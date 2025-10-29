#!/usr/bin/env python3
"""Sample multi-scale patches for the Camelyon dataset.

The Camelyon16 dataset ships Aperio TIFF whole-slide images accompanied by
XML annotations that describe tumor polygons.  This script mirrors the
multi-scale sampling strategy implemented for the TCGA pipeline while adding
Camelyon-specific handling:

* tumor seed locations are drawn from the provided polygons
* normal seed locations avoid every tumor polygon
* each saved patch hierarchy is recorded in ``bags.json`` alongside a
  ``patches.csv`` manifest that stores per-scale metadata and tumor/normal
  labels.

By default the script collects an equal number of tumor and normal bags for
every slide.  The magnification chain matches the TCGA workflow (5×, 10×, 20×,
and 40× equivalents) so downstream models can reuse the exported data without
modification.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import openslide

from sample_tcga_multiscale_patches import (
    AnnotationRegion,
    PatchTile,
    PolygonSampler,
    ValueLabelEncoder,
    generate_hierarchy,
    gather_slides,
    point_in_polygon,
)


def _extract_group_metadata(root: ET.Element) -> Dict[str, Dict[str, object]]:
    """Return descriptive metadata for ``Group`` elements keyed by their name."""

    def _clean(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        stripped = text.strip()
        return stripped if stripped else None

    group_lookup: Dict[str, Dict[str, object]] = {}
    for group in root.findall(".//Group"):
        raw_name = _clean(group.get("Name") or group.get("name"))
        if not raw_name:
            continue
        lower_name = raw_name.lower()
        labels: List[str] = []

        attr_parent = group.find("Attributes")
        if attr_parent is not None:
            for attribute in attr_parent.findall("Attribute"):
                attr_value = _clean(attribute.get("Value") or attribute.get("value"))
                attr_name = _clean(attribute.get("Name") or attribute.get("name"))
                if attr_value:
                    labels.append(attr_value)
                elif attr_name:
                    labels.append(attr_name)

        if not labels:
            labels.append(raw_name)

        tumor_like = any(
            token in label.lower()
            for label in labels
            for token in ("tumor", "metast", "positive")
        )
        negative_like = any(
            token in label.lower()
            for label in labels
            for token in ("normal", "negative", "exclusion", "exclude")
        )

        group_lookup[lower_name] = {
            "labels": labels,
            "is_tumor": tumor_like and not negative_like,
            "is_negative": negative_like and not tumor_like,
        }

    return group_lookup


def _annotation_is_tumor(
    annotation: ET.Element,
    group_lookup: Dict[str, Dict[str, object]],
) -> bool:
    """Heuristically determine if an annotation polygon marks tumor tissue."""

    def _clean(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        stripped = text.strip()
        return stripped if stripped else None

    def _classify(text: str) -> Optional[bool]:
        lower = text.lower()
        if any(token in lower for token in ("tumor", "metast", "positive")):
            return True
        if any(token in lower for token in ("normal", "negative", "exclusion", "exclude")):
            return False
        return None

    candidates: List[str] = []
    part_of_group = _clean(annotation.get("PartOfGroup") or annotation.get("partOfGroup"))
    if part_of_group:
        candidates.append(part_of_group)
        group = group_lookup.get(part_of_group.lower())
        if group:
            candidates.extend(group.get("labels", []))
            if group.get("is_tumor"):
                return True
            if group.get("is_negative"):
                return False

    name_attr = _clean(annotation.get("Name") or annotation.get("name"))
    if name_attr:
        candidates.append(name_attr)

    attr_parent = annotation.find("Attributes")
    if attr_parent is not None:
        for attribute in attr_parent.findall("Attribute"):
            attr_value = _clean(attribute.get("Value") or attribute.get("value"))
            attr_name = _clean(attribute.get("Name") or attribute.get("name"))
            if attr_value:
                candidates.append(attr_value)
            if attr_name:
                candidates.append(attr_name)

    for candidate in candidates:
        classification = _classify(candidate)
        if classification is not None:
            return classification

    # Default to tumor: Camelyon annotations typically describe tumor regions.
    return True


def parse_camelyon_xml(xml_path: Path) -> List[AnnotationRegion]:
    """Parse tumor polygons from a Camelyon XML annotation file."""

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"Warning: failed to parse annotation file {xml_path}: {exc}")
        return []
    root = tree.getroot()

    polygons: List[AnnotationRegion] = []
    group_lookup = _extract_group_metadata(root)
    for annotation in root.findall(".//Annotation"):
        type_attr = (annotation.get("Type") or annotation.get("type") or "").lower()
        if type_attr and type_attr != "polygon":
            continue

        if not _annotation_is_tumor(annotation, group_lookup):
            continue

        coordinates_parent = annotation.find(".//Coordinates")
        if coordinates_parent is None:
            continue

        points: List[Tuple[float, float]] = []
        for coord in coordinates_parent.findall("Coordinate"):
            x_val = coord.get("X") or coord.get("x")
            y_val = coord.get("Y") or coord.get("y")
            if x_val is None or y_val is None:
                continue
            try:
                points.append((float(x_val), float(y_val)))
            except ValueError:
                continue
        if len(points) >= 3:
            polygons.append(AnnotationRegion(points=points, value="Tumor"))

    return polygons


def locate_annotation_file(slide_path: Path, annotation_root: Optional[Path]) -> Optional[Path]:
    """Return the XML file paired with ``slide_path`` if available."""

    candidates: List[Path] = []
    if annotation_root is not None:
        root_path = annotation_root
        if root_path.is_dir():
            candidates.extend(
                [
                    root_path / f"{slide_path.stem}.xml",
                    root_path / f"{slide_path.name}.xml",
                    root_path / f"{slide_path.stem.upper()}.xml",
                    root_path / f"{slide_path.stem.lower()}.xml",
                ]
            )
        elif root_path.is_file():
            candidates.append(root_path)
    candidates.append(slide_path.with_suffix(".xml"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _generate_patient_id(slide_path: Path) -> str:
    seed = abs(hash(slide_path.stem.lower())) % 10**12
    return f"{seed:012d}"


def _corners_overlap(
    center_x: float,
    center_y: float,
    margin: float,
    polygon: Sequence[Tuple[float, float]],
) -> bool:
    offsets = [
        (-margin, -margin),
        (-margin, margin),
        (margin, -margin),
        (margin, margin),
    ]
    for dx, dy in offsets:
        if point_in_polygon(center_x + dx, center_y + dy, polygon):
            return True
    return False


def _intersects_polygons(
    center_x: int,
    center_y: int,
    margin: float,
    polygons: Sequence[AnnotationRegion],
) -> bool:
    for region in polygons:
        polygon = region.points
        if point_in_polygon(center_x, center_y, polygon):
            return True
        if _corners_overlap(center_x, center_y, margin, polygon):
            return True
    return False


def _sample_negative_center(
    width: int,
    height: int,
    margin_int: int,
    margin: float,
    polygons: Sequence[AnnotationRegion],
    max_attempts: int = 4000,
) -> Optional[Tuple[int, int]]:
    for _ in range(max_attempts):
        center_x = random.randint(margin_int, width - margin_int)
        center_y = random.randint(margin_int, height - margin_int)
        if not _intersects_polygons(center_x, center_y, margin, polygons):
            return center_x, center_y
    return None


def _ensure_output_dir(out_root: Path, slide_path: Path) -> Path:
    out_dir = out_root / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _record_patch_rows(
    rows: List[Dict[str, str]],
    tiles: Sequence[PatchTile],
    patient_id: str,
    slide_path: Path,
    bag_index: int,
    label_name: str,
    label_id: int,
) -> None:
    for tile in tiles:
        rows.append(
            {
                "patient_id": patient_id,
                "pathology_id": slide_path.name,
                "subtype": label_name,
                "labels": str(label_id),
                "resolved_path": str(slide_path),
                "slide_stem": slide_path.stem,
                "bag_index": str(bag_index),
                "bag_label": label_name,
                "bag_label_id": str(label_id),
                "patch_id": tile.patch_id,
                "patch_file": tile.patch_file,
                "patch_path": str(tile.patch_path),
                "patch_scale": tile.scale_suffix,
            }
        )


def sample_camelyon_slide(
    slide_path: Path,
    out_root: Path,
    tumor_polygons: Sequence[AnnotationRegion],
    tumor_bags: int,
    normal_bags: int,
    seed: int,
    target_mpps: Sequence[float],
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
    label_encoder: ValueLabelEncoder,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    random.seed(seed)

    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    width, height = slide.dimensions
    out_dir = _ensure_output_dir(out_root, slide_path)

    if not target_mpps:
        raise ValueError("At least one magnification (mpp) value must be provided")

    parent_ds = target_mpps[0] / base_mpp
    margin = patch_size * parent_ds / 2
    margin_int = int(margin)
    if width <= 2 * margin_int or height <= 2 * margin_int:
        raise ValueError(
            f"Slide {slide_path} is too small ({width}x{height}) for the requested patch size"
        )

    sampler: Optional[PolygonSampler] = None
    if tumor_polygons:
        sampler = PolygonSampler(tumor_polygons, margin, width, height)
        if not sampler.is_available():
            print(
                "Warning: tumor polygons for",
                slide_path.name,
                "are too small to fit the requested patch hierarchy.",
            )
            sampler = None

    patient_id = _generate_patient_id(slide_path)
    bags: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, str]] = []
    bag_index = 0

    tumor_label_id, tumor_label_name = label_encoder.encode("Tumor")
    normal_label_id, normal_label_name = label_encoder.encode("Normal")

    collected_tumor = 0
    if tumor_bags > 0 and sampler is None:
        print(
            f"Warning: unable to collect tumor patches for {slide_path.name} because "
            "no valid polygons were found.",
        )

    while sampler is not None and collected_tumor < tumor_bags:
        candidate = sampler.sample()
        if candidate is None:
            print(
                f"Warning: exhausted tumor sampling attempts for {slide_path.name}."
            )
            break
        center_x, center_y, _ = candidate
        prefix = f"patch_{bag_index}"
        bag = generate_hierarchy(
            slide,
            center_x,
            center_y,
            list(target_mpps),
            prefix,
            out_dir,
            base_mpp,
            patch_size,
            tissue_threshold,
            background_threshold,
        )
        if not bag:
            continue
        bag_node, tiles = bag
        bag_node.update(
            {
                "label": tumor_label_name,
                "label_id": tumor_label_id,
                "center": [center_x, center_y],
            }
        )
        bags.append(bag_node)
        _record_patch_rows(
            patch_rows,
            tiles,
            patient_id,
            slide_path,
            bag_index,
            tumor_label_name,
            tumor_label_id,
        )
        bag_index += 1
        collected_tumor += 1

    collected_normal = 0
    max_attempts = max(4000, normal_bags * 50)
    attempts = 0
    while collected_normal < normal_bags and attempts < max_attempts:
        attempts += 1
        candidate = _sample_negative_center(
            width,
            height,
            margin_int,
            margin,
            tumor_polygons,
        )
        if candidate is None:
            break
        center_x, center_y = candidate
        if sampler is not None and _intersects_polygons(center_x, center_y, margin, tumor_polygons):
            continue
        prefix = f"patch_{bag_index}"
        bag = generate_hierarchy(
            slide,
            center_x,
            center_y,
            list(target_mpps),
            prefix,
            out_dir,
            base_mpp,
            patch_size,
            tissue_threshold,
            background_threshold,
        )
        if not bag:
            continue
        bag_node, tiles = bag
        bag_node.update(
            {
                "label": normal_label_name,
                "label_id": normal_label_id,
                "center": [center_x, center_y],
            }
        )
        bags.append(bag_node)
        _record_patch_rows(
            patch_rows,
            tiles,
            patient_id,
            slide_path,
            bag_index,
            normal_label_name,
            normal_label_id,
        )
        bag_index += 1
        collected_normal += 1

    if collected_tumor < tumor_bags:
        print(
            f"Warning: collected {collected_tumor} tumor bags (requested {tumor_bags}) "
            f"for {slide_path.name}."
        )
    if collected_normal < normal_bags:
        print(
            f"Warning: collected {collected_normal} normal bags (requested {normal_bags}) "
            f"for {slide_path.name}."
        )

    with open(out_dir / "bags.json", "w") as f:
        json.dump(bags, f, indent=2)

    slide.close()
    return bags, patch_rows


def _write_manifest(csv_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "patient_id",
        "pathology_id",
        "subtype",
        "labels",
        "resolved_path",
        "slide_stem",
        "bag_index",
        "bag_label",
        "bag_label_id",
        "patch_id",
        "patch_file",
        "patch_path",
        "patch_scale",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wsi_root", type=Path, help="Slide file or directory containing Camelyon WSIs")
    parser.add_argument("output_dir", type=Path, help="Directory where patches and metadata will be stored")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Directory (or single XML file) containing Camelyon tumor annotations",
    )
    parser.add_argument("--tumor-bags", type=int, default=20, help="Number of tumor-centered bags to collect per slide")
    parser.add_argument(
        "--normal-bags",
        type=int,
        default=20,
        help="Number of normal (non-tumor) bags to collect per slide",
    )
    parser.add_argument(
        "--magnifications",
        type=float,
        nargs="+",
        default=[2.0, 1.0, 0.5, 0.25],
        help="Requested microns-per-pixel chain (coarse to fine)",
    )
    parser.add_argument("--patch-size", type=int, default=512, help="Output patch size in pixels")
    parser.add_argument("--tissue-threshold", type=float, default=0.75, help="Minimum tissue fraction required to keep a patch")
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=220,
        help="Grayscale threshold that separates background from tissue",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for reproducible sampling",
    )

    args = parser.parse_args()

    slides = gather_slides(args.wsi_root)
    if not slides:
        raise SystemExit(f"No whole-slide images found under {args.wsi_root}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_root = args.annotation_dir
    label_encoder = ValueLabelEncoder({"normal": (0, "Normal"), "tumor": (1, "Tumor")})

    all_rows: List[Dict[str, str]] = []
    for index, slide_path in enumerate(sorted(slides)):
        print(f"Processing {slide_path}")
        annotation_path = locate_annotation_file(slide_path, annotation_root)
        tumor_polygons: List[AnnotationRegion] = []
        if annotation_path is not None:
            tumor_polygons = parse_camelyon_xml(annotation_path)
            if not tumor_polygons:
                print(
                    f"Warning: annotation file {annotation_path} does not contain any tumor polygons."
                )
        else:
            print(f"Warning: no annotation XML found for {slide_path.name}; treating slide as normal.")

        _, patch_rows = sample_camelyon_slide(
            slide_path,
            output_dir,
            tumor_polygons,
            args.tumor_bags,
            args.normal_bags,
            args.seed + index,
            args.magnifications,
            args.patch_size,
            args.tissue_threshold,
            args.background_threshold,
            label_encoder,
        )
        all_rows.extend(patch_rows)

    manifest_path = output_dir / "patches.csv"
    _write_manifest(manifest_path, all_rows)
    print(f"Wrote patch manifest with {len(all_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
