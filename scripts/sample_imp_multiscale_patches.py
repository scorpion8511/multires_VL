#!/usr/bin/env python3
"""Sample multiscale patches for the IMP dataset using folder names as labels.

Dataset layout::

    imp_root/
      0/               # label "0"
        slide_a.svs
        slide_b.svs
      1/               # label "1"
        slide_c.svs
      2/               # label "2"
        slide_d.svs

Each subfolder name is treated as the slide-level label for all contained
slides. The script samples tissue-centered patch hierarchies at multiple
magnifications, names patches with the reference 5×/10×/20×/40× suffixes, and
writes a ``patches.csv`` manifest matching ``sample_camelyon_multiscale_patches.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import openslide
from PIL import Image

from sample_tcga_multiscale_patches import PatchTile, ValueLabelEncoder, compute_tissue_fraction

# Disable PIL decompression checks for very large WSIs.
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_IMAGE_EXTS = (".svs", ".tif", ".tiff")


@dataclass
class BagResult:
    node: Dict[str, object]
    tiles: List[PatchTile]


def _ensure_output_dir(out_root: Path, slide_path: Path) -> Path:
    out_dir = out_root / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _gather_labeled_slides(root: Path, exts: Sequence[str]) -> List[Tuple[Path, str]]:
    slides: List[Tuple[Path, str]] = []
    if not root.exists():
        return slides
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label_name = label_dir.name
        for ext in exts:
            for slide_path in sorted(label_dir.glob(f"**/*{ext}")):
                if slide_path.is_file():
                    slides.append((slide_path, label_name))
    return slides


def _build_label_map(
    slide: openslide.OpenSlide,
    class_id: int,
    background_threshold: int,
    max_side: int = 2048,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    width, height = slide.dimensions
    longest = max(width, height)
    if longest > max_side:
        scale = longest / float(max_side)
        thumb_size = (max(1, int(round(width / scale))), max(1, int(round(height / scale))))
    else:
        thumb_size = (width, height)

    thumb = slide.get_thumbnail(thumb_size).convert("L")
    gray = np.array(thumb)
    tissue_mask = gray < background_threshold

    label_map = np.zeros_like(gray, dtype=np.int16)
    label_map[tissue_mask] = class_id

    scale_w = width / float(label_map.shape[1])
    scale_h = height / float(label_map.shape[0])
    return label_map, (scale_w, scale_h)


def _select_level(slide: openslide.OpenSlide, requested_ds: float) -> int:
    downsamples = slide.level_downsamples
    closest = min(range(len(downsamples)), key=lambda idx: abs(downsamples[idx] - requested_ds))
    return int(closest)


def _compute_max_margin(patch_size: int, scales: Sequence[float], base_mpp: float) -> int:
    if not scales:
        return patch_size // 2
    largest = max(scales)
    requested_window = patch_size * (largest / base_mpp)
    return max(1, int(round(requested_window / 2)))


def _sample_centers(
    label_map: np.ndarray,
    count: int,
    margin: int,
    class_id: int,
    scale_factors: Tuple[float, float],
    image_size: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    coords = np.argwhere(label_map == class_id)
    if len(coords) == 0:
        return []
    height, width = label_map.shape
    scale_w, scale_h = scale_factors
    margin_x = max(1, int(np.ceil(margin / scale_w)))
    margin_y = max(1, int(np.ceil(margin / scale_h)))
    img_w, img_h = image_size
    valid: List[Tuple[int, int]] = []
    for y, x in coords:
        if margin_x <= x < width - margin_x and margin_y <= y < height - margin_y:
            cx = int(round((x + 0.5) * scale_w))
            cy = int(round((y + 0.5) * scale_h))
            if margin <= cx < img_w - margin and margin <= cy < img_h - margin:
                valid.append((cx, cy))
    if not valid:
        return []
    random.shuffle(valid)
    return valid[:count]


def _extract_patch(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    requested_mpp: float,
    base_mpp: float,
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
    out_path: Path,
) -> Optional[Path]:
    requested_ds = requested_mpp / base_mpp
    level = _select_level(slide, requested_ds)
    actual_ds = slide.level_downsamples[level]

    half = patch_size * requested_ds / 2
    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    region_size = int(round(patch_size * requested_ds / actual_ds))

    img = slide.read_region((x0, y0), level, (region_size, region_size)).convert("RGB")
    if img.size != (patch_size, patch_size):
        img = img.resize((patch_size, patch_size), Image.BILINEAR)

    tissue_fraction = compute_tissue_fraction(img, background_threshold)
    if tissue_fraction < tissue_threshold:
        return None

    img.save(out_path)
    return out_path


def generate_hierarchy_from_slide(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    mpps: Sequence[float],
    prefix: str,
    out_dir: Path,
    base_mpp: float,
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
) -> Optional[BagResult]:
    if not mpps:
        raise ValueError("At least one magnification value must be provided")

    base_scale = float(mpps[0])
    mag_map = {2.0: "5x", 1.0: "10x", 0.5: "20x", 0.25: "40x"}

    tiles: List[PatchTile] = []
    nodes: Dict[str, object] = {}
    parent: Optional[Dict[str, object]] = None

    for mpp in mpps:
        if mpp <= 0:
            raise ValueError("Magnification values must be positive")
        window_factor = mpp / base_scale
        suffix = mag_map.get(mpp, f"{mpp:.2f}mpp")
        patch_id = f"{prefix}_{suffix}"
        out_path = out_dir / f"{patch_id}.png"

        saved = _extract_patch(
            slide,
            center_x,
            center_y,
            mpp,
            base_mpp,
            patch_size,
            tissue_threshold,
            background_threshold,
            out_path,
        )
        if saved is None:
            for tile in tiles:
                try:
                    tile.patch_path.unlink()
                except FileNotFoundError:
                    pass
            return None

        tile = PatchTile(patch_id=patch_id, patch_file=out_path.name, patch_path=saved, scale_suffix=suffix)
        tiles.append(tile)

        node: Dict[str, object] = {"id": patch_id, "file": out_path.name, "children": []}
        if parent is not None:
            parent.setdefault("children", []).append(node)
        else:
            nodes = node
        parent = node

    return BagResult(node=nodes, tiles=tiles)


def _record_patch_rows(
    rows: List[Dict[str, str]],
    tiles: Iterable[PatchTile],
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


def _generate_patient_id(slide_path: Path) -> str:
    seed = abs(hash(slide_path.stem.lower())) % 10**12
    return f"{seed:012d}"


def _write_manifest(path: Path, rows: Iterable[Dict[str, str]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_slide(
    slide_path: Path,
    slide_label: str,
    out_root: Path,
    positive_bags: int,
    negative_bags: int,
    seed: int,
    mpps: Sequence[float],
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
    label_encoder: ValueLabelEncoder,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    random.seed(seed)

    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))

    label_id, label_name = label_encoder.encode(slide_label)
    label_map, scale_factors = _build_label_map(slide, label_id, background_threshold)

    out_dir = _ensure_output_dir(out_root, slide_path)
    margin = _compute_max_margin(patch_size, mpps, base_mpp)

    bags: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, str]] = []
    patient_id = _generate_patient_id(slide_path)

    bag_index = 0
    centers = _sample_centers(label_map, positive_bags, margin, label_id, scale_factors, slide.dimensions)
    for center_x, center_y in centers:
        prefix = f"patch_{bag_index}"
        bag = generate_hierarchy_from_slide(
            slide,
            center_x,
            center_y,
            list(mpps),
            prefix,
            out_dir,
            base_mpp,
            patch_size,
            tissue_threshold,
            background_threshold,
        )
        if not bag:
            continue
        bag.node.update({"label": label_name, "label_id": label_id, "center": [center_x, center_y]})
        bags.append(bag.node)
        _record_patch_rows(
            patch_rows,
            bag.tiles,
            patient_id,
            slide_path,
            bag_index,
            label_name,
            label_id,
        )
        bag_index += 1

    background_id, background_name = label_encoder.encode("Background")
    if negative_bags > 0:
        bg_map = np.zeros_like(label_map)
        centers_background = _sample_centers(bg_map, negative_bags, margin, background_id, scale_factors, slide.dimensions)
        for center_x, center_y in centers_background:
            prefix = f"patch_{bag_index}"
            bag = generate_hierarchy_from_slide(
                slide,
                center_x,
                center_y,
                list(mpps),
                prefix,
                out_dir,
                base_mpp,
                patch_size,
                tissue_threshold,
                background_threshold,
            )
            if not bag:
                continue
            bag.node.update({"label": background_name, "label_id": background_id, "center": [center_x, center_y]})
            bags.append(bag.node)
            _record_patch_rows(
                patch_rows,
                bag.tiles,
                patient_id,
                slide_path,
                bag_index,
                background_name,
                background_id,
            )
            bag_index += 1

    bags_path = out_dir / "bags.json"
    with bags_path.open("w", encoding="utf-8") as f:
        json.dump(bags, f, indent=2)

    return bags, patch_rows


def _build_label_encoder(labels: Iterable[str]) -> ValueLabelEncoder:
    unique_labels = sorted({label.strip() for label in labels if label.strip()})
    mapping: Dict[str, Tuple[int, str]] = {"background": (0, "Background")}
    for idx, label in enumerate(unique_labels, start=1):
        mapping[label] = (idx, label)
    return ValueLabelEncoder(mapping)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imp_root", type=Path, help="Root directory containing label-named folders (e.g., 0/1/2)")
    parser.add_argument("output_dir", type=Path, help="Directory where patches and manifests are written")
    parser.add_argument("--positive-bags", type=int, default=5, help="Number of labeled bags to sample per slide")
    parser.add_argument("--negative-bags", type=int, default=0, help="Number of background bags to sample per slide")
    parser.add_argument("--magnifications", type=float, nargs="+", default=[2.0, 1.0, 0.5, 0.25], help="Microns-per-pixel chain (coarse to fine)")
    parser.add_argument("--patch-size", type=int, default=512, help="Output patch size in pixels")
    parser.add_argument("--tissue-threshold", type=float, default=0.75, help="Minimum tissue fraction required to keep a patch")
    parser.add_argument("--background-threshold", type=int, default=220, help="Grayscale threshold separating tissue from background")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument("--image-exts", nargs="+", default=list(SUPPORTED_IMAGE_EXTS), help="Slide filename extensions to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    slides = _gather_labeled_slides(args.imp_root, args.image_exts)
    if not slides:
        raise SystemExit(f"No slides found under {args.imp_root} with extensions {args.image_exts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_encoder = _build_label_encoder(label for _, label in slides)

    all_rows: List[Dict[str, str]] = []
    for idx, (slide_path, label_name) in enumerate(slides):
        try:
            _, patch_rows = sample_slide(
                slide_path,
                label_name,
                args.output_dir,
                args.positive_bags,
                args.negative_bags,
                args.seed + idx,
                args.magnifications,
                args.patch_size,
                args.tissue_threshold,
                args.background_threshold,
                label_encoder,
            )
        except Exception as exc:
            print(f"Warning: skipping {slide_path} due to error: {exc}")
            continue
        all_rows.extend(patch_rows)

    manifest_path = args.output_dir / "patches.csv"
    _write_manifest(manifest_path, all_rows)
    print(f"Wrote patch manifest with {len(all_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
