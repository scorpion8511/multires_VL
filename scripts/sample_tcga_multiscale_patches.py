#!/usr/bin/env python3
"""Sample multi-scale patch hierarchies from TCGA WSIs.

This script replicates the patch sampling procedure described in the
paper referenced as ``2504.18856v1``.

For each WSI slide, the script randomly selects 20 seed locations.  At
each location a hierarchy of patches is extracted that all share the
same center but differ in magnification.  When Aperio XML annotations
are provided via ``--annotation-dir``, seed locations are sampled only
inside the annotated polygons so that all patches fall within the
specified tissue region:

* 5×  (2   µm/px)
* 10× (1   µm/px)
* 20× (0.5 µm/px)
* 40× (0.25 µm/px)

The resulting patches maintain spatial alignment across magnifications,
allowing models to reason over the same region at multiple resolutions.
The hierarchy for each seed is stored in a ``bags.json`` file alongside
the saved patches.

The script expects whole-slide images accessible via OpenSlide.
If a slide omits some pyramid levels, regions are resized so that each
output patch still reflects the requested field of view.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import openslide
from PIL import Image
import numpy as np


@dataclass
class AnnotationRegion:
    """Polygonal region extracted from an Aperio XML annotation."""

    points: List[Tuple[float, float]]
    value: Optional[str] = None


@dataclass
class PatchTile:
    """Metadata describing a saved patch tile."""

    patch_id: str
    patch_file: str
    patch_path: Path
    scale_suffix: str


class ValueLabelEncoder:
    """Assign deterministic integer labels to known annotation values."""

    DEFAULT_MAPPING = {
        "normal": (0, "Normal"),
        "benign": (1, "Benign"),
        "in situ carcinoma": (2, "In situ carcinoma"),
        "carcinoma in situ": (2, "In situ carcinoma"),
        "carcinoma in-situ": (2, "In situ carcinoma"),
        "in-situ carcinoma": (2, "In situ carcinoma"),
        "in situ": (2, "In situ carcinoma"),
        "situ": (2, "In situ carcinoma"),
        "carcinoma situ": (2, "In situ carcinoma"),
        "invasive carcinoma": (3, "Invasive carcinoma"),
        "carcinoma invasive": (3, "Invasive carcinoma"),
    }

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = value.lower().replace("-", " ")
        normalized = " ".join(normalized.split())
        return normalized

    def __init__(
        self, mapping: Optional[Dict[str, Tuple[int, str] | int]] = None
    ) -> None:
        source = mapping or self.DEFAULT_MAPPING
        processed: Dict[str, Tuple[int, str]] = {}
        token_map: Dict[Tuple[str, ...], Tuple[int, str]] = {}
        for key, value in source.items():
            if isinstance(value, tuple):
                label_id, canonical = value
            else:
                label_id = value
                canonical = key
            norm_key = self._normalize(key)
            processed[norm_key] = (label_id, canonical)
            tokens = tuple(sorted(norm_key.split()))
            if tokens and tokens not in token_map:
                token_map[tokens] = (label_id, canonical)
        self._mapping = processed
        self._token_mapping = token_map

    def encode(self, value: str) -> Tuple[int, str]:
        key = self._normalize(value.strip())
        if not key:
            raise ValueError("Cannot encode an empty annotation value")

        direct = self._mapping.get(key)
        if direct is not None:
            return direct

        if "situ" in key:
            situ = self._mapping.get("in situ carcinoma")
            if situ is not None:
                return situ

        tokens = tuple(sorted(key.split()))
        if tokens:
            match = self._token_mapping.get(tokens)
            if match is not None:
                return match

        for candidate_key, encoded in self._mapping.items():
            if candidate_key in key or key in candidate_key:
                return encoded

        raise KeyError(
            f"Annotation value '{value}' does not match any known subtype "
            f"labels ({', '.join(sorted(self._mapping))})."
        )


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Return the absolute area of a polygon described by ``points``."""

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += x0 * y1 - x1 * y0
    if points:
        x0, y0 = points[-1]
        x1, y1 = points[0]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Return ``True`` if ``(x, y)`` lies inside ``polygon`` using ray casting."""

    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)):
            slope = (xj - xi) / (yj - yi) if (yj - yi) != 0 else float("inf")
            x_intersect = slope * (y - yi) + xi if slope != float("inf") else xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _corners_inside(
    x: float, y: float, margin: float, polygon: Sequence[Tuple[float, float]]
) -> bool:
    offsets = [(-margin, -margin), (-margin, margin), (margin, -margin), (margin, margin)]
    for dx, dy in offsets:
        if not point_in_polygon(x + dx, y + dy, polygon):
            return False
    return True


class PolygonSampler:
    """Randomly sample centers constrained to annotation polygons."""

    def __init__(
        self,
        polygons: Sequence[AnnotationRegion],
        margin: float,
        slide_width: int,
        slide_height: int,
    ) -> None:
        self._margin = margin
        entries = []
        for region in polygons:
            poly = region.points
            if len(poly) < 3:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            min_x = max(min(xs) + margin, margin)
            max_x = min(max(xs) - margin, slide_width - margin)
            min_y = max(min(ys) + margin, margin)
            max_y = min(max(ys) - margin, slide_height - margin)
            if min_x >= max_x or min_y >= max_y:
                continue
            area = polygon_area(poly)
            if area <= 0:
                continue
            entries.append(
                {
                    "polygon": list(poly),
                    "area": area,
                    "bounds": (min_x, max_x, min_y, max_y),
                    "region": region,
                }
            )
        self._entries = entries
        self._weights = [entry["area"] for entry in entries]

    def is_available(self) -> bool:
        return bool(self._entries)

    def sample(
        self, max_attempts: int = 2000
    ) -> Optional[Tuple[int, int, AnnotationRegion]]:
        if not self._entries:
            return None
        for _ in range(max_attempts):
            entry = random.choices(self._entries, weights=self._weights, k=1)[0]
            min_x, max_x, min_y, max_y = entry["bounds"]
            if min_x >= max_x or min_y >= max_y:
                continue
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            polygon = entry["polygon"]
            if not point_in_polygon(x, y, polygon):
                continue
            if not _corners_inside(x, y, self._margin, polygon):
                continue
            return int(round(x)), int(round(y)), entry["region"]
        return None


def parse_aperio_xml(xml_path: Path) -> List[AnnotationRegion]:
    """Parse an Aperio annotation XML file into annotated regions."""

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"Warning: failed to parse annotation file {xml_path}: {exc}")
        return []

    root = tree.getroot()
    polygons: List[AnnotationRegion] = []
    for region in root.findall(".//Region"):
        vertices = region.find("Vertices")
        if vertices is None:
            continue
        points: List[Tuple[float, float]] = []
        for vertex in vertices.findall("Vertex"):
            x_val = vertex.get("X")
            y_val = vertex.get("Y")
            if x_val is None or y_val is None:
                continue
            try:
                points.append((float(x_val), float(y_val)))
            except ValueError:
                continue
        if len(points) >= 3:
            value: Optional[str] = None
            attributes = region.find("Attributes")
            if attributes is not None:
                for attribute in attributes.findall("Attribute"):
                    name = attribute.get("Name") or attribute.get("name")
                    attr_value = attribute.get("Value") or attribute.get("value")
                    if attr_value is None:
                        continue
                    attr_value = attr_value.strip()
                    if not attr_value:
                        continue
                    if name and name.lower() == "value":
                        value = attr_value
                        break
                    if value is None:
                        value = attr_value

            if value is None:
                text_value = region.get("Text") or region.get("text")
                if text_value:
                    text_value = text_value.strip()
                    if text_value:
                        value = text_value

            polygons.append(AnnotationRegion(points=points, value=value))
    return polygons


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        if not path:
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def index_annotation_files(
    annotation_root: Optional[Path],
) -> Dict[str, List[Path]]:
    """Return a lookup table for annotation XMLs keyed by lowercase names."""

    lookup: Dict[str, List[Path]] = defaultdict(list)

    def _register(key: str, path: Path) -> None:
        entries = lookup[key]
        if path in entries:
            return
        if entries:
            print(
                "Warning: multiple annotation files map to key '"
                f"{key}' (including {entries[0]} and {path})."
            )
        entries.append(path)

    if annotation_root is None:
        return lookup

    if annotation_root.is_file():
        key_stem = annotation_root.stem.lower()
        key_name = annotation_root.name.lower()
        _register(key_stem, annotation_root)
        _register(key_name, annotation_root)
        return lookup

    if annotation_root.is_dir():
        for xml_path in annotation_root.rglob("*.xml"):
            key_stem = xml_path.stem.lower()
            key_name = xml_path.name.lower()
            _register(key_stem, xml_path)
            _register(key_name, xml_path)

    return lookup


def load_slide_polygons(
    slide_path: Path,
    annotation_root: Optional[Path],
    annotation_lookup: Optional[Dict[str, List[Path]]] = None,
) -> List[AnnotationRegion]:
    """Return polygons for ``slide_path`` if an annotation XML is available."""

    candidates: List[Path] = []
    normalized_keys = {slide_path.stem.lower(), slide_path.name.lower()}

    if annotation_lookup:
        for key in normalized_keys:
            candidates.extend(annotation_lookup.get(key, []))

    if annotation_root is not None and not candidates:
        if annotation_root.is_file():
            candidates.append(annotation_root)
        elif annotation_root.is_dir():
            candidates.extend(
                [
                    annotation_root / f"{slide_path.stem}.xml",
                    annotation_root / f"{slide_path.name}.xml",
                    annotation_root / f"{slide_path.stem.upper()}.xml",
                    annotation_root / f"{slide_path.stem.lower()}.xml",
                ]
            )

    annotation_candidates = _unique_paths(candidates)
    candidate_keys = {path.resolve() for path in annotation_candidates if path.exists()}

    fallback_candidate = slide_path.with_suffix(".xml")
    combined_candidates = annotation_candidates + [fallback_candidate]

    matched_annotation = False
    for candidate in _unique_paths(combined_candidates):
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.resolve() in candidate_keys:
            matched_annotation = True
        polygons = parse_aperio_xml(candidate)
        if polygons:
            return polygons
        else:
            print(
                f"Warning: annotation file {candidate} does not contain "
                "any polygon regions."
            )

    if annotation_root is not None and annotation_candidates and not matched_annotation:
        print(
            "Warning: no matching annotation XML found for "
            f"{slide_path.name} in {annotation_root}."
        )

    return []


def resolve_annotation_root(annotation_root: Optional[Path]) -> Optional[Path]:
    """Normalize ``annotation_root`` and resolve file:// URIs if necessary."""

    if annotation_root is None:
        return None

    raw = str(annotation_root)
    if raw.startswith("file:"):
        parsed = urlparse(raw)
        if parsed.scheme == "file":
            decoded = unquote(parsed.path)
            if decoded:
                if decoded.startswith("/") and len(decoded) > 2 and decoded[2] == ":":
                    decoded = decoded.lstrip("/")
                return Path(decoded)
    return annotation_root

def compute_tissue_fraction(img: Image.Image, background_threshold: int = 220) -> float:
    """Return the fraction of pixels that are likely to be tissue.

    Pixels lighter than ``background_threshold`` (in grayscale) are considered
    background and excluded from the tissue count.
    """

    gray = np.array(img.convert("L"))
    tissue_mask = gray < background_threshold
    return float(tissue_mask.mean())


def save_patch(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    level: int,
    size: int,
    requested_ds: float,
    out_path: Path,
    tissue_threshold: float,
    background_threshold: int,
) -> Optional[Path]:
    """Read and save a region from ``slide`` if it contains sufficient tissue."""

    actual_ds = slide.level_downsamples[level]
    # Compute region in level-0 coordinates covering the requested area
    x0 = int(center_x - size * requested_ds / 2)
    y0 = int(center_y - size * requested_ds / 2)
    region_size = int(size * requested_ds / actual_ds)

    img = slide.read_region((x0, y0), level, (region_size, region_size)).convert("RGB")
    if region_size != size:
        img = img.resize((size, size), Image.BILINEAR)
    tissue_fraction = compute_tissue_fraction(img, background_threshold)
    if tissue_fraction < tissue_threshold:
        return None
    img.save(out_path)
    return out_path


def generate_hierarchy(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    mpps: List[float],
    prefix: str,
    out_dir: Path,
    base_mpp: float,
    size: int = 512,
    tissue_threshold: float = 0.75,
    background_threshold: int = 220,
) -> Optional[Tuple[Dict[str, object], List[PatchTile]]]:
    """Generate a chain of patches centered at ``(center_x, center_y)``.

    The first entry in ``mpps`` corresponds to the coarsest magnification
    (e.g. 5×).  Each subsequent entry represents a higher magnification of
    the *same* region.  The returned dictionary has the same structure as
    the original script: every node contains a single child describing the
    next magnification level.
    """

    mpp = mpps[0]
    requested_ds = mpp / base_mpp
    level = slide.get_best_level_for_downsample(requested_ds)

    mag_map = {2.0: "5x", 1.0: "10x", 0.5: "20x", 0.25: "40x"}
    suffix = mag_map.get(mpp, f"{mpp:.2f}mpp")
    patch_id = f"{prefix}_{suffix}"
    out_path = out_dir / f"{patch_id}.png"
    saved_path = save_patch(
        slide,
        center_x,
        center_y,
        level,
        size,
        requested_ds,
        out_path,
        tissue_threshold,
        background_threshold,
    )
    if saved_path is None:
        return None

    tile = PatchTile(
        patch_id=patch_id,
        patch_file=out_path.name,
        patch_path=saved_path,
        scale_suffix=suffix,
    )
    node: Dict[str, object] = {"id": patch_id, "file": out_path.name, "children": []}
    if len(mpps) > 1:
        child = generate_hierarchy(
            slide,
            center_x,
            center_y,
            mpps[1:],
            patch_id,
            out_dir,
            base_mpp,
            size,
            tissue_threshold,
            background_threshold,
        )
        if not child:
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass
            return None
        child_node, child_tiles = child
        node["children"].append(child_node)
        tiles = [tile] + child_tiles
    else:
        tiles = [tile]
    return node, tiles


def _generate_patient_id(slide_path: Path) -> str:
    """Return a deterministic pseudo-random identifier for ``slide_path``."""

    seed = abs(hash(slide_path.stem.lower())) % 10**12
    return f"{seed:012d}"


def process_wsi(
    slide_path: Path,
    out_root: Path,
    num_parents: int = 20,
    seed: int = 0,
    tissue_threshold: float = 0.75,
    background_threshold: int = 220,
    polygons: Optional[List[AnnotationRegion]] = None,
    label_encoder: Optional[ValueLabelEncoder] = None,
) -> Tuple[int, List[Dict[str, str]]]:
    """Process a single WSI and generate multi-scale patch bags.

    Returns the number of patch bags successfully collected.
    """
    random.seed(seed)
    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))

    slide_width, slide_height = slide.dimensions
    out_dir = out_root / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    size = 512
    target_mpps = [2.0, 1.0, 0.5, 0.25]  # 5×, 10×, 20×, 40×

    # Compute bounds for sampling centers so that all magnifications fit
    parent_ds = target_mpps[0] / base_mpp
    margin = size * parent_ds / 2
    margin_int = int(margin)
    max_cx = slide_width - margin_int
    max_cy = slide_height - margin_int

    sampler: Optional[PolygonSampler] = None
    if polygons:
        sampler = PolygonSampler(polygons, margin, slide_width, slide_height)
        if not sampler.is_available():
            print(
                "Warning: annotation polygons for "
                f"{slide_path.name} are too small for sampling."
            )
            sampler = None

    bags = []
    patch_rows: List[Dict[str, str]] = []
    patient_id = _generate_patient_id(slide_path)
    i = 0
    attempts = 0
    max_attempts = num_parents * 20 if num_parents > 0 else 0
    while i < num_parents and (max_attempts == 0 or attempts < max_attempts):
        attempts += 1
        if sampler is not None:
            candidate = sampler.sample()
            if candidate is None:
                print(
                    "Warning: unable to find more valid centers inside "
                    f"annotations for {slide_path.name}."
                )
                break
            center_x, center_y, selected_region = candidate
        else:
            center_x = random.randint(margin_int, max_cx)
            center_y = random.randint(margin_int, max_cy)
            selected_region = None
        prefix = f"patch_{i}"
        bag = generate_hierarchy(
            slide,
            center_x,
            center_y,
            target_mpps,
            prefix,
            out_dir,
            base_mpp,
            size,
            tissue_threshold,
            background_threshold,
        )
        if bag:
            bag_node, tiles = bag
            bags.append(bag_node)
            if selected_region and selected_region.value:
                region_value = selected_region.value.strip() or "Normal"
            else:
                region_value = "Normal"
            label_str = ""
            if label_encoder is not None:
                try:
                    label_id, canonical_value = label_encoder.encode(region_value)
                except (KeyError, ValueError) as exc:
                    print(
                        "Warning:",
                        exc,
                        "-- leaving label empty for",
                        slide_path.name,
                        f"(bag {i}).",
                    )
                else:
                    label_str = str(label_id)
                    region_value = canonical_value
            for tile in tiles:
                patch_rows.append(
                    {
                        "patient_id": patient_id,
                        "pathology_id": slide_path.name,
                        "subtype": region_value,
                        "labels": label_str,
                        "resolved_path": str(slide_path),
                        "slide_stem": slide_path.stem,
                        "bag_index": str(i),
                        "patch_id": tile.patch_id,
                        "patch_file": tile.patch_file,
                        "patch_path": str(tile.patch_path),
                        "patch_scale": tile.scale_suffix,
                    }
                )
            i += 1

    if i < num_parents:
        print(
            f"Warning: only collected {i} patch bags (requested {num_parents}) "
            f"for {slide_path.name}."
        )

    with open(out_dir / "bags.json", "w") as f:
        json.dump(bags, f, indent=2)

    slide.close()

    return len(bags), patch_rows


def gather_slides(wsi_root: Path) -> List[Path]:
    """Return a list of WSI files under ``wsi_root``.

    ``wsi_root`` may point to a single file or a directory.  If a directory is
    provided, it is searched recursively for known WSI file extensions.
    """
    exts = {".svs", ".tif", ".tiff", ".ndpi"}
    if wsi_root.is_file():
        return [wsi_root] if wsi_root.suffix.lower() in exts else []
    return [p for p in wsi_root.rglob("*") if p.suffix.lower() in exts]


def load_metadata_table(csv_path: Path) -> Tuple[List[Dict[str, str]], Sequence[str]]:
    """Read ``csv_path`` and return a list of row dictionaries."""

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"Metadata file {csv_path} is empty")
        return rows, reader.fieldnames or []


@dataclass
class BalancedSelection:
    """Container describing balanced slide sampling results."""

    initial: List[Tuple[Path, Dict[str, str]]]
    extras: Dict[str, List[Tuple[Path, Dict[str, str]]]]
    per_subtype: int
    subtypes: List[str]


def _index_slide_paths(slides: Sequence[Path]) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """Return lookup tables for slide files keyed by name and stem."""

    by_name = {p.name: p for p in slides}
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for slide in slides:
        by_stem[slide.stem].append(slide)
    return by_name, by_stem


def _match_slide_path(
    raw_value: str,
    by_name: Dict[str, Path],
    by_stem: Dict[str, List[Path]],
) -> Optional[Path]:
    """Resolve ``raw_value`` to a slide ``Path`` if possible."""

    if not raw_value:
        return None

    filename = Path(raw_value).name
    slide_path = by_name.get(filename)
    if slide_path is not None:
        return slide_path

    stem_matches = by_stem.get(Path(filename).stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        raise ValueError(
            f"Multiple slide files match '{raw_value}'. Please disambiguate."
        )
    return None


def select_balanced_slide_subset(
    rows: Iterable[Dict[str, str]],
    subtype_col: str,
    path_column: str,
    total: int,
    seed: int,
    slides: Sequence[Path],
) -> BalancedSelection:
    """Select ``total`` slide/metadata pairs with equal subtype representation."""

    if not slides:
        return []

    by_name, by_stem = _index_slide_paths(slides)

    grouped: Dict[str, List[Tuple[Path, Dict[str, str]]]] = defaultdict(list)
    all_subtypes: set[str] = set()
    skipped: Dict[str, int] = defaultdict(int)

    for row in rows:
        if subtype_col not in row:
            raise KeyError(f"Column '{subtype_col}' not found in metadata")
        if path_column not in row:
            raise KeyError(f"Column '{path_column}' not found in metadata")

        subtype = row[subtype_col]
        all_subtypes.add(subtype)
        raw_value = row[path_column]
        slide_path = _match_slide_path(raw_value, by_name, by_stem)
        if slide_path is None:
            skipped[raw_value or "<empty>"] += 1
            continue

        existing_paths = {path for path, _ in grouped[subtype]}
        if slide_path in existing_paths:
            continue

        grouped[subtype].append((slide_path, row))

    if not all_subtypes:
        raise ValueError("No subtypes found in metadata table")

    num_subtypes = len(all_subtypes)
    if total % num_subtypes != 0:
        raise ValueError(
            "Cannot select an equal number of WSIs per subtype: "
            f"requested {total} slides across {num_subtypes} subtypes."
        )

    per_subtype = total // num_subtypes

    for subtype in sorted(all_subtypes):
        available = grouped.get(subtype, [])
        if len(available) < per_subtype:
            raise ValueError(
                f"Not enough resolvable slides for subtype '{subtype}': "
                f"required {per_subtype}, found {len(available)}"
            )

    total_skipped = sum(skipped.values())
    if total_skipped:
        skipped_examples = ", ".join(sorted(skipped.keys())[:5])
        print(
            "Warning: skipped "
            f"{total_skipped} metadata rows without matching slides"
            f" (examples: {skipped_examples})."
        )

    rng = random.Random(seed)
    subtype_order = sorted(all_subtypes)
    initial: List[Tuple[Path, Dict[str, str]]] = []
    extras: Dict[str, List[Tuple[Path, Dict[str, str]]]] = {}

    for subtype in subtype_order:
        candidates = list(grouped[subtype])
        rng.shuffle(candidates)
        initial.extend(candidates[:per_subtype])
        extras[subtype] = candidates[per_subtype:]

    rng.shuffle(initial)

    return BalancedSelection(
        initial=initial,
        extras=extras,
        per_subtype=per_subtype,
        subtypes=subtype_order,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample multi-resolution patch bags from WSIs")
    parser.add_argument(
        "wsi_dir",
        type=Path,
        help="Path to a WSI file or directory containing WSI files",
    )
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--num_parents", type=int, default=20,
                        help="Number of parent patches per slide")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=0.75,
        help="Minimum tissue fraction required to keep a patch",
    )
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=220,
        help="Grayscale threshold separating tissue from background",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="CSV file describing slides and their subtypes",
    )
    parser.add_argument(
        "--metadata-subtype-column",
        type=str,
        default="subtype",
        help="Column in the metadata CSV that stores subtype labels",
    )
    parser.add_argument(
        "--metadata-path-column",
        type=str,
        default="pathology_id",
        help="Column in the metadata CSV that stores slide file names",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        help=(
            "Directory or file containing Aperio-style XML annotations. "
            "If provided, patches are sampled only from annotated regions."
        ),
    )
    parser.add_argument(
        "--num-wsi",
        type=int,
        default=50,
        help="Number of WSIs to sample evenly across subtypes",
    )

    args = parser.parse_args()

    slides = gather_slides(args.wsi_dir)
    if not slides:
        raise FileNotFoundError(f"No WSI files found at {args.wsi_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotation_root = resolve_annotation_root(args.annotation_dir)
    annotation_lookup = index_annotation_files(annotation_root)
    label_encoder = ValueLabelEncoder()
    all_patch_rows: List[Dict[str, str]] = []

    if args.metadata_csv is not None:
        metadata_rows, fieldnames = load_metadata_table(args.metadata_csv)
        selection = select_balanced_slide_subset(
            metadata_rows,
            args.metadata_subtype_column,
            args.metadata_path_column,
            args.num_wsi,
            args.seed,
            slides,
        )
        counts = {subtype: 0 for subtype in selection.subtypes}
        accepted_pairs: List[Tuple[Path, Dict[str, str]]] = []

        def process_candidate(
            pair: Tuple[Path, Dict[str, str]], *, is_replacement: bool = False
        ) -> bool:
            slide_path, row = pair
            subtype = row[args.metadata_subtype_column]
            print(f"Processing {slide_path}")
            annotation_polygons = load_slide_polygons(
                slide_path,
                annotation_root,
                annotation_lookup,
            )
            bag_count, patch_rows = process_wsi(
                slide_path,
                args.out_dir,
                num_parents=args.num_parents,
                seed=args.seed,
                tissue_threshold=args.tissue_threshold,
                background_threshold=args.background_threshold,
                polygons=annotation_polygons,
                label_encoder=label_encoder,
            )
            if bag_count < args.num_parents:
                slide_out_dir = args.out_dir / slide_path.stem
                if slide_out_dir.exists():
                    shutil.rmtree(slide_out_dir, ignore_errors=True)
                reason = "replacement" if is_replacement else "selected"
                print(
                    f"Scheduling an additional slide for subtype '{subtype}' "
                    f"because {slide_path.name} ({reason}) yielded "
                    f"{bag_count} patch bags."
                )
                return False

            counts[subtype] += 1
            accepted_pairs.append(pair)
            all_patch_rows.extend(patch_rows)
            return True

        for pair in selection.initial:
            process_candidate(pair)

        for subtype in selection.subtypes:
            while counts[subtype] < selection.per_subtype:
                extras = selection.extras[subtype]
                if not extras:
                    raise RuntimeError(
                        "Ran out of candidate slides for subtype "
                        f"'{subtype}' while attempting to collect "
                        f"{selection.per_subtype} slides with at least "
                        f"{args.num_parents} patch bags."
                    )
                replacement_pair = extras.pop()
                process_candidate(replacement_pair, is_replacement=True)

        if any(count < selection.per_subtype for count in counts.values()):
            raise RuntimeError(
                "Unable to assemble the requested balanced subset with "
                f"{args.num_parents} patch bags per slide."
            )

        expected_total = selection.per_subtype * len(selection.subtypes)
        if len(accepted_pairs) != expected_total:
            raise RuntimeError(
                "Internal error: collected slide count does not match the "
                "requested balanced sample size."
            )

        if not accepted_pairs:
            raise RuntimeError("No slides produced the required patch bags")

        output_fields: Sequence[str] = (
            list(fieldnames) if fieldnames else list(accepted_pairs[0][1].keys())
        )
        if "resolved_path" not in output_fields:
            output_fields = list(output_fields) + ["resolved_path"]
        selected_csv_path = args.out_dir / "selected_wsi.csv"
        with selected_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields)
            writer.writeheader()
            for slide_path, row in accepted_pairs:
                row_out = dict(row)
                row_out["resolved_path"] = str(slide_path)
                writer.writerow(row_out)
    else:
        for slide_path in slides:
            print(f"Processing {slide_path}")
            annotation_polygons = load_slide_polygons(
                slide_path,
                annotation_root,
                annotation_lookup,
            )
            bag_count, patch_rows = process_wsi(
                slide_path,
                args.out_dir,
                num_parents=args.num_parents,
                seed=args.seed,
                tissue_threshold=args.tissue_threshold,
                background_threshold=args.background_threshold,
                polygons=annotation_polygons,
                label_encoder=label_encoder,
            )
            if bag_count >= args.num_parents or args.num_parents == 0:
                all_patch_rows.extend(patch_rows)

    if all_patch_rows:
        patch_csv_path = args.out_dir / "patches.csv"
        fieldnames = [
            "patient_id",
            "pathology_id",
            "subtype",
            "labels",
            "resolved_path",
            "slide_stem",
            "bag_index",
            "patch_id",
            "patch_file",
            "patch_path",
            "patch_scale",
        ]
        with patch_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_patch_rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
