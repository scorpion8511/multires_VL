#!/usr/bin/env python3
"""Sample multiscale patches for the DHMC dataset using slide-level diagnoses.

Each PNG image has a corresponding row in a CSV file with two columns:
"File name" and "Diagnosis". The diagnosis is treated as the slide-level
label; patches are sampled from tissue regions only and inherit the slide's
label. Output naming, multiscale hierarchy, and manifest schema mirror
``sample_camelyon_multiscale_patches.py``.
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
from PIL import Image

# Disable PIL's decompression bomb check for very large whole-slide images.
Image.MAX_IMAGE_PIXELS = None

from sample_tcga_multiscale_patches import (
    PatchTile,
    ValueLabelEncoder,
    compute_tissue_fraction,
)


@dataclass
class BagResult:
    """Stores the hierarchy node and associated tiles for a sampled bag."""

    node: Dict[str, object]
    tiles: List[PatchTile]


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _ensure_output_dir(out_root: Path, image_path: Path) -> Path:
    out_dir = out_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _gather_images(image_root: Path, image_exts: Sequence[str]) -> List[Path]:
    return [
        path
        for ext in image_exts
        for path in sorted(image_root.glob(f"**/*{ext}"))
        if path.is_file()
    ]


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _build_label_map(
    image: Image.Image, class_id: int, background_threshold: int, max_side: int = 4096
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Create a memory-efficient label map and scale factors.

    Instead of converting the full-resolution slide into a numpy array (which
    can exhaust memory for very large PNGs), we optionally downscale the image
    *before* converting to numpy. The resulting label map is small while the
    returned scale factors preserve the mapping back to the original
    resolution.
    """

    orig_w, orig_h = image.size
    longest = max(orig_w, orig_h)

    if longest > max_side:
        scale = longest / float(max_side)
        new_w = max(1, int(round(orig_w / scale)))
        new_h = max(1, int(round(orig_h / scale)))
        resized = image.convert("L").resize((new_w, new_h), Image.BILINEAR)
    else:
        scale = 1.0
        resized = image.convert("L")

    gray = np.array(resized)
    tissue_mask = gray < background_threshold
    label_map = np.zeros_like(gray, dtype=np.int16)
    label_map[tissue_mask] = class_id

    scale_w = orig_w / float(label_map.shape[1])
    scale_h = orig_h / float(label_map.shape[0])
    return label_map, (scale_w, scale_h)


def _compute_max_margin(patch_size: int, scales: Sequence[float]) -> int:
    if not scales:
        return patch_size // 2
    max_scale = max(scales) / scales[0]
    margin = int(patch_size * max_scale / 2)
    return max(1, margin)


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
    valid = [
        (
            int(round((x + 0.5) * scale_w)),
            int(round((y + 0.5) * scale_h)),
        )
        for y, x in coords
        if margin_x <= x < width - margin_x and margin_y <= y < height - margin_y
    ]
    if not valid:
        return []
    random.shuffle(valid)
    trimmed: List[Tuple[int, int]] = []
    for cx, cy in valid:
        if margin <= cx < img_w - margin and margin <= cy < img_h - margin:
            trimmed.append((cx, cy))
        if len(trimmed) >= count:
            break
    return trimmed


def _extract_patch(
    image: Image.Image,
    center_x: int,
    center_y: int,
    patch_size: int,
    scale_factor: float,
    tissue_threshold: float,
    background_threshold: int,
    out_path: Path,
) -> Optional[Path]:
    window = max(1, int(round(patch_size * scale_factor)))
    half = window // 2
    left = center_x - half
    top = center_y - half
    right = left + window
    bottom = top + window

    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        return None

    crop = image.crop((left, top, right, bottom))
    if crop.size != (patch_size, patch_size):
        crop = crop.resize((patch_size, patch_size), Image.BILINEAR)

    tissue_fraction = compute_tissue_fraction(crop, background_threshold)
    if tissue_fraction < tissue_threshold:
        return None

    crop.save(out_path)
    return out_path


def generate_hierarchy_from_image(
    image: Image.Image,
    center_x: int,
    center_y: int,
    scales: Sequence[float],
    prefix: str,
    out_dir: Path,
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
) -> Optional[BagResult]:
    if not scales:
        raise ValueError("At least one scale value must be provided")

    root_tiles: List[PatchTile] = []
    nodes: Dict[str, object] = {}

    base_scale = float(scales[0])
    parent: Optional[Dict[str, object]] = None

    mag_map = {2.0: "5x", 1.0: "10x", 0.5: "20x", 0.25: "40x"}

    for scale in scales:
        if scale <= 0:
            raise ValueError("Scale values must be positive")
        window_factor = scale / base_scale
        suffix = mag_map.get(scale, f"{scale:.2f}mpp")
        patch_id = f"{prefix}_{suffix}"
        out_path = out_dir / f"{patch_id}.png"

        saved = _extract_patch(
            image,
            center_x,
            center_y,
            patch_size,
            window_factor,
            tissue_threshold,
            background_threshold,
            out_path,
        )
        if saved is None:
            for tile in root_tiles:
                try:
                    tile.patch_path.unlink()
                except FileNotFoundError:
                    pass
            return None

        tile = PatchTile(
            patch_id=patch_id,
            patch_file=out_path.name,
            patch_path=saved,
            scale_suffix=suffix,
        )
        root_tiles.append(tile)

        node: Dict[str, object] = {"id": patch_id, "file": out_path.name, "children": []}
        if parent is not None:
            parent.setdefault("children", []).append(node)
        else:
            nodes = node
        parent = node

    return BagResult(node=nodes, tiles=root_tiles)


def _record_patch_rows(
    rows: List[Dict[str, str]],
    tiles: Iterable[PatchTile],
    patient_id: str,
    image_path: Path,
    bag_index: int,
    label_name: str,
    label_id: int,
) -> None:
    for tile in tiles:
        rows.append(
            {
                "patient_id": patient_id,
                "pathology_id": image_path.name,
                "subtype": label_name,
                "labels": str(label_id),
                "resolved_path": str(image_path),
                "slide_stem": image_path.stem,
                "bag_index": str(bag_index),
                "bag_label": label_name,
                "bag_label_id": str(label_id),
                "patch_id": tile.patch_id,
                "patch_file": tile.patch_file,
                "patch_path": str(tile.patch_path),
                "patch_scale": tile.scale_suffix,
            }
        )


def _generate_patient_id(image_path: Path) -> str:
    seed = abs(hash(image_path.stem.lower())) % 10**12
    return f"{seed:012d}"


def _read_labels(csv_path: Path, file_col: str, diagnosis_col: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or file_col not in reader.fieldnames or diagnosis_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required columns '{file_col}' and '{diagnosis_col}': {reader.fieldnames}"
            )
        for row in reader:
            filename = row[file_col].strip()
            diagnosis = row[diagnosis_col].strip()
            if not filename or not diagnosis:
                continue
            name = Path(filename).name.lower()
            stem = Path(filename).stem.lower()
            mapping[filename.lower()] = diagnosis
            mapping[name] = diagnosis
            mapping[stem] = diagnosis
    return mapping


def sample_image(
    image_path: Path,
    diagnosis: str,
    out_root: Path,
    positive_bags: int,
    negative_bags: int,
    seed: int,
    scales: Sequence[float],
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
    label_encoder: ValueLabelEncoder,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    random.seed(seed)

    image = _load_image(image_path)
    label_id, label_name = label_encoder.encode(diagnosis)
    label_map, scale_factors = _build_label_map(image, label_id, background_threshold)

    out_dir = _ensure_output_dir(out_root, image_path)
    margin = _compute_max_margin(patch_size, scales)

    bags: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, str]] = []
    patient_id = _generate_patient_id(image_path)

    bag_index = 0
    centers = _sample_centers(label_map, positive_bags, margin, label_id, scale_factors, image.size)
    for center_x, center_y in centers:
        prefix = f"patch_{bag_index}"
        bag = generate_hierarchy_from_image(
            image,
            center_x,
            center_y,
            list(scales),
            prefix,
            out_dir,
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
            image_path,
            bag_index,
            label_name,
            label_id,
        )
        bag_index += 1

    background_id, background_name = label_encoder.encode("Background")
    if negative_bags > 0:
        bg_map = np.zeros_like(label_map)
        centers_background = _sample_centers(
            bg_map, negative_bags, margin, background_id, scale_factors, image.size
        )
        if centers_background:
            bg_label_id, bg_label_name = background_id, background_name
            for center_x, center_y in centers_background:
                prefix = f"patch_{bag_index}"
                bag = generate_hierarchy_from_image(
                    image,
                    center_x,
                    center_y,
                    list(scales),
                    prefix,
                    out_dir,
                    patch_size,
                    tissue_threshold,
                    background_threshold,
                )
                if not bag:
                    continue
                bag.node.update({"label": bg_label_name, "label_id": bg_label_id, "center": [center_x, center_y]})
                bags.append(bag.node)
                _record_patch_rows(
                    patch_rows,
                    bag.tiles,
                    patient_id,
                    image_path,
                    bag_index,
                    bg_label_name,
                    bg_label_id,
                )
                bag_index += 1

    bags_path = out_dir / "bags.json"
    with bags_path.open("w", encoding="utf-8") as f:
        json.dump(bags, f, indent=2)

    return bags, patch_rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path, help="Directory containing DHMC PNG images")
    parser.add_argument("metadata_csv", type=Path, help="CSV with 'File name' and 'Diagnosis' columns")
    parser.add_argument("output_dir", type=Path, help="Directory where patches and manifests are written")
    parser.add_argument(
        "--file-column",
        type=str,
        default="File name",
        help="CSV column that stores the image file name",
    )
    parser.add_argument(
        "--diagnosis-column",
        type=str,
        default="Diagnosis",
        help="CSV column that stores the diagnosis/subtype label",
    )
    parser.add_argument("--positive-bags", type=int, default=5, help="Number of labeled bags to sample per image")
    parser.add_argument("--negative-bags", type=int, default=0, help="Number of background bags to sample per image")
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
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=list(SUPPORTED_IMAGE_EXTS),
        help="Image filename extensions to include",
    )
    return parser.parse_args()


def _build_label_encoder(labels: Iterable[str]) -> ValueLabelEncoder:
    """Create a label encoder that includes all diagnoses plus background."""

    unique_labels = sorted({label.strip() for label in labels if label.strip()})
    mapping: Dict[str, Tuple[int, str]] = {"background": (0, "Background")}

    for idx, label in enumerate(unique_labels, start=1):
        mapping[label] = (idx, label)

    return ValueLabelEncoder(mapping)


def main() -> None:
    args = parse_args()

    label_lookup = _read_labels(args.metadata_csv, args.file_column, args.diagnosis_column)
    images = _gather_images(args.image_dir, args.image_exts)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir} with extensions {args.image_exts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_encoder = _build_label_encoder(label_lookup.values())

    all_rows: List[Dict[str, str]] = []
    for index, image_path in enumerate(images):
        key = image_path.name.lower()
        if key not in label_lookup:
            print(f"Warning: skipping {image_path.name} because no diagnosis was found in metadata")
            continue
        diagnosis = label_lookup[key]
        try:
            _, patch_rows = sample_image(
                image_path,
                diagnosis,
                args.output_dir,
                args.positive_bags,
                args.negative_bags,
                args.seed + index,
                args.magnifications,
                args.patch_size,
                args.tissue_threshold,
                args.background_threshold,
                label_encoder,
            )
        except ValueError as e:
            print(f"Warning: skipping {image_path} due to error: {e}")
            continue
        all_rows.extend(patch_rows)

    manifest_path = args.output_dir / "patches.csv"
    _write_manifest(manifest_path, all_rows)
    print(f"Wrote patch manifest with {len(all_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
