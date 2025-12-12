#!/usr/bin/env python3
"""Sample multiscale patches for the AGCC dataset with per-class mask folders.

Each AGCC whole-slide image (e.g., ``Subset1_Test_1.tiff``) has a sibling mask
folder with the same stem (``Subset1_Test_1``) that contains class-specific mask
files such as ``G3_Mask.tiff``, ``G4_Mask.tiff``, ``Normal_Mask.tiff``, and
``Stroma_Mask.tiff``. This script assembles those masks into a label map, then
samples positive bags per class and negative bags from background, mirroring the
multi-scale pipeline used by ``sample_camelyon_multiscale_patches.py``.
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

from sample_tcga_multiscale_patches import PatchTile, ValueLabelEncoder, compute_tissue_fraction


@dataclass
class BagResult:
    """Stores the hierarchy node and associated tiles for a sampled bag."""

    node: Dict[str, object]
    tiles: List[PatchTile]


SUPPORTED_IMAGE_EXTS = (".tif", ".tiff")
SUPPORTED_MASK_EXTS = (".tif", ".tiff", ".png")

CLASS_INFO = [
    {"id": 0, "name": "Background", "abbr": "BKG"},
    {"id": 1, "name": "G3", "abbr": "G3"},
    {"id": 2, "name": "G4", "abbr": "G4"},
    {"id": 3, "name": "Normal", "abbr": "NOR"},
    {"id": 4, "name": "Stroma", "abbr": "STR"},
]

# Expected mask file stems per class (case-insensitive).
CLASS_MASK_STEMS = {
    "G3": "G3_Mask",
    "G4": "G4_Mask",
    "Normal": "Normal_Mask",
    "Stroma": "Stroma_Mask",
}


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


def _find_class_mask(mask_dir: Path, class_name: str, mask_exts: Sequence[str]) -> Optional[Path]:
    stem = CLASS_MASK_STEMS.get(class_name)
    if stem is None:
        return None
    lower_exts = [ext.lower() for ext in mask_exts]
    for ext in lower_exts:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Try case variations for robustness
        alt = mask_dir / f"{stem.lower()}{ext}"
        if alt.exists():
            return alt
    return None


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.array(mask)


def _build_label_map(mask_dir: Path, mask_exts: Sequence[str]) -> np.ndarray:
    # Will be sized after first discovered mask; start empty.
    label_map: Optional[np.ndarray] = None

    for class_info in CLASS_INFO:
        if class_info["id"] == 0:
            continue
        mask_path = _find_class_mask(mask_dir, class_info["name"], mask_exts)
        if mask_path is None:
            continue
        mask = _load_binary_mask(mask_path)
        if label_map is None:
            label_map = np.zeros_like(mask, dtype=np.int16)
        elif label_map.shape != mask.shape:
            raise ValueError(
                f"Mask {mask_path} has shape {mask.shape}, expected {label_map.shape} to match other masks in {mask_dir}"
            )
        positives = mask > 0
        label_map[positives] = class_info["id"]

    if label_map is None:
        raise ValueError(f"No class masks found in {mask_dir}")

    return label_map


def _compute_max_margin(patch_size: int, scales: Sequence[float]) -> int:
    if not scales:
        return patch_size // 2
    max_scale = max(scales) / scales[0]
    margin = int(patch_size * max_scale / 2)
    return max(1, margin)


def _sample_centers(label_map: np.ndarray, count: int, margin: int, class_id: int) -> List[Tuple[int, int]]:
    if count <= 0:
        return []
    coords = np.argwhere(label_map == class_id)
    if len(coords) == 0:
        return []
    height, width = label_map.shape
    valid = [
        (int(x), int(y))
        for y, x in coords
        if margin <= x < width - margin and margin <= y < height - margin
    ]
    if not valid:
        return []
    random.shuffle(valid)
    return valid[:count]


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


def sample_image(
    image_path: Path,
    mask_root: Path,
    out_root: Path,
    positive_bags: int,
    negative_bags: int,
    seed: int,
    scales: Sequence[float],
    patch_size: int,
    tissue_threshold: float,
    background_threshold: int,
    label_encoder: ValueLabelEncoder,
    mask_exts: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    random.seed(seed)

    mask_dir = mask_root / image_path.stem
    if not mask_dir.exists():
        raise ValueError(f"Mask directory {mask_dir} does not exist for image {image_path.name}")

    image = _load_image(image_path)
    label_map = _build_label_map(mask_dir, mask_exts)
    if label_map.shape[:2] != (image.height, image.width):
        raise ValueError(
            f"Masks under {mask_dir} do not match image dimensions for {image_path.name}: "
            f"expected {(image.height, image.width)}, got {label_map.shape[:2]}"
        )

    out_dir = _ensure_output_dir(out_root, image_path)
    margin = _compute_max_margin(patch_size, scales)

    bags: List[Dict[str, object]] = []
    patch_rows: List[Dict[str, str]] = []
    patient_id = _generate_patient_id(image_path)
    background_id = CLASS_INFO[0]["id"]

    bag_index = 0
    for class_info in CLASS_INFO:
        class_id = class_info["id"]
        class_name = class_info["name"]
        if class_id == background_id:
            continue
        centers = _sample_centers(label_map, positive_bags, margin, class_id)
        if not centers:
            continue
        label_id, label_name = label_encoder.encode(class_name)
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

    centers_background = _sample_centers(label_map, negative_bags, margin, background_id)
    if centers_background:
        label_id, label_name = label_encoder.encode(CLASS_INFO[0]["name"])
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
    parser.add_argument("image_dir", type=Path, help="Root directory containing AGCC images (TIFF)")
    parser.add_argument(
        "mask_dir",
        type=Path,
        help="Root directory containing per-image mask folders (one folder per image stem)",
    )
    parser.add_argument("output_dir", type=Path, help="Directory where patches and manifests are written")
    parser.add_argument(
        "--positive-bags",
        type=int,
        default=5,
        help="Number of bags to sample per annotated class (excluding background)",
    )
    parser.add_argument(
        "--negative-bags", type=int, default=5, help="Number of background bags to sample per image"
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
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=list(SUPPORTED_IMAGE_EXTS),
        help="Image filename extensions to include (default: TIFF)",
    )
    parser.add_argument(
        "--mask-exts",
        nargs="+",
        default=list(SUPPORTED_MASK_EXTS),
        help="Mask filename extensions to search for inside each per-image folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = _gather_images(args.image_dir, args.image_exts)
    if not images:
        raise SystemExit(f"No images found under {args.image_dir} with extensions {args.image_exts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_encoder = ValueLabelEncoder({info["name"].lower(): (info["id"], info["name"]) for info in CLASS_INFO})

    all_rows: List[Dict[str, str]] = []
    for index, image_path in enumerate(images):
        try:
            _, patch_rows = sample_image(
                image_path,
                args.mask_dir,
                args.output_dir,
                args.positive_bags,
                args.negative_bags,
                args.seed + index,
                args.magnifications,
                args.patch_size,
                args.tissue_threshold,
                args.background_threshold,
                label_encoder,
                args.mask_exts,
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
