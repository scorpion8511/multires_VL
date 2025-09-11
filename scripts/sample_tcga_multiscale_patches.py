#!/usr/bin/env python3
"""Sample multi-scale patch hierarchies from TCGA WSIs.

This script replicates the patch sampling procedure described in the
paper referenced as ``2504.18856v1``.

For each WSI slide, the script randomly selects 20 parent patches at
5× magnification (2 µm/px) of size 512×512 pixels.  Each parent patch is
subdivided into children at higher magnifications while keeping the same
field of view:

* 10× (1 µm/px)  : 4 children per parent
* 20× (0.5 µm/px): 4 children per 10× child (16 total)
* 40× (0.25 µm/px): 4 children per 20× child (64 total)

For every parent patch, all descendants (84 children + parent) form a
"visual bag" whose structure (parent→children links) is preserved in a
JSON file.  Alignment is only defined between a parent patch and its
direct children.

The script expects whole-slide images accessible via OpenSlide.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import openslide


def pick_level(slide: openslide.OpenSlide, target_mpp: float) -> int:
    """Return the best level index approximating ``target_mpp``."""
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    downsample = target_mpp / base_mpp
    return slide.get_best_level_for_downsample(downsample)


def save_patch(slide: openslide.OpenSlide, x0: int, y0: int, level: int,
               size: int, out_path: Path) -> None:
    """Read and save a region from ``slide``."""
    img = slide.read_region((x0, y0), level, (size, size)).convert("RGB")
    img.save(out_path)


def generate_children(slide: openslide.OpenSlide, x0: int, y0: int,
                      level: int, size: int, target_mpps: List[float],
                      prefix: str, out_dir: Path,
                      base_mpp: float) -> List[Dict]:
    """Recursively generate children patches.

    Args:
        slide: OpenSlide object.
        x0, y0: top-left coordinates in level 0 reference frame.
        level: current level index.
        size: patch size (pixels) at the current level.
        target_mpps: list of desired mpp values for subsequent levels.
        prefix: patch name prefix.
        out_dir: directory in which to store patches.
        base_mpp: level-0 microns per pixel.

    Returns:
        A list of dictionaries describing the child hierarchy.
    """
    if not target_mpps:
        return []

    current_downsample = slide.level_downsamples[level]
    step = int(size * current_downsample / 2)
    child_mpp = target_mpps[0]
    child_level = slide.get_best_level_for_downsample(child_mpp / base_mpp)

    children = []
    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for idx, (ox, oy) in enumerate(offsets):
        child_x0 = x0 + ox * step
        child_y0 = y0 + oy * step
        child_prefix = f"{prefix}_{idx}"
        child_path = out_dir / f"{child_prefix}.png"
        save_patch(slide, child_x0, child_y0, child_level, size, child_path)
        grandchildren = generate_children(
            slide,
            child_x0,
            child_y0,
            child_level,
            size,
            target_mpps[1:],
            child_prefix,
            out_dir,
            base_mpp,
        )
        children.append({
            "id": child_prefix,
            "file": child_path.name,
            "children": grandchildren,
        })
    return children


def process_wsi(slide_path: Path, out_root: Path, num_parents: int = 20,
                seed: int = 0) -> None:
    """Process a single WSI and generate multi-scale patch bags."""
    random.seed(seed)
    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))

    parent_mpp = 2.0  # 5×
    parent_level = pick_level(slide, parent_mpp)
    parent_downsample = slide.level_downsamples[parent_level]
    slide_width, slide_height = slide.dimensions

    out_dir = out_root / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    bags = []
    size = 512
    max_x = slide_width - int(size * parent_downsample)
    max_y = slide_height - int(size * parent_downsample)

    for i in range(num_parents):
        x0 = random.randint(0, max_x)
        y0 = random.randint(0, max_y)
        parent_name = f"parent_{i}"
        parent_path = out_dir / f"{parent_name}.png"
        save_patch(slide, x0, y0, parent_level, size, parent_path)

        children = generate_children(
            slide,
            x0,
            y0,
            parent_level,
            size,
            [1.0, 0.5, 0.25],
            parent_name,
            out_dir,
            base_mpp,
        )

        bags.append({
            "id": parent_name,
            "file": parent_path.name,
            "children": children,
        })

    with open(out_dir / "bags.json", "w") as f:
        json.dump(bags, f, indent=2)

    slide.close()


def gather_slides(wsi_root: Path) -> List[Path]:
    """Return a list of WSI files under ``wsi_root``.

    ``wsi_root`` may point to a single file or a directory.  If a directory is
    provided, it is searched recursively for known WSI file extensions.
    """
    exts = {".svs", ".tif", ".tiff", ".ndpi"}
    if wsi_root.is_file():
        return [wsi_root] if wsi_root.suffix.lower() in exts else []
    return [p for p in wsi_root.rglob("*") if p.suffix.lower() in exts]


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

    args = parser.parse_args()

    slides = gather_slides(args.wsi_dir)
    if not slides:
        raise FileNotFoundError(f"No WSI files found at {args.wsi_dir}")

    for slide_path in slides:
        print(f"Processing {slide_path}")
        process_wsi(
            slide_path,
            args.out_dir,
            num_parents=args.num_parents,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
