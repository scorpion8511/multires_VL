#!/usr/bin/env python3
"""Sample multi-scale patch hierarchies from TCGA WSIs.

This script replicates the patch sampling procedure described in the
paper referenced as ``2504.18856v1``.

For each WSI slide, the script randomly selects 20 seed locations.  At
each location a hierarchy of patches is extracted that all share the
same center but differ in magnification:

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
import json
import random
from pathlib import Path
from typing import Dict, List

import openslide
from PIL import Image


def save_patch(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    level: int,
    size: int,
    requested_ds: float,
    out_path: Path,
) -> None:
    """Read and save a region from ``slide`` at the desired downsample."""

    actual_ds = slide.level_downsamples[level]
    # Compute region in level-0 coordinates covering the requested area
    x0 = int(center_x - size * requested_ds / 2)
    y0 = int(center_y - size * requested_ds / 2)
    region_size = int(size * requested_ds / actual_ds)

    img = slide.read_region((x0, y0), level, (region_size, region_size)).convert("RGB")
    if region_size != size:
        img = img.resize((size, size), Image.BILINEAR)
    img.save(out_path)


def generate_hierarchy(
    slide: openslide.OpenSlide,
    center_x: int,
    center_y: int,
    mpps: List[float],
    prefix: str,
    out_dir: Path,
    base_mpp: float,
    size: int = 512,
) -> Dict:
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
    save_patch(slide, center_x, center_y, level, size, requested_ds, out_path)

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
        )
        node["children"].append(child)
    return node


def process_wsi(slide_path: Path, out_root: Path, num_parents: int = 20,
                seed: int = 0) -> None:
    """Process a single WSI and generate multi-scale patch bags."""
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
    margin = int(size * parent_ds / 2)
    max_cx = slide_width - margin
    max_cy = slide_height - margin

    bags = []
    for i in range(num_parents):
        center_x = random.randint(margin, max_cx)
        center_y = random.randint(margin, max_cy)
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
        )
        bags.append(bag)

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
