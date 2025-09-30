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
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import openslide
from PIL import Image
import numpy as np


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
) -> Optional[Dict[str, object]]:
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
        node["children"].append(child)
    return node


def process_wsi(
    slide_path: Path,
    out_root: Path,
    num_parents: int = 20,
    seed: int = 0,
    tissue_threshold: float = 0.75,
    background_threshold: int = 220,
) -> int:
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
    margin = int(size * parent_ds / 2)
    max_cx = slide_width - margin
    max_cy = slide_height - margin

    bags = []
    i = 0
    attempts = 0
    max_attempts = num_parents * 20 if num_parents > 0 else 0
    while i < num_parents and (max_attempts == 0 or attempts < max_attempts):
        attempts += 1
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
            tissue_threshold,
            background_threshold,
        )
        if bag:
            bags.append(bag)
            i += 1

    if i < num_parents:
        print(
            f"Warning: only collected {i} patch bags (requested {num_parents}) "
            f"for {slide_path.name}."
        )

    with open(out_dir / "bags.json", "w") as f:
        json.dump(bags, f, indent=2)

    slide.close()

    return len(bags)


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
            bags_collected = process_wsi(
                slide_path,
                args.out_dir,
                num_parents=args.num_parents,
                seed=args.seed,
                tissue_threshold=args.tissue_threshold,
                background_threshold=args.background_threshold,
            )
            if bags_collected < args.num_parents:
                slide_out_dir = args.out_dir / slide_path.stem
                if slide_out_dir.exists():
                    shutil.rmtree(slide_out_dir, ignore_errors=True)
                reason = "replacement" if is_replacement else "selected"
                print(
                    f"Scheduling an additional slide for subtype '{subtype}' "
                    f"because {slide_path.name} ({reason}) yielded "
                    f"{bags_collected} patch bags."
                )
                return False

            counts[subtype] += 1
            accepted_pairs.append(pair)
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
            process_wsi(
                slide_path,
                args.out_dir,
                num_parents=args.num_parents,
                seed=args.seed,
                tissue_threshold=args.tissue_threshold,
                background_threshold=args.background_threshold,
            )


if __name__ == "__main__":
    main()
