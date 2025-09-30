#!/usr/bin/env python3
"""Build a manifest CSV enumerating sampled patch tiles.

The sampling pipeline stores a ``bags.json`` file for every processed slide
that describes the multi-resolution hierarchy extracted from each seed.  This
script flattens those hierarchies into a tabular representation where each row
contains:

* the patch identifier and filename
* the slide metadata columns from the balanced selection CSV
* convenience columns indicating the slide stem, bag index, and scale suffix

Example usage::

    python scripts/build_patch_manifest.py \
        /path/to/patches \
        /path/to/patches/selected_wsi.csv \
        /path/to/patch_manifest.csv

The resulting CSV makes it easy to join patch-level features with the original
slide metadata when training downstream models.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Sequence


@dataclass
class PatchRecord:
    """Description of a single patch tile."""

    slide_stem: str
    bag_index: int
    patch_id: str
    patch_file: str
    rel_path: str
    scale_suffix: str


def _load_selection_table(csv_path: Path) -> tuple[List[Dict[str, str]], Sequence[str]]:
    """Return metadata rows and the header order from ``csv_path``."""

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"Selection CSV {csv_path} is empty")
        return rows, reader.fieldnames or []


def _iter_bag_nodes(bag: MutableMapping[str, object]) -> Iterator[MutableMapping[str, object]]:
    """Yield every node contained within ``bag`` (depth-first)."""

    stack = [bag]
    while stack:
        node = stack.pop()
        yield node
        children = node.get("children", []) if isinstance(node, dict) else []
        if isinstance(children, list):
            for child in reversed(children):
                if isinstance(child, dict):
                    stack.append(child)


def _collect_patches(bags_path: Path) -> List[PatchRecord]:
    """Parse ``bags.json`` and return flattened patch descriptors."""

    with bags_path.open() as handle:
        bags = json.load(handle)

    records: List[PatchRecord] = []
    slide_stem = bags_path.parent.name
    for bag_index, bag in enumerate(bags):
        if not isinstance(bag, dict):
            continue
        for node in _iter_bag_nodes(bag):
            patch_id = str(node.get("id", ""))
            patch_file = str(node.get("file", ""))
            if not patch_file:
                continue
            rel_path = str(Path(slide_stem) / patch_file)
            scale_suffix = patch_id.split("_")[-1] if patch_id else ""
            records.append(
                PatchRecord(
                    slide_stem=slide_stem,
                    bag_index=bag_index,
                    patch_id=patch_id,
                    patch_file=patch_file,
                    rel_path=rel_path,
                    scale_suffix=scale_suffix,
                )
            )
    return records


def build_manifest(
    patch_root: Path,
    selection_rows: Iterable[Dict[str, str]],
    *,
    slide_column: str,
    bags_filename: str,
) -> List[Dict[str, str]]:
    """Return manifest rows merging metadata with patch descriptors."""

    manifest: List[Dict[str, str]] = []

    for row in selection_rows:
        slide_value = row.get(slide_column, "").strip()
        if not slide_value:
            print(
                "Warning: skipping metadata row without a value in column "
                f"'{slide_column}'."
            )
            continue

        slide_path = Path(slide_value)
        slide_stem = slide_path.stem
        slide_dir = patch_root / slide_stem
        bags_path = slide_dir / bags_filename

        if not slide_dir.exists():
            print(
                f"Warning: patch directory {slide_dir} does not exist; "
                "skipping."
            )
            continue
        if not bags_path.exists():
            print(
                f"Warning: metadata file {bags_path} not found for slide "
                f"'{slide_stem}'; skipping."
            )
            continue

        patch_records = _collect_patches(bags_path)
        if not patch_records:
            print(f"Warning: no patch entries found in {bags_path}; skipping.")
            continue

        for record in patch_records:
            out_row = dict(row)
            out_row.update(
                {
                    "slide_stem": record.slide_stem,
                    "bag_index": str(record.bag_index),
                    "patch_id": record.patch_id,
                    "patch_file": record.patch_file,
                    "patch_relpath": record.rel_path,
                    "patch_scale": record.scale_suffix,
                }
            )
            manifest.append(out_row)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV manifest describing every patch tile produced "
            "by the balanced sampling pipeline."
        )
    )
    parser.add_argument(
        "patch_root",
        type=Path,
        help="Directory containing slide-specific patch folders",
    )
    parser.add_argument(
        "selection_csv",
        type=Path,
        help="Balanced slide selection CSV generated by the sampler",
    )
    parser.add_argument("output_csv", type=Path, help="Destination path for the manifest")
    parser.add_argument(
        "--slide-column",
        type=str,
        default="resolved_path",
        help=(
            "Column in the selection CSV that stores the slide path or name "
            "used to derive the patch directory."
        ),
    )
    parser.add_argument(
        "--bags-filename",
        type=str,
        default="bags.json",
        help="Name of the JSON file that contains the sampled bag hierarchy",
    )

    args = parser.parse_args()

    if not args.patch_root.exists():
        raise FileNotFoundError(f"Patch directory {args.patch_root} does not exist")
    if not args.selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV {args.selection_csv} does not exist")

    rows, fieldnames = _load_selection_table(args.selection_csv)
    manifest = build_manifest(
        args.patch_root,
        rows,
        slide_column=args.slide_column,
        bags_filename=args.bags_filename,
    )
    if not manifest:
        raise RuntimeError("No patch rows were collected; nothing to write")

    base_fields = list(fieldnames)
    extra_fields = [
        "slide_stem",
        "bag_index",
        "patch_id",
        "patch_file",
        "patch_relpath",
        "patch_scale",
    ]
    field_order = base_fields + [f for f in extra_fields if f not in base_fields]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)

    print(f"Wrote {len(manifest)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
