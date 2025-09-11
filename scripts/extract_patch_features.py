#!/usr/bin/env python3
"""Extract features from multi-scale patch bags.

This script loads the patch hierarchy produced by
``sample_tcga_multiscale_patches.py`` and computes image features for every
patch in each bag using a specified encoder model. Features are stored in
HDF5 files preserving the patch identifiers so that the hierarchy can be
reconstructed.  Both ``bags.json`` (single JSON object) and ``bags.jsonl``
(``jsonlines``) metadata formats are supported for compatibility with
earlier runs of the sampling script.

If the encoder weights are not provided via environment variable
(`UNI_CKPT_PATH` or `CONCH_CKPT_PATH`), supply ``--ckpt_path`` to point to the
checkpoint file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow running the script directly via its path by adding the repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import get_encoder


class PatchDataset(Dataset):
    """Dataset loading patch images from a list of paths."""

    def __init__(self, paths: List[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - simple getter
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def gather_ids_paths(bag: dict, bag_dir: Path) -> Tuple[List[str], List[Path]]:
    """Return lists of patch ids and corresponding image paths."""
    ids = [bag["id"]]
    paths = [bag_dir / bag["file"]]
    for child in bag.get("children", []):
        child_ids, child_paths = gather_ids_paths(child, bag_dir)
        ids.extend(child_ids)
        paths.extend(child_paths)
    return ids, paths


def process_bag(bag: dict, bag_dir: Path, out_path: Path, model, transform,
                device: torch.device, batch_size: int) -> None:
    ids, paths = gather_ids_paths(bag, bag_dir)
    dataset = PatchDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).cpu().numpy()
            feats.append(out)

    features = np.concatenate(feats, axis=0)
    ids_arr = np.array(ids, dtype="S")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ids", data=ids_arr)
        f.create_dataset("features", data=features)


def process_slide(slide_dir: Path, out_dir: Path, model, transform,
                  device: torch.device, batch_size: int) -> None:
    """Extract features for all bags under ``slide_dir``."""

    json_path = slide_dir / "bags.json"
    loader = json.load
    if not json_path.exists():
        json_path = slide_dir / "bags.jsonl"
        if not json_path.exists():
            raise FileNotFoundError(f"No bags.json[.l] found in {slide_dir}")

        def _load_jsonl(path: Path) -> List[dict]:
            with open(path) as f:
                # ``bags.jsonl`` files may actually contain either true JSONL
                # records or a single JSON array/object despite the extension.
                # Peek at the first non-whitespace character to determine how
                # to parse the file and fall back to line-wise loading.
                first_char = None
                while True:
                    ch = f.read(1)
                    if not ch:  # empty file
                        return []
                    if not ch.isspace():
                        first_char = ch
                        break
                f.seek(0)

                if first_char in "[{":
                    data = json.load(f)
                    # Some preprocessors emit a single JSON object rather than
                    # a list; wrap it for consistency.
                    return data if isinstance(data, list) else [data]

                # Otherwise treat as traditional JSON Lines, skipping blanks
                return [json.loads(line) for line in f if line.strip()]

        loader = _load_jsonl

    bags = loader(json_path)

    slide_out_dir = out_dir / slide_dir.name
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    for bag in tqdm(bags, desc=f"{slide_dir.name}"):
        out_path = slide_out_dir / f"{bag['id']}.h5"
        process_bag(bag, slide_dir, out_path, model, transform, device,
                    batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from multi-scale patch bags")
    parser.add_argument("patch_root", type=Path,
                        help="Directory containing slide subfolders or a single slide directory")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--model_name", default="uni_v1",
                        help="Encoder model to use")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--target_img_size", type=int, default=224,
                        help="Input size expected by the model")
    parser.add_argument("--ckpt_path", type=Path, default=None,
                        help="Path to model checkpoint if not set via env var")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = get_encoder(
        args.model_name,
        target_img_size=args.target_img_size,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path is not None else None,
    )
    model.eval().to(device)

    def gather_slide_dirs(root: Path) -> List[Path]:
        """Return slide directories under ``root``.

        ``root`` itself is treated as a slide directory if it contains the
        bag metadata file.  Subdirectories starting with ``.`` or lacking the
        metadata file are ignored.
        """

        has_bags = lambda p: (p / "bags.json").exists() or (p / "bags.jsonl").exists()
        if has_bags(root):
            return [root]
        slide_dirs = []
        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith('.') and has_bags(p):
                slide_dirs.append(p)
        return slide_dirs

    slide_dirs = gather_slide_dirs(args.patch_root)
    for slide_dir in slide_dirs:
        process_slide(slide_dir, args.out_dir, model, transform, device,
                      args.batch_size)


if __name__ == "__main__":
    main()
