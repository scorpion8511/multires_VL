"""Plot Gabor energy score histograms across datasets and models.

This utility mirrors the reference style shown in the prompt: for each
dataset, it overlays energy-score distributions from multiple models on the
same axes. The script expects tabular inputs containing at least the columns
``dataset``, ``model``, and ``energy``. An optional ``accuracy`` column can be
used to annotate legend entries (``Model (acc:95.5%)``). When no input files
are provided, the script falls back to a synthetic TCGA-like example so the
command can run out-of-the-box.

Example usage with a single CSV (comma-separated) containing all datasets::

    python scripts/plot_gabor_energy.py \
      --input energies.csv \
      --datasets CIFAR10 Caltech-101 TCGA \
      --models GoogLeNet "ResNet-50" "DenseNet-169" InceptionV3 \
      --output gabor_energy.png

Example usage with multiple CSV/TSV files (auto-detected by extension)::

    python scripts/plot_gabor_energy.py \
      --input tcga_scores.csv \
      --input cifar_scores.tsv \
      --output gabor_energy.png

The script supports CSV, TSV, and JSON Lines (``.jsonl``) inputs. Each input
file should contain one energy score per row. If ``--datasets`` or
``--models`` are provided, they are used to filter and order the subplots and
legend entries respectively.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUPPORTED_EXTS = {".csv", ".tsv", ".jsonl", ".json"}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported input extension: {path}")

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".json"}:
        # Load JSON Lines if multiple objects; fall back to a JSON array.
        with path.open("r") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "{":
                return pd.read_json(path, lines=True)
            return pd.read_json(path)

    raise ValueError(f"Unhandled extension: {path}")


def _maybe_synthesize() -> pd.DataFrame:
    """Create a synthetic TCGA example so the script can run standalone."""

    rng = np.random.default_rng(0)
    entries = []
    datasets = ["TCGA"]
    models = [
        ("GoogLeNet", 95.5),
        ("ResNet-34", 96.1),
        ("DenseNet-169", 96.8),
        ("InceptionV3", 92.8),
    ]
    for dataset in datasets:
        for model, acc in models:
            # Simulate slightly different energy distributions per model.
            mean = 20 + (hash(model) % 5) * 0.5
            sigma = 3.0 + (hash(model) % 3) * 0.4
            scores = rng.normal(loc=mean, scale=sigma, size=600)
            scores = np.clip(scores, 0, None)
            for s in scores:
                entries.append({
                    "dataset": dataset,
                    "model": model,
                    "energy": float(s),
                    "accuracy": acc,
                })
    return pd.DataFrame(entries)


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "energy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return df


def load_energy_tables(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        return _normalize_labels(_maybe_synthesize())

    frames: List[pd.DataFrame] = []
    for p in paths:
        frames.append(_read_table(p))
    return _normalize_labels(pd.concat(frames, ignore_index=True))


def filter_and_order(
    df: pd.DataFrame,
    datasets: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
) -> pd.DataFrame:
    filtered = df
    if datasets:
        filtered = filtered[filtered["dataset"].isin(datasets)]
        filtered["dataset"] = pd.Categorical(filtered["dataset"], categories=datasets, ordered=True)
    if models:
        filtered = filtered[filtered["model"].isin(models)]
        filtered["model"] = pd.Categorical(filtered["model"], categories=models, ordered=True)
    return filtered


def _format_label(row: Mapping[str, object]) -> str:
    acc = row.get("accuracy")
    if acc is None or (isinstance(acc, float) and math.isnan(acc)):
        return str(row["model"])
    return f"{row['model']} (acc:{float(acc):.1f}%)"


def plot_energy(
    df: pd.DataFrame,
    output: Path,
    bins: int = 40,
    alpha: float = 0.6,
) -> None:
    if df.empty:
        raise ValueError("No data to plot after filtering.")

    datasets = list(pd.unique(df["dataset"]))
    ncols = min(2, len(datasets))
    nrows = int(math.ceil(len(datasets) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        subset = df[df["dataset"] == dataset]
        models = list(pd.unique(subset["model"]))
        for model in models:
            scores = subset[subset["model"] == model]["energy"].astype(float)
            label_row = subset[subset["model"] == model].iloc[0].to_dict()
            ax.hist(
                scores,
                bins=bins,
                density=True,
                alpha=alpha,
                label=_format_label(label_row),
                edgecolor="none",
            )
        ax.set_title(dataset)
        ax.set_xlabel("Energy Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.2, linestyle="--")

    # Hide any unused subplots
    for j in range(len(datasets), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Path(s) to CSV/TSV/JSONL files with columns dataset, model, energy, accuracy (optional).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Datasets to include and order in the plot (e.g., CIFAR10 Caltech-101 TCGA).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to include and order (e.g., GoogLeNet ResNet-34 DenseNet-169 InceptionV3).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Histogram alpha/opacity (default: 0.6).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gabor_energy.png"),
        help="Output image path (default: gabor_energy.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_energy_tables(args.input)
    df = filter_and_order(df, args.datasets, args.models)
    plot_energy(df, args.output, bins=args.bins, alpha=args.alpha)


if __name__ == "__main__":
    main()
