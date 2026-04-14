#!/usr/bin/env python
"""Plot benchmark score histograms from benchmark_iedb_test.py output JSON.

Creates:
  1. One global 2x2 summary figure aggregating all proteins
  2. One per-protein 2x2 figure per protein_id

Plots are saved under `<json_dir>/<json_stem>_histograms/` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_KEYS = (
    ("head_primary", "Head Primary"),
    ("nmp_primary", "NMP Primary"),
    ("head_conditional", "Head Conditional"),
    ("nmp_conditional", "NMP Conditional"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot score histograms from benchmark JSON output.",
    )
    p.add_argument("--input-json", required=True, help="Benchmark JSON path.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for PNGs "
                        "(default: <json_dir>/<json_stem>_histograms).")
    p.add_argument("--bins", type=int, default=40,
                   help="Number of histogram bins (default: 40).")
    return p.parse_args()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _flatten_distributions(per_protein: list[dict]) -> dict[str, dict[str, list[float]]]:
    global_dists: dict[str, dict[str, list[float]]] = {
        key: {"pos_scores": [], "neg_scores": []} for key, _ in PLOT_KEYS
    }
    for row in per_protein:
        dists = row.get("score_distributions", {})
        for key, _ in PLOT_KEYS:
            block = dists.get(key, {})
            global_dists[key]["pos_scores"].extend(block.get("pos_scores", []))
            global_dists[key]["neg_scores"].extend(block.get("neg_scores", []))
    return global_dists


def _plot_one(ax, title: str, pos_scores: list[float], neg_scores: list[float], bins: int) -> None:
    ax.set_title(title)
    if neg_scores:
        ax.hist(
            neg_scores, bins=bins, alpha=0.55, density=True,
            color="#d95f02", label=f"negative (n={len(neg_scores)})",
        )
    if pos_scores:
        ax.hist(
            pos_scores, bins=bins, alpha=0.55, density=True,
            color="#1b9e77", label=f"positive (n={len(pos_scores)})",
        )
    if not pos_scores and not neg_scores:
        ax.text(0.5, 0.5, "No scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def _plot_grid(title: str, dist_map: dict[str, dict[str, list[float]]], bins: int, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    for ax, (key, label) in zip(axes.flat, PLOT_KEYS):
        block = dist_map.get(key, {})
        _plot_one(
            ax,
            label,
            block.get("pos_scores", []),
            block.get("neg_scores", []),
            bins,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json)
    with open(input_path) as f:
        payload = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else (
        input_path.parent / f"{input_path.stem}_histograms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_protein = payload.get("per_protein", [])
    global_dists = _flatten_distributions(per_protein)

    _plot_grid(
        title=f"Benchmark Score Histograms: {input_path.stem}",
        dist_map=global_dists,
        bins=args.bins,
        out_path=str(output_dir / "global_summary.png"),
    )

    for row in per_protein:
        protein_id = row.get("protein_id", "unknown")
        dists = row.get("score_distributions", {})
        out_path = output_dir / f"{_safe_name(protein_id)}.png"
        _plot_grid(
            title=f"{protein_id} score histograms",
            dist_map=dists,
            bins=args.bins,
            out_path=str(out_path),
        )

    print(f"Saved histogram plots to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
