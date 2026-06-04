#!/usr/bin/env python
"""CLI: merge NMP-only scored shards and draw a density-stratified Tier 2 v2 subsample.

Reads one or more `tier2_nmp_screen_*` scored parquets (e.g. the length-15
shards or the multi-length S2 output), concatenates + de-duplicates by
protein_id, then samples over a density column (default `coverage_fraction`)
using the stratified water-filling samplers in
`inverse_folding.evaluation.sampling` (PLAN_DATA_SEL §12).

Writes the selected-subset parquet + a histogram JSON for review. Pure CPU and
fully deterministic given `--seed`, so it can be rerun cheaply to tune the
Gaussian (μ / peak-to-tail / min-per-bin) before the S3 selection is committed.

Usage:
    # S1: uniform downsample of the full length-15 pool to ~5000
    python scripts/select_tier2_v2.py \
        --scored-glob 'work/.../tier2_nmp_screen_HLA-DRB1_07_01_l15.shard*of16.parquet' \
        --mode uniform --n-target 5000 --n-bins 20 \
        --output-parquet work/.../tier2_v2_s1_uniform_HLA-DRB1_07_01.parquet

    # S3: Gaussian downsample of the multi-length S2 pool to ~3000
    python scripts/select_tier2_v2.py \
        --scored-parquet work/.../tier2_nmp_screen_HLA-DRB1_07_01_lall.parquet \
        --mode gaussian --n-target 3000 --n-bins 20 --peak-to-tail 3.0 --min-per-bin 40 \
        --output-parquet work/.../tier2_prescreened_v2_HLA-DRB1_07_01.parquet
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

from inverse_folding.evaluation.sampling import (
    sample_uniform_bins,
    sample_gaussian_bins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge NMP-only scored shards and density-stratified-sample Tier 2 v2."
    )
    parser.add_argument("--scored-glob", nargs="*", default=[],
                        help="Glob pattern(s) for scored shard parquets.")
    parser.add_argument("--scored-parquet", nargs="*", default=[],
                        help="Explicit scored parquet path(s).")
    parser.add_argument("--value-col", default="coverage_fraction",
                        help="Density column to stratify on (default coverage_fraction).")
    parser.add_argument("--mode", choices=["uniform", "gaussian"], required=True)
    parser.add_argument("--n-target", type=int, required=True)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--min-per-bin", type=int, default=0)
    parser.add_argument("--peak-to-tail", type=float, default=3.0,
                        help="Gaussian central:tail bin-weight ratio (gaussian mode).")
    parser.add_argument("--mu", type=float, default=None,
                        help="Gaussian center (default = median of value-col).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-histogram", default=None,
                        help="Histogram JSON path (default = output stem + .histogram.json).")
    parser.add_argument("--selection-reason", default=None)
    return parser.parse_args()


def _load_scored(globs, explicit) -> pd.DataFrame:
    paths = []
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    paths.extend(explicit)
    paths = list(dict.fromkeys(paths))  # de-dup, preserve order
    if not paths:
        print("ERROR: no scored parquets matched --scored-glob / --scored-parquet.")
        sys.exit(1)
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset="protein_id", keep="first").reset_index(drop=True)
    print(f"  Loaded {len(paths)} parquet(s): {n_before} rows -> "
          f"{len(df)} unique protein_id")
    return df


def main() -> int:
    args = parse_args()
    df = _load_scored(args.scored_glob, args.scored_parquet)

    if args.value_col not in df.columns:
        print(f"ERROR: value column '{args.value_col}' not in {list(df.columns)}")
        return 1

    vals = df[args.value_col]
    print(f"  {args.value_col}: min={vals.min():.4f} median={vals.median():.4f} "
          f"max={vals.max():.4f} mean={vals.mean():.4f}")

    if args.mode == "uniform":
        sel_idx, hist = sample_uniform_bins(
            df, args.value_col, n_target=args.n_target, n_bins=args.n_bins,
            min_per_bin=args.min_per_bin, seed=args.seed,
        )
    else:
        sel_idx, hist = sample_gaussian_bins(
            df, args.value_col, n_target=args.n_target, n_bins=args.n_bins,
            mu=args.mu, peak_to_tail=args.peak_to_tail,
            min_per_bin=args.min_per_bin, seed=args.seed,
        )

    selected = df.loc[sel_idx].copy()
    reason = args.selection_reason or f"nmp-only-{args.mode}-v2"
    selected["selection_reason"] = reason

    os.makedirs(os.path.dirname(os.path.abspath(args.output_parquet)), exist_ok=True)
    selected.to_parquet(args.output_parquet, index=False)

    hist_path = args.output_histogram or (
        args.output_parquet.rsplit(".parquet", 1)[0] + ".histogram.json"
    )
    summary = {
        "mode": args.mode, "value_col": args.value_col,
        "n_target": args.n_target, "n_bins": args.n_bins,
        "min_per_bin": args.min_per_bin, "peak_to_tail": args.peak_to_tail,
        "mu": args.mu, "seed": args.seed,
        "n_scored": int(len(df)), "n_selected": int(len(selected)),
        "selected_value_median": float(selected[args.value_col].median()),
        "selected_value_mean": float(selected[args.value_col].mean()),
        "bins": hist.to_dict(orient="records"),
    }
    with open(hist_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Selected {len(selected)}/{len(df)} ({args.mode}) -> {args.output_parquet}")
    print(f"  Histogram -> {hist_path}")
    print("  Per-bin (lo, available, selected):")
    for _, r in hist.iterrows():
        bar = "#" * int(r["n_selected"] / max(1, args.n_target / args.n_bins) * 4)
        print(f"    [{r['bin_lo']:.3f},{r['bin_hi']:.3f})  "
              f"avail={int(r['n_available']):5d}  sel={int(r['n_selected']):4d}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
