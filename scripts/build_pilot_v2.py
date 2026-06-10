#!/usr/bin/env python
"""Build the pilot diagnostic subset, band-aligned on ``coverage_fraction``.

pilot = (Tier 1 IEDB-gold anchors whose NMP ``coverage_fraction`` falls in the
same mid-load band used for Tier 2) + (Tier 2 drawn from that band with a light
``--n-sub-bins`` stratification), to a fixed ``--n-total``.

Both Tier 1 and Tier 2 ``coverage_fraction`` are read from the ``fast_v2``
parquet (the only surviving artifact that carries the column for every member),
and the selected rows are a strict subset of ``fast_v2`` -- so they inherit the
canonical if_ready schema (``pdb_path`` etc.) plus the ``coverage_fraction``
column for provenance.

Only the Tier 1 lower band edge may be widened (``--tier1-band-relax``) to admit
a near-bar experimental anchor; the Tier 2 band is never relaxed.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inverse_folding.evaluation.sampling import sample_uniform_bins  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fast", required=True,
                    help="fast_v2 parquet (source of coverage_fraction + canonical schema)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--band-lo", type=float, required=True)
    ap.add_argument("--band-hi", type=float, required=True)
    ap.add_argument("--n-total", type=int, default=50)
    ap.add_argument("--tier1-band-relax", type=float, default=0.0,
                    help="widen the Tier 1 LOWER band edge by this much to admit a near-bar "
                         "anchor (Tier 2 band is never relaxed)")
    ap.add_argument("--max-tier1", type=int, default=None,
                    help="cap the number of Tier 1 anchors; if more qualify, keep the "
                         "highest-coverage_fraction ones (i.e. drop from the lower band edge up)")
    ap.add_argument("--n-sub-bins", type=int, default=4)
    ap.add_argument("--value-col", default="coverage_fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fast = pd.read_parquet(args.fast)
    if args.value_col not in fast.columns:
        sys.exit(f"FATAL: '{args.value_col}' not in {args.fast}")
    tier = fast["tier"].astype(str)
    cf = fast[args.value_col]
    lo, hi = args.band_lo, args.band_hi

    # Tier 1 anchors: coverage_fraction in [lo - relax, hi]
    t1lo = lo - args.tier1_band_relax
    t1 = fast[(tier == "1") & cf.between(t1lo, hi)].copy()
    if args.max_tier1 is not None and len(t1) > args.max_tier1:
        # keep the highest-coverage anchors -> drops the ones nearest the lower band edge
        t1 = t1.sort_values(args.value_col, ascending=False).head(args.max_tier1)
    t1_ids = list(t1["protein_id"])

    # Tier 2: strict band, light n-sub-bin stratified fill to n_total
    n_t2 = args.n_total - len(t1_ids)
    if n_t2 < 0:
        sys.exit(f"FATAL: {len(t1_ids)} Tier 1 anchors exceed --n-total {args.n_total}")
    t2pool = fast[(tier == "2") & cf.between(lo, hi)].copy().set_index("protein_id")
    if len(t2pool) < n_t2:
        sys.exit(f"FATAL: only {len(t2pool)} Tier 2 in band [{lo},{hi}], need {n_t2}")
    sel_ids, hist = sample_uniform_bins(
        t2pool, args.value_col, n_target=n_t2, n_bins=args.n_sub_bins, seed=args.seed
    )
    t2_ids = list(sel_ids)

    keep = set(t1_ids) | set(t2_ids)
    out = (
        fast[fast["protein_id"].isin(keep)]
        .sort_values(["tier", args.value_col])
        .reset_index(drop=True)
    )
    out.to_parquet(args.output, index=False)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"[pilot] {args.output}")
    print(f"  band=[{lo},{hi}]  tier1_relax={args.tier1_band_relax}  "
          f"n_sub_bins={args.n_sub_bins}  seed={args.seed}")
    anchors = ", ".join(
        f"{r.protein_id}({getattr(r, args.value_col):.3f})"
        for r in t1.sort_values(args.value_col).itertuples()
    )
    print(f"  Tier1 anchors ({len(t1_ids)}): {anchors}")
    print(f"  Tier2 ({len(t2_ids)})  sub-bin n_selected: {list(hist['n_selected'])}  "
          f"(available: {list(hist['n_available'])})")
    cfsel = out[args.value_col]
    print(f"  total rows: {len(out)}   coverage_fraction "
          f"min={cfsel.min():.3f} median={cfsel.median():.3f} max={cfsel.max():.3f}")


if __name__ == "__main__":
    main()
