#!/usr/bin/env python
"""Build a NoD-difficulty-UNIFORM pilot subset with a decoupled 4/4 design split.

pilot-v3. Unlike pilot_v2 (coverage_fraction band stratification, generator-independent
but difficulty-blind), this stratifies UNIFORMLY across de-immunization difficulty measured
on the unguided NoD baseline, so a quick iteration set spans the easy->hard spectrum.

Difficulty = median NMP strong_frac over ``--half-a-designs`` (default: ALL 8 NoD designs ->
cleanest per-protein estimate). Difficulty axis = **NMP strong_frac only** (the non-circular
validator); the epitope head is the RF guidance, recorded as a diagnostic but never defining
the axis.

Optional regression-decoupling: the highrisk bias (selecting on the same NoD realization later
used as the eval baseline) is severe only for **extreme-tail** selection + a quantitative
small-effect claim. For a UNIFORM iteration set it is a second-order effect, so binning on all
8 designs is the default. If you later need a rigorous quantitative per-bin method-vs-NoD number,
either reserve a disjoint ``--half-b-designs`` up front (its burden is recorded as
``pilot_baseline_nmp_halfB`` and did NOT drive selection) or regenerate a fresh NoD baseline for
the ~50 selected proteins at eval time.

Selection: place candidates in equal-frequency quantile bins over half-A difficulty and sample
uniformly per bin. Output = a subset of the canonical main IF-ready parquet + difficulty/split
columns (+ a sibling FASTA). Registered in doc/SCRIPTS.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_protein_strong_frac(nmp: pd.DataFrame, design_idxs: list[int]) -> pd.Series:
    """Median NMP strong_frac (= n_strong_binders / n_windows_scored) over ``design_idxs``."""
    sub = nmp[nmp["design_idx"].isin(design_idxs)].copy()
    sfrac = sub["n_strong_binders"] / sub["n_windows_scored"].clip(lower=1)
    return sfrac.groupby(sub["protein_id"]).median()


def per_protein_head(head: pd.DataFrame, design_idxs: list[int]) -> pd.Series:
    """Median epitope-head global_risk over ``design_idxs`` (diagnostic only)."""
    sub = head[head["design_idx"].isin(design_idxs)]
    return sub.groupby("protein_id")["global_risk"].median()


def uniform_bin_sample(
    difficulty: pd.Series, n_total: int, n_bins: int, seed: int
) -> pd.DataFrame:
    """Equal-frequency quantile bins over ``difficulty``; sample uniformly per bin.

    Ties are broken by a stable first-rank so the bins are exactly equal-frequency; the
    remainder (``n_total % n_bins``) is spread across the highest-difficulty bins. Returns a
    DataFrame indexed by protein_id with ``difficulty``, ``difficulty_pct``, ``bin``.
    """
    if n_total > len(difficulty):
        raise ValueError(f"n_total={n_total} exceeds candidate pool {len(difficulty)}")
    d = difficulty.sort_values(kind="mergesort")  # stable; ascending difficulty
    ranks = d.rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False)
    pct = d.rank(pct=True)
    rng = np.random.default_rng(seed)
    per = [n_total // n_bins] * n_bins
    for i in range(n_total % n_bins):  # remainder -> hardest bins (highest index)
        per[n_bins - 1 - i] += 1
    rows = []
    for b in range(n_bins):
        ids = list(bins.index[bins == b])
        k = min(per[b], len(ids))
        chosen = list(rng.choice(ids, size=k, replace=False)) if k < len(ids) else ids
        for pid in chosen:
            rows.append({"protein_id": pid, "difficulty": float(d[pid]),
                         "difficulty_pct": float(pct[pid]), "bin": int(b)})
    return pd.DataFrame(rows).set_index("protein_id")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--canonical-if-ready", required=True, help="main IF-ready parquet (candidate pool + schema)")
    ap.add_argument("--nod-imm-dir", required=True, help="merged NoD imm dir (imm_nmp.parquet + imm_head.parquet, design_idx 0-7)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fasta-output", default=None, help="optional sibling FASTA of selected sequences")
    ap.add_argument("--n-total", type=int, default=50)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--half-a-designs", default="0,1,2,3,4,5,6,7",
                    help="design_idx defining difficulty (default: all 8 -> cleanest bins)")
    ap.add_argument("--half-b-designs", default="",
                    help="optional disjoint design_idx reserved as an INDEPENDENT eval baseline "
                         "(empty = none; only needed for a rigorous quantitative per-bin method-vs-NoD claim)")
    ap.add_argument("--min-coverage", type=float, default=0.8, help="drop if_sequence_coverage < this (match canonical)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    halfA = [int(x) for x in args.half_a_designs.split(",") if x.strip()]
    halfB = [int(x) for x in args.half_b_designs.split(",") if x.strip()]
    if not halfA:
        sys.exit("FATAL: --half-a-designs is empty (need >=1 design to define difficulty)")
    if set(halfA) & set(halfB):
        sys.exit(f"FATAL: half-a and half-b overlap: {set(halfA) & set(halfB)} (must be disjoint)")
    # halfB may be empty: difficulty then uses ALL designs in half-a and NO independent baseline
    # is reserved (fine for iteration; the regression coupling only bites a quantitative per-bin
    # method-vs-NoD claim, which should regenerate a fresh NoD baseline anyway).

    canon = pd.read_parquet(args.canonical_if_ready)
    canon["protein_id"] = canon["protein_id"].astype(str)
    n0 = len(canon)
    if args.min_coverage > 0:
        canon = canon[canon["if_sequence_coverage"] >= args.min_coverage]
    pool = set(canon["protein_id"])
    print(f"  candidate pool: {n0} -> {len(pool)} (if_sequence_coverage >= {args.min_coverage})")

    imm_dir = Path(args.nod_imm_dir)
    nmp = pd.read_parquet(imm_dir / "imm_nmp.parquet"); nmp["protein_id"] = nmp["protein_id"].astype(str)
    head = pd.read_parquet(imm_dir / "imm_head.parquet"); head["protein_id"] = head["protein_id"].astype(str)
    nmp = nmp[nmp["protein_id"].isin(pool)]; head = head[head["protein_id"].isin(pool)]

    diff = per_protein_strong_frac(nmp, halfA).rename("difficulty").dropna()          # difficulty designs
    headA = per_protein_head(head, halfA).rename("pilot_head_halfA")                   # diagnostic
    diff = diff[diff.index.isin(pool)]
    print(f"  scored proteins (difficulty NMP over designs {halfA}): {len(diff)}; "
          f"strong_frac range [{diff.min():.3f}, {diff.max():.3f}] median {diff.median():.3f}")

    sel = uniform_bin_sample(diff, n_total=args.n_total, n_bins=args.n_bins, seed=args.seed)
    sel = sel.join(headA)
    if halfB:  # optional independent eval-baseline burden (only when a disjoint half is reserved)
        baseB = per_protein_strong_frac(nmp, halfB).rename("pilot_baseline_nmp_halfB")
        sel = sel.join(baseB)
    sel["pilot_difficulty_designs"] = ",".join(map(str, halfA))
    sel["pilot_baseline_designs"] = ",".join(map(str, halfB)) if halfB else ""
    sel = sel.rename(columns={"difficulty": "pilot_difficulty_nmp", "difficulty_pct": "pilot_difficulty_pct", "bin": "pilot_difficulty_bin"})

    out = (canon.merge(sel.reset_index(), on="protein_id", how="inner")
           .sort_values(["pilot_difficulty_bin", "pilot_difficulty_pct"]))
    out.to_parquet(args.output, index=False)

    print(f"[pilot-v3] {args.output}  n={len(out)}  bins={args.n_bins}")
    cnt = out["pilot_difficulty_bin"].value_counts().sort_index()
    print(f"  per-bin count: {cnt.to_dict()}")
    print(f"  selected difficulty (NMP) range [{out['pilot_difficulty_nmp'].min():.3f}, {out['pilot_difficulty_nmp'].max():.3f}]  median {out['pilot_difficulty_nmp'].median():.3f}")
    if halfB:
        print(f"  reserved half-B baseline burden (selected): median {out['pilot_baseline_nmp_halfB'].median():.3f}")

    if args.fasta_output:
        with open(args.fasta_output, "w") as f:
            for r in out.itertuples():
                f.write(f">{r.protein_id}\n{r.sequence}\n")
        print(f"  fasta -> {args.fasta_output}")


if __name__ == "__main__":
    main()
