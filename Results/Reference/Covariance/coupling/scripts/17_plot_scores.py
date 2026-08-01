#!/usr/bin/env python3
"""
Phase 5b: per-residue view of the two masking scores.

Draws C_i (conservation) and sigma_i (cumulative coupling strength) along the
mature sequence, with the (C0, sigma0) thresholds 16_masks.py actually uses drawn
in, and the 15 structure-derived active-site residues marked so you can see by eye
which score pins each one.

  * C panel   -- absolute C_i_nogap, with the C0 grid (0.25/0.35/0.50/0.65) as
                 horizontal lines. A residue at or above a line is fixed by that C0.
  * sigma panel -- raw enrichment sigma_i, with the sigma0 PERCENTILE grid
                 (0.80/0.90/0.95) drawn at the raw value each percentile maps to,
                 so the lines mean the same thing the mask rule does.

Also re-emits the per-residue table as a plain CSV (the pipeline's canonical form
is the TSV; this is the same numbers, comma-separated, mature protein only).

Numbering: mature (= uniprot - 1); the initiator Met (mature 0) is dropped. See
00_target/reference.yml.

Usage
  python 17_plot_scores.py --scores 03_scores/per_residue_scores.tsv \
                           --out-prefix 03_scores/per_residue_scores
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
C0_GRID = [0.25, 0.35, 0.50, 0.65]        # matches 16_masks.py C0_GRID (non-zero cuts)
SIGMA0_GRID = [0.80, 0.90, 0.95]          # matches 16_masks.py SIGMA0_GRID (percentiles)


def load_reference():
    from ruamel.yaml import YAML
    with open(PROJECT / "00_target" / "reference.yml") as fh:
        return YAML(typ="safe").load(fh)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-prefix", default=str(PROJECT / "03_scores" / "per_residue_scores"))
    p.add_argument("--title", default="A. flavus uricase (Q00511) — masking scores, deep MSA")
    args = p.parse_args()

    ref = load_reference()
    site_same = set(ref["active_site_4p5A"]["same_chain"])
    site_cross = set(ref["active_site_4p5A"]["cross_chain"])

    df = pd.read_csv(args.scores, sep="\t")
    mat = df[df.pos_mature >= 1].sort_values("pos_mature").reset_index(drop=True)
    x = mat.pos_mature.to_numpy()

    # ---- CSV (mature protein, columns that matter for the mask) --------------
    csv_cols = ["pos_mature", "pos_uniprot", "wt_aa", "C_i_nogap", "C_i", "gap_frac",
                "sigma_raw", "sigma_pct", "C_pct", "C_rank", "sigma_rank",
                "active_site", "cross_subunit"]
    csv_path = Path(str(args.out_prefix) + ".csv")
    mat[csv_cols].to_csv(csv_path, index=False, float_format="%.5f")
    print(f"wrote {csv_path}  ({len(mat)} residues x {len(csv_cols)} cols)")

    # raw sigma values that the percentile thresholds map to (empirical quantiles
    # over the mature residues, so the drawn line == the mask boundary)
    sigma_at = {s0: float(np.quantile(mat.sigma_raw, s0)) for s0 in SIGMA0_GRID}

    same = mat[mat.pos_mature.isin(site_same)]
    cross = mat[mat.pos_mature.isin(site_cross)]

    # ---- figure --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # C_i panel
    ax1.fill_between(x, mat.C_i_nogap, color="#4477aa", alpha=0.30, linewidth=0)
    ax1.plot(x, mat.C_i_nogap, color="#4477aa", lw=0.7)
    for c0 in C0_GRID:
        ax1.axhline(c0, color="0.45", ls="--", lw=0.7)
        ax1.text(x.max() + 1, c0, f" C0={c0:g}", va="center", ha="left",
                 fontsize=7.5, color="0.30")
    ax1.scatter(same.pos_mature, same.C_i_nogap, s=34, color="#ee7733",
                zorder=5, label="active site (same-chain)", edgecolor="k", linewidth=0.3)
    ax1.scatter(cross.pos_mature, cross.C_i_nogap, s=42, color="#cc3311", marker="D",
                zorder=6, label="active site (cross-subunit)", edgecolor="k", linewidth=0.3)
    for _, r in pd.concat([same, cross]).iterrows():
        ax1.annotate(f"{r.wt_aa}{int(r.pos_mature)}", (r.pos_mature, r.C_i_nogap),
                     xytext=(0, 5), textcoords="offset points", ha="center",
                     fontsize=6.2, color="0.15")
    ax1.set_ylabel("$C_i$  (conservation, gap-excluded)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.set_title(args.title, fontsize=11)

    # sigma_i panel
    ax2.fill_between(x, mat.sigma_raw, color="#228833", alpha=0.30, linewidth=0)
    ax2.plot(x, mat.sigma_raw, color="#228833", lw=0.7)
    for s0, yv in sigma_at.items():
        ax2.axhline(yv, color="0.45", ls="--", lw=0.7)
        ax2.text(x.max() + 1, yv, f" σ0={s0:g} pct", va="center", ha="left",
                 fontsize=7.5, color="0.30")
    ax2.scatter(same.pos_mature, same.sigma_raw, s=34, color="#ee7733",
                zorder=5, edgecolor="k", linewidth=0.3)
    ax2.scatter(cross.pos_mature, cross.sigma_raw, s=42, color="#cc3311", marker="D",
                zorder=6, edgecolor="k", linewidth=0.3)
    for _, r in pd.concat([same, cross]).iterrows():
        ax2.annotate(f"{r.wt_aa}{int(r.pos_mature)}", (r.pos_mature, r.sigma_raw),
                     xytext=(0, 5), textcoords="offset points", ha="center",
                     fontsize=6.2, color="0.15")
    ax2.set_ylabel("$\\sigma_i$  (cumulative coupling)")
    ax2.set_xlabel("mature residue position")
    ax2.set_xlim(0, x.max() + 8)
    ax2.margins(y=0.08)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(str(args.out_prefix) + "." + ext)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
