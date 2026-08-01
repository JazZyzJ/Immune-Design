#!/usr/bin/env python3
"""
Phase 5: turn the two scores into fixed-position masks over a (C0, sigma0) grid.

    mask(C0, sigma0) = { i : C_i_nogap >= C0 }  UNION  { i : sigma_pct_i >= sigma0 }

Union, as in the paper: a residue is pinned if it is either strongly conserved or
strongly coupled. That is what lets C0 rise (freeing the bulk of the protein for
redesign) while sigma0 independently holds the functionally critical residues
that conservation alone ranks mid-pack.

C0 is an absolute C_i_nogap cut, matching the paper's C0 = 0.25/0.35/0.65 scale.
sigma0 is a PERCENTILE, because the enrichment score has no natural 0-1 range --
transplanting a numeric sigma0 from TnpB would be meaningless.

Outputs, per grid cell, the fixed positions in three conventions from one source
of truth, because an off-by-one here silently pins the wrong side chains:
  * mature 1-indexed   -- matches the PDB and the uricase literature
  * uniprot 1-indexed  -- mature + 1
  * zero_indexed       -- mature - 1, for array/tensor indexing (ESM-IF1 style)

`always_fix.json` is written SEPARATELY and never merged into the grid masks, so
you can always tell what the scores found on their own from what was forced.

Usage
  python 16_masks.py --scores 03_scores/per_residue_scores.tsv --outdir 04_masks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
C0_GRID = [0.0, 0.25, 0.35, 0.50, 0.65]
SIGMA0_GRID = [None, 0.80, 0.90, 0.95]


def load_reference():
    from ruamel.yaml import YAML
    with open(PROJECT / "00_target" / "reference.yml") as fh:
        return YAML(typ="safe").load(fh)


def build_mask(df, c0, s0):
    """Boolean over rows of df (which is ordered by uniprot position)."""
    m = df.C_i_nogap >= c0
    if s0 is not None:
        m = m | (df.sigma_pct >= s0)
    return m.to_numpy()


def conventions(df, mask):
    mat = df.pos_mature.to_numpy()[mask]
    # position 0 is the initiator Met, which is not part of the mature protein
    mat = mat[mat >= 1]
    return {
        "mature_1indexed": sorted(int(x) for x in mat),
        "uniprot_1indexed": sorted(int(x) + 1 for x in mat),
        "zero_indexed": sorted(int(x) - 1 for x in mat),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True)
    p.add_argument("--outdir", default=str(PROJECT / "04_masks"))
    p.add_argument("--chain", default="A", help="chain id for the ProteinMPNN-style file")
    args = p.parse_args()

    ref = load_reference()
    site_same = set(ref["active_site_4p5A"]["same_chain"])
    site_cross = set(ref["active_site_4p5A"]["cross_chain"])
    site = site_same | site_cross

    df = pd.read_csv(args.scores, sep="\t").sort_values("pos_uniprot").reset_index(drop=True)
    n_mature = int((df.pos_mature >= 1).sum())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    masks = {}
    for c0 in C0_GRID:
        for s0 in SIGMA0_GRID:
            m = build_mask(df, c0, s0)
            conv = conventions(df, m)
            fixed = set(conv["mature_1indexed"])
            key = (c0, s0)
            masks[key] = fixed

            rec = {
                "C0": c0,
                "sigma0": s0,
                "n_fixed": len(fixed),
                "pct_fixed": round(100 * len(fixed) / n_mature, 1),
                "pct_redesignable": round(100 * (1 - len(fixed) / n_mature), 1),
                "active_site_captured": len(fixed & site),
                "active_site_total": len(site),
                "cross_subunit_captured": len(fixed & site_cross),
                "cross_subunit_total": len(site_cross),
                "active_site_missed": sorted(site - fixed),
            }
            rows.append(rec)

            tag = f"C{c0:.2f}".replace(".", "p") + "_" + ("Snone" if s0 is None else f"S{s0:.2f}".replace(".", "p"))
            payload = {
                "target": ref["target"]["uniprot"],
                "numbering_note": "mature = uniprot - 1; literature and PDB use mature",
                "C0": c0,
                "sigma0_percentile": s0,
                "rule": "fix if C_i_nogap >= C0 OR sigma_pct >= sigma0",
                "n_fixed": len(fixed),
                "n_mature_residues": n_mature,
                **rec,
                "fixed_positions": conv,
                "proteinmpnn": {args.chain: conv["mature_1indexed"]},
            }
            with open(outdir / f"mask_{tag}.json", "w") as fh:
                json.dump(payload, fh, indent=2)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "mask_grid_summary.tsv", sep="\t", index=False)

    # ---- invariants: these are cheap and catch index/logic slips immediately ----
    for c0 in C0_GRID:
        for a, b in zip(SIGMA0_GRID, SIGMA0_GRID[1:]):
            if a is None:
                continue
            assert masks[(c0, b)] <= masks[(c0, a)], f"sigma0 monotonicity broken at C0={c0}"
    for s0 in SIGMA0_GRID:
        for a, b in zip(C0_GRID, C0_GRID[1:]):
            assert masks[(b, s0)] <= masks[(a, s0)], f"C0 monotonicity broken at sigma0={s0}"
    for key, fixed in masks.items():
        assert all(1 <= x <= n_mature for x in fixed), f"position out of range in {key}"
    print(f"invariants OK ({len(masks)} masks: monotone in both thresholds, all in 1..{n_mature})")

    always = sorted(site)
    with open(outdir / "always_fix.json", "w") as fh:
        json.dump({
            "note": "NOT merged into the grid masks - kept separate so you can see "
                    "what the scores recovered on their own.",
            "source": "residues within 4.5 A of the OXC ligand in 1R4U assembly 1",
            "mature_1indexed": always,
            "same_chain": sorted(site_same),
            "cross_subunit": sorted(site_cross),
            "pts1_mature": ref["pts1"]["mature_positions"],
            "pts1_note": "peroxisomal targeting signal; relevant only if peroxisomal "
                         "import matters for your construct",
        }, fh, indent=2)

    print(f"\nwrote {len(masks)} masks + mask_grid_summary.tsv + always_fix.json to {outdir}\n")
    show = summary.copy()
    show["sigma0"] = show.sigma0.fillna("-")
    cols = ["C0", "sigma0", "n_fixed", "pct_fixed", "active_site_captured", "cross_subunit_captured", "active_site_missed"]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
