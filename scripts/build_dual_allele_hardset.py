#!/usr/bin/env python
"""Build the two-allele high-risk set (DRB1*07:01 x DRB1*04:01).

Why this is a separate builder and not a mode of ``build_highrisk_demo.py``:
that script ranks proteins inside ONE allele's cohort. Here the two cohorts are
different populations that were sampled independently from the PDB, so the join
key is the IF-ready SEQUENCE -- ``protein_id`` is not comparable across cohorts
(0701/0401 use ``<PDB>_<chain>``, the v3 1501 build uses
``t2-<hash>-<PDB>_<entity>-<chain>``).

Measured facts this design is built around (2026-09-01/02, deduped cohorts):

* the cohorts share only **358** sequences of ~2430 each;
* cross-allele burden agreement is weak -- Spearman **+0.358** (NMP) and
  **+0.220** (head) -- so "hard for both alleles" is barely more than chance and
  a strict dual hard set the size of a single-allele one does not exist;
* 0701 x 0401 is nonetheless the only workable pair: 0701 x 1501 and
  0401 x 1501 share 195/171 sequences at NMP rho +0.216/+0.114, and 1501 has the
  weakest head (CV Pearson 0.354 vs 0.508/0.429).

So this emits the WHOLE shared pool ranked by dual burden rather than a
truncated "top-N", and the caller slices the depth a figure needs. Ranking uses
``dual_nmp_min_pct`` = min over the four (allele, arm) NMP percentiles: a protein
ranks high only if BOTH samplers left it hard under BOTH alleles.

Provenance caveat, recorded in the manifest: a shared protein carries two
INDEPENDENT baseline design draws -- the one made in the 0701 cohort (scored
under 0701) and the one made in the 0401 cohort (scored under 0401). Dual
hardness here is therefore a protein-level property measured from two unbiased
draws, not one design set scored under two alleles. Cross-scoring a single draw
under both alleles is ~3900 designs (a few CPU shards) if a figure needs it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def arm_axes(imm_dir: Path) -> pd.DataFrame:
    """Per-protein median NMP strong_frac and head global_risk over the 8 designs."""
    nmp = pd.read_parquet(imm_dir / "imm_nmp.parquet")
    nmp["protein_id"] = nmp["protein_id"].astype(str)
    frac = nmp["n_strong_binders"] / nmp["n_windows_scored"].clip(lower=1)
    n = frac.groupby(nmp["protein_id"]).median().rename("nmp")
    head = pd.read_parquet(imm_dir / "imm_head.parquet")
    head["protein_id"] = head["protein_id"].astype(str)
    h = head.groupby("protein_id")["global_risk"].median().rename("head")
    return pd.concat([n, h], axis=1)


def side(cohort_path: str, gumbel_dir: str, pmpnn_dir: str, label: str) -> pd.DataFrame:
    coh = pd.read_parquet(cohort_path)
    coh["protein_id"] = coh["protein_id"].astype(str)
    g = arm_axes(Path(gumbel_dir)).add_prefix(f"{label}_gum_")
    p = arm_axes(Path(pmpnn_dir)).add_prefix(f"{label}_pm_")
    out = coh.set_index("protein_id").join(g, how="inner").join(p, how="inner").dropna(
        subset=[f"{label}_gum_nmp", f"{label}_pm_nmp", f"{label}_gum_head", f"{label}_pm_head"])
    return out.reset_index().rename(columns={"protein_id": f"{label}_protein_id"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for lab in ("a", "b"):
        ap.add_argument(f"--{lab}-label", required=True, help="allele tag, e.g. HLA-DRB1_07_01")
        ap.add_argument(f"--{lab}-cohort", required=True)
        ap.add_argument(f"--{lab}-gumbel-imm-dir", required=True)
        ap.add_argument(f"--{lab}-pmpnn-imm-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--head-floor-pct", type=float, default=0.20,
                    help="drop proteins whose min-over-(allele,arm) head percentile is below "
                         "this; same figure-hygiene rule as the single-allele builder")
    args = ap.parse_args()

    A = side(args.a_cohort, args.a_gumbel_imm_dir, args.a_pmpnn_imm_dir, "a")
    B = side(args.b_cohort, args.b_gumbel_imm_dir, args.b_pmpnn_imm_dir, "b")
    print(f"[dual] {args.a_label}: {len(A)} scored | {args.b_label}: {len(B)} scored")

    keep_a = [c for c in A.columns if c.startswith("a_")] + ["sequence", "sequence_length"]
    extra = [c for c in ("if_sequence_coverage", "pdb_path", "tier", "resolution") if c in A.columns]
    m = A[keep_a + extra].merge(
        B[[c for c in B.columns if c.startswith("b_")] + ["sequence"]], on="sequence", how="inner")
    print(f"[dual] shared sequences: {len(m)}")
    if m.sequence.duplicated().any():
        raise SystemExit("FATAL: a sequence appears twice after the join; cohorts are not deduped")

    if args.min_coverage > 0 and "if_sequence_coverage" in m:
        n0 = len(m)
        m = m[m.if_sequence_coverage >= args.min_coverage]
        print(f"[dual] coverage >= {args.min_coverage}: {n0} -> {len(m)}")

    nmp_cols = ["a_gum_nmp", "a_pm_nmp", "b_gum_nmp", "b_pm_nmp"]
    head_cols = ["a_gum_head", "a_pm_head", "b_gum_head", "b_pm_head"]
    for c in nmp_cols + head_cols:
        m[c + "_pct"] = m[c].rank(pct=True)
    m["dual_nmp_min_pct"] = m[[c + "_pct" for c in nmp_cols]].min(axis=1)
    # The head FLOOR must be min over the two ALLELES, not over all four columns.
    # Within an allele the two arms are two draws of the same quantity, so they are
    # averaged; across alleles we require headroom under both. Taking min over all
    # four is far stricter than the single-allele builder's min-over-two --
    # measured, it excluded 52% of the shared pool instead of the intended few
    # percent, because P(min of 4 uniforms < 0.2) = 59% vs 36% for two.
    m["a_head_pct"] = m[["a_gum_head", "a_pm_head"]].mean(axis=1).rank(pct=True)
    m["b_head_pct"] = m[["b_gum_head", "b_pm_head"]].mean(axis=1).rank(pct=True)
    m["dual_head_min_pct"] = m[["a_head_pct", "b_head_pct"]].min(axis=1)

    excluded = None
    if args.head_floor_pct > 0:
        below = m.dual_head_min_pct < args.head_floor_pct
        excluded = m[below]
        m = m[~below]
        print(f"[dual] head floor {args.head_floor_pct}: {len(excluded)} excluded, pool {len(m)}")

    m = m.sort_values("dual_nmp_min_pct", ascending=False).reset_index(drop=True)
    m["dual_hardrank"] = range(1, len(m) + 1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    if excluded is not None and len(excluded):
        excluded.to_parquet(Path(str(out).replace(".parquet", "") + ".head_floor_excluded.parquet"),
                            index=False)

    def q(frame, col):
        return {"median": float(frame[col].median()), "min": float(frame[col].min()),
                "max": float(frame[col].max())}
    manifest = {
        "alleles": {"a": args.a_label, "b": args.b_label},
        "join_key": "if-ready sequence (protein_id is not comparable across cohorts)",
        "rank_law": "dual_nmp_min_pct = min over the four (allele, arm) NMP percentiles",
        "head_floor_pct": args.head_floor_pct,
        "min_coverage": args.min_coverage,
        "n_pool": int(len(m)),
        "n_head_floor_excluded": int(len(excluded)) if excluded is not None else 0,
        "top100": {c: q(m.head(100), c) for c in nmp_cols + head_cols},
        "pool": {c: q(m, c) for c in nmp_cols + head_cols},
        "design_draw_caveat": (
            "each shared protein carries two INDEPENDENT baseline design draws, one made in "
            "each allele's cohort and scored under that allele; this is not one design set "
            "scored under both alleles"),
        "sources": {"a_cohort": str(Path(args.a_cohort).resolve()),
                    "b_cohort": str(Path(args.b_cohort).resolve()),
                    "a_gumbel": str(Path(args.a_gumbel_imm_dir).resolve()),
                    "a_pmpnn": str(Path(args.a_pmpnn_imm_dir).resolve()),
                    "b_gumbel": str(Path(args.b_gumbel_imm_dir).resolve()),
                    "b_pmpnn": str(Path(args.b_pmpnn_imm_dir).resolve())},
    }
    Path(str(out).replace(".parquet", "") + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[dual] wrote {len(m)} ranked proteins -> {out}")
    for n in (50, 100, 150):
        if len(m) >= n:
            t = m.head(n)
            print(f"  top-{n:>3}: dual_nmp_min_pct >= {t.dual_nmp_min_pct.min():.3f} | "
                  f"NMP med A {t.a_gum_nmp.median():.4f}/{t.a_pm_nmp.median():.4f} "
                  f"B {t.b_gum_nmp.median():.4f}/{t.b_pm_nmp.median():.4f} | "
                  f"head med A {t[['a_gum_head','a_pm_head']].mean(axis=1).median():+.2f} "
                  f"B {t[['b_gum_head','b_pm_head']].mean(axis=1).median():+.2f}")


if __name__ == "__main__":
    main()
