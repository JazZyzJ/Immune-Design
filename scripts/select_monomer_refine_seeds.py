"""Monomer structure-rank refinement-seed selection (NO tetramer), per
PROTOCOL/monomer_structure_rank_refine_seeds.md. For wet-lab targets whose function does not
depend on an oligomeric interface, we rank designs by *monomer* structure quality alone and take
the top-N per (protein, allele) as refinement seeds — packaged separately because each
(protein, allele) is refined with a different per-allele immune Head.

Structure source + rank signal are protein-class specific (a fixed spec, re-calibratable via CLI):
  - enzyme  (PrASNase): ESMFold2 structural.parquet; rank active_site_sidechain_RMSD ↑, scTM ↓
                        (fold saturates, active-site geometry is the discriminator).
  - fp      (EGFP/mCherry): AF3 structural (ESMFold2 CANNOT fold FP beta-barrels — its scTM is an
                        artifact); rank scTM ↓, pLDDT ↓. Active-site metrics are a chromophore-Gly
                        artifact (active_site_complete=0, RMSD=NaN) and are EXCLUDED.
  - gated   (NanoLuc): ESMFold2; structure is weak, so gate scTM >= g AND max_anchor_sidechain_RMSD <= a
                        (keep only folds with an intact active site), then rank scTM ↓ — few survive.

Immune (n_strong_binders) is reported for context but NOT part of the rank. Emitted tables are
drop-in `--seed-table` inputs for scripts/refine_rf_designs.py. Cluster paths arrive via CLI.
"""
import argparse
import json
import pathlib
import subprocess

import pandas as pd

# Per-(protein, allele) spec. `struct` is relative to --backup-dir; `filt` selects the protein
# from a multi-protein parquet (None = whole file); `head_tag` records which Head refines it.
SPEC = [
    dict(protein="PrASNase", allele="0401", mode="enzyme",
         struct="runs/prasnase_04_01/structural.parquet", imm="runs/prasnase_04_01/imm_nmp.parquet",
         wt="runs/prasnase_04_01/wt_baseline/imm_nmp.parquet", filt=None,
         head_tag="a1res03_drb0401_cv5_fold0"),
    dict(protein="PrASNase", allele="0701", mode="enzyme",
         struct="runs/prasnase_07_01/structural.parquet", imm="runs/prasnase_07_01/imm_nmp.parquet",
         wt="runs/prasnase_07_01/wt_baseline/imm_nmp.parquet", filt=None,
         head_tag="a1res03_drb0701_cv5_fold0"),
    dict(protein="PrASNase", allele="1501", mode="enzyme",
         struct="runs/prasnase_15_01/structural.parquet", imm="runs/prasnase_15_01/imm_nmp.parquet",
         wt="runs/prasnase_15_01/wt_baseline/imm_nmp.parquet", filt=None,
         head_tag="a1res03_drb1501_single_split_best"),
    dict(protein="EGFP", allele="0401", mode="fp",
         struct="runs/4protein_b1aopen_0401/structural_fp_af3.parquet",
         imm="runs/4protein_b1aopen_0401/imm_nmp.parquet",
         wt="runs/4protein_b1aopen_0401/wt_baseline/imm_nmp.parquet", filt="EGFP",
         head_tag="a1res03_drb0401_cv5_fold0"),
    dict(protein="mCherry", allele="0401", mode="fp",
         struct="runs/4protein_b1aopen_0401/structural_fp_af3.parquet",
         imm="runs/4protein_b1aopen_0401/imm_nmp.parquet",
         wt="runs/4protein_b1aopen_0401/wt_baseline/imm_nmp.parquet", filt="mCherry",
         head_tag="a1res03_drb0401_cv5_fold0"),
    dict(protein="NanoLuc", allele="0401", mode="gated",
         struct="runs/4protein_b1aopen_0401/structural_esmfold2.parquet",
         imm="runs/4protein_b1aopen_0401/imm_nmp.parquet",
         wt="runs/4protein_b1aopen_0401/wt_baseline/imm_nmp.parquet", filt="NanoLuc",
         head_tag="a1res03_drb0401_cv5_fold0"),
]

DEFAULT_N = {"PrASNase": 15, "EGFP": 15, "mCherry": 15, "NanoLuc": 8}

OUT_COLS = ["protein_id", "allele", "head_tag", "design_id", "design_idx", "rank",
            "scTM", "pLDDT", "active_site_sidechain_RMSD", "max_anchor_sidechain_RMSD",
            "predicted_active_site_min_pLDDT", "n_strong_binders", "wt_n_strong", "sequence"]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=pathlib.Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return "unknown"


def select_one(spec, base, n, nl_sctm, nl_anchor):
    st = pd.read_parquet(base / spec["struct"])
    if spec["filt"]:
        st = st[st.protein_id == spec["filt"]].copy()
    imm = pd.read_parquet(base / spec["imm"])
    if spec["filt"] and "protein_id" in imm:
        imm = imm[imm.protein_id == spec["filt"]]
    st = st.merge(imm[["design_idx", "n_strong_binders"]], on="design_idx", how="left")
    wt = pd.read_parquet(base / spec["wt"])
    if spec["filt"] and "protein_id" in wt:
        wt = wt[wt.protein_id == spec["filt"]]
    st["wt_n_strong"] = pd.to_numeric(wt.n_strong_binders, errors="coerce").median()

    mode = spec["mode"]
    if mode == "enzyme":
        ranked = st.sort_values(["active_site_sidechain_RMSD", "scTM"], ascending=[True, False])
        gated_n = len(ranked)
    elif mode == "fp":
        ranked = st.sort_values(["scTM", "pLDDT"], ascending=[False, False])
        gated_n = len(ranked)
    elif mode == "gated":
        keep = st[(st.scTM >= nl_sctm) & (st.max_anchor_sidechain_RMSD <= nl_anchor)]
        ranked = keep.sort_values(["scTM", "max_anchor_sidechain_RMSD"], ascending=[False, True])
        gated_n = len(ranked)
    else:
        raise SystemExit(f"unknown mode {mode}")

    sel = ranked.head(n).copy()
    sel["allele"] = spec["allele"]
    sel["head_tag"] = spec["head_tag"]
    sel["rank"] = range(1, len(sel) + 1)
    return sel[OUT_COLS], gated_n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backup-dir", required=True, type=pathlib.Path,
                   help="backup collection root (holds runs/, collection/)")
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="default: <backup-dir>/refine_seed_selection")
    p.add_argument("--n-prasnase", type=int, default=DEFAULT_N["PrASNase"], help="N per PrASNase allele")
    p.add_argument("--n-egfp", type=int, default=DEFAULT_N["EGFP"])
    p.add_argument("--n-mcherry", type=int, default=DEFAULT_N["mCherry"])
    p.add_argument("--n-nanoluc", type=int, default=DEFAULT_N["NanoLuc"])
    p.add_argument("--nanoluc-sctm-min", type=float, default=0.95)
    p.add_argument("--nanoluc-max-anchor", type=float, default=2.6)
    a = p.parse_args()

    base = a.backup_dir.resolve()
    out_dir = a.out_dir or (base / "refine_seed_selection")
    n_by_protein = {"PrASNase": a.n_prasnase, "EGFP": a.n_egfp, "mCherry": a.n_mcherry, "NanoLuc": a.n_nanoluc}

    print("=== select_monomer_refine_seeds config ===")
    print(f"  backup_dir = {base}")
    print(f"  N: {n_by_protein} | NanoLuc gate scTM>={a.nanoluc_sctm_min} max_anchor<={a.nanoluc_max_anchor}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"protocol": "PROTOCOL/monomer_structure_rank_refine_seeds.md",
                "script": "scripts/select_monomer_refine_seeds.py", "git_sha": _git_sha(),
                "backup_dir": str(base), "lists": []}
    for spec in SPEC:
        n = n_by_protein[spec["protein"]]
        sel, gated_n = select_one(spec, base, n, a.nanoluc_sctm_min, a.nanoluc_max_anchor)
        tag = f"{spec['protein']}_{spec['allele']}"
        sel.to_parquet(out_dir / f"refine_seeds_{tag}.parquet", index=False)
        sel.to_csv(out_dir / f"refine_seeds_{tag}.csv", index=False)
        note = f" (only {gated_n} passed gate)" if spec["mode"] == "gated" and gated_n < n else ""
        print(f"  [{tag:16s}] mode={spec['mode']:7s} head={spec['head_tag']:32s} N={len(sel):2d}{note} | "
              f"scTM [{sel.scTM.min():.3f},{sel.scTM.max():.3f}] "
              f"n_strong med={sel.n_strong_binders.median():.0f} (WT {sel.wt_n_strong.iloc[0]:.0f})")
        manifest["lists"].append({"tag": tag, "protein": spec["protein"], "allele": spec["allele"],
                                  "mode": spec["mode"], "head_tag": spec["head_tag"],
                                  "struct_source": spec["struct"], "n_selected": int(len(sel)),
                                  "n_gate_passed": int(gated_n) if spec["mode"] == "gated" else None,
                                  "file": f"refine_seeds_{tag}.parquet"})
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote -> {out_dir}/  (6 per-(protein,allele) seed tables + selection_manifest.json)")


if __name__ == "__main__":
    main()
