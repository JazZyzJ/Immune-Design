"""Select List 1 (monomer tetramer-feeder shortlist) and List 2 (refinement seeds) from a
constrained single-protein RF redesign run, per PROTOCOL/shortlist_and_refine_seed_selection.md.

Reads the run's own eval artifacts (no bespoke merged CSV), derives the active-site reliability
signals from the residue-level parquet + the run's constraint manifest, and writes two
self-contained tables under <run-dir>/selection/:

  list1_tetramer_shortlist.{parquet,csv} -- active-site-reliable monomers, ranked best-first,
      tier-labelled; immune-agnostic; hand the whole thing to AF3 tetramer (uncapped).
  list2_refine_seeds.{parquet,csv}        -- T1-pristine structure-best seeds for deep
      refinement; schema is a drop-in `--seed-table` for scripts/refine_rf_designs.py
      (protein_id, design_id, sequence + metrics).

Logic is fixed by the protocol; numeric floors are CLI args (re-calibrate per run). All cluster
paths arrive via CLI; nothing is hardcoded.
"""
import argparse
import json
import pathlib
import subprocess

import pandas as pd
import yaml


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=pathlib.Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return "unknown"


def load_catalytic_positions(manifest_path):
    """1-based UniProt positions of the direct-functional catalytic residues (the activity
    firewall). Falls back to the full hard-anchor set only if the union key is absent."""
    with open(manifest_path) as fh:
        man = yaml.safe_load(fh)
    prov = man.get("annotation_provenance", {})
    cat = prov.get("direct_functional_union_uniprot_1b")
    if not cat:
        raise SystemExit(f"[fatal] {manifest_path} has no annotation_provenance."
                         f"direct_functional_union_uniprot_1b; cannot define catalytic firewall.")
    return set(int(x) for x in cat)


def per_design_anchor_maxima(residues, catalytic_1b):
    """From the residue-level parquet, worst sidechain RMSD over catalytic vs shell anchors,
    per design. Glycine/no-sidechain anchors carry NaN sidechain_RMSD and are skipped by max()."""
    anc = residues[residues.is_anchor].copy()
    anc["scr"] = pd.to_numeric(anc.sidechain_RMSD, errors="coerce")
    is_cat = anc.residue_idx_1based.isin(catalytic_1b)
    cat_max = anc[is_cat].groupby("design_idx").scr.max().rename("cat_max_scRMSD")
    shell_max = anc[~is_cat].groupby("design_idx").scr.max().rename("shell_max_scRMSD")
    return pd.concat([cat_max, shell_max], axis=1).reset_index()


def build_frame(run_dir, manifest_path):
    st = pd.read_parquet(run_dir / "eval_structure" / "structural.parquet")
    res = pd.read_parquet(run_dir / "eval_structure" / "structural_residues.parquet")
    nmp = pd.read_parquet(run_dir / "eval_immune" / "imm_nmp.parquet")[
        ["design_idx", "n_strong_binders", "n_weak_binders", "mean_best_rank"]]
    catalytic_1b = load_catalytic_positions(manifest_path)
    maxima = per_design_anchor_maxima(res, catalytic_1b)
    df = st.merge(maxima, on="design_idx", how="left").merge(nmp, on="design_idx", how="left")
    return df, sorted(catalytic_1b)


LIST1_RANK = ["cat_max_scRMSD", "predicted_active_site_min_pLDDT",
              "active_site_sidechain_RMSD", "global_ca_RMSD"]
LIST1_ASC = [True, False, True, True]

OUT_COLS = ["protein_id", "design_id", "design_idx", "tier", "rank",
            "scTM", "pLDDT", "global_ca_RMSD", "predicted_active_site_min_pLDDT",
            "active_site_sidechain_RMSD", "cat_max_scRMSD", "shell_max_scRMSD",
            "recovery", "n_strong_binders", "n_weak_binders", "sequence"]


def select(df, a):
    # T1 pristine (structure-best; also List 2 seed pool)
    t1 = ((df.scTM >= a.t1_sctm_min) & (df.pLDDT >= a.t1_plddt_min)
          & (df.global_ca_RMSD <= a.t1_ca_max) & (df.predicted_active_site_min_pLDDT >= a.t1_minplddt_min)
          & (df.active_site_sidechain_RMSD <= a.t1_agg_max)
          & (df.cat_max_scRMSD <= a.t1_cat_max) & (df.shell_max_scRMSD <= a.t1_shell_max))
    # List 1 floor: active-site reliability (global fold saturates; not the cut)
    floor = ((df.cat_max_scRMSD <= a.cat_max) & (df.predicted_active_site_min_pLDDT >= a.min_plddt)
             & (df.scTM >= a.sctm_min))
    df = df.assign(tier=pd.Series(["T1" if v else "floor" for v in t1], index=df.index))

    list1 = df[floor].sort_values(LIST1_RANK, ascending=LIST1_ASC).reset_index(drop=True)
    list1["rank"] = list1.index + 1
    # T1 rows keep tier=T1; the rest of the floor set are the extended net
    list1["tier"] = ["T1" if t else "T2plus" for t in list1.tier.eq("T1")]

    # List 2 seeds: T1 pristine, structure-best; tie-break lower starting immune (fewer edits)
    list2 = df[t1].sort_values(["cat_max_scRMSD", "n_strong_binders",
                                "predicted_active_site_min_pLDDT"],
                               ascending=[True, True, False]).reset_index(drop=True)
    list2["rank"] = list2.index + 1
    list2["tier"] = "T1"
    return list1, list2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=pathlib.Path)
    p.add_argument("--constraint-manifest", type=pathlib.Path, default=None,
                   help="default: <run-dir>/meta/constraint_manifest.yaml")
    p.add_argument("--out-dir", type=pathlib.Path, default=None,
                   help="default: <run-dir>/selection")
    # List 1 floor (active-site reliability)
    p.add_argument("--cat-max", type=float, default=2.0, help="max catalytic worst-sidechain RMSD (A)")
    p.add_argument("--min-plddt", type=float, default=85.0, help="min predicted_active_site_min_pLDDT")
    p.add_argument("--sctm-min", type=float, default=0.94, help="min scTM sanity floor")
    # T1 pristine (List 2 seeds + List 1 top tier)
    p.add_argument("--t1-sctm-min", type=float, default=0.95)
    p.add_argument("--t1-plddt-min", type=float, default=92.0)
    p.add_argument("--t1-ca-max", type=float, default=2.0)
    p.add_argument("--t1-minplddt-min", type=float, default=85.0)
    p.add_argument("--t1-agg-max", type=float, default=1.1)
    p.add_argument("--t1-cat-max", type=float, default=1.5)
    p.add_argument("--t1-shell-max", type=float, default=2.5)
    a = p.parse_args()

    run_dir = a.run_dir.resolve()
    manifest = a.constraint_manifest or (run_dir / "meta" / "constraint_manifest.yaml")
    out_dir = a.out_dir or (run_dir / "selection")
    for pth in [run_dir / "eval_structure" / "structural.parquet",
                run_dir / "eval_structure" / "structural_residues.parquet",
                run_dir / "eval_immune" / "imm_nmp.parquet", manifest]:
        if not pathlib.Path(pth).exists():
            raise SystemExit(f"[fatal] missing required input: {pth}")

    df, catalytic = build_frame(run_dir, manifest)
    list1, list2 = select(df, a)

    print("=== select_redesign_shortlists config ===")
    print(f"  run_dir            = {run_dir}")
    print(f"  constraint_manifest= {manifest}")
    print(f"  catalytic (1-based)= {catalytic}")
    print(f"  n_designs          = {len(df)}  protein(s)={sorted(df.protein_id.unique())}")
    print(f"  List1 floor        : cat_max<={a.cat_max}  min_pLDDT>={a.min_plddt}  scTM>={a.sctm_min}")
    print(f"  List1 rank         : {list(zip(LIST1_RANK, ['asc' if x else 'desc' for x in LIST1_ASC]))}")
    print(f"  T1 pristine        : scTM>={a.t1_sctm_min} pLDDT>={a.t1_plddt_min} ca<={a.t1_ca_max} "
          f"minpLDDT>={a.t1_minplddt_min} agg<={a.t1_agg_max} cat<={a.t1_cat_max} shell<={a.t1_shell_max}")
    print(f"  -> List1 (tetramer feeder) N = {len(list1)}  (T1={int(list1.tier.eq('T1').sum())}, "
          f"T2plus={int(list1.tier.eq('T2plus').sum())})")
    if len(list1):
        print(f"     List1 n_strong: med={list1.n_strong_binders.median():.0f} "
              f"[{list1.n_strong_binders.min()},{list1.n_strong_binders.max()}]  "
              f"cat_max med={list1.cat_max_scRMSD.median():.2f}")
    print(f"  -> List2 (refine seeds)   N = {len(list2)}  "
          f"n_strong [{list2.n_strong_binders.min() if len(list2) else -1},"
          f"{list2.n_strong_binders.max() if len(list2) else -1}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    l1 = list1[OUT_COLS]
    l2 = list2[OUT_COLS]
    l1.to_parquet(out_dir / "list1_tetramer_shortlist.parquet", index=False)
    l1.to_csv(out_dir / "list1_tetramer_shortlist.csv", index=False)
    l2.to_parquet(out_dir / "list2_refine_seeds.parquet", index=False)
    l2.to_csv(out_dir / "list2_refine_seeds.csv", index=False)
    manifest_out = {
        "protocol": "PROTOCOL/shortlist_and_refine_seed_selection.md",
        "script": "scripts/select_redesign_shortlists.py",
        "git_sha": _git_sha(),
        "run_dir": str(run_dir),
        "constraint_manifest": str(manifest),
        "catalytic_1based": catalytic,
        "n_designs": int(len(df)),
        "list1": {"file": "list1_tetramer_shortlist.parquet", "n": int(len(l1)),
                  "floor": {"cat_max": a.cat_max, "min_plddt": a.min_plddt, "sctm_min": a.sctm_min},
                  "rank": list(zip(LIST1_RANK, ["asc" if x else "desc" for x in LIST1_ASC])),
                  "n_T1": int(l1.tier.eq("T1").sum())},
        "list2": {"file": "list2_refine_seeds.parquet", "n": int(len(l2)),
                  "seed_tier": "T1", "tie_break": ["cat_max_scRMSD", "n_strong_binders", "min_pLDDT"]},
        "t1_thresholds": {"sctm_min": a.t1_sctm_min, "plddt_min": a.t1_plddt_min, "ca_max": a.t1_ca_max,
                          "minplddt_min": a.t1_minplddt_min, "agg_max": a.t1_agg_max,
                          "cat_max": a.t1_cat_max, "shell_max": a.t1_shell_max},
    }
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"  wrote -> {out_dir}/  (list1/list2 .parquet+.csv + selection_manifest.json)")


if __name__ == "__main__":
    main()
