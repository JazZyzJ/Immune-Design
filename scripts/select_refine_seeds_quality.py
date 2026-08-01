"""Quality-based refinement-seed selection (delivery mode; supersedes the 0701 difficulty
tiers). Per protein, k=5 seeds:
  primary  = structurally-valid (global_ok_active_site_ok + anchors preserved), ranked by
             low residual (nmp_n_strong) then best active-site geometry then scTM;
  fallback = if <k valid, fill from the rest by structure priority (active-site RMSD, scTM).
Emits a self-contained seed table (protein_id, design_id, sequence + metrics) per allele.
"""
import pathlib

import pandas as pd

SRC = ("Results/Analysis/uricases/uricase23_a1res03_b1open_n64_seed10__20260702T075209Z/"
       "selection_metrics/full/candidate_selection_full_metrics_with_wt_immune_flags.csv")
ALLELES = {"HLA-DRB1_04_01": "0401", "HLA-DRB1_15_01": "1501"}
K = 5
COLS = ["protein_id", "allele", "design_idx", "design_id", "selection_source", "quality_rank",
        "nmp_n_strong_binders", "nmp_n_weak_binders", "nmp_mean_best_rank", "wt_n_strong_binders",
        "delta_strong_vs_wt", "strong_reduction_frac", "head_global_risk", "struct_scTM",
        "struct_scRMSD", "active_site_ca_rmsd_pm3", "active_site_spatial_shell6A_ca_rmsd",
        "foldability_selection_class", "sequence_generated"]

df = pd.read_csv(SRC)


def select_protein(sub):
    sub = sub[sub.nmp_n_strong_binders > 0]   # exclude already-clean designs — nothing to refine
    valid = sub[sub.foldability_selection_class.eq("global_ok_active_site_ok")
                & sub.active_site_hard_anchor_mismatches.eq(0)]
    primary = valid.sort_values(
        ["nmp_n_strong_binders", "active_site_spatial_shell6A_ca_rmsd", "struct_scTM"],
        ascending=[True, True, False]).head(K).assign(selection_source="valid_quality")
    if len(primary) < K:
        rest = sub[~sub.design_idx.isin(primary.design_idx)]
        fb = rest.sort_values(["active_site_spatial_shell6A_ca_rmsd", "struct_scTM"],
                              ascending=[True, False]).head(K - len(primary)).assign(
            selection_source="structure_fallback")
        return pd.concat([primary, fb])
    return primary


for allele, tag in ALLELES.items():
    d = df[df.allele == allele]
    sel = pd.concat([select_protein(sub) for _, sub in d.groupby("protein_id")], ignore_index=True)
    sel["quality_rank"] = sel.groupby("protein_id").cumcount() + 1
    sel = sel[COLS].rename(columns={"sequence_generated": "sequence"}).sort_values(
        ["protein_id", "quality_rank"]).reset_index(drop=True)

    out = pathlib.Path(f"Results/RF/Uricase/refine_seed_selection_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(out / f"refine_seeds_{tag}.parquet", index=False)
    sel.drop(columns=["sequence"]).to_csv(out / f"refine_seeds_{tag}.csv", index=False)
    sel.to_csv(out / f"refine_seeds_{tag}_with_sequence.csv", index=False)

    vc = sel.selection_source.value_counts().to_dict()
    print(f"[{tag}] {len(sel)} seeds / {sel.protein_id.nunique()} proteins | source={vc} | "
          f"residual(nmp_strong) valid-picks median={sel[sel.selection_source=='valid_quality'].nmp_n_strong_binders.median():.0f} "
          f"range=[{sel.nmp_n_strong_binders.min()},{sel.nmp_n_strong_binders.max()}] | "
          f"mean scTM={sel.struct_scTM.mean():.3f} | fallback proteins="
          f"{sorted(sel[sel.selection_source=='structure_fallback'].protein_id.unique())}")
