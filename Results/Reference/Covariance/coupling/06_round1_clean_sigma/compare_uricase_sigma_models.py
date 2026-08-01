#!/usr/bin/env python3
"""Compare Q00511 sigma models and materialize clean-mask matrix counts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evcouplings.couplings.pairs import read_raw_ec_file
from scipy.stats import spearmanr
import yaml


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def anchor_map(path: Path) -> dict[int, dict]:
    raw = load_yaml(path)
    return {int(a["index_0b"]): a for a in raw["entries"][0]["hard_anchors"]}


def pos_set(df: pd.DataFrame, threshold: float) -> set[int]:
    return set(df.loc[df["in_mature"] & (df["sigma_pct"] >= threshold), "pos_mature"].astype(int))


def overlap(a: set, b: set) -> dict:
    return {
        "a_n": len(a),
        "b_n": len(b),
        "intersection_n": len(a & b),
        "union_n": len(a | b),
        "jaccard": len(a & b) / len(a | b) if (a | b) else None,
    }


def top_ec_pairs(path: Path, n: int) -> set[tuple[int, int]]:
    ecs = read_raw_ec_file(path, sort=True, score="cn")
    long_range = ecs.query("abs(i-j) >= 6").sort_values("cn", ascending=False).head(n)
    return {(int(r.i), int(r.j)) for _, r in long_range.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-scores", required=True)
    parser.add_argument("--new-scores", required=True)
    parser.add_argument("--old-ecs", required=True)
    parser.add_argument("--new-ecs", required=True)
    parser.add_argument("--old-summary", required=True)
    parser.add_argument("--new-summary", required=True)
    parser.add_argument("--new-audit", required=True)
    parser.add_argument("--new-sensitivity-tsv", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    old = pd.read_csv(args.old_scores, sep="\t")
    new = pd.read_csv(args.new_scores, sep="\t")
    old_summary = json.loads(Path(args.old_summary).read_text())
    new_summary = json.loads(Path(args.new_summary).read_text())
    audit = json.loads(Path(args.new_audit).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mature = old["in_mature"] & new["in_mature"]
    rho, rho_p = spearmanr(old.loc[mature, "sigma_raw"], new.loc[mature, "sigma_raw"])
    model = {
        "old_iterations": old_summary["iterations"],
        "old_opt_status": old_summary["opt_status"],
        "new_iterations": new_summary["iterations"],
        "new_opt_status": new_summary["opt_status"],
        "new_runtime_sec": new_summary["runtime_sec"],
        "effective_samples": new_summary["effective_samples"],
        "neff_per_L": new_summary["effective_samples"] / new_summary["L"],
        "sigma_raw_spearman_mature": float(rho),
        "sigma_raw_spearman_p": float(rho_p),
        "top_L_EC_pair_overlap": overlap(top_ec_pairs(Path(args.old_ecs), 302), top_ec_pairs(Path(args.new_ecs), 302)),
        "thresholds": {},
    }
    for threshold in (0.9, 0.8):
        key = str(threshold)
        old_raw = pos_set(old, threshold)
        new_raw = pos_set(new, threshold)
        new_clean = set(audit["thresholds"][key]["clean_positions_index_0b"])
        old_clean = {
            int(r.pos_mature)
            for _, r in old.iterrows()
            if bool(r.in_mature)
            and r.sigma_pct >= threshold
            and r.C_i_nogap < 0.5
            and r.gap_frac <= 0.2
        }
        model["thresholds"][key] = {
            "raw_old_vs_new": overlap(old_raw, new_raw),
            "clean_old_vs_new": overlap(old_clean, new_clean),
            "new_clean_n": len(new_clean),
            "new_robust_clean_n": audit["thresholds"][key]["robust_clean_n"],
            "new_removed_by_C_n": len(audit["thresholds"][key]["removed_by_C_ge_0p5"]),
            "new_removed_by_gap_n": len(audit["thresholds"][key]["removed_by_gap_gt_0p2"]),
        }
    (out / "model_comparison.json").write_text(json.dumps(model, indent=2) + "\n")

    cfg = Path(args.config_dir)
    loose = set(anchor_map(cfg / "uricase_q00511_active_site_safety_v1.yaml"))
    full_map = anchor_map(cfg / "uricase_q00511_r1_Sfull_sigOFF.yaml")
    full = set(full_map)
    ab = {idx for idx, rec in full_map.items() if "iface_AB" in rec["biological_role"]}
    structures = {"loose": loose, "mid": loose | ab, "full": full}
    rows = []
    for row, structure in structures.items():
        for threshold in (0.9, 0.8):
            sigma = set(audit["thresholds"][str(threshold)]["clean_positions_index_0b"])
            union = structure | sigma
            rows.append({
                "structure_level": row,
                "sigma_threshold": threshold,
                "structure_n": len(structure),
                "clean_sigma_n": len(sigma),
                "intersection_n": len(structure & sigma),
                "fixed_union_n": len(union),
                "fixed_union_pct_of_301": round(100 * len(union) / 301, 1),
            })
    rows.append({
        "structure_level": "full",
        "sigma_threshold": "OFF",
        "structure_n": len(full),
        "clean_sigma_n": 0,
        "intersection_n": 0,
        "fixed_union_n": len(full),
        "fixed_union_pct_of_301": round(100 * len(full) / 301, 1),
    })
    pd.DataFrame(rows).to_csv(out / "matrix_counts.tsv", sep="\t", index=False)

    detail = pd.read_csv(args.new_sensitivity_tsv, sep="\t")
    clean90 = set(audit["thresholds"]["0.9"]["clean_positions_index_0b"])
    clean80 = set(audit["thresholds"]["0.8"]["clean_positions_index_0b"])
    detail["in_clean_sigma90"] = detail["pos_mature"].isin(clean90)
    detail["in_clean_sigma80"] = detail["pos_mature"].isin(clean80)
    detail["passes_C_negative_filter"] = detail["C_i_nogap"] < 0.5
    detail["passes_gap_filter"] = detail["gap_frac"] <= 0.2
    detail.to_csv(out / "clean_sigma_positions.tsv", sep="\t", index=False, float_format="%.6f")

    print(json.dumps(model, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
