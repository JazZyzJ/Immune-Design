#!/usr/bin/env python3
"""Compare original and iter500 Q00511 models under the rescue-v2 sigma rule."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from evcouplings.couplings.pairs import read_raw_ec_file
from scipy.stats import spearmanr


def overlap(a: set, b: set) -> dict[str, float | int | None]:
    union = a | b
    return {
        "old_n": len(a),
        "new_n": len(b),
        "intersection_n": len(a & b),
        "union_n": len(union),
        "jaccard": len(a & b) / len(union) if union else None,
    }


def position_set(df: pd.DataFrame, threshold: float, gap_cap: float | None) -> set[int]:
    mask = (df["pos_mature"] >= 1) & (df["sigma_pct"] >= threshold)
    if gap_cap is not None:
        mask &= df["gap_frac"] <= gap_cap
    return set(df.loc[mask, "pos_mature"].astype(int))


def top_pairs(path: Path, n: int = 302) -> set[tuple[int, int]]:
    ecs = read_raw_ec_file(path, sort=True, score="cn")
    rows = ecs.query("abs(i-j) >= 6").sort_values("cn", ascending=False).head(n)
    return {(int(row.i), int(row.j)) for _, row in rows.iterrows()}


def top_l_tetramer_precision(path: Path) -> float:
    pattern = re.compile(r"^\s+L\s+[-+0-9.]+\s+([-+0-9.]+)\s+\d+\s+\d+\s*$")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise ValueError(f"top-L precision row not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-scores", required=True)
    parser.add_argument("--new-scores", required=True)
    parser.add_argument("--old-ecs", required=True)
    parser.add_argument("--new-ecs", required=True)
    parser.add_argument("--old-summary", required=True)
    parser.add_argument("--new-summary", required=True)
    parser.add_argument("--new-validation", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    old = pd.read_csv(args.old_scores, sep="\t").sort_values("pos_mature")
    new = pd.read_csv(args.new_scores, sep="\t").sort_values("pos_mature")
    merged = old[["pos_mature", "sigma_raw"]].merge(
        new[["pos_mature", "sigma_raw"]], on="pos_mature", suffixes=("_old", "_new")
    )
    merged = merged.loc[merged["pos_mature"] >= 1]
    rho, pvalue = spearmanr(merged["sigma_raw_old"], merged["sigma_raw_new"])
    old_summary = json.loads(Path(args.old_summary).read_text())
    new_summary = json.loads(Path(args.new_summary).read_text())

    payload = {
        "old_iterations": old_summary["iterations"],
        "old_opt_status": old_summary["opt_status"],
        "new_iterations": new_summary["iterations"],
        "new_opt_status": new_summary["opt_status"],
        "new_runtime_sec": new_summary["runtime_sec"],
        "new_effective_samples": new_summary["effective_samples"],
        "new_neff_per_L": new_summary["effective_samples"] / new_summary["L"],
        "new_top_L_tetramer_contact_precision": top_l_tetramer_precision(
            Path(args.new_validation)
        ),
        "sigma_raw_spearman_mature": float(rho),
        "sigma_raw_spearman_p": float(pvalue),
        "top_L_EC_pair_overlap": overlap(top_pairs(Path(args.old_ecs)), top_pairs(Path(args.new_ecs))),
        "thresholds": {},
    }
    for threshold in (0.9, 0.8):
        key = f"{threshold:.1f}"
        payload["thresholds"][key] = {
            "raw_old_vs_new": overlap(
                position_set(old, threshold, None), position_set(new, threshold, None)
            ),
            "rescue_v2_gap0p5_old_vs_new": overlap(
                position_set(old, threshold, 0.5), position_set(new, threshold, 0.5)
            ),
        }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
