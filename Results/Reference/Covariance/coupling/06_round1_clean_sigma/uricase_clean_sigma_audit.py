#!/usr/bin/env python3
"""Audit raw and filtered Q00511 per-residue sigma sets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evcouplings.couplings.pairs import enrichment, read_raw_ec_file


def sigma_percentile(ecs: pd.DataFrame, length: int, num_pairs: float) -> np.ndarray:
    enriched = enrichment(ecs, num_pairs=num_pairs, score="cn", min_seqdist=6)
    sigma = pd.Series(0.0, index=pd.RangeIndex(1, length + 1, name="i"))
    sigma.update(enriched.set_index("i")["enrichment"])
    return sigma.rank(pct=True).to_numpy()


def positions(df: pd.DataFrame, mask: pd.Series) -> list[int]:
    return sorted(df.loc[mask & df["in_mature"], "pos_mature"].astype(int).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--ecs", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tsv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scores, sep="\t").sort_values("pos_uniprot").reset_index(drop=True)
    ecs = read_raw_ec_file(args.ecs, sort=True, score="cn")
    for num_pairs, tag in ((0.5, "0p5L"), (1.0, "1p0L"), (2.0, "2p0L")):
        df[f"sigma_pct_{tag}"] = sigma_percentile(ecs, len(df), num_pairs)

    # The clean gate is deliberately only a negative control on obvious non-sigma
    # explanations: moderate/strong single-site conservation and gappy columns.
    base_clean = (df["C_i_nogap"] < 0.5) & (df["gap_frac"] <= 0.2)
    payload = {
        "source_scores": str(Path(args.scores)),
        "source_ecs": str(Path(args.ecs)),
        "numbering": "pos_mature == index_0b; initiator Met excluded",
        "clean_rule": "sigma_pct_1p0L >= threshold AND C_i_nogap < 0.5 AND gap_frac <= 0.2",
        "sensitivity_rule": "same percentile threshold under 0.5L, 1.0L, and 2.0L top-pair definitions",
        "thresholds": {},
    }
    for threshold in (0.9, 0.8):
        raw = df["sigma_pct_1p0L"] >= threshold
        clean = raw & base_clean
        robust = clean.copy()
        for tag in ("0p5L", "2p0L"):
            robust &= df[f"sigma_pct_{tag}"] >= threshold
        payload["thresholds"][str(threshold)] = {
            "raw_n": len(positions(df, raw)),
            "raw_positions_index_0b": positions(df, raw),
            "clean_n": len(positions(df, clean)),
            "clean_positions_index_0b": positions(df, clean),
            "robust_clean_n": len(positions(df, robust)),
            "robust_clean_positions_index_0b": positions(df, robust),
            "removed_by_C_ge_0p5": positions(df, raw & (df["C_i_nogap"] >= 0.5)),
            "removed_by_gap_gt_0p2": positions(df, raw & (df["gap_frac"] > 0.2)),
            "clean_not_robust": positions(df, clean & ~robust),
        }

    out_json = Path(args.out_json)
    out_tsv = Path(args.out_tsv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.6f")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
