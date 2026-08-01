#!/usr/bin/env python3
"""Build Q00511 rescue-first sigma/conservation masks and exact matrix counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


SIGMA_THRESHOLDS = (0.9, 0.8)
GAP_CAPS = (0.2, 0.3, 0.4, 0.5, 0.6, 1.0)
MAIN_GAP_CAP = 0.5
BASE_C_THRESHOLD = 0.8
ALT_C_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anchor_records(path: Path) -> dict[int, dict]:
    raw = yaml.safe_load(path.read_text())
    return {int(row["index_0b"]): row for row in raw["entries"][0]["hard_anchors"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-matrix-tsv", required=True)
    parser.add_argument("--out-positions-tsv", required=True)
    args = parser.parse_args()

    scores_path = Path(args.scores)
    config_dir = Path(args.config_dir)
    scores = pd.read_csv(scores_path, sep="\t")
    mature = scores.loc[scores["pos_mature"] >= 1].copy()
    mature["index_0b"] = mature["pos_mature"].astype(int)

    loose_records = load_anchor_records(config_dir / "uricase_q00511_active_site_safety_v1.yaml")
    full_records = load_anchor_records(config_dir / "uricase_q00511_r1_Sfull_sigOFF.yaml")
    loose = set(loose_records)
    full = set(full_records)
    mid = loose | {
        idx for idx, row in full_records.items() if "iface_AB" in str(row["biological_role"])
    }
    structures = {"loose": loose, "mid": mid, "full": full}
    expected_structure_counts = {"loose": 24, "mid": 79, "full": 137}
    actual_structure_counts = {name: len(values) for name, values in structures.items()}
    if actual_structure_counts != expected_structure_counts:
        raise ValueError(f"unexpected structure counts: {actual_structure_counts}")

    c08 = set(
        mature.loc[mature["C_i_nogap"] >= BASE_C_THRESHOLD, "index_0b"].astype(int)
    ) - full
    c05 = set(
        mature.loc[mature["C_i_nogap"] >= ALT_C_THRESHOLD, "index_0b"].astype(int)
    ) - full
    expected_c08 = {40, 50, 77, 145, 157, 186, 224, 301}
    if c08 != expected_c08:
        raise ValueError(f"C>=0.8 outside-full set changed: {sorted(c08)}")
    if len(c05) != 50:
        raise ValueError(f"expected 50 C>=0.5 outside-full positions, found {len(c05)}")

    sigma_by_threshold: dict[str, set[int]] = {}
    gap_sensitivity: dict[str, dict[str, object]] = {}
    for threshold in SIGMA_THRESHOLDS:
        key = f"{threshold:.1f}"
        raw = set(
            mature.loc[mature["sigma_pct"] >= threshold, "index_0b"].astype(int)
        )
        main = set(
            mature.loc[
                (mature["sigma_pct"] >= threshold)
                & (mature["gap_frac"] <= MAIN_GAP_CAP),
                "index_0b",
            ].astype(int)
        )
        sigma_by_threshold[key] = main
        cap_rows = {}
        for cap in GAP_CAPS:
            selected = set(
                mature.loc[
                    (mature["sigma_pct"] >= threshold) & (mature["gap_frac"] <= cap),
                    "index_0b",
                ].astype(int)
            )
            cap_rows[f"{cap:.1f}"] = {
                "n": len(selected),
                "positions_index_0b": sorted(selected),
                "excluded_from_raw_index_0b": sorted(raw - selected),
            }
        gap_sensitivity[key] = {
            "raw_n": len(raw),
            "raw_positions_index_0b": sorted(raw),
            "main_n": len(main),
            "main_positions_index_0b": sorted(main),
            "caps": cap_rows,
        }

    if not sigma_by_threshold["0.9"] < sigma_by_threshold["0.8"]:
        raise ValueError("sigma90 must be a strict subset of sigma80")
    if len(sigma_by_threshold["0.9"]) != 31 or len(sigma_by_threshold["0.8"]) != 61:
        raise ValueError("main sigma counts changed")

    matrix_rows: list[dict[str, object]] = []
    for structure_name, structure in structures.items():
        base = structure | c08
        for threshold in SIGMA_THRESHOLDS:
            key = f"{threshold:.1f}"
            sigma = sigma_by_threshold[key]
            fixed = base | sigma
            matrix_rows.append(
                {
                    "structure_level": structure_name,
                    "sigma_threshold": key,
                    "original_structure_n": len(structure),
                    "always_fix_C08_n": len(c08),
                    "structure_plus_C08_n": len(base),
                    "sigma_n": len(sigma),
                    "base_sigma_overlap_n": len(base & sigma),
                    "fixed_union_n": len(fixed),
                    "fixed_union_pct_of_301": round(100 * len(fixed) / 301, 1),
                }
            )
    full_base = full | c08
    matrix_rows.append(
        {
            "structure_level": "full",
            "sigma_threshold": "OFF",
            "original_structure_n": len(full),
            "always_fix_C08_n": len(c08),
            "structure_plus_C08_n": len(full_base),
            "sigma_n": 0,
            "base_sigma_overlap_n": 0,
            "fixed_union_n": len(full_base),
            "fixed_union_pct_of_301": round(100 * len(full_base) / 301, 1),
        }
    )
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(args.out_matrix_tsv, sep="\t", index=False)

    position_rows = mature[
        [
            "index_0b",
            "pos_uniprot",
            "wt_aa",
            "C_i_nogap",
            "gap_frac",
            "sigma_pct",
            "sigma_raw",
        ]
    ].copy()
    position_rows["in_structure_loose"] = position_rows["index_0b"].isin(loose)
    position_rows["in_structure_mid"] = position_rows["index_0b"].isin(mid)
    position_rows["in_structure_full"] = position_rows["index_0b"].isin(full)
    position_rows["in_always_fix_C08"] = position_rows["index_0b"].isin(c08)
    position_rows["in_outside_full_C05"] = position_rows["index_0b"].isin(c05)
    position_rows["in_sigma90_gap0p5"] = position_rows["index_0b"].isin(
        sigma_by_threshold["0.9"]
    )
    position_rows["in_sigma80_gap0p5"] = position_rows["index_0b"].isin(
        sigma_by_threshold["0.8"]
    )
    position_rows.to_csv(args.out_positions_tsv, sep="\t", index=False, float_format="%.5f")

    alt_counts = {}
    for threshold in SIGMA_THRESHOLDS:
        key = f"{threshold:.1f}"
        sigma = sigma_by_threshold[key]
        alt_counts[key] = {
            name: len(structure | c05 | sigma) for name, structure in structures.items()
        }
    alt_counts["OFF"] = {"full": len(full | c05)}

    payload = {
        "source_scores": str(scores_path),
        "source_scores_sha256": sha256(scores_path),
        "numbering": "index_0b == pos_mature == pos_uniprot - 1",
        "main_sigma_rule": "sigma_pct >= threshold AND gap_frac <= 0.5; no C filter",
        "main_gap_cap": MAIN_GAP_CAP,
        "structure_counts": actual_structure_counts,
        "always_fix_conservation": {
            "rule": "C_i_nogap >= 0.8 AND outside original full structure",
            "n": len(c08),
            "positions_index_0b": sorted(c08),
            "positions_uniprot_1b": sorted(int(x) + 1 for x in c08),
        },
        "alternative_C05_outside_full": {
            "n": len(c05),
            "positions_index_0b": sorted(c05),
            "matrix_union_counts": alt_counts,
        },
        "sigma": gap_sensitivity,
        "matrix": matrix_rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload["always_fix_conservation"], indent=2))
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
