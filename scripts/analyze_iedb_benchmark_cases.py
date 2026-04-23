#!/usr/bin/env python
"""Rank IEDB benchmark proteins for follow-up error analysis.

This script reads the JSON produced by scripts/benchmark_iedb_test.py and
emits:
  - a per-protein CSV sorted by NMP AP - head AP
  - a short markdown summary

The benchmark JSON does not store per-window coordinates, so this script can
identify candidate proteins for follow-up, but cannot decide whether head
false positives are near-miss windows adjacent to the annotated epitope.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze IEDB benchmark protein-level error cases.",
    )
    p.add_argument("--input-json", required=True, help="benchmark_iedb_test.py JSON")
    p.add_argument("--output-prefix", default=None,
                   help="Output prefix without extension "
                        "(default: <json_stem>_case_analysis in JSON dir)")
    p.add_argument("--min-positives", type=int, default=4,
                   help="Minimum GT positives for robust-candidate list "
                        "(default: 4).")
    p.add_argument("--top-n", type=int, default=30,
                   help="Rows to show in markdown top lists (default: 30).")
    return p.parse_args()


def _metric(row: dict[str, Any], predictor: str, metric: str) -> float | None:
    bundle = row.get("metrics", {}).get(f"{predictor}_primary")
    if not bundle:
        return None
    value = bundle.get(metric)
    return None if value is None else float(value)


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("per_protein", []):
        head_ap = _metric(row, "head", "ap")
        nmp_ap = _metric(row, "nmp", "ap")
        head_auc = _metric(row, "head", "auc")
        nmp_auc = _metric(row, "nmp", "auc")
        head_recall_50 = _metric(row, "head", "recall_50")
        nmp_recall_50 = _metric(row, "nmp", "recall_50")
        head_precision_10 = _metric(row, "head", "precision_10")
        nmp_precision_10 = _metric(row, "nmp", "precision_10")
        head_par50 = _metric(row, "head", "precision_at_recall_50")
        nmp_par50 = _metric(row, "nmp", "precision_at_recall_50")

        rows.append({
            "protein_id": row.get("protein_id"),
            "sequence_length": row.get("sequence_length"),
            "n_pos": row.get("n_positives_in_windows"),
            "n_windows": row.get("n_windows_total"),
            "head_ap": head_ap,
            "nmp_ap": nmp_ap,
            "ap_gap_nmp_minus_head": _sub(nmp_ap, head_ap),
            "head_auc": head_auc,
            "nmp_auc": nmp_auc,
            "auc_gap_nmp_minus_head": _sub(nmp_auc, head_auc),
            "head_recall_50": head_recall_50,
            "nmp_recall_50": nmp_recall_50,
            "recall_50_gap": _sub(nmp_recall_50, head_recall_50),
            "head_precision_10": head_precision_10,
            "nmp_precision_10": nmp_precision_10,
            "precision_10_gap": _sub(nmp_precision_10, head_precision_10),
            "head_precision_at_recall_50": head_par50,
            "nmp_precision_at_recall_50": nmp_par50,
            "precision_at_recall_50_gap": _sub(nmp_par50, head_par50),
            "head_status": row.get("head_status"),
            "nmp_status": row.get("nmp_status"),
        })
    return rows


def _distribution_counts(payload: dict[str, Any]) -> dict[str, tuple[int, int]]:
    keys = ("head_primary", "nmp_primary", "head_conditional", "nmp_conditional")
    counts = {key: [0, 0] for key in keys}
    for row in payload.get("per_protein", []):
        for key in keys:
            block = row.get("score_distributions", {}).get(key, {})
            counts[key][0] += len(block.get("pos_scores", []))
            counts[key][1] += len(block.get("neg_scores", []))
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _summary_lines(
    rows: list[dict[str, Any]],
    counts: dict[str, tuple[int, int]],
    min_positives: int,
    top_n: int,
) -> list[str]:
    lines: list[str] = []
    lines.append("# IEDB Benchmark Case Analysis")
    lines.append("")
    lines.append("## Global Score Distribution Counts")
    lines.append("")
    lines.append("| Distribution | positives | negatives |")
    lines.append("|---|---:|---:|")
    for key, (pos, neg) in counts.items():
        lines.append(f"| {key} | {pos} | {neg} |")
    lines.append("")

    npos_counter = collections.Counter(int(r["n_pos"]) for r in rows)
    lines.append("## Positive-Window Count Per Protein")
    lines.append("")
    lines.append("| n_pos | protein_count |")
    lines.append("|---:|---:|")
    for n_pos in sorted(npos_counter):
        lines.append(f"| {n_pos} | {npos_counter[n_pos]} |")
    lines.append("")

    def gap(row: dict[str, Any]) -> float:
        value = row.get("ap_gap_nmp_minus_head")
        return -1e9 if value is None else float(value)

    all_gaps = [float(r["ap_gap_nmp_minus_head"]) for r in rows
                if r.get("ap_gap_nmp_minus_head") is not None]
    if all_gaps:
        lines.append("## AP Gap Summary")
        lines.append("")
        lines.append(
            f"n={len(all_gaps)}, mean={sum(all_gaps) / len(all_gaps):.4f}, "
            f"median={statistics.median(all_gaps):.4f}, "
            f"min={min(all_gaps):.4f}, max={max(all_gaps):.4f}"
        )
        lines.append("")

    robust = [
        row for row in rows
        if row.get("n_pos") is not None
        and int(row["n_pos"]) >= min_positives
        and row.get("ap_gap_nmp_minus_head") is not None
    ]
    robust = sorted(robust, key=gap, reverse=True)[:top_n]
    lines.append(f"## Robust NMP-Better Candidates (n_pos >= {min_positives})")
    lines.append("")
    lines.append("| protein_id | gap | head_ap | nmp_ap | n_pos | length | recall50_gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in robust:
        lines.append(
            f"| {row['protein_id']} | {row['ap_gap_nmp_minus_head']:.4f} | "
            f"{row['head_ap']:.4f} | {row['nmp_ap']:.4f} | "
            f"{row['n_pos']} | {row['sequence_length']} | "
            f"{row['recall_50_gap']:.4f} |"
        )
    lines.append("")
    lines.append("## Limitation")
    lines.append("")
    lines.append(
        "This JSON contains per-label score distributions but not per-window "
        "coordinates. Use the candidate list above for a follow-up run that "
        "exports `(protein_id, start_0b, end_0b, label, head_score, nmp_rank)` "
        "to test whether head high-scoring false positives are near-miss "
        "windows around annotated epitopes."
    )
    return lines


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json)
    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else input_path.with_name(f"{input_path.stem}_case_analysis")
    )

    with input_path.open() as f:
        payload = json.load(f)

    rows = _build_rows(payload)
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -1e9 if row.get("ap_gap_nmp_minus_head") is None
            else float(row["ap_gap_nmp_minus_head"])
        ),
        reverse=True,
    )
    counts = _distribution_counts(payload)

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    _write_csv(csv_path, rows_sorted)
    md_path.write_text(
        "\n".join(_summary_lines(rows, counts, args.min_positives, args.top_n)) + "\n"
    )

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
