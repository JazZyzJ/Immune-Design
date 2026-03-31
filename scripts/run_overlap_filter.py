#!/usr/bin/env python
"""CLI: Run MMseqs2 overlap filter against CATH training sequences.

Outputs per-candidate overlap decisions for ALL candidates in the input
FASTA, including those with no CATH hits (marked as pass).

Usage:
    python scripts/run_overlap_filter.py \
        --candidate-fasta candidates.fasta \
        --cath-train-fasta cath_train.fasta \
        --output-json overlap_results.json \
        [--mmseqs-bin mmseqs] \
        [--threshold 0.3]
"""

import argparse
import json
import sys

from inverse_folding.evaluation.overlap import run_mmseqs_overlap, OverlapResult


def _read_fasta_ids(fasta_path: str) -> list:
    """Extract all sequence IDs from a FASTA file."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMseqs2 sequence identity overlap detection."
    )
    parser.add_argument(
        "--candidate-fasta", required=True,
        help="FASTA file of candidate protein sequences.",
    )
    parser.add_argument(
        "--cath-train-fasta", required=True,
        help="FASTA file of CATH training sequences.",
    )
    parser.add_argument(
        "--output-json", required=True,
        help="Output JSON file with per-candidate overlap decisions.",
    )
    parser.add_argument(
        "--mmseqs-bin", default="mmseqs",
        help="Path to the mmseqs binary (default: mmseqs).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Sequence identity threshold for exclusion (default: 0.3).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Get all candidate IDs from the input FASTA
    all_candidate_ids = _read_fasta_ids(args.candidate_fasta)

    results = run_mmseqs_overlap(
        candidate_fasta=args.candidate_fasta,
        cath_train_fasta=args.cath_train_fasta,
        mmseqs_bin=args.mmseqs_bin,
        threshold=args.threshold,
    )

    # Build complete results: include no-hit candidates as explicit pass
    complete_results = {}
    for qid in all_candidate_ids:
        if qid in results:
            complete_results[qid] = results[qid].to_dict()
        else:
            complete_results[qid] = OverlapResult(
                query_id=qid,
                best_target=None,
                best_identity=0.0,
                exclude=False,
            ).to_dict()

    n_total = len(all_candidate_ids)
    n_excluded = sum(1 for r in complete_results.values() if r["exclude"])
    n_passed = n_total - n_excluded

    output = {
        "parameters": {
            "candidate_fasta": args.candidate_fasta,
            "cath_train_fasta": args.cath_train_fasta,
            "threshold": args.threshold,
        },
        "n_total_candidates": n_total,
        "n_with_hits": len(results),
        "n_excluded": n_excluded,
        "n_passed": n_passed,
        "results": complete_results,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overlap filter: {n_total} candidates, "
          f"{n_excluded} excluded, {n_passed} passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
