#!/usr/bin/env python
"""CLI: Validate Tier 1 candidate JSON for the IF test set.

Performs JSON schema validation and, when --cath-train-fasta is provided,
runs MMseqs2 overlap detection to annotate/exclude CATH-overlapping proteins.

Usage:
    python scripts/validate_tier1.py \
        --candidates outputs/if/test_set/tier1_candidates.json \
        --candidate-fasta outputs/if/test_set/tier1_sequences.fasta \
        --cath-train-fasta work/cath/cath_train_seqs.fasta
"""

import argparse
import json
import sys

from inverse_folding.evaluation.tier_validators import validate_tier1_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Tier 1 (gold standard) candidate list."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to tier1_candidates.json.",
    )
    parser.add_argument(
        "--candidate-fasta", default=None,
        help="FASTA of Tier 1 candidate sequences (required for overlap check).",
    )
    parser.add_argument(
        "--cath-train-fasta", default=None,
        help="CATH training sequences FASTA. When provided with "
             "--candidate-fasta, runs MMseqs2 overlap detection.",
    )
    parser.add_argument(
        "--mmseqs-bin", default="mmseqs",
        help="Path to mmseqs binary (default: mmseqs).",
    )
    args = parser.parse_args()

    # Run MMseqs2 overlap if both FASTA files provided
    overlap_results = None
    if args.candidate_fasta and args.cath_train_fasta:
        print("Running MMseqs2 overlap detection...")
        from inverse_folding.evaluation.overlap import run_mmseqs_overlap
        overlap_results = run_mmseqs_overlap(
            candidate_fasta=args.candidate_fasta,
            cath_train_fasta=args.cath_train_fasta,
            mmseqs_bin=args.mmseqs_bin,
        )
        n_excluded = sum(1 for r in overlap_results.values() if r.exclude)
        print(f"  {n_excluded} candidates flagged for CATH overlap.")
    elif args.cath_train_fasta and not args.candidate_fasta:
        print("WARNING: --cath-train-fasta provided without --candidate-fasta; "
              "skipping overlap check.")

    result = validate_tier1_candidates(args.candidates, overlap_results)

    n_overlap = result.get("n_overlap_excluded", 0)
    print(f"Tier 1 validation: {result['n_valid']} valid, "
          f"{result['n_invalid']} invalid "
          f"({n_overlap} excluded by CATH overlap)")

    if result["errors"]:
        print("\nErrors:")
        for pid, errs in result["errors"].items():
            for e in errs:
                print(f"  {pid}: {e}")

    out_path = args.candidates.replace(".json", "_validated.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nValidation report written to: {out_path}")

    return 1 if result["n_invalid"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
