#!/usr/bin/env python
"""CLI: Validate Tier 3 candidate JSON for the IF test set.

Performs metadata validation and, when --nmp-scores-json is provided,
checks that each candidate meets the minimum DRB1*07:01 NMP signal
(>= 3 windows at %Rank < 5%).

Usage:
    python scripts/validate_tier3.py \
        --candidates outputs/if/test_set/tier3_candidates.json \
        [--nmp-scores-json outputs/if/test_set/tier3_nmp_scores.json]
"""

import argparse
import json
import sys

from inverse_folding.evaluation.tier_validators import validate_tier3_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Tier 3 (therapeutic) candidate list."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to tier3_candidates.json.",
    )
    parser.add_argument(
        "--nmp-scores-json", default=None,
        help="Path to pre-computed NMP scores JSON "
             "(dict mapping protein_id → {n_strong_binders: int, ...}). "
             "When provided, enforces NMP signal threshold.",
    )
    args = parser.parse_args()

    nmp_scores = None
    if args.nmp_scores_json:
        with open(args.nmp_scores_json) as f:
            nmp_scores = json.load(f)
        print(f"Loaded NMP scores for {len(nmp_scores)} proteins.")
    else:
        print("WARNING: --nmp-scores-json not provided; "
              "NMP signal threshold NOT enforced.")

    result = validate_tier3_candidates(args.candidates, nmp_scores=nmp_scores)

    print(f"Tier 3 validation: {result['n_valid']} valid, "
          f"{result['n_invalid']} invalid")

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
