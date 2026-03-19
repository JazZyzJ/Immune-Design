"""Negative span sampler with positive exclusion and hard/easy ratio.

Implements PLAN.md Task E2 and codemap §7:
  - Global positive exclusion per protein
  - Hard negatives: offset-based near positives
  - Easy negatives: random spans
  - Length sampling: match_positive distribution or uniform
  - Fail-fast on ratio shortfall (strict mode)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NegativeSamplingShortfall(RuntimeError):
    """Raised when target negative count cannot be met in strict mode."""
    pass


def sample_negatives(
    protein_length: int,
    positives: list[dict],
    neg_ratio: int = 15,
    hard_negative_fraction: float = 0.3,
    hard_neg_max_overlap_ratio: float = 0.8,
    hard_neg_offset_range: int = 20,
    neg_length_sampling: str = "match_positive",
    min_k: int = 12,
    max_k: int = 25,
    rng: np.random.RandomState | None = None,
    strict: bool = True,
) -> list[dict]:
    """Sample negative spans for one protein.

    Args:
        protein_length: full protein length L.
        positives: list of {start_0b, end_0b, pep_len, ...} dicts.
        neg_ratio: negatives per positive.
        hard_negative_fraction: fraction of negatives that are hard.
        hard_neg_max_overlap_ratio: hard negs must have overlap < this * pos_len.
        hard_neg_offset_range: max offset for hard negative generation.
        neg_length_sampling: "match_positive" or "uniform".
        min_k, max_k: allowed peptide length range.
        rng: random state for reproducibility.
        strict: if True, raise NegativeSamplingShortfall when target count
            or hard/easy ratio cannot be met. If False, backfill hard
            shortfall with easy negatives and log a warning.

    Returns:
        List of negative span dicts {start_0b, end_0b, pep_len}.

    Raises:
        NegativeSamplingShortfall: in strict mode, when total count is not met
            or hard negative target is not met.
    """
    if rng is None:
        rng = np.random.RandomState()

    if not positives:
        return []

    # Build positive exclusion set
    positive_set = {(p["start_0b"], p["end_0b"]) for p in positives}

    # Build positive length distribution for match_positive sampling
    pos_lengths = [p["pep_len"] for p in positives]

    total_neg = neg_ratio * len(positives)
    n_hard = int(round(total_neg * hard_negative_fraction))
    n_easy = total_neg - n_hard

    hard_negatives = []
    easy_negatives = []

    # --- Hard negatives: offset from random positive ---
    hard_attempts = 0
    max_hard_attempts = n_hard * 20
    while len(hard_negatives) < n_hard and hard_attempts < max_hard_attempts:
        hard_attempts += 1
        anchor = positives[rng.randint(len(positives))]
        a_start = anchor["start_0b"]
        a_end = anchor["end_0b"]
        a_len = anchor["pep_len"]

        offset = rng.randint(1, hard_neg_offset_range + 1)
        if rng.random() < 0.5:
            offset = -offset

        new_start = a_start + offset
        new_end = new_start + a_len

        if new_start < 0 or new_end > protein_length:
            continue

        overlap_start = max(a_start, new_start)
        overlap_end = min(a_end, new_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap >= hard_neg_max_overlap_ratio * a_len:
            continue

        if (new_start, new_end) in positive_set:
            continue

        hard_negatives.append({
            "start_0b": new_start,
            "end_0b": new_end,
            "pep_len": a_len,
        })

    # Check hard negative shortfall
    hard_shortfall = n_hard - len(hard_negatives)
    if hard_shortfall > 0:
        if strict:
            raise NegativeSamplingShortfall(
                f"Hard negative shortfall: wanted {n_hard}, got {len(hard_negatives)} "
                f"(protein_length={protein_length}, n_positives={len(positives)})"
            )
        else:
            logger.warning(
                "Hard negative shortfall: wanted %d, got %d for protein_length=%d. "
                "Backfilling %d with easy negatives.",
                n_hard, len(hard_negatives), protein_length, hard_shortfall,
            )
            # Backfill: increase easy target to compensate
            n_easy += hard_shortfall

    # --- Easy negatives: random spans ---
    easy_attempts = 0
    max_easy_attempts = n_easy * 20
    while len(easy_negatives) < n_easy and easy_attempts < max_easy_attempts:
        easy_attempts += 1

        if neg_length_sampling == "match_positive":
            k = pos_lengths[rng.randint(len(pos_lengths))]
        else:
            k = rng.randint(min_k, max_k + 1)

        if k > protein_length:
            continue

        start = rng.randint(0, protein_length - k + 1)
        end = start + k

        if (start, end) in positive_set:
            continue

        easy_negatives.append({
            "start_0b": start,
            "end_0b": end,
            "pep_len": k,
        })

    negatives = hard_negatives + easy_negatives

    # Check total shortfall
    total_shortfall = total_neg - len(negatives)
    if total_shortfall > 0:
        if strict:
            raise NegativeSamplingShortfall(
                f"Total negative shortfall: wanted {total_neg}, got {len(negatives)} "
                f"(protein_length={protein_length}, n_positives={len(positives)})"
            )
        else:
            logger.warning(
                "Total negative shortfall: wanted %d, got %d for protein_length=%d.",
                total_neg, len(negatives), protein_length,
            )

    return negatives
