"""M1: Epitope-head scoring bridge for classifier guidance.

Bridges candidate sequences from the DPLM sampler to the frozen epitope-head
predictor. Accepts a pre-built InferencePredictor (caller owns construction)
and returns risk scores compatible with M2 reweighting.

The bridge guarantees that its output matches calling predictor.predict_protein()
directly on each sequence.
"""

from typing import Any, Dict, List, Protocol

import numpy as np
import torch


class PredictorProtocol(Protocol):
    """Minimal interface required from the epitope-head predictor."""

    def predict_protein(self, seq: str, allele_idx: int = 0) -> dict: ...


def score_candidates(
    predictor: PredictorProtocol,
    sequences: List[str],
    allele_idx: int = 0,
    batch_size: int = 8,
) -> np.ndarray:
    """Score candidate sequences and return global risk array.

    Args:
        predictor: object with predict_protein(seq, allele_idx) → dict.
        sequences: list of candidate amino acid strings.
        allele_idx: HLA allele index (default 0).
        batch_size: number of sequences to score per micro-batch.
            Each sequence still calls predict_protein individually, but
            micro-batching controls memory pressure by allowing callers
            to interleave scoring with other work.

    Returns:
        1-D float64 array of global_risk values, one per candidate.

    Raises:
        ValueError: if sequences is empty.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    risks = np.empty(len(sequences), dtype=np.float64)
    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        for i in range(batch_start, batch_end):
            result = predictor.predict_protein(
                sequences[i], allele_idx=allele_idx,
            )
            risks[i] = result["global_risk"]

    return risks


def score_candidates_detailed(
    predictor: PredictorProtocol,
    sequences: List[str],
    allele_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Score candidates and return full prediction details per candidate.

    Same as score_candidates but preserves per-residue hotspot scores and
    window logits for downstream analysis (M4 failure analysis).

    Args:
        predictor: object with predict_protein(seq, allele_idx) → dict.
        sequences: list of candidate amino acid strings.
        allele_idx: HLA allele index (default 0).

    Returns:
        List of dicts, one per candidate, each containing:
          - global_risk: float
          - residue_hotspot: torch.Tensor [L]
          - n_windows: int
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    results = []
    for seq in sequences:
        pred = predictor.predict_protein(seq, allele_idx=allele_idx)
        results.append({
            "global_risk": pred["global_risk"],
            "residue_hotspot": pred["residue_hotspot"],
            "n_windows": pred["meta"]["n_windows"],
        })

    return results
