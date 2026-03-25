"""M2: Risk-weighted candidate reweighting and selection.

Implements the v1 classifier guidance selection algorithm:
  weights_i = exp(-eta * R_i) / sum_j exp(-eta * R_j)

Numerically stabilized via the log-sum-exp trick.
"""

import hashlib
from typing import Any, Dict, List

import numpy as np


def compute_weights(risks: np.ndarray, eta: float) -> np.ndarray:
    """Compute normalized risk-weighted selection probabilities.

    Args:
        risks: 1-D array of risk scores for each candidate.
        eta: guidance strength. eta=0 → uniform, higher → prefer low risk.

    Returns:
        1-D array of probabilities summing to 1.0.

    Raises:
        ValueError: if eta < 0 or risks is empty.
    """
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if risks.size == 0:
        raise ValueError("risks array must not be empty")

    risks = np.asarray(risks, dtype=np.float64)

    if eta == 0.0:
        return np.full(risks.shape, 1.0 / risks.size)

    # log-sum-exp trick for numerical stability:
    # w_i = exp(-eta * R_i)  →  log w_i = -eta * R_i
    # Subtract max(log w_i) before exponentiating
    log_weights = -eta * risks
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()

    if total == 0.0 or not np.isfinite(total):
        # Fallback: all candidates have same extreme weight → uniform
        return np.full(risks.shape, 1.0 / risks.size)

    return weights / total


def select_candidate(
    risks: np.ndarray,
    sequences: List[str],
    eta: float,
    seed: int,
) -> Dict[str, Any]:
    """Select one candidate via risk-weighted resampling.

    Args:
        risks: 1-D array of risk scores (one per candidate).
        sequences: list of candidate sequences (same length as risks).
        eta: guidance strength.
        seed: random seed for reproducibility.

    Returns:
        Dict with selection provenance:
          - selected_index: int
          - selected_risk: float
          - selected_seq_hash: str (sha256[:12])
          - weights: list[float]
          - candidate_risks: list[float]
    """
    risks = np.asarray(risks, dtype=np.float64)
    weights = compute_weights(risks, eta)

    rng = np.random.RandomState(seed)
    selected_index = int(rng.choice(len(risks), p=weights))

    selected_seq = sequences[selected_index]
    seq_hash = hashlib.sha256(selected_seq.encode()).hexdigest()[:12]

    return {
        "selected_index": selected_index,
        "selected_risk": float(risks[selected_index]),
        "selected_seq_hash": seq_hash,
        "weights": weights.tolist(),
        "candidate_risks": risks.tolist(),
    }


def select_candidates_batch(
    protein_risks: Dict[str, np.ndarray],
    protein_sequences: Dict[str, List[str]],
    eta: float,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Select one candidate per protein via risk-weighted resampling.

    Uses a per-protein sub-seed derived from the global seed and protein_id
    for reproducibility without cross-protein correlation.

    Args:
        protein_risks: {protein_id: risk_array}.
        protein_sequences: {protein_id: sequence_list}.
        eta: guidance strength.
        seed: global random seed.

    Returns:
        {protein_id: selection_provenance_dict}.
    """
    results = {}
    for pid in sorted(protein_risks.keys()):
        # Derive a per-protein sub-seed
        sub_seed = int(
            hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:8], 16
        ) % (2**31)

        results[pid] = select_candidate(
            protein_risks[pid],
            protein_sequences[pid],
            eta=eta,
            seed=sub_seed,
        )

    return results
