"""Training-time helpers for the IPA refiner."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_refiner_cross_entropy(
    logits: torch.Tensor,
    target_aa_tokens: torch.Tensor,
    *,
    selected_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over selected & valid residue positions only.

    ``logits`` has shape ``[B, L, 20]``; ``target_aa_tokens`` are AA
    indices in ``[0, 20)`` of shape ``[B, L]``. The loss is averaged over
    the positions where ``selected_mask & valid_mask`` is true. If no
    positions qualify, returns ``logits.sum() * 0.0`` so the autograd
    graph stays connected.
    """
    if logits.shape[-1] != 20:
        raise ValueError(
            f"logits last dim must be 20; got {logits.shape[-1]}"
        )
    if target_aa_tokens.shape != logits.shape[:-1]:
        raise ValueError(
            f"target_aa_tokens shape {tuple(target_aa_tokens.shape)} != "
            f"logits[:-1] {tuple(logits.shape[:-1])}"
        )

    mask = selected_mask & valid_mask
    if mask.sum() == 0:
        return logits.sum() * 0.0

    flat_logits = logits[mask]  # [N, 20]
    flat_target = target_aa_tokens[mask]  # [N]
    return F.cross_entropy(flat_logits, flat_target.long())
