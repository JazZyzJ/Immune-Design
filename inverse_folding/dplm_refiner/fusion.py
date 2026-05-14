"""Entropy-weighted base/refiner logit fusion.

Mirrors ``MapDiff/utils.py::fuse_logits_by_log_probs`` (lines 229-238):
weights for the two logits streams come from a softmax over their
negative entropies. Lower entropy ⇒ higher weight.
"""

from __future__ import annotations

import torch

from inverse_folding.dplm_refiner.entropy import mapdiff_entropy_from_log_probs


def fuse_logits_by_entropy(
    base_logits: torch.Tensor,
    refiner_logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Entropy-weighted fusion of two logit streams.

    ``base_logits`` and ``refiner_logits`` share shape ``[B, L, C]``.
    ``valid_mask`` is ``[B, L]`` boolean. Invalid positions return the
    base logits unchanged.
    """
    if base_logits.shape != refiner_logits.shape:
        raise ValueError(
            f"base_logits {tuple(base_logits.shape)} != refiner_logits "
            f"{tuple(refiner_logits.shape)}"
        )
    if valid_mask.shape != base_logits.shape[:-1]:
        raise ValueError(
            f"valid_mask {tuple(valid_mask.shape)} != base_logits[:-1] "
            f"{tuple(base_logits.shape[:-1])}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    refiner_log_probs = torch.log_softmax(refiner_logits, dim=-1)

    base_h = mapdiff_entropy_from_log_probs(base_log_probs)
    refiner_h = mapdiff_entropy_from_log_probs(refiner_log_probs)

    stacked_h = torch.stack([base_h, refiner_h], dim=0)  # [2, B, L]
    weights = torch.softmax(-stacked_h / temperature, dim=0)  # [2, B, L]

    stacked_logits = torch.stack([base_logits, refiner_logits], dim=0)  # [2, B, L, C]
    fused = (weights.unsqueeze(-1) * stacked_logits).sum(dim=0)  # [B, L, C]

    fused = torch.where(
        valid_mask.unsqueeze(-1).expand_as(fused),
        fused,
        base_logits,
    )
    return fused


def residual_fuse_logits(
    base_logits: torch.Tensor,
    refiner_logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Conservative residual fusion in logit space.

    Returns ``(1 - alpha) * base + alpha * refiner`` at valid positions
    and ``base_logits`` unchanged elsewhere. Logit-space convex
    combination is the geometric mean of softmaxes weighted by
    ``(1 - alpha)`` and ``alpha`` after re-normalization, which is more
    conservative than entropy fusion when the refiner is overconfident
    (entropy fusion would weight an overconfident refiner *higher*,
    residual fusion just attenuates it by ``alpha``).
    """
    if base_logits.shape != refiner_logits.shape:
        raise ValueError(
            f"base_logits {tuple(base_logits.shape)} != refiner_logits "
            f"{tuple(refiner_logits.shape)}"
        )
    if valid_mask.shape != base_logits.shape[:-1]:
        raise ValueError(
            f"valid_mask {tuple(valid_mask.shape)} != base_logits[:-1] "
            f"{tuple(base_logits.shape[:-1])}"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    mixed = (1.0 - float(alpha)) * base_logits + float(alpha) * refiner_logits
    return torch.where(
        valid_mask.unsqueeze(-1).expand_as(mixed),
        mixed,
        base_logits,
    )
