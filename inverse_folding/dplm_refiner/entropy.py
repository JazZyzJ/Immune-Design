"""Entropy-based mask selection, sine schedule, and MC-dropout helper.

The entropy definition follows MapDiff exactly: ``-mean(p * log p)`` over
the AA axis (not the standard summed entropy). The sine mask-ratio
adapter follows ``MapDiff/utils.py:241-246``. ``enable_dropout_modules``
is a wider contract than ``MapDiff/utils.py::enable_dropout`` — it
returns the count of switched modules so tests can assert idempotency.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def mapdiff_entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute MapDiff entropy from log-probabilities.

    MapDiff source uses the mean (not sum) over the class axis:
    ``-mean(p * log p, dim=-1)``. Reference: ``MapDiff/utils.py:222-226``.

    Input  : ``[..., C]`` log-probs.
    Output : ``[...]`` entropy.
    """
    probs = log_probs.exp()
    return -(probs * log_probs).mean(dim=-1)


def sine_mask_ratio(
    beta_t_bar: torch.Tensor,
    *,
    center: float = 0.4,
    max_deviation: float = 0.2,
) -> torch.Tensor:
    """Sine-shaped mask-ratio schedule.

    ``ratio = center + sin(beta * pi/2) * max_deviation``. Returns a
    1-D tensor of length ``B``. Output is clamped to ``[0, 1]``.
    Reference: ``MapDiff/utils.py:241-246``.
    """
    if beta_t_bar.dim() == 2 and beta_t_bar.shape[-1] == 1:
        b = beta_t_bar.squeeze(-1)
    else:
        b = beta_t_bar
    b = b.to(torch.float32)
    ratios = center + torch.sin(b * math.pi / 2.0) * max_deviation
    return ratios.clamp(0.0, 1.0)


def select_entropy_mask(
    entropy: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_ratios: torch.Tensor,
) -> torch.Tensor:
    """Select the highest-entropy valid positions per batch row.

    Inputs
    ------
    entropy     : ``[B, L]``
    valid_mask  : ``[B, L]`` boolean
    mask_ratios : ``[B]`` in ``[0, 1]``

    Returns ``[B, L]`` boolean mask. The number selected per row is
    ``ceil(valid_count * ratio)``; ties broken by sort stability. Rows
    with zero valid positions or zero selected count return all ``False``.
    """
    if entropy.shape != valid_mask.shape:
        raise ValueError(
            f"entropy shape {tuple(entropy.shape)} != valid_mask "
            f"{tuple(valid_mask.shape)}"
        )
    if mask_ratios.dim() != 1 or mask_ratios.shape[0] != entropy.shape[0]:
        raise ValueError(
            f"mask_ratios must be [B={entropy.shape[0]}], got "
            f"{tuple(mask_ratios.shape)}"
        )

    B, L = entropy.shape
    out = torch.zeros_like(valid_mask, dtype=torch.bool)

    # Mask out invalid positions with -inf so they sort last.
    masked_entropy = torch.where(
        valid_mask, entropy, torch.full_like(entropy, -math.inf)
    )

    valid_counts = valid_mask.sum(dim=-1)  # [B]
    ratios = mask_ratios.clamp(0.0, 1.0).to(entropy.dtype)
    select_counts = torch.ceil(valid_counts.to(entropy.dtype) * ratios).long()
    select_counts = torch.minimum(select_counts, valid_counts)

    if L == 0:
        return out

    # Sort descending so that index 0 is the largest entropy.
    sorted_idx = torch.argsort(masked_entropy, dim=-1, descending=True)

    for b in range(B):
        k = int(select_counts[b].item())
        if k <= 0:
            continue
        chosen = sorted_idx[b, :k]
        out[b, chosen] = True
    # Defense-in-depth: never select an invalid position even if k > valid_count.
    return out & valid_mask


def enable_dropout_modules(model: nn.Module) -> int:
    """Switch only Dropout modules to training mode.

    Wider contract than ``MapDiff/utils.py::enable_dropout``: returns the
    count of modules that were flipped from ``eval`` to ``train``. Only
    modules whose class name starts with ``Dropout`` are touched.
    """
    count = 0
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train(True)
            count += 1
    return count
