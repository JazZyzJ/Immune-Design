"""Training losses: InfoNCE primary + optional multi-positive and smoothness.

Implements PLAN.md Task E4:
  - Intra-protein InfoNCE with temperature tau (stable log-sum-exp)
  - Optional multi-positive softmax branch (lambda_mp, disabled in v0)
  - Optional smoothness regularizer (lambda_smooth, disabled in v0)
  - Decomposed loss dict for logging
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _stable_log_sum_exp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-sum-exp: log(sum(exp(x))) = max + log(sum(exp(x - max)))."""
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + torch.log(torch.exp(x - x_max).sum(dim=dim))


def info_nce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Intra-protein InfoNCE loss.

    For each positive, computes:
      L = -log(exp(z_pos/tau) / (exp(z_pos/tau) + sum(exp(z_neg/tau))))

    Uses log-sum-exp trick for numerical stability.

    Args:
        pos_logits: [P] logits for positive spans.
        neg_logits: [N] logits for negative spans.
        tau: temperature (default 0.1).

    Returns:
        Scalar loss averaged over positives.
    """

    # check tau is valid:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    
    if pos_logits.numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    P = pos_logits.shape[0]

    # Scale by temperature
    pos_scaled = pos_logits / tau  # [P]
    neg_scaled = neg_logits / tau  # [N]

    # For each positive: L_i = -pos_scaled[i] + log(exp(pos_scaled[i]) + sum(exp(neg_scaled)))
    # The denominator is shared across all positives (same neg set per chunk)
    # all_scaled = cat([pos_scaled_i, neg_scaled]) for each i
    # But since negatives are shared, we can optimize:
    #   log_denom_i = log(exp(pos_scaled[i]) + sum(exp(neg_scaled)))
    #              = log_sum_exp(cat([pos_scaled[i], neg_scaled]))

    losses = torch.zeros(P, device=pos_logits.device)
    for i in range(P):
        all_logits = torch.cat([pos_scaled[i:i+1], neg_scaled])  # [1 + N]
        log_denom = _stable_log_sum_exp(all_logits, dim=0)
        losses[i] = -pos_scaled[i] + log_denom

    return losses.mean()


def multi_positive_loss(
    pos_logits: torch.Tensor,
    T_mp: float = 0.1,
) -> torch.Tensor:
    """Multi-positive softmax loss (optional, disabled in v0).

    When multiple positives exist for a chunk, encourages uniform
    confidence across them via entropy-like regularization.

    L_mp = -mean_i(log(softmax(pos_logits / T_mp)[i]))

    Args:
        pos_logits: [P] logits for positive spans (P >= 2 for meaningful loss).
        T_mp: temperature for multi-positive softmax.

    Returns:
        Scalar loss (0 if P < 2).
    """
    if T_mp <= 0.0:
        raise ValueError("T_mp must be positive")
    if pos_logits.numel() < 2:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    log_probs = F.log_softmax(pos_logits / T_mp, dim=0)  # [P]
    return -log_probs.mean()


def smoothness_loss(
    per_residue_scores: torch.Tensor,
    chunk_len: int,
) -> torch.Tensor:
    """Smoothness regularizer on per-residue score profile (optional, disabled in v0).

    Penalizes large differences between adjacent residue scores.
    L_smooth = mean((s[i+1] - s[i])^2)

    Args:
        per_residue_scores: [L] score per residue in chunk.
        chunk_len: actual residue count.

    Returns:
        Scalar loss (0 if chunk_len < 2).
    """
    if chunk_len < 2:
        return torch.tensor(0.0, device=per_residue_scores.device, requires_grad=True)

    valid = per_residue_scores[:chunk_len]
    diffs = valid[1:] - valid[:-1]
    return (diffs ** 2).mean()


def margin_hard_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    margin_m: float = 0.5,
    hard_topk: int = 8,
) -> torch.Tensor:
    """Margin-based hard-example loss for AP-centric ranking.

    Selects top-k hardest negatives (highest logit) and penalizes
    when the gap between positive and hard negative is below margin.

    L_margin = mean(relu(m - (z_pos - z_neg_hard))) over [P, k] pairs.

    Args:
        pos_logits: [P] logits for positive spans.
        neg_logits: [N] logits for negative spans.
        margin_m: margin threshold (default 0.5).
        hard_topk: number of hardest negatives per positive (default 8).

    Returns:
        Scalar margin loss (0 if no positives or no negatives).
    """
    if hard_topk < 1:
        raise ValueError("hard_topk must be >= 1")

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    # Select top-k hardest negatives (fallback: use all if N < hard_topk)
    k = min(hard_topk, neg_logits.numel())
    hard_neg, _ = torch.topk(neg_logits, k)  # [k], highest logits

    # Pairwise margin: [P, k]
    gaps = pos_logits.unsqueeze(1) - hard_neg.unsqueeze(0)  # [P, k]
    losses = F.relu(margin_m - gaps)  # [P, k]

    return losses.mean()


def compute_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    tau: float = 0.1,
    T_mp: float = 0.1,
    lambda_mp: float = 0.0,
    lambda_smooth: float = 0.0,
    per_residue_scores: torch.Tensor | None = None,
    chunk_len: int = 0,
    objective_mode: str = "infonce",
    margin_m: float = 0.5,
    hard_topk: int = 8,
    lambda_margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute total loss with decomposed terms.

    Args:
        pos_logits: [P] positive span logits.
        neg_logits: [N] negative span logits.
        tau: InfoNCE temperature.
        T_mp: multi-positive temperature.
        lambda_mp: weight for multi-positive term (0 = disabled).
        lambda_smooth: weight for smoothness term (0 = disabled).
        per_residue_scores: [L] optional residue scores (needed if lambda_smooth > 0).
        chunk_len: actual residue count (needed if lambda_smooth > 0).
        objective_mode: "infonce" | "mixed_margin" | "margin_only".
        margin_m: margin threshold for hard-example loss.
        hard_topk: number of hardest negatives per positive.
        lambda_margin: weight for margin term in mixed_margin mode.

    Returns:
        Dict with keys:
          loss_total: scalar, the backprop target.
          loss_intra: scalar, InfoNCE term.
          loss_mp: scalar, multi-positive term (0 if disabled).
          loss_smooth: scalar, smoothness term (0 if disabled).
          loss_margin: scalar, margin hard-example term (0 if infonce mode).
    """
    device = pos_logits.device

    # InfoNCE (always computed unless margin_only)
    if objective_mode in ("infonce", "mixed_margin"):
        loss_intra = info_nce_loss(pos_logits, neg_logits, tau=tau)
    else:
        loss_intra = torch.tensor(0.0, device=device)

    # Margin hard-example loss
    if objective_mode in ("mixed_margin", "margin_only"):
        loss_margin = margin_hard_loss(pos_logits, neg_logits, margin_m=margin_m, hard_topk=hard_topk)
    else:
        loss_margin = torch.tensor(0.0, device=device)

    # Multi-positive (skip computation when disabled)
    if lambda_mp > 0.0:
        loss_mp = multi_positive_loss(pos_logits, T_mp=T_mp)
    else:
        loss_mp = torch.tensor(0.0, device=device)

    # Smoothness (skip computation when disabled)
    if lambda_smooth > 0.0 and per_residue_scores is not None:
        loss_smooth = smoothness_loss(per_residue_scores, chunk_len)
    elif lambda_smooth > 0.0 and per_residue_scores is None:
        raise ValueError("per_residue_scores is required when lambda_smooth > 0.0")
    else:
        loss_smooth = torch.tensor(0.0, device=device)

    # Total loss by mode
    if objective_mode == "infonce":
        loss_total = loss_intra + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "mixed_margin":
        loss_total = loss_intra + lambda_margin * loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "margin_only":
        loss_total = loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    return {
        "loss_total": loss_total,
        "loss_intra": loss_intra,
        "loss_mp": loss_mp,
        "loss_smooth": loss_smooth,
        "loss_margin": loss_margin,
    }
