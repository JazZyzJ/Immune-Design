"""Canonical amino-acid ordering and DPLM token-id bridge.

The bridge maps between DPLM/ESM vocabulary indices and the 20 canonical
amino acids in a stable, alphabetic order. It is the single source of
truth for AA-axis layout used by the IPA refiner and logit fusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch


CANONICAL_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _safe_attr(alphabet: Any, name: str) -> int | None:
    value = getattr(alphabet, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DPLMTokenBridge:
    """Stable AA <-> DPLM-vocab bridge."""

    aa_token_ids: torch.LongTensor  # [20]
    banned_token_ids: torch.LongTensor  # [B_banned]

    @classmethod
    def from_alphabet(
        cls,
        alphabet: Any,
        *,
        extra_banned_token_ids: Sequence[int] = (),
    ) -> "DPLMTokenBridge":
        if not hasattr(alphabet, "get_idx"):
            raise ValueError("alphabet must expose get_idx(token)")

        aa_ids = [int(alphabet.get_idx(aa)) for aa in CANONICAL_AA_ORDER]
        if len(set(aa_ids)) != len(aa_ids):
            raise ValueError(
                f"alphabet maps multiple canonical AAs to the same id: {aa_ids}"
            )

        banned: list[int] = []
        for name in ("padding_idx", "cls_idx", "eos_idx", "mask_idx", "unk_idx"):
            val = _safe_attr(alphabet, name)
            if val is not None:
                banned.append(val)
        for val in extra_banned_token_ids:
            banned.append(int(val))
        # de-duplicate while preserving order for reproducibility
        seen: set[int] = set()
        deduped: list[int] = []
        for tid in banned:
            if tid in seen:
                continue
            seen.add(tid)
            deduped.append(tid)

        return cls(
            aa_token_ids=torch.tensor(aa_ids, dtype=torch.long),
            banned_token_ids=torch.tensor(deduped, dtype=torch.long),
        )

    # ── AA-logits / AA-tokens helpers ────────────────────────────────────────

    def to_aa_logits(self, dplm_logits: torch.Tensor) -> torch.Tensor:
        """Gather the 20 canonical-AA logits from a DPLM-vocab logits tensor.

        Input  : [..., V]
        Output : [..., 20]
        """
        if dplm_logits.shape[-1] < int(self.aa_token_ids.max().item()) + 1:
            raise ValueError(
                f"dplm_logits last dim {dplm_logits.shape[-1]} is smaller than "
                f"the largest AA token id {int(self.aa_token_ids.max().item())}"
            )
        idx = self.aa_token_ids.to(dplm_logits.device)
        return dplm_logits.index_select(-1, idx)

    def to_aa_tokens(self, dplm_tokens: torch.Tensor) -> torch.Tensor:
        """Map DPLM-vocab token ids to AA indices in [0, 20).

        Non-canonical token positions are mapped to -1.
        """
        device = dplm_tokens.device
        ids = self.aa_token_ids.to(device)
        out = torch.full_like(dplm_tokens, fill_value=-1, dtype=torch.long)
        for aa_idx, token_id in enumerate(ids.tolist()):
            out = torch.where(dplm_tokens == token_id, torch.full_like(out, aa_idx), out)
        return out

    def from_aa_tokens(self, aa_tokens: torch.Tensor) -> torch.Tensor:
        """Map AA indices in [0, 20) back to DPLM-vocab token ids.

        AA values outside [0, 20) are passed through unchanged so the caller
        can preserve sentinel positions.
        """
        device = aa_tokens.device
        ids = self.aa_token_ids.to(device)
        out = aa_tokens.clone()
        for aa_idx, token_id in enumerate(ids.tolist()):
            out = torch.where(aa_tokens == aa_idx, torch.full_like(out, token_id), out)
        return out

    def one_hot_from_dplm_tokens(self, dplm_tokens: torch.Tensor) -> torch.Tensor:
        """Convert DPLM-vocab token ids to a [..., 20] one-hot AA tensor.

        Non-canonical positions become all-zero rows.
        """
        aa_tokens = self.to_aa_tokens(dplm_tokens)
        valid = aa_tokens >= 0
        safe = aa_tokens.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(safe, num_classes=20).to(
            dplm_tokens.device
        )
        one_hot = one_hot.to(torch.float32)
        one_hot = one_hot * valid.unsqueeze(-1).to(one_hot.dtype)
        return one_hot

    # ── Logit scatter back to DPLM vocab ─────────────────────────────────────

    def scatter_aa_logits(
        self,
        aa_logits: torch.Tensor,
        template_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter [..., 20] AA logits back into a [..., V] DPLM-vocab tensor.

        Banned token columns are filled with -inf as a defense-in-depth
        measure. The template's other entries are preserved.
        """
        if aa_logits.shape[-1] != 20:
            raise ValueError(
                f"aa_logits last dim must be 20, got {aa_logits.shape[-1]}"
            )
        if aa_logits.shape[:-1] != template_logits.shape[:-1]:
            raise ValueError(
                f"aa_logits batch shape {aa_logits.shape[:-1]} != template "
                f"batch shape {template_logits.shape[:-1]}"
            )

        out = template_logits.clone()
        ids = self.aa_token_ids.to(out.device)
        # index_copy along last dim
        # out[..., ids] = aa_logits, shape-broadcast across leading dims
        idx_view = ids.view(*([1] * (out.dim() - 1)), 20).expand_as(aa_logits)
        out.scatter_(-1, idx_view, aa_logits)

        if self.banned_token_ids.numel() > 0:
            banned = self.banned_token_ids.to(out.device)
            banned_view = banned.view(*([1] * (out.dim() - 1)), -1).expand(
                *out.shape[:-1], banned.numel()
            )
            ninf = torch.full_like(banned_view, fill_value=-math.inf, dtype=out.dtype)
            out.scatter_(-1, banned_view, ninf)
        return out
