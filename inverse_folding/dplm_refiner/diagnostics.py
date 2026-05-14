"""Diagnostic probes for ablation analysis.

``DPLMBaseEntropyProbe`` is a passthrough callable that matches the
``forward_decoder(logit_processor=...)`` contract. It records the AA-axis
entropy of the DPLM base logits at every decoder step but does NOT
modify them, so it can be installed on baseline / sidecar-only arms
to obtain entropy measurements that are directly comparable to the
``DPLMRefinerLogitProcessor`` measurements emitted on refiner / refiner+
sidecar arms.

Without this probe, baseline and sidecar-only arms have no diagnostics
sink and cross-arm comparisons collapse to fake-zero entropy, which is
how PLAN_IF_IMP indicators "sidecar lowers base entropy" and "combined
vs best single" used to be silently misleading.
"""

from __future__ import annotations

from typing import Any

import torch

from inverse_folding.dplm_refiner.entropy import mapdiff_entropy_from_log_probs
from inverse_folding.dplm_refiner.geometry import dplm_coords_to_ipa_positions
from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge


class DPLMBaseEntropyProbe:
    """Passthrough decoder hook that records base-logit entropy per step.

    Compatible with the ``logit_processor`` argument of
    ``DPLMInvFold.forward_decoder``. Returns the input ``logits``
    unchanged, so default DPLM behavior is preserved bit-for-bit.
    """

    def __init__(
        self,
        *,
        alphabet: Any,
        diagnostics_sink: Any,
    ) -> None:
        if diagnostics_sink is None:
            raise ValueError(
                "DPLMBaseEntropyProbe requires a diagnostics_sink; "
                "without it the probe has no purpose."
            )
        self.alphabet = alphabet
        self.diagnostics_sink = diagnostics_sink
        self.bridge = DPLMTokenBridge.from_alphabet(alphabet)
        structural: list[int] = []
        for name in ("padding_idx", "cls_idx", "eos_idx"):
            val = getattr(alphabet, name, None)
            if val is None:
                continue
            try:
                structural.append(int(val))
            except (TypeError, ValueError):
                continue
        self._structural_token_ids = torch.tensor(
            sorted(set(structural)), dtype=torch.long
        )

    def _structural_special_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        ids = self._structural_token_ids.to(tokens.device)
        if ids.numel() == 0:
            return torch.zeros_like(tokens, dtype=torch.bool)
        match = tokens.unsqueeze(-1) == ids.view(*([1] * tokens.dim()), -1)
        return match.any(dim=-1)

    def __call__(
        self,
        *,
        logits: torch.Tensor,
        output_tokens: torch.Tensor,
        batch: dict[str, Any],
        step: int,
        max_step: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            tokens = output_tokens.to(logits.device)
            coords = batch["coords"].to(logits.device)
            coord_mask = batch["coord_mask"].to(logits.device)

            structural_mask = self._structural_special_mask(tokens)
            ipa_pos = dplm_coords_to_ipa_positions(
                coords=coords,
                coord_mask=coord_mask,
                special_sym_mask=structural_mask,
            )
            seq_mask = ipa_pos.seq_mask  # [B, L] real-residue mask

            base_aa_logits = self.bridge.to_aa_logits(logits)
            base_log_probs = torch.log_softmax(base_aa_logits, dim=-1)
            entropy = mapdiff_entropy_from_log_probs(base_log_probs)  # [B, L]

            per_row = []
            B = int(seq_mask.shape[0])
            for b in range(B):
                row_mask = seq_mask[b]
                row_count = int(row_mask.sum().item())
                safe_count = max(row_count, 1)
                base_m = float(
                    (entropy[b] * row_mask.float()).sum().item() / safe_count
                )
                base_q90 = (
                    float(entropy[b, row_mask].quantile(0.9).item())
                    if row_count > 0
                    else float("nan")
                )
                per_row.append(
                    {
                        "row_idx": b,
                        "n_residues": row_count,
                        "n_selected": None,
                        "base_entropy_mean": base_m,
                        "fused_entropy_mean": None,
                        "refiner_entropy_mean": None,
                        "base_entropy_q90": base_q90,
                        "fused_entropy_q90": None,
                        "mask_ratio": None,
                        "probe_only": True,
                    }
                )
            self.diagnostics_sink(
                {
                    "step": int(step),
                    "max_step": int(max_step),
                    "per_row": per_row,
                }
            )
        return logits
