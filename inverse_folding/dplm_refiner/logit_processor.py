"""Decoder-time refiner logit processor.

Plugged into DPLM ``forward_decoder`` after special-token bans and
before sampling. The processor:

1. Builds a residue-validity mask from ``batch["coord_mask"]`` and the
   special-token mask derived from ``output_tokens``.
2. Builds IPA atom positions ``[B, L, 5, 3]`` with virtual CB.
3. Computes entropy from the base AA logits.
4. Selects high-entropy valid positions via ``sine_mask_ratio``.
5. Runs the IPA refiner ``mc_dropout_passes`` times (averaging logits).
6. Fuses base/refiner AA logits by entropy weighting.
7. Scatters the fused AA logits back into the DPLM-vocab tensor with
   special-token bans reapplied.
"""

from __future__ import annotations

from typing import Any

import torch

from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.entropy import (
    enable_dropout_modules,
    mapdiff_entropy_from_log_probs,
    select_entropy_mask,
    sine_mask_ratio,
)
from inverse_folding.dplm_refiner.fusion import fuse_logits_by_entropy
from inverse_folding.dplm_refiner.geometry import dplm_coords_to_ipa_positions
from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge


class DPLMRefinerLogitProcessor:
    """Callable plugged into DPLM ``forward_decoder``."""

    def __init__(
        self,
        *,
        refiner: torch.nn.Module,
        alphabet: Any,
        config: DPLMRefinerConfig,
        extra_banned_token_ids: tuple[int, ...] = (),
        diagnostics_sink: Any = None,
    ) -> None:
        config.validate()
        self.refiner = refiner
        self.alphabet = alphabet
        self.config = config
        self.bridge = DPLMTokenBridge.from_alphabet(
            alphabet, extra_banned_token_ids=extra_banned_token_ids
        )
        # Optional callable invoked once per __call__ with a dict of
        # ablation diagnostics. None disables diagnostics emission.
        # The sink runs INSIDE torch.no_grad scope and must be cheap.
        self.diagnostics_sink = diagnostics_sink
        # Structural special tokens (pad/BOS/EOS) are positions that are
        # never real residues. mask_idx / unk_idx are excluded because
        # mask positions ARE valid residues awaiting refinement.
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
        if not self.config.enabled:
            return logits
        if max_step <= 0:
            raise ValueError(f"max_step must be > 0; got {max_step}")
        if "coords" not in batch or "coord_mask" not in batch:
            raise ValueError(
                "batch must contain 'coords' and 'coord_mask' keys"
            )

        device = logits.device
        coords = batch["coords"].to(device)
        coord_mask = batch["coord_mask"].to(device)
        tokens = output_tokens.to(device)

        structural_mask = self._structural_special_mask(tokens)
        ipa_pos = dplm_coords_to_ipa_positions(
            coords=coords,
            coord_mask=coord_mask,
            special_sym_mask=structural_mask,
        )
        seq_mask = ipa_pos.seq_mask  # [B, L]
        atom_pos = ipa_pos.atom_pos  # [B, L, 5, 3]

        base_aa_logits = self.bridge.to_aa_logits(logits)  # [B, L, 20]
        base_log_probs = torch.log_softmax(base_aa_logits, dim=-1)
        entropy = mapdiff_entropy_from_log_probs(base_log_probs)  # [B, L]

        B = logits.shape[0]
        beta_t_bar = torch.full(
            (B,), float(step) / float(max_step), device=device
        )
        ratios = sine_mask_ratio(
            beta_t_bar,
            center=self.config.mask_ratio_center,
            max_deviation=self.config.mask_ratio_deviation,
        )
        selected = select_entropy_mask(
            entropy=entropy, valid_mask=seq_mask, mask_ratios=ratios
        )

        # Build IPA inputs.
        x_aa = self.bridge.one_hot_from_dplm_tokens(tokens)  # [B, L, 20]
        # Selected positions get the learned mask embedding via x_aa_mask=1
        # AND a zeroed AA one-hot so the AA projection cannot leak the
        # current-token identity into the refiner forward.
        x_aa = torch.where(
            selected.unsqueeze(-1),
            torch.zeros_like(x_aa),
            x_aa,
        )
        x_aa_mask = selected.long()

        # Run the refiner mc_dropout_passes times. If passes > 1, enable
        # only Dropout modules in train() while leaving the rest in eval().
        passes = max(1, int(self.config.mc_dropout_passes))
        if passes > 1:
            enable_dropout_modules(self.refiner)

        refiner_logits_sum: torch.Tensor | None = None
        for _ in range(passes):
            with torch.no_grad():
                ref_logits = self.refiner(
                    x_aa=x_aa,
                    x_pos=atom_pos,
                    x_aa_mask=x_aa_mask,
                    seq_mask=seq_mask,
                )
            refiner_logits_sum = (
                ref_logits if refiner_logits_sum is None
                else refiner_logits_sum + ref_logits
            )
        assert refiner_logits_sum is not None
        refiner_aa_logits = refiner_logits_sum / float(passes)

        # Refiner-vs-base entropy fusion only over selected & valid
        # positions; elsewhere keep base logits untouched.
        fusion_mask = selected & seq_mask
        fused_aa_logits = fuse_logits_by_entropy(
            base_aa_logits,
            refiner_aa_logits,
            valid_mask=fusion_mask,
            temperature=self.config.fusion_temperature,
        )

        # Scatter back to DPLM vocab; special tokens get re-banned to -inf.
        out = self.bridge.scatter_aa_logits(
            fused_aa_logits, template_logits=logits
        )

        # Ablation diagnostics: per-row records so batched generation
        # (B > 1) preserves per-design granularity. Each call emits one
        # sink record carrying a ``per_row`` list of length B; the runner
        # maps row_idx -> (protein_id, design_idx) using its batch
        # composition.
        if self.diagnostics_sink is not None:
            with torch.no_grad():
                fused_log_probs = torch.log_softmax(fused_aa_logits, dim=-1)
                fused_entropy = mapdiff_entropy_from_log_probs(fused_log_probs)
                refiner_log_probs = torch.log_softmax(refiner_aa_logits, dim=-1)
                refiner_entropy = mapdiff_entropy_from_log_probs(refiner_log_probs)
                per_row = []
                B = int(seq_mask.shape[0])
                for b in range(B):
                    row_mask = seq_mask[b]
                    row_count = int(row_mask.sum().item())
                    safe_count = max(row_count, 1)
                    n_sel = int(selected[b].sum().item())
                    base_m = float(
                        (entropy[b] * row_mask.float()).sum().item() / safe_count
                    )
                    fused_m = float(
                        (fused_entropy[b] * row_mask.float()).sum().item() / safe_count
                    )
                    refiner_m = float(
                        (refiner_entropy[b] * row_mask.float()).sum().item() / safe_count
                    )
                    if row_count > 0:
                        base_q90 = float(entropy[b, row_mask].quantile(0.9).item())
                        fused_q90 = float(fused_entropy[b, row_mask].quantile(0.9).item())
                    else:
                        base_q90 = float("nan")
                        fused_q90 = float("nan")
                    per_row.append(
                        {
                            "row_idx": b,
                            "n_residues": row_count,
                            "n_selected": n_sel,
                            "base_entropy_mean": base_m,
                            "fused_entropy_mean": fused_m,
                            "refiner_entropy_mean": refiner_m,
                            "base_entropy_q90": base_q90,
                            "fused_entropy_q90": fused_q90,
                            "mask_ratio": float(ratios[b].item()),
                            "probe_only": False,
                        }
                    )
                self.diagnostics_sink(
                    {
                        "step": int(step),
                        "max_step": int(max_step),
                        "per_row": per_row,
                    }
                )

        return out
