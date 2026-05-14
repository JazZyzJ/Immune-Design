"""Rich diagnostic probe for 4-arm IF improvement ablation.

Single processor class that:

- Replicates the refiner / fusion / scatter path when a refiner is
  supplied (so refiner-bearing arms remain bit-equivalent to the
  production ``DPLMRefinerLogitProcessor`` path when the diagnostic
  sinks are silent).
- Acts as a base-only passthrough when no refiner is supplied (so
  baseline and sidecar-only arms still emit comparable per-step
  diagnostics).
- Emits per-(step, row) records that are richer than the production
  sink: ground-truth-aware top-1 recovery / log-likelihood at all
  residue positions AND at selected positions, base-vs-refiner top-1
  agreement, fusion flip rate, and the previous step's reparam-
  preservation rate at the positions the refiner had selected.
- Optionally emits per-(step, row, position) records at selected
  positions for deeper position-level analysis.

This is a DIAGNOSTIC tool. It is NOT used in production runs and not
wired into ``run_if_imp_refiner.py``; the production processor is the
canonical inference path.

PLAN_IF_IMP diagnostic addendum (see conversation log 2026-05-14).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from inverse_folding.dplm_refiner.config import (
    DPLMRefinerConfig,
    should_apply_refiner,
)
from inverse_folding.dplm_refiner.entropy import (
    enable_dropout_modules,
    mapdiff_entropy_from_log_probs,
    select_entropy_mask,
    sine_mask_ratio,
)
from inverse_folding.dplm_refiner.fusion import (
    fuse_logits_by_entropy,
    residual_fuse_logits,
)
from inverse_folding.dplm_refiner.geometry import dplm_coords_to_ipa_positions
from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge


PerStepSink = Callable[[dict[str, Any]], None]
PerPositionSink = Callable[[dict[str, Any]], None]


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class DPLMArmDiagnosticProcessor:
    """4-arm-aware decoder-time probe with rich GT-aware diagnostics.

    Construction:

    - ``refiner=None`` → baseline/sidecar mode: returns input logits
      untouched, only records base-side diagnostics.
    - ``refiner=<DPLMIPARefiner>`` + ``config=DPLMRefinerConfig`` →
      refiner/refiner+sidecar mode: runs the same selection /
      MC-dropout / fusion / scatter pipeline as the production
      processor, AND records refiner-side diagnostics.

    The probe is stateful within one ``generate()`` call: it tracks the
    previous step's selected positions and chosen fused tokens so the
    next step can measure how many of those edits survived DPLM's
    ``_reparam_decoding`` re-mask. Callers must invoke ``reset_state()``
    before each new ``generate()`` invocation.

    Sinks
    -----
    Both ``per_step_sink`` and ``per_position_sink`` receive one record
    per call. They receive **plain dicts of native Python scalars** so
    downstream parquet writes never see GPU tensors.

    Per-step record schema (always emitted, refiner-only fields are
    ``None`` on baseline/sidecar arms)::

        {
            "step": int, "max_step": int,
            "row_idx": int, "n_residues": int,
            "n_selected": int | None,
            "mask_ratio": float | None,

            # base path, all positions
            "base_entropy_mean": float,
            "base_entropy_q90": float,
            "base_top1_recovery_all": float,
            "base_log_prob_gt_mean": float,

            # refiner-only path, only at selected positions
            "base_top1_recovery_at_selected": float | None,
            "refiner_top1_recovery_at_selected": float | None,
            "fused_top1_recovery_at_selected": float | None,
            "base_vs_refiner_top1_agree_at_selected": float | None,
            "fusion_flip_rate_at_selected": float | None,
            "base_log_prob_gt_at_selected": float | None,
            "refiner_log_prob_gt_at_selected": float | None,
            "fused_log_prob_gt_at_selected": float | None,

            # next-step view of last step's selected positions
            "prev_n_selected": int | None,
            "prev_n_preserved": int | None,
            "prev_preservation_rate": float | None,

            # init-state recovery (only meaningful at step=1)
            "init_state_top1_recovery": float | None,
        }

    Per-position record schema (only when ``per_position_sink`` is set
    AND refiner is set; one record per selected position per step)::

        {
            "step": int, "row_idx": int, "position": int,
            "gt_aa_idx": int,
            "base_top1_aa_idx": int,
            "refiner_top1_aa_idx": int,
            "fused_top1_aa_idx": int,
            "base_log_prob_gt": float,
            "refiner_log_prob_gt": float,
            "fused_log_prob_gt": float,
            "base_entropy": float,
        }
    """

    def __init__(
        self,
        *,
        alphabet: Any,
        refiner: torch.nn.Module | None = None,
        config: DPLMRefinerConfig | None = None,
        per_step_sink: PerStepSink | None = None,
        per_position_sink: PerPositionSink | None = None,
    ) -> None:
        if refiner is not None and config is None:
            raise ValueError(
                "DPLMArmDiagnosticProcessor: ``config`` is required when "
                "``refiner`` is provided"
            )
        if refiner is None and per_position_sink is not None:
            raise ValueError(
                "per_position_sink only emits records when a refiner is "
                "supplied; pass refiner=<DPLMIPARefiner> or drop the sink"
            )
        if config is not None:
            config.validate()
        self.alphabet = alphabet
        self.refiner = refiner
        self.config = config
        self.per_step_sink = per_step_sink
        self.per_position_sink = per_position_sink
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
        self.reset_state()

    # ── State ─────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Drop cross-step memory. Call once before each generate() call."""
        # row_idx -> dict("positions": LongTensor[K], "fused_tokens": LongTensor[K])
        # where positions/fused_tokens are stored on CPU as int64 so the
        # next-step preservation check is GPU-free.
        self._prev_selected_state: dict[int, dict[str, torch.Tensor]] = {}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _structural_special_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        ids = self._structural_token_ids.to(tokens.device)
        if ids.numel() == 0:
            return torch.zeros_like(tokens, dtype=torch.bool)
        match = tokens.unsqueeze(-1) == ids.view(*([1] * tokens.dim()), -1)
        return match.any(dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        denom = int(mask.sum().item())
        if denom <= 0:
            return float("nan")
        return float((values * mask.float()).sum().item() / denom)

    @staticmethod
    def _masked_quantile(
        values: torch.Tensor, mask: torch.Tensor, q: float
    ) -> float:
        if int(mask.sum().item()) <= 0:
            return float("nan")
        return float(values[mask].quantile(q).item())

    # ── Main hook ─────────────────────────────────────────────────────────

    def __call__(
        self,
        *,
        logits: torch.Tensor,
        output_tokens: torch.Tensor,
        batch: dict[str, Any],
        step: int,
        max_step: int,
    ) -> torch.Tensor:
        if max_step <= 0:
            raise ValueError(f"max_step must be > 0; got {max_step}")
        if "coords" not in batch or "coord_mask" not in batch:
            raise ValueError(
                "batch must contain 'coords' and 'coord_mask' for the probe"
            )
        if "tokens" not in batch:
            raise ValueError(
                "diagnostic probe requires batch['tokens'] (native sequence) "
                "for ground-truth-aware metrics; ensure the featurizer was "
                "called with sequences populated"
            )

        device = logits.device
        coords = batch["coords"].to(device)
        coord_mask = batch["coord_mask"].to(device)
        native_tokens = batch["tokens"].to(device)
        tokens = output_tokens.to(device)
        B = int(logits.shape[0])

        structural_mask = self._structural_special_mask(tokens)
        ipa_pos = dplm_coords_to_ipa_positions(
            coords=coords,
            coord_mask=coord_mask,
            special_sym_mask=structural_mask,
        )
        seq_mask = ipa_pos.seq_mask  # [B, L]

        with torch.no_grad():
            base_aa_logits = self.bridge.to_aa_logits(logits)  # [B, L, 20]
            base_log_probs = torch.log_softmax(base_aa_logits, dim=-1)
            base_entropy = mapdiff_entropy_from_log_probs(base_log_probs)
            base_top1_aa = base_aa_logits.argmax(dim=-1)  # [B, L] in [0, 20)

            gt_aa = self.bridge.to_aa_tokens(native_tokens)  # [B, L] in [-1, 20)
            gt_aa_safe = gt_aa.clamp(min=0)
            base_log_prob_gt = torch.gather(
                base_log_probs, dim=-1, index=gt_aa_safe.unsqueeze(-1)
            ).squeeze(-1)
            base_top1_correct = (base_top1_aa == gt_aa) & seq_mask

            # Init-state recovery: only at step=1 the probe sees the
            # initial output_tokens BEFORE any sampling step has run.
            init_state_top1_recovery_per_row: list[float | None] = []
            if int(step) == 1:
                init_aa = self.bridge.to_aa_tokens(tokens)
                init_correct = (init_aa == gt_aa) & seq_mask
                for b in range(B):
                    init_state_top1_recovery_per_row.append(
                        self._masked_mean(
                            init_correct[b].float(), seq_mask[b]
                        )
                    )
            else:
                init_state_top1_recovery_per_row = [None] * B

        # Refiner branch (running with refiner). Honors inference-time
        # gating: if the current step is gated off by config.apply_steps,
        # the probe takes the baseline path so the per-step record at
        # that step is base-only -- the refiner does not run.
        refiner_present = self.refiner is not None
        if refiner_present and self.config is not None:
            refiner_gate_on = should_apply_refiner(
                int(step), int(max_step), self.config.apply_steps
            )
        else:
            refiner_gate_on = False
        refiner_active = refiner_present and refiner_gate_on

        selected = None
        refiner_aa_logits = None
        fused_aa_logits = None
        fused_top1_aa = None
        refiner_top1_aa = None
        ratios = None
        out_logits = logits
        if refiner_active:
            atom_pos = ipa_pos.atom_pos
            beta_t_bar = torch.full(
                (B,), float(step) / float(max_step), device=device
            )
            ratios = sine_mask_ratio(
                beta_t_bar,
                center=self.config.mask_ratio_center,
                max_deviation=self.config.mask_ratio_deviation,
            )
            selected = select_entropy_mask(
                entropy=base_entropy,
                valid_mask=seq_mask,
                mask_ratios=ratios,
            )

            x_aa = self.bridge.one_hot_from_dplm_tokens(tokens)
            x_aa = torch.where(
                selected.unsqueeze(-1),
                torch.zeros_like(x_aa),
                x_aa,
            )
            x_aa_mask = selected.long()

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
                    ref_logits
                    if refiner_logits_sum is None
                    else refiner_logits_sum + ref_logits
                )
            assert refiner_logits_sum is not None
            refiner_aa_logits = refiner_logits_sum / float(passes)

            fusion_mask = selected & seq_mask
            if self.config.fusion_mode == "residual":
                fused_aa_logits = residual_fuse_logits(
                    base_aa_logits,
                    refiner_aa_logits,
                    valid_mask=fusion_mask,
                    alpha=self.config.fusion_alpha,
                )
            else:
                fused_aa_logits = fuse_logits_by_entropy(
                    base_aa_logits,
                    refiner_aa_logits,
                    valid_mask=fusion_mask,
                    temperature=self.config.fusion_temperature,
                )
            with torch.no_grad():
                refiner_top1_aa = refiner_aa_logits.argmax(dim=-1)
                fused_top1_aa = fused_aa_logits.argmax(dim=-1)

            out_logits = self.bridge.scatter_aa_logits(
                fused_aa_logits, template_logits=logits
            )

        # Reparam preservation check: compare current ``output_tokens``
        # at the positions we selected on the PREVIOUS step against the
        # DPLM-vocab tokens we had argmax'd into them last step. A
        # position is "preserved" when output_tokens still equals the
        # last step's chosen token; otherwise it was re-masked by
        # ``_reparam_decoding`` (mask_id == structural after reparam).
        prev_preservation_per_row: dict[int, tuple[int, int]] = {}
        if self._prev_selected_state:
            tokens_cpu = tokens.detach().cpu()
            for b, prev in self._prev_selected_state.items():
                positions = prev["positions"]
                prev_tokens_aa = prev["fused_tokens"]
                if positions.numel() == 0:
                    prev_preservation_per_row[b] = (0, 0)
                    continue
                cur_at_positions = tokens_cpu[b, positions]
                # Convert current DPLM tokens to AA indices via bridge
                # (cheap CPU loop using bridge maps).
                cur_aa = self.bridge.to_aa_tokens(cur_at_positions)
                n_preserved = int(((cur_aa == prev_tokens_aa)).sum().item())
                prev_preservation_per_row[b] = (
                    int(positions.numel()),
                    n_preserved,
                )

        # Emit per-step records (always) and per-position records
        # (refiner-only).
        if self.per_step_sink is not None:
            for b in range(B):
                row_mask = seq_mask[b]
                n_residues = int(row_mask.sum().item())
                rec: dict[str, Any] = {
                    "step": int(step),
                    "max_step": int(max_step),
                    "row_idx": int(b),
                    "n_residues": n_residues,
                    "n_selected": None,
                    "mask_ratio": None,
                    "base_entropy_mean": self._masked_mean(
                        base_entropy[b], row_mask
                    ),
                    "base_entropy_q90": self._masked_quantile(
                        base_entropy[b], row_mask, 0.9
                    ),
                    "base_top1_recovery_all": self._masked_mean(
                        base_top1_correct[b].float(), row_mask
                    ),
                    "base_log_prob_gt_mean": self._masked_mean(
                        base_log_prob_gt[b], row_mask
                    ),
                    "base_top1_recovery_at_selected": None,
                    "refiner_top1_recovery_at_selected": None,
                    "fused_top1_recovery_at_selected": None,
                    "base_vs_refiner_top1_agree_at_selected": None,
                    "fusion_flip_rate_at_selected": None,
                    "base_log_prob_gt_at_selected": None,
                    "refiner_log_prob_gt_at_selected": None,
                    "fused_log_prob_gt_at_selected": None,
                    "prev_n_selected": None,
                    "prev_n_preserved": None,
                    "prev_preservation_rate": None,
                    "init_state_top1_recovery": (
                        init_state_top1_recovery_per_row[b]
                    ),
                }
                if refiner_active:
                    sel_row = selected[b] & row_mask
                    n_sel = int(sel_row.sum().item())
                    rec["n_selected"] = n_sel
                    rec["mask_ratio"] = float(ratios[b].item())
                    if n_sel > 0:
                        sel_idx = torch.nonzero(sel_row, as_tuple=False).flatten()
                        base_top1_sel = base_top1_aa[b, sel_idx]
                        refiner_top1_sel = refiner_top1_aa[b, sel_idx]
                        fused_top1_sel = fused_top1_aa[b, sel_idx]
                        gt_sel = gt_aa[b, sel_idx]
                        base_correct_sel = (base_top1_sel == gt_sel).float()
                        refiner_correct_sel = (refiner_top1_sel == gt_sel).float()
                        fused_correct_sel = (fused_top1_sel == gt_sel).float()
                        agree_sel = (
                            base_top1_sel == refiner_top1_sel
                        ).float()
                        flip_sel = (
                            base_top1_sel != fused_top1_sel
                        ).float()
                        rec["base_top1_recovery_at_selected"] = float(
                            base_correct_sel.mean().item()
                        )
                        rec["refiner_top1_recovery_at_selected"] = float(
                            refiner_correct_sel.mean().item()
                        )
                        rec["fused_top1_recovery_at_selected"] = float(
                            fused_correct_sel.mean().item()
                        )
                        rec["base_vs_refiner_top1_agree_at_selected"] = float(
                            agree_sel.mean().item()
                        )
                        rec["fusion_flip_rate_at_selected"] = float(
                            flip_sel.mean().item()
                        )
                        # Log-prob[GT] only well-defined when GT is canonical.
                        gt_valid = (gt_sel >= 0).float()
                        rec["base_log_prob_gt_at_selected"] = (
                            float(
                                (base_log_prob_gt[b, sel_idx] * gt_valid).sum().item()
                                / max(gt_valid.sum().item(), 1.0)
                            )
                        )
                        gt_safe_sel = gt_sel.clamp(min=0)
                        refiner_log_probs = torch.log_softmax(
                            refiner_aa_logits[b, sel_idx], dim=-1
                        )
                        fused_log_probs = torch.log_softmax(
                            fused_aa_logits[b, sel_idx], dim=-1
                        )
                        refiner_lp_gt = refiner_log_probs.gather(
                            dim=-1, index=gt_safe_sel.unsqueeze(-1)
                        ).squeeze(-1)
                        fused_lp_gt = fused_log_probs.gather(
                            dim=-1, index=gt_safe_sel.unsqueeze(-1)
                        ).squeeze(-1)
                        rec["refiner_log_prob_gt_at_selected"] = float(
                            (refiner_lp_gt * gt_valid).sum().item()
                            / max(gt_valid.sum().item(), 1.0)
                        )
                        rec["fused_log_prob_gt_at_selected"] = float(
                            (fused_lp_gt * gt_valid).sum().item()
                            / max(gt_valid.sum().item(), 1.0)
                        )

                        # Per-position records at selected positions only.
                        if self.per_position_sink is not None:
                            base_entropy_sel = base_entropy[b, sel_idx]
                            for pi in range(int(n_sel)):
                                pos = int(sel_idx[pi].item())
                                self.per_position_sink(
                                    {
                                        "step": int(step),
                                        "row_idx": int(b),
                                        "position": pos,
                                        "gt_aa_idx": int(gt_sel[pi].item()),
                                        "base_top1_aa_idx": int(base_top1_sel[pi].item()),
                                        "refiner_top1_aa_idx": int(refiner_top1_sel[pi].item()),
                                        "fused_top1_aa_idx": int(fused_top1_sel[pi].item()),
                                        "base_log_prob_gt": float(
                                            base_log_prob_gt[b, sel_idx[pi]].item()
                                        ),
                                        "refiner_log_prob_gt": float(
                                            refiner_lp_gt[pi].item()
                                        ),
                                        "fused_log_prob_gt": float(
                                            fused_lp_gt[pi].item()
                                        ),
                                        "base_entropy": float(
                                            base_entropy_sel[pi].item()
                                        ),
                                    }
                                )

                # Previous-step preservation, if recorded
                if b in prev_preservation_per_row:
                    n_prev_sel, n_preserved = prev_preservation_per_row[b]
                    rec["prev_n_selected"] = int(n_prev_sel)
                    rec["prev_n_preserved"] = int(n_preserved)
                    rec["prev_preservation_rate"] = (
                        float(n_preserved) / float(n_prev_sel)
                        if n_prev_sel > 0
                        else float("nan")
                    )

                self.per_step_sink(rec)

        # Update prev-state for next step's preservation check.
        # We store AA indices (not DPLM vocab) so the next step's
        # comparison can simply re-run bridge.to_aa_tokens(output_tokens)
        # and compare integers.
        if refiner_active and selected is not None and fused_top1_aa is not None:
            new_state: dict[int, dict[str, torch.Tensor]] = {}
            for b in range(B):
                sel_row = selected[b] & seq_mask[b]
                if int(sel_row.sum().item()) == 0:
                    new_state[b] = {
                        "positions": torch.zeros((0,), dtype=torch.long),
                        "fused_tokens": torch.zeros((0,), dtype=torch.long),
                    }
                    continue
                sel_idx = torch.nonzero(sel_row, as_tuple=False).flatten().cpu()
                fused_tokens_at_sel = fused_top1_aa[b, sel_idx].detach().cpu()
                new_state[b] = {
                    "positions": sel_idx,
                    "fused_tokens": fused_tokens_at_sel,
                }
            self._prev_selected_state = new_state
        else:
            self._prev_selected_state = {}

        return out_logits
