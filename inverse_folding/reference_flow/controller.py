"""Phase D1 monitor-only controller.

Implements the §4.7 D1 reliability-gated online refresh + active-block surface
without changing C1 logits, schedule, or remask ranking. The controller is
stateful per ``(protein_id, design_idx)``; the C1 driver instantiates a fresh
one per design.

Public surface:
- ``SamplerStepContext`` — what the sampler passes into the hook each step.
- ``ControllerStepResult`` — what the hook returns (D1 always returns
  identity logits).
- ``ActiveBlock`` — one merged scoring-window block with its reliability
  gate factors.
- ``D1RefreshRecord`` — telemetry payload for ``refresh_log.jsonl``.
- ``D1MonitorController`` — the hook implementation.
- ``WindowMismatchError`` — raised when dynamic and static window enumerations
  do not align by ``(start_0b, end_0b, k)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .controller_config import ControllerConfig
from .head_scoring import HeadScore, OnlineHeadScorer, WindowRiskRecord


class WindowMismatchError(RuntimeError):
    """Raised when dynamic vs. static window enumeration does not align."""


@dataclass(frozen=True)
class SamplerStepContext:
    """Input passed to the controller hook on each sampler step."""

    x_t: torch.Tensor
    logits: torch.Tensor
    scores: np.ndarray
    step: int
    t: float
    mask_token_id: int
    protein_id: str
    design_idx: int
    sequence_length: int


@dataclass(frozen=True)
class ActiveBlock:
    """One merged scoring-window block."""

    block_id: int
    residue_start_0b: int
    residue_end_0b: int
    window_indices: tuple[int, ...]
    g_time: float
    g_comp: float
    g_ent: float
    g_ESS: float
    rho_B: float
    completion_fraction: float
    mean_struct_entropy: float


@dataclass(frozen=True)
class D1RefreshRecord:
    """Per-refresh telemetry payload."""

    protein_id: str
    design_idx: int
    refresh_step: int
    step: int
    t: float
    r_windows_dyn: tuple[WindowRiskRecord, ...]
    r_windows_static_count: int
    window_excess: tuple[float, ...]
    active_blocks: tuple[ActiveBlock, ...]
    new_hotspot_count: int
    completion_fraction_global: float
    mean_struct_entropy_global: float
    head_risk_LME: float
    head_risk_max: float


@dataclass(frozen=True)
class ControllerStepResult:
    """Hook output. ``refresh_record`` is None on non-refresh steps."""

    logits: torch.Tensor
    refresh_record: D1RefreshRecord | None


class D1MonitorController:
    """Reliability-gated online refresh + active-block discovery (monitor only).

    The hook is invoked after structural logit validation and before unmask
    sampling. On non-refresh steps it short-circuits and returns identity
    logits. At a refresh step it constructs a hard completion, scores it
    through the bound :class:`OnlineHeadScorer`, looks up the static WT window
    cache, computes per-window excess risk, merges active windows into
    residue-level blocks, and records reliability gate factors. D1 never
    modifies logits or the sampler's remask ranking.
    """

    def __init__(
        self,
        *,
        protein_id: str,
        design_idx: int,
        static_sequence: str,
        scorer: OnlineHeadScorer,
        config: ControllerConfig,
        decode_tokens: Callable[[torch.Tensor], str],
    ) -> None:
        self.protein_id = str(protein_id)
        self.design_idx = int(design_idx)
        self.static_sequence = str(static_sequence)
        self.scorer = scorer
        self.config = config
        self._decode_tokens = decode_tokens

        self._refresh_records: list[D1RefreshRecord] = []
        self._event_rows: list[dict] = []
        self._refresh_step_counter = 0
        self._static_head_score: HeadScore | None = None

    # ---------- public accessors for telemetry flush ----------

    def refresh_records(self) -> list[D1RefreshRecord]:
        return list(self._refresh_records)

    def controller_event_rows(self) -> list[dict]:
        return list(self._event_rows)

    # ---------- hook ----------

    def step(self, context: SamplerStepContext) -> ControllerStepResult:
        if not self.config.enabled:
            return ControllerStepResult(logits=context.logits, refresh_record=None)
        if context.t < self.config.t_start:
            return ControllerStepResult(logits=context.logits, refresh_record=None)
        if context.step % self.config.refresh_interval != 0:
            return ControllerStepResult(logits=context.logits, refresh_record=None)

        completed_tokens = self._build_hard_completion(context)
        completed_sequence = self._decode_tokens(completed_tokens)

        dyn_batch = self.scorer.score_batch_same_protein(
            protein_id=self.protein_id,
            records=[("dyn_argmax", completed_sequence)],
        )
        dyn_score = dyn_batch.scores[0]

        if self._static_head_score is None:
            self._static_head_score = self.scorer.get_or_compute_static(
                self.protein_id, self.static_sequence
            )
        static_score = self._static_head_score

        if len(dyn_score.windows) != len(static_score.windows):
            raise WindowMismatchError(
                f"dynamic windows ({len(dyn_score.windows)}) != "
                f"static windows ({len(static_score.windows)}) for "
                f"protein_id={self.protein_id!r}"
            )
        for d, s in zip(dyn_score.windows, static_score.windows):
            if (d.start_0b, d.end_0b, d.k) != (s.start_0b, s.end_0b, s.k):
                raise WindowMismatchError(
                    f"window coordinate mismatch dyn=({d.start_0b},{d.end_0b},{d.k}) "
                    f"vs static=({s.start_0b},{s.end_0b},{s.k}) for protein_id={self.protein_id!r}"
                )

        excess_threshold = self.config.active_windows.excess_threshold
        window_excess = tuple(
            max(0.0, float(d.z) - float(s.z) - excess_threshold)
            for d, s in zip(dyn_score.windows, static_score.windows)
        )

        # Threshold first, then top-N by excess.
        active_indices = [i for i, e in enumerate(window_excess) if e > 0.0]
        max_windows = self.config.active_windows.max_windows
        if len(active_indices) > max_windows:
            active_indices = sorted(
                active_indices, key=lambda i: window_excess[i], reverse=True
            )[:max_windows]

        # Window-id-indexed (start, end) spans.
        active_window_spans = [
            (dyn_score.windows[i].start_0b, dyn_score.windows[i].end_0b, i)
            for i in active_indices
        ]
        merged_blocks = _merge_overlapping_spans(active_window_spans)

        per_pos_entropy = _per_position_entropy(context.logits)

        active_blocks: list[ActiveBlock] = []
        for block_id, (start_0b, end_0b, window_indices) in enumerate(merged_blocks):
            comp_frac = _completion_fraction(context, start_0b, end_0b)
            mean_entropy = _mean_block_entropy(per_pos_entropy, start_0b, end_0b)
            g_time = _g_time(context.t, self.config.t_start, self.config.reliability.time_k)
            g_comp = _g_comp(
                comp_frac, self.config.reliability.min_completion_fraction
            )
            g_ent = _g_ent(mean_entropy, self.config.reliability.entropy_h0)
            g_ESS = 1.0  # D1: no candidate reweighting yet.
            rho_B = float(_clip_unit(g_time * g_comp * g_ent * g_ESS))
            active_blocks.append(
                ActiveBlock(
                    block_id=block_id,
                    residue_start_0b=int(start_0b),
                    residue_end_0b=int(end_0b),
                    window_indices=tuple(sorted(int(i) for i in window_indices)),
                    g_time=float(_clip_unit(g_time)),
                    g_comp=float(_clip_unit(g_comp)),
                    g_ent=float(_clip_unit(g_ent)),
                    g_ESS=float(_clip_unit(g_ESS)),
                    rho_B=rho_B,
                    completion_fraction=float(comp_frac),
                    mean_struct_entropy=float(mean_entropy),
                )
            )

        # D1 step 13: new hotspot count uses the same threshold as active windows.
        new_hotspot_threshold = self.config.active_windows.excess_threshold
        new_hotspot_count = sum(
            1
            for i in active_indices
            if float(static_score.windows[i].z) < new_hotspot_threshold
        )

        completion_fraction_global = _completion_fraction(
            context, 0, context.sequence_length
        )
        mean_struct_entropy_global = _mean_block_entropy(
            per_pos_entropy, 0, context.sequence_length
        )
        head_risk_LME = _lme(tuple(float(w.z) for w in dyn_score.windows))
        head_risk_max = max(
            (float(w.z) for w in dyn_score.windows), default=float("-inf")
        )

        record = D1RefreshRecord(
            protein_id=self.protein_id,
            design_idx=self.design_idx,
            refresh_step=self._refresh_step_counter,
            step=int(context.step),
            t=float(context.t),
            r_windows_dyn=tuple(dyn_score.windows),
            r_windows_static_count=len(static_score.windows),
            window_excess=window_excess,
            active_blocks=tuple(active_blocks),
            new_hotspot_count=int(new_hotspot_count),
            completion_fraction_global=float(completion_fraction_global),
            mean_struct_entropy_global=float(mean_struct_entropy_global),
            head_risk_LME=float(head_risk_LME),
            head_risk_max=float(head_risk_max),
        )
        self._refresh_records.append(record)

        min_rho = self.config.reliability.min_rho_to_emit_event
        for block in active_blocks:
            if block.rho_B >= min_rho:
                self._event_rows.append(
                    {
                        "protein_id": self.protein_id,
                        "design_idx": self.design_idx,
                        "refresh_step": self._refresh_step_counter,
                        "step": int(context.step),
                        "t": float(context.t),
                        "event_type": "monitor",
                        "block_id": int(block.block_id),
                        "residue_start_0b": int(block.residue_start_0b),
                        "residue_end_0b": int(block.residue_end_0b),
                        "rho_B": float(block.rho_B),
                        "g_time": float(block.g_time),
                        "g_comp": float(block.g_comp),
                        "g_ent": float(block.g_ent),
                        "g_ESS": float(block.g_ESS),
                        "reason": "monitor_only",
                    }
                )

        self._refresh_step_counter += 1
        return ControllerStepResult(logits=context.logits, refresh_record=record)

    # ---------- internals ----------

    def _build_hard_completion(self, context: SamplerStepContext) -> torch.Tensor:
        x_t = context.x_t.detach().clone()
        mask = x_t == context.mask_token_id
        if mask.any():
            argmax_tokens = context.logits.argmax(dim=-1)
            x_t[mask] = argmax_tokens[mask].to(dtype=x_t.dtype)
        return x_t


# ---------- helpers ----------


def _merge_overlapping_spans(
    spans: list[tuple[int, int, int]],
) -> list[tuple[int, int, list[int]]]:
    """Merge half-open ``[start, end)`` spans into connected components.

    Each input tuple is ``(start, end, window_index)``. Output is one tuple
    per merged group: ``(start, end, [window_indices])`` where the residue
    span is the union of all spans in the group (which is contiguous on the
    integer line under transitive overlap).
    """
    if not spans:
        return []
    # Sort by start; merge greedily — transitive overlap on a line is captured
    # because once a group extends past a window's start, subsequent windows
    # whose start lies inside the extended range are merged in turn.
    sorted_spans = sorted(spans, key=lambda s: (int(s[0]), int(s[1])))
    groups: list[tuple[int, int, list[int]]] = []
    cur_start, cur_end, cur_indices = sorted_spans[0]
    cur_indices_list = [cur_indices]
    for start, end, idx in sorted_spans[1:]:
        if start < cur_end:
            cur_end = max(cur_end, end)
            cur_indices_list.append(idx)
        else:
            groups.append((cur_start, cur_end, cur_indices_list))
            cur_start, cur_end, cur_indices_list = start, end, [idx]
    groups.append((cur_start, cur_end, cur_indices_list))
    return groups


def _per_position_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return torch.nan_to_num(entropy, nan=0.0)


def _completion_fraction(context: SamplerStepContext, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    block = context.x_t[start:end]
    committed = (block != context.mask_token_id).sum().item()
    return float(committed) / float(end - start)


def _mean_block_entropy(per_pos_entropy: torch.Tensor, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    block_entropy = per_pos_entropy[start:end]
    return float(block_entropy.mean().item())


def _g_time(t: float, t_start: float, time_k: float) -> float:
    return 1.0 / (1.0 + math.exp(-time_k * (t - t_start)))


def _g_comp(comp_frac: float, min_comp_frac: float) -> float:
    if comp_frac < min_comp_frac:
        return 0.0
    return float(comp_frac)


def _g_ent(mean_entropy: float, entropy_h0: float) -> float:
    if entropy_h0 <= 0.0:
        return 1.0
    return float(math.exp(-mean_entropy / entropy_h0))


def _clip_unit(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _lme(values: tuple[float, ...]) -> float:
    """Log-mean-exp aggregator (matches predictor's risk-aggregation form)."""
    if not values:
        return float("-inf")
    arr = np.asarray(values, dtype=np.float64)
    m = float(arr.max())
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.exp(arr - m).mean()))
