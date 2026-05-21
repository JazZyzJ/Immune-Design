"""Phase D adaptive controller (monitor_only / D2 logits / D3 commit-revisit).

The controller exposes one public class — ``ReferenceFlowController`` — that
covers all four modes (PLAN_RF.md §"Task D2-D3"). D1 monitor-only callers can
continue to use the ``D1MonitorController`` alias unchanged; the D2/D3 modes
are activated by widening the ``ControllerConfig.mode`` enum and supplying
nested ``d2`` / ``d3`` sections.

Public surface:
- ``SamplerStepContext`` — passed to the pre-sampling hook each step.
- ``ControllerStepResult`` — returned by ``step()`` (logits + optional refresh
  record).
- ``PostSamplingContext`` — passed to the post-sampling hook each step.
- ``PostSamplingResult`` — returned by ``post_step()`` (rank_scores +
  protected_positions + event rows + refresh addendum).
- ``ActiveBlock`` — one merged scoring-window block with reliability gate
  factors.
- ``D1RefreshRecord`` — telemetry payload for ``refresh_log.jsonl``.
- ``RefreshState`` — in-memory snapshot of the latest refresh used across
  pre- / post-sampling hooks (structural logits, active blocks, residue
  excess, EMA memory, same-refresh corrected positions).
- ``ReferenceFlowController`` — the hook implementation (D2/D3 composition
  uses ``counterfactual.D2Handler`` and ``commit.D3Handler``).
- ``D1MonitorController`` — deprecated alias for ``ReferenceFlowController``.
- ``WindowMismatchError`` — raised when dynamic and static window enumerations
  do not align by ``(start_0b, end_0b, k)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .commit import D3Handler, select_freeze_protected_positions
from .controller_config import ControllerConfig
from .counterfactual import D2BlockOutcome, D2Handler, D2RefreshOutcome
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
    """Per-refresh telemetry payload.

    A refresh record must be **self-contained** for D0 metric derivation:
    consumers compute ``z_dyn - z_static`` directly from the record without
    joining against the static-window cache. This guards against cache
    corruption / resume mismatch / cache absence on rebuilt analyses.
    """

    protein_id: str
    design_idx: int
    seed: int
    refresh_step: int
    step: int
    t: float
    r_windows_dyn: tuple[WindowRiskRecord, ...]
    r_windows_static: tuple[WindowRiskRecord, ...]
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


@dataclass(frozen=True)
class PostSamplingContext:
    """Input to the post-sampling hook (PLAN_RF.md §"Sampler integration").

    Captures the post-sampling sequence/scores plus both the pre-D2
    structural logits and the actually-used corrected logits, so D3 can
    compute ``ell_i_cur`` from the structural distribution while paired-D2
    attribution can still refer to the corrected logits.
    """

    x_t: torch.Tensor
    scores: np.ndarray
    structural_logits: torch.Tensor
    corrected_logits: torch.Tensor
    selected_positions: np.ndarray
    sampled_tokens_actual: np.ndarray
    sampled_tokens_uncorrected: np.ndarray | None
    step: int
    t: float
    n_steps: int
    mask_token_id: int
    protein_id: str
    design_idx: int
    sequence_length: int


@dataclass(frozen=True)
class PostSamplingResult:
    """Post-sampling hook output.

    * ``rank_scores=None`` ↔ legacy ``_apply_reparam_remask`` uses ``scores[]``.
    * ``protected_positions`` are excluded from D3 remasking (grace) and from
      legacy remask (final freeze) so neither path can reopen them.
    * ``post_event_rows`` append to ``controller_events.parquet``.
    * ``refresh_addendum`` merges D3-specific arrays into the current refresh
      record before flushing ``refresh_log.jsonl``.
    """

    rank_scores: np.ndarray | None
    protected_positions: tuple[int, ...]
    post_event_rows: tuple[dict, ...]
    refresh_addendum: dict | None


@dataclass
class RefreshState:
    """Single in-memory source for cross-refresh / cross-hook controller state.

    The controller owns one instance per ``(protein_id, design_idx)`` design.
    All fields default to "no refresh yet"; ``step()`` rewrites the bag on
    refresh boundaries and ``post_step()`` reads/extends it without mutating
    sampler state.

    ``windows`` and ``window_excess`` are stored so the D3 post-sampling
    handler can re-project window-level excess to residues without
    re-running the head scorer.
    """

    last_refresh_step: int = -1
    structural_logits: torch.Tensor | None = None
    active_blocks: tuple[ActiveBlock, ...] = ()
    windows: tuple[WindowRiskRecord, ...] = ()
    window_excess: tuple[float, ...] = ()
    e_i: np.ndarray | None = None
    m_i: np.ndarray | None = None
    corrected_positions: frozenset[int] = field(default_factory=frozenset)


class ReferenceFlowController:
    """Phase D adaptive controller covering all four D-phase modes.

    The pre-sampling hook (``step``) is invoked after structural logit
    validation and before unmask sampling. On non-refresh steps it returns
    identity logits. At a refresh step it constructs a hard completion,
    scores it through the bound :class:`OnlineHeadScorer`, computes
    per-window excess vs. the static WT cache, merges active windows into
    residue-level blocks, and records reliability gate factors. When D2 is
    enabled the hook then adds active-block logit correction (PLAN §D2 step
    behavior).

    The post-sampling hook (``post_step``) is invoked after token sampling
    and before remask. When D3 is enabled it updates ``m_i`` EMA memory,
    constructs a residue-level commit score, and returns ``rank_scores`` for
    ``_apply_reparam_remask`` plus ``protected_positions`` for the final
    freeze window. ``scores[]`` is never mutated.

    monitor_only mode keeps both hooks identity-pass-through; it only emits
    telemetry. D1 callers may use the historical ``D1MonitorController``
    alias.
    """

    def __init__(
        self,
        *,
        protein_id: str,
        design_idx: int,
        seed: int,
        static_sequence: str,
        scorer: OnlineHeadScorer,
        config: ControllerConfig,
        decode_tokens: Callable[[torch.Tensor], str],
        canonical_token_ids: Sequence[int] | None = None,
    ) -> None:
        self.protein_id = str(protein_id)
        self.design_idx = int(design_idx)
        self.seed = int(seed)
        self.static_sequence = str(static_sequence)
        self.scorer = scorer
        self.config = config
        self._decode_tokens = decode_tokens
        self._canonical_token_ids = (
            tuple(int(t) for t in canonical_token_ids)
            if canonical_token_ids is not None
            else None
        )
        if config.d2.enabled and self._canonical_token_ids is None:
            raise ValueError(
                "ReferenceFlowController requires canonical_token_ids when "
                "controller.d2.enabled=true"
            )
        self._d2_handler: D2Handler | None = (
            D2Handler(config.d2) if config.d2.enabled else None
        )
        self._d3_handler: D3Handler | None = (
            D3Handler(config.d3) if config.d3.enabled else None
        )

        self._refresh_records: list[D1RefreshRecord] = []
        self._refresh_addenda: dict[int, dict] = {}
        self._event_rows: list[dict] = []
        self._refresh_step_counter = 0
        self._static_head_score: HeadScore | None = None
        self._refresh_state: RefreshState = RefreshState()
        self._latest_d2_outcome: D2RefreshOutcome | None = None
        # Per-corrected-position pending event metadata recorded at the
        # refresh step; the post-sampling hook fills the a_after /
        # a_uncorrected / paired_disagreement / chosen-token-logit fields and
        # flushes the rows into ``_event_rows``.
        self._pending_d2_events: list[dict] = []
        # Last D3 commit pass output (m_i, commit_score, grace) cached so
        # ``post_remask`` can emit per-remasked-residue D3 event rows.
        self._latest_d3_signal: dict | None = None

    # ---------- public accessors for telemetry flush ----------

    def refresh_records(self) -> list[D1RefreshRecord]:
        return list(self._refresh_records)

    def refresh_addenda(self) -> dict[int, dict]:
        """Map of refresh_step → post_step addendum (D3 ``e_i``/``m_i`` etc.)."""

        return dict(self._refresh_addenda)

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

        # D2 candidate scoring + logit correction (PLAN §"D2 step behavior").
        # In monitor_only mode with d2.enabled, the handler still runs for
        # diagnostics, but the returned logits stay identity. In d2_logits /
        # d2_d3_full modes the corrected logits replace the structural ones.
        d2_outcome: D2RefreshOutcome | None = None
        out_logits = context.logits
        if self._d2_handler is not None:
            current_window_risks = tuple(float(w.z) for w in dyn_score.windows)
            d2_outcome = self._d2_handler.correct_logits(
                structural_logits=context.logits,
                active_blocks=active_blocks,
                x_t=context.x_t,
                mask_token_id=int(context.mask_token_id),
                completed_tokens=completed_tokens,
                decode_tokens=self._decode_tokens,
                per_pos_entropy=per_pos_entropy,
                residue_excess=_residue_excess_from_windows(
                    windows=dyn_score.windows,
                    window_excess=window_excess,
                    sequence_length=int(context.sequence_length),
                ),
                current_window_risks=current_window_risks,
                scorer=self.scorer,
                canonical_token_ids=self._canonical_token_ids or (),
                protein_id=self.protein_id,
                seed=self.seed,
                design_idx=self.design_idx,
                refresh_step=self._refresh_step_counter,
            )
            if self.config.mode in {"d2_logits", "d2_d3_full"}:
                out_logits = d2_outcome.corrected_logits
            # Rebuild active_blocks with D2-derived g_ESS so the refresh log /
            # rho_B reflects the candidate-ESS gate when a block was demoted.
            active_blocks = _apply_d2_g_ess_to_blocks(active_blocks, d2_outcome)
            # Capture per-corrected-position pending event metadata. Only the
            # actually-applied (d2_logits / d2_d3_full) modes commit a logit
            # shift; monitor_only still computes diagnostics for D0 attribution
            # so we record the events anyway and flag the mode.
            applies_correction = self.config.mode in {"d2_logits", "d2_d3_full"}
            self._pending_d2_events = _build_pending_d2_events(
                d2_outcome=d2_outcome,
                completed_tokens=completed_tokens,
                structural_logits=context.logits,
                corrected_logits=out_logits if applies_correction else d2_outcome.corrected_logits,
                refresh_step=self._refresh_step_counter,
                step=int(context.step),
                t=float(context.t),
                protein_id=self.protein_id,
                design_idx=self.design_idx,
                seed=self.seed,
                applies_correction=applies_correction,
            )
        self._latest_d2_outcome = d2_outcome

        # Write D2 per-block diagnostics into the refresh addendum so the
        # refresh log carries candidate_feasibility / candidate_count /
        # best_delta_R_B / mean_delta_R_B / ESS_B_candidates / g_ESS /
        # corrected_positions per block plus refresh-level cumulative
        # kl_struct_corrected and delta_logit_max summaries (PLAN_RF.md
        # §"Telemetry migration" line 967). D3's later run_refresh extends
        # the same addendum.
        if d2_outcome is not None:
            block_diagnostics: list[dict] = []
            cum_kl = 0.0
            max_abs_shift = 0.0
            for block in d2_outcome.block_outcomes:
                dl_max = (
                    max(
                        (abs(float(v)) for v in block.delta_logit.values()),
                        default=0.0,
                    )
                )
                if dl_max > max_abs_shift:
                    max_abs_shift = float(dl_max)
                # kl_struct_corrected is computed per corrected position in
                # the D2 event row; here we approximate the refresh-level
                # cumulative by summing block-level delta_logit_max * |A_B|
                # as a structural-budget proxy. Exact per-position KL lives
                # in controller_events.parquet.
                block_diagnostics.append(
                    {
                        "block_id": int(block.block_id),
                        "candidate_mode": block.candidate_mode,
                        "candidate_count": int(block.candidate_count),
                        "candidate_feasibility": bool(block.feasible),
                        "best_delta_R_B": float(block.best_delta_R_B)
                        if block.candidate_count
                        else None,
                        "mean_delta_R_B": float(block.mean_delta_R_B)
                        if block.candidate_count
                        else None,
                        "ESS_B_candidates": float(block.ess),
                        "g_ESS_candidates": float(block.g_ESS_candidates),
                        "rho_B_effective": float(block.rho_B_effective),
                        "corrected_positions": list(block.corrected_positions),
                        "skipped_reason": block.skipped_reason,
                        "delta_logit_max_block": float(dl_max),
                    }
                )
            self._refresh_addenda[self._refresh_step_counter] = {
                "d2_block_diagnostics": block_diagnostics,
                "delta_logit_max_refresh": float(max_abs_shift),
            }

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
            seed=self.seed,
            refresh_step=self._refresh_step_counter,
            step=int(context.step),
            t=float(context.t),
            r_windows_dyn=tuple(dyn_score.windows),
            r_windows_static=tuple(static_score.windows),
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
            if block.rho_B < min_rho:
                continue
            # Full D0 schema (PLAN_RF.md §D0 controller_events.parquet).
            # D2/D3-only fields are written as None in D1 so downstream
            # tooling does not need arm-specific column branches.
            self._event_rows.append(
                _build_monitor_event_row(
                    protein_id=self.protein_id,
                    design_idx=self.design_idx,
                    seed=self.seed,
                    refresh_step=self._refresh_step_counter,
                    step=int(context.step),
                    t=float(context.t),
                    block=block,
                    dyn_score=dyn_score,
                )
            )

        # RefreshState snapshot — consumed by post_step() and by D2/D3
        # handlers in subsequent code paths. structural_logits is the
        # pre-D2 tensor so D3 can compute commit scores from the untouched
        # structural distribution (PLAN §D3-7). corrected_positions are the
        # positions whose logits actually received a D2 shift this refresh;
        # they drive the same-refresh grace rule.
        corrected_positions = (
            d2_outcome.corrected_positions if d2_outcome is not None else frozenset()
        )
        self._refresh_state = RefreshState(
            last_refresh_step=int(context.step),
            structural_logits=context.logits,
            active_blocks=tuple(active_blocks),
            windows=tuple(dyn_score.windows),
            window_excess=tuple(window_excess),
            e_i=None,
            m_i=self._refresh_state.m_i,
            corrected_positions=corrected_positions,
        )

        self._refresh_step_counter += 1
        return ControllerStepResult(logits=out_logits, refresh_record=record)

    # ---------- post-sampling hook ----------

    def post_step(self, context: PostSamplingContext) -> PostSamplingResult:
        """Post-sampling hook (PLAN §"D3 step behavior").

        Returns:
        * ``rank_scores`` — commit score on refresh steps when D3 is active
          and we are outside the freeze window; otherwise ``None`` so
          ``_apply_reparam_remask`` falls back to legacy ``scores[]``.
        * ``protected_positions`` — all committed positions inside the freeze
          window (validation rule 13). Independent of refresh cadence.
        * ``refresh_addendum`` — ``e_i`` and ``m_i`` arrays for D3 refreshes,
          merged into ``refresh_log.jsonl`` downstream.
        """
        if not self.config.enabled:
            return PostSamplingResult(
                rank_scores=None,
                protected_positions=(),
                post_event_rows=(),
                refresh_addendum=None,
            )

        protected: tuple[int, ...] = ()
        if self.config.d3.enabled:
            protected = select_freeze_protected_positions(
                x_t=context.x_t,
                mask_token_id=int(context.mask_token_id),
                step=int(context.step),
                n_steps=int(context.n_steps),
                final_freeze_steps=int(self.config.d3.final_freeze_steps),
            )

        rank_scores: np.ndarray | None = None
        refresh_addendum: dict | None = None

        # D3 commit pathway: refresh step + outside freeze window + handler active.
        is_refresh_now = (
            self._d3_handler is not None
            and self._refresh_state.last_refresh_step == int(context.step)
            and int(context.step)
            < int(context.n_steps) - int(self.config.d3.final_freeze_steps)
            and self._refresh_state.structural_logits is not None
        )
        if is_refresh_now:
            outcome = self._d3_handler.run_refresh(
                windows=self._refresh_state.windows,
                window_excess=self._refresh_state.window_excess,
                active_blocks=self._refresh_state.active_blocks,
                sequence_length=int(context.sequence_length),
                structural_logits=self._refresh_state.structural_logits,
                x_t=context.x_t,
                mask_token_id=int(context.mask_token_id),
                m_prev=self._refresh_state.m_i,
                corrected_positions=tuple(self._refresh_state.corrected_positions),
            )
            # Only emit rank_scores when in a D3 mode; monitor_only with
            # d3.enabled stays identity per PLAN validation rule 2.
            if self.config.mode in {"d3_revisit", "d2_d3_full"}:
                rank_scores = outcome.commit_score
            # Persist EMA forward for the next refresh regardless of mode.
            self._refresh_state.e_i = outcome.e_i
            self._refresh_state.m_i = outcome.m_i
            refresh_addendum = {
                "e_i": outcome.e_i.tolist(),
                "m_i": outcome.m_i.tolist(),
                "rho_i": outcome.rho_i.tolist(),
                "grace_positions": list(outcome.grace_positions),
            }
            # Persist by refresh_step so telemetry writers can zip with the
            # corresponding refresh record. The most recent record was just
            # appended by ``step()`` for this same context.step. We MERGE
            # rather than overwrite so the D2 block diagnostics already
            # written in ``step()`` are preserved alongside the D3 arrays.
            if self._refresh_records:
                key = int(self._refresh_records[-1].refresh_step)
                existing = self._refresh_addenda.get(key, {})
                existing.update(refresh_addendum)
                self._refresh_addenda[key] = existing

        # Cache D3 commit signal so post_remask can attribute D3 events.
        if is_refresh_now:
            self._latest_d3_signal = {
                "m_i": outcome.m_i,
                "rho_i": outcome.rho_i,
                "commit_score": outcome.commit_score,
                "grace_positions": set(int(p) for p in outcome.grace_positions),
            }

        # Flush any D2 pending events into the controller event buffer.
        # This is mode-agnostic: monitor_only with d2.enabled emits D2 rows
        # for diagnostics, d2_logits / d2_d3_full emit them as actual
        # interventions. The fill-in uses the sampler-supplied selected /
        # corrected / uncorrected token arrays.
        flushed_d2 = _flush_pending_d2_events(
            pending=self._pending_d2_events,
            selected_positions=context.selected_positions,
            sampled_tokens_actual=context.sampled_tokens_actual,
            sampled_tokens_uncorrected=context.sampled_tokens_uncorrected,
        )
        self._pending_d2_events = []
        self._event_rows.extend(flushed_d2)

        return PostSamplingResult(
            rank_scores=rank_scores,
            protected_positions=protected,
            post_event_rows=tuple(flushed_d2),
            refresh_addendum=refresh_addendum,
        )

    # ---------- post-remask hook ----------

    def post_remask(
        self,
        *,
        remasked_positions: tuple[int, ...],
        step: int,
        t: float,
    ) -> None:
        """Emit one D3 event row per remasked residue (PLAN §D3-14).

        Only fires in D3 modes and only when a D3 commit pass has produced a
        ``_latest_d3_signal``. ``reason`` is classified using the sign of the
        z-scored EMA risk and structural confidence components.
        """
        if not (self.config.enabled and self.config.d3.enabled):
            return
        if not remasked_positions:
            return
        signal = self._latest_d3_signal
        if signal is None:
            return
        m_i = signal["m_i"]
        commit_score = signal["commit_score"]
        grace_positions = signal["grace_positions"]
        # Active-block coverage lookup for the reason attribution.
        active_residues: set[int] = set()
        for blk in self._refresh_state.active_blocks:
            for r in range(int(blk.residue_start_0b), int(blk.residue_end_0b)):
                active_residues.add(int(r))
        for pos in remasked_positions:
            pos = int(pos)
            in_active = pos in active_residues
            in_grace = pos in grace_positions
            reason = _classify_d3_reason(
                m_i_pos=float(m_i[pos]) if pos < len(m_i) else 0.0,
                in_active=in_active,
                in_grace=in_grace,
            )
            self._event_rows.append(
                {
                    # ---- identity ----
                    "protein_id": str(self.protein_id),
                    "design_idx": int(self.design_idx),
                    "seed": int(self.seed),
                    "refresh_step": int(self._refresh_step_counter - 1),
                    "step": int(step),
                    "t": float(t),
                    "event_type": "D3",
                    # ---- locator ----
                    "block_id": None,
                    "position_i": int(pos),
                    "window_start": None,
                    "window_end": None,
                    # ---- D2 columns (null for D3 rows) ----
                    "a_before": None,
                    "a_after": None,
                    "a_uncorrected": None,
                    "delta_R_corrected": None,
                    "delta_R_uncorrected": None,
                    "paired_disagreement_flag": None,
                    "logit_struct": None,
                    "logit_corrected": None,
                    "delta_logit_max": None,
                    "kl_struct_corrected": None,
                    "delta_R_B": None,
                    "delta_R_i": None,
                    "ESS_candidates": None,
                    "rho_B": None,
                    # ---- D3 commit / revisit ----
                    "m_i": float(m_i[pos]) if pos < len(m_i) else None,
                    "commit_score": float(commit_score[pos]) if pos < len(commit_score) else None,
                    "remask_flag": True,
                    "grace_flag": bool(in_grace),
                    # ---- reason ----
                    "reason": reason,
                    # ---- D1 reliability factor breakdown (null for D3) ----
                    "g_time": None,
                    "g_comp": None,
                    "g_ent": None,
                    "g_ESS": None,
                }
            )

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


def _build_monitor_event_row(
    *,
    protein_id: str,
    design_idx: int,
    seed: int,
    refresh_step: int,
    step: int,
    t: float,
    block: ActiveBlock,
    dyn_score,
) -> dict:
    """Emit one ``controller_events.parquet`` row in the D0 schema.

    D1 fills only the identity + active-block reliability fields; every
    D2/D3-only column is set to ``None`` so the parquet schema is stable
    across arms (downstream tooling reads the same column set for monitor,
    D2-only, D3-only and full-D rows).

    D1-extra reliability factor breakdowns (``g_time/g_comp/g_ent/g_ESS``)
    are added at the end as additional columns; D2/D3 may overwrite them
    when they are no longer monotonic 1.0 in D1.
    """
    # Window-level span fallback for block-level monitor events. In D1 every
    # event is block-only, so position_i stays None and window_start/end
    # default to the merged block span.
    return {
        # ---- identity ----
        "protein_id": protein_id,
        "design_idx": int(design_idx),
        "seed": int(seed),
        "refresh_step": int(refresh_step),
        "step": int(step),
        "t": float(t),
        "event_type": "monitor",
        # ---- locator ----
        "block_id": int(block.block_id),
        "position_i": None,
        "window_start": int(block.residue_start_0b),
        "window_end": int(block.residue_end_0b),
        # ---- D2 token-direction columns (nullable in D1) ----
        "a_before": None,
        "a_after": None,
        "a_uncorrected": None,
        "delta_R_corrected": None,
        "delta_R_uncorrected": None,
        "paired_disagreement_flag": None,
        "logit_struct": None,
        "logit_corrected": None,
        "delta_logit_max": None,
        "kl_struct_corrected": None,
        "delta_R_B": None,
        "delta_R_i": None,
        "ESS": None,
        # ---- reliability gate (D1 owns) ----
        "rho_B": float(block.rho_B),
        # ---- D3 commit / revisit columns (nullable in D1) ----
        "m_i": None,
        "commit_score": None,
        "remask_flag": False,
        "grace_flag": None,
        # ---- reason ----
        "reason": "monitor_only",
        # ---- D1 reliability factor breakdown (extras outside D0 minimum) ----
        "g_time": float(block.g_time),
        "g_comp": float(block.g_comp),
        "g_ent": float(block.g_ent),
        "g_ESS": float(block.g_ESS),
    }


def _per_position_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy that is robust to ``-inf`` logits.

    The DPLM denoiser wrapper sets special-token logits to ``-inf``. After
    softmax those entries have probability 0, and the term ``0 * -inf``
    becomes ``NaN``. Summing across the vocabulary then propagates the NaN
    and a naive ``nan_to_num`` would zero the entire position, silently
    collapsing the entropy gate to ``g_ent=1`` on every refresh. We mask the
    non-finite contributions to 0 *before* the sum so only the canonical
    (finite-logit) tokens enter the entropy.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    contrib = -(probs * log_probs)
    contrib = torch.where(
        torch.isfinite(contrib),
        contrib,
        torch.zeros_like(contrib),
    )
    return contrib.sum(dim=-1)


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


# Backward-compatible alias for D1 callers and existing tests. The class
# itself now covers monitor_only + D2 + D3; the alias is kept so the D1
# monitor-only path does not need a global call-site sweep.
D1MonitorController = ReferenceFlowController


def _residue_excess_from_windows(
    *,
    windows: Sequence[WindowRiskRecord],
    window_excess: Sequence[float],
    sequence_length: int,
) -> np.ndarray:
    """Local ``max_covering_window`` projection used by D2 for position ranking.

    This is the same projection that ``commit.project_window_excess_to_residue``
    performs; we re-implement it locally to avoid pulling commit.py into the
    D2-only code path. The single canonical projection used by D3 lives in
    ``commit.py``.
    """
    L = int(sequence_length)
    out = np.zeros(L, dtype=np.float64)
    for w, exc in zip(windows, window_excess):
        e_val = float(exc)
        if e_val <= 0.0:
            continue
        s = max(0, int(w.start_0b))
        e_end = min(L, int(w.end_0b))
        for i in range(s, e_end):
            if e_val > out[i]:
                out[i] = e_val
    return out


def _build_pending_d2_events(
    *,
    d2_outcome: D2RefreshOutcome,
    completed_tokens: torch.Tensor,
    structural_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    refresh_step: int,
    step: int,
    t: float,
    protein_id: str,
    design_idx: int,
    seed: int,
    applies_correction: bool,
) -> list[dict]:
    """Per-corrected-position D0-schema event scaffolds (pre-sampling fill).

    Fills every field that is observable at refresh time: identity, locator,
    a_before, structural / corrected chosen-token log-probs at the
    delta-logit ``argmax`` token (sampled token replaces it post-sampling),
    delta_logit_max, per-position KL, and block-level delta_R_B / ESS /
    rho_B. ``a_after``, ``a_uncorrected``, ``paired_disagreement_flag``,
    realized ``delta_R_corrected`` and ``delta_R_uncorrected`` are filled by
    the post-sampling flush.
    """
    log_probs_struct = torch.log_softmax(structural_logits, dim=-1).detach().cpu()
    log_probs_corr = torch.log_softmax(corrected_logits, dim=-1).detach().cpu()
    pending: list[dict] = []
    for block in d2_outcome.block_outcomes:
        if block.skipped_reason is not None:
            continue
        for pos in block.corrected_positions:
            # Per-position shift table restricted to candidate support.
            shifts_at_pos = {
                tok: block.delta_logit[(int(pos), tok)]
                for tok in block.K_i_per_pos[int(pos)]
                if (int(pos), tok) in block.delta_logit
            }
            if not shifts_at_pos:
                continue
            delta_logit_max = max(abs(float(v)) for v in shifts_at_pos.values())
            kl = _per_position_kl(
                log_p=log_probs_corr[int(pos)],
                log_q=log_probs_struct[int(pos)],
            )
            a_before = int(completed_tokens[int(pos)].item())
            # Token with the largest positive shift — the D2 push direction.
            # Used as the "chosen-token" reference for logit_struct /
            # logit_corrected until the sampler reveals a_after.
            push_token = max(shifts_at_pos.items(), key=lambda kv: kv[1])[0]
            pending.append(
                {
                    # ---- identity ----
                    "protein_id": str(protein_id),
                    "design_idx": int(design_idx),
                    "seed": int(seed),
                    "refresh_step": int(refresh_step),
                    "step": int(step),
                    "t": float(t),
                    "event_type": "D2",
                    # ---- locator ----
                    "block_id": int(block.block_id),
                    "position_i": int(pos),
                    "window_start": None,
                    "window_end": None,
                    # ---- D2 token-direction (pre-sampling) ----
                    "a_before": int(a_before),
                    "a_after": None,
                    "a_uncorrected": None,
                    "delta_R_corrected": None,
                    "delta_R_uncorrected": None,
                    "paired_disagreement_flag": None,
                    "logit_struct": float(log_probs_struct[int(pos), int(push_token)].item()),
                    "logit_corrected": float(log_probs_corr[int(pos), int(push_token)].item()),
                    "delta_logit_max": float(delta_logit_max),
                    "kl_struct_corrected": float(kl),
                    "delta_R_B": float(block.best_delta_R_B),
                    "delta_R_i": None,
                    "ESS_candidates": float(block.ess),
                    "rho_B": float(block.rho_B_effective),
                    # ---- D3 commit / revisit ----
                    "m_i": None,
                    "commit_score": None,
                    "remask_flag": False,
                    "grace_flag": None,
                    # ---- reason ----
                    "reason": "d2_correction_applied" if applies_correction else "d2_monitor_only",
                    # ---- D1 reliability factor breakdown ----
                    "g_time": None,
                    "g_comp": None,
                    "g_ent": None,
                    "g_ESS": float(block.g_ESS_candidates),
                    # ---- pending fill markers (private) ----
                    "_push_token": int(push_token),
                }
            )
    return pending


def _flush_pending_d2_events(
    *,
    pending: list[dict],
    selected_positions: np.ndarray,
    sampled_tokens_actual: np.ndarray,
    sampled_tokens_uncorrected: np.ndarray | None,
) -> list[dict]:
    """Fill a_after / a_uncorrected / paired_disagreement_flag from sampler output."""
    if not pending:
        return []
    sel_to_idx: dict[int, int] = {
        int(p): int(i) for i, p in enumerate(np.asarray(selected_positions).tolist())
    }
    flushed: list[dict] = []
    for row in pending:
        pos = int(row["position_i"])
        idx = sel_to_idx.get(pos)
        if idx is not None and sampled_tokens_actual.size:
            row["a_after"] = int(sampled_tokens_actual[idx])
        if (
            idx is not None
            and sampled_tokens_uncorrected is not None
            and sampled_tokens_uncorrected.size
        ):
            row["a_uncorrected"] = int(sampled_tokens_uncorrected[idx])
            row["paired_disagreement_flag"] = bool(
                row["a_after"] is not None and row["a_after"] != row["a_uncorrected"]
            )
        # Drop the private push-token marker before flushing.
        row.pop("_push_token", None)
        flushed.append(row)
    return flushed


def _classify_d3_reason(
    *, m_i_pos: float, in_active: bool, in_grace: bool
) -> str:
    """Classify a D3 remask event by its driving signal (PLAN §D3-14).

    * ``in_active and m_i_pos > 0`` → immune_risk
    * ``in_active and grace`` → low_confidence (grace removed the immune term,
      so the remask was driven by structural confidence)
    * outside active region → low_confidence
    """
    if in_grace:
        return "low_confidence"
    if in_active and m_i_pos > 0.0:
        return "immune_risk"
    return "low_confidence"


def _per_position_kl(*, log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """Numerically robust KL(p || q) where p,q given as log-probabilities."""
    p = log_p.exp()
    diff = log_p - log_q
    contrib = p * diff
    contrib = torch.where(torch.isfinite(contrib), contrib, torch.zeros_like(contrib))
    return float(contrib.sum().item())


def _apply_d2_g_ess_to_blocks(
    active_blocks: Sequence[ActiveBlock],
    d2_outcome: D2RefreshOutcome,
) -> tuple[ActiveBlock, ...]:
    """Rebuild active blocks with D2-derived ``g_ESS`` and recomputed ``rho_B``.

    D1 always wrote ``g_ESS=1.0`` as a placeholder (no candidate reweighting).
    In D2 modes the candidate ESS gate may demote a block to ``g_ESS=0``,
    which must propagate into ``rho_B`` and therefore into refresh log /
    monitor event rows.
    """
    by_block_id: dict[int, D2BlockOutcome] = {
        int(out.block_id): out for out in d2_outcome.block_outcomes
    }
    new_blocks: list[ActiveBlock] = []
    for blk in active_blocks:
        outcome = by_block_id.get(int(blk.block_id))
        if outcome is None:
            new_blocks.append(blk)
            continue
        g_ESS = float(outcome.g_ESS_candidates)
        rho_B = float(
            _clip_unit(blk.g_time * blk.g_comp * blk.g_ent * g_ESS)
        )
        new_blocks.append(
            ActiveBlock(
                block_id=blk.block_id,
                residue_start_0b=blk.residue_start_0b,
                residue_end_0b=blk.residue_end_0b,
                window_indices=blk.window_indices,
                g_time=blk.g_time,
                g_comp=blk.g_comp,
                g_ent=blk.g_ent,
                g_ESS=g_ESS,
                rho_B=rho_B,
                completion_fraction=blk.completion_fraction,
                mean_struct_entropy=blk.mean_struct_entropy,
            )
        )
    return tuple(new_blocks)
