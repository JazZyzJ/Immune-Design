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

from .commit import (
    D3Handler,
    compute_stage_a_rank_score,
    select_freeze_protected_positions,
)
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


@dataclass
class PendingD2Correction:
    """Sticky pre-step D2 correction state.

    This state stores only the pending logit shift and refresh provenance. It
    is intentionally separate from D2 evidence, which stores post-sampling
    rank credit after paired disagreement and realized benefit are known.
    """

    position: int
    created_step: int
    refresh_step: int
    block_id: int
    omega_indices: tuple[int, ...]
    delta_logit: dict[int, float]
    r_current: float
    gap_a: float
    rho_B_effective: float
    block_outcome: D2BlockOutcome


@dataclass
class D2EvidenceWrite:
    """One positive D2 evidence write used by the Stage A rank face."""

    position: int
    created_step: int
    refresh_step: int
    block_id: int
    gap_a: float


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
        self._pending_d2_corrections: dict[int, PendingD2Correction] = {}
        self._d2_evidence_writes: list[D2EvidenceWrite] = []
        # Per-corrected-position pending event metadata recorded at the
        # refresh step; the post-sampling hook fills the a_after /
        # a_uncorrected / paired_disagreement / chosen-token-logit fields and
        # flushes the rows into ``_event_rows``.
        self._pending_d2_events: list[dict] = []
        # Last D3 commit pass output (m_i, commit_score, grace) cached so
        # ``post_remask`` can emit per-remasked-residue D3 event rows.
        self._latest_d3_signal: dict | None = None
        # Last Stage A rank face output cached for the independent remask
        # ledger. This is deliberately separate from the D3 signal because
        # d2_logits / d2_d3_full construct rank scores on non-refresh steps.
        self._latest_stage_a_rank_scores: np.ndarray | None = None
        self._latest_stage_a_rank_step: int | None = None
        # Productive-revisit tracking (PLAN_RF.md §D0 Layer B D3 metric 3).
        # ``_d3_pending_snapshots`` records pre-remask state per D3-remasked
        # residue; the next refresh that observes the residue re-committed
        # resolves the snapshot into ``_d3_resolved`` with immune-only /
        # structure-only / joint flags.
        self._d3_pending_snapshots: list[dict] = []
        self._d3_resolved: list[dict] = []
        # Cache of post-sampling pre-remask state needed by ``post_remask``.
        self._pre_remask_x_t: torch.Tensor | None = None
        # Did the latest post_step actually return D3 rank_scores? PLAN
        # §D3-2 limits D3 to refresh steps and the active t window; legacy
        # non-refresh remasks share the same sampler hook so post_remask
        # must gate on this flag to avoid attributing legacy remasks to D3.
        self._d3_used_in_last_post_step: bool = False

    # ---------- public accessors for telemetry flush ----------

    def refresh_records(self) -> list[D1RefreshRecord]:
        return list(self._refresh_records)

    def refresh_addenda(self) -> dict[int, dict]:
        """Map of refresh_step → post_step addendum (D3 ``e_i``/``m_i`` etc.)."""

        return dict(self._refresh_addenda)

    def controller_event_rows(self) -> list[dict]:
        return list(self._event_rows)

    def productive_revisit_outcomes(self) -> list[dict]:
        """Resolved productive-revisit pairs (PLAN_RF.md §D0 Layer B D3-3).

        Each entry carries pre- and post-resample state plus the
        immune-only / structure-only / joint flags. Pending snapshots that
        were never re-committed by the end of the run are NOT included; the
        caller can derive a "stuck" count via ``len(self._d3_pending_snapshots)``
        if needed.
        """

        return list(self._d3_resolved)

    # ---------- hook ----------

    def step(self, context: SamplerStepContext) -> ControllerStepResult:
        if not self.config.enabled:
            return ControllerStepResult(logits=context.logits, refresh_record=None)
        self._expire_pending_d2_corrections(step=int(context.step))
        if context.t < self.config.t_start:
            return ControllerStepResult(logits=context.logits, refresh_record=None)
        if context.step % self.config.refresh_interval != 0:
            out_logits = self._apply_pending_d2_corrections_for_step(context)
            return ControllerStepResult(logits=out_logits, refresh_record=None)

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

        # Resolve any productive-revisit pre-snapshots whose position has
        # been re-committed since the last D3 remask. ``ell_post`` uses the
        # CURRENT refresh's structural logits per PLAN §D3-7 convention;
        # ``R_post_local`` uses the current refresh's dyn windows. Pending
        # snapshots that remain masked stay queued.
        self._d3_pending_snapshots = _resolve_productive_revisit_snapshots(
            pending=self._d3_pending_snapshots,
            x_t=context.x_t,
            mask_token_id=int(context.mask_token_id),
            structural_logits=context.logits,
            current_windows=dyn_score.windows,
            current_refresh_step=int(self._refresh_step_counter),
            delta_logp_threshold=float(self.config.attribution.productive_delta_logp),
            resolved_sink=self._d3_resolved,
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
            # Rebuild active_blocks with D2-derived g_ESS so the refresh log /
            # rho_B reflects the candidate-ESS gate when a block was demoted.
            active_blocks = _apply_d2_g_ess_to_blocks(active_blocks, d2_outcome)
            applies_correction = self.config.mode in {"d2_logits", "d2_d3_full"}
            if applies_correction:
                self._install_pending_d2_corrections(
                    d2_outcome=d2_outcome,
                    created_step=int(context.step),
                    refresh_step=int(self._refresh_step_counter),
                )
            else:
                # monitor_only with d2.enabled emits diagnostics, but it must
                # not install sticky state or alter logits.
                self._pending_d2_events = _build_pending_d2_events(
                    d2_outcome=d2_outcome,
                    completed_tokens=completed_tokens,
                    structural_logits=context.logits,
                    corrected_logits=d2_outcome.corrected_logits,
                    canonical_token_ids=self._canonical_token_ids or (),
                    refresh_step=self._refresh_step_counter,
                    step=int(context.step),
                    t=float(context.t),
                    protein_id=self.protein_id,
                    design_idx=self.design_idx,
                    seed=self.seed,
                    applies_correction=False,
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
                        "safe_support_sizes": {
                            str(k): int(v) for k, v in block.safe_support_sizes.items()
                        },
                        "candidate_count_argmax": int(block.candidate_count_argmax),
                        "candidate_count_ensemble": int(block.candidate_count_ensemble),
                        "candidate_feasibility": bool(block.feasible),
                        "argmax_best_delta_R_B": float(block.argmax_best_delta_R_B)
                        if block.candidate_count_argmax
                        else None,
                        "ensemble_best_delta_R_B": float(block.ensemble_best_delta_R_B)
                        if block.candidate_count_ensemble
                        else None,
                        "argmax_to_ensemble_rank_flip_flag": bool(
                            block.argmax_to_ensemble_rank_flip_flag
                        ),
                        "argmax_to_ensemble_rank_flip_rate": float(
                            block.argmax_to_ensemble_rank_flip_rate
                        ),
                        "best_delta_R_B": float(block.best_delta_R_B)
                        if block.candidate_count
                        else None,
                        "mean_delta_R_B": float(block.mean_delta_R_B)
                        if block.candidate_count
                        else None,
                        "ESS_B_candidates": float(block.ess),
                        "g_ESS_candidates": float(block.g_ESS_candidates),
                        "rho_B_effective": float(block.rho_B_effective),
                        "rho_B_stageA": float(block.rho_B_effective),
                        "context_pnll": block.context_pnll,
                        "g_pnll": block.g_pnll,
                        "context_jsd": block.context_jsd,
                        "g_stability": float(block.g_stability),
                        "ensemble_delta_R_std": block.ensemble_delta_R_std,
                        "ensemble_sign_consistency": block.ensemble_sign_consistency,
                        "corrected_positions": list(block.corrected_positions),
                        "skipped_reason": block.skipped_reason,
                        "delta_logit_max_block": float(dl_max),
                    }
                )
            self._refresh_addenda[self._refresh_step_counter] = {
                "d2_block_diagnostics": block_diagnostics,
                "delta_logit_max_refresh": float(max_abs_shift),
                "active_sticky_positions": sorted(
                    int(p) for p in self._pending_d2_corrections
                ),
                "g_ESS_suppression_rate": (
                    float(
                        sum(1 for d in block_diagnostics if float(d["g_ESS_candidates"]) == 0.0)
                    )
                    / float(len(block_diagnostics))
                    if block_diagnostics
                    else 0.0
                ),
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

        if self.config.mode in {"d2_logits", "d2_d3_full"}:
            out_logits = self._apply_pending_d2_corrections_for_step(context)

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
        if self.config.mode in {"d2_logits", "d3_revisit", "d2_d3_full"}:
            protected = select_freeze_protected_positions(
                x_t=context.x_t,
                mask_token_id=int(context.mask_token_id),
                step=int(context.step),
                n_steps=int(context.n_steps),
                final_freeze_steps=int(self.config.d3.final_freeze_steps),
            )

        rank_scores: np.ndarray | None = None
        refresh_addendum: dict | None = None
        self._latest_stage_a_rank_scores = None
        self._latest_stage_a_rank_step = None
        # Reset per-step flag; set below iff the immediately preceding
        # post_step produced D3-attributable rank scores for this same step.
        self._d3_used_in_last_post_step = False

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
            # D3-only comparator keeps the previous refresh-step rank
            # semantics. D2 modes use the Stage A rank face below after D2
            # realized benefit/evidence has been filled.
            if self.config.mode == "d3_revisit":
                rank_scores = outcome.commit_score
                self._d3_used_in_last_post_step = True
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
                "step": int(context.step),
                "m_i": outcome.m_i,
                "rho_i": outcome.rho_i,
                "commit_score": outcome.commit_score,
                "grace_positions": set(int(p) for p in outcome.grace_positions),
            }

        # Cache the post-sampling / pre-remask x_t snapshot so post_remask
        # can recover the pre-remask tokens after the sampler has reverted
        # them to mask. Required for productive_revisit pre/post comparison.
        self._pre_remask_x_t = context.x_t.detach().clone()

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
        # Realized ΔR^Ω(B) for actual & paired-uncorrected branches
        # (PLAN_RF.md §D0 schema delta_R_corrected / delta_R_uncorrected).
        # One batch head re-score per refresh covers all blocks.
        if flushed_d2 and self._latest_d2_outcome is not None:
            _fill_realized_delta_R(
                flushed=flushed_d2,
                d2_outcome=self._latest_d2_outcome,
                structural_logits=context.structural_logits,
                x_t_post=context.x_t,
                mask_token_id=int(context.mask_token_id),
                sampled_tokens_uncorrected=context.sampled_tokens_uncorrected,
                selected_positions=context.selected_positions,
                scorer=self.scorer,
                decode_tokens=self._decode_tokens,
                protein_id=self.protein_id,
                refresh_step=self._refresh_step_counter - 1,
            )
        self._record_d2_evidence_from_flushed_rows(
            flushed=flushed_d2,
            context=context,
        )
        self._clear_selected_pending_d2(context.selected_positions)
        if self.config.mode in {"d2_logits", "d2_d3_full"}:
            rank_scores = self._build_stage_a_rank_scores(context)
            self._latest_stage_a_rank_scores = rank_scores
            self._latest_stage_a_rank_step = int(context.step)
            self._fill_rank_fields(flushed=flushed_d2, rank_scores=rank_scores, context=context)
            latest_signal_matches_step = (
                self._latest_d3_signal is not None
                and int(self._latest_d3_signal.get("step", -1)) == int(context.step)
            )
            if latest_signal_matches_step:
                self._latest_d3_signal["commit_score"] = rank_scores
            if self.config.mode == "d2_d3_full" and self.config.d3.enabled:
                self._d3_used_in_last_post_step = (
                    bool(is_refresh_now) and latest_signal_matches_step
                )
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

        Only fires when the immediately preceding ``post_step`` actually
        returned D3 ``rank_scores`` (i.e. a refresh step inside the active
        ``[t_start, n_steps - final_freeze_steps)`` window and a D3 mode).
        Otherwise the remask was driven by legacy ``scores[]`` and must not
        be attributed to D3 telemetry / productive_revisit snapshots.
        """
        if self.config.enabled and self.config.mode in {"d2_logits", "d2_d3_full"}:
            self._record_remask_ledger(
                remasked_positions=remasked_positions,
                step=int(step),
                t=float(t),
            )
        if self.config.enabled and self.config.d2.enabled and remasked_positions:
            self._clear_remasked_pending_d2(remasked_positions)
        if not (self.config.enabled and self.config.d3.enabled):
            return
        if not remasked_positions:
            return
        if not self._d3_used_in_last_post_step:
            # Legacy ``scores[]``-driven remask on a non-refresh step; PLAN
            # §D3-2 explicitly excludes these from D3 telemetry.
            return
        signal = self._latest_d3_signal
        if signal is None:
            return
        if int(signal.get("step", -1)) != int(step):
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
            # Productive-revisit pre-snapshot (resolved at the next refresh
            # where the position is re-committed). PLAN §D0 Layer B D3-3
            # criteria use ℓ_i^pre under the refresh's structural logits and
            # R_pre^Ω over the window set covering ``i``.
            if (
                self._pre_remask_x_t is not None
                and self._refresh_state.structural_logits is not None
            ):
                a_pre = int(self._pre_remask_x_t[pos].item())
                ell_pre = _chosen_token_logprob(
                    self._refresh_state.structural_logits, pos, a_pre
                )
                r_pre_local = _local_lme_over_windows_covering(
                    windows=self._refresh_state.windows, position=pos
                )
                self._d3_pending_snapshots.append(
                    {
                        "position_i": int(pos),
                        "refresh_step_remask": int(self._refresh_step_counter - 1),
                        "a_pre": a_pre,
                        "ell_pre": float(ell_pre),
                        "R_pre_local": float(r_pre_local),
                    }
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
                    # ---- Stage A nullable columns ----
                    **_stage_a_event_defaults(),
                }
            )

    # ---------- internals ----------

    def _install_pending_d2_corrections(
        self,
        *,
        d2_outcome: D2RefreshOutcome,
        created_step: int,
        refresh_step: int,
    ) -> None:
        """Install or refresh sticky D2 corrections keyed by position."""
        for block in d2_outcome.block_outcomes:
            if block.skipped_reason is not None:
                continue
            by_pos: dict[int, dict[int, float]] = {}
            for (pos, tok), shift in block.delta_logit.items():
                by_pos.setdefault(int(pos), {})[int(tok)] = float(shift)
            for pos in block.corrected_positions:
                pos_i = int(pos)
                if (
                    pos_i in self._pending_d2_corrections
                    and not self.config.d2.sticky_overwrite_on_refresh
                ):
                    continue
                self._pending_d2_corrections[pos_i] = PendingD2Correction(
                    position=pos_i,
                    created_step=int(created_step),
                    refresh_step=int(refresh_step),
                    block_id=int(block.block_id),
                    omega_indices=tuple(int(i) for i in block.omega_indices),
                    delta_logit=dict(by_pos.get(pos_i, {})),
                    r_current=float(block.r_current),
                    gap_a=0.0,
                    rho_B_effective=float(block.rho_B_effective),
                    block_outcome=block,
                )

    def _expire_pending_d2_corrections(self, *, step: int) -> None:
        ttl = int(self.config.d2.sticky_ttl_steps)
        expired = [
            pos
            for pos, corr in self._pending_d2_corrections.items()
            if int(step) - int(corr.created_step) >= ttl
        ]
        for pos in expired:
            self._pending_d2_corrections.pop(int(pos), None)

    def _apply_pending_d2_corrections_for_step(
        self, context: SamplerStepContext
    ) -> torch.Tensor:
        if self.config.mode not in {"d2_logits", "d2_d3_full"}:
            return context.logits
        if not self._pending_d2_corrections:
            self._pending_d2_events = []
            return context.logits
        corrected = context.logits.clone()
        delivered: list[PendingD2Correction] = []
        ttl = int(self.config.d2.sticky_ttl_steps)
        for pos, corr in sorted(self._pending_d2_corrections.items()):
            age = int(context.step) - int(corr.created_step)
            if age < 0 or age >= ttl:
                continue
            if int(context.x_t[int(pos)].item()) != int(context.mask_token_id):
                continue
            if not corr.delta_logit:
                continue
            for tok, shift in corr.delta_logit.items():
                corrected[int(pos), int(tok)] = corrected[int(pos), int(tok)] + float(shift)
            delivered.append(corr)
        self._pending_d2_events = _build_sticky_d2_events(
            corrections=delivered,
            structural_logits=context.logits,
            corrected_logits=corrected,
            canonical_token_ids=self._canonical_token_ids or (),
            step=int(context.step),
            t=float(context.t),
            protein_id=self.protein_id,
            design_idx=self.design_idx,
            seed=self.seed,
            d2_config=self.config.d2,
            d3_config=self.config.d3,
        )
        return corrected if delivered else context.logits

    def _clear_selected_pending_d2(self, selected_positions: np.ndarray) -> None:
        if not self.config.d2.sticky_clear_on_selected:
            return
        for pos in np.asarray(selected_positions, dtype=np.int64).tolist():
            self._pending_d2_corrections.pop(int(pos), None)

    def _record_remask_ledger(
        self,
        *,
        remasked_positions: tuple[int, ...],
        step: int,
        t: float,
    ) -> None:
        if not remasked_positions:
            return
        signal = self._latest_d3_signal
        signal_matches_step = (
            signal is not None and int(signal.get("step", -1)) == int(step)
        )
        current_rank_scores = (
            self._latest_stage_a_rank_scores
            if self._latest_stage_a_rank_step == int(step)
            else None
        )
        grace_positions = (
            signal["grace_positions"] if signal_matches_step else set()
        )
        m_i = signal["m_i"] if signal is not None else None
        commit_score = (
            current_rank_scores
            if current_rank_scores is not None
            else signal["commit_score"]
            if signal_matches_step
            else None
        )
        for pos in remasked_positions:
            pos_i = int(pos)
            rank_score = (
                float(commit_score[pos_i])
                if commit_score is not None and pos_i < len(commit_score)
                else None
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
                    "event_type": "remask",
                    # ---- locator ----
                    "block_id": None,
                    "position_i": int(pos_i),
                    "window_start": None,
                    "window_end": None,
                    # ---- D2 columns ----
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
                    "ESS_candidates": None,
                    "rho_B": None,
                    # ---- D3 / remask columns ----
                    "m_i": float(m_i[pos_i]) if m_i is not None and pos_i < len(m_i) else None,
                    "commit_score": rank_score,
                    "remask_flag": True,
                    "grace_flag": bool(pos_i in grace_positions),
                    "reason": "stage_a_remask_ledger",
                    # ---- reliability extras ----
                    "g_time": None,
                    "g_comp": None,
                    "g_ent": None,
                    "g_ESS": None,
                    # ---- Stage A ----
                    **_stage_a_event_defaults(),
                    "rank_score": rank_score,
                }
            )

    def _clear_remasked_pending_d2(self, remasked_positions: tuple[int, ...]) -> None:
        if not self.config.d2.sticky_clear_on_remask:
            return
        for pos in remasked_positions:
            self._pending_d2_corrections.pop(int(pos), None)

    def _d2_evidence_array(self, *, step: int, sequence_length: int) -> np.ndarray:
        L = int(sequence_length)
        out = np.zeros(L, dtype=np.float64)
        ttl = int(self.config.d3.d2_evidence_ttl_steps)
        still_live: list[D2EvidenceWrite] = []
        for write in self._d2_evidence_writes:
            age = int(step) - int(write.created_step)
            if age < 0:
                still_live.append(write)
                continue
            if age >= ttl:
                continue
            decay = max(0.0, 1.0 - float(age) / float(ttl))
            value = float(write.gap_a) * decay
            pos = int(write.position)
            if 0 <= pos < L and value > out[pos]:
                out[pos] = value
            still_live.append(write)
        self._d2_evidence_writes = still_live
        return out

    def _record_d2_evidence_from_flushed_rows(
        self,
        *,
        flushed: list[dict],
        context: PostSamplingContext,
    ) -> None:
        if not flushed:
            return
        log_probs_struct = torch.log_softmax(
            context.structural_logits, dim=-1
        ).detach().cpu()
        log_probs_corr = torch.log_softmax(
            context.corrected_logits, dim=-1
        ).detach().cpu()
        for row in flushed:
            if row.get("event_type") != "D2":
                continue
            row.setdefault("realized_benefit_flag", None)
            row.setdefault("d2_evidence", 0.0)
            if not row.get("sticky_selected_flag"):
                row["realized_benefit_flag"] = None
                row["d2_evidence"] = 0.0
                continue
            a_after = row.get("a_after")
            if a_after is None:
                row["realized_benefit_flag"] = None
                row["d2_evidence"] = 0.0
                continue
            pos = int(row["position_i"])
            tok = int(a_after)
            ell_struct = float(log_probs_struct[pos, tok].item())
            ell_corr = float(log_probs_corr[pos, tok].item())
            row["logit_struct"] = ell_struct
            row["logit_corrected"] = ell_corr
            row["ell_struct"] = ell_struct
            gap_a = float(ell_corr - ell_struct)
            dR_corr = row.get("delta_R_corrected")
            dR_uncorr = row.get("delta_R_uncorrected")
            benefit = (
                dR_corr is not None
                and dR_uncorr is not None
                and float(dR_corr) < float(dR_uncorr)
            )
            row["realized_benefit_flag"] = bool(benefit)
            disagreement = bool(row.get("paired_disagreement_flag"))
            benefit_ok = (
                benefit
                if self.config.d3.d2_evidence_requires_benefit
                else True
            )
            if disagreement and benefit_ok and gap_a > 0.0:
                self._d2_evidence_writes.append(
                    D2EvidenceWrite(
                        position=pos,
                        created_step=int(context.step),
                        refresh_step=int(row["refresh_step"]),
                        block_id=int(row["block_id"]),
                        gap_a=float(gap_a),
                    )
                )
                row["d2_evidence"] = float(gap_a)
            else:
                row["d2_evidence"] = 0.0

    def _build_stage_a_rank_scores(self, context: PostSamplingContext) -> np.ndarray:
        if self.config.mode == "d2_d3_full":
            m_i = self._refresh_state.m_i
        else:
            m_i = None
        d2_evidence = self._d2_evidence_array(
            step=int(context.step),
            sequence_length=int(context.sequence_length),
        )
        return compute_stage_a_rank_score(
            structural_logits=context.structural_logits,
            x_t=context.x_t,
            scores=context.scores,
            mask_token_id=int(context.mask_token_id),
            m_i=m_i,
            d2_evidence=d2_evidence,
            alpha_struct=float(self.config.d3.alpha_struct),
            lambda_commit=float(self.config.d3.lambda_commit),
            d2_evidence_nu=float(self.config.d3.d2_evidence_nu),
            zscore_epsilon=float(self.config.d3.zscore_epsilon),
        )

    def _fill_rank_fields(
        self,
        *,
        flushed: list[dict],
        rank_scores: np.ndarray,
        context: PostSamplingContext,
    ) -> None:
        if rank_scores is None:
            return
        committed = (context.x_t != int(context.mask_token_id)).detach().cpu().numpy()
        committed_scores = np.asarray(rank_scores, dtype=np.float64)[committed]
        for row in flushed:
            if row.get("event_type") != "D2":
                continue
            pos = int(row["position_i"])
            if 0 <= pos < len(rank_scores):
                row["rank_score"] = float(rank_scores[pos])
                if committed_scores.size:
                    row["rank_percentile_d2_written"] = float(
                        (committed_scores <= float(rank_scores[pos])).mean()
                    )
            if row.get("a_after") is not None:
                idx = {
                    int(p): int(i)
                    for i, p in enumerate(np.asarray(context.selected_positions).tolist())
                }.get(pos)
                if idx is not None and idx < len(context.scores):
                    row["ell_sample"] = float(context.scores[pos])

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
        "ESS_candidates": None,
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
        # ---- Stage A nullable columns ----
        **_stage_a_event_defaults(),
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
    canonical_token_ids: Sequence[int],
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
                    # ---- Stage A nullable columns ----
                    **_stage_a_event_defaults(),
                    **_structural_top4_support_fields(
                        log_probs=log_probs_struct,
                        position=int(pos),
                        canonical_token_ids=canonical_token_ids,
                    ),
                    # ---- pending fill markers (private) ----
                    "_push_token": int(push_token),
                }
            )
    return pending


def _stage_a_event_defaults() -> dict:
    return {
        "sticky_age_steps": None,
        "sticky_created_step": None,
        "sticky_selected_flag": None,
        "sticky_expired_flag": None,
        "realized_benefit_flag": None,
        "d2_evidence": None,
        "rank_score": None,
        "rank_percentile_d2_written": None,
        "ell_struct": None,
        "struct_top1_token": None,
        "struct_top1_logprob": None,
        "struct_top2_token": None,
        "struct_top2_logprob": None,
        "struct_top2_gap": None,
        "struct_top3_token": None,
        "struct_top3_logprob": None,
        "struct_top3_gap": None,
        "struct_top4_token": None,
        "struct_top4_logprob": None,
        "struct_top4_gap": None,
        "ell_sample": None,
        "alpha_struct": None,
        "d2_evidence_nu": None,
        "delta_struct": None,
        "struct_temperature": None,
        "context_pnll": None,
        "g_pnll": None,
        "context_jsd": None,
        "g_stability": None,
        "ensemble_delta_R_std": None,
        "ensemble_sign_consistency": None,
    }


def _build_sticky_d2_events(
    *,
    corrections: Sequence[PendingD2Correction],
    structural_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    canonical_token_ids: Sequence[int],
    step: int,
    t: float,
    protein_id: str,
    design_idx: int,
    seed: int,
    d2_config,
    d3_config,
) -> list[dict]:
    if not corrections:
        return []
    log_probs_struct = torch.log_softmax(structural_logits, dim=-1).detach().cpu()
    log_probs_corr = torch.log_softmax(corrected_logits, dim=-1).detach().cpu()
    pending: list[dict] = []
    for corr in corrections:
        pos = int(corr.position)
        shifts = corr.delta_logit
        if not shifts:
            continue
        push_token = max(shifts.items(), key=lambda kv: kv[1])[0]
        delta_logit_max = max(abs(float(v)) for v in shifts.values())
        kl = _per_position_kl(
            log_p=log_probs_corr[pos],
            log_q=log_probs_struct[pos],
        )
        block = corr.block_outcome
        row = {
            # ---- identity ----
            "protein_id": str(protein_id),
            "design_idx": int(design_idx),
            "seed": int(seed),
            "refresh_step": int(corr.refresh_step),
            "step": int(step),
            "t": float(t),
            "event_type": "D2",
            # ---- locator ----
            "block_id": int(corr.block_id),
            "position_i": int(pos),
            "window_start": None,
            "window_end": None,
            # ---- D2 token-direction (pre-sampling delivery) ----
            "a_before": int(torch.argmax(structural_logits[pos]).item()),
            "a_after": None,
            "a_uncorrected": None,
            "delta_R_corrected": None,
            "delta_R_uncorrected": None,
            "paired_disagreement_flag": None,
            "logit_struct": float(log_probs_struct[pos, int(push_token)].item()),
            "logit_corrected": float(log_probs_corr[pos, int(push_token)].item()),
            "delta_logit_max": float(delta_logit_max),
            "kl_struct_corrected": float(kl),
            "delta_R_B": float(block.best_delta_R_B),
            "delta_R_i": None,
            "ESS_candidates": float(block.ess),
            "rho_B": float(corr.rho_B_effective),
            # ---- D3 commit / revisit ----
            "m_i": None,
            "commit_score": None,
            "remask_flag": False,
            "grace_flag": None,
            # ---- reason ----
            "reason": "d2_sticky_delivery",
            # ---- D1 reliability factor breakdown ----
            "g_time": None,
            "g_comp": None,
            "g_ent": None,
            "g_ESS": float(block.g_ESS_candidates),
            # ---- Stage A ----
            **_stage_a_event_defaults(),
            **_structural_top4_support_fields(
                log_probs=log_probs_struct,
                position=pos,
                canonical_token_ids=canonical_token_ids,
            ),
            "sticky_age_steps": int(step) - int(corr.created_step),
            "sticky_created_step": int(corr.created_step),
            "sticky_selected_flag": False,
            "sticky_expired_flag": False,
            "alpha_struct": float(d3_config.alpha_struct),
            "d2_evidence_nu": float(d3_config.d2_evidence_nu),
            "delta_struct": float(d2_config.delta_struct),
            "struct_temperature": float(d2_config.struct_temperature),
            "context_pnll": getattr(block, "context_pnll", None),
            "g_pnll": getattr(block, "g_pnll", None),
            "context_jsd": getattr(block, "context_jsd", None),
            "g_stability": getattr(block, "g_stability", None),
            "ensemble_delta_R_std": getattr(block, "ensemble_delta_R_std", None),
            "ensemble_sign_consistency": getattr(block, "ensemble_sign_consistency", None),
            # ---- private ----
            "_push_token": int(push_token),
        }
        pending.append(row)
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
        row.setdefault("sticky_selected_flag", False)
        row.setdefault("sticky_expired_flag", False)
        row["sticky_selected_flag"] = bool(idx is not None)
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


def _fill_realized_delta_R(
    *,
    flushed: list[dict],
    d2_outcome: D2RefreshOutcome,
    structural_logits: torch.Tensor,
    x_t_post: torch.Tensor,
    mask_token_id: int,
    sampled_tokens_uncorrected: np.ndarray | None,
    selected_positions: np.ndarray,
    scorer: OnlineHeadScorer,
    decode_tokens: Callable[[torch.Tensor], str],
    protein_id: str,
    refresh_step: int,
) -> None:
    """Compute the realized post-sampling ΔR for the actual + paired branches.

    Strategy (PLAN_RF.md D0 schema + ``doc/Reference_Flow_Derivation.md``
    §D2 ΔR^Ω(B) definition):

    1. Build the "actual" full sequence by argmax-completing any residue still
       masked in ``x_t_post`` under the structural logits — this matches the
       hard-completion convention used at refresh time so the baseline
       ``r_current`` stored on each block remains directly comparable.
    2. Build the "uncorrected" full sequence by overriding ALL D2-corrected
       positions with ``sampled_tokens_uncorrected``; for positions outside
       the corrected set, uncorrected and actual sequences agree because the
       paired RNG snapshot guarantees identical tokens at uncorrected logits.
    3. One ``score_batch_same_protein`` call returns both head scores; we
       restrict to each block's ``omega_indices`` and re-aggregate via LME.
    4. ``delta_R_corrected = R_actual^Ω - r_current``; same for uncorrected.

    No-op when the refresh produced no corrected positions (e.g. low_ess or
    not_feasible blocks only), or when ``sampled_tokens_uncorrected`` is None
    (D2 paired sampling disabled).
    """
    if not flushed:
        return
    # Skip blocks that never produced a correction.
    corrected_blocks = {
        int(b.block_id): b for b in d2_outcome.block_outcomes if b.skipped_reason is None
    }
    if not corrected_blocks:
        return

    # 1. Actual sequence: clone post-sampling x_t; argmax-fill any residual masks.
    actual_tokens = x_t_post.detach().clone()
    masked = actual_tokens == int(mask_token_id)
    if masked.any():
        argmax = structural_logits.argmax(dim=-1)
        actual_tokens[masked] = argmax[masked].to(dtype=actual_tokens.dtype)

    records: list[tuple[str, str]] = [
        ("realized_actual", decode_tokens(actual_tokens))
    ]

    # 2. Paired uncorrected sequence — only meaningful when the sampler ran
    #    the paired branch AND the corrected positions were sampled this step.
    uncorrected_tokens: torch.Tensor | None = None
    if sampled_tokens_uncorrected is not None and selected_positions.size:
        pos_to_idx = {int(p): int(i) for i, p in enumerate(selected_positions.tolist())}
        uncorrected_tokens = actual_tokens.detach().clone()
        for blk in corrected_blocks.values():
            for pos in blk.corrected_positions:
                idx = pos_to_idx.get(int(pos))
                if idx is None:
                    continue
                uncorrected_tokens[int(pos)] = int(
                    sampled_tokens_uncorrected[idx]
                )
        records.append(("realized_uncorrected", decode_tokens(uncorrected_tokens)))

    # 3. One batched head re-score.
    batch = scorer.score_batch_same_protein(protein_id=str(protein_id), records=records)
    actual_windows = batch.scores[0].windows
    uncorrected_windows = batch.scores[1].windows if len(batch.scores) > 1 else None

    # 4. Compute ΔR per block and fill the flushed event rows in-place.
    block_dR_corrected: dict[int, float] = {}
    block_dR_uncorrected: dict[int, float] = {}
    for blk_id, block in corrected_blocks.items():
        omega = block.omega_indices
        r_actual = _local_lme(actual_windows, omega)
        block_dR_corrected[blk_id] = float(r_actual - block.r_current)
        if uncorrected_windows is not None:
            r_uncorr = _local_lme(uncorrected_windows, omega)
            block_dR_uncorrected[blk_id] = float(r_uncorr - block.r_current)
    # PLAN_RF.md D0 schema: delta_R_corrected is the realized risk delta
    # AFTER applying ``a_after`` from D2 corrected logits. monitor_only mode
    # emits D2 diagnostic rows with reason="d2_monitor_only" but never
    # applies a corrected-logit shift, so writing realized ΔR there would
    # conflate the two semantics. Only the actual-correction rows are
    # filled; monitor diagnostics stay null for both ΔR fields.
    for row in flushed:
        if row.get("reason") not in {"d2_correction_applied", "d2_sticky_delivery"}:
            continue
        bid = int(row["block_id"])
        if bid in block_dR_corrected and row.get("a_after") is not None:
            row["delta_R_corrected"] = float(block_dR_corrected[bid])
        if bid in block_dR_uncorrected and row.get("a_uncorrected") is not None:
            row["delta_R_uncorrected"] = float(block_dR_uncorrected[bid])


def _chosen_token_logprob(
    logits: torch.Tensor, position: int, token: int
) -> float:
    """log p_struct(token | position) under ``logits`` (used for ℓ_i^pre / ℓ_i^post)."""
    log_probs = torch.log_softmax(logits[int(position)], dim=-1).detach().cpu()
    return float(log_probs[int(token)].item())


def _local_lme_over_windows_covering(
    *, windows: Sequence[WindowRiskRecord], position: int
) -> float:
    """LME risk over windows that cover ``position`` (half-open ``[start, end)``).

    Returns ``-inf`` when no window covers the position so downstream
    consumers can detect the unresolvable case (productive_revisit treats
    these as missing rather than zero).
    """
    omega: list[int] = []
    for i, w in enumerate(windows):
        if int(w.start_0b) <= int(position) < int(w.end_0b):
            omega.append(i)
    if not omega:
        return float("-inf")
    return _local_lme(windows, tuple(omega))


def _resolve_productive_revisit_snapshots(
    *,
    pending: list[dict],
    x_t: torch.Tensor,
    mask_token_id: int,
    structural_logits: torch.Tensor,
    current_windows: Sequence[WindowRiskRecord],
    current_refresh_step: int,
    delta_logp_threshold: float,
    resolved_sink: list[dict],
) -> list[dict]:
    """For each pending snapshot whose position is now committed, fill the
    post-resample state and classify productive-revisit flags.

    Returns the still-unresolved snapshots so the controller can continue
    waiting on them. Resolution criteria follow PLAN_RF.md §D0 Layer B D3-3:

    * ``immune_only = 1[R_post^Ω - R_pre^Ω < 0]``
    * ``structure_only = 1[ℓ_i^post(a_i^post) - ℓ_i^pre(a_i^pre) > -δ_ℓ]``
    * ``joint = immune_only AND structure_only``

    Snapshots whose Ω window set is empty at remask time are dropped because
    ``R_pre^Ω = -inf`` makes the immune criterion ill-defined.
    """
    if not pending:
        return pending
    still_pending: list[dict] = []
    for snap in pending:
        pos = int(snap["position_i"])
        if int(x_t[pos].item()) == int(mask_token_id):
            still_pending.append(snap)
            continue
        # Position re-committed → resolve.
        a_post = int(x_t[pos].item())
        ell_post = _chosen_token_logprob(structural_logits, pos, a_post)
        r_post_local = _local_lme_over_windows_covering(
            windows=current_windows, position=pos
        )
        r_pre = float(snap["R_pre_local"])
        ell_pre = float(snap["ell_pre"])
        # Snapshots with degenerate Ω at remask time cannot be classified
        # meaningfully (R_pre = -inf or R_post = -inf). Drop them.
        if not (math.isfinite(r_pre) and math.isfinite(r_post_local)):
            continue
        delta_R_local = float(r_post_local - r_pre)
        delta_ell = float(ell_post - ell_pre)
        immune_only = delta_R_local < 0.0
        structure_only = delta_ell > -float(delta_logp_threshold)
        resolved_sink.append(
            {
                "position_i": pos,
                "refresh_step_remask": int(snap["refresh_step_remask"]),
                "refresh_step_resolve": int(current_refresh_step),
                "a_pre": int(snap["a_pre"]),
                "a_post": int(a_post),
                "ell_pre": float(ell_pre),
                "ell_post": float(ell_post),
                "R_pre_local": float(r_pre),
                "R_post_local": float(r_post_local),
                "delta_R_local": float(delta_R_local),
                "delta_ell": float(delta_ell),
                "immune_only": bool(immune_only),
                "structure_only": bool(structure_only),
                "joint": bool(immune_only and structure_only),
            }
        )
    return still_pending


def _local_lme(windows: Sequence[WindowRiskRecord], omega: tuple[int, ...]) -> float:
    """LME risk over window indices in ``omega`` (PLAN §D2-10 / Derivation §D2)."""
    if not omega:
        return float("-inf")
    vals = np.asarray([float(windows[int(i)].z) for i in omega], dtype=np.float64)
    m = float(vals.max())
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.exp(vals - m).mean()))


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


def _structural_top4_support_fields(
    *,
    log_probs: torch.Tensor,
    position: int,
    canonical_token_ids: Sequence[int],
) -> dict:
    """Structural top-4 canonical-token support telemetry for one target.

    ``delta_struct`` defines a trust region relative to the structural top
    token. Persisting top-rank log-prob gaps makes that threshold auditable
    without re-running the model.
    """
    empty = {
        "struct_top1_token": None,
        "struct_top1_logprob": None,
        "struct_top2_token": None,
        "struct_top2_logprob": None,
        "struct_top2_gap": None,
        "struct_top3_token": None,
        "struct_top3_logprob": None,
        "struct_top3_gap": None,
        "struct_top4_token": None,
        "struct_top4_logprob": None,
        "struct_top4_gap": None,
    }
    if log_probs.ndim != 2:
        return dict(empty)
    pos = int(position)
    if pos < 0 or pos >= int(log_probs.shape[0]):
        return dict(empty)
    vocab = int(log_probs.shape[1])
    canonical = [int(t) for t in canonical_token_ids if 0 <= int(t) < vocab]
    if not canonical:
        return dict(empty)

    canonical_tensor = torch.tensor(canonical, dtype=torch.long, device=log_probs.device)
    vals = log_probs[pos, canonical_tensor]
    k = min(4, int(vals.numel()))
    top_vals, top_idx = torch.topk(vals, k=k)
    top_tokens = canonical_tensor[top_idx]
    out = dict(empty)
    top1 = float(top_vals[0].item())
    for rank in range(1, k + 1):
        val = float(top_vals[rank - 1].item())
        out[f"struct_top{rank}_token"] = int(top_tokens[rank - 1].item())
        out[f"struct_top{rank}_logprob"] = val
        if rank > 1:
            out[f"struct_top{rank}_gap"] = float(top1 - val)
    return out


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
        rho_B = float(_clip_unit(outcome.rho_B_effective))
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
