"""Phase D2/D3 controller integration tests (PLAN_RF.md §"Test plan" 4).

Exercises ``ReferenceFlowController`` mode switching with a stub scorer that
returns deterministic per-sequence window risks. Pure-function correctness is
covered separately in ``test_reference_flow_counterfactual.py`` and
``test_reference_flow_commit.py``; this file checks that the mode-vs-handler
wiring in ``controller.step()`` and ``controller.post_step()`` respects the
PLAN's identity / correction / freeze semantics.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.controller import (
    ControllerStepResult,
    PostSamplingContext,
    PostSamplingResult,
    ReferenceFlowController,
    SamplerStepContext,
)
from inverse_folding.reference_flow.controller_config import (
    ActiveWindowsConfig,
    AttributionConfig,
    CompletionConfig,
    ControllerConfig,
    ControlsConfig,
    D2Config,
    D3Config,
    HeadConfig,
    ReliabilityConfig,
    TelemetryConfig,
)
from inverse_folding.reference_flow.head_scoring import (
    BatchHeadScores,
    HeadScore,
    WindowRiskRecord,
)


_AA = "AFGHILMNPQRSTVWYDCEK"  # 20 canonical AAs (token id = index+1, mask=0)
_MASK_ID = 0
_VOCAB_SIZE = 1 + len(_AA)
_CANONICAL = tuple(range(1, 1 + len(_AA)))
_HIGH_RISK = _AA.index("Y") + 1  # token id of "Y"
_LOW_RISK = _AA.index("A") + 1   # token id of "A"


def _decode(tokens: torch.Tensor) -> str:
    out: list[str] = []
    for tok in tokens.tolist():
        if tok == _MASK_ID:
            out.append("A")
        else:
            out.append(_AA[(int(tok) - 1) % len(_AA)])
    return "".join(out)


class _StubScorer:
    """Window risk = count of 'Y' residues in the window range (raw_logit scale)."""

    def __init__(self, *, window_specs=((0, 5, 5), (3, 8, 5))) -> None:
        self.window_specs = window_specs
        self.allele = "DRB1*01:01"

    def score_batch_same_protein(
        self, *, protein_id: str, records: list[tuple[str, str]]
    ) -> BatchHeadScores:
        scores = []
        for label, seq in records:
            wins = []
            for start, end, k in self.window_specs:
                z = float(seq[start:end].count("Y"))
                wins.append(
                    WindowRiskRecord(start_0b=start, end_0b=end, k=k, z=z)
                )
            scores.append(
                HeadScore(
                    protein_id=protein_id,
                    sequence_md5=hashlib.md5(seq.encode("utf-8")).hexdigest(),
                    sequence_length=len(seq),
                    allele=self.allele,
                    score_scale="raw_logit",
                    windows=tuple(wins),
                )
            )
        return BatchHeadScores(scores=tuple(scores))

    def get_or_compute_static(self, protein_id: str, sequence: str) -> HeadScore:
        return self.score_batch_same_protein(
            protein_id=protein_id, records=[("static", sequence)]
        ).scores[0]


def _make_config(
    *,
    mode: str,
    d2_enabled: bool,
    d3_enabled: bool,
    beta: float = 1.0,
    min_ess_fraction: float = 0.0,
    max_positions: int = 2,
    max_candidates: int = 16,
    final_freeze_steps: int = 1,
    min_completion_fraction: float = 0.0,
    refresh_interval: int = 5,
    t_start: float = 0.5,
) -> ControllerConfig:
    return ControllerConfig(
        enabled=True,
        mode=mode,
        t_start=t_start,
        refresh_interval=refresh_interval,
        completion=CompletionConfig(method="argmax"),
        head=HeadConfig(),
        active_windows=ActiveWindowsConfig(
            excess_threshold=0.0,
            max_windows=16,
            selection="threshold_then_top_n",
            merge_overlapping_scoring_windows=True,
        ),
        reliability=ReliabilityConfig(
            time_k=20.0,
            entropy_h0=1.5,
            min_completion_fraction=min_completion_fraction,
            min_rho_to_emit_event=0.0,
        ),
        d2=D2Config(
            enabled=d2_enabled,
            beta=beta,
            eta=1.0,
            epsilon=1.0e-8,
            top_k_tokens=4,
            max_positions_per_block=max_positions,
            max_candidates_per_block=max_candidates,
            min_delta_R_improvement=0.0,
            min_ess_fraction=min_ess_fraction,
            max_abs_logit_shift=10.0,
            paired_uncorrected_sample=True,
        ),
        d3=D3Config(
            enabled=d3_enabled,
            window_to_residue_projection="max_covering_window",
            gamma_min=0.4,
            gamma_max=0.9,
            lambda_commit=1.0,
            zscore_epsilon=1.0e-6,
            same_refresh_grace=True,
            final_freeze_steps=final_freeze_steps,
        ),
        attribution=AttributionConfig(),
        controls=ControlsConfig(),
        telemetry=TelemetryConfig(),
    )


def _struct_logits(L: int) -> torch.Tensor:
    """Strong preference for 'Y' across all positions (drives active window)."""
    logits = torch.full((L, _VOCAB_SIZE), -50.0)
    for i in range(L):
        # Mask token deeply negative.
        logits[i, _MASK_ID] = -100.0
        # Bias toward Y > A > others so hard-completion is full of Y.
        logits[i, _HIGH_RISK] = 4.0
        logits[i, _LOW_RISK] = 3.0
        # Fill other canonical with smaller logits to give a top-K spread.
        for tok in _CANONICAL:
            if tok in (_HIGH_RISK, _LOW_RISK):
                continue
            logits[i, tok] = -2.0
    return logits


def _make_context(*, x_t: torch.Tensor, logits: torch.Tensor, step: int, t: float):
    return SamplerStepContext(
        x_t=x_t,
        logits=logits,
        scores=np.full(int(logits.shape[0]), -np.inf, dtype=np.float64),
        step=step,
        t=t,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=int(logits.shape[0]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_active_blocks_emits_no_d2_events_and_keeps_identity_logits():
    # Static sequence already full of Y → dynamic equals static → window_excess = 0.
    L = 10
    static_seq = "Y" * L
    logits = _struct_logits(L)
    x_t = torch.tensor([_MASK_ID] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    assert torch.equal(res.logits, logits)
    assert all(
        row["event_type"] != "D2" for row in controller.controller_event_rows()
    )


def test_d2_logits_mode_corrects_supported_tokens_at_active_block_positions():
    L = 10
    static_seq = "A" * L  # static low-risk
    logits = _struct_logits(L)
    # Partial commitment so g_comp > 0 inside the active block; positions
    # [0,4) are committed with high-risk Y, [4, L) are still masked.
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # Some delta MUST be present: D1 detects the all-Y completion as risky and
    # D2 candidates that replace Y with A reduce ΔR_B → mass should shift.
    diff = (res.logits - logits).abs()
    assert diff.max().item() > 1e-3, "expected non-trivial D2 correction"
    # Outside the active-block residue range or outside the candidate support
    # the entries must be unchanged. The active windows are [0,5) and [3,8) so
    # the active block spans [0,8). Positions [8, L) are untouched at ALL tokens.
    for i in range(8, L):
        assert torch.allclose(res.logits[i], logits[i], atol=1e-12)


def test_monitor_only_with_d2_enabled_returns_identity_logits():
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor([_MASK_ID] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="monitor_only", d2_enabled=True, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # monitor_only must NOT modify logits even though D2 ran for diagnostics.
    assert torch.equal(res.logits, logits)


def test_d2_beta_zero_produces_identity_within_numerical_tolerance():
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor([_MASK_ID] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False, beta=0.0),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    assert torch.allclose(res.logits, logits, atol=1e-6)


def test_d2_beta_zero_strict_null_under_sampled_mode():
    """PLAN §D2-19: at beta=0 corrected logits must equal structural logits
    exactly within tolerance. Force the sampled enumeration path by setting
    max_candidates_per_block well below the Cartesian product size."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(
            mode="d2_logits",
            d2_enabled=True,
            d3_enabled=False,
            beta=0.0,
            max_candidates=4,  # 2 positions x top-K=4 = 16 cartesian → sampled
        ),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    assert torch.allclose(res.logits, logits, atol=1e-9), (
        "sampled-mode D2 at beta=0 must produce identity logits (pi==Q empirically)"
    )


def test_low_ess_disables_block_correction_and_zeros_g_ess():
    # Force low ESS via a tight min_ess_fraction threshold paired with a budget
    # large enough that one peaked candidate dominates the weighted set.
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor([_MASK_ID] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(
            mode="d2_logits",
            d2_enabled=True,
            d3_enabled=False,
            beta=10.0,
            min_ess_fraction=0.99,
        ),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    res = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # ESS gate triggers: g_ESS=0, rho_B=0, no correction even though active blocks exist.
    assert torch.equal(res.logits, logits)
    refresh_records = controller.refresh_records()
    assert refresh_records, "expected a refresh record on the refresh step"
    assert any(
        any(blk.g_ESS == 0.0 for blk in rec.active_blocks)
        for rec in refresh_records
    )


def test_post_step_returns_empty_protection_outside_freeze_window():
    L = 8
    logits = _struct_logits(L)
    x_t = torch.tensor([_LOW_RISK] * L, dtype=torch.long)
    scores = np.zeros(L, dtype=np.float64)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_d3_full", d2_enabled=True, d3_enabled=True, final_freeze_steps=1),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    ctx = PostSamplingContext(
        x_t=x_t,
        scores=scores,
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=4,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(ctx)
    assert res.protected_positions == ()


def test_post_step_freeze_protects_all_committed_in_final_window():
    L = 8
    logits = _struct_logits(L)
    x_t = torch.tensor([_LOW_RISK if i % 2 == 0 else _MASK_ID for i in range(L)], dtype=torch.long)
    scores = np.zeros(L, dtype=np.float64)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_d3_full", d2_enabled=True, d3_enabled=True, final_freeze_steps=2),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    ctx = PostSamplingContext(
        x_t=x_t,
        scores=scores,
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=8,            # n_steps=10, final_freeze_steps=2 → freeze active.
        t=0.8,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(ctx)
    # All currently committed positions (even indices) must be protected.
    assert set(res.protected_positions) == {0, 2, 4, 6}


def test_d3_revisit_mode_returns_commit_score_on_refresh_step():
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_LOW_RISK] * 4 + [_MASK_ID] * 2, dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    pre = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # pre.refresh_record must exist (we are at a refresh step).
    assert pre.refresh_record is not None
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=pre.logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(post_ctx)
    assert isinstance(res, PostSamplingResult)
    assert res.rank_scores is not None
    assert res.rank_scores.shape == (L,)
    assert res.refresh_addendum is not None
    assert "m_i" in res.refresh_addendum and "e_i" in res.refresh_addendum


def test_d3_revisit_mode_returns_none_on_non_refresh_step():
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_LOW_RISK] * 4 + [_MASK_ID] * 2, dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    # Refresh-trigger step==5 has not happened yet.
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=3,
        t=0.3,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(post_ctx)
    assert res.rank_scores is None
    assert res.refresh_addendum is None


def test_monitor_only_with_d3_enabled_returns_no_rank_scores():
    """Rule 2: monitor_only computes diagnostics but stays identity."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_LOW_RISK] * 4 + [_MASK_ID] * 2, dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="monitor_only", d2_enabled=True, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(post_ctx)
    # rank_scores must stay None for monitor_only, but the refresh addendum
    # is still produced so D0 metrics get the EMA arrays.
    assert res.rank_scores is None
    assert res.refresh_addendum is not None


def test_d2_d3_full_grace_keeps_commit_score_finite_at_corrected_positions():
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_d3_full", d2_enabled=True, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    pre = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # Simulate sampler post-state: a corrected position commits a token (e.g. A).
    x_t_after = x_t.clone()
    refresh_state = controller._refresh_state
    if refresh_state.corrected_positions:
        first_corr = next(iter(refresh_state.corrected_positions))
        x_t_after[int(first_corr)] = _LOW_RISK

    post_ctx = PostSamplingContext(
        x_t=x_t_after,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=pre.logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(post_ctx)
    if refresh_state.corrected_positions:
        # Grace positions are not NaN/inf; they are present in addendum.
        grace = res.refresh_addendum["grace_positions"]
        assert set(grace) == {int(p) for p in refresh_state.corrected_positions}


def test_d2_logits_emits_per_corrected_position_event_rows_after_post_step():
    """F3 contract: D2 mode emits a D2 event row per corrected position with
    a_before / logit_struct / logit_corrected / delta_logit_max / KL /
    delta_R_B / rho_B / ESS_candidates, and post_step fills a_after /
    a_uncorrected / paired_disagreement_flag from sampler output."""
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    pre = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # D2 must have applied a correction (verified in earlier test); at least
    # one corrected position must exist in refresh state.
    assert controller._refresh_state.corrected_positions, (
        "expected at least one corrected position"
    )
    corrected_positions = sorted(int(p) for p in controller._refresh_state.corrected_positions)

    # Simulate sampler post-state at exactly those positions: corrected -> A,
    # uncorrected -> Y (so paired disagreement is true).
    selected = np.array(corrected_positions, dtype=np.int64)
    actual = np.array([_LOW_RISK] * len(corrected_positions), dtype=np.int64)
    uncorr = np.array([_HIGH_RISK] * len(corrected_positions), dtype=np.int64)
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=pre.logits,
        selected_positions=selected,
        sampled_tokens_actual=actual,
        sampled_tokens_uncorrected=uncorr,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    res = controller.post_step(post_ctx)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D2"]
    assert rows, "expected D2 event rows"
    by_pos = {int(r["position_i"]): r for r in rows}
    for pos in corrected_positions:
        row = by_pos[pos]
        # a_before is the hard-completion (argmax) token used as the baseline
        # for ΔR_B; for the test fixture all masked positions argmax to Y.
        assert row["a_before"] == _HIGH_RISK
        assert row["a_after"] == _LOW_RISK
        assert row["a_uncorrected"] == _HIGH_RISK
        assert row["paired_disagreement_flag"] is True
        assert row["delta_logit_max"] > 0.0
        assert row["kl_struct_corrected"] >= 0.0
        # block-level delta_R_B is precomputed from candidate enumeration
        assert row["delta_R_B"] is not None
        # PostSamplingResult also reports the same rows for downstream writers.
    assert len(res.post_event_rows) == len(rows)


def test_d2_event_row_carries_realized_delta_R_corrected_and_uncorrected():
    """G1 contract: post-sampling re-score fills realized ΔR^Ω(B) for both
    actual and paired-uncorrected branches (PLAN_RF.md D0 schema)."""
    L = 10
    static_seq = "A" * L
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence=static_seq,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    pre = controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    corrected_positions = sorted(int(p) for p in controller._refresh_state.corrected_positions)
    assert corrected_positions, "expected at least one corrected position"

    # Simulate the actual sampler: corrected positions get A (low risk), paired
    # uncorrected gets Y (high risk). Apply tokens to x_t to mirror sampler
    # state at post_step time.
    x_t_post = x_t.clone()
    for p in corrected_positions:
        x_t_post[p] = _LOW_RISK
    selected = np.array(corrected_positions, dtype=np.int64)
    actual = np.array([_LOW_RISK] * len(corrected_positions), dtype=np.int64)
    uncorr = np.array([_HIGH_RISK] * len(corrected_positions), dtype=np.int64)
    post_ctx = PostSamplingContext(
        x_t=x_t_post,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=pre.logits,
        selected_positions=selected,
        sampled_tokens_actual=actual,
        sampled_tokens_uncorrected=uncorr,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    controller.post_step(post_ctx)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D2"]
    assert rows
    for row in rows:
        assert row["delta_R_corrected"] is not None
        assert row["delta_R_uncorrected"] is not None
        # Actual branch replaces Y with A → lower head risk; corrected ΔR
        # must be strictly less than uncorrected ΔR for the stub scorer
        # (which counts Y residues over the window).
        assert row["delta_R_corrected"] < row["delta_R_uncorrected"]


def test_monitor_only_d2_diagnostic_rows_keep_realized_delta_R_null():
    """H2 contract: monitor_only with d2.enabled emits D2 diagnostic rows
    (reason='d2_monitor_only') but applies no corrected logits. The
    realized-ΔR fill must NOT touch these rows because there's no
    "corrected branch" to be realized."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="monitor_only", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    corrected = sorted(int(p) for p in controller._refresh_state.corrected_positions)
    selected = np.array(corrected, dtype=np.int64) if corrected else np.array([], dtype=np.int64)
    actual = np.array([_LOW_RISK] * len(corrected), dtype=np.int64)
    uncorr = np.array([_HIGH_RISK] * len(corrected), dtype=np.int64)
    post_ctx = PostSamplingContext(
        x_t=x_t, scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits, corrected_logits=logits,
        selected_positions=selected,
        sampled_tokens_actual=actual, sampled_tokens_uncorrected=uncorr,
        step=5, t=0.5, n_steps=10,
        mask_token_id=_MASK_ID, protein_id="P1", design_idx=0, sequence_length=L,
    )
    controller.post_step(post_ctx)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D2"]
    if rows:
        for row in rows:
            assert row["reason"] == "d2_monitor_only"
            assert row["delta_R_corrected"] is None
            assert row["delta_R_uncorrected"] is None


def test_d2_event_row_a_after_remains_null_when_position_not_in_sampler_selection():
    """If a D2-corrected position never made it into ``selected_positions``,
    a_after stays None (event still emitted with pre-sample fields)."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_MASK_ID] * (L - 4), dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d2_logits", d2_enabled=True, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    # Sampler picked NO positions this step.
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    controller.post_step(post_ctx)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D2"]
    for row in rows:
        assert row["a_after"] is None
        assert row["a_uncorrected"] is None


def test_d3_post_remask_emits_event_rows_per_remasked_position():
    """F4 contract: post_remask emits a D3 event row per remasked residue
    populated with m_i, commit_score, remask_flag=True, grace_flag, and a
    reason string from {immune_risk, low_confidence}."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_LOW_RISK] * 4 + [_MASK_ID] * 2, dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    post_ctx = PostSamplingContext(
        x_t=x_t,
        scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits,
        corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5,
        t=0.5,
        n_steps=10,
        mask_token_id=_MASK_ID,
        protein_id="P1",
        design_idx=0,
        sequence_length=L,
    )
    controller.post_step(post_ctx)
    # Simulate sampler reporting two positions remasked.
    controller.post_remask(remasked_positions=(1, 5), step=5, t=0.5)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D3"]
    assert len(rows) == 2
    positions_seen = sorted(int(r["position_i"]) for r in rows)
    assert positions_seen == [1, 5]
    for row in rows:
        assert row["remask_flag"] is True
        assert row["m_i"] is not None
        assert row["commit_score"] is not None
        assert row["reason"] in {"immune_risk", "low_confidence"}


def test_d3_productive_revisit_pre_post_snapshot_round_trip():
    """G2 contract: post_remask records a pre-snapshot; the next refresh that
    observes the position re-committed resolves it into immune_only /
    structure_only / joint flags exposed via productive_revisit_outcomes."""
    L = 8
    logits = _struct_logits(L)
    # All committed with high-risk Y → high R^Ω covering remasked positions.
    x_t = torch.tensor([_HIGH_RISK] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    # Refresh 1: D3 commit runs, post_remask records pre-snapshot at pos=2.
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    post_ctx = PostSamplingContext(
        x_t=x_t, scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits, corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5, t=0.5, n_steps=20,
        mask_token_id=_MASK_ID, protein_id="P1", design_idx=0, sequence_length=L,
    )
    controller.post_step(post_ctx)
    controller.post_remask(remasked_positions=(2,), step=5, t=0.5)
    assert len(controller._d3_pending_snapshots) == 1
    assert not controller.productive_revisit_outcomes()  # not resolved yet

    # Refresh 2: position 2 has been re-sampled to A (low risk). The next
    # refresh must observe this and resolve the snapshot.
    x_t_next = x_t.clone()
    x_t_next[2] = _LOW_RISK
    controller.step(_make_context(x_t=x_t_next, logits=logits, step=10, t=0.6))
    resolved = controller.productive_revisit_outcomes()
    assert len(resolved) == 1
    entry = resolved[0]
    assert entry["position_i"] == 2
    assert entry["a_pre"] == _HIGH_RISK
    assert entry["a_post"] == _LOW_RISK
    # Replacing Y with A inside the window LOWERS risk → immune_only=True.
    assert entry["delta_R_local"] < 0
    assert entry["immune_only"] is True
    # Under the test fixture log p(Y)-log p(A) ≈ 1 nat, so the structural
    # drop exceeds δ_ℓ=0.5 → structure_only=False, joint=False. This shows
    # the immune gain came at a non-trivial structural cost.
    assert entry["structure_only"] is False
    assert entry["joint"] is False
    assert entry["delta_ell"] < -0.5


def test_d3_post_remask_noop_on_non_refresh_step_legacy_remask():
    """H1 contract: non-refresh steps fall back to legacy ``scores[]`` remask;
    those positions must NOT be attributed to D3 telemetry / productive
    snapshots even though ``_latest_d3_signal`` is still populated from the
    last refresh."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor(
        [_HIGH_RISK] * 4 + [_LOW_RISK] * 4 + [_MASK_ID] * 2, dtype=torch.long
    )
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    # Refresh step: D3 runs, post_remask emits D3 event.
    controller.step(_make_context(x_t=x_t, logits=logits, step=5, t=0.5))
    refresh_ctx = PostSamplingContext(
        x_t=x_t, scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits, corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=5, t=0.5, n_steps=20,
        mask_token_id=_MASK_ID, protein_id="P1", design_idx=0, sequence_length=L,
    )
    controller.post_step(refresh_ctx)
    controller.post_remask(remasked_positions=(2,), step=5, t=0.5)
    d3_after_refresh = [r for r in controller.controller_event_rows() if r["event_type"] == "D3"]
    assert len(d3_after_refresh) == 1

    # Non-refresh step (step=6 with refresh_interval=5): post_step returns
    # rank_scores=None, so legacy remask is what would happen in the
    # sampler. post_remask is still called (sampler hook), but the
    # controller MUST NOT log the legacy remask as D3.
    nonrefresh_ctx = PostSamplingContext(
        x_t=x_t, scores=np.zeros(L, dtype=np.float64),
        structural_logits=logits, corrected_logits=logits,
        selected_positions=np.array([], dtype=np.int64),
        sampled_tokens_actual=np.array([], dtype=np.int64),
        sampled_tokens_uncorrected=None,
        step=6, t=0.55, n_steps=20,
        mask_token_id=_MASK_ID, protein_id="P1", design_idx=0, sequence_length=L,
    )
    controller.post_step(nonrefresh_ctx)
    controller.post_remask(remasked_positions=(3,), step=6, t=0.55)
    d3_after_legacy = [r for r in controller.controller_event_rows() if r["event_type"] == "D3"]
    # Count unchanged: no new D3 event row, no new pre-snapshot from legacy remask.
    assert len(d3_after_legacy) == 1
    assert all(int(r["position_i"]) != 3 for r in d3_after_legacy)


def test_d3_post_remask_noop_when_d3_disabled():
    """monitor_only / d2_logits modes must not emit D3 event rows."""
    L = 10
    logits = _struct_logits(L)
    x_t = torch.tensor([_LOW_RISK] * L, dtype=torch.long)
    controller = ReferenceFlowController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * L,
        scorer=_StubScorer(),
        config=_make_config(mode="monitor_only", d2_enabled=False, d3_enabled=False),
        decode_tokens=_decode,
        canonical_token_ids=_CANONICAL,
    )
    controller.post_remask(remasked_positions=(2, 4), step=5, t=0.5)
    rows = [r for r in controller.controller_event_rows() if r["event_type"] == "D3"]
    assert rows == []


def test_d1_alias_constructor_still_works():
    # PLAN §"Implementation decision" 6 — the alias must keep D1 callers
    # buildable without the canonical_token_ids kwarg.
    from inverse_folding.reference_flow.controller import D1MonitorController

    controller = D1MonitorController(
        protein_id="P1",
        design_idx=0,
        seed=42,
        static_sequence="A" * 10,
        scorer=_StubScorer(),
        config=_make_config(mode="monitor_only", d2_enabled=False, d3_enabled=False),
        decode_tokens=_decode,
    )
    assert controller is not None
