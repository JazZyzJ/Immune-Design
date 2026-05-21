"""T4 contract tests: D1MonitorController (Phase D1).

Verifies refresh cadence gating, hard-completion semantics, window→block
geometry (single / two-overlap / three-transitive), max_windows cap,
reliability gate factors, completion-fraction cutoff, no-active-windows path,
and window-count mismatch fail-fast.

The tests use a stub scorer instead of a real epitope head: D1 only needs
``score_batch_same_protein`` and ``get_or_compute_static`` to return
HeadScore objects with a fixed window set.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from inverse_folding.reference_flow.controller import (
    ActiveBlock,
    ControllerStepResult,
    D1MonitorController,
    D1RefreshRecord,
    SamplerStepContext,
    WindowMismatchError,
)
from inverse_folding.reference_flow.controller_config import (
    ActiveWindowsConfig,
    CompletionConfig,
    ControllerConfig,
    HeadConfig,
    ReliabilityConfig,
    TelemetryConfig,
)
from inverse_folding.reference_flow.head_scoring import (
    BatchHeadScores,
    HeadScore,
    WindowRiskRecord,
)


# ---------- helpers ----------


_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_MASK_ID = 0  # arbitrary fixed mask id for tests
_VOCAB_SIZE = 1 + len(_AA_ALPHABET)


def _decode_tokens(tokens: torch.Tensor) -> str:
    """Map test token ids → amino acid string (mask → 'A')."""
    out: list[str] = []
    for tok in tokens.tolist():
        if tok == _MASK_ID:
            out.append("A")
        else:
            out.append(_AA_ALPHABET[(tok - 1) % len(_AA_ALPHABET)])
    return "".join(out)


def _make_controller_config(
    *,
    enabled: bool = True,
    t_start: float = 0.5,
    refresh_interval: int = 5,
    excess_threshold: float = 0.0,
    max_windows: int = 16,
    min_completion_fraction: float = 0.5,
    min_rho_to_emit_event: float = 0.0,
) -> ControllerConfig:
    return ControllerConfig(
        enabled=enabled,
        mode="monitor_only",
        t_start=t_start,
        refresh_interval=refresh_interval,
        completion=CompletionConfig(method="argmax"),
        head=HeadConfig(score_scale="raw_logit", static_cache_policy="lazy_write"),
        active_windows=ActiveWindowsConfig(
            excess_threshold=excess_threshold,
            max_windows=max_windows,
            selection="threshold_then_top_n",
            merge_overlapping_scoring_windows=True,
        ),
        reliability=ReliabilityConfig(
            time_k=20.0,
            entropy_h0=1.5,
            min_completion_fraction=min_completion_fraction,
            min_rho_to_emit_event=min_rho_to_emit_event,
        ),
        telemetry=TelemetryConfig(
            write_refresh_log=True,
            write_controller_events=True,
            write_per_protein_summary=True,
        ),
    )


def _make_head_score(protein_id: str, sequence: str, windows: list[WindowRiskRecord]) -> HeadScore:
    import hashlib
    return HeadScore(
        protein_id=protein_id,
        sequence_md5=hashlib.md5(sequence.encode()).hexdigest(),
        sequence_length=len(sequence),
        allele="DRB1_0101",
        score_scale="raw_logit",
        windows=tuple(windows),
    )


class StubScorer:
    """Minimal scorer mimicking the OnlineHeadScorer surface used by the controller."""

    def __init__(
        self,
        *,
        static_windows: list[WindowRiskRecord],
        dyn_windows_per_refresh: list[list[WindowRiskRecord]],
    ):
        self._static_windows = static_windows
        self._dyn_queue = list(dyn_windows_per_refresh)
        self.dyn_call_count = 0
        self.static_call_count = 0
        self.last_dyn_sequence: str | None = None

    def score_batch_same_protein(self, *, protein_id, records):
        self.dyn_call_count += 1
        label, seq = records[0]
        self.last_dyn_sequence = seq
        dyn_windows = self._dyn_queue.pop(0) if self._dyn_queue else self._static_windows
        return BatchHeadScores(scores=(_make_head_score(protein_id, seq, dyn_windows),))

    def get_or_compute_static(self, protein_id, sequence):
        self.static_call_count += 1
        return _make_head_score(protein_id, sequence, self._static_windows)


def _uniform_logits(length: int, vocab_size: int = _VOCAB_SIZE) -> torch.Tensor:
    """Per-position uniform logits → max entropy."""
    return torch.zeros(length, vocab_size, dtype=torch.float32)


def _sharp_logits(length: int, peak_token: int = 1, vocab_size: int = _VOCAB_SIZE) -> torch.Tensor:
    """Per-position one-hot logits at ``peak_token`` → near-zero entropy."""
    logits = torch.full((length, vocab_size), -1e4, dtype=torch.float32)
    logits[:, peak_token] = 1e4
    return logits


def _make_context(
    *,
    x_t: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    t: float,
    protein_id: str = "P1",
    design_idx: int = 0,
) -> SamplerStepContext:
    sequence_length = int(x_t.shape[0])
    scores = torch.full((sequence_length,), float("-inf"), dtype=torch.float64).numpy()
    return SamplerStepContext(
        x_t=x_t,
        logits=logits,
        scores=scores,
        step=step,
        t=t,
        mask_token_id=_MASK_ID,
        protein_id=protein_id,
        design_idx=design_idx,
        sequence_length=sequence_length,
    )


# ---------- 1) refresh gating ----------


def test_no_refresh_before_t_start():
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1)
    scorer = StubScorer(static_windows=[], dyn_windows_per_refresh=[])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    ctx = _make_context(
        x_t=torch.full((12,), _MASK_ID, dtype=torch.long),
        logits=_uniform_logits(12),
        step=0, t=0.3,
    )
    result = controller.step(ctx)
    assert isinstance(result, ControllerStepResult)
    assert result.refresh_record is None
    assert torch.equal(result.logits, ctx.logits)
    assert scorer.dyn_call_count == 0
    assert scorer.static_call_count == 0


def test_refresh_cadence_matches_t_start_and_interval():
    # n_steps=20, t_start=0.5, interval=5. Refresh at step in {10, 15} (t=step/20).
    n_steps = 20
    cfg = _make_controller_config(t_start=0.5, refresh_interval=5)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn_runs = [[WindowRiskRecord(0, 12, 12, 0.0)] for _ in range(10)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=dyn_runs)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    refresh_steps_observed = []
    for step in range(n_steps):
        ctx = _make_context(
            x_t=torch.zeros(12, dtype=torch.long).fill_(_MASK_ID + 1),  # all committed
            logits=_sharp_logits(12),
            step=step, t=step / n_steps,
        )
        result = controller.step(ctx)
        if result.refresh_record is not None:
            refresh_steps_observed.append(step)
    assert refresh_steps_observed == [10, 15]


# ---------- 2) hard completion ----------


def test_hard_completion_keeps_committed_argmaxes_masked():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [WindowRiskRecord(0, 5, 5, 0.0)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[static])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="ACDEF",
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    # Position 1 and 3 are masked; others committed.
    x_t = torch.tensor([2, _MASK_ID, 4, _MASK_ID, 6], dtype=torch.long)
    # logits[1] argmax → token 9, logits[3] argmax → token 13
    logits = torch.full((5, _VOCAB_SIZE), -1e4, dtype=torch.float32)
    logits[0, 2] = 1e4
    logits[1, 9] = 1e4
    logits[2, 4] = 1e4
    logits[3, 13] = 1e4
    logits[4, 6] = 1e4
    ctx = _make_context(x_t=x_t, logits=logits, step=0, t=0.5)
    controller.step(ctx)
    expected_tokens = torch.tensor([2, 9, 4, 13, 6], dtype=torch.long)
    expected_seq = _decode_tokens(expected_tokens)
    assert scorer.last_dyn_sequence == expected_seq


# ---------- 3) window→block geometry ----------


def test_single_active_window_produces_single_block():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [
        WindowRiskRecord(0, 12, 12, 0.0),
        WindowRiskRecord(5, 17, 12, 0.0),
    ]
    dyn = [
        WindowRiskRecord(0, 12, 12, 2.0),  # excess = 2 > 0 → active
        WindowRiskRecord(5, 17, 12, 0.0),  # excess = 0 → inactive
    ]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 20,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((20,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(20), step=0, t=0.5)
    record = controller.step(ctx).refresh_record
    assert record is not None
    assert len(record.active_blocks) == 1
    block = record.active_blocks[0]
    assert (block.residue_start_0b, block.residue_end_0b) == (0, 12)
    assert block.window_indices == (0,)


def test_two_overlapping_windows_merge_into_one_block():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [
        WindowRiskRecord(0, 15, 15, 0.0),
        WindowRiskRecord(10, 25, 15, 0.0),
        WindowRiskRecord(30, 45, 15, 0.0),
    ]
    dyn = [
        WindowRiskRecord(0, 15, 15, 2.0),
        WindowRiskRecord(10, 25, 15, 2.0),
        WindowRiskRecord(30, 45, 15, 2.0),
    ]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 50,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((50,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(50), step=0, t=0.5)
    record = controller.step(ctx).refresh_record
    assert record is not None
    spans = sorted((b.residue_start_0b, b.residue_end_0b) for b in record.active_blocks)
    assert spans == [(0, 25), (30, 45)]


def test_three_transitively_overlapping_windows_merge_into_one_block():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    # A∩B nonempty, B∩C nonempty, A∩C empty.
    static = [
        WindowRiskRecord(0, 15, 15, 0.0),
        WindowRiskRecord(10, 25, 15, 0.0),
        WindowRiskRecord(24, 39, 15, 0.0),
    ]
    dyn = [
        WindowRiskRecord(0, 15, 15, 2.0),
        WindowRiskRecord(10, 25, 15, 2.0),
        WindowRiskRecord(24, 39, 15, 2.0),
    ]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 50,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((50,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(50), step=0, t=0.5)
    record = controller.step(ctx).refresh_record
    assert record is not None
    assert len(record.active_blocks) == 1
    block = record.active_blocks[0]
    assert (block.residue_start_0b, block.residue_end_0b) == (0, 39)
    assert set(block.window_indices) == {0, 1, 2}


def test_max_windows_caps_after_thresholding_keeping_highest_excess():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1, max_windows=3)
    static = [WindowRiskRecord(i * 10, i * 10 + 12, 12, 0.0) for i in range(6)]
    # Excess: 0.5, 1.0, 2.0, 3.0, 1.5, 0.7 (all > 0 so all "active" pre-cap)
    dyn_z = [0.5, 1.0, 2.0, 3.0, 1.5, 0.7]
    dyn = [
        WindowRiskRecord(i * 10, i * 10 + 12, 12, z)
        for i, z in enumerate(dyn_z)
    ]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 80,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((80,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(80), step=0, t=0.5)
    record = controller.step(ctx).refresh_record
    assert record is not None
    # Top-3 excess: windows index {3 (z=3), 2 (z=2), 4 (z=1.5)}.
    kept_window_indices = sorted(
        idx for block in record.active_blocks for idx in block.window_indices
    )
    assert kept_window_indices == [2, 3, 4]


# ---------- 4) reliability gates ----------


def test_reliability_gate_factors_in_unit_interval():
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1, min_completion_fraction=0.0)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn = [WindowRiskRecord(0, 12, 12, 2.0)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 20,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    # Mixed completion: half masked, half committed.
    x_t = torch.zeros(20, dtype=torch.long).fill_(_MASK_ID + 1)
    x_t[:10] = _MASK_ID
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(20), step=0, t=0.7)
    block = controller.step(ctx).refresh_record.active_blocks[0]
    for factor in (block.g_time, block.g_comp, block.g_ent, block.g_ESS, block.rho_B):
        assert 0.0 <= factor <= 1.0, factor


def test_low_completion_reduces_rho_relative_to_high_completion():
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1, min_completion_fraction=0.0)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn = [WindowRiskRecord(0, 12, 12, 2.0)]
    base_logits = _sharp_logits(12)

    high_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    high = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=high_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    high_ctx = _make_context(
        x_t=torch.full((12,), _MASK_ID + 1, dtype=torch.long),
        logits=base_logits, step=0, t=0.7,
    )
    high_block = high.step(high_ctx).refresh_record.active_blocks[0]

    low_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    low = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=low_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    low_x = torch.full((12,), _MASK_ID, dtype=torch.long)
    low_x[0] = _MASK_ID + 1  # 1/12 committed
    low_ctx = _make_context(x_t=low_x, logits=base_logits, step=0, t=0.7)
    low_block = low.step(low_ctx).refresh_record.active_blocks[0]

    assert high_block.g_comp > low_block.g_comp
    assert high_block.rho_B > low_block.rho_B


def test_high_entropy_logits_reduce_g_ent():
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1, min_completion_fraction=0.0)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn = [WindowRiskRecord(0, 12, 12, 2.0)]
    x_t = torch.full((12,), _MASK_ID + 1, dtype=torch.long)

    sharp_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    sharp = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=sharp_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    sharp_ctx = _make_context(x_t=x_t, logits=_sharp_logits(12), step=0, t=0.7)
    sharp_block = sharp.step(sharp_ctx).refresh_record.active_blocks[0]

    uniform_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    uniform = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=uniform_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    uniform_ctx = _make_context(x_t=x_t, logits=_uniform_logits(12), step=0, t=0.7)
    uniform_block = uniform.step(uniform_ctx).refresh_record.active_blocks[0]

    assert sharp_block.g_ent > uniform_block.g_ent


def test_completion_fraction_below_min_forces_g_comp_zero():
    cfg = _make_controller_config(
        t_start=0.5, refresh_interval=1, min_completion_fraction=0.5,
    )
    static = [WindowRiskRecord(0, 10, 10, 0.0)]
    dyn = [WindowRiskRecord(0, 10, 10, 2.0)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((12,), _MASK_ID, dtype=torch.long)
    x_t[0] = _MASK_ID + 1  # block [0,10): 1/10 committed → below 0.5
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(12), step=0, t=0.7)
    block = controller.step(ctx).refresh_record.active_blocks[0]
    assert block.g_comp == 0.0
    assert block.rho_B == 0.0


# ---------- 5) no active windows + mismatch ----------


def test_no_active_windows_writes_refresh_but_no_event_rows():
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1, min_rho_to_emit_event=0.0)
    static = [WindowRiskRecord(0, 12, 12, 1.0)]
    dyn = [WindowRiskRecord(0, 12, 12, 1.0)]  # excess = 0 → no active windows
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((12,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(12), step=0, t=0.7)
    record = controller.step(ctx).refresh_record
    assert record is not None
    assert record.active_blocks == ()
    assert controller.controller_event_rows() == []


def test_window_count_mismatch_raises():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [
        WindowRiskRecord(0, 12, 12, 0.0),
        WindowRiskRecord(5, 17, 12, 0.0),
    ]
    dyn = [WindowRiskRecord(0, 12, 12, 1.0)]  # missing the second window
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 20,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((20,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(20), step=0, t=0.5)
    with pytest.raises(WindowMismatchError):
        controller.step(ctx)


# ---------- 6) telemetry hooks ----------


def test_controller_buffers_refresh_records_for_later_flush():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn_runs = [[WindowRiskRecord(0, 12, 12, 2.0)] for _ in range(3)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=dyn_runs)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((12,), _MASK_ID + 1, dtype=torch.long)
    for step in range(3):
        controller.step(_make_context(x_t=x_t, logits=_sharp_logits(12), step=step, t=0.6))
    records = controller.refresh_records()
    assert len(records) == 3
    assert [r.refresh_step for r in records] == [0, 1, 2]
    # rho_B >= 0 so event rows should accumulate (one block per refresh).
    events = controller.controller_event_rows()
    assert len(events) == 3
    assert all(row["event_type"] == "monitor" for row in events)
