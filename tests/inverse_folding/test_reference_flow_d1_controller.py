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

import numpy as np
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
from inverse_folding.reference_flow.actionability import (
    global_pressure_mass,
    prominence_thresholded_mass,
    protein_pressure_burden,
    smoothstep_pressure,
)
from inverse_folding.reference_flow.controller_config import (
    ActiveWindowsConfig,
    CompletionConfig,
    ControllerConfig,
    GlobalPressureConfig,
    HeadConfig,
    ReliabilityConfig,
    TargetingConfig,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="ACDEF",
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 20,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 50,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 50,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 80,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 20,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
        scorer=high_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    high_ctx = _make_context(
        x_t=torch.full((12,), _MASK_ID + 1, dtype=torch.long),
        logits=base_logits, step=0, t=0.7,
    )
    high_block = high.step(high_ctx).refresh_record.active_blocks[0]

    low_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    low = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
        scorer=sharp_scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    sharp_ctx = _make_context(x_t=x_t, logits=_sharp_logits(12), step=0, t=0.7)
    sharp_block = sharp.step(sharp_ctx).refresh_record.active_blocks[0]

    uniform_scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    uniform = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 20,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((20,), _MASK_ID + 1, dtype=torch.long)
    ctx = _make_context(x_t=x_t, logits=_sharp_logits(20), step=0, t=0.5)
    with pytest.raises(WindowMismatchError):
        controller.step(ctx)


# ---------- 6) telemetry hooks ----------


def test_entropy_robust_to_neg_inf_special_token_logits():
    """DPLM denoiser wrapper masks special tokens to -inf. The entropy gate
    must compute over the finite (canonical residue) subset rather than
    collapsing the whole position's entropy to 0 via NaN propagation."""
    cfg = _make_controller_config(t_start=0.5, refresh_interval=1, min_completion_fraction=0.0)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn = [WindowRiskRecord(0, 12, 12, 2.0)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((12,), _MASK_ID + 1, dtype=torch.long)
    # First 3 vocab slots are "special tokens" masked to -inf; remaining 18
    # slots get uniform logits → near-maximal entropy over canonical subset.
    # Without the fix, NaN propagation through .sum() drives entropy → 0 and
    # g_ent → 1.0.
    logits = torch.zeros(12, _VOCAB_SIZE, dtype=torch.float32)
    logits[:, :3] = float("-inf")
    ctx = _make_context(x_t=x_t, logits=logits, step=0, t=0.7)
    block = controller.step(ctx).refresh_record.active_blocks[0]
    # Entropy over 18 uniform canonicals ≈ log(18) ≈ 2.89 nats → g_ent ≈ exp(-1.93) ≈ 0.146
    assert block.g_ent < 0.5, f"g_ent={block.g_ent} — masked-logit entropy bug present"
    assert block.mean_struct_entropy > 1.0  # substantially non-zero


def test_refresh_record_is_self_contained():
    """D0 metric reconstruction must not depend on the static cache: the
    refresh record itself must carry both r_windows_dyn and r_windows_static
    and the seed/protein_id/design_idx identity triple."""
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [WindowRiskRecord(0, 12, 12, 0.5)]
    dyn = [WindowRiskRecord(0, 12, 12, 2.5)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    controller = D1MonitorController(
        protein_id="PX", design_idx=3, seed=99, static_sequence="A" * 12,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    ctx = _make_context(
        x_t=torch.full((12,), _MASK_ID + 1, dtype=torch.long),
        logits=_sharp_logits(12), step=0, t=0.5,
    )
    record = controller.step(ctx).refresh_record
    assert record.protein_id == "PX"
    assert record.design_idx == 3
    assert record.seed == 99
    assert len(record.r_windows_static) == len(record.r_windows_dyn) == 1
    assert record.r_windows_static[0].z == pytest.approx(0.5)
    assert record.r_windows_dyn[0].z == pytest.approx(2.5)
    # z_dyn - z_static is recoverable from the record alone (no cache join)
    assert record.r_windows_dyn[0].z - record.r_windows_static[0].z == pytest.approx(2.0)


def test_controller_buffers_refresh_records_for_later_flush():
    cfg = _make_controller_config(t_start=0.0, refresh_interval=1)
    static = [WindowRiskRecord(0, 12, 12, 0.0)]
    dyn_runs = [[WindowRiskRecord(0, 12, 12, 2.0)] for _ in range(3)]
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=dyn_runs)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 12,
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


# ---------- Stage B typed actionability (PLAN_RF_UNI_CTRL.md Task B4) ----------


_CANONICAL = tuple(range(1, _VOCAB_SIZE))  # AA token ids (mask id = 0)


class TypedStubScorer:
    """Scorer stub that returns one score per record and logs batch labels.

    Unlike :class:`StubScorer` this returns ``len(records)`` scores so the
    Stage B envelope path (one batched call with ``env_ensemble_size`` records)
    works; ``calls`` records the label list of every batch so a test can assert
    exactly ``env_ensemble_size`` envelope head sequences per refresh.
    """

    def __init__(self, *, static_windows, dyn_windows):
        self._static_windows = static_windows
        self._dyn_windows = dyn_windows
        self.calls: list[list[str]] = []

    def score_batch_same_protein(self, *, protein_id, records):
        self.calls.append([label for label, _ in records])
        return BatchHeadScores(
            scores=tuple(
                _make_head_score(protein_id, seq, self._dyn_windows)
                for _, seq in records
            )
        )

    def get_or_compute_static(self, protein_id, sequence):
        return _make_head_score(protein_id, sequence, self._static_windows)


def _tiled_windows(length: int, width: int, z_by_start: dict[int, float]):
    """Tile ``[0, length)`` with width-``width`` windows; z from ``z_by_start``."""
    out = []
    for start in range(0, length - width + 1, width):
        out.append(WindowRiskRecord(start, start + width, width, z_by_start.get(start, 0.0)))
    return out


def _typed_config():
    base = _make_controller_config(t_start=0.5, refresh_interval=5, min_completion_fraction=0.0)
    return replace(base, targeting=TargetingConfig(mode="typed_actionability"))


def test_static_excess_mode_skips_typed_field_and_envelope():
    # z_static == z_dyn so the legacy excess path selects nothing; this also
    # proves static mode never computes the typed field or calls the envelope.
    z = {15: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = StubScorer(static_windows=static, dyn_windows_per_refresh=[dyn])
    cfg = _make_controller_config(t_start=0.5, refresh_interval=5, min_completion_fraction=0.0)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    record = controller.step(
        _make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6)
    ).refresh_record
    assert record.active_blocks == ()
    assert controller.actionability_states() == []
    # only the argmax dyn scoring call — no envelope probing in static mode
    assert scorer.dyn_call_count == 1


def test_typed_actionability_selects_from_v_target_and_aggregates_pressure():
    # One isolated focal window at residues [15, 18) with high risk; z_static
    # equals z_dyn so the legacy path would select nothing. Typed targeting
    # selects it from v_target.
    z = {15: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=_typed_config(), decode_tokens=_decode_tokens,
        canonical_token_ids=_CANONICAL,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    record = controller.step(
        _make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6)
    ).refresh_record

    # 1) typed selection produced a block covering the focal residue 16.
    covered = {
        i
        for blk in record.active_blocks
        for i in range(blk.residue_start_0b, blk.residue_end_0b)
    }
    assert 16 in covered

    state = controller.actionability_states()[-1]
    # 2) the focal residue is visible to targeting and was selected.
    assert state.v_target[16] > 0.0
    assert bool(state.active_target_flag[16]) is True
    # 3) G(t) is the length-normalized mass of u_pressure.
    assert np.isclose(state.G, global_pressure_mass(state.u_pressure))
    # 4) cluster support reduces the isolated focal pressure below v_target,
    #    yet selection (which uses v_target) still kept it — proving the split.
    assert state.u_pressure[16] < state.v_target[16]
    # 5) envelope probing costs exactly env_ensemble_size head sequences, not
    #    num_seed_windows * env_ensemble_size.
    env_calls = [labels for labels in scorer.calls if labels and labels[0].startswith("env_")]
    assert len(env_calls) == 1
    assert len(env_calls[0]) == controller.config.targeting.env_ensemble_size


def test_typed_selection_records_pre_cap_actionable_window_count():
    # Many positive-actionability windows but a small max_windows cap: the
    # pre-cap count must reveal the saturation that num_active_windows (post-cap)
    # hides (PLAN_RF_UNI_CTRL.md §"Background Reference").
    z = {15: 5.0, 18: 5.0, 21: 5.0, 24: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    cfg = replace(
        _make_controller_config(
            t_start=0.5, refresh_interval=5, min_completion_fraction=0.0, max_windows=2
        ),
        targeting=TargetingConfig(mode="typed_actionability"),
    )
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=cfg, decode_tokens=_decode_tokens,
        canonical_token_ids=_CANONICAL,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    controller.step(_make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6))
    state = controller.actionability_states()[-1]
    # >= 4 windows carry positive actionability, but only max_windows=2 survive.
    assert state.num_active_windows == 2
    assert state.num_actionable_windows_pre_cap >= 4
    assert state.num_actionable_windows_pre_cap > state.num_active_windows


def test_actionability_carries_legacy_residue_excess_distinct_from_b_cur():
    # z_static == z_dyn → legacy window_excess = 0 everywhere (per-window
    # subtraction), but b_cur (excess over the static MEDIAN) is positive at the
    # high-z window. This divergence is why the D2 in-block confound metric must
    # use legacy_residue_excess, not b_cur (PLAN_RF_UNI_CTRL.md §B4.4).
    z = {15: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=_typed_config(), decode_tokens=_decode_tokens,
        canonical_token_ids=_CANONICAL,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    controller.step(_make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6))
    state = controller.actionability_states()[-1]
    # legacy per-window-subtraction excess is 0 everywhere here ...
    np.testing.assert_allclose(state.legacy_residue_excess, 0.0)
    # ... yet b_cur is positive at the focal window (z_dyn >> static median).
    assert state.b_cur[16] > 0.0
    # active_block_id marks the focal block and is -1 outside any block.
    assert state.active_block_id[16] >= 0
    assert state.active_block_id[0] == -1


def test_envelope_short_circuits_to_one_head_call_when_no_masked_residue():
    # When the seed-window union is fully committed there is nothing to
    # resample; the K_env envelope completions would be identical, so the
    # controller should issue a single head call (P3 efficiency).
    z = {15: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=_typed_config(), decode_tokens=_decode_tokens,
        canonical_token_ids=_CANONICAL,
    )
    # Fully committed sequence (token 1 != mask id 0) → no masked residues.
    x_t = torch.full((30,), 1, dtype=torch.long)
    controller.step(_make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6))
    state = controller.actionability_states()[-1]
    assert state.num_env_head_calls == 1
    env_calls = [labels for labels in scorer.calls if labels and labels[0].startswith("env_")]
    assert len(env_calls) == 1 and len(env_calls[0]) == 1


# ---------------------------------------------------------------------------
# Task C1.4: Stage C.1 trajectory-level global pressure state + telemetry
# ---------------------------------------------------------------------------


class _GVaryingTypedScorer:
    """Typed scorer returning a different dyn-window set per refresh.

    Advances to the next per-refresh dyn windows on each ``dyn_argmax`` batch so
    the per-refresh ``G`` varies across refreshes; the envelope (``env_*``) and
    static calls reuse the current refresh's dyn windows. Lets C1.4 exercise the
    trajectory-median ``B_GR`` (median, not latest spike).
    """

    def __init__(self, *, static_windows, dyn_windows_per_refresh):
        self._static_windows = static_windows
        self._dyn_per_refresh = list(dyn_windows_per_refresh)
        self._idx = -1
        self._current = self._dyn_per_refresh[0]

    def score_batch_same_protein(self, *, protein_id, records):
        labels = [label for label, _ in records]
        if labels and labels[0] == "dyn_argmax":
            self._idx += 1
            self._current = self._dyn_per_refresh[self._idx]
        return BatchHeadScores(
            scores=tuple(
                _make_head_score(protein_id, seq, self._current) for _, seq in records
            )
        )

    def get_or_compute_static(self, protein_id, sequence):
        return _make_head_score(protein_id, sequence, self._static_windows)


def _pressure_config(
    *,
    enabled=True,
    B_low=0.0,
    B_high=1.0,
    g_min=0.0,
    g_max=1.0,
    tau_prom=0.0,
    min_reliable_refreshes=1,
    unready_g=0.0,
    scale_beta=True,
    scale_lambda=True,
):
    base = _make_controller_config(
        t_start=0.5, refresh_interval=5, min_completion_fraction=0.0
    )
    gp = GlobalPressureConfig(
        enabled=enabled,
        g_min=g_min,
        g_max=g_max,
        # tau_prom=0 → G_step == mean(u_pressure), so the median/saturation tests
        # keep using state.G directly (the thresholding effect is covered
        # separately by the pure-function + thresholding controller tests).
        tau_prom=tau_prom,
        B_low=B_low,
        B_high=B_high,
        min_reliable_refreshes=min_reliable_refreshes,
        unready_g=unready_g,
        scale_beta=scale_beta,
        scale_lambda=scale_lambda,
    )
    return replace(
        base,
        targeting=TargetingConfig(mode="typed_actionability"),
        global_pressure=gp,
    )


def _drive_pressure_refreshes(controller, *, length=30, n_refreshes=3):
    x_t = torch.full((length,), _MASK_ID, dtype=torch.long)
    for k in range(1, n_refreshes + 1):
        controller.step(
            _make_context(
                x_t=x_t, logits=_sharp_logits(length), step=5 * k, t=0.6
            )
        )
    return controller.actionability_states()


def test_pressure_disabled_leaves_actionability_telemetry_unset():
    # global_pressure.enabled=false (Stage B) must not populate any Stage C.1
    # pressure telemetry: byte-identical to pre-C1 behavior.
    z = {15: 5.0}
    static = _tiled_windows(30, 3, z)
    dyn = _tiled_windows(30, 3, z)
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer, config=_typed_config(), decode_tokens=_decode_tokens,
        canonical_token_ids=_CANONICAL,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    controller.step(_make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6))
    state = controller.actionability_states()[-1]
    assert state.B_GR is None
    assert state.g_GR_effective is None
    assert state.pressure_burden_bin is None
    assert state.beta_eff is None
    assert state.lambda_eff is None


def test_pressure_tau_prom_thresholds_g_below_unthresholded_mean():
    # G3 / RAR 0006: with tau_prom > 0 the controller G is the thresholded mass,
    # strictly below the un-thresholded mean (G_step_mean) when any residue has
    # positive u_pressure. Proves the thresholded driver flows into the field.
    static = _tiled_windows(30, 3, {15: 0.0})
    dyn = _tiled_windows(30, 3, {15: 8.0})
    scorer = TypedStubScorer(static_windows=static, dyn_windows=dyn)
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer,
        config=_pressure_config(B_low=0.0, B_high=1.0, tau_prom=0.05),
        decode_tokens=_decode_tokens, canonical_token_ids=_CANONICAL,
    )
    x_t = torch.full((30,), _MASK_ID, dtype=torch.long)
    controller.step(_make_context(x_t=x_t, logits=_sharp_logits(30), step=5, t=0.6))
    state = controller.actionability_states()[-1]
    assert state.G_step_mean > 0.0
    assert state.G < state.G_step_mean  # thresholding strictly reduces the mass
    assert np.isclose(
        state.G, prominence_thresholded_mass(state.u_pressure, tau_prom=0.05)
    )


def test_pressure_b_gr_is_trajectory_median_not_latest_spike():
    # Three refreshes with increasing focal risk → increasing per-refresh G;
    # B_GR must be the median (middle), not the latest (max) spike.
    static = _tiled_windows(30, 3, {15: 0.0})
    dyn_runs = [
        _tiled_windows(30, 3, {15: 1.0}),
        _tiled_windows(30, 3, {15: 3.0}),
        _tiled_windows(30, 3, {15: 9.0}),
    ]
    scorer = _GVaryingTypedScorer(
        static_windows=static, dyn_windows_per_refresh=dyn_runs
    )
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer,
        config=_pressure_config(B_low=0.0, B_high=2.0),
        decode_tokens=_decode_tokens, canonical_token_ids=_CANONICAL,
    )
    states = _drive_pressure_refreshes(controller, n_refreshes=3)
    g_values = [float(s.G) for s in states]
    # Increasing focal risk → strictly increasing per-refresh G (latest is max).
    assert g_values[-1] == max(g_values)
    assert len(set(g_values)) == 3
    expected_B_GR = protein_pressure_burden(g_values)
    expected_g = smoothstep_pressure(
        expected_B_GR, B_low=0.0, B_high=2.0, g_min=0.0, g_max=1.0
    )
    assert np.isclose(states[-1].B_GR, expected_B_GR)
    assert np.isclose(states[-1].B_GR, np.median(g_values))
    assert states[-1].B_GR < g_values[-1]  # median < latest spike
    assert np.isclose(states[-1].g_GR_effective, expected_g)


def test_pressure_scales_beta_and_lambda_with_independent_gates():
    static = _tiled_windows(30, 3, {15: 0.0})
    dyn_runs = [_tiled_windows(30, 3, {15: 4.0})] * 2
    scorer = _GVaryingTypedScorer(
        static_windows=static, dyn_windows_per_refresh=dyn_runs
    )
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer,
        config=_pressure_config(B_low=0.0, B_high=1.0, scale_beta=True, scale_lambda=False),
        decode_tokens=_decode_tokens, canonical_token_ids=_CANONICAL,
    )
    states = _drive_pressure_refreshes(controller, n_refreshes=2)
    s = states[-1]
    g = float(s.g_GR_effective)
    assert s.beta_base == 1.0 and s.lambda_base == 1.0
    assert np.isclose(s.beta_eff, 1.0 * g)        # scale_beta=True
    assert np.isclose(s.lambda_eff, 1.0)          # scale_lambda=False → unscaled


def test_pressure_unready_uses_unready_g_until_min_reliable_refreshes():
    # min_reliable_refreshes=2: the first refresh is below the reliability count,
    # so g_GR falls back to unready_g and B_GR stays None.
    static = _tiled_windows(30, 3, {15: 0.0})
    dyn_runs = [_tiled_windows(30, 3, {15: 4.0})] * 2
    scorer = _GVaryingTypedScorer(
        static_windows=static, dyn_windows_per_refresh=dyn_runs
    )
    controller = D1MonitorController(
        protein_id="P1", design_idx=0, seed=42, static_sequence="A" * 30,
        scorer=scorer,
        config=_pressure_config(
            B_low=0.0, B_high=1.0, min_reliable_refreshes=2, unready_g=0.0
        ),
        decode_tokens=_decode_tokens, canonical_token_ids=_CANONICAL,
    )
    states = _drive_pressure_refreshes(controller, n_refreshes=2)
    first, second = states[0], states[1]
    assert first.B_GR is None
    assert first.g_GR_effective == 0.0       # unready_g
    assert first.beta_eff == 0.0             # beta * unready_g
    # Second refresh reaches the reliability count → real B_GR + g_GR.
    assert second.B_GR is not None
    assert second.pressure_burden_bin in {"low", "mid", "high"}
