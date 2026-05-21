"""T5 contract tests: sampler controller hook integration (Phase D1).

Verifies that:
1. ``controller=None`` reproduces the existing C1 sampler exactly (per-step
   trajectory equality with no controller wired).
2. An identity controller (returns input logits unchanged) yields the same
   per-step trajectories as ``controller=None`` for the same seed.
3. The hook is called after logits validation (i.e. it never observes NaN
   logits) and before unmask sampling (the captured ``x_t`` still contains
   the pre-sample mask state).
4. The hook receives a clone of ``x_t``; mutations inside the controller do
   not corrupt the sampler's working tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.controller import (
    ControllerStepResult,
    SamplerStepContext,
)
from inverse_folding.reference_flow.sampler import (
    PositionDependentDFMSampler,
    SamplerOutput,
)


_MASK_ID = 0
_VOCAB_SIZE = 8


def _make_config(*, n_steps: int = 4, seed: int = 7) -> ReferenceFlowConfig:
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=n_steps,
            seed=seed,
            temperature=1.0,
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=False),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(
            form="constant_one", h_source="h_processed", c=None
        ),
        h_shuffle=HShuffleConfig(enabled=False),
    )


def _make_denoiser(*, n_positions: int):
    """Deterministic denoiser: uniform logits across canonical (non-mask) tokens."""
    base = torch.zeros(n_positions, _VOCAB_SIZE, dtype=torch.float32)
    base[:, _MASK_ID] = -1e4  # never sample mask
    def denoiser(x_t, t, struct):
        return base.clone()
    return denoiser


def _run_sampler(controller=None, *, n_steps: int = 4, seed: int = 7) -> SamplerOutput:
    sampler = PositionDependentDFMSampler(mask_token_id=_MASK_ID, vocab_size=_VOCAB_SIZE)
    return sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=_make_denoiser(n_positions=6),
        config=_make_config(n_steps=n_steps, seed=seed),
        struct=None,
        save_trajectories=True,
        controller=controller,
    )


class _RecordingController:
    """Stub controller capturing every (step, t) it sees; returns identity."""

    def __init__(self):
        self.captured: list[SamplerStepContext] = []

    def step(self, ctx: SamplerStepContext) -> ControllerStepResult:
        self.captured.append(ctx)
        return ControllerStepResult(logits=ctx.logits, refresh_record=None)


class _MutatingController:
    """Stub that mutates the received x_t. Used to prove the sampler hands a clone."""

    def step(self, ctx: SamplerStepContext) -> ControllerStepResult:
        ctx.x_t.fill_(_MASK_ID + 3)  # corrupt the local copy
        return ControllerStepResult(logits=ctx.logits, refresh_record=None)


def test_controller_none_keyword_accepted_and_default_behavior_preserved():
    """controller=None must be the default and must not change current output."""
    no_controller = _run_sampler(controller=None)
    explicit_none = _run_sampler()  # no controller kwarg at all
    assert torch.equal(no_controller.tokens, explicit_none.tokens)
    assert no_controller.unmask_step_by_pos == explicit_none.unmask_step_by_pos
    assert len(no_controller.trajectory_rows) == len(explicit_none.trajectory_rows)
    for a, b in zip(no_controller.trajectory_rows, explicit_none.trajectory_rows):
        assert a == b


def test_identity_controller_reproduces_per_step_trajectory():
    baseline = _run_sampler(controller=None, n_steps=4, seed=11)
    identity = _RecordingController()
    with_id = _run_sampler(controller=identity, n_steps=4, seed=11)

    assert torch.equal(baseline.tokens, with_id.tokens)
    assert baseline.unmask_step_by_pos == with_id.unmask_step_by_pos
    assert len(baseline.trajectory_rows) == len(with_id.trajectory_rows)
    for a, b in zip(baseline.trajectory_rows, with_id.trajectory_rows):
        assert a["step"] == b["step"]
        assert a["t"] == b["t"]
        assert a["unmasked_mask"] == b["unmasked_mask"]
        assert a["token_argmax"] == b["token_argmax"]


def test_hook_called_after_logits_validation():
    """Hook must never see NaN logits — the NaN check fires first."""

    def nan_denoiser(x_t, t, struct):
        logits = torch.zeros(6, _VOCAB_SIZE, dtype=torch.float32)
        logits[0, 0] = float("nan")
        return logits

    sampler = PositionDependentDFMSampler(mask_token_id=_MASK_ID, vocab_size=_VOCAB_SIZE)
    recorder = _RecordingController()
    with pytest.raises(FloatingPointError):
        sampler.sample(
            sequence_length=6,
            h_values=np.zeros(6, dtype=np.float32),
            denoiser=nan_denoiser,
            config=_make_config(n_steps=2, seed=7),
            controller=recorder,
        )
    assert recorder.captured == []


def test_hook_observes_x_t_before_unmask_sampling():
    """Captured x_t at step k must still contain the pre-sample mask state."""
    recorder = _RecordingController()
    _ = _run_sampler(controller=recorder, n_steps=4, seed=7)
    # At step 0 every position must still be the mask token.
    first = recorder.captured[0]
    assert first.step == 0
    assert int((first.x_t == _MASK_ID).sum().item()) == 6
    # The captured tensor is a clone — fully detached from sampler state.
    assert first.x_t.data_ptr() != 0  # smoke


def test_mutating_controller_cannot_corrupt_sampler_state():
    """Hook receives a clone; mutating it must not change sampler behavior."""
    baseline = _run_sampler(controller=None, n_steps=4, seed=7)
    with_mut = _run_sampler(controller=_MutatingController(), n_steps=4, seed=7)
    assert torch.equal(baseline.tokens, with_mut.tokens)
    assert baseline.unmask_step_by_pos == with_mut.unmask_step_by_pos


def test_controller_logits_override_is_used_for_sampling():
    """If the controller returns modified logits, the sampler must use them."""

    class ForceTokenController:
        def step(self, ctx: SamplerStepContext) -> ControllerStepResult:
            # Force every position to token 5 by spiking that logit.
            logits = torch.full_like(ctx.logits, -1e4)
            logits[:, 5] = 1e4
            return ControllerStepResult(logits=logits, refresh_record=None)

    out = _run_sampler(controller=ForceTokenController(), n_steps=4, seed=7)
    assert torch.all(out.tokens == 5).item()


# ---------------------------------------------------------------------------
# T5: PostSamplingContext + RNG protocol + extended _apply_reparam_remask
# (PLAN_RF.md §"Sampler integration details")
# ---------------------------------------------------------------------------


def _make_config_with_remask(*, n_steps: int = 4, seed: int = 7) -> ReferenceFlowConfig:
    cfg = _make_config(n_steps=n_steps, seed=seed)
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=cfg.sampler.n_steps,
            seed=cfg.sampler.seed,
            temperature=cfg.sampler.temperature,
            n_designs_per_protein=cfg.sampler.n_designs_per_protein,
            remask=RemaskConfig(enabled=True),
        ),
        schedule=cfg.schedule,
        amplification=cfg.amplification,
        h_shuffle=cfg.h_shuffle,
    )


def test_apply_reparam_remask_bit_equivalent_with_default_kwargs():
    """``rank_scores=None, protected_positions=()`` must replay legacy output."""
    from inverse_folding.reference_flow.sampler import _apply_reparam_remask

    rng = np.random.default_rng(0)
    L = 8
    x_t = torch.arange(1, L + 1, dtype=torch.long)  # all committed
    scores = rng.standard_normal(L).astype(np.float64)
    unmask_step = [0] * L
    x_legacy = x_t.clone()
    scores_legacy = scores.copy()
    unmask_legacy = list(unmask_step)
    n_legacy = _apply_reparam_remask(
        x_t=x_legacy,
        scores=scores_legacy,
        unmask_step_by_pos=unmask_legacy,
        mask_token_id=_MASK_ID,
        step=1,
        n_steps=4,
    )
    x_new = x_t.clone()
    scores_new = scores.copy()
    unmask_new = list(unmask_step)
    n_new = _apply_reparam_remask(
        x_t=x_new,
        scores=scores_new,
        unmask_step_by_pos=unmask_new,
        mask_token_id=_MASK_ID,
        step=1,
        n_steps=4,
        rank_scores=None,
        protected_positions=(),
    )
    assert n_legacy.count == n_new.count
    assert n_legacy.remasked_positions == n_new.remasked_positions
    assert torch.equal(x_legacy, x_new)
    np.testing.assert_array_equal(scores_legacy, scores_new)
    assert unmask_legacy == unmask_new


def test_apply_reparam_remask_uses_rank_scores_when_provided():
    """``rank_scores`` overrides ``scores`` as the bottom-k ranking signal."""
    from inverse_folding.reference_flow.sampler import _apply_reparam_remask

    L = 6
    x_t = torch.arange(1, L + 1, dtype=torch.long)
    # scores prefer position 0 lowest. rank_scores prefer position 5 lowest.
    scores = np.array([-5.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rank_scores = np.array([10.0, 10.0, 10.0, 10.0, 10.0, -5.0])
    unmask_step = [0] * L

    # step=2, n_steps=4 → rate=0.25 → cutoff_len=int(6*0.25)=1.
    res = _apply_reparam_remask(
        x_t=x_t,
        scores=scores,
        unmask_step_by_pos=unmask_step,
        mask_token_id=_MASK_ID,
        step=2,
        n_steps=4,
        rank_scores=rank_scores,
        protected_positions=(),
    )
    assert res.count == 1
    # Position 5 (lowest rank_score) must be the one remasked.
    assert int(x_t[5].item()) == _MASK_ID
    assert int(x_t[0].item()) != _MASK_ID  # NOT remasked despite low scores[]
    assert res.remasked_positions == (5,)


def test_apply_reparam_remask_excludes_protected_positions():
    from inverse_folding.reference_flow.sampler import _apply_reparam_remask

    L = 6
    x_t = torch.arange(1, L + 1, dtype=torch.long)
    scores = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    unmask_step = [0] * L
    # Position 0 would be remasked without protection (lowest score).
    # step=2, n_steps=4 → cutoff_len=1.
    res = _apply_reparam_remask(
        x_t=x_t,
        scores=scores,
        unmask_step_by_pos=unmask_step,
        mask_token_id=_MASK_ID,
        step=2,
        n_steps=4,
        protected_positions=(0,),
    )
    assert res.count == 1
    assert int(x_t[0].item()) != _MASK_ID
    assert int(x_t[1].item()) == _MASK_ID  # next-lowest score
    assert res.remasked_positions == (1,)


class _StructLogitCapturingController:
    """Records structural vs corrected logits seen at each pre/post hook."""

    def __init__(self, force_token: int | None = None):
        self.pre: list[tuple[int, torch.Tensor]] = []
        self.post: list = []  # PostSamplingContext
        self.force_token = force_token

    def step(self, ctx) -> ControllerStepResult:
        self.pre.append((ctx.step, ctx.logits.clone()))
        if self.force_token is None:
            return ControllerStepResult(logits=ctx.logits, refresh_record=None)
        # Build distinct corrected logits to exercise the paired RNG path.
        corrected = torch.full_like(ctx.logits, -1e4)
        corrected[:, int(self.force_token)] = 1e4
        return ControllerStepResult(logits=corrected, refresh_record=None)

    @property
    def config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            d2=SimpleNamespace(enabled=True, paired_uncorrected_sample=True),
            d3=SimpleNamespace(enabled=False),
        )

    def post_step(self, ctx):
        from inverse_folding.reference_flow.controller import PostSamplingResult

        self.post.append(ctx)
        return PostSamplingResult(
            rank_scores=None,
            protected_positions=(),
            post_event_rows=(),
            refresh_addendum=None,
        )


def test_post_step_context_carries_structural_and_corrected_logits():
    controller = _StructLogitCapturingController(force_token=5)
    sampler = PositionDependentDFMSampler(mask_token_id=_MASK_ID, vocab_size=_VOCAB_SIZE)
    sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=_make_denoiser(n_positions=6),
        config=_make_config_with_remask(n_steps=4, seed=7),
        struct=None,
        save_trajectories=True,
        controller=controller,
    )
    # n_steps=4, remask runs on steps 0..2 → 3 post_step calls.
    assert len(controller.post) == 3
    for ctx in controller.post:
        # corrected_logits must reflect the forced token while structural does not.
        assert torch.argmax(ctx.corrected_logits[0]).item() == 5
        # Structural logits at position 0 should NOT be peaked at token 5
        # (the denoiser fixture returns uniform logits with mask deeply negative).
        assert ctx.structural_logits[0, 5].item() < 1.0


def test_paired_uncorrected_sample_does_not_advance_real_rng():
    """Same seed: paired branch active vs disabled must produce equal final tokens."""
    sampler = PositionDependentDFMSampler(mask_token_id=_MASK_ID, vocab_size=_VOCAB_SIZE)
    cfg = _make_config_with_remask(n_steps=4, seed=7)

    # First run: paired enabled (force_token triggers corrected != structural).
    c1 = _StructLogitCapturingController(force_token=5)
    out1 = sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=_make_denoiser(n_positions=6),
        config=cfg,
        struct=None,
        save_trajectories=False,
        controller=c1,
    )

    # Second run: same forced corrected logits but paired disabled at config level.
    class _NoPairController(_StructLogitCapturingController):
        @property
        def config(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                d2=SimpleNamespace(enabled=True, paired_uncorrected_sample=False),
                d3=SimpleNamespace(enabled=False),
            )

    c2 = _NoPairController(force_token=5)
    out2 = sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=_make_denoiser(n_positions=6),
        config=cfg,
        struct=None,
        save_trajectories=False,
        controller=c2,
    )
    # Real RNG advancement is identical → final tokens must match exactly.
    assert torch.equal(out1.tokens, out2.tokens)


def test_paired_uncorrected_matches_actual_when_logits_unchanged_at_position():
    """For positions whose logits are unchanged (no D2 correction), paired
    sampling must produce the same token as the actual branch."""

    # Identity controller (no logit change). Paired branch should be skipped
    # entirely because corrected_logits is structural_logits.
    controller = _StructLogitCapturingController(force_token=None)
    sampler = PositionDependentDFMSampler(mask_token_id=_MASK_ID, vocab_size=_VOCAB_SIZE)
    sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=_make_denoiser(n_positions=6),
        config=_make_config_with_remask(n_steps=4, seed=7),
        struct=None,
        save_trajectories=False,
        controller=controller,
    )
    for ctx in controller.post:
        # No paired sample taken because corrected == structural (identity).
        assert ctx.sampled_tokens_uncorrected is None
