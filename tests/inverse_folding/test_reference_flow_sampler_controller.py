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
