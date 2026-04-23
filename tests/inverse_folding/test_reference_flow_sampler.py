"""Phase C1 sampler contract tests."""

from __future__ import annotations

import numpy as np
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler


VOCAB_SIZE = 5
MASK_ID = 4


def uniform_denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    logits = torch.zeros((x_t.shape[0], VOCAB_SIZE), dtype=torch.float32)
    logits[:, MASK_ID] = -1e9
    return logits


def test_sampler_produces_mask_free_outputs_and_is_deterministic():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=8, seed=7, temperature=1.0, n_designs_per_protein=1),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )
    h = np.zeros(12, dtype=np.float32)
    out1 = sampler.sample(sequence_length=12, h_values=h, denoiser=uniform_denoiser, config=cfg)
    out2 = sampler.sample(sequence_length=12, h_values=h, denoiser=uniform_denoiser, config=cfg)
    assert len(out1.tokens) == 12
    assert MASK_ID not in out1.tokens.tolist()
    assert torch.equal(out1.tokens, out2.tokens)


def test_sampler_position_dependent_schedule_delays_high_g_positions():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=20, seed=13, temperature=1.0, n_designs_per_protein=1),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(
            form="linear_clamp",
            c=3.0,
            mu=0.0,
            h_source="h_processed",
        ),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )
    h = np.linspace(-1.0, 3.0, 20, dtype=np.float32)
    out = sampler.sample(sequence_length=20, h_values=h, denoiser=uniform_denoiser, config=cfg)
    steps = np.asarray(out.unmask_step_by_pos)
    g = np.asarray(out.g_values)
    lo = steps[g <= np.quantile(g, 0.25)].mean()
    hi = steps[g >= np.quantile(g, 0.75)].mean()
    assert hi > lo


def test_sampler_can_capture_trajectory():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=4, seed=5, temperature=1.0, n_designs_per_protein=1),
        schedule=ScheduleConfig(base_form="cosine"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )
    out = sampler.sample(
        sequence_length=6,
        h_values=np.zeros(6, dtype=np.float32),
        denoiser=uniform_denoiser,
        config=cfg,
        save_trajectories=True,
    )
    assert len(out.trajectory_rows) == 4
    assert set(out.trajectory_rows[0].keys()) == {"step", "t", "unmasked_mask", "token_argmax"}


def test_sampler_h_shuffle_can_vary_per_design_seed():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=8, seed=11, temperature=1.0, n_designs_per_protein=1),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(
            form="linear_clamp",
            c=2.0,
            mu=0.0,
            h_source="h_processed",
        ),
        h_shuffle=HShuffleConfig(enabled=True, seed=101),
    )
    h = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    out1 = sampler.sample(
        sequence_length=6,
        h_values=h,
        denoiser=uniform_denoiser,
        config=cfg,
        shuffle_seed=1001,
    )
    out2 = sampler.sample(
        sequence_length=6,
        h_values=h,
        denoiser=uniform_denoiser,
        config=cfg,
        shuffle_seed=2002,
    )
    assert out1.g_values != out2.g_values
    assert sorted(out1.g_values) == sorted(out2.g_values)
