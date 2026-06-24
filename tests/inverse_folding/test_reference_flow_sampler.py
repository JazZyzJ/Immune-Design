"""Phase C1 sampler contract tests."""

from __future__ import annotations

import numpy as np
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.sampler import (
    PositionDependentDFMSampler,
    SamplerBatchLane,
)


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
    assert set(out.trajectory_rows[0].keys()) == {
        "step",
        "t",
        "unmasked_mask",
        "token_argmax",
        "remasked_count",
    }


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


def _baseline_cfg(*, n_steps: int = 8, remask_enabled: bool = False) -> ReferenceFlowConfig:
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=n_steps,
            seed=7,
            temperature=1.0,
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=remask_enabled),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def test_remask_disabled_matches_legacy_behavior():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _baseline_cfg(remask_enabled=False)
    h = np.zeros(12, dtype=np.float32)
    out_off = sampler.sample(
        sequence_length=12,
        h_values=h,
        denoiser=uniform_denoiser,
        config=cfg,
        save_trajectories=True,
    )
    assert all(row["remasked_count"] == 0 for row in out_off.trajectory_rows)


def biased_denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    """Per-position deterministic logits.

    Position 0 has a strong preference for token 0 (high confidence),
    position 1 has a near-uniform distribution over the non-mask tokens
    (low confidence). Positions ≥ 2 mirror position 0.
    """
    L = x_t.shape[0]
    logits = torch.zeros((L, VOCAB_SIZE), dtype=torch.float32)
    logits[:, MASK_ID] = -1e9
    # high-confidence positions
    logits[:, 0] = 5.0
    # low-confidence position (index 1)
    logits[1, 0] = 0.0
    return logits


def test_remask_reaches_all_positions_with_low_confidence_committed():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _baseline_cfg(n_steps=10, remask_enabled=True)
    out = sampler.sample(
        sequence_length=8,
        h_values=np.zeros(8, dtype=np.float32),
        denoiser=biased_denoiser,
        config=cfg,
        save_trajectories=True,
    )
    # Final tokens are mask-free.
    assert MASK_ID not in out.tokens.tolist()
    # At least one intermediate step must have remasked some positions.
    total_remasked = sum(row["remasked_count"] for row in out.trajectory_rows)
    assert total_remasked > 0
    # The final step never remasks (so remasked_count[-1] == 0).
    assert out.trajectory_rows[-1]["remasked_count"] == 0


def test_remask_preferentially_targets_low_confidence_positions():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _baseline_cfg(n_steps=12, remask_enabled=True)
    out = sampler.sample(
        sequence_length=8,
        h_values=np.zeros(8, dtype=np.float32),
        denoiser=biased_denoiser,
        config=cfg,
    )
    # Position 1 has the lowest score → expected to be re-masked at least once,
    # so its final unmask step should be later (on average) than the high-
    # confidence positions. We only need to assert that position 1 is not
    # uniformly the earliest committed position.
    steps = np.asarray(out.unmask_step_by_pos)
    other = np.delete(steps, 1)
    assert steps[1] >= other.min()


def test_remask_is_deterministic_for_same_seed():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _baseline_cfg(n_steps=12, remask_enabled=True)
    out1 = sampler.sample(
        sequence_length=8,
        h_values=np.zeros(8, dtype=np.float32),
        denoiser=biased_denoiser,
        config=cfg,
    )
    out2 = sampler.sample(
        sequence_length=8,
        h_values=np.zeros(8, dtype=np.float32),
        denoiser=biased_denoiser,
        config=cfg,
    )
    assert torch.equal(out1.tokens, out2.tokens)
    assert out1.unmask_step_by_pos == out2.unmask_step_by_pos


def test_batched_sampler_matches_scalar_lanes_with_independent_rngs():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg0 = _baseline_cfg(n_steps=10, remask_enabled=True)
    cfg1 = ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=10,
            seed=31,
            temperature=1.0,
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=True),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )
    h0 = np.zeros(8, dtype=np.float32)
    h1 = np.zeros(5, dtype=np.float32)

    scalar0 = sampler.sample(
        sequence_length=8,
        h_values=h0,
        denoiser=biased_denoiser,
        config=cfg0,
        save_trajectories=True,
        protein_id="p0",
        design_idx=0,
    )
    scalar1 = sampler.sample(
        sequence_length=5,
        h_values=h1,
        denoiser=biased_denoiser,
        config=cfg1,
        save_trajectories=True,
        protein_id="p1",
        design_idx=0,
    )

    batched_calls: list[list[list[int]]] = []

    def batched_denoiser(x_ts, t, structs):
        del t, structs
        batched_calls.append([x.tolist() for x in x_ts])
        return [biased_denoiser(x_t, 0.0, None) for x_t in x_ts]

    batched = sampler.sample_batch(
        lanes=[
            SamplerBatchLane(
                sequence_length=8,
                h_values=h0,
                config=cfg0,
                protein_id="p0",
                design_idx=0,
            ),
            SamplerBatchLane(
                sequence_length=5,
                h_values=h1,
                config=cfg1,
                protein_id="p1",
                design_idx=0,
            ),
        ],
        batched_denoiser=batched_denoiser,
        save_trajectories=True,
    )

    assert len(batched_calls) == 10
    assert torch.equal(batched[0].tokens, scalar0.tokens)
    assert torch.equal(batched[1].tokens, scalar1.tokens)
    assert batched[0].unmask_step_by_pos == scalar0.unmask_step_by_pos
    assert batched[1].unmask_step_by_pos == scalar1.unmask_step_by_pos
    assert batched[0].trajectory_rows == scalar0.trajectory_rows
    assert batched[1].trajectory_rows == scalar1.trajectory_rows


def test_remask_config_round_trips_through_yaml(tmp_path):
    from inverse_folding.reference_flow.config import (
        load_reference_flow_config,
        reference_flow_config_to_dict,
    )

    cfg_path = tmp_path / "remask.yaml"
    cfg_path.write_text(
        "sampler:\n"
        "  n_steps: 16\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "  remask:\n"
        "    enabled: true\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: constant_one\n"
        "  h_source: h_processed\n"
    )
    cfg = load_reference_flow_config(cfg_path)
    assert cfg.sampler.remask.enabled is True
    payload = reference_flow_config_to_dict(cfg)
    assert payload["sampler"]["remask"]["enabled"] is True


def test_remask_defaults_to_disabled_when_absent_in_yaml(tmp_path):
    from inverse_folding.reference_flow.config import load_reference_flow_config

    cfg_path = tmp_path / "no_remask.yaml"
    cfg_path.write_text(
        "sampler:\n"
        "  n_steps: 8\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: constant_one\n"
        "  h_source: h_processed\n"
    )
    cfg = load_reference_flow_config(cfg_path)
    assert cfg.sampler.remask.enabled is False


# ---- remask.fraction_scale (clean no-remask probe substrate) ----


def _remask_unit_inputs():
    """Eight committed (non-mask) positions with spread-out scores."""
    x_t = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4], dtype=np.float64)
    unmask_step_by_pos = [0] * 8
    return x_t, scores, unmask_step_by_pos


def test_apply_reparam_remask_cutoff_scale_one_matches_default():
    from inverse_folding.reference_flow.sampler import _apply_reparam_remask

    x1, s1, u1 = _remask_unit_inputs()
    r_default = _apply_reparam_remask(
        x_t=x1.clone(), scores=s1.copy(), unmask_step_by_pos=list(u1),
        mask_token_id=MASK_ID, step=2, n_steps=10,
    )
    x2, s2, u2 = _remask_unit_inputs()
    r_scaled = _apply_reparam_remask(
        x_t=x2.clone(), scores=s2.copy(), unmask_step_by_pos=list(u2),
        mask_token_id=MASK_ID, step=2, n_steps=10, cutoff_scale=1.0,
    )
    # The default (1.0) path must be byte-identical to the legacy unscaled path.
    assert r_default.count == r_scaled.count
    assert r_default.remasked_positions == r_scaled.remasked_positions
    assert r_default.count > 0  # sanity: remask actually fired in this setup


def test_apply_reparam_remask_cutoff_scale_zero_remasks_nothing():
    from inverse_folding.reference_flow.sampler import _apply_reparam_remask

    x, s, u = _remask_unit_inputs()
    r = _apply_reparam_remask(
        x_t=x.clone(), scores=s.copy(), unmask_step_by_pos=list(u),
        mask_token_id=MASK_ID, step=2, n_steps=10, cutoff_scale=0.0,
    )
    assert r.count == 0
    assert r.remasked_positions == ()


def test_remask_fraction_scale_zero_keeps_enabled_path_but_remasks_nothing():
    """enabled=True keeps the post_step/remask code path live; fraction_scale=0
    must remask zero positions (the clean no-remask probe substrate)."""
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=10, seed=7, temperature=1.0, n_designs_per_protein=1,
            remask=RemaskConfig(enabled=True, fraction_scale=0.0),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
        denoiser=biased_denoiser, config=cfg, save_trajectories=True,
    )
    assert MASK_ID not in out.tokens.tolist()
    assert all(row["remasked_count"] == 0 for row in out.trajectory_rows)


def test_remask_fraction_scale_defaults_to_one_when_absent(tmp_path):
    from inverse_folding.reference_flow.config import load_reference_flow_config

    cfg_path = tmp_path / "remask_default_scale.yaml"
    cfg_path.write_text(
        "sampler:\n  n_steps: 8\n  seed: 42\n  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n  remask:\n    enabled: true\n"
        "schedule:\n  base_form: linear\n"
        "amplification:\n  form: constant_one\n  h_source: h_processed\n"
    )
    cfg = load_reference_flow_config(cfg_path)
    assert cfg.sampler.remask.fraction_scale == 1.0


def test_remask_fraction_scale_round_trips_through_yaml(tmp_path):
    from inverse_folding.reference_flow.config import (
        load_reference_flow_config,
        reference_flow_config_to_dict,
    )

    cfg_path = tmp_path / "remask_scale.yaml"
    cfg_path.write_text(
        "sampler:\n  n_steps: 16\n  seed: 42\n  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n  remask:\n    enabled: true\n"
        "    fraction_scale: 0.0\n"
        "schedule:\n  base_form: linear\n"
        "amplification:\n  form: constant_one\n  h_source: h_processed\n"
    )
    cfg = load_reference_flow_config(cfg_path)
    assert cfg.sampler.remask.fraction_scale == 0.0
    payload = reference_flow_config_to_dict(cfg)
    assert payload["sampler"]["remask"]["fraction_scale"] == 0.0


def test_remask_fraction_scale_out_of_range_raises():
    import pytest

    from inverse_folding.reference_flow.config import (
        ReferenceFlowConfigError,
        materialize_reference_flow_config,
    )

    payload = {
        "sampler": {
            "n_steps": 8,
            "seed": 42,
            "remask": {"enabled": True, "fraction_scale": 1.5},
        },
        "schedule": {"base_form": "linear"},
        "amplification": {"form": "constant_one", "h_source": "h_processed"},
    }
    with pytest.raises(ReferenceFlowConfigError):
        materialize_reference_flow_config(payload)
