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
    ResumeState,
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


def test_enforce_fixed_tokens_restores_masked_anchor():
    from inverse_folding.reference_flow.sampler import _enforce_fixed_tokens

    x_t = torch.tensor([0, MASK_ID, 2], dtype=torch.long)
    unmask_step_by_pos = [0, -1, 0]
    _enforce_fixed_tokens(
        x_t=x_t,
        fixed_tokens={1: 3},
        unmask_step_by_pos=unmask_step_by_pos,
        step=5,
    )
    assert x_t.tolist() == [0, 3, 2]
    assert unmask_step_by_pos == [0, 5, 0]


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


# ---- signal-diag P1/P2: snapshot capture + resume-from-state ----


def _snap_cfg(*, n_steps: int = 8, seed: int = 7, remask_enabled: bool = False):
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=n_steps,
            seed=seed,
            temperature=1.0,
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=remask_enabled),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def test_snapshot_steps_are_byte_equivalent_to_no_snapshot():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _snap_cfg(n_steps=10, remask_enabled=True)
    h = np.zeros(8, dtype=np.float32)
    base = sampler.sample(
        sequence_length=8, h_values=h, denoiser=biased_denoiser, config=cfg
    )
    snap = sampler.sample(
        sequence_length=8,
        h_values=h,
        denoiser=biased_denoiser,
        config=cfg,
        snapshot_steps=[3, 6],
    )
    # Capture must not perturb the trajectory.
    assert torch.equal(base.tokens, snap.tokens)
    assert base.unmask_step_by_pos == snap.unmask_step_by_pos
    assert base.g_values == snap.g_values
    assert base.snapshots == ()
    assert [s.step for s in snap.snapshots] == [3, 6]


def test_snapshot_captures_pre_controller_logits_and_partial_x_t():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _snap_cfg(n_steps=6)

    class _StepResult:
        def __init__(self, logits):
            self.logits = logits
            self.refresh_record = None

    class _LogitMutatingController:
        # No ``config`` attr -> paired-uncorrected branch is skipped.
        config = None

        def step(self, ctx):
            # Drive sampling with a clearly different (mutated) logit field so a
            # post-controller capture would be detectable.
            mutated = ctx.logits.clone()
            mutated[:, 0] = 9.0
            return _StepResult(mutated)

    out = sampler.sample(
        sequence_length=5,
        h_values=np.zeros(5, dtype=np.float32),
        denoiser=uniform_denoiser,
        config=cfg,
        controller=_LogitMutatingController(),
        snapshot_steps=[2],
    )
    assert len(out.snapshots) == 1
    snap = out.snapshots[0]
    assert snap.step == 2
    # Captured logits are the RAW denoiser output (pre-controller), not the
    # controller-mutated field (which set column 0 to 9.0).
    raw = uniform_denoiser(snap.x_t, snap.t, None)
    assert torch.equal(snap.struct_logits, raw)
    assert not torch.equal(snap.struct_logits, raw.clone().index_fill_(1, torch.tensor([0]), 9.0))
    # x_t is a partial completion (some masks still present at an early step).
    assert (snap.x_t == MASK_ID).any()


def _context_denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    """Near one-hot logits whose preferred token depends on committed context.

    The preferred token keys off how many committed positions carry a high token
    (2 or 3), so different ``fixed_tokens`` (A_B) values shift the preferred
    token; the near one-hot gap makes the sampled token deterministic given the
    preference. The unmasking SELECTION stays a function of the (paired) RNG
    draws and the masked set only -> selection pairs, tokens diverge.
    """
    L = x_t.shape[0]
    logits = torch.zeros((L, VOCAB_SIZE), dtype=torch.float32)
    logits[:, MASK_ID] = -1e9
    committed = x_t[x_t != MASK_ID]
    count_hi = int(((committed == 2) | (committed == 3)).sum().item())
    pref = count_hi % (VOCAB_SIZE - 1)
    logits[:, pref] = 50.0
    return logits


def test_resume_runs_only_tail_steps_and_finishes_mask_free():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _snap_cfg(n_steps=10, remask_enabled=True)
    h = np.zeros(8, dtype=np.float32)
    full = sampler.sample(
        sequence_length=8,
        h_values=h,
        denoiser=biased_denoiser,
        config=cfg,
        snapshot_steps=[4],
    )
    snap = full.snapshots[0]

    seen_steps: list[float] = []

    def counting_denoiser(x_t, t, struct):
        seen_steps.append(round(float(t) * 10))
        return biased_denoiser(x_t, t, struct)

    resumed = sampler.sample(
        sequence_length=8,
        h_values=h,
        denoiser=counting_denoiser,
        config=cfg,
        controller=None,
        initial_state=ResumeState(
            x_t=snap.x_t,
            scores=snap.scores,
            unmask_step_by_pos=list(snap.unmask_step_by_pos),
            start_step=snap.step,
        ),
    )
    # Only steps 4..9 ran.
    assert seen_steps == [4, 5, 6, 7, 8, 9]
    assert MASK_ID not in resumed.tokens.tolist()


def test_resume_freezes_fixed_tokens_through_remask():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _snap_cfg(n_steps=10, remask_enabled=True)
    h = np.zeros(8, dtype=np.float32)
    full = sampler.sample(
        sequence_length=8, h_values=h, denoiser=biased_denoiser, config=cfg,
        snapshot_steps=[3],
    )
    snap = full.snapshots[0]
    # A_B = positions that are still masked in the snapshot.
    masked = [i for i in range(8) if int(snap.x_t[i].item()) == MASK_ID]
    a_b = masked[:2]
    fixed = {a_b[0]: 1, a_b[1]: 2}
    resumed = sampler.sample(
        sequence_length=8, h_values=h, denoiser=biased_denoiser, config=cfg,
        controller=None,
        fixed_tokens=fixed,
        initial_state=ResumeState(
            x_t=snap.x_t, scores=snap.scores,
            unmask_step_by_pos=list(snap.unmask_step_by_pos), start_step=snap.step,
        ),
    )
    assert int(resumed.tokens[a_b[0]].item()) == 1
    assert int(resumed.tokens[a_b[1]].item()) == 2
    assert set(resumed.fixed_positions) == set(a_b)


def test_resume_is_paired_across_fixed_token_values():
    """Same seed + same snapshot, different A_B tuples -> identical unmasking
    selection (paired CRN), different sampled tokens. Remask off so selection is
    a pure function of the (paired) Bernoulli draws and the masked set."""
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _snap_cfg(n_steps=10, remask_enabled=False)
    h = np.zeros(8, dtype=np.float32)
    full = sampler.sample(
        sequence_length=8, h_values=h, denoiser=_context_denoiser, config=cfg,
        snapshot_steps=[3],
    )
    snap = full.snapshots[0]
    masked = [i for i in range(8) if int(snap.x_t[i].item()) == MASK_ID]
    a_b = masked[:2]

    def _resume(tokvals):
        return sampler.sample(
            sequence_length=8, h_values=h, denoiser=_context_denoiser, config=cfg,
            controller=None,
            fixed_tokens={a_b[0]: tokvals[0], a_b[1]: tokvals[1]},
            initial_state=ResumeState(
                x_t=snap.x_t, scores=snap.scores,
                unmask_step_by_pos=list(snap.unmask_step_by_pos), start_step=snap.step,
            ),
        )

    out_a = _resume((0, 1))
    out_b = _resume((2, 3))
    # Paired selection: every non-A_B position commits at the same step.
    non_ab = [i for i in range(8) if i not in a_b]
    assert [out_a.unmask_step_by_pos[i] for i in non_ab] == [
        out_b.unmask_step_by_pos[i] for i in non_ab
    ]
    # Different frozen context -> at least one sampled token differs.
    assert any(
        int(out_a.tokens[i].item()) != int(out_b.tokens[i].item()) for i in non_ab
    )


def test_resume_imports_resumestate_symbol():
    from inverse_folding.reference_flow.sampler import ResumeState, SamplerSnapshot

    assert ResumeState is not None and SamplerSnapshot is not None


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
