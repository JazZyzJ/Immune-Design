"""Sampler hard-anchor (uricase enzyme-mode v0) tests — PLAN_URICASE_ENZYME_MODE Task U2.

The sampler must:
- initialize hard-anchor positions to their fixed token before the loop,
- never overwrite or remask them (even under remask + a broadly-remasking
  controller, and even when controller=None),
- mark them with unmask_step_by_pos == 0 AND expose a separate ``fixed_positions``
  set so constraint-committed anchors stay distinguishable from ordinary residues
  first sampled at step 0,
- stay bit-equivalent to legacy behavior when no constraints are supplied.
"""

from __future__ import annotations

from types import SimpleNamespace

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

# Hard anchors for an 8-residue toy sequence: position 2 -> token 1, position 5 -> token 3.
# (Distinct from token 0, which the denoiser below produces for every free position.)
FIXED = {2: 1, 5: 3}


def prefer0_denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    """Every free position resolves deterministically to token 0 (never MASK)."""
    L = x_t.shape[0]
    logits = torch.zeros((L, VOCAB_SIZE), dtype=torch.float32)
    logits[:, MASK_ID] = -1e9
    logits[:, 0] = 1e9
    return logits


def _cfg(*, n_steps: int = 10, remask_enabled: bool = False, seed: int = 7) -> ReferenceFlowConfig:
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


def _sampler() -> PositionDependentDFMSampler:
    return PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)


def test_no_constraint_path_is_bit_equivalent():
    sampler = _sampler()
    cfg = _cfg(n_steps=10, remask_enabled=True)
    h = np.zeros(8, dtype=np.float32)
    out_none = sampler.sample(
        sequence_length=8, h_values=h, denoiser=prefer0_denoiser, config=cfg,
        save_trajectories=True, fixed_tokens=None,
    )
    out_empty = sampler.sample(
        sequence_length=8, h_values=h, denoiser=prefer0_denoiser, config=cfg,
        save_trajectories=True, fixed_tokens={},
    )
    assert torch.equal(out_none.tokens, out_empty.tokens)
    assert out_none.unmask_step_by_pos == out_empty.unmask_step_by_pos
    assert out_none.trajectory_rows == out_empty.trajectory_rows
    assert tuple(out_none.fixed_positions) == ()


def test_anchors_fixed_with_remask_disabled():
    sampler = _sampler()
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
        denoiser=prefer0_denoiser, config=_cfg(remask_enabled=False),
        fixed_tokens=FIXED,
    )
    assert MASK_ID not in out.tokens.tolist()
    assert out.tokens[2].item() == 1
    assert out.tokens[5].item() == 3
    # every other position resolves to token 0
    assert [out.tokens[i].item() for i in (0, 1, 3, 4, 6, 7)] == [0, 0, 0, 0, 0, 0]
    assert tuple(out.fixed_positions) == (2, 5)
    assert out.unmask_step_by_pos[2] == 0
    assert out.unmask_step_by_pos[5] == 0


def test_anchors_fixed_with_remask_enabled():
    # The linchpin: with remask on and controller=None, anchors carry score=-inf
    # and would be the first reparam remask targets without permanent protection.
    sampler = _sampler()
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
        denoiser=prefer0_denoiser, config=_cfg(n_steps=10, remask_enabled=True),
        fixed_tokens=FIXED, save_trajectories=True,
    )
    assert MASK_ID not in out.tokens.tolist()
    assert out.tokens[2].item() == 1
    assert out.tokens[5].item() == 3
    assert tuple(out.fixed_positions) == (2, 5)
    assert out.unmask_step_by_pos[2] == 0
    assert out.unmask_step_by_pos[5] == 0


class _BroadRemaskController:
    """Stub controller: passes logits through, asks to remask everything, and
    protects nothing — so only the sampler's permanent anchor set can save them."""

    config = None  # disables the D2 paired-uncorrected branch

    def step(self, ctx):
        return SimpleNamespace(logits=ctx.logits)

    def post_step(self, ctx):
        rank = np.full(ctx.sequence_length, -1e9, dtype=np.float64)
        return SimpleNamespace(rank_scores=rank, protected_positions=())


def test_anchors_fixed_under_broad_remask_stub_controller():
    sampler = _sampler()
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
        denoiser=prefer0_denoiser, config=_cfg(n_steps=10, remask_enabled=True),
        controller=_BroadRemaskController(), fixed_tokens=FIXED,
    )
    assert MASK_ID not in out.tokens.tolist()
    assert out.tokens[2].item() == 1
    assert out.tokens[5].item() == 3
    assert tuple(out.fixed_positions) == (2, 5)


def test_batch_and_single_lane_enforce_same_anchors():
    sampler = _sampler()
    cfg = _cfg(n_steps=10, remask_enabled=True)
    h = np.zeros(8, dtype=np.float32)

    single = sampler.sample(
        sequence_length=8, h_values=h, denoiser=prefer0_denoiser, config=cfg,
        protein_id="p0", design_idx=0, fixed_tokens=FIXED,
    )

    def batched_denoiser(x_ts, t, structs):
        del t, structs
        return [prefer0_denoiser(x, 0.0, None) for x in x_ts]

    batched = sampler.sample_batch(
        lanes=[
            SamplerBatchLane(
                sequence_length=8, h_values=h, config=cfg,
                protein_id="p0", design_idx=0, fixed_tokens=FIXED,
            )
        ],
        batched_denoiser=batched_denoiser,
    )
    assert batched[0].tokens[2].item() == 1
    assert batched[0].tokens[5].item() == 3
    assert tuple(batched[0].fixed_positions) == (2, 5)
    assert torch.equal(batched[0].tokens, single.tokens)
    assert batched[0].unmask_step_by_pos == single.unmask_step_by_pos


def test_fixed_positions_excludes_ordinary_step0_residues():
    # B7: fixed_positions must contain ONLY constraint anchors, even though many
    # ordinary positions also commit at step 0 (unmask_step_by_pos == 0).
    sampler = _sampler()
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
        denoiser=prefer0_denoiser, config=_cfg(n_steps=10, remask_enabled=False),
        fixed_tokens=FIXED,
    )
    step0_positions = {i for i, s in enumerate(out.unmask_step_by_pos) if s == 0}
    # Some ordinary residue commits at step 0 too, so the step-0 set is a strict
    # superset of the anchor set — proving unmask_step alone cannot distinguish them.
    assert {2, 5}.issubset(step0_positions)
    assert set(out.fixed_positions) == {2, 5}
    assert step0_positions != set(out.fixed_positions)


def test_fixed_tokens_out_of_range_index_fails():
    sampler = _sampler()
    with __import__("pytest").raises(ValueError, match=r"(?i)range|bound"):
        sampler.sample(
            sequence_length=8, h_values=np.zeros(8, dtype=np.float32),
            denoiser=prefer0_denoiser, config=_cfg(), fixed_tokens={99: 1},
        )
