"""V1F1 sampler continuation contract tests.

Pre-denoiser (post-previous-remask) checkpoint boundary, editable-maturity crossing, and
early stop for pre-terminal continuation-value allocation
(PLAN_RF_REFINE_FUSION_V1 §2.3-2.5, task V1F1). Additive to the legacy ``snapshot_steps`` /
``ResumeState`` diagnostic surface, which must stay byte-compatible.

Maturity is AA20-membership (not merely non-mask); anchors are firewalled; non-crossing is
fail-closed for every maturity-trigger request; resume+continuation is out of V1F1 scope.
"""

from __future__ import annotations

import dataclasses

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
from inverse_folding.reference_flow.sampler import (
    ContinuationCheckpoint,
    ContinuationRequest,
    ContinuationResume,
    FixedTokenError,
    InvalidPreterminalRootError,
    MaturityNotReachedError,
    MaturityTelemetry,
    PositionDependentDFMSampler,
    ResumeState,
    SamplerBatchLane,
    editable_maturity,
)
from inverse_folding.reference_flow.sampler import (  # private, load-bearing invariants
    _is_upward_crossing,
    _normalize_and_validate_fixed_tokens,
    _validate_preterminal_root,
)

VOCAB_SIZE = 5
MASK_ID = 4
AA_IDS = frozenset({0, 1, 2, 3})  # canonical residue token-id set for the toy vocab
CONTINUATION_PHASE = "pre_denoiser_after_previous_remask"


def uniform_denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    logits = torch.zeros((x_t.shape[0], VOCAB_SIZE), dtype=torch.float32)
    logits[:, MASK_ID] = -1e9
    return logits


def _cfg(*, n_steps: int = 12, seed: int = 7, remask_enabled: bool = False):
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


def _sample(sampler, cfg, length, *, residue_token_ids=AA_IDS, **kw):
    return sampler.sample(
        sequence_length=length,
        h_values=np.zeros(length, dtype=np.float32),
        denoiser=uniform_denoiser,
        config=cfg,
        residue_token_ids=residue_token_ids,
        **kw,
    )


# --------------------------------------------------------------------------- #
# editable_maturity — AA20 membership, current-x_t only, anchor exclusion
# --------------------------------------------------------------------------- #
def test_editable_maturity_counts_current_x_t_not_unmask_history():
    # Position 2 was once unmasked (history >= 0) but is MASK now (reparam remask).
    # Maturity is a function of the current x_t alone -> counts it UNRESOLVED.
    x_t = torch.tensor([0, 1, MASK_ID, 3, MASK_ID], dtype=torch.long)
    mt = editable_maturity(
        x_t=x_t, editable_positions=range(5), fixed_positions=(), mask_token_id=MASK_ID,
        aa_token_ids=AA_IDS,
    )
    assert isinstance(mt, MaturityTelemetry)
    assert mt.n_editable == 5
    assert mt.n_resolved_editable == 3  # positions 0, 1, 3
    assert mt.n_unresolved_editable == 2
    assert mt.unresolved_editable_mask == (False, False, True, False, True)
    assert mt.rho_edit == pytest.approx(3 / 5)


def test_editable_maturity_excludes_fixed_from_rho_edit_but_counts_known_identity():
    x_t = torch.tensor([0, 1, MASK_ID, 3, MASK_ID], dtype=torch.long)
    mt = editable_maturity(
        x_t=x_t, editable_positions=(0, 2, 3, 4), fixed_positions=(1,), mask_token_id=MASK_ID,
        aa_token_ids=AA_IDS,
    )
    assert mt.n_fixed == 1
    assert mt.n_editable == 4
    assert mt.n_resolved_editable == 2
    assert mt.rho_edit == pytest.approx(2 / 4)
    assert mt.rho_known_sequence_identity == pytest.approx((1 + 2) / 5)
    assert mt.fixed_mask == (False, True, False, False, False)
    assert mt.editable_mask == (True, False, True, True, True)


def test_editable_maturity_rejects_non_canonical_editable_token():
    # A non-mask, non-AA20 token (unknown/padding/out-of-range) must never be counted mature.
    x_t = torch.tensor([0, 1, 99, MASK_ID], dtype=torch.long)
    with pytest.raises(ValueError):
        editable_maturity(
            x_t=x_t, editable_positions=range(4), fixed_positions=(), mask_token_id=MASK_ID,
            aa_token_ids=AA_IDS,
        )


def test_editable_maturity_zero_editable_is_hard_failure():
    x_t = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        editable_maturity(
            x_t=x_t, editable_positions=(), fixed_positions=(0, 1), mask_token_id=MASK_ID,
            aa_token_ids=AA_IDS,
        )


# --------------------------------------------------------------------------- #
# _validate_preterminal_root and _is_upward_crossing (pure helpers)
# --------------------------------------------------------------------------- #
def test_validate_preterminal_root_rejects_fully_resolved_editable():
    x_t = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    mt = editable_maturity(
        x_t=x_t, editable_positions=range(4), fixed_positions=(), mask_token_id=MASK_ID,
        aa_token_ids=AA_IDS,
    )
    assert mt.n_unresolved_editable == 0
    with pytest.raises(InvalidPreterminalRootError):
        _validate_preterminal_root(mt)


def test_validate_preterminal_root_accepts_partial():
    x_t = torch.tensor([0, MASK_ID, 2, 3], dtype=torch.long)
    mt = editable_maturity(
        x_t=x_t, editable_positions=range(4), fixed_positions=(), mask_token_id=MASK_ID,
        aa_token_ids=AA_IDS,
    )
    _validate_preterminal_root(mt)  # no raise


def test_upward_crossing_ignores_within_step_transient():
    # PLAN §8.2: maturity 0.4 -> (within-step transient 0.8) -> post-remask 0.5 -> 0.75.
    # Only pre-denoiser boundaries are evaluated, so the transient 0.8 is never a crossing
    # candidate; the boundary sequence is [0.4, 0.5, 0.75] and the crossing is the 0.75 boundary.
    assert _is_upward_crossing(None, 0.0, 0.75) is False
    assert _is_upward_crossing(0.4, 0.5, 0.75) is False
    assert _is_upward_crossing(0.5, 0.75, 0.75) is True  # first boundary >= target
    assert _is_upward_crossing(0.5, 0.8, 0.75) is True  # prev < target <= cur
    assert _is_upward_crossing(0.8, 0.9, 0.75) is False  # already above -> not a new crossing


# --------------------------------------------------------------------------- #
# _normalize_and_validate_fixed_tokens — anchor firewall (PLAN §2.3)
# --------------------------------------------------------------------------- #
def test_normalize_fixed_tokens_rejects_mask_anchor():
    with pytest.raises(FixedTokenError):
        _normalize_and_validate_fixed_tokens({0: MASK_ID}, 6, MASK_ID, AA_IDS)


def test_normalize_fixed_tokens_rejects_normalized_key_conflict():
    with pytest.raises(FixedTokenError):
        _normalize_and_validate_fixed_tokens({1: 0, "1": 2}, 6, MASK_ID, AA_IDS)


def test_normalize_fixed_tokens_rejects_out_of_range_and_non_canonical():
    with pytest.raises(FixedTokenError):
        _normalize_and_validate_fixed_tokens({9: 1}, 6, MASK_ID, AA_IDS)
    with pytest.raises(FixedTokenError):
        _normalize_and_validate_fixed_tokens({0: 99}, 6, MASK_ID, AA_IDS)


def test_normalize_fixed_tokens_accepts_valid_and_normalizes_keys():
    out = _normalize_and_validate_fixed_tokens({0: 1, "3": 2}, 6, MASK_ID, AA_IDS)
    assert out == {0: 1, 3: 2}


def test_sample_rejects_invalid_fixed_tokens_end_to_end():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=10)
    with pytest.raises(FixedTokenError):
        _sample(sampler, cfg, 6, fixed_tokens={1: 0, "1": 2},
                continuation=ContinuationRequest(at_step=3))
    with pytest.raises(FixedTokenError):
        _sample(sampler, cfg, 6, fixed_tokens={0: MASK_ID},
                continuation=ContinuationRequest(at_step=3))


# --------------------------------------------------------------------------- #
# pre-denoiser checkpoint: exact DFE, masked-root, early stop, tail contract
# --------------------------------------------------------------------------- #
def test_checkpoint_at_step_charges_exactly_s_forwards_and_returns_masked_root():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    seen: list[int] = []

    def counting(x_t, t, struct):
        seen.append(int(round(float(t) * 12)))
        return uniform_denoiser(x_t, t, struct)

    out = sampler.sample(
        sequence_length=10,
        h_values=np.zeros(10, dtype=np.float32),
        denoiser=counting,
        config=cfg,
        residue_token_ids=AA_IDS,
        continuation=ContinuationRequest(at_step=4, early_stop=True),
    )
    assert seen == [0, 1, 2, 3]  # step-4 forward NOT paid
    assert out.stopped_early is True
    cp = out.continuation_checkpoint
    assert isinstance(cp, ContinuationCheckpoint)
    assert cp.step == 4
    assert cp.paid_prefix_dfe == 4
    assert cp.snapshot_phase == CONTINUATION_PHASE
    assert (cp.x_t == MASK_ID).any()
    assert cp.maturity.n_unresolved_editable >= 1


def test_checkpoint_carries_n_steps_for_tail_contract():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    out = _sample(sampler, cfg, 10, continuation=ContinuationRequest(at_step=4, early_stop=True))
    cp = out.continuation_checkpoint
    assert cp.n_steps == 12
    assert cp.n_steps - cp.step == 8  # S-s tail contract is self-contained
    assert cp.paid_prefix_dfe == cp.step


def test_early_stop_does_not_run_terminal_residual_completion():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    out = _sample(sampler, cfg, 10, continuation=ContinuationRequest(at_step=3, early_stop=True))
    assert out.stopped_early is True
    assert (out.tokens == MASK_ID).any()  # masked tokens survive (no residual completion)


# --------------------------------------------------------------------------- #
# observation neutrality (RED 5 + trajectory rows + forward count)
# --------------------------------------------------------------------------- #
def test_observe_checkpoint_is_byte_equivalent_to_no_continuation():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12, remask_enabled=True)
    base = _sample(sampler, cfg, 10)
    obs = _sample(sampler, cfg, 10, continuation=ContinuationRequest(at_step=5, early_stop=False))
    assert torch.equal(base.tokens, obs.tokens)
    assert base.unmask_step_by_pos == obs.unmask_step_by_pos
    assert base.g_values == obs.g_values
    assert base.continuation_checkpoint is None and base.stopped_early is False
    assert obs.stopped_early is False
    assert obs.continuation_checkpoint is not None and obs.continuation_checkpoint.step == 5
    assert MASK_ID not in obs.tokens.tolist()


def test_observation_is_neutral_on_trajectory_rows_and_forward_count():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12, remask_enabled=True)
    n_base: list[int] = []
    n_obs: list[int] = []

    def d_base(x, t, s):
        n_base.append(1)
        return uniform_denoiser(x, t, s)

    def d_obs(x, t, s):
        n_obs.append(1)
        return uniform_denoiser(x, t, s)

    base = sampler.sample(
        sequence_length=10, h_values=np.zeros(10, dtype=np.float32), denoiser=d_base,
        config=cfg, save_trajectories=True, residue_token_ids=AA_IDS,
    )
    obs = sampler.sample(
        sequence_length=10, h_values=np.zeros(10, dtype=np.float32), denoiser=d_obs,
        config=cfg, save_trajectories=True, residue_token_ids=AA_IDS,
        continuation=ContinuationRequest(at_step=5, early_stop=False),
    )
    assert len(n_base) == len(n_obs)  # observation adds no forward
    assert base.trajectory_rows == obs.trajectory_rows
    assert torch.equal(base.tokens, obs.tokens)


# --------------------------------------------------------------------------- #
# maturity crossing: first post-remask upward boundary; anchors excluded (RED 2)
# --------------------------------------------------------------------------- #
def test_rho_crossing_captures_first_post_remask_upward_boundary():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=16, remask_enabled=True)
    target = 0.5
    run = _sample(sampler, cfg, 12, continuation=ContinuationRequest(at_rho_edit=target, early_stop=True))
    cp = run.continuation_checkpoint
    assert cp is not None
    s = cp.step
    assert 1 <= s < 16
    assert cp.maturity.rho_edit >= target
    assert cp.paid_prefix_dfe == s
    prev = _sample(sampler, cfg, 12, continuation=ContinuationRequest(at_step=s - 1, early_stop=True))
    assert prev.continuation_checkpoint.maturity.rho_edit < target
    same = _sample(sampler, cfg, 12, continuation=ContinuationRequest(at_step=s, early_stop=True))
    assert torch.equal(cp.x_t, same.continuation_checkpoint.x_t)


def test_rho_crossing_with_anchors_excludes_them():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=16, remask_enabled=True)
    out = sampler.sample(
        sequence_length=12, h_values=np.zeros(12, dtype=np.float32), denoiser=uniform_denoiser,
        config=cfg, residue_token_ids=AA_IDS, fixed_tokens={0: 1, 11: 2},
        continuation=ContinuationRequest(at_rho_edit=0.5, early_stop=True),
    )
    cp = out.continuation_checkpoint
    assert cp.maturity.n_fixed == 2
    assert cp.maturity.n_editable == 10
    assert cp.maturity.rho_edit >= 0.5
    assert not cp.maturity.editable_mask[0] and not cp.maturity.editable_mask[11]
    assert int(cp.x_t[0].item()) == 1 and int(cp.x_t[11].item()) == 2


def test_checkpoint_maturity_excludes_anchor_positions():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    out = sampler.sample(
        sequence_length=10, h_values=np.zeros(10, dtype=np.float32), denoiser=uniform_denoiser,
        config=cfg, residue_token_ids=AA_IDS, fixed_tokens={0: 1, 5: 2},
        continuation=ContinuationRequest(at_step=6, early_stop=True),
    )
    cp = out.continuation_checkpoint
    assert cp.maturity.n_fixed == 2
    assert cp.maturity.n_editable == 8
    assert cp.maturity.fixed_mask[0] and cp.maturity.fixed_mask[5]
    assert not cp.maturity.editable_mask[0] and not cp.maturity.editable_mask[5]
    assert int(cp.x_t[0].item()) == 1 and int(cp.x_t[5].item()) == 2
    assert cp.fixed_tokens == ((0, 1), (5, 2))


# --------------------------------------------------------------------------- #
# non-crossing is fail-closed for BOTH early-stop and observe (PLAN §2.5)
# --------------------------------------------------------------------------- #
def test_no_cross_fails_closed_for_both_early_stop_and_observe():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=6)
    length = 8
    full = _sample(sampler, cfg, length)
    # Precondition: some position first commits at the last step / residual, so rho_edit never
    # hits 1.0 at a pre-denoiser boundary.
    assert max(full.unmask_step_by_pos) >= cfg.sampler.n_steps - 1, "seed committed all positions early"
    for early in (True, False):
        with pytest.raises(MaturityNotReachedError):
            _sample(sampler, cfg, length, continuation=ContinuationRequest(at_rho_edit=1.0, early_stop=early))


# --------------------------------------------------------------------------- #
# scope / legacy / request validation
# --------------------------------------------------------------------------- #
def test_resume_with_continuation_is_rejected_in_v1f1():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=10)
    full = _sample(sampler, cfg, 8, snapshot_steps=[3])
    snap = full.snapshots[0]
    with pytest.raises(ValueError):
        sampler.sample(
            sequence_length=8, h_values=np.zeros(8, dtype=np.float32), denoiser=uniform_denoiser,
            config=cfg, residue_token_ids=AA_IDS,
            initial_state=ResumeState(
                x_t=snap.x_t, scores=snap.scores,
                unmask_step_by_pos=list(snap.unmask_step_by_pos), start_step=snap.step,
            ),
            continuation=ContinuationRequest(at_step=5),
        )


def test_continuation_requires_residue_token_ids():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=10)
    with pytest.raises(ValueError):
        sampler.sample(
            sequence_length=6, h_values=np.zeros(6, dtype=np.float32), denoiser=uniform_denoiser,
            config=cfg, continuation=ContinuationRequest(at_step=3),
        )


def test_legacy_snapshot_path_unaffected_and_new_fields_default():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=10)
    # No continuation, no residue_token_ids -> pure legacy behavior.
    out = sampler.sample(
        sequence_length=8, h_values=np.zeros(8, dtype=np.float32), denoiser=uniform_denoiser,
        config=cfg, snapshot_steps=[3],
    )
    assert [s.step for s in out.snapshots] == [3]
    assert out.continuation_checkpoint is None
    assert out.stopped_early is False
    assert MASK_ID not in out.tokens.tolist()


def test_continuation_request_requires_exactly_one_trigger():
    with pytest.raises(ValueError):
        ContinuationRequest()
    with pytest.raises(ValueError):
        ContinuationRequest(at_step=2, at_rho_edit=0.5)


def test_continuation_at_step_out_of_range_raises():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=8)
    with pytest.raises(ValueError):
        _sample(sampler, cfg, 6, continuation=ContinuationRequest(at_step=8, early_stop=True))
    with pytest.raises(ValueError):
        ContinuationRequest(at_step=-1)


# =========================================================================== #
# V1F2 — exact identity replay, fork seed namespaces, resumed batch parity
# (PLAN_RF_REFINE_FUSION_V1 §2.4-2.7, task V1F2)
# =========================================================================== #
def _capture(sampler, cfg, length, at_step, **kw):
    """Run to completion in OBSERVE mode; return (completed_output, checkpoint)."""
    out = _sample(
        sampler, cfg, length,
        continuation=ContinuationRequest(at_step=at_step, early_stop=False), **kw,
    )
    return out, out.continuation_checkpoint


def test_identity_resume_reproduces_uninterrupted_suffix():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=14, remask_enabled=True)
    obs, cp = _capture(sampler, cfg, 10, at_step=5)
    res = _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.identity(cp))
    # Exact replay: terminal tokens, unmask history, and FINAL RNG state all match the
    # uninterrupted run (final-RNG equality means the whole draw stream matched).
    assert torch.equal(res.tokens, obs.tokens)
    assert res.unmask_step_by_pos == obs.unmask_step_by_pos
    assert res.final_rng_state == obs.final_rng_state
    assert res.continuation_checkpoint is None and res.stopped_early is False


def test_fork_is_deterministic_and_independent():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=14, remask_enabled=True)
    _, cp = _capture(sampler, cfg, 10, at_step=5)
    f1a = _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.fork(cp, 100))
    f1b = _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.fork(cp, 100))
    f2 = _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.fork(cp, 200))
    ident = _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.identity(cp))
    # same fork seed -> byte-identical (reproducible)
    assert torch.equal(f1a.tokens, f1b.tokens)
    assert f1a.final_rng_state == f1b.final_rng_state
    # different fork seed -> independent stream (guaranteed distinct final RNG state)
    assert f1a.final_rng_state != f2.final_rng_state
    # a fork never impersonates the identity stream
    assert f1a.final_rng_state != ident.final_rng_state
    # divergence shows up on positions still masked at the checkpoint
    masked_at_cp = [i for i in range(10) if int(cp.x_t[i].item()) == MASK_ID]
    assert masked_at_cp
    assert any(int(f1a.tokens[i].item()) != int(f2.tokens[i].item()) for i in masked_at_cp)
    assert any(int(f1a.tokens[i].item()) != int(ident.tokens[i].item()) for i in masked_at_cp)


def test_identity_resume_with_tampered_rng_state_raises():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    # A DIFFERENT VALID RNG state would pass restore yet silently produce a non-identity suffix;
    # the replay hash binds the identity rng_state, so the swap fails the tamper gate up front.
    other_valid = np.random.default_rng(9999).bit_generator.state
    bad = dataclasses.replace(ContinuationResume.identity(cp), rng_state=other_valid)
    with pytest.raises(ValueError):  # state_hash mismatch BEFORE any denoiser call
        _sample(sampler, cfg, 8, continuation_resume=bad)


def test_identity_resume_with_tampered_unmask_history_raises():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    forged = tuple(reversed(cp.unmask_step_by_pos))  # same length, different (forged) history
    bad = dataclasses.replace(ContinuationResume.identity(cp), unmask_step_by_pos=forged)
    with pytest.raises(ValueError):
        _sample(sampler, cfg, 8, continuation_resume=bad)


def test_resume_rejects_n_steps_mismatch():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    _, cp = _capture(sampler, _cfg(n_steps=14), 8, at_step=4)
    with pytest.raises(ValueError):
        sampler.sample(
            sequence_length=8, h_values=np.zeros(8, dtype=np.float32), denoiser=uniform_denoiser,
            config=_cfg(n_steps=10), continuation_resume=ContinuationResume.identity(cp),
        )


def test_resume_rejects_tampered_x_t():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    tampered = cp.x_t.clone()
    tampered[0] = (int(tampered[0].item()) + 1) % VOCAB_SIZE  # mutate one token
    bad = dataclasses.replace(ContinuationResume.identity(cp), x_t=tampered)
    with pytest.raises(ValueError):  # state_hash mismatch before any denoiser call
        _sample(sampler, cfg, 8, continuation_resume=bad)


def test_resume_rejects_extra_fixed_tokens_param():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    with pytest.raises(ValueError):  # resume is self-contained; a fixed_tokens param conflicts
        sampler.sample(
            sequence_length=8, h_values=np.zeros(8, dtype=np.float32), denoiser=uniform_denoiser,
            config=cfg, fixed_tokens={0: 1}, continuation_resume=ContinuationResume.identity(cp),
        )


def test_resume_with_wrong_length_raises():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    with pytest.raises(ValueError):
        _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.identity(cp))


def test_identity_resume_preserves_anchors_and_their_history():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=14, remask_enabled=True)
    obs, cp = _capture(sampler, cfg, 10, at_step=5, fixed_tokens={0: 1, 9: 2})
    assert cp.unmask_step_by_pos[0] == 0 and cp.unmask_step_by_pos[9] == 0
    # Resume is self-contained: it carries the anchors, so fixed_tokens is NOT re-passed.
    res = sampler.sample(
        sequence_length=10, h_values=np.zeros(10, dtype=np.float32), denoiser=uniform_denoiser,
        config=cfg, continuation_resume=ContinuationResume.identity(cp),
    )
    assert int(res.tokens[0].item()) == 1 and int(res.tokens[9].item()) == 2
    # anchor unmask history preserved (NOT overwritten to the resume start step)
    assert res.unmask_step_by_pos[0] == 0 and res.unmask_step_by_pos[9] == 0
    assert torch.equal(res.tokens, obs.tokens)


def test_continuation_resume_mutually_exclusive():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=12)
    _, cp = _capture(sampler, cfg, 8, at_step=4)
    resume = ContinuationResume.identity(cp)
    with pytest.raises(ValueError):
        _sample(sampler, cfg, 8, continuation_resume=resume,
                continuation=ContinuationRequest(at_step=5))
    with pytest.raises(ValueError):
        sampler.sample(
            sequence_length=8, h_values=np.zeros(8, dtype=np.float32), denoiser=uniform_denoiser,
            config=cfg, continuation_resume=resume,
            initial_state=ResumeState(
                x_t=cp.x_t, scores=cp.scores,
                unmask_step_by_pos=list(cp.unmask_step_by_pos), start_step=cp.step,
            ),
        )


def test_continuation_resume_field_validation():
    x = torch.tensor([0, MASK_ID], dtype=torch.long)
    common = dict(
        x_t=x, scores=np.zeros(2), unmask_step_by_pos=(0, -1), start_step=1, n_steps=4,
        fixed_tokens=(), editable_positions=(0, 1), state_hash="x",
    )
    with pytest.raises(ValueError):
        ContinuationResume(**common, mode="bogus")
    with pytest.raises(ValueError):
        ContinuationResume(**common, mode="identity")  # missing rng_state
    with pytest.raises(ValueError):
        ContinuationResume(**common, mode="fork")  # missing fork_seed


def _batched_uniform(x_ts, t, structs):
    return [uniform_denoiser(x, t, s) for x, s in zip(x_ts, structs)]


def test_batch_resume_fork_parity_with_scalar():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=14, remask_enabled=True)
    _, cp = _capture(sampler, cfg, 10, at_step=5)
    fork_seeds = [11, 22, 33]
    scalar = [
        _sample(sampler, cfg, 10, continuation_resume=ContinuationResume.fork(cp, fs))
        for fs in fork_seeds
    ]
    lanes = [
        SamplerBatchLane(
            sequence_length=10, h_values=np.zeros(10, dtype=np.float32), config=cfg,
            resume=ContinuationResume.fork(cp, fs),
        )
        for fs in fork_seeds
    ]
    batched = sampler.sample_batch(lanes=lanes, batched_denoiser=_batched_uniform)
    assert len(batched) == len(scalar)
    for sc, ba in zip(scalar, batched):
        assert torch.equal(sc.tokens, ba.tokens)
        assert sc.unmask_step_by_pos == ba.unmask_step_by_pos
        assert sc.final_rng_state == ba.final_rng_state


def test_batch_rejects_mixed_resume_start_steps():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    cfg = _cfg(n_steps=14, remask_enabled=True)
    _, cp5 = _capture(sampler, cfg, 10, at_step=5)
    _, cp7 = _capture(sampler, cfg, 10, at_step=7)
    lanes = [
        SamplerBatchLane(sequence_length=10, h_values=np.zeros(10, dtype=np.float32),
                         config=cfg, resume=ContinuationResume.fork(cp5, 1)),
        SamplerBatchLane(sequence_length=10, h_values=np.zeros(10, dtype=np.float32),
                         config=cfg, resume=ContinuationResume.fork(cp7, 2)),
    ]
    with pytest.raises(ValueError):
        sampler.sample_batch(lanes=lanes, batched_denoiser=_batched_uniform)
