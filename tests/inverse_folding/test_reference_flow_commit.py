"""Phase D3 commit/EMA primitives contract tests (PLAN_RF.md §D3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.commit import (
    compute_chosen_token_logprob,
    compute_commit_score,
    compute_d3_evidence_input,
    compute_ema_gamma,
    compute_residue_reliability,
    compute_stage_a_rank_score,
    project_window_excess_to_residue,
    select_freeze_protected_positions,
    update_ema,
    z_score,
)
from inverse_folding.reference_flow.controller_config import D3Config


VOCAB_SIZE = 16
MASK_TOKEN_ID = 0


@dataclass(frozen=True)
class _Window:
    start_0b: int
    end_0b: int


@dataclass(frozen=True)
class _Block:
    residue_start_0b: int
    residue_end_0b: int
    rho_B: float


# ---------------------------------------------------------------------------
# project_window_excess_to_residue (PLAN §D3-3)
# ---------------------------------------------------------------------------


def test_project_window_excess_max_over_covering_windows():
    windows = [_Window(0, 3), _Window(2, 5), _Window(4, 6)]
    window_excess = [0.5, 0.8, 0.0]
    e_i = project_window_excess_to_residue(
        windows=windows, window_excess=window_excess, sequence_length=6
    )
    # residues 0,1 only covered by window 0 → 0.5
    # residue 2 covered by 0 (0.5) and 1 (0.8) → 0.8
    # residues 3,4 only by window 1 → 0.8
    # residue 5 only by window 2 (0.0) → 0.0
    np.testing.assert_allclose(e_i, [0.5, 0.5, 0.8, 0.8, 0.8, 0.0])


def test_project_window_excess_zero_when_no_window_covers_residue():
    windows = [_Window(2, 4)]
    e_i = project_window_excess_to_residue(
        windows=windows, window_excess=[1.0], sequence_length=6
    )
    np.testing.assert_allclose(e_i, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])


def test_project_window_excess_skips_nonpositive():
    # Windows with excess <= 0 must not lower a residue that other windows already raised.
    windows = [_Window(0, 4), _Window(0, 4)]
    e_i = project_window_excess_to_residue(
        windows=windows, window_excess=[0.7, -0.3], sequence_length=4
    )
    np.testing.assert_allclose(e_i, [0.7, 0.7, 0.7, 0.7])


# ---------------------------------------------------------------------------
# compute_residue_reliability (PLAN §D3-4)
# ---------------------------------------------------------------------------


def test_residue_reliability_max_over_covering_blocks():
    blocks = [
        _Block(residue_start_0b=0, residue_end_0b=3, rho_B=0.4),
        _Block(residue_start_0b=2, residue_end_0b=5, rho_B=0.9),
    ]
    rho_i = compute_residue_reliability(active_blocks=blocks, sequence_length=6)
    np.testing.assert_allclose(rho_i, [0.4, 0.4, 0.9, 0.9, 0.9, 0.0])


def test_residue_reliability_zero_without_blocks():
    rho_i = compute_residue_reliability(active_blocks=(), sequence_length=4)
    np.testing.assert_allclose(rho_i, np.zeros(4))


# ---------------------------------------------------------------------------
# EMA: gamma + update + cold start (PLAN §D3-5/D3-6)
# ---------------------------------------------------------------------------


def test_gamma_high_reliability_uses_gamma_min():
    rho_i = np.array([1.0, 1.0])
    gamma = compute_ema_gamma(rho_i=rho_i, gamma_min=0.4, gamma_max=0.9)
    np.testing.assert_allclose(gamma, [0.4, 0.4])


def test_gamma_low_reliability_uses_gamma_max():
    rho_i = np.array([0.0, 0.0])
    gamma = compute_ema_gamma(rho_i=rho_i, gamma_min=0.4, gamma_max=0.9)
    np.testing.assert_allclose(gamma, [0.9, 0.9])


def test_gamma_interpolates_linearly():
    rho_i = np.array([0.5])
    gamma = compute_ema_gamma(rho_i=rho_i, gamma_min=0.4, gamma_max=0.9)
    np.testing.assert_allclose(gamma, [0.65])


def test_update_ema_cold_start_uses_e_i_directly():
    e_i = np.array([0.2, 0.5, 0.0])
    m = update_ema(m_prev=None, e_i=e_i, rho_i=np.zeros_like(e_i), gamma_min=0.4, gamma_max=0.9)
    np.testing.assert_allclose(m, e_i)


def test_update_ema_combines_prev_and_current():
    m_prev = np.array([1.0, 1.0])
    e_i = np.array([0.0, 0.0])
    # rho=1 → gamma=gamma_min=0.4 → m = 0.4 * m_prev + 0.6 * e_i = 0.4
    m = update_ema(
        m_prev=m_prev,
        e_i=e_i,
        rho_i=np.array([1.0, 1.0]),
        gamma_min=0.4,
        gamma_max=0.9,
    )
    np.testing.assert_allclose(m, [0.4, 0.4])


# ---------------------------------------------------------------------------
# compute_chosen_token_logprob (PLAN §D3-7)
# ---------------------------------------------------------------------------


def test_chosen_token_logprob_uses_struct_logits_not_scores():
    L = 3
    logits = torch.full((L, VOCAB_SIZE), -10.0)
    logits[0, 5] = 5.0
    logits[1, 6] = 2.0; logits[1, 7] = 2.0
    logits[2, 8] = 0.0
    x_t = torch.tensor([5, 7, MASK_TOKEN_ID])
    out = compute_chosen_token_logprob(struct_logits=logits, x_t=x_t, mask_token_id=MASK_TOKEN_ID)
    # Position 0: ~0 log-prob (token 5 dominates), Position 1: ~log(0.5), Position 2: NaN
    assert out[0] == pytest.approx(0.0, abs=1e-3)
    assert out[1] < 0.0
    assert np.isnan(out[2])


# ---------------------------------------------------------------------------
# z_score (PLAN §D3-8)
# ---------------------------------------------------------------------------


def test_zscore_returns_zero_when_too_few_eligible():
    vals = np.array([1.0, 2.0, np.nan])
    mask = np.array([True, False, False])  # only one eligible
    out = z_score(values=vals, eligibility_mask=mask, zscore_epsilon=1e-6)
    np.testing.assert_allclose(out, np.zeros(3))


def test_zscore_returns_zero_when_std_below_epsilon():
    vals = np.array([5.0, 5.0, 5.0])
    mask = np.array([True, True, True])
    out = z_score(values=vals, eligibility_mask=mask, zscore_epsilon=1e-3)
    np.testing.assert_allclose(out, np.zeros(3))


def test_zscore_eligible_only_includes_committed():
    vals = np.array([10.0, 20.0, 30.0, np.nan])
    mask = np.array([True, True, True, False])  # last one masked
    out = z_score(values=vals, eligibility_mask=mask, zscore_epsilon=1e-6)
    # Standardize over [10, 20, 30] → mean=20, std=sqrt(200/3)≈8.165
    mu, sigma = 20.0, float(np.std([10.0, 20.0, 30.0]))
    expected = np.array([(10 - mu) / sigma, 0.0, (30 - mu) / sigma, 0.0])
    np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# compute_commit_score + same-refresh grace (PLAN §D3-9/D3-10)
# ---------------------------------------------------------------------------


def test_commit_score_combines_z_ell_and_z_m():
    z_ell = np.array([0.5, -0.2, 1.0])
    z_m = np.array([1.0, 0.0, -0.5])
    score = compute_commit_score(
        z_ell=z_ell, z_m=z_m, lambda_commit=1.0, grace_positions=None
    )
    np.testing.assert_allclose(score, z_ell - z_m)


def test_grace_removes_immune_penalty_at_corrected_positions_only():
    z_ell = np.array([0.5, -0.2, 1.0])
    z_m = np.array([1.0, 1.0, 1.0])
    score = compute_commit_score(
        z_ell=z_ell, z_m=z_m, lambda_commit=2.0, grace_positions=(1,)
    )
    # Position 0: 0.5 - 2*1.0 = -1.5
    # Position 1: grace → just z_ell = -0.2
    # Position 2: 1.0 - 2*1.0 = -1.0
    np.testing.assert_allclose(score, [-1.5, -0.2, -1.0])


# ---------------------------------------------------------------------------
# select_freeze_protected_positions (PLAN §D3-12, validation 13)
# ---------------------------------------------------------------------------


def test_freeze_empty_before_freeze_window():
    x_t = torch.tensor([5, MASK_TOKEN_ID, 6])
    protected = select_freeze_protected_positions(
        x_t=x_t, mask_token_id=MASK_TOKEN_ID, step=4, n_steps=10, final_freeze_steps=2
    )
    assert protected == ()


def test_freeze_protects_all_committed_in_window():
    x_t = torch.tensor([5, MASK_TOKEN_ID, 6, MASK_TOKEN_ID, 7])
    protected = select_freeze_protected_positions(
        x_t=x_t, mask_token_id=MASK_TOKEN_ID, step=8, n_steps=10, final_freeze_steps=2
    )
    # step 8 in freeze window (n_steps - final_freeze_steps = 8) → freeze active.
    assert set(protected) == {0, 2, 4}


def test_freeze_final_freeze_steps_one_aligns_with_last_step():
    x_t = torch.tensor([5, 6])
    # final_freeze_steps=1; step==n_steps-1 must trigger freeze.
    protected = select_freeze_protected_positions(
        x_t=x_t, mask_token_id=MASK_TOKEN_ID, step=9, n_steps=10, final_freeze_steps=1
    )
    assert set(protected) == {0, 1}


# ---------------------------------------------------------------------------
# Stage A multi-objective rank face
# ---------------------------------------------------------------------------


def test_stage_a_rank_score_combines_active_terms_over_all_committed_residues():
    L = 4
    logits = torch.full((L, VOCAB_SIZE), -3.0)
    logits[0, 5] = 4.0
    logits[1, 6] = 3.0
    logits[2, 7] = 2.0
    logits[3, 8] = 1.0
    x_t = torch.tensor([5, 6, MASK_TOKEN_ID, 8])
    scores = np.array([-0.1, -2.0, -np.inf, -0.5], dtype=np.float64)
    m_i = np.array([0.0, 2.0, 100.0, 1.0], dtype=np.float64)
    d2 = np.array([0.0, 1.5, 0.0, 0.5], dtype=np.float64)
    scores_before = scores.copy()

    rank = compute_stage_a_rank_score(
        structural_logits=logits,
        x_t=x_t,
        scores=scores,
        mask_token_id=MASK_TOKEN_ID,
        m_i=m_i,
        d2_evidence=d2,
        alpha_struct=0.25,
        lambda_commit=0.5,
        d2_evidence_nu=2.0,
        zscore_epsilon=1.0e-6,
    )

    eligibility = np.array([True, True, False, True])
    ell_struct = compute_chosen_token_logprob(
        struct_logits=logits, x_t=x_t, mask_token_id=MASK_TOKEN_ID
    )
    expected = (
        0.25 * z_score(values=ell_struct, eligibility_mask=eligibility, zscore_epsilon=1e-6)
        + 0.75 * z_score(values=scores, eligibility_mask=eligibility, zscore_epsilon=1e-6)
        - 0.5 * z_score(values=m_i, eligibility_mask=eligibility, zscore_epsilon=1e-6)
        + 2.0 * z_score(values=d2, eligibility_mask=eligibility, zscore_epsilon=1e-6)
    )
    np.testing.assert_allclose(rank, expected, rtol=1e-6)
    assert rank[2] == 0.0
    np.testing.assert_array_equal(scores, scores_before)


def test_stage_a_rank_degenerate_terms_contribute_zero():
    L = 3
    logits = torch.zeros((L, VOCAB_SIZE))
    x_t = torch.tensor([5, MASK_TOKEN_ID, MASK_TOKEN_ID])
    scores = np.array([-0.1, -np.inf, -np.inf])
    rank = compute_stage_a_rank_score(
        structural_logits=logits,
        x_t=x_t,
        scores=scores,
        mask_token_id=MASK_TOKEN_ID,
        m_i=None,
        d2_evidence=np.zeros(L, dtype=np.float64),
        alpha_struct=0.3,
        lambda_commit=1.0,
        d2_evidence_nu=1.0,
        zscore_epsilon=1.0e-6,
    )
    np.testing.assert_allclose(rank, np.zeros(L, dtype=np.float64))


# ---------------------------------------------------------------------------
# Stage B D3 evidence-source firewall (PLAN_RF_UNI_CTRL.md Task B5)
# ---------------------------------------------------------------------------


def test_d3_evidence_input_legacy_keeps_window_excess():
    cfg = D3Config(enabled=True, evidence_source="legacy_window_excess")
    out = compute_d3_evidence_input(
        d3_config=cfg, legacy_e_i=np.array([1.0]), typed_fresh_i=np.array([9.0])
    )
    np.testing.assert_allclose(out, [1.0])


def test_d3_evidence_input_typed_fresh_uses_e_fresh():
    cfg = D3Config(enabled=True, evidence_source="typed_fresh")
    out = compute_d3_evidence_input(
        d3_config=cfg, legacy_e_i=np.array([1.0]), typed_fresh_i=np.array([9.0])
    )
    np.testing.assert_allclose(out, [9.0])


def test_d3_evidence_input_typed_fresh_requires_typed_array():
    cfg = D3Config(enabled=True, evidence_source="typed_fresh")
    with pytest.raises(ValueError, match="typed_fresh"):
        compute_d3_evidence_input(
            d3_config=cfg, legacy_e_i=np.array([1.0]), typed_fresh_i=None
        )
