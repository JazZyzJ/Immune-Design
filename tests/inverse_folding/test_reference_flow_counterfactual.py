"""Phase D2 counterfactual pure-function contract tests (PLAN_RF.md §D2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.counterfactual import (
    apply_logit_correction,
    build_candidate_support,
    build_safe_candidate_support,
    compute_context_pnll,
    compute_context_pnll_from_log_probs,
    compute_delta_logit,
    compute_ensemble_diagnostics,
    compute_ess,
    compute_feasibility,
    compute_local_risk,
    compute_Q_B_per_candidate,
    compute_weights,
    enumerate_candidates,
    project_marginals,
    select_editable_positions,
)


CANONICAL_TOKENS = (10, 11, 12, 13, 14)  # 5-symbol pseudo-AA alphabet
SPECIAL_TOKENS = (0, 1, 2, 3)            # mask/pad/cls/eos style
MASK_TOKEN_ID = 0
VOCAB_SIZE = 20


def _struct_logits(seed: int = 0, length: int = 6) -> torch.Tensor:
    """Reproducible (L, V) logits where canonical tokens dominate over specials."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=-5.0, scale=0.1, size=(length, VOCAB_SIZE)).astype(np.float32)
    for i in range(length):
        # boost canonical tokens with distinct per-position structure
        for j, tok in enumerate(CANONICAL_TOKENS):
            logits[i, tok] = float(rng.normal(loc=1.0 + 0.2 * j, scale=0.3))
        # specials stay deeply negative (won't be picked as top-K canonical)
        for tok in SPECIAL_TOKENS:
            logits[i, tok] = -50.0
    return torch.from_numpy(logits)


# ---------------------------------------------------------------------------
# select_editable_positions
# ---------------------------------------------------------------------------


def test_select_editable_positions_uses_excess_then_entropy_tiebreak():
    L = 6
    x_t = torch.tensor([MASK_TOKEN_ID] * L, dtype=torch.long)
    residue_excess = np.array([0.0, 0.5, 0.5, 0.9, 0.1, 0.0])
    per_pos_entropy = torch.tensor([3.0, 1.0, 0.5, 2.0, 2.5, 1.5])

    chosen = select_editable_positions(
        start_0b=1,
        end_0b=5,
        x_t=x_t,
        mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess,
        per_pos_entropy=per_pos_entropy,
        max_positions=2,
    )
    # Top excess at idx 3 (0.9), then 1 vs 2 tied at 0.5 -> lower entropy (idx 2 = 0.5).
    assert chosen == (3, 2)


def test_select_editable_positions_skips_committed_positions():
    L = 4
    x_t = torch.tensor([20, MASK_TOKEN_ID, 21, MASK_TOKEN_ID], dtype=torch.long)
    residue_excess = np.array([2.0, 1.0, 0.5, 0.3])
    per_pos_entropy = torch.zeros(L)
    chosen = select_editable_positions(
        start_0b=0,
        end_0b=4,
        x_t=x_t,
        mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess,
        per_pos_entropy=per_pos_entropy,
        max_positions=4,
    )
    assert set(chosen) == {1, 3}  # committed positions 0, 2 excluded


def test_select_editable_positions_v_target_source_picks_high_v_low_excess():
    # B.1 (PLAN_RF_UNI_CTRL.md Stage B.1): within-block ranking by v_target.
    L = 6
    x_t = torch.tensor([MASK_TOKEN_ID] * L, dtype=torch.long)
    # legacy excess peaks at idx 1; v_target peaks at idx 4 (low legacy there).
    residue_excess = np.array([0.0, 0.9, 0.1, 0.0, 0.05, 0.0])
    v_target = np.array([0.0, 0.1, 0.2, 0.0, 0.95, 0.0])
    per_pos_entropy = torch.zeros(L)
    legacy = select_editable_positions(
        start_0b=0, end_0b=6, x_t=x_t, mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess, per_pos_entropy=per_pos_entropy, max_positions=1,
    )
    assert legacy == (1,)  # legacy ranks by residue_excess
    typed = select_editable_positions(
        start_0b=0, end_0b=6, x_t=x_t, mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess, per_pos_entropy=per_pos_entropy, max_positions=1,
        score_source="v_target", typed_field=v_target,
    )
    assert typed == (4,)  # v_target ranks by the typed field (low legacy excess there)


def test_select_editable_positions_v_target_keeps_low_entropy_tiebreak():
    L = 5
    x_t = torch.tensor([MASK_TOKEN_ID] * L, dtype=torch.long)
    residue_excess = np.zeros(L)
    v_target = np.array([0.0, 0.5, 0.5, 0.0, 0.0])  # idx 1,2 tied
    per_pos_entropy = torch.tensor([3.0, 2.0, 0.5, 1.0, 1.0])
    chosen = select_editable_positions(
        start_0b=0, end_0b=5, x_t=x_t, mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess, per_pos_entropy=per_pos_entropy, max_positions=1,
        score_source="v_target", typed_field=v_target,
    )
    assert chosen == (2,)  # tie on v_target → lower entropy wins (idx 2 = 0.5)


def test_select_editable_positions_v_target_falls_back_to_legacy_when_field_absent():
    L = 4
    x_t = torch.tensor([MASK_TOKEN_ID] * L, dtype=torch.long)
    residue_excess = np.array([0.1, 0.9, 0.2, 0.0])
    per_pos_entropy = torch.zeros(L)
    chosen = select_editable_positions(
        start_0b=0, end_0b=4, x_t=x_t, mask_token_id=MASK_TOKEN_ID,
        residue_excess=residue_excess, per_pos_entropy=per_pos_entropy, max_positions=1,
        score_source="v_target", typed_field=None,  # absent → legacy fallback
    )
    assert chosen == (1,)


# ---------------------------------------------------------------------------
# build_candidate_support
# ---------------------------------------------------------------------------


def test_candidate_support_uses_top_k_canonical_and_excludes_specials():
    logits = _struct_logits(seed=42, length=4)
    K = build_candidate_support(
        struct_logits=logits,
        positions=(0, 2),
        top_k=3,
        canonical_token_ids=CANONICAL_TOKENS,
    )
    assert set(K.keys()) == {0, 2}
    for pos, tokens in K.items():
        assert len(tokens) == 3
        # No special token leaks in.
        assert all(tok in CANONICAL_TOKENS for tok in tokens)
        assert all(tok not in SPECIAL_TOKENS for tok in tokens)


def test_candidate_support_top_k_clamped_to_canonical_alphabet():
    logits = _struct_logits(seed=1, length=2)
    K = build_candidate_support(
        struct_logits=logits,
        positions=(0,),
        top_k=99,
        canonical_token_ids=CANONICAL_TOKENS,
    )
    assert len(K[0]) == len(CANONICAL_TOKENS)


def test_safe_candidate_support_filters_by_structural_logprob_drop_and_keeps_top1():
    logits = torch.full((1, VOCAB_SIZE), -50.0)
    logits[0, 10] = 5.0
    logits[0, 11] = 4.3
    logits[0, 12] = 3.4
    logits[0, 13] = -1.0
    K = build_safe_candidate_support(
        struct_logits=logits,
        positions=(0,),
        top_k=4,
        canonical_token_ids=CANONICAL_TOKENS,
        delta_struct=1.0,
    )
    assert K[0] == (10, 11)

    K_top1 = build_safe_candidate_support(
        struct_logits=logits,
        positions=(0,),
        top_k=1,
        canonical_token_ids=CANONICAL_TOKENS,
        delta_struct=1.0,
    )
    assert K_top1[0] == (10,)


# ---------------------------------------------------------------------------
# enumerate_candidates
# ---------------------------------------------------------------------------


def test_enumerate_cartesian_below_budget():
    K = {0: (10, 11), 1: (12, 13)}
    cands, mode = enumerate_candidates(
        positions=(0, 1),
        K_i_per_pos=K,
        max_candidates=8,
        struct_logits=_struct_logits(seed=0, length=2),
        seed_tuple=("P", 0, 0, 0),
    )
    assert mode == "cartesian"
    assert set(cands) == {(10, 12), (10, 13), (11, 12), (11, 13)}


def test_enumerate_sampled_above_budget_is_deterministic():
    # 5 positions * top-4 = 1024 cart product; force sampled path with budget 6.
    L = 5
    logits = _struct_logits(seed=7, length=L)
    K = build_candidate_support(
        struct_logits=logits,
        positions=tuple(range(L)),
        top_k=4,
        canonical_token_ids=CANONICAL_TOKENS,
    )
    cands_a, mode_a = enumerate_candidates(
        positions=tuple(range(L)),
        K_i_per_pos=K,
        max_candidates=6,
        struct_logits=logits,
        seed_tuple=("Q", 1, 2, 3),
    )
    cands_b, mode_b = enumerate_candidates(
        positions=tuple(range(L)),
        K_i_per_pos=K,
        max_candidates=6,
        struct_logits=logits,
        seed_tuple=("Q", 1, 2, 3),
    )
    assert mode_a == mode_b == "sampled"
    assert cands_a == cands_b  # determinism under same seed_tuple
    assert len(cands_a) == 6


def test_enumerate_sampled_retains_duplicates():
    # Use highly peaked logits so the same candidate is sampled repeatedly.
    L = 1
    logits = torch.full((L, VOCAB_SIZE), -50.0)
    logits[0, 10] = 30.0
    logits[0, 11] = -20.0
    K = {0: (10, 11)}
    cands, mode = enumerate_candidates(
        positions=(0,),
        K_i_per_pos=K,
        max_candidates=10,   # > cartesian product (2) -> sampled
        struct_logits=logits,
        seed_tuple=("X",),
    )
    # Cart product is 2 ≤ 10, so this lands on cartesian. Make the budget tighter.
    # Force sampled by making cartesian larger than budget via duplicated dimensions.
    K2 = {0: (10, 11), 1: (10, 11), 2: (10, 11), 3: (10, 11)}
    logits2 = torch.full((4, VOCAB_SIZE), -50.0)
    for i in range(4):
        logits2[i, 10] = 30.0
        logits2[i, 11] = -20.0
    cands2, mode2 = enumerate_candidates(
        positions=(0, 1, 2, 3),
        K_i_per_pos=K2,
        max_candidates=8,    # cart = 16
        struct_logits=logits2,
        seed_tuple=("X",),
    )
    assert mode2 == "sampled"
    # Logits force token 10 at every position with overwhelming probability.
    # Duplicates are kept; the all-10 tuple should appear multiple times.
    assert cands2.count((10, 10, 10, 10)) >= 7


def test_enumerate_sampled_uses_safe_support_and_temperature():
    logits = torch.full((3, VOCAB_SIZE), -50.0)
    for i in range(3):
        logits[i, 10] = 20.0
        logits[i, 11] = 19.0
        logits[i, 12] = -20.0
    K = {0: (10, 11), 1: (10, 11), 2: (10, 11)}
    cands, mode = enumerate_candidates(
        positions=(0, 1, 2),
        K_i_per_pos=K,
        max_candidates=4,
        struct_logits=logits,
        seed_tuple=("safe", 1),
        struct_temperature=0.5,
    )
    assert mode == "sampled"
    assert len(cands) == 4
    assert all(all(tok in {10, 11} for tok in cand) for cand in cands)
    assert len(set(cands)) < len(cands)


# ---------------------------------------------------------------------------
# Local risk + per-candidate Q_B
# ---------------------------------------------------------------------------


def test_compute_local_risk_only_uses_omega_indices():
    window_risks = [1.0, 2.0, 0.5, 3.0]
    r_full = compute_local_risk(
        window_risks=window_risks,
        omega_indices=range(len(window_risks)),
        aggregation="LME",
    )
    r_partial = compute_local_risk(
        window_risks=window_risks,
        omega_indices=(1, 3),
        aggregation="LME",
    )
    assert r_partial != r_full
    # LME{2.0, 3.0} = 3 + log((e^(-1)+e^0)/2) ≈ 3 + log(0.6839) ≈ 2.6201
    assert r_partial == pytest.approx(
        3.0 + math.log((math.exp(-1.0) + math.exp(0.0)) / 2.0), abs=1e-9
    )


def test_compute_Q_B_factorizes_over_positions():
    L = 2
    logits = torch.full((L, VOCAB_SIZE), -50.0)
    logits[0, 10] = 1.0
    logits[0, 11] = 0.0
    logits[1, 12] = 0.0
    logits[1, 13] = -1.0
    K = {0: (10, 11), 1: (12, 13)}
    candidates = ((10, 12), (10, 13), (11, 12), (11, 13))
    Q = compute_Q_B_per_candidate(
        candidates=candidates,
        positions=(0, 1),
        K_i_per_pos=K,
        struct_logits=logits,
    )
    # Per-position softmax over K is the same after subtracting max.
    p0 = np.exp([1.0, 0.0])
    p0 /= p0.sum()
    p1 = np.exp([0.0, -1.0])
    p1 /= p1.sum()
    expected = np.array([
        p0[0] * p1[0],
        p0[0] * p1[1],
        p0[1] * p1[0],
        p0[1] * p1[1],
    ])
    np.testing.assert_allclose(Q, expected, rtol=1e-6)
    assert Q.sum() == pytest.approx(1.0, abs=1e-6)


def test_compute_Q_B_uses_struct_temperature_after_safe_filtering():
    logits = torch.full((1, VOCAB_SIZE), -50.0)
    logits[0, 10] = 2.0
    logits[0, 11] = 0.0
    K = {0: (10, 11)}
    Q = compute_Q_B_per_candidate(
        candidates=((10,), (11,)),
        positions=(0,),
        K_i_per_pos=K,
        struct_logits=logits,
        struct_temperature=2.0,
    )
    expected = np.exp([1.0, 0.0])
    expected = expected / expected.sum()
    np.testing.assert_allclose(Q, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Weights / ESS / marginals
# ---------------------------------------------------------------------------


def test_weights_cartesian_includes_Q_B_factor():
    Q = np.array([0.5, 0.3, 0.2])
    dR = np.array([-1.0, 0.0, 2.0])
    w = compute_weights(mode="cartesian", Q_B=Q, delta_R_B=dR, beta=1.0)
    np.testing.assert_allclose(w, Q * np.exp(-dR), rtol=1e-9)


def test_weights_sampled_excludes_Q_B_factor():
    Q = np.array([0.5, 0.3, 0.2])
    dR = np.array([-1.0, 0.0, 2.0])
    w = compute_weights(mode="sampled", Q_B=Q, delta_R_B=dR, beta=1.0)
    np.testing.assert_allclose(w, np.exp(-dR), rtol=1e-9)


def test_compute_ess_zero_safe():
    assert compute_ess(np.array([0.0, 0.0, 0.0])) == 0.0
    assert compute_ess(np.array([1.0, 1.0, 1.0])) == pytest.approx(3.0)
    # Highly peaked → ESS ≈ 1
    w = np.array([10.0, 1e-8, 1e-8])
    assert compute_ess(w) == pytest.approx(1.0, abs=1e-3)


def test_project_marginals_sums_to_one_per_position():
    candidates = ((10, 12), (10, 13), (11, 12), (11, 13))
    K = {0: (10, 11), 1: (12, 13)}
    w = np.array([0.25, 0.25, 0.25, 0.25])
    marginals = project_marginals(
        candidates=candidates,
        weights_normalized=w,
        positions=(0, 1),
        K_i_per_pos=K,
    )
    for p in (0, 1):
        assert sum(marginals[p].values()) == pytest.approx(1.0, abs=1e-9)
    assert marginals[0][10] == pytest.approx(0.5)
    assert marginals[1][12] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Delta-logit construction
# ---------------------------------------------------------------------------


def test_beta_zero_produces_zero_correction():
    Q = np.array([0.5, 0.3, 0.2])
    dR = np.array([-1.0, 0.0, 2.0])
    w_cart = compute_weights(mode="cartesian", Q_B=Q, delta_R_B=dR, beta=0.0)
    np.testing.assert_allclose(w_cart, Q, rtol=1e-9)
    # Normalized cart weights == Q_B; so pi_B == Q_B; delta_logit = 0 everywhere.
    # We verify through the full pipeline below.

    K = {0: (10, 11, 12)}
    candidates = ((10,), (11,), (12,))
    w_norm = w_cart / w_cart.sum()
    pi = project_marginals(
        candidates=candidates, weights_normalized=w_norm, positions=(0,), K_i_per_pos=K
    )
    Q_marg = project_marginals(
        candidates=candidates, weights_normalized=Q / Q.sum(), positions=(0,), K_i_per_pos=K
    )
    delta = compute_delta_logit(
        pi_marginals=pi, Q_marginals=Q_marg, eta=1.0, rho_B=1.0, epsilon=1e-8, max_abs_shift=5.0,
    )
    for shift in delta.values():
        assert abs(shift) < 1e-9


def test_delta_logit_clipped_to_max_abs_shift():
    # pi puts all mass on tok 11; Q puts all mass on tok 10. Unclipped log-ratio is huge.
    K = {0: (10, 11)}
    pi = {0: {10: 1e-12, 11: 1.0 - 1e-12}}
    Q = {0: {10: 1.0 - 1e-12, 11: 1e-12}}
    delta = compute_delta_logit(
        pi_marginals=pi, Q_marginals=Q, eta=1.0, rho_B=1.0, epsilon=1e-8, max_abs_shift=3.0,
    )
    assert delta[(0, 11)] == pytest.approx(3.0, abs=1e-6)
    assert delta[(0, 10)] == pytest.approx(-3.0, abs=1e-6)


def test_apply_logit_correction_only_touches_supported_entries():
    L = 3
    logits = torch.zeros(L, VOCAB_SIZE)
    delta = {(1, 10): 0.5, (2, 11): -0.25}
    corrected = apply_logit_correction(struct_logits=logits, delta_logit=delta)
    # Cloned, not in-place
    assert corrected is not logits
    assert torch.allclose(logits, torch.zeros_like(logits))
    # Supported entries shifted
    assert corrected[1, 10].item() == pytest.approx(0.5)
    assert corrected[2, 11].item() == pytest.approx(-0.25)
    # Outside-support entries unchanged
    for i in range(L):
        for tok in range(VOCAB_SIZE):
            if (i, tok) in delta:
                continue
            assert corrected[i, tok].item() == 0.0


# ---------------------------------------------------------------------------
# Feasibility gate
# ---------------------------------------------------------------------------


def test_feasibility_passes_when_any_candidate_improves_enough():
    dR = np.array([0.1, -0.5, 0.3])
    feasible, best = compute_feasibility(delta_R_B=dR, min_delta_R_improvement=0.0)
    assert feasible is True
    assert best == pytest.approx(-0.5)


def test_feasibility_fails_when_no_candidate_meets_threshold():
    dR = np.array([0.1, -0.05, 0.3])
    feasible, best = compute_feasibility(delta_R_B=dR, min_delta_R_improvement=0.1)
    # best is -0.05 but threshold needs ΔR < -0.1; fails.
    assert feasible is False
    assert best == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Stage A context reliability and local ensemble diagnostics
# ---------------------------------------------------------------------------


def test_context_pnll_uses_committed_context_and_skips_masks():
    logits = torch.full((4, VOCAB_SIZE), -5.0)
    logits[0, 10] = 5.0
    logits[1, 11] = -1.0
    logits[2, 12] = 5.0
    x_t = torch.tensor([10, MASK_TOKEN_ID, 12, MASK_TOKEN_ID])
    pnll = compute_context_pnll(
        struct_logits=logits,
        x_t=x_t,
        mask_token_id=MASK_TOKEN_ID,
        start_0b=0,
        end_0b=4,
    )
    assert pnll is not None
    assert pnll < 1e-2
    assert compute_context_pnll(
        struct_logits=logits,
        x_t=torch.tensor([MASK_TOKEN_ID] * 4),
        mask_token_id=MASK_TOKEN_ID,
        start_0b=0,
        end_0b=4,
    ) is None


def test_context_pnll_precomputed_log_probs_matches_legacy_helper():
    logits = _struct_logits(seed=9, length=7)
    x_t = torch.tensor([10, MASK_TOKEN_ID, 12, 13, MASK_TOKEN_ID, 11, 14])
    log_probs = torch.log_softmax(logits, dim=-1).detach().cpu()

    for start, end in [(0, 7), (1, 5), (4, 6), (6, 7)]:
        legacy = compute_context_pnll(
            struct_logits=logits,
            x_t=x_t,
            mask_token_id=MASK_TOKEN_ID,
            start_0b=start,
            end_0b=end,
        )
        precomputed = compute_context_pnll_from_log_probs(
            log_probs=log_probs,
            x_t=x_t,
            mask_token_id=MASK_TOKEN_ID,
            start_0b=start,
            end_0b=end,
        )
        if legacy is None:
            assert precomputed is None
        else:
            assert precomputed == pytest.approx(legacy)


def test_ensemble_diagnostics_use_mean_variance_and_sign_consistency_without_gating():
    argmax_delta = np.array([-2.0, -0.5, 0.2])
    ensemble = {
        0: np.array([-1.0, -1.5, -2.0]),
        1: np.array([0.2, -0.1, 0.4]),
    }
    diag = compute_ensemble_diagnostics(
        argmax_delta_R=argmax_delta,
        ensemble_delta_R_by_candidate=ensemble,
    )
    assert diag.mean_delta_R[0] == pytest.approx(-1.5)
    assert diag.std_delta_R[0] == pytest.approx(np.std([-1.0, -1.5, -2.0]))
    assert diag.sign_consistency[0] == pytest.approx(1.0)
    assert diag.sign_consistency[1] == pytest.approx(1.0 / 3.0)
    assert diag.rank_flip_rate >= 0.0


# ---------------------------------------------------------------------------
# End-to-end: beneficial candidate increases mass on lower-risk token
# ---------------------------------------------------------------------------


def test_beneficial_candidate_shifts_mass_toward_lower_risk():
    # Two candidates at one position; cand A reduces risk a lot, cand B does not.
    K = {0: (10, 11)}
    candidates = ((10,), (11,))
    Q = np.array([0.5, 0.5])  # structural neutral
    dR = np.array([-2.0, 0.0])
    w = compute_weights(mode="cartesian", Q_B=Q, delta_R_B=dR, beta=1.0)
    w_norm = w / w.sum()
    pi = project_marginals(
        candidates=candidates, weights_normalized=w_norm, positions=(0,), K_i_per_pos=K
    )
    Q_marg = project_marginals(
        candidates=candidates, weights_normalized=Q / Q.sum(), positions=(0,), K_i_per_pos=K
    )
    delta = compute_delta_logit(
        pi_marginals=pi, Q_marginals=Q_marg, eta=1.0, rho_B=1.0, epsilon=1e-8, max_abs_shift=10.0,
    )
    # Beneficial token 10 must receive a positive shift; risky 11 negative.
    assert delta[(0, 10)] > 0.0
    assert delta[(0, 11)] < 0.0
