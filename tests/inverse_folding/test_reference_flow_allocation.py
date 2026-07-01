"""Allocation-layer pure-helper tests (PLAN_PLANNER_SC_GR.md Task 2)."""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.allocation import (
    allocation_mass,
    reweight_by_allocation,
    smooth_window_max,
    stable_seed,
    terminal_union_field,
    triage_field,
)


def test_smooth_window_max_register_grain():
    v = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(
        smooth_window_max(v, half_width=1), [0.0, 5.0, 5.0, 5.0, 0.0]
    )


def test_allocation_mass_is_mean_one():
    phi = allocation_mass(np.array([0.0, 0.0, 3.0, 1.0]), half_width=0)
    assert abs(float(phi.mean()) - 1.0) < 1e-9
    assert phi[2] > phi[0]


def test_allocation_mass_all_zero_is_uniform():
    np.testing.assert_allclose(
        allocation_mass(np.zeros(6), half_width=4), np.ones(6)
    )


def test_reweight_uniform_alloc_preserves_vtarget_order():
    v = np.array([0.2, 0.9, 0.5, 0.1])
    s = reweight_by_allocation(v, np.ones(4), c=1.0, eps=0.05)
    assert list(np.argsort(s)) == list(np.argsort(v))


def test_reweight_tilts_toward_high_alloc():
    s = reweight_by_allocation(
        np.array([1.0, 1.0]), np.array([0.2, 1.8]), c=1.0, eps=0.05
    )
    assert s[1] > s[0]


def test_stable_seed_is_process_stable():
    # value is fixed (NOT salted like builtin hash); recompute must match
    assert stable_seed("flat", "P123", 0) == stable_seed("flat", "P123", 0)
    assert stable_seed("flat", "P123", 0) != stable_seed("flat", "P123", 1)


# ---------------------------------------------------------------------------
# triage_field — §A1 revised r_i consumption (doc §8.4 Path A)
# ---------------------------------------------------------------------------


def test_triage_lambda0_is_v_target_order_within_eligible():
    v = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    s = triage_field(
        v, np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        eligible_quantile=0.5, triage_lambda=0.0,
    )
    elig = v >= np.quantile(v, 0.5)
    # ineligible get 0; eligible ordered exactly by v_target
    assert (s[~elig] == 0).all()
    e = np.where(elig)[0]
    assert list(e[np.argsort(s[e])]) == list(e[np.argsort(v[e])])


def test_triage_reorders_toward_high_phi_within_eligible():
    v = np.array([0.8, 0.82])  # both eligible, nearly tied
    s = triage_field(v, np.array([0.2, 1.8]), eligible_quantile=0.0, triage_lambda=0.5)
    assert s[1] > s[0]  # high-phi wins the tie-break


def test_triage_never_selects_ineligible_over_eligible():
    v = np.array([0.05, 0.9])  # idx0 ineligible
    s = triage_field(v, np.array([9.0, 0.1]), eligible_quantile=0.5, triage_lambda=1.0)
    assert s[1] > s[0]  # huge phi at idx0 cannot lift it past eligible idx1


def test_triage_empty_returns_empty():
    assert triage_field(
        np.empty(0), np.empty(0), eligible_quantile=0.5, triage_lambda=0.3
    ).size == 0


def test_triage_all_zero_v_target_selects_nothing():
    # strict eligibility (doc §8.4): zero-actionability sites are NEVER eligible,
    # even when the median is 0 — no τ_v widening.
    s = triage_field(
        np.zeros(6), np.array([0.2, 1.8, 0.5, 9.0, 0.1, 3.0]),
        eligible_quantile=0.5, triage_lambda=0.3,
    )
    assert (s == 0).all()


def test_triage_sparse_zero_excludes_zero_sites():
    # >50% zeros ⇒ median is 0; the OLD `v >= quantile(v, q)` admitted every zero.
    # Now eligibility is top-q AMONG strictly-positive actionability.
    v = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 3.0])
    s = triage_field(
        v, np.array([9.0, 9.0, 9.0, 9.0, 0.1, 0.1]),
        eligible_quantile=0.5, triage_lambda=1.0,
    )
    assert (s[:4] == 0).all()  # zero-v_target sites never selected, huge Φ notwithstanding
    assert s[4] > 0.0          # top positive eligible


def test_triage_floor_excludes_below_active_window_min_excess():
    # floor mirrors the pipeline's actionability floor (active_window_min_excess).
    v = np.array([0.05, 0.2, 1.0, 2.0])
    s = triage_field(
        v, np.ones(4), eligible_quantile=0.0, triage_lambda=0.0, floor=0.1,
    )
    assert s[0] == 0.0      # 0.05 <= floor 0.1 -> ineligible
    assert (s[1:] > 0).all()  # 0.2/1.0/2.0 > floor -> eligible


# ---------------------------------------------------------------------------
# terminal_union_field — §A3 terminal-aware active-register targeting
# (Fork A; doc §8.4). The principled INVERSE of triage: promotes high-Φ
# low-v_target registers via a SEPARATE terminal gate (never widens τ_v).
# ---------------------------------------------------------------------------


def test_terminal_union_promotes_low_vtarget_that_triage_zeros():
    # below-median (idx0) + above-median (idx4) v_target; Φ high at the below site.
    # A1 triage EXCLUDES idx0 (below the eligibility quantile); A3 PROMOTES it.
    v = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    phi = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    tri = triage_field(v, phi, eligible_quantile=0.5, triage_lambda=0.3)
    uni = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.9, terminal_lambda=0.3
    )
    assert tri[0] == 0.0   # triage: low-v_target site never activates
    assert uni[0] > 0.0    # A3: promoted into the active set


def test_terminal_union_uniform_phi_guard_admits_only_local():
    # flat Φ ⇒ terminal set empty ⇒ only local (v>floor) sites score; a dead
    # (v<=floor) site stays 0 (never widen to dead sites — RAR 0019 M4).
    v = np.array([0.0, 0.5, 1.0])
    phi = np.array([5.0, 5.0, 5.0])
    s = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.9, terminal_lambda=0.3
    )
    assert s[0] == 0.0     # v=0 is neither local nor terminal
    assert s[2] > s[1]     # over local, ordered by v_target (Φ flat)


def test_terminal_union_all_zero_vtarget_only_terminal_scores():
    # local empty (all v==0) ⇒ only terminal-hot sites get positive scores.
    v = np.zeros(6)
    phi = np.array([0.1, 0.2, 0.5, 9.0, 3.0, 0.1])
    s = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.5, terminal_lambda=0.3
    )
    assert s[3] > 0.0                    # top-Φ terminal site scores
    assert (s[[0, 1, 5]] == 0.0).all()   # below-median Φ, no local ⇒ 0


def test_norm_rank_ties_share_rank():
    # review P1: tie-aware ranks — equal values get the SAME rank (not broken by
    # original index), so a downstream additive Φ term can be the real tie-break.
    from inverse_folding.reference_flow.allocation import _norm_rank
    r = _norm_rank(np.zeros(5))
    assert np.allclose(r, r[0])                 # all equal ⇒ identical rank
    r2 = _norm_rank(np.array([3.0, 1.0, 2.0]))  # distinct ⇒ unchanged, ascending
    assert r2[1] < r2[2] < r2[0]


def test_terminal_union_tied_vtarget_orders_by_phi():
    # review P1 repro: all-zero v_target + multiple terminal sites ⇒ high Φ MUST
    # score above low Φ (Φ decides, not the position index). Was inverted before.
    v = np.zeros(5)
    phi = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    s = terminal_union_field(v, phi, terminal_eligible_quantile=0.0, terminal_lambda=0.3)
    assert s[0] > s[1] > s[2] > s[3] > s[4]     # strictly decreasing in Φ order


def test_triage_tied_positive_vtarget_orders_by_phi():
    # review P1: same for A1 — at tied positive v_target, Φ is the real tie-break.
    v = np.full(4, 0.5)
    phi = np.array([4.0, 3.0, 2.0, 1.0])
    s = triage_field(v, phi, eligible_quantile=0.0, triage_lambda=0.3)
    assert s[0] > s[1] > s[2] > s[3]


def test_terminal_union_sparse_phi_does_not_widen_to_dead_sites():
    # REGRESSION (verify-panel P1): the realistic allocation_mass output is SPARSE
    # (mostly zero — r_i is window-max-smoothed over a clipped, mostly-zero map).
    # A full-array quantile lands in the zero baseline and admits every dead site
    # (the falsified τ_v-widening, §8.2.1 / RAR 0019 M4). The positive-subset gate
    # must confine the terminal set to the genuine hot register only.
    L = 100
    v = np.zeros(L)  # all dead locally (low b_env everywhere)
    phi = np.zeros(L)
    phi[45:54] = np.array([2.0, 5.0, 9.0, 9.0, 9.0, 9.0, 8.0, 4.0, 2.0])  # one register
    s = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.9, terminal_lambda=0.3
    )
    promoted = np.flatnonzero(s > 0.0)
    assert promoted.size > 0                                  # the hot register scores
    assert set(promoted.tolist()).issubset(set(range(45, 54)))  # only inside it
    assert (s[:45] == 0.0).all() and (s[54:] == 0.0).all()   # 91 dead sites stay 0


def test_terminal_union_floor_only_via_terminal_gate():
    v = np.array([0.05, 0.5, 1.0])
    # uniform Φ ⇒ terminal empty ⇒ below-floor idx0 stays 0.
    s_flat = terminal_union_field(
        v, np.ones(3), terminal_eligible_quantile=0.5, terminal_lambda=0.3,
        floor=0.1,
    )
    assert s_flat[0] == 0.0
    # high Φ at idx0 ⇒ enters via the terminal gate despite v<=floor.
    s_hot = terminal_union_field(
        v, np.array([9.0, 1.0, 1.0]),
        terminal_eligible_quantile=0.5, terminal_lambda=0.3, floor=0.1,
    )
    assert s_hot[0] > 0.0


def test_terminal_union_lambda0_orders_by_vtarget_over_eligible():
    v = np.array([0.1, 0.9, 0.5, 0.7])
    phi = np.array([9.0, 0.1, 3.0, 0.5])  # non-uniform Φ affects eligibility only
    s = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.5, terminal_lambda=0.0
    )
    assert list(np.argsort(s)) == list(np.argsort(v))


def test_terminal_union_is_scale_invariant():
    v = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    phi = np.array([0.2, 1.8, 0.5, 9.0, 0.1])
    base = terminal_union_field(
        v, phi, terminal_eligible_quantile=0.7, terminal_lambda=0.4
    )
    scaled_v = terminal_union_field(
        2.0 * v, phi, terminal_eligible_quantile=0.7, terminal_lambda=0.4
    )
    scaled_phi = terminal_union_field(
        v, 3.0 * phi, terminal_eligible_quantile=0.7, terminal_lambda=0.4
    )
    np.testing.assert_allclose(base, scaled_v)
    np.testing.assert_allclose(base, scaled_phi)


def test_terminal_union_empty_returns_empty():
    assert terminal_union_field(
        np.empty(0), np.empty(0),
        terminal_eligible_quantile=0.9, terminal_lambda=0.3,
    ).size == 0
