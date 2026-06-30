"""Allocation-layer pure-helper tests (PLAN_PLANNER_SC_GR.md Task 2)."""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.allocation import (
    allocation_mass,
    reweight_by_allocation,
    smooth_window_max,
    stable_seed,
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
