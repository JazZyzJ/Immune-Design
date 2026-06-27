"""Allocation-layer pure-helper tests (PLAN_PLANNER_SC_GR.md Task 2)."""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.allocation import (
    allocation_mass,
    reweight_by_allocation,
    smooth_window_max,
    stable_seed,
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
