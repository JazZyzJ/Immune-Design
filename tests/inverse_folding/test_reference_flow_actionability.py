"""Stage B typed-actionability pure operators (PLAN_RF_UNI_CTRL.md Task B1/B3).

These cover the residue-level ``A_i(t)`` field math in isolation from the
controller: max-covering projection, the anchored SoftOR union, the
proposal-envelope peak*consistency aggregation, the fresh-evidence /
memory firewall, and the target-vs-pressure split. Controller integration
is exercised separately in the D1/D2-D3 controller tests.
"""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.actionability import (
    cluster_support_multiplier,
    cluster_supported_pressure,
    compute_fresh_evidence,
    compute_target_evidence,
    envelope_burden_from_excess_samples,
    excess_over_tau,
    global_pressure_mass,
    global_pressure_scalar,
    max_covering_window_projection,
    positive_evidence_weight,
    prominence_thresholded_mass,
    protein_pressure_burden,
    smoothstep_pressure,
    soft_or,
    update_memory,
)


def test_max_covering_window_projection_preserves_focal_peak():
    windows = [
        {"start": 0, "end": 3, "score": 0.2},
        {"start": 2, "end": 5, "score": 1.7},
    ]
    out = max_covering_window_projection(length=5, windows=windows)
    np.testing.assert_allclose(out, [0.2, 0.2, 1.7, 1.7, 1.7])


def test_soft_or_zero_single_and_multi_source_properties():
    assert soft_or([0.0, 0.0, 0.0], tau=0.5) == 0.0
    assert np.isclose(soft_or([1.2, 0.0], tau=0.5), 1.2)
    both = soft_or([1.2, 0.8], tau=0.5)
    assert both >= 1.2
    assert both < 2.0


# ---------------------------------------------------------------------------
# Task B3: field functions + firewall
# ---------------------------------------------------------------------------


def test_positive_evidence_weight_applies_floor():
    out = positive_evidence_weight(np.array([0.0, 0.5, 1.0]), floor=0.25)
    np.testing.assert_allclose(out, [0.25, 0.625, 1.0])


def test_excess_over_tau_clamps_at_zero():
    out = excess_over_tau(np.array([0.1, 0.5, 1.0]), tau_ref=0.4)
    np.testing.assert_allclose(out, [0.0, 0.1, 0.6])


def test_env_burden_uses_peak_times_consistency_for_small_k():
    samples = np.array([
        [1.0, 0.0],
        [0.0, 0.5],
        [0.0, 0.7],
    ])
    out = envelope_burden_from_excess_samples(samples, consistency_floor=1.0 / 3.0)
    np.testing.assert_allclose(out, [1.0 * (1 / 3 + 2 / 3 * 1 / 3), 0.7 * (1 / 3 + 2 / 3 * 2 / 3)])


def test_update_memory_cold_start_zero_then_decays_by_half_life():
    gamma = 2.0 ** (-1.0 / 3.0)
    first = update_memory(
        previous_b_mem=None, e_fresh=np.array([1.0]), half_life_refreshes=3.0
    )
    np.testing.assert_allclose(first, [(1.0 - gamma) * 1.0])
    m = np.array([1.0])
    for _ in range(3):
        m = update_memory(
            previous_b_mem=m, e_fresh=np.array([0.0]), half_life_refreshes=3.0
        )
    np.testing.assert_allclose(m, [0.5], atol=1e-12)


def test_update_memory_cannot_grow_when_fresh_is_zero():
    m = update_memory(
        previous_b_mem=np.array([2.0]), e_fresh=np.array([0.0]), half_life_refreshes=3.0
    )
    assert m[0] < 2.0


def test_fresh_evidence_excludes_memory():
    fresh = compute_fresh_evidence(
        b_cur=np.array([1.0]),
        b_env=np.array([0.0]),
        r_ctx=np.array([0.0]),
        r_ctx_floor=0.25,
        tau=0.5,
    )
    target = compute_target_evidence(
        fresh_evidence=fresh,
        b_mem=np.array([10.0]),
        tau=0.5,
    )
    assert fresh[0] < 1.0
    assert target[0] > fresh[0]


def test_cluster_support_does_not_feed_active_targeting():
    v = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
    pressure = cluster_supported_pressure(
        v_target=v,
        tau=0.5,
        radius=1,
        min_mass=3.0,
        floor=0.25,
    )
    assert v[1] == 2.0
    assert pressure[1] < v[1]


def test_global_pressure_mass_is_length_normalized():
    out = global_pressure_mass(np.array([0.0, 1.0, 2.0, 1.0]))
    assert np.isclose(out, 1.0)


def test_cluster_support_multiplier_saturates_at_one():
    v = np.array([5.0, 5.0, 5.0])
    support = cluster_support_multiplier(v, tau=0.5, radius=1, min_mass=1.0)
    np.testing.assert_allclose(support, [1.0, 1.0, 1.0])


def test_cluster_supported_pressure_uses_multiplier():
    v = np.array([0.0, 2.0, 0.0])
    support = cluster_support_multiplier(v, tau=0.5, radius=1, min_mass=3.0)
    pressure = cluster_supported_pressure(
        v_target=v, tau=0.5, radius=1, min_mass=3.0, floor=0.25
    )
    np.testing.assert_allclose(pressure, v * (0.25 + 0.75 * support))


def test_global_pressure_scalar_monotonic_and_bounded():
    lo = global_pressure_scalar(-10.0, g_min=0.25, g_max=1.0, G0=0.0, s_G=1.0)
    mid = global_pressure_scalar(0.0, g_min=0.25, g_max=1.0, G0=0.0, s_G=1.0)
    hi = global_pressure_scalar(10.0, g_min=0.25, g_max=1.0, G0=0.0, s_G=1.0)
    assert 0.25 <= lo < mid < hi <= 1.0
    assert np.isclose(mid, 0.625)


# ---------------------------------------------------------------------------
# Task C1.1: Stage C.1 global-pressure smoothstep actuator
# ---------------------------------------------------------------------------


def test_smoothstep_pressure_hits_zero_for_low_burden():
    # Below B_low the clip pins q=0 -> s=0 -> g exactly g_min.
    assert smoothstep_pressure(0.1, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0) == 0.0


def test_smoothstep_pressure_hits_one_for_high_burden():
    # Above B_high the clip pins q=1 -> s=1 -> g exactly g_max.
    assert smoothstep_pressure(0.9, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0) == 1.0


def test_smoothstep_pressure_midpoint_is_half_envelope():
    # q=0.5 -> s=3(.25)-2(.125)=0.5 -> g = g_min + (g_max-g_min)*0.5.
    g = smoothstep_pressure(0.5, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0)
    assert np.isclose(g, 0.5)
    g2 = smoothstep_pressure(0.5, B_low=0.2, B_high=0.8, g_min=0.2, g_max=1.0)
    assert np.isclose(g2, 0.2 + 0.8 * 0.5)


def test_smoothstep_pressure_monotonic_and_clamped_between_floor_and_ceiling():
    xs = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    gs = [smoothstep_pressure(x, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0) for x in xs]
    assert gs == sorted(gs)
    assert all(0.0 <= g <= 1.0 for g in gs)


def test_smoothstep_pressure_rejects_non_positive_band():
    try:
        smoothstep_pressure(0.5, B_low=0.5, B_high=0.5, g_min=0.0, g_max=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError when B_high <= B_low")


def test_protein_pressure_burden_uses_median_not_latest_spike():
    assert protein_pressure_burden([0.02, 0.03, 1.20]) == 0.03


def test_prominence_thresholded_mass_counts_only_excess_above_tau():
    # (1/L) Σ ReLU(u - tau): ReLU([-10,-5,2,10]) = [0,0,2,10]; mean = 3.0.
    u = np.array([0.0, 5.0, 12.0, 20.0])
    assert np.isclose(prominence_thresholded_mass(u, tau_prom=10.0), 3.0)


def test_prominence_thresholded_mass_tau_zero_equals_mean_for_nonneg():
    # The Stage B byte-identity hinge: tau_prom=0 with u>=0 reduces to the mean.
    u = np.array([0.0, 1.0, 2.0, 1.0])
    assert np.isclose(
        prominence_thresholded_mass(u, tau_prom=0.0), float(u.mean())
    )
    assert np.isclose(
        prominence_thresholded_mass(u, tau_prom=0.0), global_pressure_mass(u)
    )


def test_prominence_thresholded_mass_empty_is_zero():
    assert prominence_thresholded_mass(np.array([]), tau_prom=5.0) == 0.0


def test_protein_pressure_burden_drops_non_finite_and_empty_is_zero():
    assert protein_pressure_burden([0.02, float("nan"), 0.04]) == 0.03
    assert protein_pressure_burden([]) == 0.0
    assert protein_pressure_burden([float("inf"), float("nan")]) == 0.0
