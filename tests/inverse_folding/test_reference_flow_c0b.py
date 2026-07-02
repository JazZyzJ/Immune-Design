"""C0b (Path-C headroom existence test) pure helpers (PLANNER_README §C0b).

These are the unit-testable, scorer/decoder-FREE primitives the ``--mode c0b``
driver composes: high-r_i register selection, single-position edit enumeration,
the two-arm (single-position vs joint) per-register verdict, and the GO fraction.
The DPLM joint decode + head scoring are integration-tested on the cluster.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from inverse_folding.reference_flow import signal_diag as sd


# ---------------------------------------------------------------------------
# select_high_ri_registers
# ---------------------------------------------------------------------------


def test_select_high_ri_registers_picks_top_nonoverlapping_windows():
    # Two clear high-r_i plateaus: pos 5-9 (mean 10) and pos 15-19 (mean 8).
    risks = [0.0] * 5 + [10.0] * 5 + [0.0] * 5 + [8.0] * 5
    regs = sd.select_high_ri_registers(
        residue_risks=risks, register_width=5, max_registers=2, quantile=0.5
    )
    assert regs == ((5, 6, 7, 8, 9), (15, 16, 17, 18, 19))


def test_select_high_ri_registers_respects_max_and_separation():
    risks = [0.0] * 5 + [10.0] * 5 + [0.0] * 5 + [8.0] * 5
    regs = sd.select_high_ri_registers(
        residue_risks=risks, register_width=5, max_registers=1, quantile=0.5
    )
    assert regs == ((5, 6, 7, 8, 9),)  # only the single highest window


def test_select_high_ri_registers_no_overlap_between_selected():
    # A single broad hot band: greedy must NOT return overlapping width-5 windows.
    risks = [0.0] * 4 + [9.0, 9.5, 10.0, 9.5, 9.0, 8.0, 7.0] + [0.0] * 4
    regs = sd.select_high_ri_registers(
        residue_risks=risks, register_width=5, max_registers=3, quantile=0.0
    )
    starts = [r[0] for r in regs]
    for a, b in zip(starts, starts[1:]):
        assert b - a >= 5  # min_separation defaults to register_width


def test_select_high_ri_registers_too_short_returns_empty():
    assert sd.select_high_ri_registers(
        residue_risks=[1.0, 2.0, 3.0], register_width=5, max_registers=2
    ) == ()


def test_select_high_ri_registers_quantile_filters_low_windows():
    # Only the top window clears a high quantile; the rest are filtered out.
    risks = [0.0] * 5 + [10.0] * 5 + [1.0] * 5
    regs = sd.select_high_ri_registers(
        residue_risks=risks, register_width=5, max_registers=5, quantile=0.9
    )
    assert regs == ((5, 6, 7, 8, 9),)


# ---------------------------------------------------------------------------
# single_position_candidate_tuples
# ---------------------------------------------------------------------------


def test_single_position_candidate_tuples_one_diff_excludes_identity():
    out = sd.single_position_candidate_tuples(
        design_register_tokens=(1, 2, 3),
        register_positions=(10, 11, 12),
        safe_support={10: (1, 4), 11: (2, 5, 6), 12: (3,)},
    )
    # pos10: 4 (skip identity 1); pos11: 5,6 (skip identity 2); pos12: only 3 == identity -> none
    assert out == ((4, 2, 3), (1, 5, 3), (1, 6, 3))


def test_single_position_candidate_tuples_empty_when_all_identity():
    out = sd.single_position_candidate_tuples(
        design_register_tokens=(7, 8),
        register_positions=(0, 1),
        safe_support={0: (7,), 1: (8,)},
    )
    assert out == ()


# ---------------------------------------------------------------------------
# compute_c0b_register_verdict  (three arms: single / fake / true)
# ---------------------------------------------------------------------------


def test_c0b_verdict_coordination_wins_when_true_beats_fake():
    # design baseline 0.90, tol 0.05 -> structure bar = 0.85.
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[4.0, 3.0], sctm_fake=[0.88, 0.60],   # idx1 (best immune) wrecks fold
        r_true=[3.7, 3.5], sctm_true=[0.90, 0.87],   # reconditioned folds stay good
        design_sctm=0.90, coordination_margin_thresh=0.2, sctm_tol=0.05,
    )
    assert v["R_single_best"] == pytest.approx(5.0)
    assert v["R_fake_best"] == pytest.approx(4.0)   # NOT 3.0 (that one folds badly)
    assert v["R_true_best"] == pytest.approx(3.5)
    assert v["coordination_margin"] == pytest.approx(0.5)
    assert v["dof_margin"] == pytest.approx(1.0)
    assert v["coordination_wins"] is True


def test_c0b_verdict_fake_infeasible_is_a_coordination_win():
    # Fake's only low-immune draw wrecks the fold (no structure-acceptable fake),
    # while the reconditioned true joint folds fine -> coordination strictly needed.
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[2.0], sctm_fake=[0.60],     # below bar 0.85 -> no acceptable fake
        r_true=[3.5], sctm_true=[0.90],
        design_sctm=0.90, sctm_tol=0.05,
    )
    assert v["fake_acceptable"] is False
    assert v["true_acceptable"] is True
    assert math.isnan(v["coordination_margin"])
    assert v["coordination_wins"] is True


def test_c0b_verdict_dof_without_coordination_is_not_a_win():
    # Fake (9 independent edits) folds AND nearly ties true post-structure: large
    # dof but coordination ~ 0 -> NO win (raise the cap, don't build a sampler).
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[3.6], sctm_fake=[0.90],
        r_true=[3.5], sctm_true=[0.90],
        design_sctm=0.90, coordination_margin_thresh=0.2, sctm_tol=0.05,
    )
    assert v["coordination_margin"] == pytest.approx(0.1)
    assert v["dof_margin"] == pytest.approx(1.4)
    assert v["coordination_wins"] is False


def test_c0b_verdict_infeasible_true_is_not_a_win():
    # The true joint itself can't fold acceptably -> Path C not justified here.
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[3.5], sctm_fake=[0.90],
        r_true=[3.0], sctm_true=[0.40],     # below floor
        design_sctm=0.90, sctm_tol=0.05,
    )
    assert v["true_acceptable"] is False
    assert v["coordination_wins"] is False


def test_c0b_verdict_no_baseline_uses_floor_only():
    # design_sctm missing -> acceptance is the absolute foldability floor (>0.5).
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.60],
        r_fake=[4.0], sctm_fake=[0.60],
        r_true=[3.5], sctm_true=[0.60],
        design_sctm=None, coordination_margin_thresh=0.2,
    )
    assert v["coordination_margin"] == pytest.approx(0.5)
    assert v["coordination_wins"] is True


def test_c0b_verdict_depth_sweep_rescues_undercooked_true():
    # true@18 only ties fake (coord 0.1 < 0.2), but the deeper true@36 clears it:
    # NOT a NO-GO — the 18-step true was under-cooked (the false-NO-GO trap).
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[3.6], sctm_fake=[0.90],
        r_true=[3.5], sctm_true=[0.90],          # @18: coord = 0.1
        r_true_deep=[3.0], sctm_true_deep=[0.90],  # @36: coord = 0.6
        design_sctm=0.90, coordination_margin_thresh=0.2, sctm_tol=0.05,
    )
    assert v["R_true_best"] == pytest.approx(3.0)            # best of {18, 36}
    assert v["coordination_wins_18"] is False
    assert v["coordination_wins"] is True
    assert v["needs_depth"] is True
    assert v["depth_gap"] == pytest.approx(0.5)              # >0 ⇒ deeper helps


def test_c0b_verdict_depth_flat_is_clean_no_go():
    # true~fake at BOTH depths and the depth gap ~ 0 -> coordination is genuinely
    # useless (clean NO-GO -> raise the cap), not an under-cooked artifact.
    v = sd.compute_c0b_register_verdict(
        r_single=[5.0], sctm_single=[0.90],
        r_fake=[3.6], sctm_fake=[0.90],
        r_true=[3.5], sctm_true=[0.90],
        r_true_deep=[3.55], sctm_true_deep=[0.90],
        design_sctm=0.90, coordination_margin_thresh=0.2, sctm_tol=0.05,
    )
    assert v["coordination_wins"] is False
    assert v["needs_depth"] is False
    assert abs(v["depth_gap"]) < 0.1                        # ≈ 0 ⇒ deeper buys nothing


# ---------------------------------------------------------------------------
# c0b_go_fraction
# ---------------------------------------------------------------------------


def test_c0b_go_fraction_go_when_at_least_one_third_win():
    out = sd.c0b_go_fraction(joint_wins_flags=[True, False, True, False, False], win_fraction=1 / 3)
    assert out["fraction"] == pytest.approx(0.4)
    assert out["go"] is True
    assert out["n_total"] == 5
    assert out["n_wins"] == 2


def test_c0b_go_fraction_no_go_below_threshold():
    out = sd.c0b_go_fraction(joint_wins_flags=[True, False, False, False, False, False])
    assert out["fraction"] == pytest.approx(1 / 6)
    assert out["go"] is False


def test_c0b_go_fraction_empty_is_no_go():
    out = sd.c0b_go_fraction(joint_wins_flags=[])
    assert out["go"] is False
    assert out["n_total"] == 0
