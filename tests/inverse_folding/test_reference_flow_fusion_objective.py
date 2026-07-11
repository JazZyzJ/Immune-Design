"""F2 unit tests — fusion/objective.py Head responsibility, local delta, new-hotspot gate.

Contract: PLAN_RF_REFINE_FUSION.md §1.2 / §1.3. Whole-sequence global_risk drives selection;
the target register is the deterministic highest-raw-risk window with editable (non-anchor)
positions; local LME delta is telemetry, not acceptance; the parent-relative off-halo
new-hotspot burden (max/mass/count) is the hard gate signal. Window coordinates must align
exactly between parent and child.

Objects are duck-typed to head_scoring.WindowRiskRecord / HeadScore so this stays torch-free.
"""
from types import SimpleNamespace

import pytest

from inverse_folding.reference_flow.fusion import objective as ob


def W(start, end, k, z):
    return SimpleNamespace(start_0b=start, end_0b=end, k=k, z=z)


def S(windows, global_risk=1.0):
    return SimpleNamespace(windows=list(windows), global_risk=global_risk)


# --------------------------------------------------------------------------- #
# global risk fail-fast
# --------------------------------------------------------------------------- #
def test_global_risk_of_returns_value():
    assert ob.global_risk_of(S([], global_risk=2.5)) == pytest.approx(2.5)


def test_global_risk_of_none_is_hard_error():
    with pytest.raises(ob.ObjectiveError):
        ob.global_risk_of(S([], global_risk=None))


# --------------------------------------------------------------------------- #
# target register responsibility
# --------------------------------------------------------------------------- #
def test_select_target_picks_highest_z_window():
    score = S([W(0, 9, 9, 1.0), W(10, 19, 9, 3.0), W(20, 29, 9, 2.0)])
    tgt = ob.select_target_register(score, anchors=set())
    assert (tgt.start_0b, tgt.end_0b) == (10, 19)
    assert tgt.z == pytest.approx(3.0)
    assert tgt.editable_positions == tuple(range(10, 19))


def test_select_target_deterministic_tie_break():
    # equal z -> lower start_0b, then end_0b, then k
    score = S([W(30, 39, 9, 3.0), W(10, 19, 9, 3.0), W(10, 18, 8, 3.0)])
    tgt = ob.select_target_register(score, anchors=set())
    assert (tgt.start_0b, tgt.end_0b) == (10, 18)  # lowest start, then lowest end


def test_select_target_skips_fully_anchored_window():
    # highest-z window is entirely anchors -> fall back to the next window with editable slots
    score = S([W(0, 3, 3, 5.0), W(10, 13, 3, 4.0)])
    anchors = {0, 1, 2}
    tgt = ob.select_target_register(score, anchors=anchors)
    assert (tgt.start_0b, tgt.end_0b) == (10, 13)


def test_select_target_editable_excludes_anchors():
    score = S([W(10, 15, 5, 4.0)])
    tgt = ob.select_target_register(score, anchors={11, 13})
    assert tgt.editable_positions == (10, 12, 14)


def test_select_target_none_when_no_editable():
    score = S([W(0, 3, 3, 5.0)])
    assert ob.select_target_register(score, anchors={0, 1, 2}) is None


# --------------------------------------------------------------------------- #
# local target delta (LME) — telemetry, separate from global
# --------------------------------------------------------------------------- #
def test_local_target_delta_over_overlapping_windows():
    parent = S([W(0, 9, 9, 2.0), W(20, 29, 9, 5.0)], global_risk=3.0)
    child = S([W(0, 9, 9, 1.0), W(20, 29, 9, 5.0)], global_risk=3.5)  # lower local, worse global
    d = ob.local_target_delta(child, parent, target_start=1, target_end=4)
    assert d == pytest.approx(-1.0)  # only the overlapping (0,9,9) window counts
    # local improvement does not imply global improvement
    assert ob.global_risk_of(child) > ob.global_risk_of(parent)


def test_local_target_delta_alignment_mismatch_raises():
    parent = S([W(0, 9, 9, 2.0)])
    child = S([W(0, 10, 9, 1.0)])  # end differs -> coordinate mismatch
    with pytest.raises(ob.ObjectiveError):
        ob.local_target_delta(child, parent, target_start=1, target_end=4)


# --------------------------------------------------------------------------- #
# parent-relative off-halo new-hotspot gate
# --------------------------------------------------------------------------- #
def test_new_hotspot_offhalo_max_mass_count():
    parent = S([W(0, 9, 9, 1.0), W(20, 29, 9, 1.0), W(40, 49, 9, 1.0)])
    child = S([W(0, 9, 9, 5.0),   # +4 but INSIDE halo -> ignored
               W(20, 29, 9, 3.0),  # +2 off halo -> counts
               W(40, 49, 9, 0.5)]) # decrease -> 0
    nh = ob.new_hotspot(child, parent, halo_start=0, halo_end=10)
    assert nh.max_increase == pytest.approx(2.0)
    assert nh.positive_mass == pytest.approx(2.0)
    assert nh.positive_count == 1


def test_new_hotspot_zero_when_no_offhalo_increase():
    parent = S([W(20, 29, 9, 2.0)])
    child = S([W(20, 29, 9, 1.0)])  # decrease
    nh = ob.new_hotspot(child, parent, halo_start=0, halo_end=100)  # everything in halo
    assert nh.max_increase == pytest.approx(0.0)
    assert nh.positive_count == 0


def test_new_hotspot_alignment_mismatch_raises():
    parent = S([W(0, 9, 9, 1.0), W(20, 29, 9, 1.0)])
    child = S([W(0, 9, 9, 1.0)])  # missing a window -> coordinate-set mismatch
    with pytest.raises(ob.ObjectiveError):
        ob.new_hotspot(child, parent, halo_start=0, halo_end=10)


# --------------------------------------------------------------------------- #
# adversarial: non-finite Head values must fail closed, never silently pass
# --------------------------------------------------------------------------- #
def test_global_risk_non_finite_is_hard_error():
    with pytest.raises(ob.ObjectiveError):
        ob.global_risk_of(S([], global_risk=float("nan")))
    with pytest.raises(ob.ObjectiveError):
        ob.global_risk_of(S([], global_risk=float("inf")))


def test_non_finite_window_z_in_new_hotspot_is_error():
    parent = S([W(0, 9, 9, 1.0)])
    child = S([W(0, 9, 9, float("nan"))])  # corrupted z must NOT become max(0, nan)=0
    with pytest.raises(ob.ObjectiveError):
        ob.new_hotspot(child, parent, halo_start=50, halo_end=60)


def test_non_finite_window_z_in_local_delta_is_error():
    parent = S([W(0, 9, 9, 1.0)])
    child = S([W(0, 9, 9, float("inf"))])
    with pytest.raises(ob.ObjectiveError):
        ob.local_target_delta(child, parent, target_start=1, target_end=4)


def test_select_target_rejects_non_finite_z():
    with pytest.raises(ob.ObjectiveError):
        ob.select_target_register(S([W(10, 15, 5, float("nan"))]), anchors=set())
