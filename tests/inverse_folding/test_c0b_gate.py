"""C0b coordination gate join + aggregation (scripts/analysis/c0b_gate.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import c0b_gate  # noqa: E402


def _cand(arm, facade, r_b, register_id="P__d0__r10", pid="P", r0=6.0, short=True):
    return dict(
        protein_id=pid, register_id=register_id, arm=arm, R_B=r_b, R0=r0,
        shortlisted=short, facade_design_idx=facade,
    )


def _candidates():
    return pd.DataFrame([
        _cand("design", 0, float("nan"), register_id="P__design"),  # baseline fold
        _cand("single", 1, 5.0),
        _cand("single", 2, 4.8),
        _cand("fake", 3, 4.0),
        _cand("fake", 4, 3.0),     # lowest-immune fake but wrecks the fold (below)
        _cand("joint", 5, 3.7),    # "joint" == true reconditioned
        _cand("joint", 6, 3.5),
        # an unfolded, never-scored true draw — MUST be ignored by the gate
        _cand("joint", -1, 1.0, short=False),
    ])


def _structural():
    return pd.DataFrame([
        dict(protein_id="P", design_idx=0, scTM=0.90),   # design baseline -> bar 0.85
        dict(protein_id="P", design_idx=1, scTM=0.90),
        dict(protein_id="P", design_idx=2, scTM=0.40),   # single below floor
        dict(protein_id="P", design_idx=3, scTM=0.88),   # fake accept
        dict(protein_id="P", design_idx=4, scTM=0.60),   # fake wrecks fold -> reject
        dict(protein_id="P", design_idx=5, scTM=0.90),   # true accept
        dict(protein_id="P", design_idx=6, scTM=0.87),   # true accept
    ])


def test_compute_gate_coordination_win_with_design_baseline():
    verdicts, go = c0b_gate.compute_gate(candidates=_candidates(), structural=_structural())
    assert len(verdicts) == 1  # the __design pseudo-register is excluded
    row = verdicts.iloc[0]
    assert row["R_single_best"] == pytest.approx(5.0)
    assert row["R_fake_best"] == pytest.approx(4.0)   # the 3.0 fake folds badly
    assert row["R_true_best"] == pytest.approx(3.5)   # NOT the unfolded 1.0
    assert row["coordination_margin"] == pytest.approx(0.5)
    assert row["dof_margin"] == pytest.approx(1.0)
    assert bool(row["coordination_wins"]) is True
    assert go["go"] is True and go["n_total"] == 1 and go["n_wins"] == 1


def test_compute_gate_fake_infeasible_still_wins():
    # All folded fakes wreck the structure -> no acceptable fake, but true folds.
    struct = _structural()
    struct.loc[struct["design_idx"] == 3, "scTM"] = 0.60  # the one good fake now bad
    verdicts, go = c0b_gate.compute_gate(candidates=_candidates(), structural=struct)
    row = verdicts.iloc[0]
    assert bool(row["fake_acceptable"]) is False
    assert bool(row["true_acceptable"]) is True
    assert bool(row["coordination_wins"]) is True
    assert go["go"] is True


def test_compute_gate_dof_without_coordination_is_no_go():
    # Make the best fold-acceptable fake nearly tie the true joint -> coord ~ 0.
    cands = _candidates()
    cands.loc[cands["facade_design_idx"] == 3, "R_B"] = 3.6  # acceptable fake ties true 3.5
    verdicts, go = c0b_gate.compute_gate(candidates=cands, structural=_structural())
    row = verdicts.iloc[0]
    assert row["coordination_margin"] == pytest.approx(0.1)
    assert row["dof_margin"] > 0.5            # large degrees-of-freedom value
    assert bool(row["coordination_wins"]) is False
    assert go["go"] is False


def test_compute_gate_depth_sweep_rescue_through_gate():
    # true@18 ties fake (coord 0.1) but the deeper true@36 clears it -> GO + needs_depth.
    cands = pd.DataFrame([
        _cand("design", 0, float("nan"), register_id="P__design"),
        _cand("single", 1, 5.0),
        _cand("fake", 2, 3.6),
        _cand("joint", 3, 3.5),        # true @18
        _cand("joint_deep", 4, 3.0),   # true @36
    ])
    struct = pd.DataFrame([
        dict(protein_id="P", design_idx=i, scTM=0.90) for i in range(5)
    ])
    verdicts, go = c0b_gate.compute_gate(candidates=cands, structural=struct)
    row = verdicts.iloc[0]
    assert bool(row["coordination_wins_18"]) is False
    assert bool(row["coordination_wins"]) is True
    assert bool(row["needs_depth"]) is True
    assert go["go"] is True
