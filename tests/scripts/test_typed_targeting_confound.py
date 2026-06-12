"""Unit tests for the Stage B B4.4 in-block confound diagnostic.

Hand-computed expectations on a synthetic two-artifact join exercise the
per-block v_target percentile, the legacy_excess=0 mechanism rate (NOT b_cur),
the stratified realized_benefit framework, and the empty/filter edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.typed_targeting_confound import compute_typed_targeting_confound


def _actionability_df() -> pd.DataFrame:
    # One refresh, residues 0..5. Block 0 = residues {0,1,2}; residues 3,4,5
    # are outside any block. v_target rises across the block; legacy excess is 0
    # except residue 2 (the b_cur-vs-legacy divergence is encoded here).
    rows = []
    spec = {
        0: (1.0, 0.0, 0),
        1: (2.0, 0.0, 0),
        2: (3.0, 0.5, 0),
        3: (0.0, 0.0, -1),
        4: (0.0, 0.0, -1),
        5: (0.0, 0.0, -1),
    }
    for res, (vt, leg, blk) in spec.items():
        rows.append(
            {
                "protein_id": "P1",
                "design_idx": 0,
                "seed": 42,
                "refresh_step": 0,
                "residue_index_0b": res,
                "v_target": vt,
                "legacy_residue_excess": leg,
                "active_block_id": blk,
                "active_target_flag": blk >= 0,
            }
        )
    return pd.DataFrame(rows)


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # D2 edit at the low-v_target / legacy_excess=0 position: no benefit.
            {
                "protein_id": "P1", "design_idx": 0, "seed": 42, "refresh_step": 0,
                "event_type": "D2", "position_i": 0, "block_id": 0,
                "realized_benefit_flag": False,
            },
            # D2 edit at the high-v_target position: benefit.
            {
                "protein_id": "P1", "design_idx": 0, "seed": 42, "refresh_step": 0,
                "event_type": "D2", "position_i": 2, "block_id": 0,
                "realized_benefit_flag": True,
            },
            # A non-D2 row must be ignored.
            {
                "protein_id": "P1", "design_idx": 0, "seed": 42, "refresh_step": 0,
                "event_type": "monitor", "position_i": 1, "block_id": 0,
                "realized_benefit_flag": None,
            },
        ]
    )


def test_confound_metrics_match_hand_computed_values():
    m = compute_typed_targeting_confound(
        _actionability_df(), _events_df(), high_vtarget_quantile=0.75
    )
    assert m["n_d2_edits"] == 2
    # pos0 percentile = 1/3, pos2 percentile = 3/3 → per-block mean = 0.6667.
    assert m["d2_edit_vtarget_percentile"] == pytest.approx((1 / 3 + 1.0) / 2)
    # in-block edits: pos0 legacy=0 (True), pos2 legacy=0.5 (False) → 0.5.
    assert m["d2_edit_on_zero_legacy_excess_rate"] == pytest.approx(0.5)
    assert m["d2_edit_active_target_rate"] == pytest.approx(1.0)
    # high-v_target stratum = {pos2} → benefit 1.0; legacy=0 stratum = {pos0} → 0.0.
    assert m["realized_benefit_rate_high_vtarget"] == pytest.approx(1.0)
    assert m["n_edits_high_vtarget"] == 1
    assert m["realized_benefit_rate_zero_legacy_excess"] == pytest.approx(0.0)
    assert m["n_edits_zero_legacy_excess"] == 1
    assert m["realized_benefit_rate_overall"] == pytest.approx(0.5)


def test_confound_uses_legacy_excess_not_b_cur():
    # The legacy-excess mechanism rate must reflect legacy_residue_excess, so a
    # block whose legacy excess is 0 everywhere reads as 100% entropy-only picks
    # even though v_target (b_cur-like) is high.
    act = _actionability_df().copy()
    act.loc[act["residue_index_0b"] == 2, "legacy_residue_excess"] = 0.0
    m = compute_typed_targeting_confound(act, _events_df())
    assert m["d2_edit_on_zero_legacy_excess_rate"] == pytest.approx(1.0)


def test_confound_empty_events_returns_zero_count():
    empty = _events_df().iloc[0:0]
    m = compute_typed_targeting_confound(_actionability_df(), empty)
    assert m["n_d2_edits"] == 0
    assert m["d2_edit_vtarget_percentile"] is None
    assert m["realized_benefit_rate_high_vtarget"] is None
