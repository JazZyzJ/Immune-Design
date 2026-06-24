"""Task U6 controller-misinterpretation telemetry tests — PLAN_URICASE_ENZYME_MODE U6.

Synthetic in-memory fixtures only (hand-built records / DataFrames). No real run
files are read. Validates ``build_enzyme_controller_overlap``:
- anchor-overlapping vs non-overlapping blocks,
- window_excess sum/max over window_indices,
- safe_support_size reductions,
- selected_hard_anchor_count == 0 and remasked_hard_anchor_count == 0 (clean fixture),
- realized_benefit_rate from D2 event rows,
- final_persistence_rate when a selected position is remasked at a later refresh_step,
- visible_noneditable_pressure_flag True/False cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from inverse_folding.reference_flow.enzyme_controller import (
    build_enzyme_controller_overlap,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
ANCHOR_SET = frozenset({10, 11, 12})  # hard-anchor residue index_0b set


def _r_windows():
    # window id -> span; ids 0,1,2,3 used by blocks below
    return [
        {"start_0b": 5, "end_0b": 15, "k": 4, "z": 0.5},   # 0 (overlaps anchors 10,11,12)
        {"start_0b": 30, "end_0b": 40, "k": 3, "z": 0.2},  # 1
        {"start_0b": 60, "end_0b": 70, "k": 2, "z": 0.1},  # 2 (non-overlap)
        {"start_0b": 70, "end_0b": 80, "k": 2, "z": 0.1},  # 3 (non-overlap)
    ]


def _refresh_record(
    *,
    protein_id="Q00511",
    design_idx=0,
    seed=7,
    refresh_step=0,
    step=100,
    t=0.4,
    window_excess,
    active_blocks,
    d2_block_diagnostics,
):
    return {
        "protein_id": protein_id,
        "design_idx": design_idx,
        "seed": seed,
        "refresh_step": refresh_step,
        "step": step,
        "t": t,
        "window_excess": window_excess,
        "r_windows_dyn": _r_windows(),
        "active_blocks": active_blocks,
        "d2_block_diagnostics": d2_block_diagnostics,
    }


def _block(block_id, start, end, window_indices):
    return {
        "block_id": block_id,
        "residue_start_0b": start,
        "residue_end_0b": end,
        "window_indices": window_indices,
        "g_time": 0.0,
        "g_comp": 0.0,
        "g_ent": 0.0,
        "g_ESS": 0.0,
        "rho_B": 0.0,
        "completion_fraction": 0.0,
        "mean_struct_entropy": 0.0,
    }


def _d2diag(
    block_id,
    *,
    feasibility=True,
    best_delta=-0.3,
    safe_support_sizes,
    corrected_positions,
):
    return {
        "block_id": block_id,
        "candidate_feasibility": feasibility,
        "best_delta_R_B": best_delta,
        "safe_support_sizes": safe_support_sizes,
        "corrected_positions": corrected_positions,
    }


def _events_empty():
    return pd.DataFrame(
        {
            "protein_id": pd.Series([], dtype=object),
            "design_idx": pd.Series([], dtype=int),
            "refresh_step": pd.Series([], dtype=int),
            "step": pd.Series([], dtype=int),
            "t": pd.Series([], dtype=float),
            "event_type": pd.Series([], dtype=object),
            "block_id": pd.Series([], dtype=object),
            "position_i": pd.Series([], dtype=int),
            "delta_R_B": pd.Series([], dtype=float),
            "realized_benefit_flag": pd.Series([], dtype=object),
            "remask_flag": pd.Series([], dtype=object),
            "sticky_selected_flag": pd.Series([], dtype=object),
            "reason": pd.Series([], dtype=object),
        }
    )


def _clean_two_block_record():
    """One refresh record: block A overlaps anchors, block B does not."""
    window_excess = [2.5, 0.0, 0.0, 1.0]  # indexed by window id
    blocks = [
        _block("A", 5, 15, [0]),     # overlaps anchors 10,11,12 via residue span
        _block("B", 60, 80, [2, 3]),  # non-overlapping
    ]
    diags = [
        _d2diag(
            "A",
            feasibility=True,
            best_delta=-0.3,
            safe_support_sizes={6: 4, 7: 2, 13: 8},
            corrected_positions=[6, 7, 13],  # all non-anchor, inside block A
        ),
        _d2diag(
            "B",
            feasibility=True,
            best_delta=-0.5,
            safe_support_sizes={61: 5, 62: 9},
            corrected_positions=[61, 62],
        ),
    ]
    return _refresh_record(
        window_excess=window_excess,
        active_blocks=blocks,
        d2_block_diagnostics=diags,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_one_row_per_block_and_basic_keys():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    assert len(out) == 2
    keyset = set(zip(out["protein_id"], out["design_idx"], out["refresh_step"], out["block_id"]))
    assert keyset == {("Q00511", 0, 0, "A"), ("Q00511", 0, 0, "B")}
    rowA = out[out["block_id"] == "A"].iloc[0]
    assert rowA["seed"] == 7
    assert rowA["step"] == 100
    assert rowA["t"] == pytest.approx(0.4)
    assert rowA["block_residue_start_0b"] == 5
    assert rowA["block_residue_end_0b"] == 15


def test_anchor_overlap_vs_non_overlap_block():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]

    assert bool(rowA["overlaps_hard_anchor"]) is True
    assert sorted(rowA["covered_anchor_indices"]) == [10, 11, 12]

    assert bool(rowB["overlaps_hard_anchor"]) is False
    assert list(rowB["covered_anchor_indices"]) == []


def test_window_indices_and_spans_expansion():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]

    assert list(rowA["window_indices"]) == [0]
    assert [list(s) for s in rowA["window_spans_0b"]] == [[5, 15]]

    assert list(rowB["window_indices"]) == [2, 3]
    assert [list(s) for s in rowB["window_spans_0b"]] == [[60, 70], [70, 80]]


def test_window_excess_sum_and_max_over_window_indices():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]

    # window_excess = [2.5, 0.0, 0.0, 1.0]
    # block A uses window 0 -> {2.5}
    assert rowA["window_excess_sum"] == pytest.approx(2.5)
    assert rowA["window_excess_max"] == pytest.approx(2.5)
    # block B uses windows 2,3 -> {0.0, 1.0}
    assert rowB["window_excess_sum"] == pytest.approx(1.0)
    assert rowB["window_excess_max"] == pytest.approx(1.0)


def test_safe_support_size_reductions_and_values():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    # safe_support_sizes A = {6:4,7:2,13:8} -> values 4,2,8
    assert sorted(rowA["safe_support_size_values"]) == [2, 4, 8]
    assert rowA["safe_support_size_min"] == 2
    assert rowA["safe_support_size_mean"] == pytest.approx((4 + 2 + 8) / 3)


def test_empty_safe_support_sizes_yields_na():
    window_excess = [0.0, 0.0, 0.0, 0.0]
    blocks = [_block("A", 5, 15, [0])]
    diags = [
        _d2diag(
            "A",
            safe_support_sizes={},
            corrected_positions=[],
        )
    ]
    rec = _refresh_record(
        window_excess=window_excess,
        active_blocks=blocks,
        d2_block_diagnostics=diags,
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert pd.isna(row["safe_support_size_min"])
    assert pd.isna(row["safe_support_size_mean"])
    assert list(row["safe_support_size_values"]) == []


def test_selected_positions_and_zero_anchor_count_clean():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]

    assert list(rowA["selected_positions"]) == [6, 7, 13]
    assert int(rowA["selected_hard_anchor_count"]) == 0
    assert int(rowB["selected_hard_anchor_count"]) == 0
    assert int(rowA["remasked_hard_anchor_count"]) == 0
    assert int(rowB["remasked_hard_anchor_count"]) == 0


def test_selected_flank_in_anchor_window_count():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]
    # block A overlaps anchor; all 3 selected positions are non-anchor flanks
    assert int(rowA["selected_flank_in_anchor_window_count"]) == 3
    # block B does not overlap anchor -> 0
    assert int(rowB["selected_flank_in_anchor_window_count"]) == 0


def test_candidate_feasibility_and_best_delta_joined_by_block_id():
    rec = _clean_two_block_record()
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]
    assert bool(rowA["candidate_feasibility"]) is True
    assert rowA["best_delta_R_B"] == pytest.approx(-0.3)
    assert rowB["best_delta_R_B"] == pytest.approx(-0.5)


def test_missing_d2_diag_yields_na_feasibility_and_delta():
    window_excess = [0.0, 0.0, 0.0, 0.0]
    blocks = [_block("A", 5, 15, [0])]
    rec = _refresh_record(
        window_excess=window_excess,
        active_blocks=blocks,
        d2_block_diagnostics=[],  # no diag for block A
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert pd.isna(row["candidate_feasibility"])
    assert pd.isna(row["best_delta_R_B"])
    assert list(row["selected_positions"]) == []


def test_realized_benefit_rate_from_d2_events():
    rec = _clean_two_block_record()
    # D2 events for block A: flags True, False, None -> mean over non-null = 1/2
    events = pd.DataFrame(
        [
            dict(protein_id="Q00511", design_idx=0, refresh_step=0, step=100, t=0.4,
                 event_type="D2", block_id="A", position_i=6, delta_R_B=-0.1,
                 realized_benefit_flag=True, remask_flag=False,
                 sticky_selected_flag=True, reason="d2"),
            dict(protein_id="Q00511", design_idx=0, refresh_step=0, step=100, t=0.4,
                 event_type="D2", block_id="A", position_i=7, delta_R_B=-0.2,
                 realized_benefit_flag=False, remask_flag=False,
                 sticky_selected_flag=True, reason="d2"),
            dict(protein_id="Q00511", design_idx=0, refresh_step=0, step=100, t=0.4,
                 event_type="D2", block_id="A", position_i=13, delta_R_B=-0.05,
                 realized_benefit_flag=None, remask_flag=False,
                 sticky_selected_flag=True, reason="d2"),
            # a non-D2 row must be ignored for the rate
            dict(protein_id="Q00511", design_idx=0, refresh_step=0, step=100, t=0.4,
                 event_type="monitor", block_id="A", position_i=6, delta_R_B=0.0,
                 realized_benefit_flag=True, remask_flag=False,
                 sticky_selected_flag=False, reason="mon"),
        ]
    )
    out = build_enzyme_controller_overlap([rec], events, ANCHOR_SET)
    rowA = out[out["block_id"] == "A"].iloc[0]
    rowB = out[out["block_id"] == "B"].iloc[0]
    assert rowA["realized_benefit_rate"] == pytest.approx(0.5)
    # block B has no D2 rows -> NA
    assert pd.isna(rowB["realized_benefit_rate"])


def test_final_persistence_rate_uses_sampler_step_not_refresh_step():
    # block A at refresh_step 0, sampler step 100, selects [6,7,13]. "later remask"
    # must be judged by sampler STEP, not refresh_step: a remask in the SAME refresh
    # but at a later step still un-commits the position.
    rec = _refresh_record(
        refresh_step=0,
        step=100,
        window_excess=[2.5, 0.0, 0.0, 1.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", safe_support_sizes={6: 4, 7: 2, 13: 8},
                    corrected_positions=[6, 7, 13])
        ],
    )
    events = pd.DataFrame(
        [
            # remask of position 7 at a later refresh (step 120 > 100) -> not persistent
            dict(protein_id="Q00511", design_idx=0, refresh_step=1, step=120, t=0.3,
                 event_type="remask", block_id="A", position_i=7, delta_R_B=0.0,
                 realized_benefit_flag=None, remask_flag=True,
                 sticky_selected_flag=False, reason="remask"),
            # remask of position 6 in the SAME refresh_step 0 but at a LATER sampler
            # step (110 > 100) -> MUST count as later (the refresh_step-only bug missed this)
            dict(protein_id="Q00511", design_idx=0, refresh_step=0, step=110, t=0.35,
                 event_type="remask", block_id="A", position_i=6, delta_R_B=0.0,
                 realized_benefit_flag=None, remask_flag=True,
                 sticky_selected_flag=False, reason="remask"),
        ]
    )
    out = build_enzyme_controller_overlap([rec], events, ANCHOR_SET)
    row = out.iloc[0]
    # positions 6 and 7 both later-remasked, only 13 persists -> 1/3
    assert row["final_persistence_rate"] == pytest.approx(1 / 3)


def test_final_persistence_rate_na_when_no_selected_positions():
    rec = _refresh_record(
        window_excess=[0.0, 0.0, 0.0, 0.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", safe_support_sizes={}, corrected_positions=[])
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert pd.isna(row["final_persistence_rate"])


def test_remasked_hard_anchor_count_attributed_design_level():
    # A remask hitting an anchor position (11) somewhere in the design trajectory.
    # This must NOT happen on a clean run; this fixture deliberately violates it to
    # check the count is computed and attributed design-level.
    recs = [
        _refresh_record(
            refresh_step=0,
            window_excess=[2.5, 0.0, 0.0, 1.0],
            active_blocks=[_block("A", 5, 15, [0]), _block("B", 60, 80, [2, 3])],
            d2_block_diagnostics=[
                _d2diag("A", safe_support_sizes={6: 4}, corrected_positions=[6]),
                _d2diag("B", safe_support_sizes={61: 5}, corrected_positions=[61]),
            ],
        )
    ]
    events = pd.DataFrame(
        [
            dict(protein_id="Q00511", design_idx=0, refresh_step=2, step=130, t=0.2,
                 event_type="remask", block_id="A", position_i=11, delta_R_B=0.0,
                 realized_benefit_flag=None, remask_flag=True,
                 sticky_selected_flag=False, reason="remask"),
            # another remask on a non-anchor position -> not counted
            dict(protein_id="Q00511", design_idx=0, refresh_step=2, step=130, t=0.2,
                 event_type="remask", block_id="A", position_i=6, delta_R_B=0.0,
                 realized_benefit_flag=None, remask_flag=True,
                 sticky_selected_flag=False, reason="remask"),
        ]
    )
    out = build_enzyme_controller_overlap(recs, events, ANCHOR_SET)
    # both rows belong to design (Q00511, 0); each gets the design-level count = 1
    assert set(out["remasked_hard_anchor_count"]) == {1}
    assert len(out) == 2


def test_visible_noneditable_pressure_flag_true_infeasible():
    # overlaps anchor, window_excess_max > 0, no anchor selected, infeasible candidate
    rec = _refresh_record(
        window_excess=[2.5, 0.0, 0.0, 0.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", feasibility=False, best_delta=None,
                    safe_support_sizes={6: 4}, corrected_positions=[6]),
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert bool(row["visible_noneditable_pressure_flag"]) is True


def test_visible_noneditable_pressure_flag_true_nonimproving_delta():
    # feasible but best_delta_R_B >= 0 -> no useful improvement -> flag True
    rec = _refresh_record(
        window_excess=[2.5, 0.0, 0.0, 0.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", feasibility=True, best_delta=0.1,
                    safe_support_sizes={6: 4}, corrected_positions=[6]),
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert bool(row["visible_noneditable_pressure_flag"]) is True


def test_visible_noneditable_pressure_flag_false_feasible_improving():
    # feasible AND best_delta_R_B < 0 -> repairable -> flag False
    rec = _refresh_record(
        window_excess=[2.5, 0.0, 0.0, 0.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", feasibility=True, best_delta=-0.4,
                    safe_support_sizes={6: 4}, corrected_positions=[6]),
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert bool(row["visible_noneditable_pressure_flag"]) is False


def test_visible_noneditable_pressure_flag_false_no_overlap():
    # non-overlapping block can never raise the flag even if infeasible
    rec = _refresh_record(
        window_excess=[0.0, 0.0, 5.0, 0.0],
        active_blocks=[_block("B", 60, 80, [2, 3])],
        d2_block_diagnostics=[
            _d2diag("B", feasibility=False, best_delta=None,
                    safe_support_sizes={61: 5}, corrected_positions=[61]),
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert bool(row["visible_noneditable_pressure_flag"]) is False


def test_visible_noneditable_pressure_flag_false_zero_excess():
    # overlaps anchor + infeasible, but window_excess_max == 0 -> flag False
    rec = _refresh_record(
        window_excess=[0.0, 0.0, 0.0, 0.0],
        active_blocks=[_block("A", 5, 15, [0])],
        d2_block_diagnostics=[
            _d2diag("A", feasibility=False, best_delta=None,
                    safe_support_sizes={6: 4}, corrected_positions=[6]),
        ],
    )
    out = build_enzyme_controller_overlap([rec], _events_empty(), ANCHOR_SET)
    row = out.iloc[0]
    assert bool(row["visible_noneditable_pressure_flag"]) is False


def test_empty_inputs_return_empty_frame_with_columns():
    out = build_enzyme_controller_overlap([], _events_empty(), ANCHOR_SET)
    assert len(out) == 0
    for col in (
        "protein_id", "design_idx", "refresh_step", "block_id",
        "overlaps_hard_anchor", "window_excess_sum", "window_excess_max",
        "visible_noneditable_pressure_flag", "realized_benefit_rate",
        "final_persistence_rate", "remasked_hard_anchor_count",
    ):
        assert col in out.columns


def test_multiple_designs_isolated_trajectories():
    # remask in design 1 must not affect persistence of design 0
    recs = [
        _refresh_record(
            design_idx=0, refresh_step=0,
            window_excess=[2.5, 0.0, 0.0, 0.0],
            active_blocks=[_block("A", 5, 15, [0])],
            d2_block_diagnostics=[
                _d2diag("A", safe_support_sizes={6: 4}, corrected_positions=[6]),
            ],
        ),
        _refresh_record(
            design_idx=1, refresh_step=0,
            window_excess=[2.5, 0.0, 0.0, 0.0],
            active_blocks=[_block("A", 5, 15, [0])],
            d2_block_diagnostics=[
                _d2diag("A", safe_support_sizes={6: 4}, corrected_positions=[6]),
            ],
        ),
    ]
    events = pd.DataFrame(
        [
            # remask position 6 in design 1 only, later refresh_step
            dict(protein_id="Q00511", design_idx=1, refresh_step=3, step=140, t=0.1,
                 event_type="remask", block_id="A", position_i=6, delta_R_B=0.0,
                 realized_benefit_flag=None, remask_flag=True,
                 sticky_selected_flag=False, reason="remask"),
        ]
    )
    out = build_enzyme_controller_overlap(recs, events, ANCHOR_SET)
    row0 = out[out["design_idx"] == 0].iloc[0]
    row1 = out[out["design_idx"] == 1].iloc[0]
    # design 0 position 6 never remasked -> persistence 1.0
    assert row0["final_persistence_rate"] == pytest.approx(1.0)
    # design 1 position 6 remasked later -> persistence 0.0
    assert row1["final_persistence_rate"] == pytest.approx(0.0)
