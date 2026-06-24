"""Shared enzyme-telemetry foundation tests — PLAN_URICASE_ENZYME_MODE U5/U6 (B5 + overlap)."""

from __future__ import annotations

import pandas as pd
import pytest

from inverse_folding.reference_flow.enzyme_telemetry import (
    anchor_overlap,
    normalize_design_keys,
)


def test_normalize_reconstructs_design_id_from_idx():
    df = pd.DataFrame({"protein_id": ["Q00511"], "design_idx": [3]})
    out = normalize_design_keys(df)
    assert out.loc[0, "design_id"] == "design_0003"
    assert int(out.loc[0, "design_idx"]) == 3


def test_normalize_reconstructs_design_idx_from_id():
    df = pd.DataFrame({"protein_id": ["Q00511"], "design_id": ["design_0007"]})
    out = normalize_design_keys(df)
    assert int(out.loc[0, "design_idx"]) == 7
    assert out.loc[0, "design_id"] == "design_0007"


def test_normalize_duplicate_design_is_hard_error():
    df = pd.DataFrame(
        {"protein_id": ["Q00511", "Q00511"], "design_idx": [0, 0]}
    )
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        normalize_design_keys(df, unique=True)


def test_normalize_inconsistent_design_id_and_idx_is_hard_error():
    # Both keys present but disagree (design_0007 vs idx 3) -> hard error.
    df = pd.DataFrame(
        {"protein_id": ["Q00511"], "design_idx": [3], "design_id": ["design_0007"]}
    )
    with pytest.raises(ValueError, match=r"(?i)inconsistent|mismatch"):
        normalize_design_keys(df)


def test_normalize_consistent_design_id_and_idx_ok():
    df = pd.DataFrame(
        {"protein_id": ["Q00511"], "design_idx": [7], "design_id": ["design_0007"]}
    )
    out = normalize_design_keys(df)
    assert int(out.loc[0, "design_idx"]) == 7
    assert out.loc[0, "design_id"] == "design_0007"


def test_normalize_window_frame_allows_duplicate_designs():
    # Window-level frames (peptides) legitimately have many rows per design.
    df = pd.DataFrame(
        {"protein_id": ["Q00511"] * 3, "design_idx": [0, 0, 0], "pos": [0, 1, 2]}
    )
    out = normalize_design_keys(df, unique=False)
    assert len(out) == 3
    assert list(out["design_id"]) == ["design_0000"] * 3


def test_anchor_overlap_covering_window():
    overlaps, covered, count, dist = anchor_overlap(8, 20, frozenset({10, 57}))
    assert overlaps is True
    assert covered == [10]
    assert count == 1
    assert dist == 0


def test_anchor_overlap_non_covering_reports_min_distance():
    # window [20, 30) ; nearest anchor is 10 (distance 10) or 57 (distance 28)
    overlaps, covered, count, dist = anchor_overlap(20, 30, frozenset({10, 57}))
    assert overlaps is False
    assert covered == []
    assert count == 0
    assert dist == 10  # 20 - 10


def test_anchor_overlap_empty_anchor_set():
    overlaps, covered, count, dist = anchor_overlap(0, 12, frozenset())
    assert overlaps is False
    assert covered == []
    assert count == 0
    assert dist is None
