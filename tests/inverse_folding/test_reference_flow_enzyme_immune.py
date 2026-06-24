"""Task U5 active-site immune telemetry tests — PLAN_URICASE_ENZYME_MODE.

Synthetic in-memory fixtures only; no real run files are read. Covers:
- window overlap flags vs a known anchor set,
- strong/margin anchor vs non-anchor splits,
- delta = NA when wt is None AND when the protein is missing from wt,
- a real delta when wt is present,
- hotspot concentration fraction on a hand-built design.
"""

from __future__ import annotations

import pandas as pd
import pytest

from inverse_folding.reference_flow.enzyme_immune import (
    build_enzyme_immune_summary,
    build_enzyme_nmp_windows,
)

# Hard-anchor index set (0-based) used across tests.
ANCHORS = frozenset({10, 57, 58})


def _peptides_df(rows: list[dict]) -> pd.DataFrame:
    """Build an imm_nmp_peptides-style window frame from record dicts."""
    cols = [
        "protein_id",
        "design_id",
        "design_idx",
        "pep_length",
        "pos",
        "peptide",
        "core",
        "rank_EL",
        "el_score",
    ]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# FUNCTION 1: build_enzyme_nmp_windows
# --------------------------------------------------------------------------- #


def test_windows_overlap_flags_against_known_anchor_set():
    # Window A [8,23) covers anchor 10 ; window B [30,45) covers nothing.
    df = _peptides_df(
        [
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=15,
                pos=8,
                peptide="A" * 15,
                core="AAAAAAAAA",
                rank_EL=0.5,
                el_score=0.9,
            ),
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=15,
                pos=30,
                peptide="B" * 15,
                core="BBBBBBBBB",
                rank_EL=8.0,
                el_score=0.1,
            ),
        ]
    )
    out = build_enzyme_nmp_windows(
        df, ANCHORS, rank_threshold=2.0, source="design"
    )

    assert list(out["source"]) == ["design", "design"]
    assert list(out["rank_threshold"]) == [2.0, 2.0]

    a = out[out["pos"] == 8].iloc[0]
    assert bool(a["overlaps_hard_anchor"]) is True
    assert int(a["num_hard_anchors_covered"]) == 1
    assert list(a["covered_anchor_indices"]) == [10]
    assert int(a["min_distance_to_hard_anchor"]) == 0
    assert bool(a["strong_binder"]) is True  # rank 0.5 <= 2.0

    b = out[out["pos"] == 30].iloc[0]
    assert bool(b["overlaps_hard_anchor"]) is False
    assert int(b["num_hard_anchors_covered"]) == 0
    assert list(b["covered_anchor_indices"]) == []
    # window [30,45) -> last residue 44; nearest anchor is 57 at distance
    # min(|57-30|, |57-44|) = min(27, 13) = 13.
    assert int(b["min_distance_to_hard_anchor"]) == 13
    assert bool(b["strong_binder"]) is False  # rank 8.0 > 2.0


def test_windows_min_distance_is_na_for_empty_anchor_set():
    df = _peptides_df(
        [
            dict(
                protein_id="P1",
                design_id="design_0000",
                design_idx=0,
                pep_length=9,
                pos=0,
                peptide="C" * 9,
                core="CCCCCCCCC",
                rank_EL=1.0,
                el_score=0.5,
            ),
        ]
    )
    out = build_enzyme_nmp_windows(
        df, frozenset(), rank_threshold=2.0, source="WT"
    )
    row = out.iloc[0]
    assert bool(row["overlaps_hard_anchor"]) is False
    assert int(row["num_hard_anchors_covered"]) == 0
    assert pd.isna(row["min_distance_to_hard_anchor"])
    assert list(row["covered_anchor_indices"]) == []
    assert row["source"] == "WT"


def test_windows_reconstructs_design_idx_when_missing():
    # Window frame carrying only design_id; design_idx must be reconstructed.
    df = pd.DataFrame(
        {
            "protein_id": ["Q00511", "Q00511"],
            "design_id": ["design_0002", "design_0002"],
            "pep_length": [9, 9],
            "pos": [55, 100],
            "peptide": ["X" * 9, "Y" * 9],
            "core": ["XXXXXXXXX", "YYYYYYYYY"],
            "rank_EL": [0.2, 5.0],
            "el_score": [0.8, 0.2],
        }
    )
    out = build_enzyme_nmp_windows(
        df, ANCHORS, rank_threshold=1.0, source="design"
    )
    assert set(out["design_idx"]) == {2}
    # window [55,64) covers anchors 57 and 58.
    covered = out[out["pos"] == 55].iloc[0]
    assert int(covered["num_hard_anchors_covered"]) == 2
    assert list(covered["covered_anchor_indices"]) == [57, 58]


# --------------------------------------------------------------------------- #
# FUNCTION 2: build_enzyme_immune_summary
# --------------------------------------------------------------------------- #


def _design_windows() -> pd.DataFrame:
    """Hand-built design windows for protein Q00511, design 0.

    Three strong windows (rank <= threshold=2.0):
      - pos 8  [8,23)   covers anchor 10  -> anchor-overlap strong, margin 2-0.5 = 1.5
      - pos 50 [50,59)  covers anchor 57,58 -> anchor-overlap strong, margin 2-1.0 = 1.0
      - pos 100 [100,109) covers nothing  -> non-anchor strong, margin 2-1.5 = 0.5
    One weak window:
      - pos 200 [200,209) covers nothing, rank 9.0 -> not strong, margin 0
    """
    rows = [
        dict(pos=8, pep_length=15, rank_EL=0.5, el_score=0.9),
        dict(pos=50, pep_length=9, rank_EL=1.0, el_score=0.7),
        dict(pos=100, pep_length=9, rank_EL=1.5, el_score=0.6),
        dict(pos=200, pep_length=9, rank_EL=9.0, el_score=0.05),
    ]
    pep = _peptides_df(
        [
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=r["pep_length"],
                pos=r["pos"],
                peptide="Z" * r["pep_length"],
                core="ZZZZZZZZZ",
                rank_EL=r["rank_EL"],
                el_score=r["el_score"],
            )
            for r in rows
        ]
    )
    return build_enzyme_nmp_windows(
        pep, ANCHORS, rank_threshold=2.0, source="design"
    )


def test_summary_strong_and_margin_splits():
    dw = _design_windows()
    out = build_enzyme_immune_summary(dw, None, rank_threshold=2.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["protein_id"] == "Q00511"
    assert int(row["design_idx"]) == 0

    # 3 strong total: 2 anchor-overlap + 1 non-anchor.
    assert int(row["nmp_total_strong"]) == 3
    assert int(row["nmp_anchor_overlap_strong"]) == 2
    assert int(row["nmp_non_anchor_strong"]) == 1

    # margin mass: total = 1.5 + 1.0 + 0.5 + 0 = 3.0
    assert row["nmp_total_rank_margin_mass"] == pytest.approx(3.0)
    # anchor overlap margin = 1.5 + 1.0 = 2.5
    assert row["nmp_anchor_overlap_rank_margin_mass"] == pytest.approx(2.5)
    # non anchor margin = 0.5 (the rank-9 window contributes 0)
    assert row["nmp_non_anchor_rank_margin_mass"] == pytest.approx(0.5)


def test_summary_delta_is_na_when_wt_none():
    dw = _design_windows()
    out = build_enzyme_immune_summary(dw, None, rank_threshold=2.0)
    row = out.iloc[0]
    for col in [
        "delta_total_strong_vs_wt",
        "delta_anchor_overlap_strong_vs_wt",
        "delta_non_anchor_strong_vs_wt",
        "delta_total_rank_margin_mass_vs_wt",
        "delta_anchor_overlap_rank_margin_mass_vs_wt",
        "delta_non_anchor_rank_margin_mass_vs_wt",
    ]:
        assert pd.isna(row[col]), col


def test_summary_delta_is_na_when_protein_absent_from_wt():
    dw = _design_windows()
    # WT windows for a DIFFERENT protein only.
    wt_pep = _peptides_df(
        [
            dict(
                protein_id="OTHER",
                design_id="design_0000",
                design_idx=0,
                pep_length=9,
                pos=8,
                peptide="W" * 9,
                core="WWWWWWWWW",
                rank_EL=0.5,
                el_score=0.9,
            )
        ]
    )
    wt = build_enzyme_nmp_windows(
        wt_pep, ANCHORS, rank_threshold=2.0, source="WT"
    )
    out = build_enzyme_immune_summary(dw, wt, rank_threshold=2.0)
    row = out.iloc[0]
    assert pd.isna(row["delta_total_strong_vs_wt"])
    assert pd.isna(row["delta_total_rank_margin_mass_vs_wt"])
    assert pd.isna(row["delta_anchor_overlap_strong_vs_wt"])
    assert pd.isna(row["delta_non_anchor_rank_margin_mass_vs_wt"])


def _wt_windows_matched() -> pd.DataFrame:
    """WT facade windows on the SAME (pep_length, pos) grid as ``_design_windows``.

    Same 4 windows as the design (matched support) but different ranks:
      - (15,8)  rank 0.0 -> anchor-overlap strong, margin 2.0
      - (9,50)  rank 3.0 -> weak, margin 0
      - (9,100) rank 3.0 -> weak, margin 0
      - (9,200) rank 9.0 -> weak, margin 0
    WT totals: 1 strong (anchor-overlap), 0 non-anchor strong, total margin 2.0.
    """
    rows = [
        dict(pos=8, pep_length=15, rank_EL=0.0),
        dict(pos=50, pep_length=9, rank_EL=3.0),
        dict(pos=100, pep_length=9, rank_EL=3.0),
        dict(pos=200, pep_length=9, rank_EL=9.0),
    ]
    pep = _peptides_df(
        [
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=r["pep_length"],
                pos=r["pos"],
                peptide="W" * r["pep_length"],
                core="WWWWWWWWW",
                rank_EL=r["rank_EL"],
                el_score=0.5,
            )
            for r in rows
        ]
    )
    return build_enzyme_nmp_windows(pep, ANCHORS, rank_threshold=2.0, source="WT")


def test_summary_real_delta_when_wt_window_support_matches():
    dw = _design_windows()
    wt = _wt_windows_matched()
    out = build_enzyme_immune_summary(dw, wt, rank_threshold=2.0)
    row = out.iloc[0]

    # matched (same (pep_length, pos) grid) -> deltas are computed.
    assert bool(row["wt_window_support_matched"]) is True

    # design totals: 3 strong, 1 non-anchor strong, 2 anchor-overlap strong.
    # WT totals: 1 strong, 1 anchor-overlap strong, 0 non-anchor strong.
    assert int(row["delta_total_strong_vs_wt"]) == 3 - 1
    assert int(row["delta_anchor_overlap_strong_vs_wt"]) == 2 - 1
    assert int(row["delta_non_anchor_strong_vs_wt"]) == 1 - 0

    # margin mass deltas. design total 3.0 ; WT total margin 2.0.
    assert row["delta_total_rank_margin_mass_vs_wt"] == pytest.approx(3.0 - 2.0)
    assert row["delta_anchor_overlap_rank_margin_mass_vs_wt"] == pytest.approx(0.5)
    assert row["delta_non_anchor_rank_margin_mass_vs_wt"] == pytest.approx(0.5)


def test_summary_delta_na_when_wt_window_support_differs():
    # WT covers only ONE of the design's four (pep_length, pos) windows -> support
    # mismatch -> deltas must be NA (never subtract aggregates over different grids).
    dw = _design_windows()
    wt_pep = _peptides_df(
        [
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=15,
                pos=8,
                peptide="W" * 15,
                core="WWWWWWWWW",
                rank_EL=0.0,
                el_score=0.99,
            ),
        ]
    )
    wt = build_enzyme_nmp_windows(wt_pep, ANCHORS, rank_threshold=2.0, source="WT")
    out = build_enzyme_immune_summary(dw, wt, rank_threshold=2.0)
    row = out.iloc[0]
    assert bool(row["wt_window_support_matched"]) is False
    for col in [
        "delta_total_strong_vs_wt",
        "delta_anchor_overlap_strong_vs_wt",
        "delta_non_anchor_strong_vs_wt",
        "delta_total_rank_margin_mass_vs_wt",
        "delta_anchor_overlap_rank_margin_mass_vs_wt",
        "delta_non_anchor_rank_margin_mass_vs_wt",
    ]:
        assert pd.isna(row[col]), col


def test_summary_hotspot_concentration_fraction():
    dw = _design_windows()
    out = build_enzyme_immune_summary(
        dw, None, rank_threshold=2.0, top_ks=(1, 3, 5)
    )
    row = out.iloc[0]
    # per-window margin masses: [1.5, 1.0, 0.5, 0.0] ; total = 3.0
    # top1 = 1.5 / 3.0 = 0.5
    assert row["hotspot_concentration_top1_frac"] == pytest.approx(1.5 / 3.0)
    # top3 = (1.5 + 1.0 + 0.5) / 3.0 = 1.0
    assert row["hotspot_concentration_top3_frac"] == pytest.approx(1.0)
    # top5 (only 4 windows) = 3.0 / 3.0 = 1.0
    assert row["hotspot_concentration_top5_frac"] == pytest.approx(1.0)


def test_summary_hotspot_concentration_zero_when_no_mass():
    # All windows weak -> total margin mass 0 -> concentration fracs are 0.0.
    pep = _peptides_df(
        [
            dict(
                protein_id="P9",
                design_id="design_0001",
                design_idx=1,
                pep_length=9,
                pos=0,
                peptide="Q" * 9,
                core="QQQQQQQQQ",
                rank_EL=10.0,
                el_score=0.0,
            )
        ]
    )
    dw = build_enzyme_nmp_windows(
        pep, ANCHORS, rank_threshold=2.0, source="design"
    )
    out = build_enzyme_immune_summary(dw, None, rank_threshold=2.0)
    row = out.iloc[0]
    assert int(row["nmp_total_strong"]) == 0
    assert row["nmp_total_rank_margin_mass"] == pytest.approx(0.0)
    assert row["hotspot_concentration_top1_frac"] == pytest.approx(0.0)
    assert row["hotspot_concentration_top3_frac"] == pytest.approx(0.0)
    assert row["hotspot_concentration_top5_frac"] == pytest.approx(0.0)


def test_summary_external_only_flag_is_na_for_v0():
    dw = _design_windows()
    out = build_enzyme_immune_summary(dw, None, rank_threshold=2.0)
    assert pd.isna(out.iloc[0]["external_only_candidate_flag"])


def test_summary_multiple_designs_one_row_each():
    # Two designs for one protein, plus a window each.
    pep = _peptides_df(
        [
            dict(
                protein_id="Q00511",
                design_id="design_0000",
                design_idx=0,
                pep_length=9,
                pos=8,
                peptide="A" * 9,
                core="AAAAAAAAA",
                rank_EL=0.5,
                el_score=0.9,
            ),
            dict(
                protein_id="Q00511",
                design_id="design_0001",
                design_idx=1,
                pep_length=9,
                pos=100,
                peptide="B" * 9,
                core="BBBBBBBBB",
                rank_EL=1.0,
                el_score=0.7,
            ),
        ]
    )
    dw = build_enzyme_nmp_windows(
        pep, ANCHORS, rank_threshold=2.0, source="design"
    )
    out = build_enzyme_immune_summary(dw, None, rank_threshold=2.0)
    assert len(out) == 2
    assert set(out["design_idx"]) == {0, 1}
    d0 = out[out["design_idx"] == 0].iloc[0]
    d1 = out[out["design_idx"] == 1].iloc[0]
    assert int(d0["nmp_anchor_overlap_strong"]) == 1
    assert int(d1["nmp_non_anchor_strong"]) == 1
