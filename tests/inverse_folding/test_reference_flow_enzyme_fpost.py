"""F_post v0 measurement tests — PLAN_URICASE_ENZYME_MODE Task U7.

Synthetic in-memory fixtures only (hand-built DataFrames / dict records). No real
run files are read. Covers:
- per-design f_anchor pass vs fail from mixed ``preserved`` constraint rows,
- unconstrained design (no constraint rows) -> f_anchor pass,
- delta_*_vs_wt == pd.NA when wt is None,
- real delta_*_vs_wt when wt structural frame is present (matched by protein_id),
- f_fold_status == 'ranking_only' when scTM present,
- f_assembly / f_site_scaffold literal 'unavailable' for v0 monomer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from inverse_folding.reference_flow.enzyme_fpost import build_f_post_v0


def _structural_df() -> pd.DataFrame:
    """Two designs for Q00511, schema mimics structural.parquet."""
    return pd.DataFrame(
        {
            "protein_id": ["Q00511", "Q00511"],
            "design_id": ["design_0000", "design_0001"],
            "design_idx": [0, 1],
            "sequence": ["AAAA", "CCCC"],
            "scTM": [0.80, 0.65],
            "pLDDT": [85.0, 70.0],
            "global_ca_RMSD": [1.2, 2.5],
            "recovery": [0.40, 0.55],
            "foldability": [0.9, 0.7],
            "refold_backend": ["esmfold", "esmfold"],
        }
    )


def _constraint_rows() -> list[dict]:
    """design 0 preserves all anchors; design 1 has one mismatch.

    design 0: two anchor rows, both preserved=True  -> pass, 0 mismatches.
    design 1: two anchor rows, one preserved=False   -> fail, 1 mismatch.
    """
    return [
        {
            "protein_id": "Q00511",
            "design_idx": 0,
            "anchor_index_0b": 10,
            "expected_aa": "K",
            "generated_aa": "K",
            "preserved": True,
        },
        {
            "protein_id": "Q00511",
            "design_idx": 0,
            "anchor_index_0b": 57,
            "expected_aa": "T",
            "generated_aa": "T",
            "preserved": True,
        },
        {
            "protein_id": "Q00511",
            "design_idx": 1,
            "anchor_index_0b": 10,
            "expected_aa": "K",
            "generated_aa": "K",
            "preserved": True,
        },
        {
            "protein_id": "Q00511",
            "design_idx": 1,
            "anchor_index_0b": 57,
            "expected_aa": "T",
            "generated_aa": "A",
            "preserved": False,
        },
    ]


def _wt_structural_df() -> pd.DataFrame:
    """Single WT design per protein (WT facade structural run)."""
    return pd.DataFrame(
        {
            "protein_id": ["Q00511"],
            "design_id": ["design_0000"],
            "design_idx": [0],
            "sequence": ["WWWW"],
            "scTM": [0.90],
            "pLDDT": [88.0],
            "global_ca_RMSD": [1.0],
            "recovery": [1.0],
            "foldability": [1.0],
            "refold_backend": ["esmfold"],
        }
    )


def _row(out: pd.DataFrame, design_idx: int) -> pd.Series:
    sel = out[out["design_idx"] == design_idx]
    assert len(sel) == 1, f"expected exactly one row for design_idx={design_idx}"
    return sel.iloc[0]


def test_one_row_per_design_present_in_structural():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    assert len(out) == 2
    assert set(out["design_idx"]) == {0, 1}
    assert list(out["protein_id"].unique()) == ["Q00511"]


def test_f_anchor_pass_when_all_preserved():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    r0 = _row(out, 0)
    assert r0["f_anchor_status"] == "pass"
    assert int(r0["num_anchor_mismatches"]) == 0


def test_f_anchor_fail_when_any_mismatch():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    r1 = _row(out, 1)
    assert r1["f_anchor_status"] == "fail"
    assert int(r1["num_anchor_mismatches"]) == 1


def test_unconstrained_design_passes_with_zero_mismatches():
    # No constraint rows at all -> every design is unconstrained -> pass.
    out = build_f_post_v0([], _structural_df())
    for idx in (0, 1):
        r = _row(out, idx)
        assert r["f_anchor_status"] == "pass"
        assert int(r["num_anchor_mismatches"]) == 0


def test_constraint_rows_accept_dataframe_input():
    # constraint_rows may arrive as a DataFrame, not only list[dict].
    out = build_f_post_v0(pd.DataFrame(_constraint_rows()), _structural_df())
    assert _row(out, 0)["f_anchor_status"] == "pass"
    assert _row(out, 1)["f_anchor_status"] == "fail"


def test_delta_is_na_when_wt_none():
    out = build_f_post_v0(_constraint_rows(), _structural_df(), None)
    for col in (
        "delta_scTM_vs_wt",
        "delta_global_ca_RMSD_vs_wt",
        "delta_pLDDT_vs_wt",
        "delta_recovery_vs_wt",
    ):
        assert out[col].isna().all(), f"{col} must be NA when wt is None"


def test_delta_is_na_when_protein_absent_from_wt():
    wt = _wt_structural_df().assign(protein_id="OTHER")
    out = build_f_post_v0(_constraint_rows(), _structural_df(), wt)
    for col in (
        "delta_scTM_vs_wt",
        "delta_global_ca_RMSD_vs_wt",
        "delta_pLDDT_vs_wt",
        "delta_recovery_vs_wt",
    ):
        assert out[col].isna().all()


def test_real_delta_when_wt_present():
    out = build_f_post_v0(_constraint_rows(), _structural_df(), _wt_structural_df())
    r0 = _row(out, 0)
    # design - wt : 0.80 - 0.90 = -0.10 (improvement is negative)
    assert r0["delta_scTM_vs_wt"] == pytest.approx(0.80 - 0.90)
    assert r0["delta_global_ca_RMSD_vs_wt"] == pytest.approx(1.2 - 1.0)
    assert r0["delta_pLDDT_vs_wt"] == pytest.approx(85.0 - 88.0)
    assert r0["delta_recovery_vs_wt"] == pytest.approx(0.40 - 1.0)
    r1 = _row(out, 1)
    assert r1["delta_scTM_vs_wt"] == pytest.approx(0.65 - 0.90)
    assert r1["delta_global_ca_RMSD_vs_wt"] == pytest.approx(2.5 - 1.0)


def test_absolute_columns_passthrough():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    r0 = _row(out, 0)
    assert r0["scTM"] == pytest.approx(0.80)
    assert r0["global_ca_RMSD"] == pytest.approx(1.2)
    assert r0["pLDDT"] == pytest.approx(85.0)
    assert r0["recovery"] == pytest.approx(0.40)
    assert r0["foldability"] == pytest.approx(0.9)


def test_f_fold_status_ranking_only_when_sctm_present():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    assert (out["f_fold_status"] == "ranking_only").all()


def test_f_fold_status_unavailable_when_sctm_missing():
    struct = _structural_df()
    struct["scTM"] = [pd.NA, pd.NA]
    out = build_f_post_v0(_constraint_rows(), struct)
    assert (out["f_fold_status"] == "unavailable").all()


def test_v0_constant_envelope_columns():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    assert (out["reference_type"] == "predicted_target").all()
    assert (out["calibration_status"] == "ranking_only").all()
    assert (out["f_assembly"] == "unavailable").all()
    assert (out["f_site_scaffold"] == "unavailable").all()


def test_predictor_fields_propagated():
    out = build_f_post_v0(
        _constraint_rows(),
        _structural_df(),
        predictor="esmfold",
        predictor_version="v1.0.3",
    )
    assert (out["predictor"] == "esmfold").all()
    assert (out["predictor_version"] == "v1.0.3").all()


def test_predictor_defaults():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    assert (out["predictor"] == "esmfold2").all()
    assert (out["predictor_version"] == "na").all()


def test_output_columns_exact_set():
    out = build_f_post_v0(_constraint_rows(), _structural_df())
    expected = [
        "protein_id",
        "design_idx",
        "f_anchor_status",
        "num_anchor_mismatches",
        "f_fold_status",
        "scTM",
        "global_ca_RMSD",
        "pLDDT",
        "recovery",
        "foldability",
        "delta_scTM_vs_wt",
        "delta_global_ca_RMSD_vs_wt",
        "delta_pLDDT_vs_wt",
        "delta_recovery_vs_wt",
        "reference_type",
        "predictor",
        "predictor_version",
        "calibration_status",
        "f_assembly",
        "f_site_scaffold",
    ]
    assert list(out.columns) == expected


def test_duplicate_structural_design_is_hard_error():
    struct = _structural_df()
    dup = pd.concat([struct, struct.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        build_f_post_v0(_constraint_rows(), dup)
