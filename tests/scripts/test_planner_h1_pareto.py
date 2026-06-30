"""Planner H1 Pareto gate tests (PLAN_PLANNER_SC_GR.md Task 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import json

from scripts.analysis.planner_h1_pareto import (
    _frame_from_runs_json,
    compute_h1,
    compute_pairwise_gate,
    extract_arm_mode_g_max,
    merge_run_eval,
    merge_run_eval_frames,
)


def test_compute_h1_aggregates_designs_then_pairs():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(50):
        base = rng.normal(0, 1)
        for d in range(8):  # 8 designs/protein
            jit = rng.normal(0, 0.3)
            for mode, imm in [
                ("flat", base + 1.0 + jit),
                ("v_target", base + jit),
                ("v_target_x_alloc", base - 0.6 + jit),
            ]:
                rows.append(
                    dict(
                        protein_id=f"p{p}",
                        design_idx=d,
                        g_max=2.0,
                        arm_mode=mode,
                        immune_nmp=imm,
                        sctm=0.95,
                    )
                )
    out = compute_h1(pd.DataFrame(rows))
    cell = [c for c in out["by_g_max"] if c["g_max"] == 2.0][0]
    assert cell["n_proteins_paired"] == 50  # paired over proteins, not 400 rows
    assert cell["alloc_minus_vtarget_median"] < 0
    assert cell["wilcoxon_alloc_vs_vtarget_p"] < 0.05
    assert cell["vtarget_minus_flat_median"] < 0
    assert cell["sctm_noninferior"] is True
    assert cell["h1_pass"] is True


def test_compute_h1_macro_leg_flat_equals_vtarget_fails():
    # flat ≈ v_target (selection not actuating) -> macro leg fails -> no pass,
    # even though alloc < v_target (PLAN §5 FAIL: macro leg).
    rng = np.random.default_rng(1)
    rows = []
    for p in range(40):
        base = rng.normal(0, 1)
        for d in range(4):
            jit = rng.normal(0, 0.2)
            for mode, imm in [
                ("flat", base + jit),
                ("v_target", base + jit),  # identical to flat
                ("v_target_x_alloc", base - 0.8 + jit),
            ]:
                rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=2.0,
                                 arm_mode=mode, immune_nmp=imm, sctm=0.95))
    cell = [c for c in compute_h1(pd.DataFrame(rows))["by_g_max"] if c["g_max"] == 2.0][0]
    assert cell["alloc_minus_vtarget_median"] < 0  # primary leg holds
    assert cell["macro_pass"] is False             # but flat ≈ v_target
    assert cell["h1_pass"] is False


def test_compute_h1_incomplete_cohort_fails():
    # only 6 proteins, but the pre-registered cohort is 50 -> gate fails (PLAN §5).
    rng = np.random.default_rng(2)
    rows = []
    for p in range(6):
        base = rng.normal(0, 1)
        for d in range(8):
            jit = rng.normal(0, 0.2)
            for mode, imm in [("flat", base + 1.0 + jit), ("v_target", base + jit),
                              ("v_target_x_alloc", base - 0.8 + jit)]:
                rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=2.0,
                                 arm_mode=mode, immune_nmp=imm, sctm=0.95))
    out = compute_h1(
        pd.DataFrame(rows),
        expected_n_proteins=50,
        expected_n_designs=8,
        expected_g_max=(1.5, 2.0, 2.5),
    )
    cell = [c for c in out["by_g_max"] if c["g_max"] == 2.0][0]
    assert cell["cohort_complete"] is False  # 6 != 50
    assert cell["h1_pass"] is False
    assert out["overall_pass"] is False
    assert set(out["missing_g_max"]) == {1.5, 2.5}


def test_compute_h1_complete_cohort_overall_pass():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(50):
        base = rng.normal(0, 1)
        for d in range(8):
            jit = rng.normal(0, 0.2)
            for mode, imm in [("flat", base + 1.0 + jit), ("v_target", base + jit),
                              ("v_target_x_alloc", base - 0.8 + jit)]:
                rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=2.0,
                                 arm_mode=mode, immune_nmp=imm, sctm=0.95))
    out = compute_h1(
        pd.DataFrame(rows),
        expected_n_proteins=50,
        expected_n_designs=8,
        expected_g_max=(2.0,),
    )
    cell = out["by_g_max"][0]
    assert cell["cohort_complete"] is True
    assert cell["macro_pass"] is True
    assert cell["h1_pass"] is True
    assert out["overall_pass"] is True


def test_merge_run_eval_frames_fails_on_dropped_rows():
    # a design present in immune but not structure (or vice versa) must NOT be
    # silently dropped by the inner join (PLAN §5 fixed cohort, fail-fast norm).
    imm = pd.DataFrame({"protein_id": ["p0", "p1"], "design_idx": [0, 0],
                        "n_strong_binders": [1, 1], "n_windows_scored": [10, 10]})
    struct = pd.DataFrame({"protein_id": ["p0"], "design_idx": [0], "scTM": [0.9]})
    with pytest.raises(ValueError, match="(?i)drop|mismatch|missing"):
        merge_run_eval_frames(imm, struct, arm_mode="flat", g_max=1.5)


def test_compute_h1_sctm_drop_fails_noninferiority():
    # alloc immune lower (good) but its scTM collapses -> non-inferiority fails -> no pass
    rows = []
    for p in range(30):
        for d in range(4):
            rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=1.5,
                             arm_mode="v_target", immune_nmp=1.0, sctm=0.9))
            rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=1.5,
                             arm_mode="v_target_x_alloc", immune_nmp=0.3, sctm=0.5))
            rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=1.5,
                             arm_mode="flat", immune_nmp=2.0, sctm=0.9))
    cell = [c for c in compute_h1(pd.DataFrame(rows))["by_g_max"] if c["g_max"] == 1.5][0]
    assert cell["alloc_minus_vtarget_median"] < 0  # immune did drop
    assert cell["sctm_noninferior"] is False  # but scTM collapsed
    assert cell["h1_pass"] is False


def test_merge_run_eval_derives_strong_frac_and_joins(tmp_path):
    imm = pd.DataFrame(
        {
            "protein_id": ["p0", "p1"],
            "design_id": ["design_0000", "design_0000"],
            "design_idx": [0, 0],
            "n_strong_binders": [2, 0],
            "n_windows_scored": [10, 8],
            "mean_best_rank": [1.0, 5.0],
        }
    )
    struct = pd.DataFrame(
        {"protein_id": ["p0", "p1"], "design_idx": [0, 0], "scTM": [0.9, 0.8]}
    )
    ip = tmp_path / "imm_nmp.parquet"
    sp = tmp_path / "structural.parquet"
    imm.to_parquet(ip)
    struct.to_parquet(sp)
    df = merge_run_eval(ip, sp, arm_mode="v_target_x_alloc", g_max=2.0)
    assert {
        "protein_id", "design_idx", "g_max", "arm_mode", "immune_nmp", "sctm"
    }.issubset(df.columns)
    row0 = df[df.protein_id == "p0"].iloc[0]
    assert row0["immune_nmp"] == pytest.approx(0.2)  # n_strong_binders / n_windows_scored
    assert row0["sctm"] == pytest.approx(0.9)
    assert row0["arm_mode"] == "v_target_x_alloc"
    assert float(row0["g_max"]) == 2.0


def test_merge_run_eval_zero_windows_is_nan():
    # n_windows_scored==0 -> immune_nmp NaN (not a div-by-zero crash)
    imm = pd.DataFrame(
        {"protein_id": ["p0"], "design_idx": [0],
         "n_strong_binders": [0], "n_windows_scored": [0]}
    )
    struct = pd.DataFrame({"protein_id": ["p0"], "design_idx": [0], "scTM": [0.9]})
    df = merge_run_eval_frames(imm, struct, arm_mode="flat", g_max=1.5)
    assert np.isnan(df.iloc[0]["immune_nmp"])


def _three_arm_frame(treatment, *, n=40, d=4, g_max=2.0, treat_shift=-0.8, flat_shift=1.0):
    rng = np.random.default_rng(7)
    rows = []
    for p in range(n):
        base = rng.normal(0, 1)
        for di in range(d):
            jit = rng.normal(0, 0.2)
            for mode, imm in [("flat", base + flat_shift + jit),
                              ("v_target", base + jit),
                              (treatment, base + treat_shift + jit)]:
                rows.append(dict(protein_id=f"p{p}", design_idx=di, g_max=g_max,
                                 arm_mode=mode, immune_nmp=imm, sctm=0.95))
    return pd.DataFrame(rows)


def test_compute_h1_treatment_mode_triage():
    # §A1: the treatment arm is v_target_triage; gate pairs it vs v_target + macro.
    df = _three_arm_frame("v_target_triage")
    out = compute_h1(df, treatment_mode="v_target_triage")
    cell = out["by_g_max"][0]
    assert cell["treatment_mode"] == "v_target_triage"
    assert cell["alloc_minus_vtarget_median"] < 0  # triage below v_target
    assert cell["macro_pass"] is True
    assert cell["h1_pass"] is True


def _two_arm_frame(*, terminal_shift, n=30, d=4, terminal_sctm=0.95):
    rng = np.random.default_rng(11)
    rows = []
    for p in range(n):
        base = rng.normal(0, 1)
        for di in range(d):
            jit = rng.normal(0, 0.15)
            rows.append(dict(protein_id=f"p{p}", design_idx=di,
                             arm_mode="b1_local", immune_nmp=base + jit, sctm=0.9))
            rows.append(dict(protein_id=f"p{p}", design_idx=di, arm_mode="b1_terminal",
                             immune_nmp=base + terminal_shift + jit, sctm=terminal_sctm))
    return pd.DataFrame(rows)


def test_compute_pairwise_gate_terminal_better():
    out = compute_pairwise_gate(
        _two_arm_frame(terminal_shift=-0.5), treatment="b1_terminal", baseline="b1_local"
    )
    assert out["treatment_minus_baseline_median"] < 0
    assert out["sctm_noninferior"] is True
    assert out["treatment_better"] is True
    assert out["neutral_or_better"] is True


def test_compute_pairwise_gate_neutral_is_neutral_or_better_only():
    out = compute_pairwise_gate(
        _two_arm_frame(terminal_shift=0.0), treatment="b1_terminal", baseline="b1_local"
    )
    assert out["treatment_better"] is False  # no significant drop
    assert out["neutral_or_better"] is True  # but not significantly worse


def test_compute_pairwise_gate_worse_fails_both():
    out = compute_pairwise_gate(
        _two_arm_frame(terminal_shift=0.5), treatment="b1_terminal", baseline="b1_local"
    )
    assert out["treatment_better"] is False
    assert out["neutral_or_better"] is False  # significantly worse


def test_compute_pairwise_gate_sctm_collapse_blocks_pass():
    out = compute_pairwise_gate(
        _two_arm_frame(terminal_shift=-0.5, terminal_sctm=0.5),
        treatment="b1_terminal", baseline="b1_local",
    )
    assert out["sctm_noninferior"] is False
    assert out["treatment_better"] is False  # immune drop but scTM collapsed
    assert out["neutral_or_better"] is False


def test_pairwise_requires_explicit_arm_mode(tmp_path):
    # under --pairwise, an entry relying on controller_config auto-derive must fail
    # fast (b1_local/b1_terminal are indistinguishable by config fields).
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([
        {"imm_nmp_parquet": "x.parquet", "structural_parquet": "y.parquet",
         "controller_config_json": "cfg.json"}
    ]))
    with pytest.raises(ValueError, match="(?i)arm_mode"):
        _frame_from_runs_json(runs, require_explicit_arm=True)


def test_pairwise_explicit_arm_mode_assembles(tmp_path):
    imm = pd.DataFrame({"protein_id": ["p0"], "design_idx": [0],
                        "n_strong_binders": [1], "n_windows_scored": [10]})
    struct = pd.DataFrame({"protein_id": ["p0"], "design_idx": [0], "scTM": [0.9]})
    ip, sp = tmp_path / "imm.parquet", tmp_path / "struct.parquet"
    imm.to_parquet(ip)
    struct.to_parquet(sp)
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([
        {"imm_nmp_parquet": str(ip), "structural_parquet": str(sp),
         "arm_mode": "b1_terminal"}  # explicit, no g_max needed
    ]))
    df = _frame_from_runs_json(runs, require_explicit_arm=True)
    assert set(df["arm_mode"]) == {"b1_terminal"}
    assert df.iloc[0]["immune_nmp"] == pytest.approx(0.1)


def test_extract_arm_mode_g_max_from_config():
    cfg = {
        "targeting": {"selection_field_mode": "flat"},
        "global_pressure": {"g_max": 1.5},
    }
    assert extract_arm_mode_g_max(cfg) == ("flat", 1.5)
