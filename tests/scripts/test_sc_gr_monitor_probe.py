"""SC-GR monitor analysis tests (PLAN_RF_SC_GR.md Task SC0.5)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.sc_gr_monitor_probe import (
    aggregate_oracle,
    compute_monitor_table,
    emit_calibration,
    main,
    select_best_metric,
    spearman,
    tercile_bins,
)


def _probe_df(
    probe_by_protein: dict[str, float],
    *,
    arm: str = "fresh",
    refresh_step: int = 0,
    metric_col: str = "B_sc_mean_excess_median",
) -> pd.DataFrame:
    rows = []
    for pid, val in probe_by_protein.items():
        row = {
            "protein_id": pid, "design_idx": 0, "seed": 42,
            "refresh_step": refresh_step, "step": 10, "t": 0.5, "arm": arm,
            "B_sc_mean_excess_median": 0.0,
            "B_sc_topm_lse_median": 0.0,
            "B_sc_supra_mass_tau_11p75_median": 0.0,
        }
        row[metric_col] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure statistics
# ---------------------------------------------------------------------------


def test_spearman_monotone_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 33.0, 44.0, 99.0])
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_constant_is_nan():
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))


def test_tercile_bins_three_groups():
    bins = tercile_bins(np.arange(9.0))
    assert set(bins.tolist()) == {0, 1, 2}
    assert bins[0] == 0 and bins[-1] == 2


def test_aggregate_oracle_medians_over_designs():
    oracle = pd.DataFrame(
        {
            "protein_id": ["A", "A", "B", "B"],
            "global_risk": [1.0, 3.0, 10.0, 20.0],
        }
    )
    series = aggregate_oracle(
        oracle, protein_col="protein_id", risk_col="global_risk", agg="median"
    )
    assert series["A"] == pytest.approx(2.0)
    assert series["B"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# compute_monitor_table
# ---------------------------------------------------------------------------


def test_synthetic_monotone_probe_positive_spearman():
    proteins = {f"P{i}": float(i) for i in range(9)}
    probe = _probe_df(proteins)
    oracle = pd.Series({f"P{i}": float(i) for i in range(9)})
    table = compute_monitor_table(probe, oracle)
    rec = next(r for r in table if r["metric"] == "mean_excess")
    assert rec["n"] == 9
    assert rec["spearman"] == pytest.approx(1.0)
    assert rec["recall_at_high"] == pytest.approx(1.0)
    assert rec["p_low_given_high"] == pytest.approx(0.0)


def test_true_high_as_low_computed_correctly():
    # protein P6 has a high oracle burden but a low probe burden -> it should be
    # counted in P(probe low | oracle high), dropping Recall@High below 1.
    oracle_vals = {f"P{i}": float(i) for i in range(9)}
    probe_vals = dict(oracle_vals)
    probe_vals["P6"] = -10.0  # force P6 into the probe-low tercile
    probe = _probe_df(probe_vals)
    oracle = pd.Series(oracle_vals)
    table = compute_monitor_table(probe, oracle)
    rec = next(r for r in table if r["metric"] == "mean_excess")

    # Independently reproduce the expected bin stats with the module's binning.
    pb_series = probe.groupby("protein_id")["B_sc_mean_excess_median"].median()
    common = pb_series.index.intersection(oracle.index)
    pb = pb_series.loc[common].to_numpy(float)
    ob = oracle.loc[common].to_numpy(float)
    ob_bin = tercile_bins(ob)
    pb_bin = tercile_bins(pb)
    high = ob_bin == 2
    exp_p_low = float(np.mean(pb_bin[high] == 0))
    exp_recall = float(np.mean(pb_bin[high] == 2))

    assert rec["p_low_given_high"] == pytest.approx(exp_p_low)
    assert rec["recall_at_high"] == pytest.approx(exp_recall)
    assert rec["p_low_given_high"] > 0.0  # P6 misclassified
    assert rec["recall_at_high"] < 1.0


# ---------------------------------------------------------------------------
# best-metric selection
# ---------------------------------------------------------------------------


def test_best_metric_selection_prefers_recall_over_spearman():
    table = [
        {
            "arm": "fresh", "refresh_step": 1, "metric": "mean_excess", "n": 9,
            "spearman": 0.90, "recall_at_high": 0.30,
            "crosswalk_diagonal": 0.5, "p_low_given_high": 0.3,
        },
        {
            "arm": "self_conditioned", "refresh_step": 1, "metric": "topm_lse", "n": 9,
            "spearman": 0.50, "recall_at_high": 0.80,
            "crosswalk_diagonal": 0.6, "p_low_given_high": 0.1,
        },
    ]
    best = select_best_metric(table)
    assert best["metric"] == "topm_lse"
    assert best["recall_at_high"] == pytest.approx(0.80)


def test_best_metric_ties_break_on_spearman():
    table = [
        {"arm": "fresh", "refresh_step": 0, "metric": "a", "n": 5,
         "spearman": 0.4, "recall_at_high": 0.5},
        {"arm": "fresh", "refresh_step": 0, "metric": "b", "n": 5,
         "spearman": 0.7, "recall_at_high": 0.5},
    ]
    assert select_best_metric(table)["metric"] == "b"


def test_best_metric_none_when_no_valid():
    table = [{"arm": "fresh", "refresh_step": 0, "metric": "a", "n": 1,
              "spearman": float("nan"), "recall_at_high": float("nan")}]
    assert select_best_metric(table) is None


def test_best_metric_excludes_nonfinite_recall_or_spearman():
    # A "perfect" Recall@High that comes with a NaN Spearman (a degenerate /
    # constant estimator) must NOT win — finite Recall AND finite Spearman.
    table = [
        {"arm": "fresh", "refresh_step": 0, "metric": "constant", "n": 9,
         "spearman": float("nan"), "recall_at_high": 1.0},
        {"arm": "fresh", "refresh_step": 0, "metric": "real", "n": 9,
         "spearman": 0.7, "recall_at_high": 0.6},
    ]
    best = select_best_metric(table)
    assert best["metric"] == "real"


# ---------------------------------------------------------------------------
# degenerate-binning guard (the constant-probe pollution bug)
# ---------------------------------------------------------------------------


def test_constant_probe_yields_nan_bin_stats_not_perfect_recall():
    # Reproduce the reported P1: oracle 0..8, probe constant. The collapsed
    # tercile cut must NOT yield Recall@High=1.0; bin stats + Spearman are NaN.
    proteins = {f"P{i}": 1.0 for i in range(9)}  # constant probe burden
    probe = _probe_df(proteins)
    oracle = pd.Series({f"P{i}": float(i) for i in range(9)})
    table = compute_monitor_table(probe, oracle)
    rec = next(r for r in table if r["metric"] == "mean_excess")
    assert math.isnan(rec["recall_at_high"])
    assert math.isnan(rec["p_low_given_high"])
    assert math.isnan(rec["crosswalk_diagonal"])
    assert math.isnan(rec["spearman"])
    # all three aggregators are constant here -> no informative estimator.
    assert select_best_metric(table) is None


def test_constant_metric_not_preferred_over_real_metric():
    # mean_excess is monotone (real); topm_lse + supra are constant (degenerate).
    proteins = {f"P{i}": float(i) for i in range(9)}
    probe = _probe_df(proteins)  # mean_excess=i, topm_lse=0, supra=0
    oracle = pd.Series({f"P{i}": float(i) for i in range(9)})
    table = compute_monitor_table(probe, oracle)
    best = select_best_metric(table)
    assert best is not None
    assert best["metric"] == "mean_excess"
    topm = next(r for r in table if r["metric"] == "topm_lse")
    assert math.isnan(topm["recall_at_high"])


def test_non_finite_pairs_are_filtered():
    proteins = {f"P{i}": float(i) for i in range(8)}
    proteins["P8"] = float("nan")  # non-finite probe burden -> dropped
    probe = _probe_df(proteins)
    oracle = pd.Series({f"P{i}": float(i) for i in range(9)})
    table = compute_monitor_table(probe, oracle)
    rec = next(r for r in table if r["metric"] == "mean_excess")
    assert rec["n"] == 8
    assert math.isfinite(rec["spearman"])


# ---------------------------------------------------------------------------
# end-to-end CLI + registration
# ---------------------------------------------------------------------------


def test_analyze_main_end_to_end(tmp_path: Path):
    run_dir = tmp_path / "generation"
    run_dir.mkdir()
    proteins = {f"P{i}": float(i) for i in range(9)}
    frames = []
    for arm in ("fresh", "self_conditioned"):
        frames.append(_probe_df(proteins, arm=arm, refresh_step=0))
    pd.concat(frames, ignore_index=True).to_parquet(
        run_dir / "sc_gr_probe_refresh.parquet", index=False
    )
    oracle_path = tmp_path / "imm_head.parquet"
    pd.DataFrame(
        {"protein_id": list(proteins), "global_risk": list(proteins.values())}
    ).to_parquet(oracle_path, index=False)

    out = tmp_path / "summary.json"
    rc = main([
        "--run-dir", str(run_dir),
        "--oracle-parquet", str(oracle_path),
        "--oracle-risk-column", "global_risk",
        "--oracle-aggregate", "median",
        "--output", str(out),
    ])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["n_proteins_common"] == 9
    assert "mean_excess" in result["metrics"]
    assert "supra_mass_tau_11p75" in result["metrics"]
    assert result["best"] is not None
    assert result["best"]["arm"] in {"fresh", "self_conditioned"}


def test_script_registered_in_scripts_md():
    scripts_md = Path(__file__).resolve().parents[2] / "doc" / "SCRIPTS.md"
    text = scripts_md.read_text()
    assert "scripts/analysis/sc_gr_monitor_probe.py" in text


# ---------------------------------------------------------------------------
# SC1.3 --emit-calibration (PLAN_RF_SC_GR.md Task SC1.3)
# ---------------------------------------------------------------------------


def test_emit_calibration_band_from_freeze_horizon():
    proteins = {f"P{i}": float(i) for i in range(9)}
    # B_sc lives in the topm_lse column for the fresh arm at refresh_step 0.
    probe = _probe_df(proteins, arm="fresh", refresh_step=0, metric_col="B_sc_topm_lse_median")
    payload = emit_calibration(
        probe, aggregator="topm_lse", arm="fresh", refresh_step=0,
        low_quantile=0.25, high_quantile=0.75,
    )
    assert payload["pressure_source"] == "self_conditioned_probe"
    assert payload["aggregator"] == "topm_lse"
    assert payload["arm"] == "fresh"
    assert payload["B_high"] > payload["B_low"]
    assert payload["B_low"] == pytest.approx(np.quantile(np.arange(9.0), 0.25))
    assert payload["B_high"] == pytest.approx(np.quantile(np.arange(9.0), 0.75))


def test_emit_calibration_main_writes_json(tmp_path: Path):
    run_dir = tmp_path / "generation"
    run_dir.mkdir()
    proteins = {f"P{i}": float(i) for i in range(9)}
    frames = [_probe_df(proteins, arm=a, refresh_step=0, metric_col="B_sc_topm_lse_median")
              for a in ("fresh", "self_conditioned")]
    pd.concat(frames, ignore_index=True).to_parquet(
        run_dir / "sc_gr_probe_refresh.parquet", index=False
    )
    oracle_path = tmp_path / "imm_head.parquet"
    pd.DataFrame({"protein_id": list(proteins), "global_risk": list(proteins.values())}).to_parquet(
        oracle_path, index=False
    )
    out = tmp_path / "summary.json"
    calib_out = tmp_path / "sc_calib.json"
    rc = main([
        "--run-dir", str(run_dir),
        "--oracle-parquet", str(oracle_path),
        "--oracle-risk-column", "global_risk",
        "--output", str(out),
        "--emit-calibration", str(calib_out),
        "--calibration-aggregator", "topm_lse",
        "--calibration-arm", "fresh",
        "--calibration-refresh-step", "0",
    ])
    assert rc == 0
    calib = json.loads(calib_out.read_text())
    assert calib["pressure_source"] == "self_conditioned_probe"
    assert calib["aggregator"] == "topm_lse"
    assert calib["arm"] == "fresh"
    assert calib["B_high"] > calib["B_low"]


def test_emit_calibration_degenerate_band_fails():
    proteins = {f"P{i}": 1.0 for i in range(9)}  # constant -> q_low == q_high
    probe = _probe_df(proteins, arm="fresh", refresh_step=0, metric_col="B_sc_topm_lse_median")
    with pytest.raises(ValueError, match="B_high"):
        emit_calibration(
            probe, aggregator="topm_lse", arm="fresh", refresh_step=0,
            low_quantile=0.25, high_quantile=0.75,
        )
