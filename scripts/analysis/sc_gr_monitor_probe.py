#!/usr/bin/env python3
"""SC-GR monitor analysis (PLAN_RF_SC_GR.md Task SC0.5 / doc/Self-Cond_GR.md §6).

Reads a monitor run's ``sc_gr_probe_refresh.parquet`` and an oracle burden
(``imm_head.parquet`` from the NoD / D3 reference run) and scores each candidate
burden estimator (``arm`` x ``refresh_step`` x aggregator) against the oracle:

* ``Spearman(B_sc, oracle per-protein burden)`` — the rank-agreement gate;
* a tercile crosswalk diagonal (probe bin == oracle bin fraction);
* ``P(probe low | oracle high)`` — the true-high-misclassified-as-low rate that
  cost C.1 its high-burden gain;
* ``Recall@High = P(probe high | oracle high)``.

The best estimator is chosen by ``Recall@High`` first, ``Spearman`` second — the
*tail*, not the mean correlation, is what the gain must get right. This is a
monitor-only readout; it never changes any config or actuation.

Pure functions (no I/O) are unit-tested; ``analyze`` / ``main`` wire the parquet
I/O + CLI. Lightweight on purpose (pandas + numpy only) so it does not pull in
the head / torch deps of the run script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROBE_REFRESH_FILENAME = "sc_gr_probe_refresh.parquet"

# Map a monitored aggregator name -> its B_sc per-refresh column stem
# (PLAN_RF_SC_GR.md §2/§3). ``supra_mass_tau_<label>`` columns are discovered
# dynamically so multiple supra thresholds are all scored.
_FIXED_METRIC_COLUMNS = {
    "mean_excess": "B_sc_mean_excess_median",
    "topm_lse": "B_sc_topm_lse_median",
}
_SUPRA_COL_RE = re.compile(r"^B_sc_supra_mass_tau_(.+)_median$")


# ---------------------------------------------------------------------------
# Pure statistics
# ---------------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean rank)."""
    arr = np.asarray(a, dtype=float)
    n = arr.size
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN for <2 points or a constant input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def tercile_bins(values: np.ndarray) -> np.ndarray:
    """Bin into terciles: 0=low, 1=mid, 2=high (cuts at the 1/3, 2/3 quantiles)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(arr, [q1, q2]).astype(int)


# ---------------------------------------------------------------------------
# Burden series
# ---------------------------------------------------------------------------


def discover_metrics(probe_df: pd.DataFrame) -> dict[str, str]:
    """Map aggregator name -> B_sc column present in the probe-refresh frame."""
    metrics: dict[str, str] = {}
    for name, col in _FIXED_METRIC_COLUMNS.items():
        if col in probe_df.columns:
            metrics[name] = col
    for col in probe_df.columns:
        m = _SUPRA_COL_RE.match(str(col))
        if m is not None:
            metrics[f"supra_mass_tau_{m.group(1)}"] = col
    return metrics


def aggregate_oracle(
    oracle_df: pd.DataFrame,
    *,
    protein_col: str,
    risk_col: str,
    agg: str,
) -> pd.Series:
    """Per-protein oracle burden (median/mean over the oracle rows).

    Already-per-protein input is unchanged (the median/mean of one value is
    itself), so this is safe whether the oracle parquet is per-design or
    per-protein.
    """
    if protein_col not in oracle_df.columns:
        raise ValueError(f"oracle parquet missing protein column {protein_col!r}")
    if risk_col not in oracle_df.columns:
        raise ValueError(f"oracle parquet missing risk column {risk_col!r}")
    grouped = oracle_df.groupby(protein_col)[risk_col]
    if agg == "median":
        series = grouped.median()
    elif agg == "mean":
        series = grouped.mean()
    else:
        raise ValueError(f"oracle aggregate must be median|mean (got {agg!r})")
    return series.astype(float)


def _is_finite_number(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _tercile_cut_degenerate(values: np.ndarray) -> bool:
    """True when the 1/3, 2/3 quantiles collapse (no usable tercile split).

    A constant or near-constant burden gives ``q1 == q2``; ``np.digitize`` would
    then dump every value into a single bin (e.g. all-high → a spurious
    ``Recall@High = 1.0``). Treat that — and fewer than 3 finite points — as
    degenerate so the bin stats are reported as NaN instead of a false perfect.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return True
    q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    return not (q1 < q2)


def _bin_stats(probe_burden: np.ndarray, oracle_burden: np.ndarray) -> dict[str, float]:
    """Crosswalk diagonal + P(probe low | oracle high) + Recall@High."""
    ob = tercile_bins(oracle_burden)
    pb = tercile_bins(probe_burden)
    crosswalk_diagonal = float(np.mean(ob == pb)) if ob.size else float("nan")
    high = ob == 2
    n_high = int(high.sum())
    if n_high:
        p_low_given_high = float(np.mean(pb[high] == 0))
        recall_at_high = float(np.mean(pb[high] == 2))
    else:
        p_low_given_high = float("nan")
        recall_at_high = float("nan")
    return {
        "crosswalk_diagonal": crosswalk_diagonal,
        "p_low_given_high": p_low_given_high,
        "recall_at_high": recall_at_high,
        "n_oracle_high": n_high,
    }


def compute_monitor_table(
    probe_df: pd.DataFrame, oracle_series: pd.Series
) -> list[dict[str, Any]]:
    """Score every (arm, refresh_step, metric) estimator against the oracle.

    Per-protein probe burden = the median ``B_sc`` over the protein's designs at
    that (arm, refresh_step). Joined to the oracle on the common protein set.
    """
    metrics = discover_metrics(probe_df)
    rows: list[dict[str, Any]] = []
    arms = sorted(str(a) for a in probe_df["arm"].unique())
    refresh_steps = sorted(int(s) for s in probe_df["refresh_step"].unique())
    for arm in arms:
        for rstep in refresh_steps:
            sub = probe_df[
                (probe_df["arm"] == arm) & (probe_df["refresh_step"] == rstep)
            ]
            if sub.empty:
                continue
            for mname, col in metrics.items():
                probe_burden = sub.groupby("protein_id")[col].median()
                common = probe_burden.index.intersection(oracle_series.index)
                # Keep only finite (probe, oracle) pairs — a NaN burden on either
                # side must not enter the rank correlation or the binning.
                pb_all = probe_burden.loc[common].to_numpy(dtype=float)
                ob_all = oracle_series.loc[common].to_numpy(dtype=float)
                finite = np.isfinite(pb_all) & np.isfinite(ob_all)
                pb = pb_all[finite]
                ob = ob_all[finite]
                n = int(finite.sum())
                record: dict[str, Any] = {
                    "arm": arm,
                    "refresh_step": int(rstep),
                    "metric": mname,
                    "n": n,
                    "spearman": float("nan"),
                    "crosswalk_diagonal": float("nan"),
                    "p_low_given_high": float("nan"),
                    "recall_at_high": float("nan"),
                    "n_oracle_high": 0,
                }
                if n >= 2:
                    record["spearman"] = spearman(pb, ob)
                    # Bin stats are only meaningful when BOTH burdens have a real
                    # tercile split; a collapsed cut (constant probe/oracle) would
                    # otherwise fabricate a perfect Recall@High.
                    if not _tercile_cut_degenerate(pb) and not _tercile_cut_degenerate(ob):
                        record.update(_bin_stats(pb, ob))
                rows.append(record)
    return rows


def _rank_key(record: dict[str, Any]) -> tuple[float, float]:
    def _f(v: Any) -> float:
        f = float(v) if v is not None else float("nan")
        return -math.inf if math.isnan(f) else f

    return (_f(record.get("recall_at_high")), _f(record.get("spearman")))


_B_SC_MEDIAN_FIXED = {
    "mean_excess": "B_sc_mean_excess_median",
    "topm_lse": "B_sc_topm_lse_median",
}
_B_SC_SUPRA_RE = re.compile(r"^B_sc_supra_mass_tau_(.+)_median$")


def _b_sc_median_column(probe_df: pd.DataFrame, aggregator: str) -> str:
    """Map an actuation aggregator name -> its B_sc-median column in the frame."""
    if aggregator in _B_SC_MEDIAN_FIXED:
        return _B_SC_MEDIAN_FIXED[aggregator]
    if aggregator == "supra_mass":
        cols = [c for c in probe_df.columns if _B_SC_SUPRA_RE.match(str(c))]
        if len(cols) != 1:
            raise ValueError(
                "supra_mass calibration needs exactly one B_sc_supra_mass_tau_* "
                f"column (found {cols})"
            )
        return cols[0]
    raise ValueError(f"unknown aggregator {aggregator!r}")


def emit_calibration(
    probe_df: pd.DataFrame,
    *,
    aggregator: str,
    arm: str,
    refresh_step: int,
    low_quantile: float,
    high_quantile: float,
    schema_version: str = "scgr_betaonly_calibration.v1",
) -> dict[str, Any]:
    """Build the SC1 ``self_conditioned_probe`` calibration band (PLAN SC1.3).

    Bands come from the per-protein ``B_sc`` distribution at the **freeze horizon**
    — the ``(arm, refresh_step)`` SC1 actually actuates on, NOT necessarily the
    summary's best-discrimination step. Per-protein burden = median over designs.
    """
    col = _b_sc_median_column(probe_df, aggregator)
    sub = probe_df[
        (probe_df["arm"] == arm) & (probe_df["refresh_step"] == int(refresh_step))
    ]
    if sub.empty:
        raise ValueError(
            f"no probe rows for arm={arm!r} refresh_step={refresh_step}"
        )
    per_protein = sub.groupby("protein_id")[col].median().to_numpy(dtype=float)
    vals = per_protein[np.isfinite(per_protein)]
    if vals.size < 2:
        raise ValueError(
            f"need >=2 finite per-protein B_sc values to calibrate (got {vals.size})"
        )
    B_low = float(np.quantile(vals, float(low_quantile)))
    B_high = float(np.quantile(vals, float(high_quantile)))
    if not (B_high > B_low):
        raise ValueError(
            "calibration band collapsed: requires B_high > B_low "
            f"(got B_low={B_low}, B_high={B_high}); the B_sc distribution at this "
            "horizon is too concentrated"
        )
    return {
        "schema_version": schema_version,
        "pressure_source": "self_conditioned_probe",
        "aggregator": aggregator,
        "arm": arm,
        "refresh_step": int(refresh_step),
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "B_low": B_low,
        "B_high": B_high,
    }


def select_best_metric(table: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best estimator by Recall@High, tie-broken by Spearman.

    Requires BOTH a finite Recall@High and a finite Spearman so a degenerate /
    non-informative estimator (constant burden ⇒ collapsed terciles ⇒ NaN bin
    stats, or a constant ⇒ NaN Spearman) can never be selected as "best".
    """
    valid = [
        r
        for r in table
        if int(r.get("n", 0)) >= 2
        and _is_finite_number(r.get("recall_at_high"))
        and _is_finite_number(r.get("spearman"))
    ]
    if not valid:
        return None
    return max(valid, key=_rank_key)


# ---------------------------------------------------------------------------
# Pipeline + CLI
# ---------------------------------------------------------------------------


def analyze(
    *,
    run_dir: Path,
    oracle_parquet: Path,
    oracle_risk_column: str,
    oracle_aggregate: str,
    oracle_protein_column: str = "protein_id",
) -> dict[str, Any]:
    probe_path = Path(run_dir) / PROBE_REFRESH_FILENAME
    if not probe_path.exists():
        raise FileNotFoundError(f"SC-GR probe refresh parquet not found: {probe_path}")
    probe_df = pd.read_parquet(probe_path)
    oracle_df = pd.read_parquet(oracle_parquet)
    oracle_series = aggregate_oracle(
        oracle_df,
        protein_col=oracle_protein_column,
        risk_col=oracle_risk_column,
        agg=oracle_aggregate,
    )
    table = compute_monitor_table(probe_df, oracle_series)
    best = select_best_metric(table)
    common_proteins = set(probe_df["protein_id"].unique()) & set(oracle_series.index)
    return {
        "run_dir": str(run_dir),
        "oracle_parquet": str(oracle_parquet),
        "oracle_risk_column": oracle_risk_column,
        "oracle_aggregate": oracle_aggregate,
        "n_proteins_probe": int(probe_df["protein_id"].nunique()),
        "n_proteins_oracle": int(len(oracle_series)),
        "n_proteins_common": int(len(common_proteins)),
        "metrics": sorted(discover_metrics(probe_df).keys()),
        "table": table,
        "best": best,
    }


def _jsonify(obj: Any) -> Any:
    """Recursively replace NaN/inf with None so the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SC-GR monitor probe analysis")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--oracle-parquet", required=True, type=Path)
    parser.add_argument("--oracle-risk-column", default="global_risk")
    parser.add_argument("--oracle-protein-column", default="protein_id")
    parser.add_argument(
        "--oracle-aggregate", default="median", choices=["median", "mean"]
    )
    parser.add_argument("--output", required=True, type=Path)
    # SC1.3 calibration emission (optional). Defaults match RAR 0010.
    parser.add_argument("--emit-calibration", type=Path, default=None)
    parser.add_argument(
        "--calibration-aggregator", default="topm_lse",
        choices=["mean_excess", "topm_lse", "supra_mass"],
    )
    parser.add_argument(
        "--calibration-arm", default="fresh", choices=["fresh", "self_conditioned"]
    )
    parser.add_argument("--calibration-refresh-step", type=int, default=0)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    args = parser.parse_args(argv)

    result = analyze(
        run_dir=args.run_dir,
        oracle_parquet=args.oracle_parquet,
        oracle_risk_column=args.oracle_risk_column,
        oracle_aggregate=args.oracle_aggregate,
        oracle_protein_column=args.oracle_protein_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(_jsonify(result), f, indent=2)
        f.write("\n")

    if args.emit_calibration is not None:
        probe_df = pd.read_parquet(Path(args.run_dir) / PROBE_REFRESH_FILENAME)
        calib = emit_calibration(
            probe_df,
            aggregator=args.calibration_aggregator,
            arm=args.calibration_arm,
            refresh_step=args.calibration_refresh_step,
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
        )
        args.emit_calibration.parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_calibration, "w") as f:
            json.dump(_jsonify(calib), f, indent=2)
            f.write("\n")
        print(
            f"[sc-gr-monitor] wrote calibration {args.emit_calibration} "
            f"(arm={calib['arm']} aggregator={calib['aggregator']} "
            f"refresh_step={calib['refresh_step']} "
            f"B_low={calib['B_low']:.4f} B_high={calib['B_high']:.4f})",
            flush=True,
        )

    best = result["best"]
    if best is not None:
        print(
            f"[sc-gr-monitor] best estimator: arm={best['arm']} "
            f"refresh_step={best['refresh_step']} metric={best['metric']} "
            f"Recall@High={best['recall_at_high']} Spearman={best['spearman']}",
            flush=True,
        )
    else:
        print("[sc-gr-monitor] no estimator with >=2 common proteins", flush=True)
    print(f"[sc-gr-monitor] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
