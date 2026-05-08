#!/usr/bin/env python
"""Cross-arm ablation comparison for IF Improvement.

Consumes 2 to 4 run directories produced by ``run_if_imp_refiner.py
--ablation-mode``. Each run dir is expected to contain:

- ``generated.parquet`` (always)
- ``ablation_diagnostics.parquet`` (when ``--ablation-mode`` was on;
  baseline / sidecar arms get this via the
  ``DPLMBaseEntropyProbe`` passthrough so entropy fields are real
  measurements, not fake zeros)
- ``ablation_step_diagnostics.parquet`` (per-step records; optional)
- the upstream evaluator's per-design metric parquets if you ran
  ``evaluate_phase_c.py`` first (optional)

Pairing
-------
The script REQUIRES that the arms being compared cover the same
``(protein_id, design_idx)`` set. If keys differ across arms, the
default behavior is to fail; pass ``--allow-unpaired`` to fall back
to per-arm aggregates with a warning, and to compute deltas only on
the intersection of keys.

NaN-safe aggregation
--------------------
Entropy and refiner-selection columns are NaN where they were not
measured (e.g. ``fused_entropy_*`` is NaN on a baseline arm because
no fusion occurred). All aggregates use NaN-aware reductions; deltas
that would require both sides to have a real measurement are emitted
as ``null`` when one side is missing, never as fake zero.

Outputs ``comparison.parquet`` + ``comparison_summary.json`` under
``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_PAIR_KEYS = ("protein_id", "design_idx")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-arm IF Improvement ablation comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-run", default=None, help="Arm 1 run dir (DPLM only).")
    parser.add_argument("--refiner-run", default=None, help="Arm 2 run dir (DPLM + refiner).")
    parser.add_argument("--sidecar-run", default=None, help="Arm 3 run dir (DPLM + sidecar).")
    parser.add_argument("--combined-run", default=None, help="Arm 4 run dir (DPLM + sidecar + refiner).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help=(
            "If arms cover different (protein_id, design_idx) sets, "
            "compute deltas on the intersection and emit a warning "
            "instead of failing."
        ),
    )
    args = parser.parse_args(argv)

    provided = [
        p for p in (args.baseline_run, args.refiner_run, args.sidecar_run, args.combined_run)
        if p is not None
    ]
    if len(provided) < 2:
        parser.error(
            "provide at least 2 of --baseline-run/--refiner-run/--sidecar-run/--combined-run"
        )
    return args


def _maybe(value: float | None) -> float | None:
    """Convert NaN to None so JSON emits null (not 'NaN')."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isnan(float(value)):
        return None
    return float(value)


def _load_arm(run_dir: Path) -> dict[str, Any]:
    """Load a single arm's per-design data. Returns a dict with:

    - ``run_dir``       : str
    - ``arm_label``     : str (filled in by caller)
    - ``generated``     : pd.DataFrame (per-design generated rows)
    - ``diagnostics``   : pd.DataFrame or None (per-design ablation rows)
    - ``has_diagnostics``: bool
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    generated = pd.read_parquet(run_dir / "generated.parquet")
    diagnostics_path = run_dir / "ablation_diagnostics.parquet"
    has_diagnostics = diagnostics_path.is_file()
    diagnostics = pd.read_parquet(diagnostics_path) if has_diagnostics else None
    return {
        "run_dir": str(run_dir),
        "generated": generated,
        "diagnostics": diagnostics,
        "has_diagnostics": bool(has_diagnostics),
    }


def _arm_summary(arm_data: dict[str, Any], pair_keys: pd.DataFrame | None) -> dict[str, Any]:
    """Compute mean/p50/p95 over (optionally) the paired key set.

    NaN-aware: if every value is NaN, returns NaN; partial NaNs are
    skipped via ``np.nan*`` reductions.
    """
    generated = arm_data["generated"]
    if pair_keys is not None:
        gen = generated.merge(pair_keys, on=list(_PAIR_KEYS), how="inner")
    else:
        gen = generated

    wall = gen["wall_seconds"].astype(float).to_numpy() if len(gen) else np.array([])

    summary: dict[str, Any] = {
        "run_dir": arm_data["run_dir"],
        "n_designs": int(len(gen)),
        "wall_seconds_mean": _maybe(float(np.nanmean(wall))) if wall.size else None,
        "wall_seconds_p50": _maybe(float(np.nanmedian(wall))) if wall.size else None,
        "wall_seconds_p95": _maybe(float(np.nanquantile(wall, 0.95))) if wall.size else None,
        "has_diagnostics": bool(arm_data["has_diagnostics"]),
    }

    if not arm_data["has_diagnostics"] or arm_data["diagnostics"] is None:
        return summary

    diag = arm_data["diagnostics"]
    if pair_keys is not None:
        diag = diag.merge(pair_keys, on=list(_PAIR_KEYS), how="inner")
    if len(diag) == 0:
        return summary

    for col in (
        "n_selected_per_step_mean",
        "n_selected_total",
        "base_entropy_mean",
        "fused_entropy_mean",
        "refiner_entropy_mean",
        "base_entropy_q90_mean",
        "fused_entropy_q90_mean",
        "n_steps_with_refiner",
        "n_diagnostic_steps",
    ):
        if col in diag.columns:
            arr = pd.to_numeric(diag[col], errors="coerce").to_numpy()
            mean = float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")
            p50 = float(np.nanmedian(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")
            summary[f"{col}__mean"] = _maybe(mean)
            summary[f"{col}__p50"] = _maybe(p50)
    return summary


def _safe_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    va = a.get(key)
    vb = b.get(key)
    if va is None or vb is None:
        return None
    return float(vb) - float(va)


def _resolve_pair_keys(
    arms: dict[str, dict[str, Any]],
    *,
    allow_unpaired: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Return the intersection of (protein_id, design_idx) keys across
    arms, plus a dict describing per-arm coverage and any drift."""
    arm_keys = {
        label: arm["generated"][list(_PAIR_KEYS)].drop_duplicates().reset_index(drop=True)
        for label, arm in arms.items()
    }
    n_per_arm = {label: int(len(df)) for label, df in arm_keys.items()}

    pair_df: pd.DataFrame | None = None
    for label, df in arm_keys.items():
        pair_df = df if pair_df is None else pair_df.merge(df, on=list(_PAIR_KEYS), how="inner")
    n_intersection = int(len(pair_df)) if pair_df is not None else 0

    pairing_info: dict[str, Any] = {
        "n_designs_per_arm": n_per_arm,
        "n_designs_intersection": n_intersection,
        "is_paired": all(n == n_intersection for n in n_per_arm.values()) and n_intersection > 0,
    }

    if not pairing_info["is_paired"]:
        msg = (
            f"Arms cover different design sets: {n_per_arm}; "
            f"intersection has {n_intersection} designs."
        )
        if not allow_unpaired:
            raise SystemExit(
                "ERROR: " + msg + " Pass --allow-unpaired to compute deltas on the intersection only."
            )
        else:
            pairing_info["warning"] = msg

    return pair_df, pairing_info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        print(
            f"ERROR: --output-dir {output_dir} not empty; pass --overwrite to clobber.",
            file=sys.stderr,
        )
        return 2

    arm_paths = [
        ("baseline", args.baseline_run),
        ("refiner", args.refiner_run),
        ("sidecar", args.sidecar_run),
        ("sidecar_refiner", args.combined_run),
    ]
    arms_data: dict[str, dict[str, Any]] = {}
    for label, run_dir in arm_paths:
        if run_dir is None:
            continue
        arms_data[label] = _load_arm(Path(run_dir).expanduser().resolve())

    pair_df, pairing_info = _resolve_pair_keys(
        arms_data, allow_unpaired=args.allow_unpaired
    )
    pair_keys_for_summary = pair_df if pairing_info["is_paired"] else (
        pair_df if args.allow_unpaired else None
    )

    arms: dict[str, dict[str, Any]] = {}
    for label, arm_data in arms_data.items():
        summary = _arm_summary(arm_data, pair_keys_for_summary)
        summary["arm"] = label
        arms[label] = summary

    rows = list(arms.values())
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_parquet(output_dir / "comparison.parquet", index=False)

    deltas: dict[str, Any] = {}

    # Indicator 1 & 4: refiner selection + combined gain
    if "refiner" in arms and "sidecar_refiner" in arms:
        deltas["refiner_vs_combined"] = {
            "n_selected_per_step_delta": _safe_delta(
                arms["refiner"], arms["sidecar_refiner"],
                "n_selected_per_step_mean__mean",
            ),
            "fused_entropy_delta": _safe_delta(
                arms["refiner"], arms["sidecar_refiner"],
                "fused_entropy_mean__mean",
            ),
            "fused_entropy_q90_delta": _safe_delta(
                arms["refiner"], arms["sidecar_refiner"],
                "fused_entropy_q90_mean__mean",
            ),
            "wall_seconds_delta": _safe_delta(
                arms["refiner"], arms["sidecar_refiner"], "wall_seconds_mean"
            ),
        }

    # Indicator 2: high-entropy quantile gain (per arm)
    quantile_gains: dict[str, float | None] = {}
    for label, arm in arms.items():
        base_q = arm.get("base_entropy_q90_mean__mean")
        fused_q = arm.get("fused_entropy_q90_mean__mean")
        if base_q is None or fused_q is None:
            quantile_gains[label] = None
        else:
            quantile_gains[label] = float(base_q - fused_q)
    deltas["fused_minus_base_q90_per_arm"] = quantile_gains

    # Indicator 3: sidecar effect on base entropy (needs probe-derived
    # base entropy on BOTH arms; with the refiner-passthrough probe in
    # place this is now a real measurement)
    if "baseline" in arms and "sidecar" in arms:
        deltas["sidecar_minus_baseline"] = {
            "base_entropy_delta": _safe_delta(
                arms["baseline"], arms["sidecar"], "base_entropy_mean__mean"
            ),
            "base_entropy_q90_delta": _safe_delta(
                arms["baseline"], arms["sidecar"],
                "base_entropy_q90_mean__mean",
            ),
            "wall_seconds_delta": _safe_delta(
                arms["baseline"], arms["sidecar"], "wall_seconds_mean"
            ),
        }

    # Indicator 4: combined vs best-of-{refiner, sidecar}
    if "sidecar_refiner" in arms and ("refiner" in arms or "sidecar" in arms):
        candidates: list[float] = []
        for label in ("refiner", "sidecar"):
            v = arms.get(label, {}).get("fused_entropy_mean__mean")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                candidates.append(float(v))
        combined_h = arms["sidecar_refiner"].get("fused_entropy_mean__mean")
        if candidates and combined_h is not None:
            best_single = min(candidates)
            deltas["combined_vs_best_single"] = {
                "combined_minus_best_single_fused_entropy": float(
                    combined_h - best_single
                ),
                "best_single_fused_entropy": float(best_single),
                "combined_fused_entropy": float(combined_h),
            }
        else:
            deltas["combined_vs_best_single"] = None

    # Indicator 5: compute cost summary
    deltas["compute_cost_per_arm"] = {
        label: {
            "wall_seconds_mean": arm["wall_seconds_mean"],
            "wall_seconds_p50": arm["wall_seconds_p50"],
            "wall_seconds_p95": arm["wall_seconds_p95"],
            "n_designs": arm["n_designs"],
        }
        for label, arm in arms.items()
    }

    summary_payload: dict[str, Any] = {
        "pairing": pairing_info,
        "arms": arms,
        "deltas": deltas,
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True, default=str)

    print("============================================================")
    print(f"Ablation comparison written to {output_dir}")
    print(f"  arms compared          : {sorted(arms)}")
    print(f"  is_paired              : {pairing_info['is_paired']}")
    print(f"  n_designs_per_arm      : {pairing_info['n_designs_per_arm']}")
    print(f"  n_designs_intersection : {pairing_info['n_designs_intersection']}")
    print(f"  comparison.parquet rows: {len(comparison_df)}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
