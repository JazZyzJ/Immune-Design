#!/usr/bin/env python3
"""SC-GR per-residue r_i accuracy vs the NoD per-residue oracle.

The per-protein analog of RAR 0010 (`sc_gr_monitor_probe.py`), one granularity
down: RAR 0010 validated the probe's trajectory SCALAR `B_sc` against the NoD
per-protein burden (Spearman ~0.75). This script asks whether the SAME
ensemble's per-residue map `r_i` (the `residue_excess` array now persisted to
`sc_gr_probe_samples.parquet` when `self_conditioned_gr.write_residue_telemetry`)
ranks RESIDUES within a protein the way the unconditional (NoD) per-residue
burden does. This is the gate for the position-dependent ("local SC1") path:
`r_i` is a LEVEL, so it may inherit the protein-level accuracy even though the
candidate-tuple MARGINAL did not (RAR 0018). Spec: `PLAN_RF_SC_GR_ri_accuracy.md`.

Inputs (both per-residue):
* `--probe-samples`  : monitor run `sc_gr_probe_samples.parquet` with the
  `residue_excess` list-column (one row per completion). Per-protein probe map =
  median over the K samples, then median over the designs (mirrors RAR 0010's
  per-protein median-over-designs), per (arm, refresh_step, residue).
* `--oracle-residues`: NoD `imm_head_residues.parquet` from
  `evaluate_phase_c.py --imm-full` (cols protein_id, design_idx, residue_idx,
  hotspot). Per-protein oracle map = `--oracle-aggregate` over the NoD designs.

Residue indices align position-wise: same protein => same backbone/length, only
the sequence differs (guided trajectory vs NoD design), so position i is the
same backbone site in both. Spearman is rank-based, so the probe `residue_excess`
(max-covering projection minus tau) and the oracle `hotspot` (the head's native
per-residue burden) need only be monotone in residue immunogenicity, not equal.

All outputs are objective levels; no verdict is emitted (the pre-registered gate
lives in the PLAN). Pure functions are importable for tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_GROUP = ["arm", "refresh_step"]


def explode_probe_residue_map(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Per-completion `residue_excess` lists -> per-protein per-residue `r_i`.

    Reductions mirror RAR 0010: median over the K samples (within a design),
    then median over the designs. Returns columns
    ``protein_id, arm, refresh_step, residue_idx, r_i``.
    """
    if "residue_excess" not in samples_df.columns:
        raise ValueError(
            "probe samples parquet lacks 'residue_excess' -- the run must set "
            "self_conditioned_gr.write_residue_telemetry=true "
            "(config d2_d3_full_stageB_aopen_scgr_monitor_residue.yaml)"
        )
    recs: list[tuple] = []
    for row in samples_df.itertuples(index=False):
        vals = row.residue_excess
        if vals is None:
            continue
        pid = row.protein_id
        didx = int(row.design_idx)
        rstep = int(row.refresh_step)
        arm = row.arm
        for i, v in enumerate(vals):
            recs.append((pid, didx, rstep, arm, i, float(v)))
    if not recs:
        raise ValueError("no residue_excess rows found in probe samples")
    exploded = pd.DataFrame(
        recs,
        columns=["protein_id", "design_idx", "refresh_step", "arm", "residue_idx", "r_i"],
    )
    per_design = (
        exploded.groupby(
            ["protein_id", "design_idx", "refresh_step", "arm", "residue_idx"],
            sort=False,
        )["r_i"]
        .median()
        .reset_index()
    )
    per_protein = (
        per_design.groupby(
            ["protein_id", "refresh_step", "arm", "residue_idx"], sort=False
        )["r_i"]
        .median()
        .reset_index()
    )
    return per_protein


def oracle_residue_map(
    residues_df: pd.DataFrame, *, risk_col: str = "hotspot", agg: str = "median"
) -> pd.DataFrame:
    """NoD per-residue oracle: ``risk_col`` aggregated over designs per residue.

    Returns columns ``protein_id, residue_idx, oracle``.
    """
    for col in ("protein_id", "residue_idx", risk_col):
        if col not in residues_df.columns:
            raise ValueError(f"oracle residues parquet lacks column {col!r}")
    out = (
        residues_df.groupby(["protein_id", "residue_idx"], sort=False)[risk_col]
        .agg(agg)
        .reset_index()
        .rename(columns={risk_col: "oracle"})
    )
    return out


def _safe_spearman(x: np.ndarray, y: np.ndarray, *, min_n: int) -> float:
    if x.size < min_n or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def per_protein_residue_spearman(
    probe_map: pd.DataFrame, oracle_map: pd.DataFrame, *, min_residues: int = 5
) -> pd.DataFrame:
    """Per (protein, arm, refresh_step) Spearman(r_i, oracle) across residues."""
    merged = probe_map.merge(oracle_map, on=["protein_id", "residue_idx"], how="inner")
    rows: list[dict] = []
    for (pid, arm, rstep), g in merged.groupby(
        ["protein_id", "arm", "refresh_step"], sort=False
    ):
        rho = _safe_spearman(
            g["r_i"].to_numpy(), g["oracle"].to_numpy(), min_n=min_residues
        )
        rows.append(
            {
                "protein_id": pid,
                "arm": arm,
                "refresh_step": int(rstep),
                "n_residues": int(len(g)),
                "spearman": rho,
            }
        )
    return pd.DataFrame(rows)


def _tercile(values: np.ndarray) -> np.ndarray:
    """Rank-tercile labels 0/1/2 (2 = highest); first-rank tie-break."""
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, 3, labels=[0, 1, 2]).to_numpy()


def residue_tail_metrics(
    probe_map: pd.DataFrame, oracle_map: pd.DataFrame, *, min_residues: int = 9
) -> pd.DataFrame:
    """Per-residue tail metrics, per (arm, refresh_step), pooled over proteins.

    Within each protein, residues are tercile-binned by oracle and by probe.
    ``recall_at_high`` = P(probe high | oracle high); ``p_low_given_high`` =
    P(probe low | oracle high) -- the per-residue analog of RAR 0010's tail.
    """
    merged = probe_map.merge(oracle_map, on=["protein_id", "residue_idx"], how="inner")
    acc: dict[tuple, dict] = {}
    for (pid, arm, rstep), g in merged.groupby(
        ["protein_id", "arm", "refresh_step"], sort=False
    ):
        if len(g) < min_residues or g["oracle"].nunique() < 3 or g["r_i"].nunique() < 3:
            continue
        ob = _tercile(g["oracle"].to_numpy())
        pb = _tercile(g["r_i"].to_numpy())
        high = ob == 2
        n_high = int(high.sum())
        if n_high == 0:
            continue
        key = (arm, int(rstep))
        a = acc.setdefault(key, {"hit_high": 0, "low_given_high": 0, "n_high": 0, "n_prot": 0})
        a["hit_high"] += int(((pb == 2) & high).sum())
        a["low_given_high"] += int(((pb == 0) & high).sum())
        a["n_high"] += n_high
        a["n_prot"] += 1
    rows = []
    for (arm, rstep), a in acc.items():
        nh = a["n_high"]
        rows.append(
            {
                "arm": arm,
                "refresh_step": rstep,
                "n_proteins": a["n_prot"],
                "recall_at_high_residue": a["hit_high"] / nh if nh else float("nan"),
                "p_low_given_high_residue": a["low_given_high"] / nh if nh else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def region_spearman(
    probe_map: pd.DataFrame,
    oracle_map: pd.DataFrame,
    *,
    region_size: int = 9,
    reduce: str = "max",
    min_regions: int = 5,
) -> pd.DataFrame:
    """Region-level Spearman: residues binned into windows of ``region_size``.

    The coarser granularity relevant to targeting (which REGION is hot, not the
    exact residue). Per protein, reduce probe & oracle within each region, then
    Spearman across regions; aggregated per (arm, refresh_step).
    """
    merged = probe_map.merge(oracle_map, on=["protein_id", "residue_idx"], how="inner")
    merged = merged.assign(region=(merged["residue_idx"] // int(region_size)).astype(int))
    reg = (
        merged.groupby(["protein_id", "arm", "refresh_step", "region"], sort=False)
        .agg(r_i=("r_i", reduce), oracle=("oracle", reduce))
        .reset_index()
    )
    rows = []
    for (arm, rstep), g in reg.groupby(["arm", "refresh_step"], sort=False):
        rhos = []
        for _pid, gp in g.groupby("protein_id", sort=False):
            rho = _safe_spearman(
                gp["r_i"].to_numpy(), gp["oracle"].to_numpy(), min_n=min_regions
            )
            if not np.isnan(rho):
                rhos.append(rho)
        rows.append(
            {
                "arm": arm,
                "refresh_step": int(rstep),
                "n_proteins": len(rhos),
                "region_spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
                "region_spearman_median": float(np.median(rhos)) if rhos else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_residue_accuracy(
    samples_df: pd.DataFrame,
    residues_df: pd.DataFrame,
    *,
    risk_col: str = "hotspot",
    oracle_agg: str = "median",
    min_residues: int = 5,
    region_size: int = 9,
) -> tuple[dict, pd.DataFrame]:
    """End-to-end: returns (summary dict by arm x refresh_step, per-protein df)."""
    probe_map = explode_probe_residue_map(samples_df)
    oracle = oracle_residue_map(residues_df, risk_col=risk_col, agg=oracle_agg)
    per_protein = per_protein_residue_spearman(
        probe_map, oracle, min_residues=min_residues
    )
    tail = residue_tail_metrics(probe_map, oracle)
    region = region_spearman(probe_map, oracle, region_size=region_size)

    summary: list[dict] = []
    valid = per_protein.dropna(subset=["spearman"])
    for (arm, rstep), g in valid.groupby(_GROUP, sort=False):
        cell = {
            "arm": arm,
            "refresh_step": int(rstep),
            "n_proteins": int(len(g)),
            "residue_spearman_mean": float(g["spearman"].mean()),
            "residue_spearman_median": float(g["spearman"].median()),
        }
        t = tail[(tail.arm == arm) & (tail.refresh_step == rstep)]
        if len(t):
            cell["recall_at_high_residue"] = float(t.iloc[0]["recall_at_high_residue"])
            cell["p_low_given_high_residue"] = float(t.iloc[0]["p_low_given_high_residue"])
        r = region[(region.arm == arm) & (region.refresh_step == rstep)]
        if len(r):
            cell["region_spearman_mean"] = float(r.iloc[0]["region_spearman_mean"])
            cell["region_spearman_median"] = float(r.iloc[0]["region_spearman_median"])
        summary.append(cell)
    return {"by_arm_refresh": summary}, per_protein


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-samples", required=True, type=Path)
    p.add_argument("--oracle-residues", required=True, type=Path)
    p.add_argument("--oracle-risk-column", default="hotspot")
    p.add_argument("--oracle-aggregate", default="median", choices=["median", "mean"])
    p.add_argument("--min-residues", type=int, default=5)
    p.add_argument("--region-size", type=int, default=9)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    for key in (
        "probe_samples", "oracle_residues", "oracle_risk_column",
        "oracle_aggregate", "min_residues", "region_size", "output",
    ):
        print(f"[scgr_residue_accuracy] {key}: {getattr(args, key)}", flush=True)

    samples_df = pd.read_parquet(args.probe_samples)
    residues_df = pd.read_parquet(args.oracle_residues)
    summary, per_protein = compute_residue_accuracy(
        samples_df,
        residues_df,
        risk_col=args.oracle_risk_column,
        oracle_agg=args.oracle_aggregate,
        min_residues=args.min_residues,
        region_size=args.region_size,
    )
    summary["provenance"] = {
        "probe_samples": str(args.probe_samples),
        "oracle_residues": str(args.oracle_residues),
        "oracle_risk_column": args.oracle_risk_column,
        "oracle_aggregate": args.oracle_aggregate,
        "min_residues": args.min_residues,
        "region_size": args.region_size,
        "n_probe_samples": int(len(samples_df)),
        "n_oracle_rows": int(len(residues_df)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    per_protein_path = args.output.with_suffix(".per_protein.csv")
    per_protein.to_csv(per_protein_path, index=False)
    print(f"[scgr_residue_accuracy] wrote {args.output}", flush=True)
    print(f"[scgr_residue_accuracy] wrote {per_protein_path}", flush=True)
    for cell in summary["by_arm_refresh"]:
        print(
            f"[scgr_residue_accuracy]   arm={cell['arm']} "
            f"refresh_step={cell['refresh_step']} "
            f"residue_spearman_median={cell['residue_spearman_median']:.4f} "
            f"region_spearman_median={cell.get('region_spearman_median', float('nan')):.4f} "
            f"recall_at_high_residue={cell.get('recall_at_high_residue', float('nan')):.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
