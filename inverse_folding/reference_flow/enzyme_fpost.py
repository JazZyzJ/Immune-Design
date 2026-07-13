"""F_post v0 measurement (uricase enzyme mode v0) — PLAN_URICASE_ENZYME_MODE Task U7.

One pure builder, ``build_f_post_v0``, that assembles the v0 ``f_post_v0.parquet``
measurement envelope from the hard-constraint telemetry (``constraints_applied``)
and the structural eval frame (``structural.parquet``). The v0 envelope is:

    f_anchor:        hard_invariant      (pass | fail)
    f_fold:          ranking_only        (scTM present) | unavailable
    f_assembly:      unavailable         (monomer in v0)
    f_site_scaffold: unavailable         (monomer in v0)

Structural deltas vs a same-protocol WT facade structural run are reported only
when that run is supplied and the protein is present in it; otherwise every
``delta_*_vs_wt`` column is ``pd.NA`` (never a placeholder/zero). The packaged WT
facade run has no ``structural.parquet``, so NA is the expected v0 path.

The structural input is the canonical v2 schema: the retained global RMSD is
``global_ca_RMSD``; active-site geometry remains in the index-addressable
``structural_residues.parquet`` sidecar and is never replaced by a C-alpha shell.

This is a PURE function: DataFrame(s)/records in -> DataFrame out. No file IO, no
CLI, no printing. The CLI wrapper is built separately by the orchestrator.
"""

from __future__ import annotations

import pandas as pd

from inverse_folding.reference_flow.enzyme_telemetry import normalize_design_keys

_OUTPUT_COLUMNS = [
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

# Absolute structural metric -> its delta column name.
_DELTA_METRICS = {
    "scTM": "delta_scTM_vs_wt",
    "global_ca_RMSD": "delta_global_ca_RMSD_vs_wt",
    "pLDDT": "delta_pLDDT_vs_wt",
    "recovery": "delta_recovery_vs_wt",
}


def _aggregate_anchor_status(
    constraint_rows: list[dict] | pd.DataFrame,
) -> dict[tuple[str, int], int]:
    """Per-design count of anchor mismatches (``preserved == False`` rows).

    Returns ``{(protein_id, design_idx): num_anchor_mismatches}``. A design absent
    from this map has no constraint rows and is treated as unconstrained (pass,
    zero mismatches) by the caller.
    """
    if isinstance(constraint_rows, pd.DataFrame):
        frame = constraint_rows
    else:
        frame = pd.DataFrame(list(constraint_rows))

    if frame.empty:
        return {}

    frame = normalize_design_keys(frame, unique=False)
    # preserved==False counts as a mismatch; treat missing as not-preserved (mismatch).
    preserved = frame["preserved"].astype("boolean").fillna(False)
    frame = frame.assign(_mismatch=(~preserved).astype(int))
    grouped = frame.groupby(["protein_id", "design_idx"])["_mismatch"].sum()
    return {
        (str(protein_id), int(design_idx)): int(count)
        for (protein_id, design_idx), count in grouped.items()
    }


def build_f_post_v0(
    constraint_rows: list[dict] | pd.DataFrame,
    structural_df: pd.DataFrame,
    wt_structural_df: pd.DataFrame | None = None,
    *,
    predictor: str = "esmfold2",
    predictor_version: str = "na",
) -> pd.DataFrame:
    """Build the v0 ``f_post_v0`` measurement envelope.

    One output row per ``(protein_id, design_idx)`` present in ``structural_df``.

    Args:
        constraint_rows: ``constraints_applied`` rows (one per design x anchor) as
            ``list[dict]`` or DataFrame. A design with no rows here is unconstrained
            and scores ``f_anchor_status='pass'``, ``num_anchor_mismatches=0``.
        structural_df: ``structural.parquet`` frame; one row per design.
        wt_structural_df: same-protocol WT facade structural frame (single design
            per protein) or ``None``. When ``None`` or when a protein is absent from
            it, all ``delta_*_vs_wt`` columns for that protein are ``pd.NA``.
        predictor: structure predictor name stamped into every row.
        predictor_version: predictor version stamped into every row.

    Returns:
        DataFrame with the exact columns of the ``f_post_v0`` sidecar.
    """
    structural = normalize_design_keys(structural_df, unique=True)
    mismatch_by_design = _aggregate_anchor_status(constraint_rows)

    # WT lookup: metric value keyed by protein_id (one WT design per protein).
    wt_lookup: dict[str, dict[str, object]] = {}
    if wt_structural_df is not None:
        wt = normalize_design_keys(wt_structural_df, unique=True)
        for _, wt_row in wt.iterrows():
            wt_lookup[str(wt_row["protein_id"])] = {
                metric: wt_row[metric] for metric in _DELTA_METRICS
            }

    records: list[dict[str, object]] = []
    for _, row in structural.iterrows():
        protein_id = str(row["protein_id"])
        design_idx = int(row["design_idx"])

        num_mismatches = mismatch_by_design.get((protein_id, design_idx), 0)
        f_anchor_status = "pass" if num_mismatches == 0 else "fail"

        sctm = row["scTM"]
        f_fold_status = "ranking_only" if not pd.isna(sctm) else "unavailable"

        wt_metrics = wt_lookup.get(protein_id)
        deltas: dict[str, object] = {}
        for metric, delta_col in _DELTA_METRICS.items():
            if wt_metrics is None:
                deltas[delta_col] = pd.NA
            else:
                design_val = row[metric]
                wt_val = wt_metrics[metric]
                if pd.isna(design_val) or pd.isna(wt_val):
                    deltas[delta_col] = pd.NA
                else:
                    deltas[delta_col] = design_val - wt_val

        records.append(
            {
                "protein_id": protein_id,
                "design_idx": design_idx,
                "f_anchor_status": f_anchor_status,
                "num_anchor_mismatches": int(num_mismatches),
                "f_fold_status": f_fold_status,
                "scTM": sctm,
                "global_ca_RMSD": row["global_ca_RMSD"],
                "pLDDT": row["pLDDT"],
                "recovery": row["recovery"],
                "foldability": row["foldability"],
                **deltas,
                "reference_type": "predicted_target",
                "predictor": predictor,
                "predictor_version": predictor_version,
                "calibration_status": "ranking_only",
                "f_assembly": "unavailable",
                "f_site_scaffold": "unavailable",
            }
        )

    return pd.DataFrame.from_records(records, columns=_OUTPUT_COLUMNS)
