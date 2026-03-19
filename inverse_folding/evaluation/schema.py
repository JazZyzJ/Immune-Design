"""Frozen metric schema for the IF evaluation pipeline.

Defines required column sets for each output table and a validator
that rejects incomplete rows. All downstream modules (M, N) must
produce outputs conforming to these schemas.
"""

from typing import FrozenSet

import pandas as pd


class EvalSchemaError(Exception):
    """Raised when evaluation output violates the frozen metric schema."""


# ── Frozen column sets (PLAN_IF.md §L0) ─────────────────────────────────────

STRUCTURAL_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "sequence",
    "scTM",
    "pLDDT",
    "bb_RMSD",
    "recovery",
    "foldability",
})

IMMUNOGENICITY_HEAD_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "global_risk",
    "mean_hotspot",
    "max_hotspot",
    "n_hotspot_positions",
})

IMMUNOGENICITY_NMP_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "n_strong_binders",
    "n_weak_binders",
    "mean_best_rank",
    "n_windows_scored",
})

COMPARISON_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "delta_global_risk_head",
    "delta_mean_best_rank_nmp",
    "delta_n_strong_binders",
    "hotspot_reduction",
    "mutation_count",
    "risk_per_mutation",
})


# ── Public API ───────────────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: FrozenSet[str],
) -> pd.DataFrame:
    """Validate that a DataFrame contains all required columns and is non-empty.

    Returns the DataFrame unchanged if valid; raises EvalSchemaError otherwise.
    """
    if df.empty:
        raise EvalSchemaError(
            f"DataFrame is empty; expected columns: {sorted(required_columns)}"
        )

    missing = required_columns - set(df.columns)
    if missing:
        raise EvalSchemaError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df
