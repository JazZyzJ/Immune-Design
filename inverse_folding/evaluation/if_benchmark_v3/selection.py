"""Exact-sequence grouping and frozen C5/C7 deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _sequence(value: Any) -> str:
    return "".join(str(value).split()).upper()


def build_exact_sequence_groups(
    entity_rows: Iterable[Mapping[str, Any]],
    *,
    min_length: int = 100,
    max_length: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply C2 and retain every entity as an ordered structure fallback."""

    rows = [dict(row) for row in entity_rows]
    entity_ids = [str(row.get("rcsb_entity_id") or "") for row in rows]
    if not all(entity_ids) or len(entity_ids) != len(set(entity_ids)):
        raise ValueError("duplicate rcsb_entity_id or missing entity identity")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sequence = _sequence(row.get("entity_sequence"))
        if not sequence or not set(sequence) <= AA20:
            raise ValueError(f"non-AA20 entity sequence: {row['rcsb_entity_id']}")
        if not min_length <= len(sequence) <= max_length:
            raise ValueError(f"entity sequence length outside {min_length}-{max_length}")
        resolution = row.get("resolution")
        if (
            not isinstance(resolution, (int, float))
            or isinstance(resolution, bool)
            or not math.isfinite(float(resolution))
        ):
            raise ValueError(f"entity resolution missing/non-finite: {row['rcsb_entity_id']}")
        normalized = dict(row)
        normalized["rcsb_entity_id"] = str(row["rcsb_entity_id"])
        normalized["entity_sequence"] = sequence
        normalized["resolution"] = float(resolution)
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        grouped.setdefault(digest, []).append(normalized)

    group_rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for digest, members in sorted(grouped.items()):
        members.sort(key=lambda row: (row["resolution"], row["rcsb_entity_id"]))
        unit_id = f"t2-seq:{digest}"
        group_rows.append(
            {
                "selection_unit_id": unit_id,
                "entity_sequence": members[0]["entity_sequence"],
                "entity_sequence_sha256": digest,
                "entity_count": len(members),
                "ordered_entity_fallbacks": [row["rcsb_entity_id"] for row in members],
            }
        )
        for fallback_rank, member in enumerate(members, start=1):
            edges.append(
                {
                    "selection_unit_id": unit_id,
                    "fallback_rank": fallback_rank,
                    **member,
                }
            )
    return group_rows, edges


_C5_IDENTITY_COLUMNS = (
    "selection_unit_id", "source_sequence_sha256", "score_identity_sha256",
)
_C7_IDENTITY_COLUMNS = (
    "selection_unit_id", "sequence_sha256", "cath_query_sha256",
    "cath_reference_sha256", "cath_tool_sha256", "cath_parameters_sha256",
    "cath_overlap_flag", "nmp_evidence_sha256",
)


def _input_digest(
    frame: pd.DataFrame, value_col: str, identity_columns: tuple[str, ...]
) -> str:
    columns = list(dict.fromkeys([*identity_columns, value_col]))
    ordered = frame[columns].copy().sort_values("selection_unit_id", kind="stable")
    records = []
    for raw in ordered.to_dict("records"):
        row = {}
        for key, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                raise ValueError(f"eligible identity column {key} is null")
            row[key] = value
        records.append(row)
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class BinnedSampleResult:
    selected_ids: tuple[str, ...]
    histogram: tuple[dict[str, Any], ...]
    value_col: str
    target: int
    n_bins: int
    mode: str
    stage: str
    identity_columns: tuple[str, ...]
    min_per_populated_bin: int
    seed: int
    mu: float | None
    sigma: float | None
    eligible_digest: str
    numpy_version: str
    bit_generator: str


def binned_sample_to_dict(result: BinnedSampleResult) -> dict[str, Any]:
    return {
        "selected_ids": list(result.selected_ids),
        "histogram": list(result.histogram),
        "value_col": result.value_col,
        "target": result.target,
        "n_bins": result.n_bins,
        "mode": result.mode,
        "stage": result.stage,
        "identity_columns": list(result.identity_columns),
        "min_per_populated_bin": result.min_per_populated_bin,
        "seed": result.seed,
        "mu": result.mu,
        "sigma": result.sigma,
        "eligible_digest": result.eligible_digest,
        "numpy_version": result.numpy_version,
        "bit_generator": result.bit_generator,
    }


def binned_sample_from_dict(payload: Mapping[str, Any]) -> BinnedSampleResult:
    return BinnedSampleResult(
        selected_ids=tuple(str(value) for value in payload["selected_ids"]),
        histogram=tuple(dict(value) for value in payload["histogram"]),
        value_col=str(payload["value_col"]),
        target=int(payload["target"]),
        n_bins=int(payload["n_bins"]),
        mode=str(payload["mode"]),
        stage=str(payload["stage"]),
        identity_columns=tuple(str(value) for value in payload["identity_columns"]),
        min_per_populated_bin=int(payload["min_per_populated_bin"]),
        seed=int(payload["seed"]),
        mu=None if payload.get("mu") is None else float(payload["mu"]),
        sigma=None if payload.get("sigma") is None else float(payload["sigma"]),
        eligible_digest=str(payload["eligible_digest"]),
        numpy_version=str(payload["numpy_version"]),
        bit_generator=str(payload["bit_generator"]),
    )


def deterministic_binned_sample(
    frame: pd.DataFrame,
    *,
    value_col: str,
    target: int,
    n_bins: int = 20,
    mode: str,
    stage: str,
    min_per_populated_bin: int,
    seed: int = 42,
    mu: float | None = None,
    sigma: float | None = None,
) -> BinnedSampleResult:
    """Run the exact C5 uniform or C7 Gaussian allocation and draw contract."""

    if mode not in {"uniform", "gaussian"}:
        raise ValueError("mode must be uniform or gaussian")
    if stage not in {"c5", "c7"}:
        raise ValueError("stage must be c5 or c7")
    if n_bins < 1 or target < 0 or min_per_populated_bin < 0:
        raise ValueError("invalid sampling dimensions")
    identity_columns = _C5_IDENTITY_COLUMNS if stage == "c5" else _C7_IDENTITY_COLUMNS
    required = {*identity_columns, value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"eligible table missing columns: {sorted(missing)}")
    for column in identity_columns:
        if column.endswith("sha256"):
            if any(not isinstance(value, str) or len(value) != 64 for value in frame[column]):
                raise ValueError(f"eligible identity column {column} is not SHA-256")
    if stage == "c7":
        if any(not isinstance(value, (bool, np.bool_)) for value in frame["cath_overlap_flag"]):
            raise ValueError("C7 CATH flags must be measured booleans")
        if bool(frame["cath_overlap_flag"].any()):
            raise ValueError("C7 eligible table contains CATH-overlap rows")
    if frame["selection_unit_id"].isna().any() or not frame["selection_unit_id"].is_unique:
        raise ValueError("eligible table requires unique selection_unit_id")
    if frame[value_col].isna().any():
        raise ValueError(f"eligible table has null {value_col}")
    values = frame[value_col].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("eligible table is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"eligible table has non-finite {value_col}")
    if target > len(frame):
        raise ValueError(f"target {target} exceeds eligible capacity {len(frame)}")

    lo = float(values.min())
    hi = float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    if hi == lo:
        bin_indices = np.zeros(len(frame), dtype=int)
    else:
        bin_indices = np.digitize(values, edges[1:-1], right=False).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    available = np.array([(bin_indices == i).sum() for i in range(n_bins)], dtype=int)
    floors = np.where(
        available > 0, np.minimum(available, min_per_populated_bin), 0
    ).astype(int)
    floor_sum = int(floors.sum())
    if floor_sum > target:
        raise ValueError(f"floor sum {floor_sum} exceeds target {target}")
    if target > int(available.sum()):
        raise ValueError(f"target {target} exceeds eligible capacity {available.sum()}")

    mu_value: float | None = None
    sigma_value: float | None = None
    if mode == "uniform":
        weights = np.ones(n_bins, dtype=float)
    else:
        mu_value = float(np.median(values)) if mu is None else float(mu)
        if sigma is None:
            d = max(mu_value - float(centers.min()), float(centers.max()) - mu_value)
            sigma_value = d / math.sqrt(2.0 * math.log(3.0)) if d > 0 else None
        else:
            sigma_value = float(sigma)
            if sigma_value <= 0:
                raise ValueError("sigma must be positive")
        if sigma_value is None:
            weights = np.ones(n_bins, dtype=float)
        else:
            weights = np.exp(-0.5 * ((centers - mu_value) / sigma_value) ** 2)
    weights = np.where(available > 0, weights, 0.0)

    allocated = floors.copy()
    while int(allocated.sum()) < target:
        score = np.full(n_bins, np.inf, dtype=float)
        candidates = allocated < available
        score[candidates] = allocated[candidates] / weights[candidates]
        chosen_bin = int(np.argmin(score))
        if not math.isfinite(float(score[chosen_bin])):
            raise ValueError("allocation exhausted before target")
        allocated[chosen_bin] += 1

    ordered = frame[["selection_unit_id", value_col]].copy()
    ordered["selection_unit_id"] = ordered["selection_unit_id"].astype(str)
    ordered["_bin"] = bin_indices
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    histogram: list[dict[str, Any]] = []
    for bin_idx in range(n_bins):
        members = sorted(
            ordered.loc[ordered["_bin"] == bin_idx, "selection_unit_id"].tolist()
        )
        take = int(allocated[bin_idx])
        if take == len(members):
            chosen = members
        elif take == 0:
            chosen = []
        else:
            chosen = sorted(str(x) for x in rng.choice(members, size=take, replace=False))
        selected.extend(chosen)
        histogram.append(
            {
                "bin": bin_idx,
                "bin_lo": float(edges[bin_idx]),
                "bin_hi": float(edges[bin_idx + 1]),
                "bin_center": float(centers[bin_idx]),
                "weight": float(weights[bin_idx]),
                "n_available": len(members),
                "floor": int(floors[bin_idx]),
                "n_selected": take,
            }
        )
    if len(selected) != target or len(selected) != len(set(selected)):
        raise AssertionError("sampling did not produce the exact unique target")
    return BinnedSampleResult(
        selected_ids=tuple(selected), histogram=tuple(histogram), value_col=value_col,
        target=target, n_bins=n_bins, mode=mode, stage=stage,
        identity_columns=identity_columns,
        min_per_populated_bin=min_per_populated_bin, seed=seed,
        mu=mu_value, sigma=sigma_value,
        eligible_digest=_input_digest(frame, value_col, identity_columns),
        numpy_version=str(np.__version__),
        bit_generator=rng.bit_generator.__class__.__name__,
    )


def replay_binned_sample(
    frame: pd.DataFrame, recorded: BinnedSampleResult
) -> BinnedSampleResult:
    """Recompute membership and reject a changed ordered eligible table."""

    observed_digest = _input_digest(frame, recorded.value_col, recorded.identity_columns)
    if observed_digest != recorded.eligible_digest:
        raise ValueError("eligible table digest changed; replay is not authorized")
    if recorded.numpy_version != str(np.__version__):
        raise ValueError("NumPy identity changed; replay is not authorized")
    if recorded.bit_generator != np.random.default_rng(recorded.seed).bit_generator.__class__.__name__:
        raise ValueError("NumPy bit-generator identity changed; replay is not authorized")
    return deterministic_binned_sample(
        frame,
        value_col=recorded.value_col,
        target=recorded.target,
        n_bins=recorded.n_bins,
        mode=recorded.mode,
        stage=recorded.stage,
        min_per_populated_bin=recorded.min_per_populated_bin,
        seed=recorded.seed,
        mu=recorded.mu,
        sigma=recorded.sigma,
    )
