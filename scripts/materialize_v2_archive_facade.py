#!/usr/bin/env python3
"""Materialize an exact V2 archive selection as a Phase-C/v0 ``generated.parquet``.

The archive table is a membership view, not sequence evidence.  This exporter therefore joins it
to ``complete_endpoints.parquet`` on the persisted ``endpoint_id`` and validates the redundant
content/protein/feasibility identities before selecting anything.  It never reconstructs lineage
from sequence bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402
from inverse_folding.reference_flow.fusion.v1_alloc import (  # noqa: E402
    HeadInputError,
    validate_complete_aa20,
)
from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = ["V2FacadeError", "materialize_archive_facade", "main"]


class V2FacadeError(V2Error):
    """The named evidence bundles cannot support an exact facade selection."""


_ARCHIVE_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "sequence_equivalence_key",
    "feasibility_level",
    "is_elite",
    "elite_rank",
}
_ENDPOINT_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "depth",
    "sequence",
    "sequence_md5",
    "sequence_equivalence_key",
    "sequence_length",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "head_global_risk",
    "feasibility_level",
    "structure_evaluated",
    "structure_feasible",
}
_OUTPUT_COLUMNS = (
    "protein_id",
    "design_idx",
    "sequence",
    "entry_source_id",
    "endpoint_id",
    "endpoint_content_digest",
    "depth",
    "head_global_risk",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "sequence_md5",
)

# These fields define the one exploratory experiment whose per-protein bundles may be merged into
# a cohort facade.  Protein-bound content (backbone, constraints, band cell, safety reference and
# the resolved config digest) is deliberately absent: requiring those to match would make every
# legitimate multi-protein cohort unmergeable.
_COMMON_MANIFEST_FIELDS = (
    "campaign_id",
    "phase",
    "split_role",
    "code_revision",
    "schedule_id",
    "coordinate_law",
    "depth_cap",
    "exploratory_depth_override",
    "production_depth_authorized",
    "structure_backend",
    "v0_structure_gate_config",
    "structure_config",
)
_REQUIRED_STRUCTURE_CONTENT = ("structure_backend", "v0_structure_gate_config")


def _bundle_root(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file() and path.name in {"archive.parquet", "complete_endpoints.parquet"}:
        path = path.parent
    if not path.is_dir():
        raise V2FacadeError(f"V2 bundle is not a directory: {path}")
    return path


def _required_manifest_text(manifest: Mapping[str, Any], field: str, *, path: Path) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise V2FacadeError(f"{path} requires non-empty string {field}")
    return value


def _load_manifest_identity(root: Path) -> dict[str, Any]:
    """Read one bundle's run identity before any endpoint tables are merged.

    The exporter is intentionally specific to the unblinded uricase capability ladder.  A bundle
    from qualification, production authorization, or another exploratory split is valid evidence
    for its own question, but it is not a cell in this cohort and must not be made one by a parquet
    concatenation.
    """
    path = root / "run_manifest.json"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing run_manifest.json: {root}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2FacadeError(f"cannot read valid JSON run_manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise V2FacadeError(f"{path} must contain a JSON object")

    phase = _required_manifest_text(manifest, "phase", path=path)
    if phase != "capability_ladder":
        raise V2FacadeError(
            f"{path} phase must be 'capability_ladder', got {phase!r}"
        )
    split_role = _required_manifest_text(manifest, "split_role", path=path)
    if split_role != "exploratory_uricase":
        raise V2FacadeError(
            f"{path} split_role must be 'exploratory_uricase', got {split_role!r}"
        )
    if manifest.get("exploratory_depth_override") is not True:
        raise V2FacadeError(
            f"{path} requires exploratory_depth_override=true; an ordinary or production D>1 "
            "run cannot enter this exploratory facade"
        )
    if manifest.get("production_depth_authorized") is not False:
        raise V2FacadeError(
            f"{path} requires production_depth_authorized=false; exploratory and production "
            "evidence are non-interchangeable"
        )

    depth_cap = manifest.get("depth_cap")
    if isinstance(depth_cap, bool) or not isinstance(depth_cap, int) or depth_cap < 2:
        raise V2FacadeError(f"{path} requires integer depth_cap >= 2, got {depth_cap!r}")

    content = manifest.get("content_identities")
    if not isinstance(content, Mapping):
        raise V2FacadeError(f"{path} requires object content_identities")
    structure: dict[str, str | None] = {}
    for role in _REQUIRED_STRUCTURE_CONTENT:
        value = content.get(role)
        if not isinstance(value, str) or not value.strip():
            raise V2FacadeError(
                f"{path} requires content identity {role!r}; the common structure instrument "
                "cannot be inferred from endpoint rows"
            )
        structure[role] = value
    optional_structure = content.get("structure_config")
    if optional_structure is not None and (
        not isinstance(optional_structure, str) or not optional_structure.strip()
    ):
        raise V2FacadeError(
            f"{path} content identity 'structure_config' must be a non-empty string when present"
        )
    structure["structure_config"] = optional_structure

    return {
        "campaign_id": _required_manifest_text(manifest, "campaign_id", path=path),
        "phase": phase,
        "split_role": split_role,
        "code_revision": _required_manifest_text(manifest, "code_revision", path=path),
        "schedule_id": _required_manifest_text(manifest, "schedule_id", path=path),
        "coordinate_law": _required_manifest_text(manifest, "coordinate_law", path=path),
        "depth_cap": depth_cap,
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        **structure,
    }


def _validate_common_manifest_identity(roots: Sequence[Path]) -> None:
    identities = [(root, _load_manifest_identity(root)) for root in roots]
    baseline_root, baseline = identities[0]
    for root, identity in identities[1:]:
        for field in _COMMON_MANIFEST_FIELDS:
            if identity[field] != baseline[field]:
                raise V2FacadeError(
                    f"V2 bundle manifest identity mismatch for {field}: "
                    f"{baseline_root} has {baseline[field]!r}, {root} has {identity[field]!r}"
                )


def _require_columns(frame: pd.DataFrame, required: set[str], *, table: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V2FacadeError(f"{table} is missing required columns {missing}")


def _require_unique_endpoint_ids(frame: pd.DataFrame, *, table: Path) -> None:
    if frame["endpoint_id"].isna().any():
        raise V2FacadeError(f"{table} contains null endpoint_id")
    if frame["endpoint_id"].astype(str).str.len().eq(0).any():
        raise V2FacadeError(f"{table} contains empty endpoint_id")
    duplicate = frame.loc[frame["endpoint_id"].duplicated(keep=False), "endpoint_id"]
    if not duplicate.empty:
        raise V2FacadeError(
            f"{table} contains duplicate endpoint_id values {sorted(set(map(str, duplicate)))[:3]}"
        )


def _strict_bool(value: object, *, field: str, endpoint_id: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V2FacadeError(
            f"endpoint {endpoint_id} has non-boolean {field}={value!r}; no truthy coercion allowed"
        )
    return bool(value)


def _load_joined_bundle(root: Path) -> pd.DataFrame:
    archive_path = root / "archive.parquet"
    endpoint_path = root / "complete_endpoints.parquet"
    for path in (archive_path, endpoint_path):
        if not path.is_file():
            raise V2FacadeError(f"V2 bundle is missing {path.name}: {root}")
    archive = pd.read_parquet(archive_path)
    endpoints = pd.read_parquet(endpoint_path)
    _require_columns(archive, _ARCHIVE_REQUIRED, table=archive_path)
    _require_columns(endpoints, _ENDPOINT_REQUIRED, table=endpoint_path)
    _require_unique_endpoint_ids(archive, table=archive_path)
    _require_unique_endpoint_ids(endpoints, table=endpoint_path)

    joined = archive.merge(
        endpoints,
        on="endpoint_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_archive", "_endpoint"),
        indicator=True,
    )
    unmatched = joined[joined["_merge"] != "both"]
    if not unmatched.empty:
        examples = sorted(map(str, unmatched["endpoint_id"]))[:3]
        side = sorted(set(map(str, unmatched["_merge"])))
        raise V2FacadeError(
            f"bundle {root} has endpoint ids with no complete endpoint/archive peer: "
            f"merge_status={side} examples={examples}"
        )
    joined = joined.drop(columns=["_merge"])

    for field in (
        "endpoint_content_digest",
        "protein_id",
        "sequence_equivalence_key",
        "feasibility_level",
    ):
        left = joined[f"{field}_archive"].astype("string")
        right = joined[f"{field}_endpoint"].astype("string")
        mismatch = left.isna() | right.isna() | left.ne(right)
        if mismatch.any():
            ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has {field} identity mismatch for endpoint(s) {ids}"
            )

    joined["source_bundle"] = str(root)
    return joined


def _validate_endpoint_row(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    definitive = (
        str(row["feasibility_level_archive"]) == "definitive"
        and str(row["feasibility_level_endpoint"]) == "definitive"
    )
    evaluated = _strict_bool(
        row["structure_evaluated"], field="structure_evaluated", endpoint_id=endpoint_id,
    )
    feasible = _strict_bool(
        row["structure_feasible"], field="structure_feasible", endpoint_id=endpoint_id,
    )
    if not (definitive and evaluated and feasible):
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not definitive-feasible: "
            f"archive_level={row['feasibility_level_archive']!r} "
            f"endpoint_level={row['feasibility_level_endpoint']!r} "
            f"evaluated={evaluated} feasible={feasible}"
        )

    sequence = row["sequence"]
    try:
        expected_length = int(row["sequence_length"])
        validate_complete_aa20(sequence, expected_length)
    except (HeadInputError, TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not complete uppercase AA20: {exc}"
        ) from exc
    observed_md5 = sequence_md5(sequence)
    if str(row["sequence_md5"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_md5 does not digest its AA20 sequence"
        )
    if str(row["sequence_equivalence_key_endpoint"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_equivalence_key does not equal sequence_md5"
        )
    try:
        risk = float(row["head_global_risk"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid Head risk") from exc
    if not math.isfinite(risk):
        raise V2FacadeError(f"endpoint {endpoint_id} has non-finite Head risk {risk!r}")
    try:
        depth_value = float(row["depth"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth") from exc
    if not math.isfinite(depth_value) or not depth_value.is_integer() or depth_value < 0:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth {row['depth']!r}")


def _candidate_rows(joined: pd.DataFrame) -> pd.DataFrame:
    eligible_mask = (
        joined["feasibility_level_archive"].eq("definitive")
        & joined["feasibility_level_endpoint"].eq("definitive")
        & joined["structure_evaluated"].eq(True)  # noqa: E712 - exact evidence value
        & joined["structure_feasible"].eq(True)  # noqa: E712 - exact evidence value
    )
    candidates = joined.loc[eligible_mask].copy()
    for _, row in candidates.iterrows():
        _validate_endpoint_row(row)
    return candidates


def materialize_archive_facade(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    k: int | None = None,
    require_k: bool = False,
) -> pd.DataFrame:
    """Return a deterministic Phase-C facade selected from one or more disjoint V2 bundles.

    ``top-k`` first collapses sequence-equivalent logical endpoints to the stable representative
    under ``(head_global_risk, endpoint_id)``.  Raw multiplicity remains intact in the source V2
    bundle, but cannot buy duplicate round-0 ancestry mass in the optional v0 suffix.
    """
    if mode not in {"elite", "top-k"}:
        raise V2FacadeError("mode must be 'elite' or 'top-k'")
    if mode == "top-k":
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise V2FacadeError("top-k mode requires integer k >= 1")
    elif k is not None or require_k:
        raise V2FacadeError("k/require_k are valid only in top-k mode")
    if not bundles:
        raise V2FacadeError("at least one V2 bundle is required")

    roots = [_bundle_root(path) for path in bundles]
    if len(set(roots)) != len(roots):
        raise V2FacadeError("the same V2 bundle was supplied more than once")
    # Establish that these are cells of ONE frozen experiment before opening their endpoint
    # tables.  Once frames are concatenated, source-bundle provenance is too late to prevent a
    # mixed campaign from looking like one cohort.
    _validate_common_manifest_identity(roots)
    joined_by_root = [(root, _load_joined_bundle(root)) for root in roots]

    protein_owner: dict[str, Path] = {}
    for root, frame in joined_by_root:
        proteins = set(map(str, frame["protein_id_archive"].dropna().unique()))
        for protein_id in proteins:
            previous = protein_owner.get(protein_id)
            if previous is not None:
                raise V2FacadeError(
                    f"protein {protein_id!r} appears in more than one bundle: {previous}, {root}"
                )
            protein_owner[protein_id] = root
    joined = pd.concat([frame for _, frame in joined_by_root], ignore_index=True)
    if joined.empty or not protein_owner:
        raise V2FacadeError("the named V2 bundles contain no archived endpoints")
    duplicated_ids = joined.loc[
        joined["endpoint_id"].duplicated(keep=False), "endpoint_id"
    ]
    if not duplicated_ids.empty:
        raise V2FacadeError(
            "endpoint_id is duplicated across V2 bundles: "
            f"{sorted(set(map(str, duplicated_ids)))[:3]}"
        )

    candidates = _candidate_rows(joined)

    selected: list[pd.DataFrame] = []
    for protein_id in sorted(protein_owner):
        all_rows = joined[joined["protein_id_archive"].astype(str) == protein_id]
        pool = candidates[candidates["protein_id_archive"].astype(str) == protein_id].copy()
        if mode == "elite":
            elite_mask = [
                _strict_bool(
                    row["is_elite"], field="is_elite", endpoint_id=str(row["endpoint_id"]),
                )
                for _, row in all_rows.iterrows()
            ]
            elite_rows = all_rows.loc[elite_mask]
            if len(elite_rows) != 1:
                raise V2FacadeError(
                    f"protein {protein_id!r} requires exactly one current archive elite; "
                    f"found {len(elite_rows)}"
                )
            elite_row = elite_rows.iloc[0]
            try:
                elite_rank = float(elite_row["elite_rank"])
            except (TypeError, ValueError) as exc:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite has invalid elite_rank"
                ) from exc
            if not math.isfinite(elite_rank) or not elite_rank.is_integer() or elite_rank != 0:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite must carry elite_rank=0"
                )
            _validate_endpoint_row(elite_row)
            pick = elite_rows.copy()
        else:
            if pool.empty:
                raise V2FacadeError(
                    f"protein {protein_id!r} has no definitive-feasible archive endpoint"
                )
            pool = pool.sort_values(
                ["head_global_risk", "endpoint_id"], kind="mergesort",
            )
            pool = pool.drop_duplicates("sequence_equivalence_key_endpoint", keep="first")
            if require_k and len(pool) < int(k):
                raise V2FacadeError(
                    f"top-k requires {k} distinct definitive-feasible endpoints for protein "
                    f"{protein_id!r}, but it has {len(pool)}"
                )
            pick = pool.head(int(k))
        pick = pick.sort_values(["head_global_risk", "endpoint_id"], kind="mergesort").copy()
        pick["design_idx"] = range(len(pick))
        selected.append(pick)

    chosen = pd.concat(selected, ignore_index=True)
    if chosen.empty:
        raise V2FacadeError("selection produced zero facade rows")
    head_domains = chosen[["head_evaluator_digest", "head_window_grid_digest"]].drop_duplicates()
    if chosen[["head_evaluator_digest", "head_window_grid_digest"]].isna().any().any():
        raise V2FacadeError("selected V2 endpoints carry null Head identity")
    if any(
        not str(value)
        for value in chosen["head_evaluator_digest"].tolist()
        + chosen["head_window_grid_digest"].tolist()
    ):
        raise V2FacadeError("selected V2 endpoints carry empty Head identity")
    if len(head_domains) != 1:
        raise V2FacadeError(
            "selected V2 endpoints do not share one frozen Head evaluator/window-grid identity"
        )
    out = pd.DataFrame({
        "protein_id": chosen["protein_id_endpoint"].astype(str),
        "design_idx": chosen["design_idx"].astype(int),
        "sequence": chosen["sequence"].astype(str),
        "entry_source_id": chosen["endpoint_id"].astype(str),
        "endpoint_id": chosen["endpoint_id"].astype(str),
        "endpoint_content_digest": chosen["endpoint_content_digest_endpoint"].astype(str),
        "depth": chosen["depth"].astype(int),
        "head_global_risk": chosen["head_global_risk"].astype(float),
        "head_evaluator_digest": chosen["head_evaluator_digest"].astype(str),
        "head_window_grid_digest": chosen["head_window_grid_digest"].astype(str),
        "sequence_md5": chosen["sequence_md5"].astype(str),
    })
    out = out.sort_values(["protein_id", "design_idx"], kind="mergesort").reset_index(drop=True)
    if out[["protein_id", "design_idx"]].duplicated().any():
        raise V2FacadeError("selection produced duplicate (protein_id, design_idx) facade keys")
    if out["entry_source_id"].duplicated().any():
        raise V2FacadeError("selection produced duplicate endpoint lineage keys")
    return out.loc[:, _OUTPUT_COLUMNS]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", action="append", required=True,
        help="V2 bundle directory (repeat for disjoint-protein bundles)",
    )
    parser.add_argument("--mode", required=True, choices=("elite", "top-k"))
    parser.add_argument("--k", type=int, help="maximum distinct endpoints per protein in top-k mode")
    parser.add_argument(
        "--require-k", action="store_true",
        help="fail if any protein has fewer than K distinct definitive-feasible endpoints",
    )
    parser.add_argument("--output", required=True, help="output generated.parquet path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace it: {output}")
    try:
        facade = materialize_archive_facade(
            args.bundle, mode=args.mode, k=args.k, require_k=bool(args.require_k),
        )
    except V2FacadeError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    facade.to_parquet(output, index=False)
    print(
        f"wrote {len(facade)} rows across {facade['protein_id'].nunique()} proteins to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
