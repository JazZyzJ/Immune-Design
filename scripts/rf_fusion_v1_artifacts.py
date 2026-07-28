"""V1F5 entry-artifact writers (PLAN_RF_REFINE_FUSION_V1 §5).

IO layer for the V1 entry artifact family. Lives in ``scripts`` (not the pure ``fusion`` package)
so it may import pandas; the pure package stays pandas/torch-free. Every table has a frozen schema
(stable columns even when empty), an explicit column order, canonical row ordering on write, and a
hard error on any undeclared column. Large root state is written as a content-addressed sidecar
whose integrity is verified before reuse (§2.4/§5.3). ``continuations`` never deduplicates distinct
logical seeds.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from inverse_folding.reference_flow.fusion.v1_admission import AdmissionResult, CoverageRow
from inverse_folding.reference_flow.fusion.v1_alloc import (
    FacadeRow,
    RootValue,
    ScoredContinuation,
)
from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent
from inverse_folding.reference_flow.fusion.v1_records import (
    PartialRootPayload,
    make_root_id,
    verify_payload_integrity,
)


# --------------------------------------------------------------------------- #
# frozen schema registry (§5.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableSchema:
    columns: tuple[str, ...]
    sort_by: tuple[str, ...]


def _schema(columns, sort_by) -> TableSchema:
    return TableSchema(columns=tuple(columns), sort_by=tuple(sort_by))


#: V1-A conditioning identity. There is no h-map key: the null entry runtime consumes none, and
#: its absence is asserted by the two sentinels rather than hashed as a placeholder.
_CONDITIONING_KEYS = (
    "dplm_checkpoint", "tokenizer", "backbone_row", "coordinate_mask", "entry_config",
    "fixed_token_policy", "controller_enabled", "h_maps_present",
)

V1_TABLE_SCHEMAS: dict[str, TableSchema] = {
    "root_attempts": _schema(
        ["protein_id", "arm_id", "rho_id", "attempt_index", "root_id", "seed", "paid_prefix_dfe",
         "crossing_status", "payload_status", "equivalence_status", "root_equivalence_hash",
         "snapshot_payload_hash", "failure_reason", "duplicate_of_root_id"],
        ["protein_id", "arm_id", "rho_id", "attempt_index"],
    ),
    "partial_roots": _schema(
        ["schema_version", "root_id", "protein_id", "arm_id", "rho_id", "seed", "phase", "step",
         "n_steps", "t", "target_maturity", "actual_maturity", "n_unresolved_editable",
         "paid_prefix_dfe", "root_equivalence_hash", "snapshot_payload_hash", "payload_path",
         *(f"cond_{k}" for k in _CONDITIONING_KEYS)],
        ["protein_id", "arm_id", "rho_id", "root_id"],
    ),
    "maturity_telemetry": _schema(
        ["protein_id", "root_equivalence_hash", "length", "n_fixed", "n_editable",
         "n_resolved_editable", "rho_edit", "known_identity_fraction", "coordinate_valid_fraction",
         "unresolved_head_relevant_mass", "anchor_preservation"],
        ["protein_id", "root_equivalence_hash"],
    ),
        "continuations": _schema(
        ["protein_id", "continuation_id", "root_equivalence_hash", "set_tag", "seed",
         "replicate_index", "sequence", "sequence_md5", "global_risk", "dfe", "status",
         "convergence_multiplicity"],
        ["protein_id", "root_equivalence_hash", "set_tag", "replicate_index", "continuation_id"],
    ),
    "root_values": _schema(
        ["protein_id", "root_equivalence_hash", "rank", "expected_k", "observed_k", "mean",
         "value", "valid", "equivalence_group", "selected"],
        ["protein_id", "rank"],
    ),
    "root_selection": _schema(
        ["protein_id", "root_equivalence_hash", "policy", "control", "rank", "selected",
         "tie_break", "source_value", "fresh_materialization_authorized"],
        ["protein_id", "policy", "rank"],
    ),
    # ``rank`` is NULL for a trajectory that was generated (and paid for) but ranked below F_cap,
    # so the canonical sort uses source_id -- a total, always-present key.
    # ``collapsed_into_source_id`` says WHY ``rank`` is null: non-null => the trajectory ranked
    # INSIDE F_cap but converged onto another row's sequence; null => it genuinely ranked below the
    # cap. Without that column the two are indistinguishable and a top-ranked trajectory reads as
    # "generated but not submitted".
    "complete_entry_pool": _schema(
        ["protein_id", "arm_id", "source_id", "seed", "sequence", "sequence_md5",
         "terminal_global_risk", "rank", "dfe", "collapsed_into_source_id"],
        ["protein_id", "arm_id", "source_id"],
    ),
    # Every facade row dropped by sequence convergence, with the representative that absorbed it.
    # PLAN §2.11:489-501 requires an unbroken root -> continuation -> facade -> admission chain and
    # §5.3:828 makes a missing edge invalidate the run: without this table a collapsed row reads
    # `selected=True, fresh_materialization_authorized=True` in `root_selection` and then simply has
    # no facade row and no recorded reason anywhere.
    "facade_collapse": _schema(
        ["protein_id", "arm_id", "design_idx", "source_id", "root_equivalence_hash",
         "sequence_md5", "representative_source_id", "representative_design_idx"],
        ["protein_id", "arm_id", "source_id"],
    ),
    "t0_control_membership": _schema(
        ["protein_id", "endpoint_id", "policy", "selected_partial_member",
         "random_partial_member", "independent_full_member", "common_eval_provenance",
         "subset_key"],
        ["protein_id", "policy", "endpoint_id"],
    ),
    "t0_structure_subset": _schema(
        ["protein_id", "policy", "subset_rank", "source_endpoint_id", "request_id",
         "head_independent_sampling"],
        ["protein_id", "policy", "subset_rank"],
    ),
    "t0_structure_results": _schema(
        ["protein_id", "policy", "request_id", "source_endpoint_id", "target_backbone_metric",
         "gate_pass", "request_status", "cache_status", "model_status", "failure_reason"],
        ["protein_id", "policy", "request_id"],
    ),
    "terminal_parent_facade": _schema(
        ["protein_id", "arm_id", "design_idx", "entry_rank", "seed", "sequence", "sequence_md5",
         "source_id", "root_equivalence_hash", "continuation_id", "config_hash"],
        ["protein_id", "arm_id", "design_idx"],
    ),
    "terminal_parent_admission": _schema(
        ["protein_id", "arm_id", "design_idx", "attempt_order", "source_id", "continuation_id",
         "root_equivalence_hash", "sequence_md5", "structure_feasible", "verdict", "slot_idx",
         "particle_id", "cache_status", "model_executed", "walltime_s", "failure_reason"],
        ["protein_id", "arm_id", "attempt_order"],
    ),
    "cohort_coverage": _schema(
        ["protein_id", "arm_id", "phase", "frozen_split_id", "requested", "preflight_status",
         "root_status", "entry_status", "terminal_status", "failure_stage", "status"],
        ["protein_id", "arm_id"],
    ),
}

# Parquet entry tables that must always exist with their full schema (§5.1). ``snapshot payloads``
# are content-addressed sidecars, and ``cost_ledger.jsonl`` / ``manifest.json`` are not parquet.
REQUIRED_ENTRY_TABLES: tuple[str, ...] = tuple(V1_TABLE_SCHEMAS)


# --------------------------------------------------------------------------- #
# stable-schema parquet writer
# --------------------------------------------------------------------------- #
# Column dtypes are resolved by name so empty and populated tables share ONE stable Arrow schema
# (an empty parquet is not all-null-typed) and nullable ints stay ints rather than NaN floats.
_UINT_COLS = frozenset({"seed"})
_INT_COLS = frozenset({
    "attempt_index", "paid_prefix_dfe", "step", "n_steps", "n_unresolved_editable", "length",
    "n_fixed", "n_editable", "n_resolved_editable", "n_unresolved", "window_start", "window_end",
    "replicate_index", "dfe", "convergence_multiplicity", "rank", "expected_k", "observed_k",
    "subset_rank", "design_idx", "entry_rank", "attempt_order", "slot_idx",
    "representative_design_idx",
})
_FLOAT_COLS = frozenset({
    "t", "target_maturity", "actual_maturity", "rho_edit", "known_identity_fraction",
    "coordinate_valid_fraction", "unresolved_head_relevant_mass", "anchor_preservation",
    "window_risk", "global_risk", "mean", "value", "source_value", "terminal_global_risk",
    "target_backbone_metric", "walltime_s",
})
_BOOL_COLS = frozenset({
    "cond_controller_enabled", "cond_h_maps_present",
    "valid", "selected", "structure_feasible", "requested", "gate_pass", "model_executed",
    "fresh_materialization_authorized", "head_independent_sampling",
    "selected_partial_member", "random_partial_member", "independent_full_member", "subset_key",
})


def _arrow_type(column: str):
    if column in _UINT_COLS:
        return pa.uint64()  # seeds span the full uint64 range and would overflow int64
    if column in _INT_COLS:
        return pa.int64()
    if column in _FLOAT_COLS:
        return pa.float64()
    if column in _BOOL_COLS:
        return pa.bool_()
    return pa.string()


def write_stable_parquet(
    path, rows: Sequence[Mapping], *, columns: Sequence[str], sort_by: Sequence[str]
) -> None:
    """Write ``rows`` as parquet with the frozen ``columns`` (present even when empty, in declared
    order) under an EXPLICIT per-column Arrow schema, sorted by ``sort_by`` with a stable sort. The
    schema is identical whether or not there are rows, so a downstream reader never sees an
    all-null empty table drift into typed columns. An undeclared column, or a ``sort_by`` key not
    in ``columns``, is a hard error rather than a silent drop."""
    columns = list(columns)
    col_set = set(columns)
    missing_sort = [k for k in sort_by if k not in col_set]
    if missing_sort:
        raise ValueError(f"sort_by keys not in columns: {missing_sort}")
    for row in rows:
        extra = set(row) - col_set
        if extra:
            raise ValueError(f"undeclared columns in row: {sorted(extra)}")

    norm = [{c: row.get(c) for c in columns} for row in rows]
    if sort_by and norm:
        norm.sort(key=lambda r: tuple(r[k] for k in sort_by))  # join/order keys are non-null
    schema = pa.schema([(c, _arrow_type(c)) for c in columns])
    table = pa.Table.from_arrays(
        [pa.array([r[c] for r in norm], type=schema.field(c).type) for c in columns],
        schema=schema,
    )
    pq.write_table(table, str(Path(path)))


# --------------------------------------------------------------------------- #
# content-addressed snapshot payload sidecar (§2.4 / §5.3)
# --------------------------------------------------------------------------- #
def write_payload_sidecar(base_dir, payload: PartialRootPayload) -> tuple[str, str]:
    """Write ``payload`` as a content-addressed JSON sidecar. The relative path carries the
    integrity hash (``snapshot_payload_hash``); returns ``(relative_path, integrity_hash)``."""
    digest = payload.snapshot_payload_hash
    rel = f"payloads/{digest}.json"
    dest = Path(base_dir) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(dest, payload.to_dict())
    return rel, digest


def read_payload_sidecar(base_dir, rel_path: str, *, expected_hash: str) -> PartialRootPayload:
    """Load a payload sidecar and fail-fast unless its recomputed integrity hash matches
    ``expected_hash`` (a tampered or wrong-hash sidecar never enters replay)."""
    data = json.loads((Path(base_dir) / rel_path).read_text())
    payload = PartialRootPayload.from_dict(data)
    verify_payload_integrity(payload, expected_hash)
    return payload


# --------------------------------------------------------------------------- #
# cost ledger + manifest
# --------------------------------------------------------------------------- #
def write_cost_ledger_jsonl(path, events: Sequence[LedgerEvent]) -> None:
    """One JSON line per logical/physical ledger event, in emission order (append-safe §3.3)."""
    with open(Path(path), "w") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def read_cost_ledger_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_manifest(path, manifest: Mapping) -> None:
    _write_json_atomic(Path(path), dict(manifest))


# --------------------------------------------------------------------------- #
# row builders — replay-graph edges (§5.3)
# --------------------------------------------------------------------------- #
def partial_root_row(
    payload: PartialRootPayload, *, payload_path: str, target_maturity: float,
    actual_maturity: float, seed: int | None = None,
) -> dict:
    """A ``partial_roots`` row: root/equivalence/payload identity, seed, phase/step, target vs
    actual maturity, the content-addressed payload path, and all six conditioning digests."""
    row = {
        "schema_version": payload.schema_version, "root_id": payload.root_id,
        "protein_id": payload.protein_id, "arm_id": payload.arm_id, "rho_id": payload.rho_id,
        "seed": seed, "phase": payload.snapshot_phase, "step": payload.step,
        "n_steps": payload.n_steps, "t": payload.t, "target_maturity": target_maturity,
        "actual_maturity": actual_maturity, "n_unresolved_editable": payload.n_unresolved_editable,
        "paid_prefix_dfe": payload.paid_prefix_dfe,
        "root_equivalence_hash": payload.root_equivalence_hash,
        "snapshot_payload_hash": payload.snapshot_payload_hash, "payload_path": payload_path,
    }
    for key in _CONDITIONING_KEYS:
        row[f"cond_{key}"] = getattr(payload.conditioning, key)
    return row


def continuation_rows(
    scored: Sequence[ScoredContinuation], *, protein_id: str | None = None,
    dfe_by_id: Mapping[str, int] | None = None,
    convergence_multiplicity: Mapping[str, int] | None = None,
) -> list[dict]:
    """One row per LOGICAL continuation (never deduplicated by sequence): distinct seeds keep
    distinct rows even when the terminal sequence collides (§5.1). ``dfe_by_id`` supplies each
    continuation's tail DFE (est and final continuations are both recorded)."""
    mult = convergence_multiplicity or {}
    dfe = dfe_by_id or {}
    rows = []
    for sc in scored:
        cont = sc.continuation
        rows.append({
            "protein_id": protein_id, "continuation_id": cont.continuation_id,
            "root_equivalence_hash": cont.root_equivalence_hash, "set_tag": cont.set_tag,
            "seed": cont.seed, "replicate_index": cont.replicate_index, "sequence": cont.sequence,
            "sequence_md5": sc.sequence_md5, "global_risk": sc.global_risk,
            "dfe": dfe.get(cont.continuation_id), "status": "ok",
            "convergence_multiplicity": mult.get(cont.sequence),
        })
    return rows


def root_value_rows(
    root_values: Sequence[RootValue], *, expected_k: int, protein_id: str | None = None,
    selected: Sequence[str] | None = None,
) -> list[dict]:
    """One ``root_values`` row per root; rank is the deterministic ``(value, hash)`` order (never
    the collapse/root-id order), and ``valid`` requires the exact expected replicate count (a short
    estimator set is surfaced, not silently averaged)."""
    selected_set = set(selected or ())
    ranked = sorted(root_values, key=lambda rv: (rv.value, rv.root_equivalence_hash))
    rows = []
    for rank, rv in enumerate(ranked):
        rows.append({
            "protein_id": protein_id, "root_equivalence_hash": rv.root_equivalence_hash,
            "rank": rank, "expected_k": expected_k, "observed_k": rv.n_samples, "mean": rv.value,
            "value": rv.value, "valid": rv.n_samples == expected_k,
            "equivalence_group": rv.root_equivalence_hash,
            "selected": rv.root_equivalence_hash in selected_set,
        })
    return rows


def facade_rows(
    facade: Sequence[FacadeRow], *, protein_id: str | None = None, arm_id: str | None = None,
    config_hash: str | None = None, continuation_id_by_hash: Mapping[str, str] | None = None,
) -> list[dict]:
    """One ``terminal_parent_facade`` row per ordered complete attempt; ``entry_rank`` mirrors the
    facade ``design_idx`` (root-value rank for the pre-terminal arm, terminal-Head rank for the
    terminal arm). The
    ``continuation_id`` join is carried explicitly, never reconstructed from the sequence (§2.11)."""
    cid = continuation_id_by_hash or {}
    return [
        {
            "protein_id": protein_id, "arm_id": arm_id, "design_idx": f.design_idx,
            "entry_rank": f.design_idx, "seed": f.seed, "sequence": f.sequence,
            "sequence_md5": f.sequence_md5, "source_id": f.source_id,
            "root_equivalence_hash": f.root_equivalence_hash,
            "continuation_id": cid.get(f.root_equivalence_hash), "config_hash": config_hash,
        }
        for f in facade
    ]


def root_attempt_rows(
    attempts: Sequence, *, representative_root_id_by_hash: Mapping[str, str], protein_id: str,
    arm_id: str, rho_id: str,
) -> list[dict]:
    """One ``root_attempts`` row per FROZEN prefix attempt (§2.3): a no-crossing/invalid attempt
    and an equivalence duplicate stay visible with their charged prefix DFE, never hidden."""
    rows = []
    for outcome in attempts:
        payload = outcome.payload
        if payload is None:
            rows.append({
                "protein_id": protein_id, "arm_id": arm_id, "rho_id": rho_id,
                "attempt_index": outcome.attempt_index,
                "root_id": make_root_id(protein_id, arm_id, rho_id, outcome.attempt_index),
                "seed": outcome.seed, "paid_prefix_dfe": outcome.paid_prefix_dfe,
                "crossing_status": outcome.status, "payload_status": "none",
                "equivalence_status": "none", "root_equivalence_hash": None,
                "snapshot_payload_hash": None, "failure_reason": outcome.failure_reason,
                "duplicate_of_root_id": None,
            })
            continue
        h = payload.root_equivalence_hash
        representative = representative_root_id_by_hash.get(h)
        is_duplicate = representative is not None and payload.root_id != representative
        rows.append({
            "protein_id": protein_id, "arm_id": arm_id, "rho_id": rho_id,
            "attempt_index": outcome.attempt_index, "root_id": payload.root_id,
            "seed": outcome.seed, "paid_prefix_dfe": outcome.paid_prefix_dfe,
            "crossing_status": "crossed", "payload_status": "valid",
            "equivalence_status": "duplicate" if is_duplicate else "unique",
            "root_equivalence_hash": h, "snapshot_payload_hash": payload.snapshot_payload_hash,
            "failure_reason": None,
            "duplicate_of_root_id": representative if is_duplicate else None,
        })
    return rows


def root_selection_rows(records: Sequence, *, protein_id: str | None = None) -> list[dict]:
    """One ``root_selection`` row per root: the deterministic value rank and whether the beam
    selected it and authorized a fresh final materialization."""
    return [
        {
            "protein_id": protein_id, "root_equivalence_hash": r.root_equivalence_hash,
            "policy": r.policy, "control": None, "rank": r.rank, "selected": r.selected,
            "tie_break": None, "source_value": r.source_value,
            "fresh_materialization_authorized": r.fresh_materialization_authorized,
        }
        for r in records
    ]


def maturity_rows(records: Sequence, *, protein_id: str | None = None) -> list[dict]:
    """One ``maturity_telemetry`` row per unique root, from payload-derived counts; Head/structure
    columns are left null for the driver to fill."""
    return [
        {
            "protein_id": protein_id, "root_equivalence_hash": r.root_equivalence_hash,
            "length": r.length, "n_fixed": r.n_fixed, "n_editable": r.n_editable,
            "n_resolved_editable": r.n_resolved_editable, "rho_edit": r.rho_edit,
            "known_identity_fraction": r.known_identity_fraction,
            "coordinate_valid_fraction": None, "unresolved_head_relevant_mass": None,
            "anchor_preservation": None,
        }
        for r in records
    ]


def admission_rows(
    result: AdmissionResult, *, protein_id: str | None = None, arm_id: str | None = None,
) -> list[dict]:
    """One ``terminal_parent_admission`` row per facade attempt: rank, structure verdict, and the
    admitted slot or the visible failure reason (§2.11)."""
    return [
        {
            "protein_id": protein_id, "arm_id": arm_id, "design_idx": a.design_idx,
            "attempt_order": a.attempt_order, "source_id": a.source_id,
            "continuation_id": a.continuation_id, "root_equivalence_hash": a.root_equivalence_hash,
            "sequence_md5": a.sequence_md5, "structure_feasible": a.structure_feasible,
            "verdict": a.verdict, "slot_idx": a.slot_idx, "particle_id": a.particle_id,
            "cache_status": a.cache_status, "model_executed": a.model_executed,
            "walltime_s": a.walltime_s,
            "failure_reason": a.failure_reason or (None if a.verdict == "admitted" else a.verdict),
        }
        for a in result.attempts
    ]


def coverage_rows(
    coverage: Sequence[CoverageRow], *, arm_id: str | None = None, phase: str | None = None,
    frozen_split_id: str | None = None,
) -> list[dict]:
    """One ``cohort_coverage`` row per requested protein; a never-processed protein carries
    ``status == "missing"`` from :func:`build_cohort_coverage` (never silently dropped)."""
    return [
        {
            "protein_id": row.protein_id, "arm_id": arm_id, "phase": phase,
            "frozen_split_id": frozen_split_id, "requested": True, "preflight_status": None,
            "root_status": None, "entry_status": None, "terminal_status": None,
            "failure_stage": None, "status": row.status,
        }
        for row in coverage
    ]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)
