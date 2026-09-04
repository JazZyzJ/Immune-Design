"""Path-bound, replayable evidence for IF benchmark v3 release gates.

The release validator deliberately accepts file identities rather than caller assertions.  Every
gate implemented here reopens persisted producer artifacts, verifies their bytes, and recomputes
the corresponding decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .entities import (
    build_entity_source_pool,
    build_rcsb_entity_query,
    parse_graphql_entity_page,
    parse_search_pages,
    validate_search_graphql_identity,
)
from .preflight import (
    CHAIN_ATTEMPT_COLUMNS,
    SOURCE_STAGE_COLUMNS,
    VALID_SOURCE_UNIT_COLUMNS,
    _source_stage_row,
    evaluate_sifts_mapping,
    parse_sifts_mappings,
    validate_preflight_raw_file_bindings,
)
from .leakage import validate_mmseqs_sidecar
from .nmp import (
    data_directory_file_table,
    load_nmp_shard_manifest,
    replay_native_nmp_shard,
)
from .rcsb_snapshot import GRAPHQL_QUERY, GRAPHQL_URL, SEARCH_URL
from .selection import (
    binned_sample_from_dict,
    binned_sample_to_dict,
    deterministic_binned_sample,
)
from .tier1 import finalize_tier1_pool, resolve_tier1_tier2_collisions
from .source_identity import resolve_canonical_source_units
from .uniprot_snapshot import _parse_mapping_tsv, _parse_sequence_tsv
from .sifts_residue import validate_tier1_residue_map_snapshot


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
CATH_REFERENCE_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EVIDENCE_SCHEMA = "if-benchmark-v3-release-evidence/1"
EVIDENCE_KEYS: dict[str, tuple[str, ...]] = {
    "source": ("rcsb_manifest", "tier1_manifest", "tier1_residue_manifest"),
    "ledgers": (
        "chain_attempts", "collision_replay", "final_dedup_replay",
        "tier1_candidate_pool", "tier1_internal_collision_ledger",
        "structure_downloads",
    ),
    "selection": (
        "c5_eligible", "c5_result", "c5_expansion", "c5_prior_rungs",
        "c5_nmp_shard_manifest", "c5_nmp_tool", "c5_nmp_parameters",
        "source_cath_queries", "source_cath_sidecar",
        "final_cath_queries", "final_cath_sidecar",
        "c7_eligible", "c7_result",
    ),
    "nmp": ("shard_manifest", "tool", "parameters"),
    "nmp_install": ("tool", "data_manifest", "contract_producer", "runner_producer"),
    "family": ("query_fasta", "cluster_tsv", "tool", "parameters"),
    "mmseqs": (
        "tool", "identity", "reference_producer", "search_producer", "cath_reference_manifest",
        "cath_reference_fasta", "head_reference_manifest", "head_reference_fasta",
    ),
    "head": (
        "checkpoint", "config", "run_summary", "annotation_manifest",
        "model_yaml", "model_ablation_yaml", "inference_yaml",
        "cohort_primary", "cohort_diagnostic",
    ),
    "approved_sources": ("manifest",),
    "descendants": ("inventory",),
    "publication": ("context",),
    "release_tables": (
        "primary", "diagnostic", "cath", "head", "head_homology", "load_coords",
    ),
}
IMMUTABLE_EXTERNAL_EVIDENCE = frozenset({
    ("selection", "c5_nmp_tool"), ("nmp", "tool"), ("family", "tool"),
    ("nmp_install", "tool"), ("nmp_install", "contract_producer"),
    ("nmp_install", "runner_producer"),
    ("mmseqs", "tool"), ("mmseqs", "reference_producer"),
    ("mmseqs", "search_producer"), ("head", "checkpoint"), ("head", "config"),
    ("head", "run_summary"),
})
FAILED_C5_SELECTION_KEYS = tuple(
    key for key in EVIDENCE_KEYS["selection"]
    if key not in {"c5_expansion", "c5_prior_rungs", "c7_result", "c5_nmp_tool"}
)
FAILED_C5_NMP_KEYS = tuple(key for key in EVIDENCE_KEYS["nmp"] if key != "tool")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def resolve_candidate_stage_path(raw_path: Any, *, candidate_root: Path, label: str) -> Path:
    """Resolve one persisted stage path and reject absolute or ``..`` escapes."""

    raw = Path(str(raw_path))
    if raw.is_absolute():
        raise ValueError(f"{label} must be candidate-relative")
    root = Path(candidate_root).resolve()
    resolved = (root / raw).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"{label} is missing or escapes candidate root")
    return resolved


def _relative_or_absolute(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_identity(path: Path, *, candidate_root: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative_or_absolute(path, Path(candidate_root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_release_evidence_bundle(
    path: Path,
    *,
    candidate_root: Path,
    sections: Mapping[str, Mapping[str, Path]],
    protocol_profile: str = "drb1501_production",
) -> dict[str, Any]:
    """Write the closed evidence-path schema and bind every referenced file byte."""

    path = Path(path)
    candidate_root = Path(candidate_root).resolve()
    if path.exists():
        raise FileExistsError(path)
    if set(sections) != set(EVIDENCE_KEYS):
        raise ValueError("release evidence sections do not match the v3 schema")
    if protocol_profile not in {"drb1501_production", "tiny_fixture"}:
        raise ValueError("unknown release evidence protocol profile")
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA, "protocol_profile": protocol_profile,
    }
    for section, required in EVIDENCE_KEYS.items():
        supplied = sections[section]
        if set(supplied) != set(required):
            raise ValueError(f"release evidence {section} keys do not match the v3 schema")
        payload[section] = {
            key: file_identity(Path(supplied[key]), candidate_root=candidate_root)
            for key in required
        }
    payload["bundle_sha256"] = sha256_bytes(canonical_json(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _resolve_identity(record: Mapping[str, Any], *, candidate_root: Path, label: str) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"{label} is not a typed file identity")
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else Path(candidate_root) / raw
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} size mismatch")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path.resolve()


def load_release_evidence_bundle(path: Path, *, candidate_root: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("release evidence schema mismatch")
    if set(payload) != {"schema_version", "protocol_profile", "bundle_sha256", *EVIDENCE_KEYS}:
        raise ValueError("release evidence has missing or extra sections")
    unsigned = dict(payload)
    expected = unsigned.pop("bundle_sha256", None)
    if expected != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("release evidence bundle self-hash mismatch")
    if payload.get("protocol_profile") not in {"drb1501_production", "tiny_fixture"}:
        raise ValueError("release evidence protocol profile mismatch")
    resolved: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "protocol_profile": payload["protocol_profile"],
    }
    for section, required in EVIDENCE_KEYS.items():
        records = payload[section]
        if not isinstance(records, Mapping) or set(records) != set(required):
            raise ValueError(f"release evidence {section} keys mismatch")
        for key in required:
            raw_path = Path(str(records[key].get("path") or ""))
            if raw_path.is_absolute() and (section, key) not in IMMUTABLE_EXTERNAL_EVIDENCE:
                raise ValueError(
                    f"stage output {section}.{key} must be self-contained/candidate-relative"
                )
        resolved[section] = {
            key: _resolve_identity(
                records[key], candidate_root=Path(candidate_root), label=f"{section}.{key}"
            )
            for key in required
        }
    return resolved


def write_failed_c5_attempt_bundle(
    path: Path, *, candidate_root: Path, target: int,
    ledgers: Mapping[str, Path], selection: Mapping[str, Path], nmp: Mapping[str, Path],
) -> dict[str, Any]:
    """Bind one fully materialized but capacity-short C5 rung for a later build."""

    if set(ledgers) != set(EVIDENCE_KEYS["ledgers"]):
        raise ValueError("failed C5 ledger keys mismatch")
    if set(selection) != set(FAILED_C5_SELECTION_KEYS):
        raise ValueError("failed C5 selection keys mismatch")
    if set(nmp) != set(FAILED_C5_NMP_KEYS):
        raise ValueError("failed C5 full-NMP keys mismatch")
    payload: dict[str, Any] = {
        "schema_version": "if-benchmark-v3-failed-c5-attempt/1", "target": int(target),
        "artifacts": {
            "ledgers": {key: file_identity(value, candidate_root=candidate_root)
                        for key, value in ledgers.items()},
            "selection": {key: file_identity(value, candidate_root=candidate_root)
                          for key, value in selection.items()},
            "nmp": {key: file_identity(value, candidate_root=candidate_root)
                    for key, value in nmp.items()},
        },
    }
    payload["bundle_sha256"] = sha256_bytes(canonical_json(payload))
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _load_failed_c5_attempt_bundle(
    path: Path, *, candidate_root: Path, expected_target: int
) -> dict[str, dict[str, Path]]:
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema_version") != "if-benchmark-v3-failed-c5-attempt/1"
        or int(payload.get("target", -1)) != int(expected_target)
    ):
        raise ValueError("failed C5 attempt schema/target mismatch")
    unsigned = dict(payload)
    digest = unsigned.pop("bundle_sha256", None)
    if digest != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("failed C5 attempt self-hash mismatch")
    expected = {
        "ledgers": EVIDENCE_KEYS["ledgers"],
        "selection": FAILED_C5_SELECTION_KEYS,
        "nmp": FAILED_C5_NMP_KEYS,
    }
    resolved: dict[str, dict[str, Path]] = {}
    for section, keys in expected.items():
        records = payload["artifacts"].get(section)
        if not isinstance(records, Mapping) or set(records) != set(keys):
            raise ValueError("failed C5 attempt artifact keys mismatch")
        if any(Path(str(records[key].get("path") or "")).is_absolute() for key in keys):
            raise ValueError("failed C5 attempt artifacts must be copied into the new candidate")
        resolved[section] = {
            key: _resolve_identity(
                records[key], candidate_root=candidate_root,
                label=f"failed_c5.{expected_target}.{section}.{key}",
            )
            for key in keys
        }
    return resolved


def write_c5_prior_rung_index(
    path: Path, *, candidate_root: Path, attempt_bundles: Mapping[int, Path],
    ladder: Sequence[int] = (5000, 7500, 10000),
    attempt_roots: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    """Register copied, independently replayable failed rungs in ascending ladder order."""

    ordered = sorted((int(target), Path(bundle)) for target, bundle in attempt_bundles.items())
    targets = [target for target, _bundle in ordered]
    frozen = [int(value) for value in ladder]
    if targets != frozen[:len(targets)] or len(targets) >= len(frozen):
        raise ValueError("prior C5 rung set is not a valid ladder prefix")
    roots = {} if attempt_roots is None else {
        int(target): Path(root).resolve() for target, root in attempt_roots.items()
    }
    if roots and set(roots) != set(targets):
        raise ValueError("prior C5 attempt roots do not match bundle targets")
    payload = {
        "schema_version": "if-benchmark-v3-c5-prior-rungs/1",
        "rungs": [
            {
                "target": target,
                "attempt_bundle": file_identity(bundle, candidate_root=candidate_root),
                **({
                    "attempt_root": str(roots[target].relative_to(Path(candidate_root).resolve()))
                } if roots else {}),
            }
            for target, bundle in ordered
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _verify_manifest_self_hash(manifest: Mapping[str, Any], label: str) -> None:
    unsigned = dict(manifest)
    expected = unsigned.pop("manifest_sha256", None)
    if expected != sha256_bytes(canonical_json(unsigned)):
        raise ValueError(f"{label} manifest self-hash mismatch")


def _verify_file_table(root: Path, table: Mapping[str, Any], label: str) -> None:
    for name, raw in table.items():
        record = dict(raw)
        path = (root / str(record["path"])).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"{label} path is missing or escapes root: {name}")
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"{label} file identity mismatch: {name}")


def _assert_frame_equal(observed: pd.DataFrame, expected: pd.DataFrame, *, sort: Sequence[str], label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.sort_values(list(sort), kind="stable").reset_index(drop=True),
            expected.sort_values(list(sort), kind="stable").reset_index(drop=True),
            check_dtype=False,
            check_like=True,
        )
    except AssertionError as exc:
        raise ValueError(f"{label} does not replay from source artifacts") from exc


def validate_rcsb_source_manifest(manifest_path: Path) -> None:
    """Reparse Search/GraphQL bodies and rebuild C1 eligibility plus C2 grouping."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "if-benchmark-v3-rcsb-source/1":
        raise ValueError("RCSB source manifest schema mismatch")
    _verify_manifest_self_hash(manifest, "RCSB source")
    _verify_file_table(root, manifest["output_files"], "RCSB output")
    raw_files = manifest["raw_response_files"]["files"]
    if raw_files != sorted(raw_files, key=lambda row: row["path"]):
        raise ValueError("RCSB raw file table is not sorted")
    if manifest["raw_response_files"]["merkle_sha256"] != sha256_bytes(canonical_json(raw_files)):
        raise ValueError("RCSB raw response Merkle mismatch")
    _verify_file_table(root, {row["path"]: row for row in raw_files}, "RCSB raw")

    ledger = pd.read_parquet(root / "request_ledger.parquet")
    search_rows = ledger[ledger["request_kind"] == "search"].sort_values(
        "page_or_batch", kind="stable"
    )
    expected_start = 0
    for row in search_rows.itertuples(index=False):
        query_path = root / str(row.query_path)
        query = json.loads(query_path.read_bytes())
        expected = build_rcsb_entity_query(
            page_start=expected_start,
            page_rows=int(manifest["parameters"]["search_page_rows"]),
        )
        if query != expected or str(row.url) != SEARCH_URL:
            raise ValueError("RCSB Search query/API identity does not match the frozen contract")
        response = json.loads((root / str(row.response_path)).read_bytes())
        expected_start += len(response.get("result_set") or [])
    search_pages: list[Mapping[str, Any]] = []
    graphql_entities: list[dict[str, Any]] = []
    graphql_edges: list[dict[str, Any]] = []
    for row in ledger.sort_values(["request_kind", "page_or_batch"], kind="stable").itertuples(index=False):
        query_path = (root / str(row.query_path)).resolve()
        response_path = (root / str(row.response_path)).resolve()
        if root not in query_path.parents or root not in response_path.parents:
            raise ValueError("RCSB request ledger path escapes snapshot")
        if sha256_file(query_path) != row.query_sha256 or sha256_file(response_path) != row.response_sha256:
            raise ValueError("RCSB request ledger body digest mismatch")
        if row.request_kind == "search":
            search_pages.append(json.loads(response_path.read_bytes()))
        elif row.request_kind == "graphql_entity_instances":
            entities, edges = parse_graphql_entity_page(response_path.read_bytes())
            graphql_entities.extend(entities)
            graphql_edges.extend(edges)
        else:
            raise ValueError("RCSB request ledger has unknown request_kind")
    search_ids = parse_search_pages(search_pages)
    graphql_rows = ledger[ledger["request_kind"] == "graphql_entity_instances"].sort_values(
        "page_or_batch", kind="stable"
    )
    batch_size = int(manifest["parameters"]["graphql_batch_size"])
    expected_batches = [search_ids[index:index + batch_size] for index in range(0, len(search_ids), batch_size)]
    if len(graphql_rows) != len(expected_batches):
        raise ValueError("RCSB GraphQL batch count mismatch")
    for row, expected_ids in zip(graphql_rows.itertuples(index=False), expected_batches, strict=True):
        query = json.loads((root / str(row.query_path)).read_bytes())
        if (
            str(row.url) != GRAPHQL_URL
            or query.get("query") != GRAPHQL_QUERY
            or (query.get("variables") or {}).get("ids") != expected_ids
        ):
            raise ValueError("RCSB GraphQL query/API identity does not match Search IDs")
    validate_search_graphql_identity(search_ids, graphql_entities)
    _assert_frame_equal(
        pd.read_parquet(root / "entities_all.parquet"), pd.DataFrame(graphql_entities),
        sort=["rcsb_entity_id"], label="RCSB entity table",
    )
    _assert_frame_equal(
        pd.read_parquet(root / "entity_chain_edges_all.parquet"), pd.DataFrame(graphql_edges),
        sort=["rcsb_entity_id", "label_asym_id", "auth_asym_id"], label="RCSB chain edge table",
    )
    groups, fallbacks, rejected = build_entity_source_pool(graphql_entities, graphql_edges)
    group_frame = pd.DataFrame(groups)
    if len(group_frame):
        group_frame["ordered_entity_fallbacks_json"] = group_frame.pop("ordered_entity_fallbacks").map(
            lambda value: json.dumps(value, separators=(",", ":"))
        )
    fallback_frame = pd.DataFrame(fallbacks)
    if len(fallback_frame):
        fallback_frame["entity_chain_edges_json"] = fallback_frame.pop("entity_chain_edges").map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    rejected_frame = pd.DataFrame(rejected, columns=["rcsb_entity_id", "rejection_reasons_json"])
    _assert_frame_equal(pd.read_parquet(root / "sequence_groups.parquet"), group_frame,
                        sort=["selection_unit_id"], label="RCSB sequence groups")
    _assert_frame_equal(pd.read_parquet(root / "ordered_entity_fallbacks.parquet"), fallback_frame,
                        sort=["selection_unit_id", "fallback_rank"], label="RCSB fallbacks")
    _assert_frame_equal(pd.read_parquet(root / "entities_rejected.parquet"), rejected_frame,
                        sort=["rcsb_entity_id"], label="RCSB rejection ledger")
    expected_summary = {
        "searched_entity_count": len(search_ids),
        "eligible_entity_count": len(fallbacks),
        "rejected_entity_count": len(rejected),
        "exact_sequence_group_count": len(groups),
        "entity_chain_edge_count": len(graphql_edges),
    }
    if manifest.get("summary") != expected_summary:
        raise ValueError("RCSB source summary does not replay")
    if len(search_ids) > 1000 and float(manifest["parameters"].get("api_delay_s", -1)) < 0.1:
        raise ValueError("production RCSB GraphQL snapshot lacks the frozen 0.1s API delay")


def _verify_external_file_identity(
    record: Mapping[str, Any], label: str, *, root: Path | None = None
) -> Path:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
        raise ValueError(f"{label} input file identity mismatch")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} input SHA-256 mismatch")
    return path


def validate_tier1_source_manifest(manifest_path: Path) -> None:
    """Re-evaluate all canonical units and SIFTS mappings from persisted raw bodies."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "if-benchmark-v3-tier1-preflight/2":
        raise ValueError("Tier1 source manifest schema mismatch")
    _verify_manifest_self_hash(manifest, "Tier1 source")
    _verify_file_table(root, manifest["output_files"], "Tier1 output")
    validate_preflight_raw_file_bindings(manifest, root)
    input_paths = {
        label: _verify_external_file_identity(record, f"Tier1 {label}", root=root)
        for label, record in manifest["input_files"].items()
    }
    if any(root not in path.resolve().parents for path in input_paths.values()):
        raise ValueError("Tier1 source inputs are not self-contained in the candidate snapshot")

    source_ids = [
        line.strip() for line in input_paths["test_ids"].read_text().splitlines() if line.strip()
    ]
    proteins = pd.read_parquet(input_paths["protein_samples"])
    spans_input = pd.read_parquet(input_paths["span_records"])
    raw_sequences: dict[str, str] = {}
    for source_id, group in proteins[
        proteins["protein_id"].astype(str).isin(source_ids)
        & (proteins["allele"] == manifest["allele"])
    ].groupby("protein_id", sort=False):
        values = set(group["protein_seq"].astype(str))
        if len(values) != 1:
            raise ValueError("Tier1 frozen protein input has conflicting source sequence")
        raw_sequences[str(source_id)] = values.pop()
    raw_spans: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in source_ids}
    filtered_spans = spans_input[
        spans_input["protein_id"].astype(str).isin(source_ids)
        & (spans_input["allele"] == manifest["allele"])
    ]
    for item in filtered_spans.itertuples(index=False):
        raw_spans[str(item.protein_id)].append(
            {"start_0b": int(item.start_0b), "end_0b": int(item.end_0b),
             "peptide": str(item.peptide_seq), "source": str(item.source)}
        )
    if len(source_ids) != len(set(source_ids)) or set(raw_sequences) != set(source_ids):
        raise ValueError("Tier1 frozen split/protein input identity mismatch")
    if any(not raw_spans[source_id] for source_id in source_ids):
        raise ValueError("Tier1 frozen split source lacks allele-matched spans")

    units = pd.read_parquet(root / "tier1_canonical_source_units.parquet")
    if units["source_id"].isna().any() or not units["source_id"].is_unique:
        raise ValueError("Tier1 canonical source IDs are invalid")
    unit_by_id: dict[str, dict[str, Any]] = {}
    for row in units.to_dict("records"):
        sequence = str(row["canonical_sequence"])
        if not sequence or not set(sequence) <= AA20:
            raise ValueError("Tier1 canonical source sequence is non-AA20")
        if int(row["canonical_sequence_length"]) != len(sequence):
            raise ValueError("Tier1 canonical source length mismatch")
        if row["canonical_sequence_sha256"] != sha256_bytes(sequence.encode()):
            raise ValueError("Tier1 canonical source SHA mismatch")
        spans = json.loads(str(row["spans_json"]))
        aliases = json.loads(str(row["raw_source_ids_json"]))
        if int(row["raw_source_id_count"]) != len(aliases) or int(row["span_count"]) != len(spans):
            raise ValueError("Tier1 canonical source alias/span count mismatch")
        unit_by_id[str(row["source_id"])] = {**row, "spans": spans, "aliases": aliases}

    # Canonical sequences must be present in the content-bound UniProt response snapshot.
    returned_sequences: dict[str, str] = {}
    external_mappings: dict[str, set[str]] = {}
    uniprot_raw_entries = manifest["uniprot_response_files"]["files"]
    uniprot_requests = pd.read_parquet(root / "uniprot_requests.parquet")
    referenced_uniprot_raw: set[str] = set()
    for request in uniprot_requests.to_dict("records"):
        response_path = request.get("response_path")
        if response_path:
            prefixed = f"uniprot/{response_path}"
            path = root / prefixed
            if not path.is_file() or sha256_file(path) != request.get("response_sha256"):
                raise ValueError("UniProt request ledger response digest mismatch")
            referenced_uniprot_raw.add(prefixed)
        elif not request.get("error"):
            raise ValueError("UniProt request ledger lacks response and error provenance")
    if referenced_uniprot_raw != {str(entry["path"]) for entry in uniprot_raw_entries}:
        raise ValueError("UniProt raw response table differs from request ledger")
    for entry in uniprot_raw_entries:
        body = (root / str(entry["path"])).read_bytes()
        if body.startswith(b"Entry\tSequence\tLength"):
            returned_sequences.update(_parse_sequence_tsv(body))
        elif body.startswith(b"From\tEntry"):
            mappings, _targets = _parse_mapping_tsv(body)
            for source_id, accessions in mappings.items():
                external_mappings.setdefault(source_id, set()).update(accessions)
    rebuilt_units, rebuilt_ledger = resolve_canonical_source_units(
        raw_sequences=raw_sequences,
        raw_spans=raw_spans,
        external_mappings={key: sorted(value) for key, value in external_mappings.items()},
        canonical_sequences=returned_sequences,
    )
    rebuilt_unit_rows = [
        {"source_id": source_id, "canonical_sequence": unit["canonical_sequence"],
         "canonical_sequence_length": unit["canonical_sequence_length"],
         "canonical_sequence_sha256": unit["canonical_sequence_sha256"],
         "raw_source_ids_json": json.dumps(unit["raw_source_ids"], separators=(",", ":")),
         "raw_source_id_count": unit["raw_source_id_count"], "span_count": unit["span_count"],
         "spans_json": json.dumps(unit["spans"], sort_keys=True, separators=(",", ":"))}
        for source_id, unit in sorted(rebuilt_units.items())
    ]
    rebuilt_units_frame = pd.DataFrame(rebuilt_unit_rows, columns=list(units.columns))
    _assert_frame_equal(units, rebuilt_units_frame, sort=["source_id"],
                        label="Tier1 canonical source units")
    rebuilt_ledger_frame = pd.DataFrame(rebuilt_ledger).sort_values(
        "raw_source_id", kind="stable"
    ).reset_index(drop=True)
    _assert_frame_equal(
        pd.read_parquet(root / "tier1_accession_resolution_ledger.parquet"),
        rebuilt_ledger_frame, sort=["raw_source_id"], label="Tier1 accession identity ledger",
    )
    for source_id, unit in unit_by_id.items():
        if returned_sequences.get(source_id) != unit["canonical_sequence"]:
            raise ValueError(f"Tier1 canonical sequence lacks matching UniProt raw provenance: {source_id}")

    identity = pd.read_parquet(root / "tier1_accession_resolution_ledger.parquet")
    resolved = identity[identity["resolution_status"] == "resolved"]
    for source_id, unit in unit_by_id.items():
        aliases = set(unit["aliases"])
        rows = resolved[resolved["canonical_uniprot_id"].astype(str) == source_id]
        if set(rows["raw_source_id"].astype(str)) != aliases:
            raise ValueError("Tier1 canonical alias ledger mismatch")
        if any(rows["canonical_sequence_sha256"].astype(str) != unit["canonical_sequence_sha256"]):
            raise ValueError("Tier1 alias ledger canonical sequence mismatch")

    requests = pd.read_parquet(root / "sifts_requests.parquet")
    if set(requests["source_id"].astype(str)) != set(unit_by_id):
        raise ValueError("Tier1 SIFTS request set does not match canonical units")
    expected_attempts: list[dict[str, Any]] = []
    expected_stages: list[dict[str, Any]] = []
    for request in requests.sort_values("source_id", kind="stable").to_dict("records"):
        source_id = str(request["source_id"])
        attempt_records = json.loads(str(request["attempts_json"]))
        if (
            len(attempt_records) != int(request["attempt_count"])
            or not attempt_records
            or any(not item.get("queried_at_utc") or not item.get("outcome")
                   for item in attempt_records)
            or not str(request["url"]).endswith(f"/{source_id}")
        ):
            raise ValueError("Tier1 SIFTS request retry/query-time provenance mismatch")
        response = (root / str(request["response_path"])).resolve()
        if root not in response.parents or not response.is_file():
            raise ValueError("Tier1 SIFTS response path escapes source root")
        if response.stat().st_size != int(request["response_size_bytes"]) or sha256_file(response) != request["response_sha256"]:
            raise ValueError("Tier1 SIFTS request/response identity mismatch")
        mappings: list[dict[str, Any]] = []
        if request["final_status"] == "http_200":
            mappings = parse_sifts_mappings(response.read_bytes(), source_id)
            if request["parse_status"] != "complete":
                raise ValueError("Tier1 SIFTS parse status mismatch")
        elif request["final_status"] != "http_404":
            raise ValueError("Tier1 SIFTS snapshot contains unresolved network/HTTP failure")
        if bool(request["network_failure"]) or int(request["parsed_mapping_count"]) != len(mappings):
            raise ValueError("Tier1 SIFTS request funnel mismatch")
        current: list[dict[str, Any]] = []
        unit = unit_by_id[source_id]
        for mapping in mappings:
            evaluated = evaluate_sifts_mapping(
                source_id=source_id, mapping=mapping,
                full_sequence=unit["canonical_sequence"], spans=unit["spans"],
            )
            evaluated["sifts_item_index"] = mapping["sifts_item_index"]
            evaluated["source_sequence_sha256"] = unit["canonical_sequence_sha256"]
            current.append(evaluated)
            expected_attempts.append(evaluated)
        expected_stages.append(_source_stage_row(source_id, request, current))

    observed_attempts = pd.read_parquet(root / "tier1_sifts_chain_attempts.parquet")
    expected_attempts_frame = pd.DataFrame(expected_attempts, columns=CHAIN_ATTEMPT_COLUMNS)
    _assert_frame_equal(observed_attempts, expected_attempts_frame,
                        sort=["source_id", "sifts_item_index"], label="Tier1 SIFTS attempts")
    observed_stages = pd.read_parquet(root / "tier1_source_funnel.parquet")
    expected_stages_frame = pd.DataFrame(expected_stages, columns=SOURCE_STAGE_COLUMNS)
    _assert_frame_equal(observed_stages, expected_stages_frame,
                        sort=["source_id"], label="Tier1 source funnel")
    valid_counts = (
        expected_attempts_frame[expected_attempts_frame["source_valid"]]
        .groupby("source_id", sort=True).size().to_dict()
        if len(expected_attempts_frame) else {}
    )
    expected_valid = pd.DataFrame([
        {
            "source_id": source_id,
            "canonical_sequence_length": unit_by_id[source_id]["canonical_sequence_length"],
            "canonical_sequence_sha256": unit_by_id[source_id]["canonical_sequence_sha256"],
            "raw_source_ids_json": unit_by_id[source_id]["raw_source_ids_json"],
            "viable_sifts_mapping_count": int(count),
        }
        for source_id, count in sorted(valid_counts.items())
    ], columns=VALID_SOURCE_UNIT_COLUMNS)
    _assert_frame_equal(pd.read_parquet(root / "tier1_source_frame_valid_units.parquet"),
                        expected_valid, sort=["source_id"], label="Tier1 valid source units")
    funnel = manifest["funnel"]
    if int(funnel["canonical_source_units"]) != len(unit_by_id):
        raise ValueError("Tier1 manifest canonical-source count mismatch")
    if int(funnel["source_frame_valid"]) != int(expected_stages_frame["source_frame_valid"].sum()):
        raise ValueError("Tier1 manifest valid-source count mismatch")
    if int(funnel["network_failure"]) or int(funnel["malformed_response"]):
        raise ValueError("Tier1 source snapshot contains unresolved request failure")


def validate_tier1_release_sources(
    manifest_path: Path, *, rows: Mapping[str, Mapping[str, Any]], allele: str,
    chain_attempt_path: Path, residue_manifest_path: Path,
) -> None:
    """Bind each released Tier1 row to one valid canonical UniProt/SIFTS source attempt."""

    root = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("allele") != allele:
        raise ValueError("Tier1 source manifest allele differs from release")
    units = pd.read_parquet(root / "tier1_canonical_source_units.parquet").set_index("source_id")
    attempts = pd.read_parquet(root / "tier1_sifts_chain_attempts.parquet")
    residue_root = Path(residue_manifest_path).resolve().parent
    residue_candidates = pd.read_parquet(
        residue_root / "tier1_residue_map_candidates.parquet"
    )
    chain_attempts = pd.read_parquet(chain_attempt_path)
    tier1_chain_attempts = chain_attempts[chain_attempts["source_tier"] == "tier1"]
    expected_chain_attempts: list[dict[str, Any]] = []
    for source_id, group in residue_candidates.groupby(
        "source_id", sort=True
    ):
        for order, source in enumerate(
            group.sort_values(
                ["sifts_item_index", "pdb_id", "auth_chain_id", "label_asym_id"],
                kind="stable",
            )
            .to_dict("records"), start=1,
        ):
            expected_chain_attempts.append({
                "source_uniprot_id": str(source_id), "pdb_id": str(source["pdb_id"]),
                "rcsb_entity_id": str(source["rcsb_entity_id"]),
                "label_asym_id": str(source["label_asym_id"]),
                "auth_chain_id": str(source["auth_chain_id"]),
                "sifts_item_index": int(source["sifts_item_index"]), "attempt_order": order,
                "source_revision_date": str(source["source_revision_date"]),
                "download_sha256": str(source["coordinate_cif_sha256"]),
            })
    observed_chain_attempts = tier1_chain_attempts.copy()
    observed_chain_attempts["pdb_id"] = observed_chain_attempts["rcsb_entity_id"].astype(str).str.split("_").str[0]
    _assert_frame_equal(
        observed_chain_attempts[[
            "source_uniprot_id", "pdb_id", "rcsb_entity_id", "label_asym_id",
            "auth_chain_id", "sifts_item_index", "attempt_order",
            "source_revision_date", "download_sha256",
        ]],
        pd.DataFrame(expected_chain_attempts),
        sort=["source_uniprot_id", "attempt_order"],
        label="Tier1 complete source-valid SIFTS materialization attempt set",
    )
    for pid, row in rows.items():
        memberships = row["selected_tier_memberships"]
        if isinstance(memberships, np.ndarray):
            memberships = memberships.tolist()
        if list(memberships) != ["tier1"]:
            continue
        source_id = str(row.get("source_uniprot_id") or "")
        if source_id not in units.index:
            raise ValueError(f"Tier1 release row {pid} lacks canonical source provenance")
        unit = units.loc[source_id]
        if (
            str(unit["canonical_sequence"]) != str(row["source_sequence"])
            or unit["canonical_sequence_sha256"] != row["source_sequence_sha256"]
        ):
            raise ValueError(f"Tier1 release row {pid} canonical sequence mismatch")
        sifts_range = json.loads(str(row["sifts_range_json"]))
        matches = attempts[
            (attempts["source_id"].astype(str) == source_id)
            & (attempts["pdb_id"].astype(str) == str(row["pdb_id"]))
            & (attempts["auth_chain_id"].astype(str) == str(row["auth_chain_id"]))
            & (attempts["unp_start"].astype(int) == int(sifts_range[0]))
            & (attempts["unp_end"].astype(int) == int(sifts_range[1]))
            & (attempts["source_valid"] == True)  # noqa: E712
        ]
        if matches.empty:
            raise ValueError(f"Tier1 release row {pid} lacks a valid frozen SIFTS witness")
        detailed_matches = residue_candidates[
            (residue_candidates["source_id"].astype(str) == source_id)
            & (residue_candidates["pdb_id"].astype(str) == str(row["pdb_id"]))
            & (residue_candidates["rcsb_entity_id"].astype(str) == str(row["rcsb_entity_id"]))
            & (residue_candidates["label_asym_id"].astype(str) == str(row["label_asym_id"]))
            & (residue_candidates["auth_chain_id"].astype(str) == str(row["auth_chain_id"]))
            & (residue_candidates["unp_start"].astype(int) == int(sifts_range[0]))
            & (residue_candidates["unp_end"].astype(int) == int(sifts_range[1]))
        ]
        if len(detailed_matches) != 1:
            raise ValueError(f"Tier1 release row {pid} lacks one detailed SIFTS residue witness")
        expected_map = json.loads(str(detailed_matches.iloc[0]["residue_map_json"]))
        observed_map = json.loads(str(row["sifts_residue_map_json"]))
        if observed_map != expected_map:
            raise ValueError(f"Tier1 release row {pid} detailed SIFTS residue map differs from raw")


def validate_chain_collision_dedup_ledgers(
    paths: Mapping[str, Path], *, rows: Mapping[str, Mapping[str, Any]],
    enforce_release_membership: bool = True,
) -> None:
    """Replay C6 winner selection, C8 collisions, and final exact-sequence deduplication."""

    attempts = pd.read_parquet(paths["chain_attempts"])
    required = {
        "selection_unit_id", "rcsb_entity_id", "auth_chain_id", "label_asym_id", "viable",
        "if_sequence_coverage", "resolution", "sequence_sha256", "structure_sha256",
        "failure_reason", "source_tier", "attempt_order", "attempt_status",
        "source_revision_date", "download_sha256", "fallback_rank",
        "source_uniprot_id", "sifts_item_index",
    }
    if not required <= set(attempts):
        raise ValueError("chain attempt ledger schema mismatch")
    if attempts.duplicated(
        ["source_tier", "selection_unit_id", "rcsb_entity_id", "label_asym_id", "auth_chain_id"]
    ).any():
        raise ValueError("chain attempt ledger has duplicate entity-chain attempt keys")
    for attempt in attempts.to_dict("records"):
        downloaded = str(attempt["attempt_status"]) != "download_failure"
        if (
            not str(attempt["source_revision_date"] or "")
            or (downloaded and (
                not isinstance(attempt["download_sha256"], str)
                or len(attempt["download_sha256"]) != 64
            ))
            or (not downloaded and not str(attempt["failure_reason"] or ""))
        ):
            raise ValueError("chain attempt lacks revision/download/failure provenance")
    missing_released_units = (
        {str(row["selection_unit_id"]) for row in rows.values()}
        - set(attempts["selection_unit_id"].astype(str))
    )
    if enforce_release_membership and missing_released_units:
        raise ValueError(
            f"chain attempt ledger misses released selection units: {sorted(missing_released_units)}"
        )
    for unit, group in attempts.groupby("selection_unit_id", sort=False):
        viable = group[group["viable"] == True].copy()  # noqa: E712
        released = [row for row in rows.values() if str(row["selection_unit_id"]) == str(unit)]
        if not released:
            continue
        if len(released) != 1 or viable.empty:
            raise ValueError("chain attempt ledger lacks one released viable winner")
        viable = viable.sort_values(
            ["if_sequence_coverage", "resolution", "rcsb_entity_id", "auth_chain_id", "label_asym_id"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        winner = viable.iloc[0]
        row = released[0]
        if any(str(winner[key]) != str(row[key]) for key in ("rcsb_entity_id", "auth_chain_id", "label_asym_id")):
            raise ValueError("released chain is not the replayed C6 winner")
        if winner["sequence_sha256"] != row["sequence_sha256"] or winner["structure_sha256"] != row["structure_sha256"]:
            raise ValueError("released chain bytes do not match C6 winner ledger")
        selected = json.loads(str(row["selected_chain_provenance_json"]))
        if str(selected.get("rcsb_entity_id")) != str(winner["rcsb_entity_id"]):
            raise ValueError("selected-chain provenance does not bind the C6 winner")
        rejected = json.loads(str(row["rejected_chain_provenance_json"]))
        expected_rejected = sorted(
            str(value) for value in group.loc[group.index != winner.name, "rcsb_entity_id"].tolist()
        )
        if sorted(str(value["rcsb_entity_id"]) for value in rejected) != expected_rejected:
            raise ValueError("rejected-chain provenance does not match attempt ledger")

    collision = json.loads(paths["collision_replay"].read_text())
    if collision.get("schema_version") != "if-benchmark-v3-collision-replay/1":
        raise ValueError("collision replay schema mismatch")
    expected = resolve_tier1_tier2_collisions(
        t1_primary=collision["inputs"]["tier1_primary"],
        t1_diagnostic=collision["inputs"]["tier1_diagnostic"],
        t2_rows=collision["inputs"]["tier2_rows"],
    )
    observed = (
        collision["outputs"]["tier1_primary"], collision["outputs"]["tier1_diagnostic"],
        collision["outputs"]["tier2_rows"], collision["collision_ledger"],
    )
    if canonical_json(expected) != canonical_json(observed):
        raise ValueError("Tier1/Tier2 collision ledger does not replay")
    released_t1_primary = {
        pid for pid, row in rows.items()
        if list(row["selected_tier_memberships"]) == ["tier1"]
        and row["evaluation_role"] == "primary_generalization"
    }
    released_t1_diagnostic = {
        pid for pid, row in rows.items()
        if row["evaluation_role"] == "tier1_overlap_diagnostic"
    }
    released_t2_units = {
        str(row["selection_unit_id"]) for row in rows.values()
        if list(row["selected_tier_memberships"]) == ["tier2"]
    }
    if enforce_release_membership and {str(row["protein_id"]) for row in expected[0]} != released_t1_primary:
        raise ValueError("post-collision Tier1 primary rows differ from release")
    if enforce_release_membership and {str(row["protein_id"]) for row in expected[1]} != released_t1_diagnostic:
        raise ValueError("post-collision Tier1 diagnostics differ from release")
    if enforce_release_membership and not released_t2_units <= {
        str(row.get("selection_unit_id") or "") for row in expected[2]
    }:
        raise ValueError("released Tier2 unit is absent from post-collision pool")

    dedup = json.loads(paths["final_dedup_replay"].read_text())
    if dedup.get("schema_version") != "if-benchmark-v3-final-dedup-replay/1":
        raise ValueError("final dedup replay schema mismatch")
    source = pd.DataFrame(dedup["inputs"])
    required_dedup = {
        "protein_id", "selection_unit_id", "sequence_sha256", "if_sequence_coverage", "resolution",
        "rcsb_entity_id", "auth_chain_id", "label_asym_id",
    }
    if not required_dedup <= set(source) or source["protein_id"].isna().any():
        raise ValueError("final dedup input schema mismatch")
    chosen: list[str] = []
    ledger: list[dict[str, Any]] = []
    for sequence_sha, group in source.groupby("sequence_sha256", sort=True):
        ranked = group.sort_values(
            ["if_sequence_coverage", "resolution", "rcsb_entity_id", "auth_chain_id", "label_asym_id"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        winner = str(ranked.iloc[0]["protein_id"])
        chosen.append(winner)
        for pid in sorted(ranked["protein_id"].astype(str)):
            ledger.append({"protein_id": pid, "sequence_sha256": str(sequence_sha),
                           "representative_protein_id": winner,
                           "action": "keep" if pid == winner else "collapse"})
    if sorted(chosen) != sorted(str(value) for value in dedup["selected_protein_ids"]):
        raise ValueError("final exact-sequence representative set does not replay")
    if canonical_json(sorted(ledger, key=lambda value: value["protein_id"])) != canonical_json(
        sorted(dedup["dedup_ledger"], key=lambda value: value["protein_id"])
    ):
        raise ValueError("final exact-sequence collapse ledger does not replay")
    released_ids = {
        pid for pid, row in rows.items()
        if list(row["selected_tier_memberships"]) == ["tier2"]
    }
    if not released_ids <= set(chosen):
        raise ValueError("released row is absent from final dedup representatives")


def validate_tier1_internal_pool_replay(
    paths: Mapping[str, Path], *, rows: Mapping[str, Mapping[str, Any]], target: int
) -> None:
    candidates = pd.read_parquet(paths["tier1_candidate_pool"]).to_dict("records")
    primary, diagnostic, ledger = finalize_tier1_pool(candidates, target=target)
    recorded_ledger = json.loads(paths["tier1_internal_collision_ledger"].read_text())
    if recorded_ledger.get("schema_version") != "if-benchmark-v3-tier1-internal-collisions/1":
        raise ValueError("Tier1 internal collision ledger schema mismatch")
    if canonical_json(recorded_ledger.get("rows")) != canonical_json(ledger):
        raise ValueError("Tier1 internal entity/final-sequence collision ledger does not replay")
    released_primary = {
        pid for pid, row in rows.items()
        if list(row["selected_tier_memberships"]) == ["tier1"]
        and row["evaluation_role"] == "primary_generalization"
    }
    released_diagnostic = {
        pid for pid, row in rows.items() if row["evaluation_role"] == "tier1_overlap_diagnostic"
    }
    if {str(row["protein_id"]) for row in primary} != released_primary:
        raise ValueError("replayed Tier1 internal collapse/backfill differs from primary release")
    if {str(row["protein_id"]) for row in diagnostic} != released_diagnostic:
        raise ValueError("replayed Tier1 diagnostic collapse differs from release")


def validate_selection_replay(
    paths: Mapping[str, Path], *, ledger_paths: Mapping[str, Path],
    primary: pd.DataFrame, tier2_target: int, allele: str,
    rcsb_manifest_path: Path, mmseqs_identity: Mapping[str, Any],
    candidate_root: Path, nmp_install_identity: Mapping[str, Any],
    eligibility_only: bool = False, expected_c5_target: int | None = None,
    protocol_profile: str = "drb1501_production",
) -> int:
    """Replay C5/C7 and the complete C5 -> C6 -> C7 eligibility lineage."""

    rcsb_root = Path(rcsb_manifest_path).resolve().parent
    groups = pd.read_parquet(rcsb_root / "sequence_groups.parquet")
    source_queries = pd.read_parquet(paths["source_cath_queries"])
    expected_source_queries = groups[
        ["selection_unit_id", "entity_sequence", "entity_sequence_sha256"]
    ].rename(columns={"selection_unit_id": "protein_id", "entity_sequence": "sequence",
                      "entity_sequence_sha256": "sequence_sha256"})
    _assert_frame_equal(
        source_queries, expected_source_queries, sort=["protein_id"],
        label="C2 complete source-CATH query pool",
    )
    source_cath = pd.read_parquet(paths["source_cath_sidecar"])
    source_identity_rows = source_cath[[
        "reference_sha256", "tool_sha256", "tool_version", "cov_mode", "coverage",
        "min_seq_id", "raw_output_path", "query_fasta_path",
    ]].drop_duplicates()
    if len(source_identity_rows) != 1:
        raise ValueError("source CATH prefilter has mixed search identity")
    source_identity = source_identity_rows.iloc[0]
    source_raw_path = resolve_candidate_stage_path(
        source_identity["raw_output_path"], candidate_root=Path(candidate_root),
        label="source CATH raw output",
    )
    source_fasta_path = resolve_candidate_stage_path(
        source_identity["query_fasta_path"], candidate_root=Path(candidate_root),
        label="source CATH query FASTA",
    )
    if (
        source_identity["reference_sha256"] != mmseqs_identity["cath_reference_sha256"]
        or source_identity["tool_sha256"] != mmseqs_identity["tool_sha256"]
        or source_identity["tool_version"] != mmseqs_identity["tool_version"]
        or int(source_identity["cov_mode"]) != 0
        or float(source_identity["coverage"]) != 0.8
        or float(source_identity["min_seq_id"]) != 0.3
    ):
        raise ValueError("source CATH prefilter does not use frozen reference/tool/parameters")
    validate_mmseqs_sidecar(
        source_queries[["protein_id", "sequence"]], source_cath,
        raw_tsv_path=source_raw_path,
        query_fasta_path=source_fasta_path, search_kind="cath",
        reference_sha256=str(source_identity["reference_sha256"]),
        tool_sha256=str(source_identity["tool_sha256"]),
        tool_version=str(source_identity["tool_version"]), cov_mode=0, coverage=0.8,
        min_seq_id=0.3, path_root=Path(candidate_root),
    )
    source_cath_by_id = source_cath.set_index("protein_id")
    if not set(source_cath["best_target"].dropna().astype(str)) <= set(
        mmseqs_identity["cath_reference_ids"]
    ):
        raise ValueError("source CATH output contains target absent from actual reference")
    clean_source_ids = {
        str(row.protein_id) for row in source_queries.itertuples(index=False)
        if not bool(source_cath_by_id.loc[str(row.protein_id), "overlap_flag"])
    }
    c5_pool = pd.read_parquet(paths["c5_eligible"])
    if set(c5_pool["selection_unit_id"].astype(str)) != clean_source_ids:
        raise ValueError("C5 eligible pool is not the exact C2 source-CATH-clean set")
    source_query_by_id = source_queries.set_index("protein_id")
    for row in c5_pool.itertuples(index=False):
        query = source_query_by_id.loc[str(row.selection_unit_id)]
        if (
            str(row.source_sequence) != str(query.sequence)
            or str(row.source_sequence_sha256) != str(query.sequence_sha256)
        ):
            raise ValueError("C5 source sequence differs from C2 source-CATH query bytes")
    validate_c5_nmp_evidence(
        paths, allele=allele, install_identity=nmp_install_identity
    )
    results: dict[str, Any] = {}
    eligible_tables: dict[str, pd.DataFrame] = {}
    for stage in (("c5",) if eligibility_only else ("c5", "c7")):
        eligible = pd.read_parquet(paths[f"{stage}_eligible"])
        eligible_tables[stage] = eligible
        payload = json.loads(paths[f"{stage}_result"].read_text())
        if (
            payload.get("schema_version") != "if-benchmark-v3-binned-sample/1"
            or payload.get("protocol_profile") != protocol_profile
        ):
            raise ValueError(f"{stage.upper()} sample evidence schema mismatch")
        recorded = binned_sample_from_dict(payload["result"])
        if stage == "c5":
            frozen = deterministic_binned_sample(
                eligible, value_col="selection_source_coverage_fraction_15",
                target=recorded.target, n_bins=20, mode="uniform", stage="c5",
                min_per_populated_bin=0, seed=42, mu=None, sigma=None,
            )
        else:
            expected_target = 3000 if protocol_profile == "drb1501_production" else 1
            expected_floor = 40 if protocol_profile == "drb1501_production" else 1
            frozen = deterministic_binned_sample(
                eligible, value_col="coverage_fraction", target=expected_target,
                n_bins=20, mode="gaussian", stage="c7",
                min_per_populated_bin=expected_floor, seed=42, mu=None, sigma=None,
            )
        if binned_sample_to_dict(frozen) != payload["result"]:
            raise ValueError(f"{stage.upper()} sample violates the frozen protocol/replay")
        results[stage] = recorded
    if expected_c5_target is not None and int(results["c5"].target) != int(expected_c5_target):
        raise ValueError("C5 attempt sample target differs from its ladder rung")

    c5_selected = set(results["c5"].selected_ids)
    attempts = pd.read_parquet(ledger_paths["chain_attempts"])
    if "source_tier" not in attempts or "protein_id" not in attempts:
        raise ValueError("chain attempt ledger lacks selection-lineage columns")
    tier2_attempts = attempts[attempts["source_tier"] == "tier2"]
    if set(tier2_attempts["selection_unit_id"].astype(str)) != c5_selected:
        raise ValueError("C6 attempt units do not exactly account for C5 membership")
    fallback_table = pd.read_parquet(
        Path(rcsb_manifest_path).resolve().parent / "ordered_entity_fallbacks.parquet"
    )
    fallback_by_unit = {
        str(unit): group.sort_values("fallback_rank", kind="stable")
        for unit, group in fallback_table.groupby("selection_unit_id", sort=False)
    }
    expected_attempts: list[dict[str, Any]] = []
    for unit in sorted(c5_selected):
        group = fallback_by_unit.get(unit)
        if group is None or group.empty:
            raise ValueError("C5 unit has no C2 ordered entity fallbacks")
        attempt_order = 0
        for fallback in group.to_dict("records"):
            edges = json.loads(str(fallback["entity_chain_edges_json"]))
            for edge in sorted(edges, key=lambda row: (row["label_asym_id"], row["auth_asym_id"])):
                attempt_order += 1
                expected_attempts.append({
                    "selection_unit_id": unit, "rcsb_entity_id": str(fallback["rcsb_entity_id"]),
                    "label_asym_id": str(edge["label_asym_id"]),
                    "auth_chain_id": str(edge["auth_asym_id"]),
                    "fallback_rank": int(fallback["fallback_rank"]),
                    "attempt_order": attempt_order,
                    "source_revision_date": str(fallback["structure_revision_date"]),
                })
    observed_attempts = tier2_attempts[[
        "selection_unit_id", "rcsb_entity_id", "label_asym_id", "auth_chain_id",
        "fallback_rank", "attempt_order", "source_revision_date",
    ]].copy()
    _assert_frame_equal(
        observed_attempts, pd.DataFrame(expected_attempts),
        sort=["selection_unit_id", "attempt_order"],
        label="C5 selected complete ordered entity-chain attempt set",
    )
    successful_units = {
        str(unit) for unit, group in tier2_attempts.groupby("selection_unit_id", sort=False)
        if bool(group["viable"].any())
    }
    failed_units = c5_selected - successful_units
    if any(
        not str(value) for value in tier2_attempts.loc[
            tier2_attempts["selection_unit_id"].astype(str).isin(failed_units), "failure_reason"
        ]
    ):
        raise ValueError("failed C6 selection unit lacks an explicit failure reason")

    dedup = json.loads(ledger_paths["final_dedup_replay"].read_text())
    dedup_inputs = pd.DataFrame(dedup["inputs"])
    if set(dedup_inputs["selection_unit_id"].astype(str)) != successful_units:
        raise ValueError("final dedup inputs differ from successful C6 units")
    for unit, group in tier2_attempts.groupby("selection_unit_id", sort=False):
        viable = group[group["viable"] == True].sort_values(  # noqa: E712
            ["if_sequence_coverage", "resolution", "rcsb_entity_id", "auth_chain_id", "label_asym_id"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        if viable.empty:
            continue
        winner = viable.iloc[0]
        materialized = dedup_inputs[
            dedup_inputs["selection_unit_id"].astype(str) == str(unit)
        ].iloc[0]
        if any(
            str(winner[field]) != str(materialized[field])
            for field in (
                "protein_id", "sequence_sha256", "rcsb_entity_id",
                "auth_chain_id", "label_asym_id",
            )
        ):
            raise ValueError("final dedup input is not the replayed C6 winner")
    dedup_selected_ids = set(str(value) for value in dedup["selected_protein_ids"])
    collision = json.loads(ledger_paths["collision_replay"].read_text())
    collision_t2_input = collision["inputs"]["tier2_rows"]
    if {str(row["protein_id"]) for row in collision_t2_input} != dedup_selected_ids:
        raise ValueError("collision inputs differ from final Tier2 dedup representatives")
    dedup_by_id = dedup_inputs.set_index("protein_id")
    for row in collision_t2_input:
        source = dedup_by_id.loc[str(row["protein_id"])]
        if (
            str(row["selection_unit_id"]) != str(source["selection_unit_id"])
            or str(row["sequence_sha256"]) != str(source["sequence_sha256"])
            or str(row["rcsb_entity_id"]) != str(source["rcsb_entity_id"])
        ):
            raise ValueError("collision input bytes differ from final dedup representative")
    post_collision = collision["outputs"]["tier2_rows"]

    cath_queries = pd.read_parquet(paths["final_cath_queries"])
    expected_post_collision = {
        str(row["selection_unit_id"]): (str(row["protein_id"]), str(row["sequence"]))
        for row in post_collision
    }
    observed_queries = {
        str(row.selection_unit_id): (str(row.protein_id), str(row.sequence))
        for row in cath_queries.itertuples(index=False)
    }
    if observed_queries != expected_post_collision:
        raise ValueError("final CATH query pool differs from post-collision Tier2 pool")
    cath_sidecar = pd.read_parquet(paths["final_cath_sidecar"])
    identity_columns = (
        "reference_sha256", "tool_sha256", "tool_version", "cov_mode", "coverage",
        "min_seq_id", "raw_output_path", "query_fasta_path",
    )
    identities = cath_sidecar[list(identity_columns)].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("final-eligibility CATH sidecar has mixed search identity")
    identity = identities.iloc[0]
    final_raw_path = resolve_candidate_stage_path(
        identity["raw_output_path"], candidate_root=Path(candidate_root),
        label="final CATH raw output",
    )
    final_fasta_path = resolve_candidate_stage_path(
        identity["query_fasta_path"], candidate_root=Path(candidate_root),
        label="final CATH query FASTA",
    )
    if (
        identity["reference_sha256"] != mmseqs_identity["cath_reference_sha256"]
        or identity["tool_sha256"] != mmseqs_identity["tool_sha256"]
        or identity["tool_version"] != mmseqs_identity["tool_version"]
        or int(identity["cov_mode"]) != 0 or float(identity["coverage"]) != 0.8
        or float(identity["min_seq_id"]) != 0.3
    ):
        raise ValueError("final CATH lineage search does not use frozen actual identities")
    validate_mmseqs_sidecar(
        cath_queries[["protein_id", "sequence"]], cath_sidecar,
        raw_tsv_path=final_raw_path,
        query_fasta_path=final_fasta_path, search_kind="cath",
        reference_sha256=str(identity["reference_sha256"]),
        tool_sha256=str(identity["tool_sha256"]), tool_version=str(identity["tool_version"]),
        cov_mode=int(identity["cov_mode"]), coverage=float(identity["coverage"]),
        min_seq_id=float(identity["min_seq_id"]), path_root=Path(candidate_root),
    )
    cath_by_id = cath_sidecar.set_index("protein_id")
    if not set(cath_sidecar["best_target"].dropna().astype(str)) <= set(
        mmseqs_identity["cath_reference_ids"]
    ):
        raise ValueError("final CATH output contains target absent from actual reference")
    clean_units = {
        str(row.selection_unit_id) for row in cath_queries.itertuples(index=False)
        if not bool(cath_by_id.loc[str(row.protein_id), "overlap_flag"])
    }
    c7_eligible = eligible_tables.get("c7")
    if c7_eligible is None:
        c7_eligible = pd.read_parquet(paths["c7_eligible"])
    if set(c7_eligible["selection_unit_id"].astype(str)) != clean_units:
        raise ValueError("C7 eligible set is not the exact final CATH-clean lineage")
    for row in c7_eligible.itertuples(index=False):
        cath = cath_by_id.loc[str(row.protein_id)]
        if (
            str(row.sequence_sha256) != str(cath.query_sequence_sha256)
            or bool(row.cath_overlap_flag) != bool(cath.overlap_flag)
            or str(row.cath_query_sha256) != str(cath.query_sequence_sha256)
            or str(row.cath_reference_sha256) != str(cath.reference_sha256)
            or str(row.cath_tool_sha256) != str(cath.tool_sha256)
            or str(row.cath_parameters_sha256) != str(cath.parameters_sha256)
        ):
            raise ValueError("C7 eligible CATH evidence differs from persisted final search")
    if eligibility_only:
        return len(c7_eligible)
    expected_t2 = set(
        primary.loc[
            primary["selected_tier_memberships"].map(
                lambda value: list(value) if not isinstance(value, np.ndarray) else value.tolist()
            ).map(lambda value: value == ["tier2"]),
            "selection_unit_id",
        ].astype(str)
    )
    if set(results["c7"].selected_ids) != expected_t2:
        raise ValueError("C7 replayed membership differs from released Tier2 membership")
    if len(c7_eligible) < tier2_target:
        raise ValueError("C7 eligible capacity is below the preregistered Tier2 target")
    ladder = json.loads(paths["c5_expansion"].read_text())
    if ladder.get("schema_version") != "if-benchmark-v3-c5-expansion/1":
        raise ValueError("C5 expansion evidence schema mismatch")
    rungs = [int(value) for value in ladder["preregistered_ladder"]]
    attempts = ladder["attempts"]
    if any(set(row) != {"target", "observed_final_clean_capacity"} for row in attempts):
        raise ValueError("C5 expansion attempts contain caller assertions or unknown fields")
    if len(attempts) != 1:
        raise ValueError("current build must contain exactly one independently replayed C5 rung")
    current_target = int(results["c5"].target)
    if not rungs or current_target not in rungs or int(attempts[0]["target"]) != current_target:
        raise ValueError("current C5 attempt does not follow the preregistered ladder")
    if tier2_target == 3000:
        if rungs != [5000, 7500, 10000]:
            raise ValueError("production C5 ladder differs from 5000/7500/10000")
    elif not (tier2_target == 1 and rungs == [1] and int(attempts[0]["target"]) == 1):
        raise ValueError("non-production C5 ladder is allowed only for the one-row tiny fixture")

    validate_c5_prior_rung_history(
        index_path=paths["c5_prior_rungs"], ladder=rungs, current_target=current_target,
        tier2_target=tier2_target, candidate_root=Path(candidate_root), allele=allele,
        rcsb_manifest_path=rcsb_manifest_path, mmseqs_identity=mmseqs_identity,
        nmp_install_identity=nmp_install_identity, primary_columns=list(primary.columns),
        protocol_profile=protocol_profile,
    )
    if not attempts or int(attempts[-1]["observed_final_clean_capacity"]) != len(c7_eligible):
        raise ValueError("C5 final capacity was not recomputed from the C7 eligible set")
    if int(attempts[-1]["target"]) != int(results["c5"].target):
        raise ValueError("C5 selected rung differs from replayed sample target")
    return len(c7_eligible)


def validate_c5_prior_rung_history(
    *, index_path: Path, ladder: Sequence[int], current_target: int, tier2_target: int,
    candidate_root: Path, allele: str, rcsb_manifest_path: Path,
    mmseqs_identity: Mapping[str, Any], nmp_install_identity: Mapping[str, Any],
    primary_columns: Sequence[str], protocol_profile: str,
) -> list[dict[str, int]]:
    """Recursively replay copied failed-rung bytes before allowing the next ladder rung."""

    rungs = [int(value) for value in ladder]
    if current_target not in rungs:
        raise ValueError("current target is absent from C5 ladder")
    prior_index = json.loads(Path(index_path).read_text())
    if prior_index.get("schema_version") != "if-benchmark-v3-c5-prior-rungs/1":
        raise ValueError("C5 prior-rung index schema mismatch")
    expected_prior_targets = rungs[:rungs.index(int(current_target))]
    prior_records = prior_index.get("rungs")
    if (
        not isinstance(prior_records, list)
        or [int(record.get("target", -1)) for record in prior_records] != expected_prior_targets
    ):
        raise ValueError("C5 prior-rung index does not exactly cover earlier ladder attempts")
    measured: list[dict[str, int]] = []
    for prior_target, record in zip(expected_prior_targets, prior_records, strict=True):
        if set(record) not in (
            {"target", "attempt_bundle"},
            {"target", "attempt_bundle", "attempt_root"},
        ):
            raise ValueError("C5 prior-rung index contains untyped fields")
        bundle_record = record["attempt_bundle"]
        if Path(str(bundle_record.get("path") or "")).is_absolute():
            raise ValueError("prior C5 attempt bundle must be copied into the new candidate")
        attempt_path = _resolve_identity(
            bundle_record, candidate_root=Path(candidate_root),
            label=f"c5_prior_rung.{prior_target}",
        )
        attempt_root = Path(candidate_root)
        if "attempt_root" in record:
            raw_root = Path(str(record["attempt_root"]))
            attempt_root = (Path(candidate_root) / raw_root).resolve()
            if (
                raw_root.is_absolute() or ".." in raw_root.parts
                or Path(candidate_root).resolve() not in attempt_root.parents
                or not attempt_root.is_dir()
            ):
                raise ValueError("prior C5 attempt root escapes the new candidate")
        prior = _load_failed_c5_attempt_bundle(
            attempt_path, candidate_root=attempt_root, expected_target=prior_target
        )
        prior["selection"]["c5_nmp_tool"] = Path(nmp_install_identity["tool_path"])
        prior["nmp"]["tool"] = Path(nmp_install_identity["tool_path"])
        validate_chain_collision_dedup_ledgers(
            prior["ledgers"], rows={}, enforce_release_membership=False
        )
        prior_capacity = validate_selection_replay(
            prior["selection"], ledger_paths=prior["ledgers"],
            primary=pd.DataFrame(columns=list(primary_columns)),
            tier2_target=tier2_target, allele=allele, rcsb_manifest_path=rcsb_manifest_path,
            mmseqs_identity=mmseqs_identity, candidate_root=attempt_root,
            nmp_install_identity=nmp_install_identity, eligibility_only=True,
            expected_c5_target=prior_target, protocol_profile=protocol_profile,
        )
        validate_nmp_evidence(
            prior["nmp"], c7_eligible_path=prior["selection"]["c7_eligible"],
            rows={}, allele=allele, install_identity=nmp_install_identity,
        )
        if prior_capacity >= tier2_target:
            raise ValueError("C5 advanced past an earlier rung that already had sufficient capacity")
        measured.append({"target": prior_target, "measured_final_clean_capacity": prior_capacity})
    return measured


def _parse_nmp_raw(path: Path) -> pd.DataFrame:
    columns = ["protein_id", "peptide_length", "start_0b", "end_0b", "peptide", "rank_el"]
    frame = pd.read_csv(path, sep="\t", names=columns, header=None, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError("NMP raw output schema mismatch")
    return frame


def validate_nmp_install_evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    manifest_path = paths["data_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "if-benchmark-v3-nmp-install/1":
        raise ValueError("NetMHCIIpan install manifest schema mismatch")
    tool = paths["tool"].resolve()
    raw_tool_path = Path(manifest.get("tool_path") or "")
    manifest_tool_path = (
        raw_tool_path if raw_tool_path.is_absolute() else manifest_path.parent / raw_tool_path
    ).resolve()
    if (
        manifest_tool_path != tool
        or manifest.get("tool_sha256") != sha256_file(tool)
        or not str(manifest.get("tool_version") or "")
        or not isinstance(manifest.get("effective_environment"), Mapping)
        or manifest.get("identity_mode") not in {"official_install_merkle", "fixture"}
    ):
        raise ValueError("NetMHCIIpan binary/version/environment identity mismatch")
    raw_data_root = Path(manifest.get("data_root") or "")
    data_root = (
        raw_data_root if raw_data_root.is_absolute() else manifest_path.parent / raw_data_root
    ).resolve()
    files = data_directory_file_table(data_root)
    if (
        files != manifest.get("data_files")
        or manifest.get("data_merkle_sha256") != sha256_bytes(canonical_json(files))
    ):
        raise ValueError("NetMHCIIpan data/model directory Merkle mismatch")
    if paths["contract_producer"].resolve() != Path(__file__).with_name("nmp.py").resolve():
        raise ValueError("NetMHCIIpan install contract producer mismatch")
    expected_runner = Path(__file__).parents[3] / "epitope_head" / "data" / "netmhciipan_runner.py"
    if paths["runner_producer"].resolve() != expected_runner.resolve():
        raise ValueError("NetMHCIIpan scoring runner producer mismatch")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "tool_path": tool, "tool_sha256": manifest["tool_sha256"],
        "tool_version": str(manifest["tool_version"]),
        "data_merkle_sha256": str(manifest["data_merkle_sha256"]),
        "identity_mode": str(manifest.get("identity_mode") or ""),
        "effective_environment": dict(manifest["effective_environment"]),
    }


def _validate_nmp_pool(
    *, queries: pd.DataFrame, shard_manifest_path: Path,
    tool_path: Path, parameters_path: Path, allele: str, lengths: list[int], stage: str,
    key_column: str, sequence_column: str, sequence_sha_column: str,
    install_identity: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    params = json.loads(parameters_path.read_text())
    if params.get("schema_version") != "if-benchmark-v3-nmp-parameters/2":
        raise ValueError("NMP parameter schema mismatch")
    allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")
    expected_command = [
        "{netmhciipan_binary}", "-f", "{input_fasta}", "-a", allele_fmt,
        "-length", "{comma_separated_lengths}", "-context",
    ]
    if (
        params.get("allele") != allele or params.get("peptide_lengths") != lengths
        or params.get("stage") != stage or float(params.get("rank_el_threshold", -1)) != 2.0
        or params.get("allele_argument") != allele_fmt
        or params.get("install_manifest_sha256") != install_identity["manifest_sha256"]
        or params.get("filter_mode") != "off_by_omission_verified_netmhciipan_4_3"
        or params.get("context_mode") != "on_by_presence_verified_netmhciipan_4_3"
        or params.get("command_argv_template") != expected_command
        or not isinstance(params.get("runner"), Mapping)
        or any(int(params["runner"].get(key, 0)) < 1 for key in (
            "batch_size", "subprocess_timeout_s", "max_lengths_per_call", "n_workers"
        ))
        or Path(tool_path).resolve() != Path(install_identity["tool_path"]).resolve()
    ):
        raise ValueError("NMP install/command/environment/parameter identity mismatch")
    if queries[key_column].isna().any() or not queries[key_column].is_unique:
        raise ValueError(f"{stage} NMP query table has duplicate/missing keys")
    manifest, shard_paths = load_nmp_shard_manifest(
        shard_manifest_path, queries=queries, key_column=key_column,
        sequence_column=sequence_column, sequence_sha_column=sequence_sha_column,
        stage=stage, peptide_lengths=lengths, tool_path=tool_path,
        parameters_path=parameters_path,
    )
    tool_sha = sha256_file(tool_path)
    params_sha = sha256_file(parameters_path)
    ordered_queries = queries.sort_values(key_column, kind="stable").reset_index(drop=True)
    coverage_by_id: dict[str, float] = {}
    raw_sha_by_id: dict[str, str] = {}
    cursor = 0
    for shard_record, paths in zip(manifest["shards"], shard_paths, strict=True):
        query_count = int(shard_record["query_count"])
        shard_queries = ordered_queries.iloc[cursor:cursor + query_count]
        cursor += query_count
        raw = _parse_nmp_raw(paths["raw_output"])
        windows = pd.read_parquet(paths["windows"])
        try:
            pd.testing.assert_frame_equal(
                windows.reset_index(drop=True), raw.reset_index(drop=True), check_dtype=False
            )
        except AssertionError as exc:
            raise ValueError(f"{stage} NMP window table does not replay raw canonical rows") from exc
        native_files = shard_record.get("native_files") or []
        if native_files:
            native_replay = replay_native_nmp_shard(
                shard_record=shard_record,
                manifest_root=Path(shard_manifest_path).resolve().parent,
                expected_queries=shard_queries, key_column=key_column,
                sequence_column=sequence_column, tool_path=tool_path,
                parameters_path=parameters_path,
            )
            try:
                pd.testing.assert_frame_equal(
                    raw.reset_index(drop=True), native_replay.reset_index(drop=True),
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise ValueError(
                    f"{stage} NMP canonical windows differ from native stdout replay"
                ) from exc
        elif install_identity.get("identity_mode") == "official_install_merkle":
            raise ValueError("production NMP shard lacks native stdout replay evidence")
        summary = pd.read_parquet(paths["summary"])
        expected_ids = set(shard_queries[key_column].astype(str))
        if (
            set(summary["protein_id"].astype(str)) != expected_ids
            or summary["protein_id"].duplicated().any()
        ):
            raise ValueError(f"{stage} NMP summary key set differs from shard query pool")
        summary_by_id = summary.set_index(summary["protein_id"].astype(str), drop=False)
        raw = raw.copy()
        raw["protein_id"] = raw["protein_id"].astype(str)
        raw_ids = raw["protein_id"].to_numpy(dtype=str)
        raw_lengths = raw["peptide_length"].to_numpy(dtype=np.int64)
        raw_starts = raw["start_0b"].to_numpy(dtype=np.int64)
        same_id = raw_ids[1:] == raw_ids[:-1]
        if (
            (raw_ids[1:] < raw_ids[:-1]).any()
            or (same_id & (raw_lengths[1:] < raw_lengths[:-1])).any()
            or (
                same_id & (raw_lengths[1:] == raw_lengths[:-1])
                & (raw_starts[1:] <= raw_starts[:-1])
            ).any()
        ):
            raise ValueError(f"{stage} NMP raw output is not contiguous canonical order")
        boundaries = np.flatnonzero(np.r_[True, raw_ids[1:] != raw_ids[:-1], True])
        raw_ranges = {
            str(raw_ids[int(start)]): (int(start), int(end))
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        } if len(raw_ids) else {}
        raw_sha = sha256_file(paths["raw_output"])
        for query in shard_queries.to_dict("records"):
            pid = str(query[key_column])
            bounds = raw_ranges.get(pid)
            if bounds is None:
                raise ValueError(f"NMP raw output is missing query {pid}")
            observed = raw.iloc[bounds[0]:bounds[1]]
            sequence = str(query[sequence_column])
            sequence_sha = str(query[sequence_sha_column])
            if (
                not sequence or not set(sequence) <= AA20
                or sha256_bytes(sequence.encode()) != sequence_sha
            ):
                raise ValueError(f"{stage} NMP query sequence identity mismatch")
            expected_count = sum(max(0, len(sequence) - length + 1) for length in lengths)
            if len(observed) != expected_count:
                raise ValueError(f"NMP raw output is incomplete for {pid}")
            expected_keys = {
                (length, start, start + length)
                for length in lengths for start in range(max(0, len(sequence) - length + 1))
            }
            observed_keys = {
                (int(item.peptide_length), int(item.start_0b), int(item.end_0b))
                for item in observed.itertuples(index=False)
            }
            if observed_keys != expected_keys:
                raise ValueError(f"NMP window coordinates are incomplete/duplicated for {pid}")
            covered: set[int] = set()
            for window in observed.itertuples(index=False):
                start, end = int(window.start_0b), int(window.end_0b)
                if str(window.peptide) != sequence[start:end]:
                    raise ValueError("NMP peptide does not match final sequence")
                rank = float(window.rank_el)
                if not math.isfinite(rank) or not 0.0 <= rank <= 100.0:
                    raise ValueError("NMP output rank is outside [0,100]")
                if rank < 2.0:
                    covered.update(range(start, end))
            coverage = len(covered) / len(sequence)
            item = summary_by_id.loc[pid]
            if (
                item["status"] != "complete" or item["sequence_sha256"] != sequence_sha
                or item["raw_evidence_sha256"] != raw_sha
                or item["tool_sha256"] != tool_sha
                or item["parameters_sha256"] != params_sha
                or json.loads(str(item["scored_lengths_json"])) != lengths
                or not math.isfinite(float(item["coverage_fraction"]))
                or not 0.0 <= float(item["coverage_fraction"]) <= 1.0
                or abs(float(item["coverage_fraction"]) - coverage) > 1e-12
            ):
                raise ValueError(f"{stage} NMP summary evidence does not replay")
            coverage_by_id[pid] = coverage
            raw_sha_by_id[pid] = raw_sha
        if set(raw["protein_id"].astype(str)) != expected_ids:
            raise ValueError(f"{stage} NMP raw output contains extra/missing shard query IDs")
    if cursor != len(ordered_queries):
        raise ValueError("NMP shard query cursor does not cover eligible pool")
    return coverage_by_id, raw_sha_by_id


def validate_c5_nmp_evidence(
    paths: Mapping[str, Path], *, allele: str, install_identity: Mapping[str, Any]
) -> None:
    eligible = pd.read_parquet(paths["c5_eligible"])
    coverage, _raw_by_id = _validate_nmp_pool(
        queries=eligible,
        shard_manifest_path=paths["c5_nmp_shard_manifest"],
        tool_path=paths["c5_nmp_tool"],
        parameters_path=paths["c5_nmp_parameters"], allele=allele, lengths=[15], stage="c5",
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256",
        install_identity=install_identity,
    )
    score_identity = sha256_bytes(canonical_json({
        "manifest": sha256_file(paths["c5_nmp_shard_manifest"]),
        "tool": sha256_file(paths["c5_nmp_tool"]),
        "parameters": sha256_file(paths["c5_nmp_parameters"]),
    }))
    for row in eligible.to_dict("records"):
        if (
            row["score_identity_sha256"] != score_identity
            or abs(float(row["selection_source_coverage_fraction_15"])
                   - coverage[str(row["selection_unit_id"])]) > 1e-12
        ):
            raise ValueError("C5 eligible axis does not replay from complete length-15 NMP")


def validate_nmp_evidence(
    paths: Mapping[str, Path], *, c7_eligible_path: Path,
    rows: Mapping[str, Mapping[str, Any]], allele: str,
    install_identity: Mapping[str, Any],
) -> None:
    """Reparse full NMP for the entire C7 eligible pool, then bind released Tier2 rows."""

    eligible = pd.read_parquet(c7_eligible_path)
    coverage, raw_by_id = _validate_nmp_pool(
        queries=eligible, shard_manifest_path=paths["shard_manifest"],
        tool_path=paths["tool"],
        parameters_path=paths["parameters"], allele=allele, lengths=list(range(12, 26)),
        stage="c7", key_column="protein_id", sequence_column="sequence",
        sequence_sha_column="sequence_sha256",
        install_identity=install_identity,
    )
    for item in eligible.to_dict("records"):
        if (
            item["nmp_evidence_sha256"] != raw_by_id[str(item["protein_id"])]
            or abs(float(item["coverage_fraction"]) - coverage[str(item["protein_id"])]) > 1e-12
        ):
            raise ValueError("C7 eligible axis does not replay from full NMP")
    eligible_by_unit = eligible.set_index("selection_unit_id")
    tier2 = {
        pid: row for pid, row in rows.items()
        if list(row["selected_tier_memberships"]) == ["tier2"]
    }
    tool_sha = sha256_file(paths["tool"])
    params_sha = sha256_file(paths["parameters"])
    for pid, row in tier2.items():
        unit = str(row["selection_unit_id"])
        if unit not in eligible_by_unit.index:
            raise ValueError("released Tier2 row is absent from full NMP eligible pool")
        item = eligible_by_unit.loc[unit]
        if (
            str(item["protein_id"]) != pid
            or abs(float(row["coverage_fraction"]) - coverage[pid]) > 1e-12
            or row["nmp_evidence_sha256"] != raw_by_id[pid]
            or row["nmp_tool_sha256"] != tool_sha
            or row["nmp_parameters_sha256"] != params_sha
            or row["nmp_query_sequence_sha256"] != row["sequence_sha256"]
            or row["nmp_status"] != "complete"
        ):
            raise ValueError("Tier2 NMP row evidence does not replay")
        selection_bin = int(row["selection_bin"])
        if selection_bin < 0 or selection_bin > 19:
            raise ValueError("Tier2 selection_bin is outside 0..19")


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current: str | None = None
    chunks: list[str] = []
    for raw in path.read_text().splitlines():
        if raw.startswith(">"):
            if current is not None:
                records.append((current, "".join(chunks)))
            current = raw[1:].split()[0]
            chunks = []
        else:
            chunks.append(raw.strip())
    if current is not None:
        records.append((current, "".join(chunks)))
    return records


def _check_manifest_file_identity(
    record: Mapping[str, Any], label: str, *, root: Path | None = None
) -> Path:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if (
        not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"{label} content identity mismatch")
    return path


def validate_mmseqs_reference_evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Rebuild CATH-train and Head-seen FASTAs from their frozen source bytes."""

    identity = json.loads(paths["identity"].read_text())
    tool_sha = sha256_file(paths["tool"])
    if (
        identity.get("schema_version") != "if-benchmark-v3-mmseqs-tool/1"
        or identity.get("tool_sha256") != tool_sha
        or not str(identity.get("tool_version") or "")
    ):
        raise ValueError("shared MMseqs binary/version identity mismatch")
    expected_producer = Path(__file__).with_name("references.py").resolve()
    if paths["reference_producer"].resolve() != expected_producer:
        raise ValueError("MMseqs reference producer is not the registered v3 implementation")
    if paths["search_producer"].resolve() != Path(__file__).with_name("leakage.py").resolve():
        raise ValueError("MMseqs search parser is not the registered v3 implementation")

    cath_manifest = json.loads(paths["cath_reference_manifest"].read_text())
    if cath_manifest.get("schema_version") != "if-benchmark-v3-cath-reference/1":
        raise ValueError("CATH reference manifest schema mismatch")
    chain_set = _check_manifest_file_identity(cath_manifest["chain_set_jsonl"], "CATH chain_set")
    splits_path = _check_manifest_file_identity(cath_manifest["splits_json"], "CATH splits")
    cath_fasta = _check_manifest_file_identity(
        cath_manifest["reference_fasta"], "CATH FASTA",
        root=paths["cath_reference_manifest"].parent,
    )
    if cath_fasta.resolve() != paths["cath_reference_fasta"].resolve():
        raise ValueError("CATH reference manifest/bundle path mismatch")
    sequences: dict[str, str] = {}
    for line in chain_set.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name, sequence = str(row.get("name") or ""), str(row.get("seq") or "").upper()
        if (
            not name or not sequence
            or not set(sequence) <= CATH_REFERENCE_ALPHABET
            or name in sequences
        ):
            raise ValueError("CATH chain_set cannot replay a unique protein reference")
        sequences[name] = sequence
    train_ids = [str(value) for value in json.loads(splits_path.read_text()).get("train", [])]
    if not train_ids or len(train_ids) != len(set(train_ids)) or not set(train_ids) <= set(sequences):
        raise ValueError("CATH train split cannot replay from chain_set")
    expected_cath = [(name, sequences[name]) for name in sorted(train_ids)]
    if _read_fasta(cath_fasta) != expected_cath or int(cath_manifest["record_count"]) != len(expected_cath):
        raise ValueError("CATH train FASTA does not replay from chain_set/split")

    head_manifest = json.loads(paths["head_reference_manifest"].read_text())
    if (
        head_manifest.get("schema_version") != "if-benchmark-v3-head-seen-reference/1"
        or head_manifest.get("split_contract") != "strict_full_train_union_val_union_test"
        or [record.get("role") for record in head_manifest.get("split_id_files", [])]
        != ["train", "val", "test"]
    ):
        raise ValueError("Head seen reference manifest schema mismatch")
    seen_ids: set[str] = set()
    for record in head_manifest["split_id_files"]:
        split_path = _check_manifest_file_identity(record, "Head split IDs")
        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("Head split ID source contains duplicates")
        seen_ids.update(ids)
    protein_samples_path = _check_manifest_file_identity(
        head_manifest["protein_samples"], "Head protein_samples"
    )
    proteins = pd.read_parquet(protein_samples_path)
    proteins = proteins[
        proteins["protein_id"].astype(str).isin(seen_ids)
        & (proteins["allele"] == head_manifest["allele"])
    ]
    head_records: dict[str, str] = {}
    for protein_id, group in proteins.groupby("protein_id", sort=False):
        values = set(group["protein_seq"].astype(str).str.upper())
        if len(values) != 1:
            raise ValueError("Head protein_samples has conflicting seen sequence")
        sequence = values.pop()
        if not sequence or not set(sequence) <= CATH_REFERENCE_ALPHABET:
            raise ValueError("Head seen reference is not a valid protein sequence")
        head_records[str(protein_id)] = sequence
    head_fasta = _check_manifest_file_identity(
        head_manifest["reference_fasta"], "Head FASTA",
        root=paths["head_reference_manifest"].parent,
    )
    if head_fasta.resolve() != paths["head_reference_fasta"].resolve():
        raise ValueError("Head reference manifest/bundle path mismatch")
    expected_head = [(name, head_records[name]) for name in sorted(seen_ids)]
    if (
        set(head_records) != seen_ids or _read_fasta(head_fasta) != expected_head
        or int(head_manifest["record_count"]) != len(expected_head)
    ):
        raise ValueError("Head seen FASTA does not replay from strict split/protein inputs")
    return {
        "tool_sha256": tool_sha, "tool_version": str(identity["tool_version"]),
        "cath_reference_sha256": sha256_file(cath_fasta),
        "head_reference_sha256": sha256_file(head_fasta),
        "cath_chain_set_sha256": str(cath_manifest["chain_set_jsonl"]["sha256"]),
        "cath_splits_sha256": str(cath_manifest["splits_json"]["sha256"]),
        "cath_reference_ids": frozenset(name for name, _sequence in expected_cath),
        "head_reference_ids": frozenset(name for name, _sequence in expected_head),
    }


def stable_family_cluster_id(sequence_hashes: Sequence[str]) -> str:
    return sha256_bytes(canonical_json(sorted(str(value) for value in sequence_hashes)))


def validate_family_evidence(
    paths: Mapping[str, Path], *, primary: pd.DataFrame, diagnostic: pd.DataFrame,
    mmseqs_identity: Mapping[str, Any],
    protocol_profile: str,
) -> None:
    """Recompute primary-only MMseqs clusters and stable content-derived family IDs."""

    records = _read_fasta(paths["query_fasta"])
    expected = [
        (str(row.protein_id), str(row.sequence))
        for row in primary.sort_values("protein_id", kind="stable").itertuples(index=False)
    ]
    if records != expected:
        raise ValueError("family clustering FASTA is not the sorted primary-only cohort")
    diagnostic_ids = set(diagnostic["protein_id"].astype(str))
    if any(pid in diagnostic_ids for pid, _ in records):
        raise ValueError("diagnostic row entered primary family clustering")
    params = json.loads(paths["parameters"].read_text())
    expected_params = {
        "schema_version": "if-benchmark-v3-family-parameters/1",
        "min_seq_id": 0.3, "cov_mode": 0, "coverage": 0.8,
        "cluster_mode": 0, "threads": 1,
    }
    if any(params.get(key) != value for key, value in expected_params.items()):
        raise ValueError("family MMseqs parameters differ from protocol")
    expected_command = [
        "mmseqs", "easy-cluster", "final.fasta", "clu", "tmp",
        "--min-seq-id", "0.3", "--cov-mode", "0", "-c", "0.8",
        "--cluster-mode", "0", "--threads", "1",
    ]
    if (
        params.get("tool_sha256") != sha256_file(paths["tool"])
        or params.get("tool_sha256") != mmseqs_identity["tool_sha256"]
        or params.get("tool_version") != mmseqs_identity["tool_version"]
        or params.get("command_argv") != expected_command
        or not isinstance(params.get("effective_defaults"), Mapping)
    ):
        raise ValueError("family MMseqs tool/version/command identity mismatch")
    if protocol_profile == "drb1501_production":
        defaults = params["effective_defaults"]
        help_path = paths["parameters"].parent / str(defaults.get("help_path") or "")
        if (
            defaults.get("contract") != "complete_easy_cluster_help_stdout"
            or not help_path.is_file()
            or sha256_file(help_path) != defaults.get("help_sha256")
        ):
            raise ValueError("family MMseqs effective-default help evidence mismatch")
        help_result = subprocess.run(
            [str(paths["tool"]), "easy-cluster", "-h"], text=True,
            capture_output=True, check=False, timeout=60,
            env={**os.environ, "LC_ALL": "C"},
        )
        if (
            help_result.returncode != 0
            or sha256_bytes(help_result.stdout.encode()) != defaults["help_sha256"]
        ):
            raise ValueError("family MMseqs effective defaults differ from bound tool")
    clusters: dict[str, list[str]] = {}
    member_seen: set[str] = set()
    for raw in paths["cluster_tsv"].read_text().splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != 2:
            raise ValueError("family cluster TSV must be representative/member")
        representative, member = parts
        if member in member_seen:
            raise ValueError("family cluster member occurs more than once")
        member_seen.add(member)
        clusters.setdefault(representative, []).append(member)
    primary_by_id = primary.set_index("protein_id")
    if member_seen != set(primary_by_id.index.astype(str)):
        raise ValueError("family cluster member set differs from primary cohort")
    expected_family: dict[str, str] = {}
    for members in clusters.values():
        family_id = stable_family_cluster_id(
            [str(primary_by_id.loc[member, "sequence_sha256"]) for member in members]
        )
        for member in members:
            expected_family[member] = family_id
    for row in primary.itertuples(index=False):
        if str(row.family_cluster_id) != expected_family[str(row.protein_id)]:
            raise ValueError("family_cluster_id does not replay from persisted clustering")
    if protocol_profile == "drb1501_production":
        with tempfile.TemporaryDirectory(prefix="ifbench-v3-family-") as temporary:
            root = Path(temporary)
            prefix = root / "clu"
            command = [
                str(paths["tool"]), "easy-cluster", str(paths["query_fasta"]),
                str(prefix), str(root / "tmp"), "--min-seq-id", "0.3",
                "--cov-mode", "0", "-c", "0.8", "--cluster-mode", "0",
                "--threads", "1",
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False, timeout=600,
                env={**os.environ, "LC_ALL": "C"},
            )
            if completed.returncode != 0:
                raise ValueError("family MMseqs validation rerun failed")
            rerun_path = root / "clu_cluster.tsv"
            if not rerun_path.is_file():
                raise ValueError("family MMseqs validation rerun lacks cluster TSV")
            rerun_clusters: dict[str, set[str]] = {}
            for line in rerun_path.read_text().splitlines():
                if line.strip():
                    representative, member = line.split("\t")
                    rerun_clusters.setdefault(representative, set()).add(member)
            persisted_partition = sorted(sorted(members) for members in clusters.values())
            rerun_partition = sorted(sorted(members) for members in rerun_clusters.values())
            if persisted_partition != rerun_partition:
                raise ValueError("persisted family partition differs from bound MMseqs rerun")


def validate_head_files(
    paths: Mapping[str, Path], head_sidecar: pd.DataFrame, *, protocol_profile: str,
    predictor_factory: Any | None = None,
) -> None:
    from .head_annotation import validate_head_annotation

    checkpoint_sha = sha256_file(paths["checkpoint"])
    config_sha = sha256_file(paths["config"])
    if not np.isfinite(head_sidecar["global_risk"].to_numpy(dtype=float)).all():
        raise ValueError("Head annotation contains non-finite global risk")
    if set(head_sidecar["head_checkpoint_sha256"].astype(str)) != {checkpoint_sha}:
        raise ValueError("Head checkpoint digest does not match actual file")
    if set(head_sidecar["head_config_sha256"].astype(str)) != {config_sha}:
        raise ValueError("Head config digest does not match actual file")
    model_paths = {
        "model.yaml": paths["model_yaml"],
        "model_ablation.yaml": paths["model_ablation_yaml"],
        "inference.yaml": paths["inference_yaml"],
    }
    parents = {path.parent.resolve() for path in model_paths.values()}
    if len(parents) != 1 or any(path.name != name for name, path in model_paths.items()):
        raise ValueError("Head evidence model configs do not form one canonical directory")
    annotation = validate_head_annotation(
        manifest_path=paths["annotation_manifest"], checkpoint_path=paths["checkpoint"],
        resolved_config_path=paths["config"], run_summary_path=paths["run_summary"],
        model_config_dir=parents.pop(),
        cohort_input_paths=[paths["cohort_primary"], paths["cohort_diagnostic"]],
        # The registered c9_head stage already produced and sealed these scores.
        # Release validation replays byte/model/cohort identity, not a second 3k-row GPU pass.
        replay=False,
        predictor_factory=predictor_factory,
    )
    annotation_output = paths["annotation_manifest"].parent / str(annotation["output_path"])
    persisted = pd.read_parquet(annotation_output)
    try:
        pd.testing.assert_frame_equal(
            persisted.reset_index(drop=True), head_sidecar.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("Head release table differs from replayed annotation output") from exc


def validate_approved_source_contract(
    *, manifest_path: Path, head_paths: Mapping[str, Path],
    mmseqs_identity: Mapping[str, Any], protocol_profile: str,
) -> dict[str, Any]:
    contract = json.loads(Path(manifest_path).read_text())
    if contract.get("schema_version") != "if-benchmark-v3-approved-sources/1":
        raise ValueError("approved source contract schema mismatch")
    unsigned = dict(contract)
    digest = unsigned.pop("manifest_sha256", None)
    if digest != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("approved source contract self-hash mismatch")
    if contract.get("protocol_profile") != protocol_profile:
        raise ValueError("approved source contract protocol profile mismatch")
    head = contract.get("head") or {}
    cath = contract.get("cath") or {}
    actual = {
        "checkpoint_sha256": sha256_file(head_paths["checkpoint"]),
        "config_sha256": sha256_file(head_paths["config"]),
        "run_summary_sha256": sha256_file(head_paths["run_summary"]),
    }
    if (
        head.get("allele") != "HLA-DRB1*15:01" or int(head.get("fixed_epoch", -1)) != 24
        or any(head.get(key) != value for key, value in actual.items())
        or Path(head_paths["checkpoint"]).name == "best.pt"
        or cath.get("chain_set_sha256") != mmseqs_identity["cath_chain_set_sha256"]
        or cath.get("splits_sha256") != mmseqs_identity["cath_splits_sha256"]
        or cath.get("release") != "CATH4.3"
    ):
        raise ValueError("approved Head epoch24/CATH4.3 source identity mismatch")
    summary = json.loads(head_paths["run_summary"].read_text())
    if int(summary.get("final_epoch", -1)) != 24:
        raise ValueError("Head run_summary does not prove fixed epoch24")
    if protocol_profile == "drb1501_production":
        import torch
        checkpoint = torch.load(head_paths["checkpoint"], map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, Mapping) else None
        expected_metadata = {
            "epoch": int(head.get("fixed_epoch", -1)),
            "global_step": int(head.get("checkpoint_global_step", -1)),
            "config_hash": str(head.get("checkpoint_config_hash") or ""),
            "manifest_version": str(head.get("checkpoint_manifest_version") or ""),
        }
        if (
            not isinstance(metadata, Mapping)
            or expected_metadata["epoch"] != 24
            or expected_metadata["global_step"] != 10055
            or not expected_metadata["config_hash"]
            or expected_metadata["manifest_version"] != "v1.1"
            or int(metadata.get("epoch", -1)) != expected_metadata["epoch"]
            or int(metadata.get("global_step", -1)) != expected_metadata["global_step"]
            or str(metadata.get("config_hash") or "") != expected_metadata["config_hash"]
            or str(metadata.get("manifest_version") or "")
            != expected_metadata["manifest_version"]
        ):
            raise ValueError("Head checkpoint metadata differs from approved epoch24 contract")
    return contract


def _dataset_manifest_identity(path: Path) -> tuple[str, str]:
    manifest = json.loads(path.read_text())
    unsigned = dict(manifest)
    manifest_sha = unsigned.pop("manifest_sha256", None)
    if manifest_sha != sha256_bytes(canonical_json(unsigned)):
        raise ValueError("referenced dataset manifest self-hash mismatch")
    root = path.parent
    observed = [
        {"path": str(item.relative_to(root)), "size_bytes": item.stat().st_size,
         "sha256": sha256_file(item)}
        for item in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root)))
        if item.is_file() and item.name != "dataset_manifest.json"
    ]
    if manifest.get("artifacts") != observed:
        raise ValueError("referenced dataset manifest artifact table mismatch")
    return str(manifest.get("dataset_release_id") or ""), str(manifest_sha)


DESCENDANT_SCOPE_SCHEMA = "if-benchmark-v3-drb1501-descendant-scope/1"
DESCENDANT_SCOPE_ROOTS = (
    ".", "if_ready/main", "if_ready/fast", "if_ready/pilot", "if_ready/highrisk",
    "if_ready/h_maps_v2", "h_maps", "wt_baselines", "generation_facades", "score_cache",
)


def _allele_tokens(allele_tag: str) -> tuple[str, ...]:
    normalized = str(allele_tag).lower()
    short = normalized.removeprefix("hla-")
    compact = short.replace("_", "")
    return normalized, short, compact


def scan_registered_descendants(
    *, warehouse_root: Path, allele_tag: str
) -> list[dict[str, Any]]:
    """Scan the frozen v3 set of derived-product roots for one allele only."""

    warehouse_root = Path(warehouse_root).resolve()
    tokens = _allele_tokens(allele_tag)
    observations: list[dict[str, Any]] = []
    for relative in DESCENDANT_SCOPE_ROOTS:
        root = warehouse_root if relative == "." else warehouse_root / relative
        candidates = (
            list(root.iterdir()) if relative == "." and root.is_dir()
            else list(root.rglob("*")) if root.is_dir() else []
        )
        files = [
            {"path": str(item.resolve().relative_to(warehouse_root)),
             "size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in candidates if item.is_file()
            and any(token in str(item.relative_to(root)).lower() for token in tokens)
        ]
        files.sort(key=lambda item: item["path"])
        observations.append({"root": relative, "root_exists": root.is_dir(), "allele_files": files})
    return observations


def write_descendant_inventory(
    path: Path, *, warehouse_root: Path, allele_tag: str,
    bindings: Sequence[Mapping[str, Any]] = (),
) -> Path:
    observations = scan_registered_descendants(
        warehouse_root=warehouse_root, allele_tag=allele_tag
    )
    scope_identity = {
        "schema": DESCENDANT_SCOPE_SCHEMA, "roots": list(DESCENDANT_SCOPE_ROOTS),
        "warehouse_root": str(Path(warehouse_root).resolve()), "allele_tag": allele_tag,
    }
    payload = {
        "schema_version": "if-benchmark-v3-descendant-inventory/2",
        "scope_identity": scope_identity,
        "scope_identity_sha256": sha256_bytes(canonical_json(scope_identity)),
        "observations": observations,
        "bindings": [dict(value) for value in bindings],
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def validate_descendant_inventory(
    path: Path, *, release_id: str, publication_context_path: Path
) -> None:
    """Rescan every registered allele-specific target and verify true release bindings."""

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "if-benchmark-v3-descendant-inventory/2":
        raise ValueError("descendant inventory schema mismatch")
    context = json.loads(Path(publication_context_path).read_text())
    warehouse_root = Path(context["warehouse_root"]).resolve()
    allele_tag = str(context["allele_tag"])
    expected_scope = {
        "schema": DESCENDANT_SCOPE_SCHEMA, "roots": list(DESCENDANT_SCOPE_ROOTS),
        "warehouse_root": str(warehouse_root), "allele_tag": allele_tag,
    }
    if (
        payload.get("scope_identity") != expected_scope
        or payload.get("scope_identity_sha256") != sha256_bytes(canonical_json(expected_scope))
    ):
        raise ValueError("descendant inventory scope is not the frozen DRB1501 contract")
    observed = scan_registered_descendants(
        warehouse_root=warehouse_root, allele_tag=allele_tag
    )
    if observed != payload.get("observations"):
        raise ValueError("descendant inventory no longer matches registered roots")
    observed_files = [item for root in observed for item in root["allele_files"]]
    bindings = payload.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(observed_files):
        raise ValueError("every registered descendant requires one explicit release binding")
    by_path = {
        str((warehouse_root / item["path"]).resolve()): item for item in observed_files
    }
    for binding in bindings:
        artifact_path = str(Path(binding["path"]).resolve())
        if artifact_path not in by_path:
            raise ValueError("descendant binding references an unscanned artifact")
        manifest_path = Path(binding["release_manifest_path"]).resolve()
        bound_id, bound_sha = _dataset_manifest_identity(manifest_path)
        if binding.get("dataset_release_id") != bound_id or binding.get("manifest_sha256") != bound_sha:
            raise ValueError("descendant release digest does not match referenced manifest")
        artifact = Path(artifact_path)
        if artifact.suffix == ".json":
            content = json.loads(artifact.read_text())
            content_ids = {str(content.get("dataset_release_id") or "")}
            content_shas = {
                str(content.get("dataset_manifest_sha256") or content.get("manifest_sha256") or "")
            }
        elif artifact.suffix == ".parquet":
            content = pd.read_parquet(artifact)
            if not {"dataset_release_id", "dataset_manifest_sha256"} <= set(content):
                raise ValueError("descendant parquet lacks release identity columns")
            content_ids = set(content["dataset_release_id"].astype(str))
            content_shas = set(content["dataset_manifest_sha256"].astype(str))
        else:
            raise ValueError("descendant inventory contains an unsupported uninspectable artifact")
        if content_ids != {bound_id} or content_shas != {bound_sha}:
            raise ValueError("descendant bytes do not bind the declared release digest")
        state = binding.get("state")
        if state == "bound_candidate" and bound_id != release_id:
            raise ValueError("candidate descendant binds the wrong release")
        if state not in {"bound_candidate", "pinned_prior"}:
            raise ValueError("descendant is unbound or has an invalid state")


def alias_identity(path: Path) -> list[Any] | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink():
        return ["symlink", os.readlink(path)]
    if path.is_file():
        return ["file", path.stat().st_size, sha256_file(path)]
    return ["directory", path.stat().st_dev, path.stat().st_ino]


def validate_publication_context(
    path: Path, *, release_id: str, candidate_root: Path
) -> None:
    """Verify destination absence, live alias snapshot, and recoverable prior manifest."""

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "if-benchmark-v3-publication-context/1":
        raise ValueError("publication context schema mismatch")
    if payload.get("release_id") != release_id:
        raise ValueError("publication context release ID mismatch")
    if payload.get("allele_tag") != "HLA-DRB1_15_01":
        raise ValueError("publication context is not the frozen DRB1501 allele target")
    warehouse_root = Path(payload["warehouse_root"]).resolve()
    candidate_root = Path(candidate_root).resolve()
    if (
        Path(payload.get("candidate_root") or "").resolve() != candidate_root
        or candidate_root.parent.parent != warehouse_root / "builds"
        or candidate_root.name != str(payload["allele_tag"])
    ):
        raise ValueError("publication warehouse scope is not anchored to the candidate build")
    releases_root = Path(payload["releases_root"]).resolve()
    if releases_root != warehouse_root / "releases":
        raise ValueError("publication releases_root differs from warehouse contract")
    destination = releases_root / release_id / str(payload["allele_tag"])
    if destination.exists() or destination.is_symlink():
        raise ValueError("immutable publication destination already exists")
    main_alias = Path(payload["allele_main_alias"])
    expected_alias = warehouse_root / "if_ready" / "main" / str(payload["allele_tag"])
    if main_alias.resolve(strict=False) != expected_alias.resolve(strict=False):
        raise ValueError("publication context uses the wrong allele-specific main alias")
    if alias_identity(main_alias) != payload.get("main_alias_identity"):
        raise ValueError("main alias changed since publication preflight")
    mode = payload.get("mode")
    if mode == "first_release":
        if (
            payload.get("main_alias_identity") is not None
            or payload.get("prior_release_manifest") is not None
            or payload.get("prior_manifest_sha256") is not None
            or os.path.lexists(main_alias)
        ):
            raise ValueError("first-release publication context unexpectedly has a prior target")
    elif mode == "existing_release":
        prior_manifest = Path(payload["prior_release_manifest"]).resolve()
        prior_id, prior_sha = _dataset_manifest_identity(prior_manifest)
        if not prior_id or prior_id == release_id:
            raise ValueError("publication context does not identify a distinct prior release")
        if prior_sha != payload.get("prior_manifest_sha256"):
            raise ValueError("publication prior manifest digest mismatch")
        if not main_alias.is_symlink() or main_alias.resolve() != prior_manifest.parent:
            raise ValueError("main alias does not resolve to the recoverable prior release")
    else:
        raise ValueError("publication context mode must be first_release or existing_release")
