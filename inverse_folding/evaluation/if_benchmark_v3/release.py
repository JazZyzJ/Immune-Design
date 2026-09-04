"""v3 row/cross-artifact validation and immutable atomic publication primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .leakage import validate_mmseqs_sidecar
from .schema import V3_ROW_COLUMNS
from .structures import validate_structure_download_registry
from .evidence import (
    load_release_evidence_bundle,
    validate_chain_collision_dedup_ledgers,
    validate_descendant_inventory,
    validate_family_evidence,
    validate_head_files,
    validate_nmp_evidence,
    validate_nmp_install_evidence,
    validate_approved_source_contract,
    validate_mmseqs_reference_evidence,
    validate_publication_context,
    validate_rcsb_source_manifest,
    validate_selection_replay,
    validate_tier1_source_manifest,
    validate_tier1_residue_map_snapshot,
    validate_tier1_release_sources,
    validate_tier1_internal_pool_replay,
    resolve_candidate_stage_path,
)


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_ROW_COLUMNS = V3_ROW_COLUMNS


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _list_value(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("membership field is not a list")
    return [str(item) for item in value]


def _validate_rows(
    primary: pd.DataFrame,
    diagnostic: pd.DataFrame,
    *,
    pdb_root: Path,
    release_id: str,
    allele: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    for label, frame in (("primary", primary), ("diagnostic", diagnostic)):
        missing = REQUIRED_ROW_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} rows missing v3 columns: {sorted(missing)}")
    combined = pd.concat([primary, diagnostic], ignore_index=True)
    if combined.empty:
        raise ValueError("release contains no rows")
    for column in ("protein_id", "selection_unit_id", "sequence_sha256", "pdb_path"):
        if combined[column].isna().any() or not combined[column].is_unique:
            raise ValueError(f"release requires unique non-null {column}")
    if combined["rcsb_entity_id"].isna().any() or not combined["rcsb_entity_id"].is_unique:
        raise ValueError("release requires one row per RCSB entity collision key")
    if set(combined["dataset_release_id"].astype(str)) != {release_id}:
        raise ValueError("row dataset_release_id mismatch")
    if set(combined["allele"].astype(str)) != {allele}:
        raise ValueError("row allele mismatch")
    if set(primary["evaluation_role"].astype(str)) != {"primary_generalization"}:
        raise ValueError("primary evaluation_role mismatch")
    if len(diagnostic) and set(diagnostic["evaluation_role"].astype(str)) != {
        "tier1_overlap_diagnostic"
    }:
        raise ValueError("diagnostic evaluation_role mismatch")

    row_by_id: dict[str, dict[str, Any]] = {}
    root = Path(pdb_root).resolve()
    for row in combined.to_dict("records"):
        pid = str(row["protein_id"])
        sequence = str(row["sequence"])
        if not sequence or not set(sequence) <= AA20:
            raise ValueError(f"row {pid} has non-AA20 sequence")
        if int(row["sequence_length"]) != len(sequence):
            raise ValueError(f"row {pid} sequence length mismatch")
        if row["sequence_sha256"] != _sha_bytes(sequence.encode()):
            raise ValueError(f"row {pid} sequence SHA mismatch")
        entity_sequence = str(row["entity_sequence"])
        if not entity_sequence or not set(entity_sequence) <= AA20:
            raise ValueError(f"row {pid} entity sequence is non-AA20")
        if row["entity_sequence_sha256"] != _sha_bytes(entity_sequence.encode()):
            raise ValueError(f"row {pid} entity sequence SHA mismatch")
        source_sequence = str(row["source_sequence"])
        if not source_sequence or not set(source_sequence) <= AA20:
            raise ValueError(f"row {pid} source sequence is non-AA20")
        if row["source_sequence_sha256"] != _sha_bytes(source_sequence.encode()):
            raise ValueError(f"row {pid} source sequence SHA mismatch")
        source_start = int(row["source_range_start_0b"])
        source_end = int(row["source_range_end_0b"])
        if not 0 <= source_start < source_end <= len(source_sequence):
            raise ValueError(f"row {pid} source range is invalid")
        mapping = json.loads(str(row["final_to_source_json"]))
        if (
            not isinstance(mapping, list) or len(mapping) != len(sequence)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in mapping)
            or len(mapping) != len(set(mapping))
            or any(int(a) >= int(b) for a, b in zip(mapping, mapping[1:]))
        ):
            raise ValueError(f"row {pid} final-to-source mapping mismatch")
        denominator = int(row["source_denominator_length"])
        if denominator != source_end - source_start:
            raise ValueError(f"row {pid} source denominator/range mismatch")
        if any(int(value) < source_start or int(value) >= source_end for value in mapping):
            raise ValueError(f"row {pid} final-to-source index outside source range")
        if any(sequence[index] != source_sequence[int(source)]
               for index, source in enumerate(mapping)):
            raise ValueError(f"row {pid} final sequence/source mapping mismatch")
        coverage = len(mapping) / denominator
        if denominator < 1 or abs(float(row["if_sequence_coverage"]) - coverage) > 1e-12:
            raise ValueError(f"row {pid} mapping coverage mismatch")
        if coverage < 0.8 or str(row["mapping_status"]) != "complete":
            raise ValueError(f"row {pid} mapping gate failed")
        path = (root / str(row["pdb_path"])).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"row {pid} structure path is not resolvable")
        if _sha_file(path) != row["structure_sha256"]:
            raise ValueError(f"row {pid} structure SHA mismatch")
        if "X-RAY" not in str(row["experimental_method"]).upper():
            raise ValueError(f"row {pid} is not X-ray")
        if not math.isfinite(float(row["resolution"])) or float(row["resolution"]) > 2.5:
            raise ValueError(f"row {pid} resolution gate failed")
        if not isinstance(row["cath_overlap_flag"], (bool, np.bool_)):
            raise ValueError(f"row {pid} CATH flag is not measured")
        memberships = _list_value(row["selected_tier_memberships"])
        _list_value(row["source_pool_memberships"])
        if memberships == ["tier1"]:
            try:
                sifts_range = json.loads(str(row["sifts_range_json"]))
                uniprot_spans = json.loads(str(row["uniprot_spans_json"]))
                chain_spans = json.loads(str(row["chain_spans_json"]))
                spans = json.loads(str(row["if_spans_json"]))
                residue_map = {
                    int(position): value for position, value in
                    json.loads(str(row["sifts_residue_map_json"])).items()
                }
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"row {pid} Tier1 projection JSON is invalid") from exc
            if (
                not isinstance(sifts_range, list) or len(sifts_range) != 2
                or [int(sifts_range[0]) - 1, int(sifts_range[1])] != [source_start, source_end]
                or denominator != int(sifts_range[1]) - int(sifts_range[0]) + 1
            ):
                raise ValueError(f"row {pid} Tier1 SIFTS range/denominator mismatch")
            for position, residue in residue_map.items():
                if (
                    position < source_start or position >= source_end
                    or not {"label_seq_id", "auth_seq_id", "insertion_code",
                            "chain_local_index_0b", "amino_acid"} <= set(residue)
                    or str(residue["amino_acid"]) != source_sequence[position]
                ):
                    raise ValueError(f"row {pid} Tier1 detailed SIFTS residue map mismatch")
            if (
                not all(isinstance(value, list) for value in (uniprot_spans, chain_spans, spans))
                or not len(uniprot_spans) == len(chain_spans) == len(spans)
                or len(spans) < 2
            ):
                raise ValueError(f"row {pid} Tier1 spans missing")
            source_to_final = {int(source): index for index, source in enumerate(mapping)}
            covered: set[int] = set()
            for uniprot_span, chain_span, span in zip(
                uniprot_spans, chain_spans, spans, strict=True
            ):
                unp_start, unp_end = int(uniprot_span["start_0b"]), int(uniprot_span["end_0b"])
                chain_start, chain_end = int(chain_span["start_0b"]), int(chain_span["end_0b"])
                start, end = int(span["start_0b"]), int(span["end_0b"])
                peptide = str(span["peptide"])
                source_positions = list(range(unp_start, unp_end))
                mapped_residues = [residue_map.get(position) for position in source_positions]
                if any(residue is None for residue in mapped_residues):
                    raise ValueError(f"row {pid} Tier1 detailed SIFTS residue mapping gap")
                if any(
                    not bool(residue.get("observed", True))
                    or residue.get("auth_seq_id") is None
                    for residue in mapped_residues
                ):
                    raise ValueError(f"row {pid} Tier1 span includes unresolved SIFTS residue")
                chain_indices = [int(residue["chain_local_index_0b"]) for residue in mapped_residues]
                expected_auth = [
                    {"auth_seq_id": int(residue["auth_seq_id"]),
                     "insertion_code": str(residue["insertion_code"])}
                    for residue in mapped_residues
                ]
                if (
                    str(uniprot_span["peptide"]) != peptide
                    or str(chain_span["peptide"]) != peptide
                    or not source_start <= unp_start < unp_end <= source_end
                    or source_sequence[unp_start:unp_end] != peptide
                    or chain_indices != list(range(chain_start, chain_end))
                    or chain_span.get("uniprot_positions_0b") != source_positions
                    or chain_span.get("label_seq_ids") != [
                        int(residue["label_seq_id"]) for residue in mapped_residues
                    ]
                    or chain_span.get("auth_residues") != expected_auth
                    or "".join(str(residue["amino_acid"]) for residue in mapped_residues) != peptide
                    or any(position not in source_to_final for position in range(unp_start, unp_end))
                    or [source_to_final[position] for position in range(unp_start, unp_end)]
                    != list(range(start, end))
                    or not 0 <= start < end <= len(sequence)
                    or sequence[start:end] != peptide
                ):
                    raise ValueError(f"row {pid} Tier1 IF-frame peptide mismatch")
                covered.update(range(start, end))
            epitope_coverage = len(covered) / len(sequence)
            if (
                int(row["verified_span_count"]) != len(spans)
                or int(row["verified_covered_residue_count"]) != len(covered)
                or abs(float(row["final_epitope_coverage"]) - epitope_coverage) > 1e-12
                or not 0.10 <= epitope_coverage <= 0.50
                or not row["source_uniprot_id"]
            ):
                raise ValueError(f"row {pid} Tier1 projection summary mismatch")
        elif memberships == ["tier2"]:
            if (
                source_sequence != entity_sequence
                or source_start != 0 or source_end != len(entity_sequence)
                or denominator != len(entity_sequence)
                or str(row["nmp_status"]) != "complete"
                or row["nmp_query_sequence_sha256"] != row["sequence_sha256"]
                or not math.isfinite(float(row["coverage_fraction"]))
                or not 0.0 <= float(row["coverage_fraction"]) <= 1.0
                or pd.isna(row["selection_bin"])
                or int(row["selection_bin"]) < 0 or int(row["selection_bin"]) > 19
                or not str(row["selection_reason"])
            ):
                raise ValueError(f"row {pid} Tier2 NMP/selection evidence missing")
        else:
            raise ValueError(f"row {pid} selected tier membership is invalid")
        if row["evaluation_role"] == "primary_generalization" and not row["family_cluster_id"]:
            raise ValueError(f"row {pid} family cluster missing")
        row_by_id[pid] = row
    return combined, row_by_id


def validate_release_tables(
    primary: pd.DataFrame,
    diagnostic: pd.DataFrame,
    cath_sidecar: pd.DataFrame,
    head_sidecar: pd.DataFrame,
    head_homology_sidecar: pd.DataFrame,
    load_coords_evidence: pd.DataFrame,
    *,
    evidence_bundle_path: Path,
    candidate_root: Path,
    pdb_root: Path,
    release_id: str,
    allele: str,
    tier1_target: int,
    tier2_target: int,
    load_coords_fn: Any | None = None,
) -> dict[str, str]:
    """Machine-check the thirteen release-gate groups."""

    combined, rows = _validate_rows(
        primary, diagnostic, pdb_root=pdb_root, release_id=release_id, allele=allele
    )
    bundle = load_release_evidence_bundle(
        evidence_bundle_path, candidate_root=Path(candidate_root)
    )
    expected_profile = (
        "drb1501_production" if (tier1_target, tier2_target) == (15, 3000)
        else "tiny_fixture" if (tier1_target, tier2_target) == (1, 1) else None
    )
    if expected_profile is None or bundle["protocol_profile"] != expected_profile:
        raise ValueError("release targets/protocol profile are not a frozen v3 contract")
    supplied_tables = {
        "primary": primary, "diagnostic": diagnostic, "cath": cath_sidecar,
        "head": head_sidecar, "head_homology": head_homology_sidecar,
        "load_coords": load_coords_evidence,
    }
    for label, supplied in supplied_tables.items():
        persisted = pd.read_parquet(bundle["release_tables"][label])
        try:
            pd.testing.assert_frame_equal(
                supplied.reset_index(drop=True), persisted.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(f"release {label} table differs from persisted candidate artifact") from exc
    mmseqs_identity = validate_mmseqs_reference_evidence(bundle["mmseqs"])
    nmp_install_identity = validate_nmp_install_evidence(bundle["nmp_install"])
    validate_approved_source_contract(
        manifest_path=bundle["approved_sources"]["manifest"],
        head_paths=bundle["head"], mmseqs_identity=mmseqs_identity,
        protocol_profile=bundle["protocol_profile"],
    )
    if tier2_target == 3000 and nmp_install_identity["identity_mode"] != "official_install_merkle":
        raise ValueError("production release requires full official NetMHCIIpan install Merkle")
    validate_rcsb_source_manifest(bundle["source"]["rcsb_manifest"])
    validate_tier1_source_manifest(bundle["source"]["tier1_manifest"])
    validate_tier1_residue_map_snapshot(bundle["source"]["tier1_residue_manifest"])
    validate_tier1_release_sources(
        bundle["source"]["tier1_manifest"], rows=rows, allele=allele,
        chain_attempt_path=bundle["ledgers"]["chain_attempts"],
        residue_manifest_path=bundle["source"]["tier1_residue_manifest"],
    )
    validate_structure_download_registry(
        bundle["ledgers"]["structure_downloads"], candidate_root=candidate_root,
        chain_attempt_path=bundle["ledgers"]["chain_attempts"],
    )
    validate_chain_collision_dedup_ledgers(bundle["ledgers"], rows=rows)
    validate_tier1_internal_pool_replay(
        bundle["ledgers"], rows=rows, target=tier1_target
    )
    if bool(primary["cath_overlap_flag"].any()):
        raise ValueError("gate 3 primary CATH overlap")

    expected_ids = set(rows)
    for label, sidecar in (
        ("CATH", cath_sidecar), ("Head", head_sidecar),
        ("Head homology", head_homology_sidecar),
        ("load_coords", load_coords_evidence)
    ):
        if "protein_id" not in sidecar or not sidecar["protein_id"].is_unique:
            raise ValueError(f"{label} sidecar key invalid")
        if set(sidecar["protein_id"].astype(str)) != expected_ids:
            raise ValueError(f"{label} sidecar count/key mismatch")

    if load_coords_fn is None:
        try:
            from byprot.utils.io import load_coords as load_coords_fn
        except ImportError as exc:
            raise RuntimeError("registered DPLM load_coords is unavailable") from exc
    for load_row in load_coords_evidence.to_dict("records"):
        row = rows[str(load_row["protein_id"])]
        if (
            load_row["status"] != "pass"
            or load_row["sequence_sha256"] != row["sequence_sha256"]
            or load_row["loaded_sequence_sha256"] != row["sequence_sha256"]
            or int(load_row["loaded_length"]) != int(row["sequence_length"])
            or load_row["structure_sha256"] != row["structure_sha256"]
        ):
            raise ValueError("gate 4 load_coords/mapping evidence mismatch")
        clean_path = (Path(pdb_root).resolve() / str(row["pdb_path"])).resolve()
        loaded = load_coords_fn(str(clean_path), chain="A")
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("gate 4 registered load_coords returned an invalid result")
        coords, loaded_sequence = loaded
        if (
            str(loaded_sequence) != str(row["sequence"])
            or len(coords) != int(row["sequence_length"])
            or hashlib.sha256(str(loaded_sequence).encode()).hexdigest()
            != str(row["sequence_sha256"])
        ):
            raise ValueError("gate 4 registered load_coords replay differs from release bytes")

    if len(diagnostic):
        if not diagnostic["cath_overlap_flag"].all():
            raise ValueError("gate 5 diagnostic row is not a measured CATH hit")
        if any(_list_value(value) != ["tier1"] for value in diagnostic["selected_tier_memberships"]):
            raise ValueError("gate 5 diagnostic contains non-Tier1 row")

    primary_t1 = sum(
        _list_value(value) == ["tier1"] for value in primary["selected_tier_memberships"]
    )
    primary_t2 = sum(
        _list_value(value) == ["tier2"] for value in primary["selected_tier_memberships"]
    )
    if primary_t1 != tier1_target or primary_t2 != tier2_target:
        raise ValueError(
            f"gate 7 target counts mismatch: Tier1={primary_t1}/{tier1_target}, "
            f"Tier2={primary_t2}/{tier2_target}"
        )
    validate_nmp_evidence(
        bundle["nmp"], c7_eligible_path=bundle["selection"]["c7_eligible"],
        rows=rows, allele=allele, install_identity=nmp_install_identity,
    )
    validate_selection_replay(
        bundle["selection"], ledger_paths=bundle["ledgers"], primary=primary,
        tier2_target=tier2_target, allele=allele,
        rcsb_manifest_path=bundle["source"]["rcsb_manifest"],
        mmseqs_identity=mmseqs_identity, candidate_root=Path(candidate_root),
        nmp_install_identity=nmp_install_identity,
        protocol_profile=bundle["protocol_profile"],
    )

    head_identity: set[tuple[str, str]] = set()
    for head_row in head_sidecar.to_dict("records"):
        row = rows[str(head_row["protein_id"])]
        if (
            head_row["dataset_release_id"] != release_id
            or head_row["sequence_sha256"] != row["sequence_sha256"]
            or head_row["status"] != "complete"
        ):
            raise ValueError("gate 9 Head row identity mismatch")
        head_identity.add(
            (str(head_row["head_checkpoint_sha256"]), str(head_row["head_config_sha256"]))
        )
    if len(head_identity) != 1 or any(len(value) != 64 for value in next(iter(head_identity))):
        raise ValueError("gate 9 mixed/invalid Head identity")
    head_checkpoint_sha256 = next(iter(head_identity))[0]
    for homology_row in head_homology_sidecar.to_dict("records"):
        row = rows[str(homology_row["protein_id"])]
        if (
            homology_row.get("dataset_release_id") != release_id
            or homology_row.get("sequence_sha256") != row["sequence_sha256"]
            or homology_row.get("query_sequence_sha256") != row["sequence_sha256"]
            or homology_row.get("head_checkpoint_sha256") != head_checkpoint_sha256
        ):
            raise ValueError("gate 9 Head-homology annotation identity mismatch")
    validate_head_files(
        bundle["head"], head_sidecar, protocol_profile=bundle["protocol_profile"]
    )

    cath_by_id = cath_sidecar.set_index("protein_id")
    for pid, row in rows.items():
        cath_row = cath_by_id.loc[pid]
        if (
            cath_row["query_sequence_sha256"] != row["sequence_sha256"]
            or bool(cath_row["overlap_flag"]) != bool(row["cath_overlap_flag"])
            or (
                None if pd.isna(cath_row["best_target"]) else str(cath_row["best_target"])
            ) != (None if pd.isna(row["cath_best_target"]) else str(row["cath_best_target"]))
            or (
                (pd.isna(cath_row["best_identity"]) and not pd.isna(row["cath_best_identity"]))
                or (not pd.isna(cath_row["best_identity"]) and pd.isna(row["cath_best_identity"]))
                or (
                    not pd.isna(cath_row["best_identity"])
                    and float(cath_row["best_identity"]) != float(row["cath_best_identity"])
                )
            )
            or cath_row["search_status"] not in {"complete_hit", "complete_no_hit"}
            or any(not isinstance(cath_row[field], str) or len(cath_row[field]) != 64
                   for field in ("reference_sha256", "tool_sha256", "parameters_sha256"))
        ):
            raise ValueError("gate 10 CATH evidence identity/flag mismatch")

    queries = combined[["protein_id", "sequence"]].copy()
    for search_kind, sidecar, expected_cov_mode in (
        ("cath", cath_sidecar, 0), ("head", head_homology_sidecar, 2)
    ):
        identity_columns = (
            "reference_sha256", "tool_sha256", "tool_version", "cov_mode", "coverage",
            "min_seq_id", "raw_output_path", "query_fasta_path",
        )
        identities = sidecar[list(identity_columns)].drop_duplicates()
        if len(identities) != 1:
            raise ValueError(f"gate 10 mixed {search_kind} search identity")
        identity = identities.iloc[0]
        if int(identity["cov_mode"]) != expected_cov_mode:
            raise ValueError(f"gate 10 {search_kind} cov_mode mismatch")
        raw_output_path = resolve_candidate_stage_path(
            identity["raw_output_path"], candidate_root=Path(candidate_root),
            label=f"gate 10 {search_kind} raw output",
        )
        query_fasta_path = resolve_candidate_stage_path(
            identity["query_fasta_path"], candidate_root=Path(candidate_root),
            label=f"gate 10 {search_kind} query FASTA",
        )
        expected_reference = (
            mmseqs_identity["cath_reference_sha256"]
            if search_kind == "cath" else mmseqs_identity["head_reference_sha256"]
        )
        if (
            str(identity["reference_sha256"]) != expected_reference
            or str(identity["tool_sha256"]) != mmseqs_identity["tool_sha256"]
            or str(identity["tool_version"]) != mmseqs_identity["tool_version"]
        ):
            raise ValueError(f"gate 10 {search_kind} actual reference/tool identity mismatch")
        expected_targets = set(
            mmseqs_identity[
                "cath_reference_ids" if search_kind == "cath" else "head_reference_ids"
            ]
        )
        if not set(sidecar["best_target"].dropna().astype(str)) <= expected_targets:
            raise ValueError(f"gate 10 {search_kind} target is absent from actual reference")
        validate_mmseqs_sidecar(
            queries,
            sidecar,
            raw_tsv_path=raw_output_path,
            query_fasta_path=query_fasta_path,
            search_kind=search_kind,
            reference_sha256=str(identity["reference_sha256"]),
            tool_sha256=str(identity["tool_sha256"]),
            tool_version=str(identity["tool_version"]),
            cov_mode=int(identity["cov_mode"]),
            coverage=float(identity["coverage"]),
            min_seq_id=float(identity["min_seq_id"]),
            path_root=Path(candidate_root),
        )

    validate_family_evidence(
        bundle["family"], primary=primary, diagnostic=diagnostic,
        mmseqs_identity=mmseqs_identity,
        protocol_profile=bundle["protocol_profile"],
    )
    validate_publication_context(
        bundle["publication"]["context"], release_id=release_id,
        candidate_root=Path(candidate_root),
    )
    validate_descendant_inventory(
        bundle["descendants"]["inventory"], release_id=release_id,
        publication_context_path=bundle["publication"]["context"],
    )
    # Reaching here also establishes gate 6 (Tier2 NMP), gate 11 (row/cross-artifact schema),
    # and all remaining per-row conditions checked above.
    return {str(number): "pass" for number in range(1, 14)}


def _artifact_table(candidate_dir: Path) -> list[dict[str, Any]]:
    files = [
        path for path in candidate_dir.rglob("*")
        if path.is_file() and path.name != "dataset_manifest.json"
    ]
    return [
        {
            "path": str(path.relative_to(candidate_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in sorted(files, key=lambda path: str(path.relative_to(candidate_dir)))
    ]


def write_dataset_manifest(
    candidate_dir: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)
    if "manifest_sha256" in payload or "artifacts" in payload:
        raise ValueError("manifest payload cannot override hash/artifact fields")
    manifest = dict(payload)
    manifest["schema_version"] = "if-benchmark-v3-release-manifest/1"
    manifest["artifacts"] = _artifact_table(candidate_dir)
    manifest["manifest_sha256"] = _sha_bytes(_canonical(manifest))
    path = candidate_dir / "dataset_manifest.json"
    temporary = candidate_dir / ".dataset_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return manifest


def verify_dataset_manifest(candidate_dir: Path) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    path = candidate_dir / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected != _sha_bytes(_canonical(unsigned)):
        raise ValueError("dataset manifest self-hash mismatch")
    if manifest.get("artifacts") != _artifact_table(candidate_dir):
        raise ValueError("dataset manifest artifact table/hash mismatch")
    return manifest


def _alias_identity(path: Path | None) -> tuple[Any, ...] | None:
    if path is None or not os.path.lexists(path):
        return None
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_file():
        return ("file", path.stat().st_size, _sha_file(path))
    return ("directory", path.stat().st_dev, path.stat().st_ino)


def finalize_candidate_directory(
    candidate_dir: Path,
    *,
    releases_root: Path,
    release_id: str,
    allele_tag: str,
    main_alias: Path | None = None,
) -> Path:
    """Atomically rename a verified candidate; never repoint the canonical alias."""

    candidate_dir = Path(candidate_dir)
    releases_root = Path(releases_root)
    manifest = verify_dataset_manifest(candidate_dir)
    if str(manifest.get("dataset_release_id") or "") != str(release_id):
        raise ValueError("finalization release ID differs from packaged dataset identity")
    destination = releases_root / release_id / allele_tag
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    alias_before = _alias_identity(main_alias)
    os.replace(candidate_dir, destination)
    verify_dataset_manifest(destination)
    if _alias_identity(main_alias) != alias_before:
        raise RuntimeError("main alias changed during candidate finalization")
    return destination
