"""Registered scientific stage producers for the DRB1*15:01 v3 build.

This module is intentionally separate from the generic journal/SLURM machinery.  Every producer
accepts only the frozen DAG plus its canonical array index, discovers predecessor bytes through
sealed task payloads, and writes one self-hashed ``stage_result.json`` inside its task directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextvars import ContextVar
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .evidence import (
    EVIDENCE_KEYS,
    FAILED_C5_NMP_KEYS,
    FAILED_C5_SELECTION_KEYS,
    alias_identity,
    canonical_json,
    sha256_file,
    stable_family_cluster_id,
    write_c5_prior_rung_index,
    write_failed_c5_attempt_bundle,
    write_descendant_inventory,
    write_release_evidence_bundle,
    validate_tier1_source_manifest,
)
from .head_annotation import _path_identity as _head_path_identity
from .head_annotation import annotate_fixed_epoch_head
from .leakage import build_mmseqs_sidecar, run_mmseqs_easy_search
from .mapping import materialize_mmcif_chain
from .nmp import (
    load_nmp_shard_manifest,
    make_standalone_score_shard_fn,
    merge_nmp_sharded_evidence,
    nmp_parameter_payload,
    write_nmp_sharded_evidence,
)
from .production import (
    ALLELE, APPROVED_LOGICAL_MANIFESTS,
    STAGE_BY_NAME,
    _stage_root,
    _validate_file_identity,
    production_status,
    validate_production_dag,
)
from .preflight import _default_request
from .rcsb_snapshot import snapshot_rcsb_entities
from .references import materialize_cath_train_reference, materialize_head_seen_reference
from .release import (
    validate_release_tables,
    verify_dataset_manifest,
    write_dataset_manifest,
)
from .selection import binned_sample_to_dict, deterministic_binned_sample
from .sifts_residue import frozen_revision_map, snapshot_tier1_residue_maps
from .structures import (
    RcsbMmcifAcquisitionError,
    acquire_rcsb_mmcif,
    write_structure_download_registry,
)
from .tier1 import (
    finalize_tier1_pool,
    project_verified_spans,
    resolve_tier1_tier2_collisions,
)


STAGE_RESULT_SCHEMA = "if-benchmark-v3-production-stage-result/1"
_ACTIVE_ADAPTERS: ContextVar[Mapping[str, Any]] = ContextVar(
    "if_benchmark_v3_stage_adapters", default={}
)


def _adapter(name: str, default: Any = None) -> Any:
    return _ACTIVE_ADAPTERS.get().get(name, default)


def _candidate_annotation_id(dag: Mapping[str, Any]) -> str:
    return f"candidate-{dag['build_id']}"


def _protocol_profile(dag: Mapping[str, Any]) -> str:
    return (
        "drb1501_production"
        if dag.get("targets") == {"tier1": 15, "tier2": 3000}
        else "tiny_fixture" if dag.get("targets") == {"tier1": 1, "tier2": 1}
        else "invalid"
    )


def _prior_candidate_roots(dag: Mapping[str, Any]) -> list[tuple[int, Path]]:
    return [
        (int(record["target"]), Path(str(record["candidate_root"])).resolve())
        for record in dag.get("prior_rung_snapshots") or []
    ]


def _sha(path: Path) -> str:
    return sha256_file(Path(path))


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _input_path(dag: Mapping[str, Any], role: str) -> Path:
    return _validate_file_identity(
        dag["immutable_inputs"][role], f"production stage input {role}"
    )


def _relative(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _file_table(root: Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    return [
        {
            "path": _relative(path, root),
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root)))
        if path.is_file() and path.name != "stage_result.json"
    ]


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _dependency_payloads(dag: Mapping[str, Any], stage: str) -> list[Path]:
    payloads: list[Path] = []
    for dependency in STAGE_BY_NAME[stage].dependencies:
        row = next(item for item in dag["stages"] if item["name"] == dependency)
        for index in range(int(row["task_count"])):
            payload = _stage_root(dag, dependency) / f"{index:05d}" / "payload"
            result = payload / "stage_result.json"
            if not result.is_file():
                raise FileNotFoundError(result)
            validate_stage_result(
                result, dag=dag, expected_stage=dependency,
                expected_index=index, deep=False,
            )
            payloads.append(payload)
    return payloads


def _stage_payloads(dag: Mapping[str, Any], stage: str) -> list[Path]:
    row = next(item for item in dag["stages"] if item["name"] == stage)
    payloads = []
    for index in range(int(row["task_count"])):
        payload = _stage_root(dag, stage) / f"{index:05d}" / "payload"
        validate_stage_result(
            payload / "stage_result.json", dag=dag,
            expected_stage=stage, expected_index=index, deep=False,
        )
        payloads.append(payload)
    return payloads


def validate_stage_result(
    path: Path, *, dag: Mapping[str, Any], expected_stage: str, expected_index: int,
    deep: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("result_sha256", None)
    root = path.parent
    if (
        payload.get("schema_version") != STAGE_RESULT_SCHEMA
        or digest != hashlib.sha256(canonical_json(unsigned)).hexdigest()
        or payload.get("dag_sha256") != dag["dag_sha256"]
        or payload.get("stage") != expected_stage
        or int(payload.get("task_index", -1)) != int(expected_index)
        or payload.get("status") != "complete"
    ):
        raise ValueError("production stage result identity mismatch")
    observed = payload.get("files")
    if not isinstance(observed, list) or observed != sorted(observed, key=lambda row: row["path"]):
        raise ValueError("production stage result file table is noncanonical")
    if deep and observed != _file_table(root):
        raise ValueError("production stage result file table changed")
    return payload


def _write_result(
    output_dir: Path, *, dag: Mapping[str, Any], stage: str, task_index: int,
    summary: Mapping[str, Any],
) -> Path:
    payload = {
        "schema_version": STAGE_RESULT_SCHEMA,
        "dag_sha256": dag["dag_sha256"],
        "stage": stage,
        "task_index": int(task_index),
        "status": "complete",
        "summary": dict(summary),
        "files": _file_table(output_dir),
    }
    payload["result_sha256"] = _self_hash(payload, "result_sha256")
    path = Path(output_dir) / "stage_result.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    validate_stage_result(
        path, dag=dag, expected_stage=stage, expected_index=int(task_index)
    )
    return path


def _copy_self_contained_tier1_preflight(
    source_manifest: Path, destination: Path, *, expected_logical_sha256: str | None,
) -> Path:
    """Copy the approved preflight and internalize its three formerly external inputs."""

    source_manifest = Path(source_manifest).resolve()
    source_root = source_manifest.parent
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source_root, destination)
    manifest_path = destination / source_manifest.name
    manifest = json.loads(manifest_path.read_text())
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256", None)
    if recorded != hashlib.sha256(canonical_json(unsigned)).hexdigest():
        raise ValueError("approved Tier1 preflight self-hash mismatch")
    if expected_logical_sha256 is not None and recorded != expected_logical_sha256:
        raise ValueError("Tier1 preflight is not the approved strict-full DRB1501 snapshot")
    input_dir = destination / "candidate_inputs"
    input_dir.mkdir()
    for label, record in manifest["input_files"].items():
        raw_source = Path(str(record["path"]))
        source = (
            raw_source.resolve() if raw_source.is_absolute()
            else (source_root / raw_source).resolve()
        )
        if (
            not source.is_file() or source.stat().st_size != int(record["size_bytes"])
            or _sha(source) != record["sha256"]
        ):
            raise ValueError(f"Tier1 preflight input changed: {label}")
        target = input_dir / f"{label}{source.suffix}"
        shutil.copyfile(source, target)
        record["path"] = str(target.relative_to(destination))
    manifest.pop("manifest_sha256", None)
    manifest["approved_upstream_manifest_sha256"] = recorded
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_tier1_source_manifest(manifest_path)
    return manifest_path


def _produce_c1(dag: Mapping[str, Any], output_dir: Path, _index: int) -> dict[str, Any]:
    source_dir = output_dir / "rcsb_source"
    prior = _prior_candidate_roots(dag)
    if prior:
        source = (
            prior[-1][1] / "audit" / "orchestrator" / "stages" /
            "c1_rcsb_source" / "00000" / "payload" / "rcsb_source"
        )
        shutil.copytree(source, source_dir)
        manifest = json.loads((source_dir / "source_manifest.json").read_text())
        return {
            "rcsb_manifest": _relative(source_dir / "source_manifest.json", output_dir),
            **{key: int(value) for key, value in manifest["summary"].items()},
            "reused_prior_rung": prior[-1][0],
        }
    kwargs = {"output_dir": source_dir, "api_delay_s": 0.1}
    if _adapter("rcsb_transport") is not None:
        kwargs.update({
            "transport": _adapter("rcsb_transport"),
            "sleep_fn": _adapter("sleep_fn", lambda _seconds: None),
        })
    summary = snapshot_rcsb_entities(**kwargs)
    return {
        "rcsb_manifest": _relative(source_dir / "source_manifest.json", output_dir),
        **{key: int(value) for key, value in summary.items()},
    }


def _produce_references(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    refs = output_dir / "references"
    refs.mkdir()
    cath_fasta = refs / "cath_train.fasta"
    cath_manifest = refs / "cath_train.manifest.json"
    head_fasta = refs / "head_seen.fasta"
    head_manifest = refs / "head_seen.manifest.json"
    materialize_cath_train_reference(
        chain_set_jsonl=_input_path(dag, "cath_chain_set"),
        splits_json=_input_path(dag, "cath_splits"),
        output_fasta=cath_fasta, manifest_path=cath_manifest,
    )
    materialize_head_seen_reference(
        split_id_paths={
            "train": _input_path(dag, "strict_train_ids"),
            "val": _input_path(dag, "strict_val_ids"),
            "test": _input_path(dag, "strict_test_ids"),
        },
        protein_samples_path=_input_path(dag, "protein_samples"), allele=ALLELE,
        output_fasta=head_fasta, manifest_path=head_manifest,
    )
    tool = _input_path(dag, "mmseqs_binary")
    version = subprocess.run(
        [str(tool), "version"], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise RuntimeError("MMseqs version probe failed while materializing references")
    identity = refs / "mmseqs_identity.json"
    identity.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-mmseqs-tool/1",
        "tool_sha256": _sha(tool), "tool_version": version.stdout.strip(),
    }, indent=2, sort_keys=True) + "\n")
    return {
        "cath_reference_manifest": _relative(cath_manifest, output_dir),
        "cath_reference_fasta": _relative(cath_fasta, output_dir),
        "head_reference_manifest": _relative(head_manifest, output_dir),
        "head_reference_fasta": _relative(head_fasta, output_dir),
        "mmseqs_identity": _relative(identity, output_dir),
    }


def _produce_c8_source(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    prior = _prior_candidate_roots(dag)
    if prior:
        source = (
            prior[-1][1] / "audit" / "orchestrator" / "stages" /
            "c8_sifts_residue_source" / "00000" / "payload"
        )
        shutil.copytree(source / "tier1_preflight", output_dir / "tier1_preflight")
        shutil.copytree(source / "tier1_residue_maps", output_dir / "tier1_residue_maps")
        if (source / "coordinate_cache").is_dir():
            shutil.copytree(source / "coordinate_cache", output_dir / "coordinate_cache")
        return {
            "tier1_manifest": "tier1_preflight/preflight_manifest.json",
            "tier1_residue_manifest": "tier1_residue_maps/manifest.json",
            "reused_prior_rung": prior[-1][0],
        }
    candidate_root = Path(dag["candidate_root"]).resolve()
    preflight = _copy_self_contained_tier1_preflight(
        _input_path(dag, "tier1_preflight_manifest"), output_dir / "tier1_preflight",
        expected_logical_sha256=(
            APPROVED_LOGICAL_MANIFESTS["tier1_preflight_manifest"]
            if _protocol_profile(dag) == "drb1501_production" else None
        ),
    )
    preflight_root = preflight.parent
    source_attempts = pd.read_parquet(preflight_root / "tier1_sifts_chain_attempts.parquet")
    canonical_sources = pd.read_parquet(
        preflight_root / "tier1_canonical_source_units.parquet"
    )
    coordinate_cache = output_dir / "coordinate_cache"

    def coordinate_provider(pdb_id: str, revision: str) -> tuple[Path, Path]:
        cif, _manifest = acquire_rcsb_mmcif(
            pdb_id=pdb_id, expected_revision_date=revision,
            cache_dir=coordinate_cache, api_delay_s=0.1,
            **({
                "request_fn": _adapter("rcsb_cif_request"),
                "sleep_fn": _adapter("sleep_fn", lambda _seconds: None),
            } if _adapter("rcsb_cif_request") is not None else {}),
        )
        return cif, cif.parent / "manifest.json"

    # Frozen revision authority: the sealed C1 RCSB entity snapshot, never the PDBe file.
    c1_source = _stage_payloads(dag, "c1_rcsb_source")[0]
    frozen_revisions = frozen_revision_map(
        pd.read_parquet(c1_source / "rcsb_source" / "entities_all.parquet")
    )
    residue_manifest = snapshot_tier1_residue_maps(
        source_attempts=source_attempts, canonical_sources=canonical_sources,
        output_dir=output_dir / "tier1_residue_maps", candidate_root=candidate_root,
        coordinate_provider=coordinate_provider,
        request_fn=_adapter("pdbe_residue_request", _default_request),
        sleep_fn=_adapter("sleep_fn", __import__("time").sleep),
        api_delay_s=0.1, frozen_revisions=frozen_revisions,
        min_tier1_source_units=int(dag["targets"]["tier1"]),
        command_argv=["registered-production-stage", "c8_sifts_residue_source"],
    )
    return {
        "tier1_manifest": _relative(preflight, output_dir),
        "tier1_residue_manifest": _relative(residue_manifest, output_dir),
    }


def _write_fasta(frame: pd.DataFrame, path: Path, *, key: str, sequence: str) -> None:
    ordered = frame.sort_values(key, kind="stable")
    path.write_text("".join(
        f">{row[key]}\n{row[sequence]}\n" for row in ordered.to_dict("records")
    ))


def _produce_c4_source(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    prior = _prior_candidate_roots(dag)
    if prior:
        source = (
            prior[-1][1] / "audit" / "orchestrator" / "stages" /
            "c4_source_cath" / "00000" / "payload"
        )
        for item in source.iterdir():
            destination = output_dir / item.name
            shutil.copytree(item, destination) if item.is_dir() else shutil.copyfile(
                item, destination
            )
        return {
            "query_count": len(pd.read_parquet(output_dir / "source_cath_queries.parquet")),
            "source_clean_count": int((~pd.read_parquet(
                output_dir / "source_cath_sidecar.parquet"
            )["overlap_flag"]).sum()),
            "queries": "source_cath_queries.parquet",
            "sidecar": "source_cath_sidecar.parquet",
            "mmseqs_manifest": "mmseqs/manifest.json",
            "reused_prior_rung": prior[-1][0],
        }
    c1 = _stage_payloads(dag, "c1_rcsb_source")[0]
    refs = _stage_payloads(dag, "references")[0]
    groups = pd.read_parquet(c1 / "rcsb_source" / "sequence_groups.parquet")
    queries = groups[[
        "selection_unit_id", "entity_sequence", "entity_sequence_sha256"
    ]].rename(columns={
        "selection_unit_id": "protein_id", "entity_sequence": "sequence",
        "entity_sequence_sha256": "sequence_sha256",
    }).sort_values("protein_id", kind="stable")
    query_table = output_dir / "source_cath_queries.parquet"
    query_fasta = output_dir / "source_cath_queries.fasta"
    queries.to_parquet(query_table, index=False)
    _write_fasta(queries, query_fasta, key="protein_id", sequence="sequence")
    reference = refs / "references" / "cath_train.fasta"
    search = run_mmseqs_easy_search(
        tool_path=_input_path(dag, "mmseqs_binary"), query_fasta_path=query_fasta,
        reference_fasta_path=reference, output_dir=output_dir / "mmseqs",
        cov_mode=0, coverage=0.8, min_seq_id=0.3,
        threads=int(next(
            row for row in dag["stages"] if row["name"] == "c4_source_cath"
        )["resources"]["cpus"]),
    )
    run = json.loads(search["manifest"].read_text())
    sidecar = build_mmseqs_sidecar(
        queries[["protein_id", "sequence"]], raw_tsv_path=search["raw_output"],
        query_fasta_path=query_fasta, search_kind="cath",
        reference_sha256=_sha(reference), tool_sha256=_sha(_input_path(dag, "mmseqs_binary")),
        tool_version=str(run["tool_version"]), cov_mode=0, coverage=0.8,
        min_seq_id=0.3, path_root=Path(dag["candidate_root"]),
    )
    sidecar_path = output_dir / "source_cath_sidecar.parquet"
    sidecar.to_parquet(sidecar_path, index=False)
    return {
        "query_count": len(queries),
        "source_clean_count": int((~sidecar["overlap_flag"]).sum()),
        "queries": query_table.name, "sidecar": sidecar_path.name,
        "mmseqs_manifest": _relative(search["manifest"], output_dir),
    }


def _contiguous_partition(frame: pd.DataFrame, *, index: int, count: int) -> pd.DataFrame:
    """Split contiguously and EVENLY across the array.

    A `ceil(n/count)` chunk leaves the tail shards empty whenever `n` is not close to a multiple
    of `count` -- 4,453 C7 queries over 192 shards put shard 186's start at 4,464, so the last six
    shards were empty and the stage could not seal even though shards 0-185 had covered every
    query. Distributing the remainder over the first `n % count` shards makes every shard non-empty
    whenever `count <= n`, so array cardinality never has to be hand-tuned to a dataset's size.
    """

    total = len(frame)
    count, index = int(count), int(index)
    if total == 0:
        raise ValueError("array source table is empty")
    base, remainder = divmod(total, count)
    start = index * base + min(index, remainder)
    stop = start + base + (1 if index < remainder else 0)
    result = frame.iloc[start:stop].copy()
    if result.empty:
        # only reachable when the array is larger than the work itself
        raise ValueError("array cardinality creates an empty registered task")
    return result


def _nmp_parameters(dag: Mapping[str, Any], *, stage: str, lengths: list[int]) -> dict[str, Any]:
    runner = dag["stage_parameters"]["nmp"]
    return nmp_parameter_payload(
        stage=stage, allele=ALLELE, peptide_lengths=lengths,
        install_manifest_sha256=_sha(_input_path(dag, "nmp_install_manifest")),
        batch_size=int(runner["batch_size"]),
        subprocess_timeout_s=int(runner["subprocess_timeout_s"]),
        max_lengths_per_call=int(runner["max_lengths_per_call"]),
        n_workers=int(runner["n_workers"]),
    )


def _produce_c5_nmp(
    dag: Mapping[str, Any], output_dir: Path, index: int,
) -> dict[str, Any]:
    prior = _prior_candidate_roots(dag)
    if prior:
        marker = output_dir / "reused_source_nmp.json"
        marker.write_text(json.dumps({
            "schema_version": "if-benchmark-v3-c5-source-nmp-reuse/1",
            "prior_rung": prior[-1][0], "task_index": int(index),
        }, indent=2, sort_keys=True) + "\n")
        return {"reused_prior_rung": prior[-1][0], "marker": marker.name}
    source = _stage_payloads(dag, "c4_source_cath")[0]
    queries = pd.read_parquet(source / "source_cath_queries.parquet")
    sidecar = pd.read_parquet(source / "source_cath_sidecar.parquet").set_index("protein_id")
    clean = queries[
        queries["protein_id"].map(lambda value: not bool(sidecar.loc[value, "overlap_flag"]))
    ].rename(columns={
        "protein_id": "selection_unit_id", "sequence": "source_sequence",
        "sequence_sha256": "source_sequence_sha256",
    }).sort_values("selection_unit_id", kind="stable").reset_index(drop=True)
    count = int(next(row for row in dag["stages"] if row["name"] == "c5_nmp")["task_count"])
    subset = _contiguous_partition(clean, index=index, count=count)
    query_path = output_dir / "queries.parquet"
    subset.to_parquet(query_path, index=False)
    parameters = output_dir / "parameters.json"
    parameters.write_text(json.dumps(
        _nmp_parameters(dag, stage="c5", lengths=[15]), indent=2, sort_keys=True
    ) + "\n")
    binary = _input_path(dag, "nmp_binary")
    scorer_factory = _adapter("nmp_score_shard_factory", make_standalone_score_shard_fn)
    scorer = scorer_factory(
        binary_path=binary, allele=ALLELE, peptide_lengths=[15],
        parameters_path=parameters, key_column="selection_unit_id",
        sequence_column="source_sequence",
    )
    manifest = write_nmp_sharded_evidence(
        output_dir=output_dir / "evidence", queries=subset,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=binary, parameters_path=parameters, score_shard_fn=scorer,
        shard_query_count=len(subset),
    )
    return {
        "query_count": len(subset), "query_table": query_path.name,
        "parameters": parameters.name, "manifest": _relative(manifest, output_dir),
    }


def _produce_c5_select(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    source = _stage_payloads(dag, "c4_source_cath")[0]
    task_payloads = _stage_payloads(dag, "c5_nmp")
    source_queries = pd.read_parquet(source / "source_cath_queries.parquet")
    source_cath = pd.read_parquet(source / "source_cath_sidecar.parquet").set_index("protein_id")
    queries = source_queries[
        source_queries["protein_id"].map(
            lambda value: not bool(source_cath.loc[value, "overlap_flag"])
        )
    ].rename(columns={
        "protein_id": "selection_unit_id", "sequence": "source_sequence",
        "sequence_sha256": "source_sequence_sha256",
    }).sort_values("selection_unit_id", kind="stable").reset_index(drop=True)
    parameters = output_dir / "parameters.json"
    binary = _input_path(dag, "nmp_binary")
    prior = _prior_candidate_roots(dag)
    if prior:
        prior_c5 = (
            prior[-1][1] / "audit" / "orchestrator" / "stages" /
            "c5_select" / "00000" / "payload"
        )
        shutil.copyfile(prior_c5 / "parameters.json", parameters)
        shutil.copytree(prior_c5 / "nmp_evidence", output_dir / "nmp_evidence")
        merged = output_dir / "nmp_evidence" / "manifest.json"
    else:
        parameter_shas = {_sha(path / "parameters.json") for path in task_payloads}
        if len(parameter_shas) != 1:
            raise ValueError("C5 NMP array tasks used mixed parameter bytes")
        shutil.copyfile(task_payloads[0] / "parameters.json", parameters)
        merged = merge_nmp_sharded_evidence(
            source_manifests=[path / "evidence" / "manifest.json" for path in task_payloads],
            output_dir=output_dir / "nmp_evidence", queries=queries,
            key_column="selection_unit_id", sequence_column="source_sequence",
            sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
            tool_path=binary, parameters_path=parameters,
        )
    _payload, shards = load_nmp_shard_manifest(
        merged, queries=queries, key_column="selection_unit_id",
        sequence_column="source_sequence", sequence_sha_column="source_sequence_sha256",
        stage="c5", peptide_lengths=[15], tool_path=binary, parameters_path=parameters,
    )
    summaries = pd.concat([pd.read_parquet(record["summary"]) for record in shards], ignore_index=True)
    coverage = summaries.set_index("protein_id")["coverage_fraction"]
    score_identity = hashlib.sha256(canonical_json({
        "manifest": _sha(merged), "tool": _sha(binary), "parameters": _sha(parameters),
    })).hexdigest()
    eligible = queries.copy()
    eligible["selection_source_coverage_fraction_15"] = eligible[
        "selection_unit_id"
    ].map(coverage)
    eligible["score_identity_sha256"] = score_identity
    eligible_path = output_dir / "c5_eligible.parquet"
    eligible.to_parquet(eligible_path, index=False)
    target = int(dag["current_c5_rung"])
    selected = deterministic_binned_sample(
        eligible, value_col="selection_source_coverage_fraction_15", target=target,
        n_bins=20, mode="uniform", stage="c5", min_per_populated_bin=0,
        seed=int(dag["seed"]),
    )
    result_path = output_dir / "c5_result.json"
    result_path.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-binned-sample/1",
        "protocol_profile": _protocol_profile(dag), "result": binned_sample_to_dict(selected),
    }, indent=2, sort_keys=True) + "\n")
    selected_path = output_dir / "c5_selected.parquet"
    eligible[eligible["selection_unit_id"].isin(selected.selected_ids)].to_parquet(
        selected_path, index=False
    )
    prior_path = output_dir / "c5_prior_rungs.json"
    copied_bundles: dict[int, Path] = {}
    copied_roots: dict[int, Path] = {}
    for prior_target, prior_root in prior:
        copied_root = output_dir / "prior_rungs" / str(prior_target) / "candidate_snapshot"
        shutil.copytree(prior_root, copied_root)
        bundle = (
            copied_root / "audit" / "orchestrator" / "stages" / "c7_select" /
            "00000" / "payload" / "failed_attempt.json"
        )
        if not bundle.is_file():
            raise ValueError("prior failed-rung snapshot lacks its attempt bundle")
        copied_bundles[prior_target] = bundle
        copied_roots[prior_target] = copied_root
    write_c5_prior_rung_index(
        prior_path, candidate_root=Path(dag["candidate_root"]),
        attempt_bundles=copied_bundles, ladder=dag["c5_ladder"],
        attempt_roots=copied_roots or None,
    )
    return {
        "eligible_count": len(eligible), "selected_count": len(selected.selected_ids),
        "eligible": eligible_path.name, "selected": selected_path.name,
        "result": result_path.name, "nmp_manifest": _relative(merged, output_dir),
    }


def _produce_c6_coordinate_source(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    c1 = _stage_payloads(dag, "c1_rcsb_source")[0]
    c5 = _stage_payloads(dag, "c5_select")[0]
    c8 = _stage_payloads(dag, "c8_sifts_residue_source")[0]
    selected = set(pd.read_parquet(c5 / "c5_selected.parquet")["selection_unit_id"].astype(str))
    fallbacks = pd.read_parquet(
        c1 / "rcsb_source" / "ordered_entity_fallbacks.parquet"
    )
    tier2_rows = []
    for unit, group in fallbacks[
        fallbacks["selection_unit_id"].astype(str).isin(selected)
    ].groupby("selection_unit_id", sort=True):
        order = 0
        for fallback in group.sort_values("fallback_rank", kind="stable").to_dict("records"):
            edges = json.loads(str(fallback["entity_chain_edges_json"]))
            for edge in sorted(edges, key=lambda row: (row["label_asym_id"], row["auth_asym_id"])):
                order += 1
                tier2_rows.append({
                    "source_tier": "tier2", "selection_unit_id": str(unit),
                    "source_uniprot_id": None,
                    "rcsb_entity_id": str(fallback["rcsb_entity_id"]),
                    "label_asym_id": str(edge["label_asym_id"]),
                    "auth_chain_id": str(edge["auth_asym_id"]),
                    "fallback_rank": int(fallback["fallback_rank"]),
                    "sifts_item_index": None, "attempt_order": order,
                    "source_revision_date": str(fallback["structure_revision_date"]),
                    "resolution": float(fallback["resolution"]),
                    "source_sequence": str(fallback["entity_sequence"]),
                    "source_sequence_sha256": hashlib.sha256(
                        str(fallback["entity_sequence"]).encode()
                    ).hexdigest(),
                    "sifts_range_json": None, "sifts_residue_map_json": None,
                })
    if {row["selection_unit_id"] for row in tier2_rows} != selected:
        raise ValueError("C6 coordinate plan does not cover every C5 selected unit")

    residue_root = c8 / "tier1_residue_maps"
    residue = pd.read_parquet(residue_root / "tier1_residue_map_candidates.parquet")
    preflight_root = c8 / "tier1_preflight"
    source_units = pd.read_parquet(
        preflight_root / "tier1_canonical_source_units.parquet"
    ).set_index("source_id")
    source_attempts = pd.read_parquet(
        preflight_root / "tier1_sifts_chain_attempts.parquet"
    ).set_index(["source_id", "sifts_item_index"])
    tier1_rows = []
    for source_id, group in residue.groupby("source_id", sort=True):
        for order, candidate in enumerate(group.sort_values(
            ["sifts_item_index", "pdb_id", "auth_chain_id", "label_asym_id"],
            kind="stable",
        ).to_dict("records"), start=1):
            unit = source_units.loc[str(source_id)]
            source_attempt = source_attempts.loc[
                (str(source_id), int(candidate["sifts_item_index"]))
            ]
            tier1_rows.append({
                "source_tier": "tier1",
                "selection_unit_id": f"t1:HLA-DRB1_15_01:{source_id}",
                "source_uniprot_id": str(source_id),
                "rcsb_entity_id": str(candidate["rcsb_entity_id"]),
                "label_asym_id": str(candidate["label_asym_id"]),
                "auth_chain_id": str(candidate["auth_chain_id"]),
                "fallback_rank": int(candidate["sifts_item_index"]) + 1,
                "sifts_item_index": int(candidate["sifts_item_index"]),
                "attempt_order": order,
                "source_revision_date": str(candidate["source_revision_date"]),
                "resolution": float(source_attempt["resolution"]),
                "source_sequence": str(unit["canonical_sequence"]),
                "source_sequence_sha256": str(unit["canonical_sequence_sha256"]),
                "sifts_range_json": json.dumps([
                    int(candidate["unp_start"]), int(candidate["unp_end"])
                ], separators=(",", ":")),
                "sifts_residue_map_json": str(candidate["residue_map_json"]),
                "coordinate_cif_path": str(candidate["coordinate_cif_path"]),
                "coordinate_manifest_path": str(candidate["coordinate_manifest_path"]),
                "download_sha256": str(candidate["coordinate_cif_sha256"]),
                "request_failure_path": None,
            })

    cache = output_dir / "coordinate_cache"
    acquisition: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tier2_rows:
        pdb_id = row["rcsb_entity_id"].split("_", 1)[0]
        key = (pdb_id, row["source_revision_date"])
        if key in acquisition:
            continue
        try:
            cif, manifest = acquire_rcsb_mmcif(
                pdb_id=pdb_id, expected_revision_date=key[1], cache_dir=cache,
                api_delay_s=0.1,
                **({
                    "request_fn": _adapter("rcsb_cif_request"),
                    "sleep_fn": _adapter("sleep_fn", lambda _seconds: None),
                } if _adapter("rcsb_cif_request") is not None else {}),
            )
            acquisition[key] = {
                "coordinate_cif_path": _relative(cif, Path(dag["candidate_root"])),
                "coordinate_manifest_path": _relative(
                    cif.parent / "manifest.json", Path(dag["candidate_root"])
                ),
                "download_sha256": _sha(cif), "request_failure_path": None,
            }
        except RcsbMmcifAcquisitionError as exc:
            acquisition[key] = {
                "coordinate_cif_path": None, "coordinate_manifest_path": None,
                "download_sha256": None,
                "request_failure_path": _relative(
                    exc.ledger_path, Path(dag["candidate_root"])
                ),
            }
    for row in tier2_rows:
        key = (row["rcsb_entity_id"].split("_", 1)[0], row["source_revision_date"])
        row.update(acquisition[key])
    attempts = pd.DataFrame([*tier2_rows, *tier1_rows]).sort_values(
        ["source_tier", "selection_unit_id", "attempt_order"], kind="stable"
    )
    path = output_dir / "coordinate_attempts.parquet"
    attempts.to_parquet(path, index=False)
    return {
        "tier2_attempt_count": len(tier2_rows), "tier1_attempt_count": len(tier1_rows),
        "unique_tier2_coordinate_requests": len(acquisition),
        "download_failures": sum(value["download_sha256"] is None for value in acquisition.values()),
        "coordinate_attempts": path.name,
    }


def _load_coords():
    if _adapter("load_coords") is not None:
        return _adapter("load_coords")
    try:
        from byprot.utils.io import load_coords
    except ImportError as exc:  # pragma: no cover - cluster environment contract
        raise RuntimeError("byprot load_coords is unavailable in the production environment") from exc
    return load_coords


def _mmcif_entity_sequence(path: Path, entity_id: str) -> str:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    payload = MMCIF2Dict(str(path))
    raw_ids = payload.get("_entity_poly.entity_id")
    raw_sequences = payload.get("_entity_poly.pdbx_seq_one_letter_code_can")
    ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
    sequences = raw_sequences if isinstance(raw_sequences, list) else [raw_sequences]
    if len(ids) != len(sequences):
        raise ValueError("coordinate mmCIF entity sequence table is malformed")
    matches = [
        "".join(str(sequence).split()).upper()
        for observed, sequence in zip(ids, sequences, strict=True)
        if str(observed) == str(entity_id)
    ]
    if len(matches) != 1 or not matches[0] or not set(matches[0]) <= set(
        "ACDEFGHIKLMNPQRSTVWY"
    ):
        raise ValueError("coordinate mmCIF lacks one canonical AA20 entity sequence")
    return matches[0]


def _materialize_attempts(
    dag: Mapping[str, Any], output_dir: Path, *, source_tier: str, index: int,
) -> dict[str, Any]:
    source = _stage_payloads(dag, "c6_coordinate_source")[0]
    attempts = pd.read_parquet(source / "coordinate_attempts.parquet")
    attempts = attempts[attempts["source_tier"] == source_tier].copy()
    key = "selection_unit_id" if source_tier == "tier2" else "source_uniprot_id"
    units = sorted(set(attempts[key].astype(str)))
    stage = "c6_tier2" if source_tier == "tier2" else "c6_tier1"
    count = int(next(row for row in dag["stages"] if row["name"] == stage)["task_count"])
    unit_frame = pd.DataFrame({key: units})
    selected_units = set(_contiguous_partition(unit_frame, index=index, count=count)[key])
    attempts = attempts[attempts[key].astype(str).isin(selected_units)].copy()
    load_coords = _load_coords()
    clean_dir = output_dir / "clean"
    clean_dir.mkdir()
    candidate_root = Path(dag["candidate_root"]).resolve()
    rows = []
    for attempt in attempts.sort_values([key, "attempt_order"], kind="stable").to_dict("records"):
        source_sequence = str(attempt["source_sequence"])
        source_map = None
        denominator = None
        if source_tier == "tier2":
            source_map = {position + 1: position for position in range(len(source_sequence))}
            denominator = range(len(source_sequence))
        else:
            residue_map = json.loads(str(attempt["sifts_residue_map_json"]))
            source_map = {
                int(record["label_seq_id"]): int(position)
                for position, record in residue_map.items()
            }
            unp_start, unp_end = json.loads(str(attempt["sifts_range_json"]))
            denominator = range(int(unp_start) - 1, int(unp_end))
        protein_id = (
            f"t2-{str(attempt['selection_unit_id']).split(':')[-1][:12]}-"
            f"{attempt['rcsb_entity_id']}-{attempt['label_asym_id']}"
            if source_tier == "tier2" else
            f"t1-{attempt['source_uniprot_id']}-{attempt['rcsb_entity_id']}-"
            f"{attempt['label_asym_id']}"
        )
        base = {
            **attempt, "protein_id": protein_id, "viable": False,
            "attempt_status": "download_failure" if pd.isna(
                attempt["download_sha256"]
            ) else "materialization_failure",
            "failure_reason": None, "if_sequence_coverage": None,
            "sequence": None, "sequence_length": None, "sequence_sha256": None,
            "structure_sha256": None, "pdb_path": None,
            "mapping_status": None, "final_to_source_json": None,
            "mapping_ledger_json": None, "load_coords_evidence_json": None,
            "entity_sequence": (
                source_sequence if source_tier == "tier2" else None
            ),
            "entity_sequence_sha256": (
                hashlib.sha256(source_sequence.encode()).hexdigest()
                if source_tier == "tier2" else None
            ),
        }
        if pd.isna(attempt["download_sha256"]):
            base["failure_reason"] = "coordinate_download_failure"
            rows.append(base)
            continue
        cif = candidate_root / str(attempt["coordinate_cif_path"])
        clean = clean_dir / f"{protein_id}.cif"
        try:
            if source_tier == "tier1":
                entity_sequence = _mmcif_entity_sequence(
                    cif, str(attempt["rcsb_entity_id"]).split("_", 1)[1]
                )
                base["entity_sequence"] = entity_sequence
                base["entity_sequence_sha256"] = hashlib.sha256(
                    entity_sequence.encode()
                ).hexdigest()
            materialized = materialize_mmcif_chain(
                source_mmcif=cif,
                expected_edge={
                    "rcsb_entity_id": str(attempt["rcsb_entity_id"]),
                    "label_asym_id": str(attempt["label_asym_id"]),
                    "auth_asym_id": str(attempt["auth_chain_id"]),
                },
                source_sequence=source_sequence, observed_to_source=source_map,
                denominator_positions=denominator, output_path=clean,
                output_chain_id="A", load_coords_fn=load_coords, min_coverage=0.8,
            )
            base.update({
                "viable": True, "attempt_status": "materialization_success",
                "failure_reason": "", "if_sequence_coverage": float(
                    materialized.mapping.if_sequence_coverage
                ),
                "sequence": materialized.mapping.sequence,
                "sequence_length": len(materialized.mapping.sequence),
                "sequence_sha256": hashlib.sha256(
                    materialized.mapping.sequence.encode()
                ).hexdigest(),
                "structure_sha256": materialized.structure_sha256,
                "pdb_path": _relative(materialized.output_path, candidate_root),
                "mapping_status": materialized.mapping.mapping_status,
                "final_to_source_json": json.dumps(
                    materialized.mapping.final_to_source, separators=(",", ":")
                ),
                "mapping_ledger_json": json.dumps(
                    materialized.mapping.mapping_ledger,
                    sort_keys=True, separators=(",", ":"),
                ),
                "load_coords_evidence_json": json.dumps(
                    materialized.load_coords_evidence,
                    sort_keys=True, separators=(",", ":"),
                ),
            })
        except (RuntimeError, ValueError) as exc:
            base["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(base)
    result = pd.DataFrame(rows).sort_values([key, "attempt_order"], kind="stable")
    path = output_dir / "chain_attempts.parquet"
    result.to_parquet(path, index=False)
    return {
        "source_tier": source_tier, "attempt_count": len(result),
        "viable_count": int(result["viable"].sum()), "attempts": path.name,
    }


def _produce_c6_tier2(dag: Mapping[str, Any], output_dir: Path, index: int) -> dict[str, Any]:
    return _materialize_attempts(
        dag, output_dir, source_tier="tier2", index=index,
    )


def _produce_c6_tier1(dag: Mapping[str, Any], output_dir: Path, index: int) -> dict[str, Any]:
    return _materialize_attempts(
        dag, output_dir, source_tier="tier1", index=index,
    )


def _merge_chain_attempts(dag: Mapping[str, Any]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(payload / "chain_attempts.parquet")
        for stage in ("c6_tier2", "c6_tier1")
        for payload in _stage_payloads(dag, stage)
    ]
    merged = pd.concat(frames, ignore_index=True)
    keys = [
        "source_tier", "selection_unit_id", "rcsb_entity_id", "label_asym_id",
        "auth_chain_id",
    ]
    if merged.duplicated(keys).any():
        raise ValueError("C6 merged chain attempts contain duplicate keys")
    return merged.sort_values(
        ["source_tier", "selection_unit_id", "attempt_order"], kind="stable"
    ).reset_index(drop=True)


def _rank_winners(
    attempts: pd.DataFrame, *, group_key: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    winners = []
    failures = []
    for unit, group in attempts.groupby(group_key, sort=True):
        viable = group[group["viable"] == True].sort_values(  # noqa: E712
            ["if_sequence_coverage", "resolution", "rcsb_entity_id", "auth_chain_id",
             "label_asym_id"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        if viable.empty:
            failures.append({
                group_key: str(unit), "failure": "no_viable_c6_chain",
                "attempt_count": len(group),
            })
            continue
        winner = viable.iloc[0].to_dict()
        rejected = group[group.index != viable.index[0]][[
            "rcsb_entity_id", "label_asym_id", "auth_chain_id", "viable",
            "failure_reason",
        ]].to_dict("records")
        winner["selected_chain_provenance_json"] = json.dumps({
            "rcsb_entity_id": winner["rcsb_entity_id"],
            "label_asym_id": winner["label_asym_id"],
            "auth_chain_id": winner["auth_chain_id"],
            "attempt_order": int(winner["attempt_order"]),
        }, sort_keys=True, separators=(",", ":"))
        winner["rejected_chain_provenance_json"] = json.dumps(
            rejected, sort_keys=True, separators=(",", ":")
        )
        winners.append(winner)
    return pd.DataFrame(winners), failures


def _deduplicate_tier2(winners: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = []
    ledger = []
    for sequence_sha, group in winners.groupby("sequence_sha256", sort=True):
        ranked = group.sort_values(
            ["if_sequence_coverage", "resolution", "rcsb_entity_id", "auth_chain_id",
             "label_asym_id"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        winner = str(ranked.iloc[0]["protein_id"])
        selected.append(winner)
        for protein_id in sorted(ranked["protein_id"].astype(str)):
            ledger.append({
                "protein_id": protein_id, "sequence_sha256": str(sequence_sha),
                "representative_protein_id": winner,
                "action": "keep" if protein_id == winner else "collapse",
            })
    replay = {
        "schema_version": "if-benchmark-v3-final-dedup-replay/1",
        "inputs": _json_records(winners),
        "selected_protein_ids": selected,
        "dedup_ledger": ledger,
    }
    return winners[winners["protein_id"].astype(str).isin(selected)].copy(), replay


def _run_cath(
    dag: Mapping[str, Any], output_dir: Path, rows: pd.DataFrame, *, prefix: str,
) -> tuple[Path, Path]:
    refs = _stage_payloads(dag, "references")[0]
    queries = rows[["protein_id", "selection_unit_id", "sequence", "sequence_sha256"]].copy()
    queries = queries.sort_values("protein_id", kind="stable")
    query_path = output_dir / f"{prefix}_queries.parquet"
    fasta_path = output_dir / f"{prefix}_queries.fasta"
    queries.to_parquet(query_path, index=False)
    _write_fasta(queries, fasta_path, key="protein_id", sequence="sequence")
    reference = refs / "references" / "cath_train.fasta"
    search = run_mmseqs_easy_search(
        tool_path=_input_path(dag, "mmseqs_binary"), query_fasta_path=fasta_path,
        reference_fasta_path=reference, output_dir=output_dir / f"{prefix}_mmseqs",
        cov_mode=0, coverage=0.8, min_seq_id=0.3,
        threads=int(next(
            row for row in dag["stages"] if row["name"] == "c4_final_cath"
        )["resources"]["cpus"]),
    )
    run = json.loads(search["manifest"].read_text())
    sidecar = build_mmseqs_sidecar(
        queries[["protein_id", "sequence"]], raw_tsv_path=search["raw_output"],
        query_fasta_path=fasta_path, search_kind="cath",
        reference_sha256=_sha(reference), tool_sha256=_sha(_input_path(dag, "mmseqs_binary")),
        tool_version=str(run["tool_version"]), cov_mode=0, coverage=0.8,
        min_seq_id=0.3, path_root=Path(dag["candidate_root"]),
    )
    sidecar_path = output_dir / f"{prefix}_sidecar.parquet"
    sidecar.to_parquet(sidecar_path, index=False)
    return query_path, sidecar_path


def _produce_c4_final(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    attempts = _merge_chain_attempts(dag)
    attempts_path = output_dir / "chain_attempts.parquet"
    attempts.to_parquet(attempts_path, index=False)
    evidence_rows = attempts[[
        "source_tier", "selection_unit_id", "rcsb_entity_id", "label_asym_id",
        "auth_chain_id", "coordinate_cif_path", "coordinate_manifest_path",
        "request_failure_path",
    ]].to_dict("records")
    structure_registry = write_structure_download_registry(
        output_dir / "structure_downloads.parquet",
        candidate_root=Path(dag["candidate_root"]), chain_attempts=attempts,
        evidence_rows=evidence_rows,
    )
    t2_winners, t2_failures = _rank_winners(
        attempts[attempts["source_tier"] == "tier2"], group_key="selection_unit_id"
    )
    c1 = _stage_payloads(dag, "c1_rcsb_source")[0]
    fallbacks = pd.read_parquet(
        c1 / "rcsb_source" / "ordered_entity_fallbacks.parquet"
    )
    group_entities = {
        str(unit): group.sort_values("fallback_rank", kind="stable")[
            "rcsb_entity_id"
        ].astype(str).tolist()
        for unit, group in fallbacks.groupby("selection_unit_id", sort=False)
    }
    t2_winners["group_entity_ids"] = t2_winners["selection_unit_id"].astype(str).map(
        group_entities
    )
    t2_dedup, dedup_replay = _deduplicate_tier2(t2_winners)
    dedup_path = output_dir / "final_dedup_replay.json"
    dedup_path.write_text(json.dumps(
        dedup_replay, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    t1_winners, t1_failures = _rank_winners(
        attempts[attempts["source_tier"] == "tier1"], group_key="source_uniprot_id"
    )
    t2_path = output_dir / "tier2_final_dedup.parquet"
    t1_path = output_dir / "tier1_representatives.parquet"
    t2_dedup.to_parquet(t2_path, index=False)
    t1_winners.to_parquet(t1_path, index=False)
    combined = pd.concat([t2_dedup, t1_winners], ignore_index=True)
    query_path, sidecar_path = _run_cath(
        dag, output_dir, combined, prefix="all_final_cath",
    )
    failures = output_dir / "c6_unit_failures.json"
    failures.write_text(json.dumps({
        "tier2": t2_failures, "tier1": t1_failures,
    }, indent=2, sort_keys=True) + "\n")
    return {
        "tier2_c6_winners": len(t2_winners), "tier2_final_dedup": len(t2_dedup),
        "tier1_c6_winners": len(t1_winners),
        "chain_attempts": attempts_path.name,
        "structure_downloads": structure_registry.name,
        "dedup_replay": dedup_path.name,
        "all_cath_queries": query_path.name, "all_cath_sidecar": sidecar_path.name,
    }


def _produce_c8_finalize(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    c4 = _stage_payloads(dag, "c4_final_cath")[0]
    c8 = _stage_payloads(dag, "c8_sifts_residue_source")[0]
    representatives = pd.read_parquet(c4 / "tier1_representatives.parquet")
    cath = pd.read_parquet(c4 / "all_final_cath_sidecar.parquet").set_index("protein_id")
    source_units = pd.read_parquet(
        c8 / "tier1_preflight" / "tier1_canonical_source_units.parquet"
    ).set_index("source_id")
    candidates = []
    for raw in representatives.to_dict("records"):
        row = dict(raw)
        source_id = str(row["source_uniprot_id"])
        source = source_units.loc[source_id]
        sequence = str(row["sequence"])
        final_to_source = [int(value) for value in json.loads(str(row["final_to_source_json"]))]
        residue_map = {
            int(position): record
            for position, record in json.loads(str(row["sifts_residue_map_json"])).items()
        }
        unp_start, unp_end = json.loads(str(row["sifts_range_json"]))
        projection = project_verified_spans(
            canonical_sequence=str(source["canonical_sequence"]),
            frozen_uniprot_range=(int(unp_start), int(unp_end)),
            raw_uniprot_spans=json.loads(str(source["spans_json"])),
            final_sequence=sequence, final_to_uniprot=final_to_source,
            uniprot_to_chain_residue=residue_map,
        )
        cath_row = cath.loc[str(row["protein_id"])]
        row.update({
            "pdb_id": str(row["rcsb_entity_id"]).split("_", 1)[0],
            "entity_id": str(row["rcsb_entity_id"]).split("_", 1)[1],
            "projection_valid": bool(projection.valid),
            "final_epitope_coverage": float(projection.if_coverage),
            "verified_span_count": int(projection.verified_span_count),
            "verified_covered_residue_count": int(
                projection.verified_covered_residue_count
            ),
            "raw_uniprot_spans_json": json.dumps(
                projection.uniprot_spans, sort_keys=True, separators=(",", ":")
            ),
            "chain_spans_json": json.dumps(
                projection.chain_spans, sort_keys=True, separators=(",", ":")
            ),
            "if_spans_json": json.dumps(
                projection.if_spans, sort_keys=True, separators=(",", ":")
            ),
            "span_drop_counts_json": json.dumps(
                projection.drop_counts, sort_keys=True, separators=(",", ":")
            ),
            "tier1_epitope_allele": ALLELE,
            "cath_overlap_flag": bool(cath_row["overlap_flag"]),
            "cath_best_target": None if pd.isna(cath_row["best_target"]) else str(
                cath_row["best_target"]
            ),
            "cath_best_identity": None if pd.isna(cath_row["best_identity"]) else float(
                cath_row["best_identity"]
            ),
            "cath_query_sha256": str(cath_row["query_sequence_sha256"]),
            "cath_reference_sha256": str(cath_row["reference_sha256"]),
            "cath_tool_sha256": str(cath_row["tool_sha256"]),
            "cath_parameters_sha256": str(cath_row["parameters_sha256"]),
        })
        candidates.append(row)
    candidate_frame = pd.DataFrame(candidates)
    candidate_path = output_dir / "tier1_candidate_pool.parquet"
    candidate_frame.to_parquet(candidate_path, index=False)
    primary, diagnostic, ledger = finalize_tier1_pool(
        candidates, target=int(dag["targets"]["tier1"])
    )
    primary_path = output_dir / "tier1_primary.parquet"
    diagnostic_path = output_dir / "tier1_diagnostic.parquet"
    tier1_columns = list(candidate_frame.columns)
    for record in [*primary, *diagnostic]:
        tier1_columns.extend(key for key in record if key not in tier1_columns)
    pd.DataFrame(primary, columns=tier1_columns).to_parquet(primary_path, index=False)
    pd.DataFrame(diagnostic, columns=tier1_columns).to_parquet(
        diagnostic_path, index=False
    )
    ledger_path = output_dir / "tier1_internal_collision_ledger.json"
    ledger_path.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-tier1-internal-collisions/1",
        "rows": ledger,
    }, indent=2, sort_keys=True) + "\n")
    return {
        "candidate_count": len(candidates), "primary_count": len(primary),
        "diagnostic_count": len(diagnostic), "candidate_pool": candidate_path.name,
        "primary": primary_path.name, "diagnostic": diagnostic_path.name,
    }


def _produce_collision_dedup(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    c4 = _stage_payloads(dag, "c4_final_cath")[0]
    c8 = _stage_payloads(dag, "c8_tier1_finalize")[0]
    t1_primary = pd.read_parquet(c8 / "tier1_primary.parquet")
    t1_diagnostic = pd.read_parquet(c8 / "tier1_diagnostic.parquet")
    t2 = pd.read_parquet(c4 / "tier2_final_dedup.parquet")
    input_primary = _json_records(t1_primary)
    input_diagnostic = _json_records(t1_diagnostic)
    input_t2 = _json_records(t2)
    primary, diagnostic, remaining, ledger = resolve_tier1_tier2_collisions(
        t1_primary=input_primary, t1_diagnostic=input_diagnostic, t2_rows=input_t2,
    )
    replay = {
        "schema_version": "if-benchmark-v3-collision-replay/1",
        "inputs": {
            "tier1_primary": input_primary,
            "tier1_diagnostic": input_diagnostic,
            "tier2_rows": input_t2,
        },
        "outputs": {
            "tier1_primary": primary, "tier1_diagnostic": diagnostic,
            "tier2_rows": remaining,
        },
        "collision_ledger": ledger,
    }
    replay_path = output_dir / "collision_replay.json"
    replay_path.write_text(json.dumps(
        replay, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    primary_path = output_dir / "tier1_primary.parquet"
    diagnostic_path = output_dir / "tier1_diagnostic.parquet"
    t2_path = output_dir / "tier2_post_collision.parquet"
    pd.DataFrame(primary, columns=t1_primary.columns).to_parquet(primary_path, index=False)
    pd.DataFrame(diagnostic, columns=t1_diagnostic.columns).to_parquet(
        diagnostic_path, index=False
    )
    remaining_frame = pd.DataFrame(remaining)
    remaining_frame.to_parquet(t2_path, index=False)
    query_path, sidecar_path = _run_cath(
        dag, output_dir, remaining_frame, prefix="final_cath",
    )
    return {
        "tier1_primary_count": len(primary), "tier1_diagnostic_count": len(diagnostic),
        "tier2_post_collision_count": len(remaining), "collision_count": len(ledger),
        "collision_replay": replay_path.name,
        "final_cath_queries": query_path.name, "final_cath_sidecar": sidecar_path.name,
    }


def _c7_clean_queries(dag: Mapping[str, Any]) -> pd.DataFrame:
    collision = _stage_payloads(dag, "collision_dedup")[0]
    rows = pd.read_parquet(collision / "tier2_post_collision.parquet")
    cath = pd.read_parquet(collision / "final_cath_sidecar.parquet").set_index("protein_id")
    clean = rows[
        rows["protein_id"].map(lambda value: not bool(cath.loc[value, "overlap_flag"]))
    ].copy().sort_values("protein_id", kind="stable").reset_index(drop=True)
    clean["cath_overlap_flag"] = False
    clean["cath_query_sha256"] = clean["protein_id"].map(
        cath["query_sequence_sha256"]
    )
    clean["cath_reference_sha256"] = clean["protein_id"].map(cath["reference_sha256"])
    clean["cath_tool_sha256"] = clean["protein_id"].map(cath["tool_sha256"])
    clean["cath_parameters_sha256"] = clean["protein_id"].map(cath["parameters_sha256"])
    clean["cath_best_target"] = clean["protein_id"].map(cath["best_target"])
    clean["cath_best_identity"] = clean["protein_id"].map(cath["best_identity"])
    return clean


def _produce_c7_nmp(
    dag: Mapping[str, Any], output_dir: Path, index: int,
) -> dict[str, Any]:
    clean = _c7_clean_queries(dag)
    count = int(next(row for row in dag["stages"] if row["name"] == "c7_nmp")["task_count"])
    subset = _contiguous_partition(clean, index=index, count=count)
    query_path = output_dir / "queries.parquet"
    subset.to_parquet(query_path, index=False)
    lengths = list(range(12, 26))
    parameters = output_dir / "parameters.json"
    parameters.write_text(json.dumps(
        _nmp_parameters(dag, stage="c7", lengths=lengths), indent=2, sort_keys=True
    ) + "\n")
    binary = _input_path(dag, "nmp_binary")
    scorer_factory = _adapter("nmp_score_shard_factory", make_standalone_score_shard_fn)
    scorer = scorer_factory(
        binary_path=binary, allele=ALLELE, peptide_lengths=lengths,
        parameters_path=parameters, key_column="protein_id", sequence_column="sequence",
    )
    manifest = write_nmp_sharded_evidence(
        output_dir=output_dir / "evidence", queries=subset,
        key_column="protein_id", sequence_column="sequence",
        sequence_sha_column="sequence_sha256", stage="c7", peptide_lengths=lengths,
        tool_path=binary, parameters_path=parameters, score_shard_fn=scorer,
        shard_query_count=len(subset),
    )
    return {
        "query_count": len(subset), "query_table": query_path.name,
        "parameters": parameters.name, "manifest": _relative(manifest, output_dir),
    }


def _selection_bins(frame: pd.DataFrame, *, value_col: str, histogram: list[dict[str, Any]]) -> pd.Series:
    values = frame[value_col].to_numpy(dtype=float)
    lo = float(histogram[0]["bin_lo"])
    hi = float(histogram[-1]["bin_hi"])
    if hi == lo:
        indices = np.zeros(len(frame), dtype=int)
    else:
        edges = np.array([float(histogram[0]["bin_lo"])] + [
            float(row["bin_hi"]) for row in histogram
        ])
        indices = np.digitize(values, edges[1:-1], right=False).astype(int)
        indices = np.clip(indices, 0, len(histogram) - 1)
    return pd.Series(indices, index=frame.index, dtype=int)


def _produce_c7_select(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    queries = _c7_clean_queries(dag)
    task_payloads = _stage_payloads(dag, "c7_nmp")
    if len({_sha(path / "parameters.json") for path in task_payloads}) != 1:
        raise ValueError("C7 NMP array tasks used mixed parameter bytes")
    parameters = output_dir / "parameters.json"
    shutil.copyfile(task_payloads[0] / "parameters.json", parameters)
    binary = _input_path(dag, "nmp_binary")
    lengths = list(range(12, 26))
    merged = merge_nmp_sharded_evidence(
        source_manifests=[path / "evidence" / "manifest.json" for path in task_payloads],
        output_dir=output_dir / "nmp_evidence", queries=queries,
        key_column="protein_id", sequence_column="sequence",
        sequence_sha_column="sequence_sha256", stage="c7", peptide_lengths=lengths,
        tool_path=binary, parameters_path=parameters,
    )
    _payload, shards = load_nmp_shard_manifest(
        merged, queries=queries, key_column="protein_id", sequence_column="sequence",
        sequence_sha_column="sequence_sha256", stage="c7", peptide_lengths=lengths,
        tool_path=binary, parameters_path=parameters,
    )
    summaries = pd.concat([pd.read_parquet(record["summary"]) for record in shards], ignore_index=True)
    summaries = summaries.set_index("protein_id")
    eligible = queries.copy()
    eligible["coverage_fraction"] = eligible["protein_id"].map(
        summaries["coverage_fraction"]
    )
    eligible["nmp_evidence_sha256"] = eligible["protein_id"].map(
        summaries["raw_evidence_sha256"]
    )
    eligible_path = output_dir / "c7_eligible.parquet"
    eligible.to_parquet(eligible_path, index=False)
    expansion = output_dir / "c5_expansion.json"
    expansion.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-c5-expansion/1",
        "preregistered_ladder": [int(value) for value in dag["c5_ladder"]],
        "attempts": [{
            "target": int(dag["current_c5_rung"]),
            "observed_final_clean_capacity": len(eligible),
        }],
    }, indent=2, sort_keys=True) + "\n")
    if len(eligible) < int(dag["targets"]["tier2"]):
        c4_final = _stage_payloads(dag, "c4_final_cath")[0]
        c8_final = _stage_payloads(dag, "c8_tier1_finalize")[0]
        collision = _stage_payloads(dag, "collision_dedup")[0]
        c5 = _stage_payloads(dag, "c5_select")[0]
        c4_source = _stage_payloads(dag, "c4_source_cath")[0]
        failed_path = output_dir / "failed_attempt.json"
        ledger_paths = {
            "chain_attempts": c4_final / "chain_attempts.parquet",
            "collision_replay": collision / "collision_replay.json",
            "final_dedup_replay": c4_final / "final_dedup_replay.json",
            "tier1_candidate_pool": c8_final / "tier1_candidate_pool.parquet",
            "tier1_internal_collision_ledger": c8_final /
            "tier1_internal_collision_ledger.json",
            "structure_downloads": c4_final / "structure_downloads.parquet",
        }
        selection_paths = {
            "c5_eligible": c5 / "c5_eligible.parquet",
            "c5_result": c5 / "c5_result.json",
            "c5_nmp_shard_manifest": c5 / "nmp_evidence" / "manifest.json",
            "c5_nmp_parameters": c5 / "parameters.json",
            "source_cath_queries": c4_source / "source_cath_queries.parquet",
            "source_cath_sidecar": c4_source / "source_cath_sidecar.parquet",
            "final_cath_queries": collision / "final_cath_queries.parquet",
            "final_cath_sidecar": collision / "final_cath_sidecar.parquet",
            "c7_eligible": eligible_path,
        }
        nmp_paths = {
            "shard_manifest": merged, "parameters": parameters,
        }
        if (
            set(ledger_paths) != set(EVIDENCE_KEYS["ledgers"])
            or set(selection_paths) != set(FAILED_C5_SELECTION_KEYS)
            or set(nmp_paths) != set(FAILED_C5_NMP_KEYS)
        ):
            raise AssertionError("failed-rung evidence path registry is incomplete")
        write_failed_c5_attempt_bundle(
            failed_path, candidate_root=Path(dag["candidate_root"]),
            target=int(dag["current_c5_rung"]), ledgers=ledger_paths,
            selection=selection_paths, nmp=nmp_paths,
        )
        return {
            "scientific_status": "insufficient_capacity",
            "eligible_count": len(eligible),
            "required_tier2_count": int(dag["targets"]["tier2"]),
            "failed_attempt": failed_path.name, "c5_expansion": expansion.name,
            "nmp_manifest": _relative(merged, output_dir),
        }
    selected = deterministic_binned_sample(
        eligible, value_col="coverage_fraction", target=int(dag["targets"]["tier2"]),
        n_bins=20, mode="gaussian", stage="c7",
        min_per_populated_bin=(40 if _protocol_profile(dag) == "drb1501_production" else 1),
        seed=int(dag["seed"]),
    )
    result_path = output_dir / "c7_result.json"
    result_path.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-binned-sample/1",
        "protocol_profile": _protocol_profile(dag), "result": binned_sample_to_dict(selected),
    }, indent=2, sort_keys=True) + "\n")
    eligible["selection_bin"] = _selection_bins(
        eligible, value_col="coverage_fraction", histogram=list(selected.histogram)
    )
    chosen = eligible[
        eligible["selection_unit_id"].astype(str).isin(selected.selected_ids)
    ].copy()
    chosen["source_pool_memberships"] = [["tier2"] for _ in range(len(chosen))]
    chosen["selected_tier_memberships"] = [["tier2"] for _ in range(len(chosen))]
    chosen["evaluation_role"] = "primary_generalization"
    chosen["selection_reason"] = "c7_gaussian_coverage_fraction_seed42"
    chosen["nmp_status"] = "complete"
    chosen["nmp_tool_sha256"] = _sha(binary)
    chosen["nmp_parameters_sha256"] = _sha(parameters)
    chosen["nmp_query_sequence_sha256"] = chosen["sequence_sha256"]
    chosen_path = output_dir / "tier2_selected.parquet"
    chosen.to_parquet(chosen_path, index=False)
    return {
        "scientific_status": "complete",
        "eligible_count": len(eligible), "selected_count": len(chosen),
        "eligible": eligible_path.name, "selected": chosen_path.name,
        "result": result_path.name, "nmp_manifest": _relative(merged, output_dir),
        "c5_expansion": expansion.name,
    }


def _normalize_release_row(
    row: Mapping[str, Any], *, dataset_release_id: str, tier: str,
) -> dict[str, Any]:
    result = dict(row)
    source_sequence = str(result["source_sequence"])
    if tier == "tier2":
        source_start, source_end = 0, len(source_sequence)
        result.update({
            "source_uniprot_id": None, "sifts_range_json": None,
            "uniprot_spans_json": None, "chain_spans_json": None,
            "sifts_residue_map_json": None, "if_spans_json": None,
            "verified_span_count": None, "verified_covered_residue_count": None,
            "final_epitope_coverage": None,
        })
    else:
        start_1b, end_1b = json.loads(str(result["sifts_range_json"]))
        source_start, source_end = int(start_1b) - 1, int(end_1b)
        result.update({
            "uniprot_spans_json": result["raw_uniprot_spans_json"],
            "nmp_status": None, "coverage_fraction": None,
            "nmp_query_sequence_sha256": None, "nmp_evidence_sha256": None,
            "nmp_tool_sha256": None, "nmp_parameters_sha256": None,
            "selection_bin": None, "selection_reason": None,
        })
    result.update({
        "dataset_release_id": dataset_release_id, "allele": ALLELE,
        "pdb_id": str(result["rcsb_entity_id"]).split("_", 1)[0],
        "entity_id": str(result["rcsb_entity_id"]).split("_", 1)[1],
        "sequence_length": len(str(result["sequence"])),
        "source_range_start_0b": source_start, "source_range_end_0b": source_end,
        "source_denominator_length": source_end - source_start,
        "experimental_method": "X-RAY DIFFRACTION", "family_cluster_id": None,
    })
    return result


def _run_family_clustering(
    dag: Mapping[str, Any], output_dir: Path, primary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    family = output_dir / "family"
    family.mkdir()
    fasta = family / "primary.fasta"
    _write_fasta(primary, fasta, key="protein_id", sequence="sequence")
    tool = _input_path(dag, "mmseqs_binary")
    prefix = family / "clu"
    actual_command = [
        str(tool), "easy-cluster", str(fasta), str(prefix), str(family / "tmp"),
        "--min-seq-id", "0.3", "--cov-mode", "0", "-c", "0.8",
        "--cluster-mode", "0", "--threads", "1",
    ]
    completed = subprocess.run(
        actual_command, text=True, capture_output=True, check=False, timeout=1800,
        env={**os.environ, "LC_ALL": "C"},
    )
    (family / "stdout.txt").write_text(completed.stdout)
    (family / "stderr.txt").write_text(completed.stderr)
    cluster_tsv = family / "clu_cluster.tsv"
    if completed.returncode != 0 or not cluster_tsv.is_file():
        raise RuntimeError("primary family MMseqs clustering failed")
    version = subprocess.run(
        [str(tool), "version"], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise RuntimeError("family MMseqs version probe failed")
    help_result = subprocess.run(
        [str(tool), "easy-cluster", "-h"], text=True, capture_output=True,
        check=False, timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    help_path = family / "easy_cluster_help.txt"
    help_path.write_text(help_result.stdout)
    if help_result.returncode != 0 or not help_result.stdout.strip():
        raise RuntimeError("family MMseqs easy-cluster help/default probe failed")
    parameters = family / "parameters.json"
    parameters.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-family-parameters/1",
        "min_seq_id": 0.3, "cov_mode": 0, "coverage": 0.8,
        "cluster_mode": 0, "threads": 1, "tool_sha256": _sha(tool),
        "tool_version": version.stdout.strip(),
        "command_argv": [
            "mmseqs", "easy-cluster", "final.fasta", "clu", "tmp",
            "--min-seq-id", "0.3", "--cov-mode", "0", "-c", "0.8",
            "--cluster-mode", "0", "--threads", "1",
        ],
        "effective_defaults": {
            "contract": "complete_easy_cluster_help_stdout",
            "help_path": help_path.name,
            "help_sha256": _sha(help_path),
        },
        "actual_command_argv": actual_command,
    }, indent=2, sort_keys=True) + "\n")
    clusters: dict[str, list[str]] = {}
    for line in cluster_tsv.read_text().splitlines():
        if line.strip():
            representative, member = line.split("\t")
            clusters.setdefault(representative, []).append(member)
    if {member for members in clusters.values() for member in members} != set(
        primary["protein_id"].astype(str)
    ):
        raise ValueError("family cluster output does not cover primary membership")
    sequence_sha = primary.set_index("protein_id")["sequence_sha256"]
    family_by_id = {}
    for members in clusters.values():
        family_id = stable_family_cluster_id([str(sequence_sha.loc[member]) for member in members])
        for member in members:
            family_by_id[member] = family_id
    result = primary.copy()
    result["family_cluster_id"] = result["protein_id"].map(family_by_id)
    return result, {
        "query_fasta": _relative(fasta, output_dir),
        "cluster_tsv": _relative(cluster_tsv, output_dir),
        "parameters": _relative(parameters, output_dir),
    }


def _produce_assemble_family(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    c7 = _stage_payloads(dag, "c7_select")[0]
    collision = _stage_payloads(dag, "collision_dedup")[0]
    t2 = pd.read_parquet(c7 / "tier2_selected.parquet")
    t1_primary = pd.read_parquet(collision / "tier1_primary.parquet")
    t1_diagnostic = pd.read_parquet(collision / "tier1_diagnostic.parquet")
    release_id = _candidate_annotation_id(dag)
    primary_rows = [
        *[_normalize_release_row(row, dataset_release_id=release_id, tier="tier1")
          for row in t1_primary.to_dict("records")],
        *[_normalize_release_row(row, dataset_release_id=release_id, tier="tier2")
          for row in t2.to_dict("records")],
    ]
    diagnostic_rows = [
        _normalize_release_row(row, dataset_release_id=release_id, tier="tier1")
        for row in t1_diagnostic.to_dict("records")
    ]
    primary = pd.DataFrame(primary_rows)
    diagnostic = pd.DataFrame(diagnostic_rows, columns=primary.columns)
    combined = pd.concat([primary, diagnostic], ignore_index=True)
    cath_queries, cath_sidecar_path = _run_cath(
        dag, output_dir, combined, prefix="release_cath",
    )
    cath = pd.read_parquet(cath_sidecar_path).set_index("protein_id")
    for frame in (primary, diagnostic):
        for index, row in frame.iterrows():
            measured = cath.loc[str(row["protein_id"])]
            if bool(measured["overlap_flag"]) != bool(row["cath_overlap_flag"]):
                raise ValueError("post-membership CATH replay changed a frozen membership flag")
            frame.at[index, "cath_best_target"] = (
                None if pd.isna(measured["best_target"]) else str(measured["best_target"])
            )
            frame.at[index, "cath_best_identity"] = (
                None if pd.isna(measured["best_identity"]) else float(measured["best_identity"])
            )
    structure_root = output_dir / "pdbs_if_ready"
    structure_root.mkdir()
    candidate_root = Path(dag["candidate_root"]).resolve()
    for frame in (primary, diagnostic):
        for index, row in frame.iterrows():
            source = candidate_root / str(row["pdb_path"])
            destination = structure_root / f"{row['protein_id']}.cif"
            shutil.copyfile(source, destination)
            if _sha(destination) != str(row["structure_sha256"]):
                raise ValueError("assembled structure copy digest changed")
            frame.at[index, "pdb_path"] = destination.name
    primary, family_paths = _run_family_clustering(dag, output_dir, primary)
    primary_path = output_dir / "cohort_if_ready.parquet"
    diagnostic_path = output_dir / "tier1_overlap_diagnostic_if_ready.parquet"
    primary.to_parquet(primary_path, index=False)
    diagnostic.to_parquet(diagnostic_path, index=False)
    load_rows = []
    for row in pd.concat([primary, diagnostic], ignore_index=True).to_dict("records"):
        evidence = json.loads(str(row["load_coords_evidence_json"]))
        load_rows.append({
            "protein_id": str(row["protein_id"]),
            "sequence_sha256": str(row["sequence_sha256"]),
            "structure_sha256": str(row["structure_sha256"]),
            "loaded_sequence_sha256": str(evidence["loaded_sequence_sha256"]),
            "loaded_length": int(evidence["loaded_length"]),
            "status": str(evidence["status"]),
        })
    load_path = output_dir / "load_coords.parquet"
    pd.DataFrame(load_rows).to_parquet(load_path, index=False)
    return {
        "primary_count": len(primary), "diagnostic_count": len(diagnostic),
        "primary": primary_path.name, "diagnostic": diagnostic_path.name,
        "pdb_root": structure_root.name, "cath_queries": cath_queries.name,
        "cath_sidecar": cath_sidecar_path.name, "load_coords": load_path.name,
        **family_paths,
    }


def _head_config_dir(dag: Mapping[str, Any]) -> Path:
    paths = [
        _input_path(dag, role)
        for role in ("head_model_yaml", "head_ablation_yaml", "head_inference_yaml")
    ]
    parents = {path.parent.resolve() for path in paths}
    if len(parents) != 1 or {path.name for path in paths} != {
        "model.yaml", "model_ablation.yaml", "inference.yaml"
    }:
        raise ValueError("Head model config files are not one canonical config directory")
    return parents.pop()


def _produce_c9_head(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    assemble = _stage_payloads(dag, "assemble_family")[0]
    primary_path = assemble / "cohort_if_ready.parquet"
    diagnostic_path = assemble / "tier1_overlap_diagnostic_if_ready.parquet"
    primary = pd.read_parquet(primary_path)
    diagnostic = pd.read_parquet(diagnostic_path)
    cohort = pd.concat([primary, diagnostic], ignore_index=True)
    output = output_dir / "head.parquet"
    manifest = output_dir / "head.manifest.json"
    params = dag["stage_parameters"]["head"]
    annotate_fixed_epoch_head(
        cohort=cohort, checkpoint_path=_input_path(dag, "head_checkpoint"),
        resolved_config_path=_input_path(dag, "head_resolved_config"),
        run_summary_path=_input_path(dag, "head_run_summary"),
        model_config_dir=_head_config_dir(dag), output_path=output,
        manifest_path=manifest, device=str(params["device"]),
        window_batch_size=int(params["window_batch_size"]),
        cohort_input_paths=[primary_path, diagnostic_path],
        predictor_factory=_adapter("head_predictor_factory"),
    )
    return {"row_count": len(cohort), "head": output.name, "manifest": manifest.name}


def _produce_c9_homology(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    assemble = _stage_payloads(dag, "assemble_family")[0]
    refs = _stage_payloads(dag, "references")[0]
    primary = pd.read_parquet(assemble / "cohort_if_ready.parquet")
    diagnostic = pd.read_parquet(assemble / "tier1_overlap_diagnostic_if_ready.parquet")
    queries = pd.concat([primary, diagnostic], ignore_index=True)[
        ["protein_id", "sequence", "sequence_sha256"]
    ].sort_values("protein_id", kind="stable")
    query_path = output_dir / "head_homology_queries.parquet"
    fasta = output_dir / "head_homology_queries.fasta"
    queries.to_parquet(query_path, index=False)
    _write_fasta(queries, fasta, key="protein_id", sequence="sequence")
    reference = refs / "references" / "head_seen.fasta"
    search = run_mmseqs_easy_search(
        tool_path=_input_path(dag, "mmseqs_binary"), query_fasta_path=fasta,
        reference_fasta_path=reference, output_dir=output_dir / "mmseqs",
        cov_mode=2, coverage=0.8, min_seq_id=0.3,
        threads=int(next(
            row for row in dag["stages"] if row["name"] == "c9_homology"
        )["resources"]["cpus"]),
    )
    run = json.loads(search["manifest"].read_text())
    sidecar = build_mmseqs_sidecar(
        queries[["protein_id", "sequence"]], raw_tsv_path=search["raw_output"],
        query_fasta_path=fasta, search_kind="head", reference_sha256=_sha(reference),
        tool_sha256=_sha(_input_path(dag, "mmseqs_binary")),
        tool_version=str(run["tool_version"]), cov_mode=2, coverage=0.8,
        min_seq_id=0.3, path_root=Path(dag["candidate_root"]),
    )
    head_stage = _stage_payloads(dag, "c9_head")[0]
    head = pd.read_parquet(head_stage / "head.parquet").set_index("protein_id")
    sidecar["dataset_release_id"] = sidecar["protein_id"].map(
        head["dataset_release_id"]
    )
    sidecar["sequence_sha256"] = sidecar["query_sequence_sha256"]
    sidecar["head_checkpoint_sha256"] = sidecar["protein_id"].map(
        head["head_checkpoint_sha256"]
    )
    sidecar_path = output_dir / "head_homology.parquet"
    sidecar.to_parquet(sidecar_path, index=False)
    return {
        "row_count": len(sidecar), "queries": query_path.name,
        "sidecar": sidecar_path.name, "mmseqs_manifest": _relative(
            search["manifest"], output_dir
        ),
    }


def _approved_sources(dag: Mapping[str, Any], path: Path) -> Path:
    import torch

    checkpoint_path = _input_path(dag, "head_checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ValueError("approved Head checkpoint lacks nested metadata")
    payload = {
        "schema_version": "if-benchmark-v3-approved-sources/1",
        "protocol_profile": _protocol_profile(dag),
        "head": {
            "allele": ALLELE, "fixed_epoch": 24,
            "checkpoint_sha256": _sha(checkpoint_path),
            "config_sha256": _sha(_input_path(dag, "head_resolved_config")),
            "run_summary_sha256": _sha(_input_path(dag, "head_run_summary")),
            "checkpoint_config_hash": str(metadata.get("config_hash") or ""),
            "checkpoint_global_step": int(metadata.get("global_step", -1)),
            "checkpoint_manifest_version": str(metadata.get("manifest_version") or ""),
        },
        "cath": {
            "release": "CATH4.3",
            "chain_set_sha256": _sha(_input_path(dag, "cath_chain_set")),
            "splits_sha256": _sha(_input_path(dag, "cath_splits")),
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _publication_artifacts(
    dag: Mapping[str, Any], output_dir: Path, *, release_id: str,
) -> tuple[Path, Path]:
    candidate_root = Path(dag["candidate_root"]).resolve()
    warehouse_root = candidate_root.parents[2]
    scope = json.loads(_input_path(dag, "descendant_scope").read_text())
    if (
        scope.get("schema_version") != "if-benchmark-v3-descendant-bindings/1"
        or not isinstance(scope.get("bindings"), list)
    ):
        raise ValueError("descendant scope input schema mismatch")
    inventory = output_dir / "descendant_inventory.json"
    write_descendant_inventory(
        inventory, warehouse_root=warehouse_root, allele_tag="HLA-DRB1_15_01",
        bindings=scope["bindings"],
    )
    main_alias = warehouse_root / "if_ready" / "main" / "HLA-DRB1_15_01"
    if os.path.lexists(main_alias):
        raise ValueError("DRB1501 first-release main alias unexpectedly exists")
    context = output_dir / "publication_context.json"
    context.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-publication-context/1",
        "release_id": release_id, "allele_tag": "HLA-DRB1_15_01",
        "mode": "first_release", "warehouse_root": str(warehouse_root),
        "candidate_root": str(candidate_root),
        "releases_root": str(warehouse_root / "releases"),
        "allele_main_alias": str(main_alias), "main_alias_identity": None,
        "prior_release_manifest": None, "prior_manifest_sha256": None,
    }, indent=2, sort_keys=True) + "\n")
    return inventory, context


def _produce_release_validate(
    dag: Mapping[str, Any], output_dir: Path, _index: int,
) -> dict[str, Any]:
    candidate_root = Path(dag["candidate_root"]).resolve()
    c1 = _stage_payloads(dag, "c1_rcsb_source")[0]
    c4_source = _stage_payloads(dag, "c4_source_cath")[0]
    c5 = _stage_payloads(dag, "c5_select")[0]
    c8_source = _stage_payloads(dag, "c8_sifts_residue_source")[0]
    c4_final = _stage_payloads(dag, "c4_final_cath")[0]
    c8_final = _stage_payloads(dag, "c8_tier1_finalize")[0]
    collision = _stage_payloads(dag, "collision_dedup")[0]
    c7 = _stage_payloads(dag, "c7_select")[0]
    assemble = _stage_payloads(dag, "assemble_family")[0]
    head_stage = _stage_payloads(dag, "c9_head")[0]
    homology = _stage_payloads(dag, "c9_homology")[0]
    refs = _stage_payloads(dag, "references")[0]

    install_manifest = output_dir / "nmp_install_manifest.json"
    shutil.copyfile(_input_path(dag, "nmp_install_manifest"), install_manifest)
    head_configs = output_dir / "head_model_configs"
    head_configs.mkdir()
    for role, name in (
        ("head_model_yaml", "model.yaml"),
        ("head_ablation_yaml", "model_ablation.yaml"),
        ("head_inference_yaml", "inference.yaml"),
    ):
        shutil.copyfile(_input_path(dag, role), head_configs / name)
    approved = _approved_sources(dag, output_dir / "approved_sources.json")
    provisional_release_id = _candidate_annotation_id(dag)
    inventory, publication = _publication_artifacts(
        dag, output_dir, release_id=provisional_release_id,
    )
    primary_path = assemble / "cohort_if_ready.parquet"
    diagnostic_path = assemble / "tier1_overlap_diagnostic_if_ready.parquet"
    cath_path = assemble / "release_cath_sidecar.parquet"
    head_path = head_stage / "head.parquet"
    homology_path = homology / "head_homology.parquet"
    load_path = assemble / "load_coords.parquet"
    evidence_path = output_dir / "release_evidence.json"
    write_release_evidence_bundle(
        evidence_path, candidate_root=candidate_root,
        protocol_profile=_protocol_profile(dag),
        sections={
            "source": {
                "rcsb_manifest": c1 / "rcsb_source" / "source_manifest.json",
                "tier1_manifest": c8_source / "tier1_preflight" / "preflight_manifest.json",
                "tier1_residue_manifest": c8_source / "tier1_residue_maps" / "manifest.json",
            },
            "ledgers": {
                "chain_attempts": c4_final / "chain_attempts.parquet",
                "collision_replay": collision / "collision_replay.json",
                "final_dedup_replay": c4_final / "final_dedup_replay.json",
                "tier1_candidate_pool": c8_final / "tier1_candidate_pool.parquet",
                "tier1_internal_collision_ledger": c8_final /
                "tier1_internal_collision_ledger.json",
                "structure_downloads": c4_final / "structure_downloads.parquet",
            },
            "selection": {
                "c5_eligible": c5 / "c5_eligible.parquet",
                "c5_result": c5 / "c5_result.json",
                "c5_expansion": c7 / "c5_expansion.json",
                "c5_prior_rungs": c5 / "c5_prior_rungs.json",
                "c5_nmp_shard_manifest": c5 / "nmp_evidence" / "manifest.json",
                "c5_nmp_tool": _input_path(dag, "nmp_binary"),
                "c5_nmp_parameters": c5 / "parameters.json",
                "source_cath_queries": c4_source / "source_cath_queries.parquet",
                "source_cath_sidecar": c4_source / "source_cath_sidecar.parquet",
                "final_cath_queries": collision / "final_cath_queries.parquet",
                "final_cath_sidecar": collision / "final_cath_sidecar.parquet",
                "c7_eligible": c7 / "c7_eligible.parquet",
                "c7_result": c7 / "c7_result.json",
            },
            "nmp": {
                "shard_manifest": c7 / "nmp_evidence" / "manifest.json",
                "tool": _input_path(dag, "nmp_binary"),
                "parameters": c7 / "parameters.json",
            },
            "nmp_install": {
                "tool": _input_path(dag, "nmp_binary"), "data_manifest": install_manifest,
                "contract_producer": Path(__file__).with_name("nmp.py"),
                "runner_producer": Path(__file__).parents[3] / "epitope_head" / "data" /
                "netmhciipan_runner.py",
            },
            "family": {
                "query_fasta": assemble / "family" / "primary.fasta",
                "cluster_tsv": assemble / "family" / "clu_cluster.tsv",
                "tool": _input_path(dag, "mmseqs_binary"),
                "parameters": assemble / "family" / "parameters.json",
            },
            "mmseqs": {
                "tool": _input_path(dag, "mmseqs_binary"),
                "identity": refs / "references" / "mmseqs_identity.json",
                "reference_producer": Path(__file__).with_name("references.py"),
                "search_producer": Path(__file__).with_name("leakage.py"),
                "cath_reference_manifest": refs / "references" / "cath_train.manifest.json",
                "cath_reference_fasta": refs / "references" / "cath_train.fasta",
                "head_reference_manifest": refs / "references" / "head_seen.manifest.json",
                "head_reference_fasta": refs / "references" / "head_seen.fasta",
            },
            "head": {
                "checkpoint": _input_path(dag, "head_checkpoint"),
                "config": _input_path(dag, "head_resolved_config"),
                "run_summary": _input_path(dag, "head_run_summary"),
                "annotation_manifest": head_stage / "head.manifest.json",
                "model_yaml": head_configs / "model.yaml",
                "model_ablation_yaml": head_configs / "model_ablation.yaml",
                "inference_yaml": head_configs / "inference.yaml",
                "cohort_primary": primary_path,
                "cohort_diagnostic": diagnostic_path,
            },
            "approved_sources": {"manifest": approved},
            "descendants": {"inventory": inventory},
            "publication": {"context": publication},
            "release_tables": {
                "primary": primary_path, "diagnostic": diagnostic_path,
                "cath": cath_path, "head": head_path,
                "head_homology": homology_path, "load_coords": load_path,
            },
        },
    )
    primary = pd.read_parquet(primary_path)
    diagnostic = pd.read_parquet(diagnostic_path)
    gates = validate_release_tables(
        primary, diagnostic, pd.read_parquet(cath_path), pd.read_parquet(head_path),
        pd.read_parquet(homology_path), pd.read_parquet(load_path),
        evidence_bundle_path=evidence_path, candidate_root=candidate_root,
        pdb_root=assemble / "pdbs_if_ready", release_id=provisional_release_id,
        allele=ALLELE, tier1_target=int(dag["targets"]["tier1"]),
        tier2_target=int(dag["targets"]["tier2"]),
        load_coords_fn=_adapter("load_coords"),
    )
    gates_path = output_dir / "release_gates.json"
    gates_path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    return {
        "release_id": provisional_release_id, "gate_count": len(gates),
        "all_gates_pass": all(value == "pass" for value in gates.values()),
        "evidence": evidence_path.name, "gates": gates_path.name,
    }


def _write_packaged_head_manifest(
    *, source_manifest: Path, destination_manifest: Path, head_output: Path,
    primary_path: Path, diagnostic_path: Path,
) -> Path:
    manifest = json.loads(Path(source_manifest).read_text())
    manifest["cohort_inputs"] = [
        _head_path_identity(path, root=destination_manifest.parent)
        for path in (primary_path, diagnostic_path)
    ]
    manifest["output_path"] = head_output.name
    manifest["output_sha256"] = _sha(head_output)
    manifest["row_count"] = len(pd.read_parquet(head_output))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    destination_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination_manifest


def package_candidate_release(
    *, dag_path: Path, dataset_release_id: str, main_alias: Path,
) -> Path:
    """Assign the semantic release ID, materialize root-level candidate bytes, and rerun gates.

    This is candidate-only packaging.  It requires the complete release-validation seal and never
    renames the build or touches the canonical alias.
    """

    from .production import SAFE_ID

    dag_path = Path(dag_path).resolve()
    dag = validate_production_dag(dag_path)
    status = production_status(dag_path=dag_path, deep=True)
    if any(value != "complete" for value in status.values()):
        raise ValueError("candidate packaging requires all production stage seals")
    release_id = str(dataset_release_id)
    if (
        not SAFE_ID.fullmatch(release_id)
        or release_id in {str(dag["build_id"]), _candidate_annotation_id(dag)}
    ):
        raise ValueError("semantic dataset_release_id must be safe and distinct from build identity")
    candidate_root = Path(dag["candidate_root"]).resolve()
    warehouse_root = candidate_root.parents[2]
    expected_alias = warehouse_root / "if_ready" / "main" / "HLA-DRB1_15_01"
    main_alias = Path(main_alias).resolve(strict=False)
    if main_alias != expected_alias.resolve(strict=False):
        raise ValueError("candidate packaging received the wrong main alias")
    alias_before = alias_identity(expected_alias)
    if alias_before is not None:
        raise ValueError("DRB1501 candidate packaging requires absent first-release alias")
    top_level = [
        candidate_root / "cohort_if_ready.parquet",
        candidate_root / "tier1_overlap_diagnostic_if_ready.parquet",
        candidate_root / "pdbs_if_ready", candidate_root / "annotations",
        candidate_root / "dataset_manifest.json", candidate_root / "audit" / "release",
    ]
    if any(path.exists() or path.is_symlink() for path in top_level):
        raise FileExistsError("candidate root already contains packaged release bytes")

    release_stage = _stage_payloads(dag, "release_validate")[0]
    provisional_gates = json.loads((release_stage / "release_gates.json").read_text())
    if provisional_gates != {str(index): "pass" for index in range(1, 14)}:
        raise ValueError("candidate packaging requires passing provisional scientific gates")
    c1 = _stage_payloads(dag, "c1_rcsb_source")[0]
    c4_source = _stage_payloads(dag, "c4_source_cath")[0]
    c5 = _stage_payloads(dag, "c5_select")[0]
    c8_source = _stage_payloads(dag, "c8_sifts_residue_source")[0]
    c4_final = _stage_payloads(dag, "c4_final_cath")[0]
    c8_final = _stage_payloads(dag, "c8_tier1_finalize")[0]
    collision = _stage_payloads(dag, "collision_dedup")[0]
    c7 = _stage_payloads(dag, "c7_select")[0]
    assemble = _stage_payloads(dag, "assemble_family")[0]
    head_stage = _stage_payloads(dag, "c9_head")[0]
    homology = _stage_payloads(dag, "c9_homology")[0]
    refs = _stage_payloads(dag, "references")[0]

    primary = pd.read_parquet(assemble / "cohort_if_ready.parquet")
    diagnostic = pd.read_parquet(assemble / "tier1_overlap_diagnostic_if_ready.parquet")
    primary["dataset_release_id"] = release_id
    diagnostic["dataset_release_id"] = release_id
    primary_path = candidate_root / "cohort_if_ready.parquet"
    diagnostic_path = candidate_root / "tier1_overlap_diagnostic_if_ready.parquet"
    primary.to_parquet(primary_path, index=False)
    diagnostic.to_parquet(diagnostic_path, index=False)
    shutil.copytree(assemble / "pdbs_if_ready", candidate_root / "pdbs_if_ready")
    annotations = candidate_root / "annotations"
    annotations.mkdir()
    cath_path = annotations / "cath.parquet"
    homology_path = annotations / "head_homology.parquet"
    shutil.copyfile(assemble / "release_cath_sidecar.parquet", cath_path)
    homology_rows = pd.read_parquet(homology / "head_homology.parquet")
    homology_rows["dataset_release_id"] = release_id
    homology_rows.to_parquet(homology_path, index=False)
    head = pd.read_parquet(head_stage / "head.parquet")
    head["dataset_release_id"] = release_id
    head_path = annotations / "head.parquet"
    head.to_parquet(head_path, index=False)
    head_manifest = _write_packaged_head_manifest(
        source_manifest=head_stage / "head.manifest.json",
        destination_manifest=annotations / "head.manifest.json", head_output=head_path,
        primary_path=primary_path, diagnostic_path=diagnostic_path,
    )
    release_audit = candidate_root / "audit" / "release"
    release_audit.mkdir()
    load_path = release_audit / "load_coords.parquet"
    shutil.copyfile(assemble / "load_coords.parquet", load_path)
    install_manifest = release_audit / "nmp_install_manifest.json"
    shutil.copyfile(_input_path(dag, "nmp_install_manifest"), install_manifest)
    head_configs = release_audit / "head_model_configs"
    head_configs.mkdir()
    for role, name in (
        ("head_model_yaml", "model.yaml"),
        ("head_ablation_yaml", "model_ablation.yaml"),
        ("head_inference_yaml", "inference.yaml"),
    ):
        shutil.copyfile(_input_path(dag, role), head_configs / name)
    approved = _approved_sources(dag, release_audit / "approved_sources.json")
    inventory, publication = _publication_artifacts(
        dag, release_audit, release_id=release_id,
    )
    evidence_path = release_audit / "release_evidence.json"
    write_release_evidence_bundle(
        evidence_path, candidate_root=candidate_root,
        protocol_profile=_protocol_profile(dag),
        sections={
            "source": {
                "rcsb_manifest": c1 / "rcsb_source" / "source_manifest.json",
                "tier1_manifest": c8_source / "tier1_preflight" / "preflight_manifest.json",
                "tier1_residue_manifest": c8_source / "tier1_residue_maps" / "manifest.json",
            },
            "ledgers": {
                "chain_attempts": c4_final / "chain_attempts.parquet",
                "collision_replay": collision / "collision_replay.json",
                "final_dedup_replay": c4_final / "final_dedup_replay.json",
                "tier1_candidate_pool": c8_final / "tier1_candidate_pool.parquet",
                "tier1_internal_collision_ledger": c8_final /
                "tier1_internal_collision_ledger.json",
                "structure_downloads": c4_final / "structure_downloads.parquet",
            },
            "selection": {
                "c5_eligible": c5 / "c5_eligible.parquet", "c5_result": c5 / "c5_result.json",
                "c5_expansion": c7 / "c5_expansion.json",
                "c5_prior_rungs": c5 / "c5_prior_rungs.json",
                "c5_nmp_shard_manifest": c5 / "nmp_evidence" / "manifest.json",
                "c5_nmp_tool": _input_path(dag, "nmp_binary"),
                "c5_nmp_parameters": c5 / "parameters.json",
                "source_cath_queries": c4_source / "source_cath_queries.parquet",
                "source_cath_sidecar": c4_source / "source_cath_sidecar.parquet",
                "final_cath_queries": collision / "final_cath_queries.parquet",
                "final_cath_sidecar": collision / "final_cath_sidecar.parquet",
                "c7_eligible": c7 / "c7_eligible.parquet", "c7_result": c7 / "c7_result.json",
            },
            "nmp": {
                "shard_manifest": c7 / "nmp_evidence" / "manifest.json",
                "tool": _input_path(dag, "nmp_binary"), "parameters": c7 / "parameters.json",
            },
            "nmp_install": {
                "tool": _input_path(dag, "nmp_binary"), "data_manifest": install_manifest,
                "contract_producer": Path(__file__).with_name("nmp.py"),
                "runner_producer": Path(__file__).parents[3] / "epitope_head" / "data" /
                "netmhciipan_runner.py",
            },
            "family": {
                "query_fasta": assemble / "family" / "primary.fasta",
                "cluster_tsv": assemble / "family" / "clu_cluster.tsv",
                "tool": _input_path(dag, "mmseqs_binary"),
                "parameters": assemble / "family" / "parameters.json",
            },
            "mmseqs": {
                "tool": _input_path(dag, "mmseqs_binary"),
                "identity": refs / "references" / "mmseqs_identity.json",
                "reference_producer": Path(__file__).with_name("references.py"),
                "search_producer": Path(__file__).with_name("leakage.py"),
                "cath_reference_manifest": refs / "references" / "cath_train.manifest.json",
                "cath_reference_fasta": refs / "references" / "cath_train.fasta",
                "head_reference_manifest": refs / "references" / "head_seen.manifest.json",
                "head_reference_fasta": refs / "references" / "head_seen.fasta",
            },
            "head": {
                "checkpoint": _input_path(dag, "head_checkpoint"),
                "config": _input_path(dag, "head_resolved_config"),
                "run_summary": _input_path(dag, "head_run_summary"),
                "annotation_manifest": head_manifest,
                "model_yaml": head_configs / "model.yaml",
                "model_ablation_yaml": head_configs / "model_ablation.yaml",
                "inference_yaml": head_configs / "inference.yaml",
                "cohort_primary": primary_path, "cohort_diagnostic": diagnostic_path,
            },
            "approved_sources": {"manifest": approved},
            "descendants": {"inventory": inventory},
            "publication": {"context": publication},
            "release_tables": {
                "primary": primary_path, "diagnostic": diagnostic_path,
                "cath": cath_path, "head": head_path,
                "head_homology": homology_path, "load_coords": load_path,
            },
        },
    )
    gates = validate_release_tables(
        primary, diagnostic, pd.read_parquet(cath_path), head,
        pd.read_parquet(homology_path), pd.read_parquet(load_path),
        evidence_bundle_path=evidence_path, candidate_root=candidate_root,
        pdb_root=candidate_root / "pdbs_if_ready", release_id=release_id,
        allele=ALLELE, tier1_target=int(dag["targets"]["tier1"]),
        tier2_target=int(dag["targets"]["tier2"]),
        load_coords_fn=_adapter("load_coords"),
    )
    gates_path = release_audit / "release_gates.json"
    gates_path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    write_dataset_manifest(candidate_root, {
        "dataset_release_id": release_id, "build_id": str(dag["build_id"]),
        "protocol_id": "if-benchmark-test-set/3", "release_gates": gates,
        "publication_state": "candidate_only",
    })
    verify_dataset_manifest(candidate_root)
    if alias_identity(expected_alias) != alias_before:
        raise RuntimeError("candidate packaging changed the main alias")
    return candidate_root / "dataset_manifest.json"


def package_failed_rung_snapshot(*, dag_path: Path) -> Path:
    """Seal an insufficient C5 rung as a content-bound source for the next build."""

    dag_path = Path(dag_path).resolve()
    dag = validate_production_dag(dag_path)
    status = production_status(dag_path=dag_path)
    if status.get("c7_select") != "failed" or any(
        status.get(stage) == "complete"
        for stage in ("assemble_family", "c9_head", "c9_homology", "release_validate")
    ):
        raise ValueError("failed-rung packaging requires terminal C7 insufficient capacity")
    c7 = _stage_root(dag, "c7_select") / "00000" / "payload"
    if not (c7 / "failed_attempt.json").is_file():
        raise ValueError("failed C7 rung lacks replayable attempt bundle")
    candidate_root = Path(dag["candidate_root"]).resolve()
    path = candidate_root / "failed_rung_snapshot_manifest.json"
    if path.exists() or (candidate_root / "dataset_manifest.json").exists():
        raise FileExistsError("failed-rung candidate is already packaged")
    artifacts = [
        {
            "path": str(item.relative_to(candidate_root)),
            "size_bytes": item.stat().st_size, "sha256": _sha(item),
        }
        for item in sorted(
            candidate_root.rglob("*"), key=lambda item: str(item.relative_to(candidate_root))
        )
        if item.is_file() and item != path
    ]
    payload = {
        "schema_version": "if-benchmark-v3-failed-rung-snapshot/1",
        "candidate_root": str(candidate_root),
        "target": int(dag["current_c5_rung"]),
        "failed_attempt": _relative(c7 / "failed_attempt.json", candidate_root),
        "artifacts": artifacts,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    from .production import _validate_failed_rung_snapshot_manifest

    _validate_failed_rung_snapshot_manifest(path)
    return path


PRODUCERS: dict[str, Callable[[Mapping[str, Any], Path, int], dict[str, Any]]] = {
    "c1_rcsb_source": _produce_c1,
    "c8_sifts_residue_source": _produce_c8_source,
    "references": _produce_references,
    "c4_source_cath": _produce_c4_source,
    "c5_nmp": _produce_c5_nmp,
    "c5_select": _produce_c5_select,
    "c6_coordinate_source": _produce_c6_coordinate_source,
    "c6_tier2": _produce_c6_tier2,
    "c6_tier1": _produce_c6_tier1,
    "c4_final_cath": _produce_c4_final,
    "c8_tier1_finalize": _produce_c8_finalize,
    "collision_dedup": _produce_collision_dedup,
    "c7_nmp": _produce_c7_nmp,
    "c7_select": _produce_c7_select,
    "assemble_family": _produce_assemble_family,
    "c9_head": _produce_c9_head,
    "c9_homology": _produce_c9_homology,
    "release_validate": _produce_release_validate,
}


def execute_production_stage(
    *, dag_path: Path, stage: str, task_index: int, task_count: int, output_dir: Path,
    adapters: Mapping[str, Any] | None = None,
) -> Path:
    """Execute one registered producer after replaying DAG/dependency identities."""

    dag_path = Path(dag_path).resolve()
    dag = validate_production_dag(dag_path)
    if stage not in STAGE_BY_NAME or stage not in PRODUCERS:
        raise ValueError(f"production scientific producer is not implemented: {stage}")
    row = next(item for item in dag["stages"] if item["name"] == stage)
    if int(task_count) != int(row["task_count"]) or not 0 <= int(task_index) < int(task_count):
        raise ValueError("production stage array identity mismatch")
    expected = _stage_root(dag, stage) / f"{int(task_index):05d}" / "payload"
    output_dir = Path(output_dir).resolve()
    if output_dir != expected.resolve() or output_dir.exists():
        raise ValueError("production stage output directory differs from registered task root")
    status = production_status(dag_path=dag_path, deep=False)
    if status.get(stage) != "ready":
        raise ValueError(f"production stage is not ready: {stage}={status.get(stage)}")
    _dependency_payloads(dag, stage)
    output_dir.mkdir(parents=True)
    token = _ACTIVE_ADAPTERS.set({} if adapters is None else dict(adapters))
    try:
        summary = PRODUCERS[stage](dag, output_dir, int(task_index))
    finally:
        _ACTIVE_ADAPTERS.reset(token)
    return _write_result(
        output_dir, dag=dag, stage=stage, task_index=int(task_index), summary=summary,
    )
