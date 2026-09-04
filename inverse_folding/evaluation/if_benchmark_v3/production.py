"""Fail-closed production DAG, array-task, and resume machinery for benchmark v3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .runner import StageJournal


PROTOCOL_ID = "if-benchmark-test-set/3"
ALLELE = "HLA-DRB1*15:01"
ALLELE_TAG = "HLA-DRB1_15_01"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

APPROVED_HASHES = {
    "head_checkpoint": "1ff54085f2cde1ea0847ebca567af49b2ca85cd23e730d21d515bbb02896136b",
    "head_resolved_config": "fec4e3289d060d9d6e8da8abe93b0b60db20ad578e218e879d6412f33549ecc3",
    "head_run_summary": "5a85445890f58e2cea9515147b5e1a6837dad32949ac299876240033663b16df",
    "cath_chain_set": "e71f886a17db434aaa605752309e0c283be3f060f6710d0d5765deca7bc53b4f",
    "cath_splits": "444c6a88599f331705b09c947f8e48bd7147b77437ee2e2e375ce26c2c7b2501",
    "tier1_preflight_manifest": "b3fa9206f7351b092cde3d0633305ab0e513973da35d2b30ee0500838c15d109",
    "strict_train_ids": "65ce84dcc1fc120fbcdf1604255155f00366c2c2cd939007ced9701306c56429",
    "strict_val_ids": "3612b2ddf48dbbd7d412d9fff48f1c1976807331a7009f3ade970b486bd043ec",
    "strict_test_ids": "ee38f2765c237640f80511a57530d14a38267716b516232fbeed64802ff29c4b",
    "protein_samples": "d3aeb1aac9f7bdf6a1113ce0d879f3414cb6bee9f03bd3f0e33793bd04912fab",
    "span_records": "92479a155ea093033d65a319d4ed3384ac0b14005797387d1bc9ca9a3099d280",
}

APPROVED_LOGICAL_MANIFESTS = {
    "tier1_preflight_manifest": "f9c7875d2497f7122b1e72992d0768702b0d4ca62874e21e60229f421b96fc40",
}

REQUIRED_INPUT_ROLES = {
    *APPROVED_HASHES,
    "tier1_preflight_manifest", "strict_train_ids", "strict_val_ids",
    "strict_test_ids", "protein_samples", "span_records", "nmp_binary",
    "nmp_install_manifest", "nmp_probe_manifest", "mmseqs_binary",
    "mmseqs_probe_manifest", "head_probe_manifest", "structure_probe_manifest",
    "descendant_scope", "runner_code", "slurm_launcher", "python_executable",
    "head_model_yaml", "head_ablation_yaml", "head_inference_yaml",
    "protocol", "runner_doc",
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...]
    execution: str
    shard_key: str | None = None
    resource_class: str = "cpu"


STAGES = (
    StageSpec("c1_rcsb_source", (), "networked_login", resource_class="network"),
    # C8 depends on C1 for the FROZEN coordinate revision. PDBe enriched mmCIF carries no
    # revision history, so the content-bound RCSB entity snapshot is the only revision authority
    # that can be cross-checked against what RCSB later serves.
    StageSpec(
        "c8_sifts_residue_source", ("c1_rcsb_source",), "networked_login",
        resource_class="network",
    ),
    StageSpec("references", (), "slurm", resource_class="cpu"),
    StageSpec("c4_source_cath", ("c1_rcsb_source", "references"), "slurm"),
    StageSpec("c5_nmp", ("c4_source_cath",), "slurm", "c5_nmp_shards", "cpu"),
    StageSpec("c5_select", ("c5_nmp",), "slurm"),
    StageSpec(
        "c6_coordinate_source", ("c5_select", "c8_sifts_residue_source"),
        "networked_login", resource_class="network",
    ),
    StageSpec(
        "c6_tier2", ("c6_coordinate_source",), "slurm", "c6_tier2_shards", "cpu"
    ),
    StageSpec(
        "c6_tier1", ("c6_coordinate_source",), "slurm", "c6_tier1_shards", "cpu"
    ),
    StageSpec("c4_final_cath", ("c6_tier2", "c6_tier1", "references"), "slurm"),
    StageSpec("c8_tier1_finalize", ("c4_final_cath",), "slurm"),
    StageSpec("collision_dedup", ("c8_tier1_finalize", "c6_tier2"), "slurm"),
    StageSpec("c7_nmp", ("collision_dedup",), "slurm", "c7_nmp_shards", "cpu"),
    StageSpec("c7_select", ("c7_nmp",), "slurm"),
    StageSpec("assemble_family", ("c7_select", "c8_tier1_finalize"), "slurm"),
    StageSpec("c9_head", ("assemble_family",), "slurm", resource_class="gpu"),
    StageSpec("c9_homology", ("c9_head", "references"), "slurm"),
    StageSpec("release_validate", ("c9_homology",), "slurm"),
)
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}

# Immutable external inputs read directly by each stage.  Dependency payloads are discovered only
# through verified predecessor seals and therefore do not appear as mutable caller paths here.
STAGE_INPUT_ROLES: dict[str, tuple[str, ...]] = {
    "c1_rcsb_source": ("protocol", "runner_doc"),
    "c8_sifts_residue_source": (
        "tier1_preflight_manifest", "span_records", "protocol",
    ),
    "references": (
        "cath_chain_set", "cath_splits", "strict_train_ids", "strict_val_ids",
        "strict_test_ids", "protein_samples", "mmseqs_binary", "protocol",
    ),
    "c4_source_cath": ("mmseqs_binary", "mmseqs_probe_manifest", "protocol"),
    "c5_nmp": ("nmp_binary", "nmp_install_manifest", "nmp_probe_manifest", "protocol"),
    "c5_select": ("protocol",),
    "c6_coordinate_source": ("protocol",),
    "c6_tier2": ("protocol",),
    "c6_tier1": ("span_records", "protocol"),
    "c4_final_cath": ("mmseqs_binary", "mmseqs_probe_manifest", "protocol"),
    "c8_tier1_finalize": ("span_records", "protocol"),
    "collision_dedup": ("protocol",),
    "c7_nmp": ("nmp_binary", "nmp_install_manifest", "nmp_probe_manifest", "protocol"),
    "c7_select": ("protocol",),
    "assemble_family": ("mmseqs_binary", "mmseqs_probe_manifest", "protocol"),
    "c9_head": (
        "head_checkpoint", "head_resolved_config", "head_run_summary",
        "head_model_yaml", "head_ablation_yaml", "head_inference_yaml",
        "head_probe_manifest", "protocol",
    ),
    "c9_homology": (
        "mmseqs_binary", "mmseqs_probe_manifest", "strict_train_ids",
        "strict_val_ids", "strict_test_ids", "protein_samples", "protocol",
    ),
    "release_validate": ("descendant_scope", "protocol", "runner_doc"),
}
if set(STAGE_INPUT_ROLES) != set(STAGE_BY_NAME):  # pragma: no cover - import-time invariant
    raise RuntimeError("production stage input registry is incomplete")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    if payload.get(field) != _self_hash(payload, field):
        raise ValueError(f"{label} self-hash mismatch")


def _file_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _validate_file_identity(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path") or "")).resolve()
    if (
        not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1))
        or _sha(path) != record.get("sha256")
    ):
        raise ValueError(f"{label} file identity mismatch")
    return path


def _default_probe_validators(inputs: Mapping[str, Path]) -> dict[str, Any]:
    """Replay all four production probes against the separately registered inputs.

    The returned receipt is serialized into the production contract.  It deliberately contains
    the scientific/tool parameters that matter to this release rather than a caller supplied
    boolean.  Both DAG creation and every later DAG validation regenerate this receipt.
    """

    from .head_annotation import validate_head_annotation
    from .smoke import (
        validate_mmseqs_probe_manifest,
        validate_nmp_probe_manifest,
        validate_rcsb_structure_probe_manifest,
    )

    nmp_probe = inputs["nmp_probe_manifest"]
    registered_install = inputs["nmp_install_manifest"]
    probe_install = nmp_probe.parent / "install_manifest.json"
    if (
        not probe_install.is_file()
        or _sha(probe_install) != _sha(registered_install)
    ):
        raise ValueError("registered NetMHCIIpan install manifest differs from probe evidence")
    install_payload = json.loads(registered_install.read_text())
    raw_install_root = Path(str(install_payload.get("data_root") or ""))
    install_root = (
        raw_install_root.resolve() if raw_install_root.is_absolute()
        else (registered_install.parent / raw_install_root).resolve()
    )
    nmp = validate_nmp_probe_manifest(
        nmp_probe, binary_path=inputs["nmp_binary"], install_root=install_root,
    )
    mmseqs = validate_mmseqs_probe_manifest(
        inputs["mmseqs_probe_manifest"], replay=False,
        tool_path=inputs["mmseqs_binary"],
    )
    structure = validate_rcsb_structure_probe_manifest(inputs["structure_probe_manifest"])

    head_manifest_path = inputs["head_probe_manifest"]
    head_manifest = json.loads(head_manifest_path.read_text())
    cohort_paths = []
    for record in head_manifest.get("cohort_inputs") or []:
        raw = Path(str(record.get("path") or ""))
        cohort_paths.append(
            raw.resolve() if raw.is_absolute()
            else (head_manifest_path.parent / raw).resolve()
        )
    config_paths = {
        "model.yaml": inputs["head_model_yaml"],
        "model_ablation.yaml": inputs["head_ablation_yaml"],
        "inference.yaml": inputs["head_inference_yaml"],
    }
    config_dirs = {path.parent.resolve() for path in config_paths.values()}
    if len(config_dirs) != 1 or any(path.name != name for name, path in config_paths.items()):
        raise ValueError("Head model config inputs do not form one canonical config directory")
    head = validate_head_annotation(
        manifest_path=head_manifest_path,
        checkpoint_path=inputs["head_checkpoint"],
        resolved_config_path=inputs["head_resolved_config"],
        run_summary_path=inputs["head_run_summary"],
        model_config_dir=config_dirs.pop(), cohort_input_paths=cohort_paths, replay=False,
    )
    return {
        "schema_version": "if-benchmark-v3-probe-validation/1",
        "nmp": {
            "manifest_sha256": nmp["manifest_sha256"],
            "install_manifest_sha256": nmp["install_manifest_sha256"],
            "binary_sha256": _sha(inputs["nmp_binary"]),
            "allele": nmp["allele"],
            "window_counts": {
                stage: int(nmp["stages"][stage]["window_count"])
                for stage in ("c5", "c7")
            },
        },
        "mmseqs": {
            "manifest_sha256": mmseqs["manifest_sha256"],
            "binary_sha256": _sha(inputs["mmseqs_binary"]),
            "cov_modes": {
                stage: int(mmseqs["stages"][stage]["cov_mode"])
                for stage in ("cath", "head")
            },
        },
        "head": {
            "manifest_sha256": head["manifest_sha256"],
            "checkpoint_sha256": head["checkpoint_sha256"],
            "resolved_config_sha256": head["resolved_config_sha256"],
            "fixed_epoch": int(head["fixed_epoch"]),
        },
        "structure": {
            "manifest_sha256": structure["manifest_sha256"],
            "rcsb_entity_id": structure["rcsb_entity_id"],
            "coordinate_cif_sha256": structure["coordinate_cif_sha256"],
            "passed_instances": sum(
                1 for row in structure["results"] if row.get("status") == "pass"
            ),
        },
    }


def _approved_logical_manifest_receipt(inputs: Mapping[str, Path]) -> dict[str, str]:
    receipt: dict[str, str] = {}
    for role, expected in APPROVED_LOGICAL_MANIFESTS.items():
        payload = json.loads(inputs[role].read_text())
        unsigned = dict(payload)
        logical = unsigned.pop("manifest_sha256", None)
        recomputed = hashlib.sha256(_canonical(unsigned)).hexdigest()
        if logical != expected or recomputed != expected:
            raise ValueError(f"approved logical manifest identity mismatch: {role}")
        receipt[role] = str(logical)
    return receipt


def _validate_contract_bound_sources(
    contract: Mapping[str, Any], *,
    probe_validator: Callable[[Mapping[str, Path]], Mapping[str, Any]] | None = None,
    replay_probes: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Rebuild immutable inputs/resources and typed probe receipt from source bytes."""

    registry_path = _validate_file_identity(
        contract.get("input_registry") or {}, "production input registry"
    )
    execution_path = _validate_file_identity(
        contract.get("execution_config") or {}, "production execution config"
    )
    registry = json.loads(registry_path.read_text())
    execution = json.loads(execution_path.read_text())
    if not isinstance(registry, Mapping) or set(registry) != REQUIRED_INPUT_ROLES:
        raise ValueError("production input registry role set changed")
    if set(execution) != {
        "array_shards", "resources", "stage_parameters", "current_c5_rung",
        "prior_rung_snapshots",
    }:
        raise ValueError("production execution config schema changed")
    records = {
        str(role): {"role": str(role), **_file_identity(Path(str(path)))}
        for role, path in sorted(registry.items())
    }
    paths = {role: Path(record["path"]) for role, record in records.items()}
    observed = contract.get("immutable_inputs")
    if observed != [records[role] for role in sorted(records)]:
        raise ValueError("production immutable inputs differ from bound registry")
    if (
        contract.get("array_shards") != execution["array_shards"]
        or contract.get("resources") != execution["resources"]
        or contract.get("stage_parameters") != execution["stage_parameters"]
        or int(contract.get("current_c5_rung", -1)) != int(execution["current_c5_rung"])
    ):
        raise ValueError("production resources/shards differ from bound execution config")
    prior_receipt = _validate_prior_rung_snapshots(execution)
    if contract.get("prior_rung_snapshots") != prior_receipt:
        raise ValueError("production prior-rung snapshot receipt changed")
    for role, digest in APPROVED_HASHES.items():
        if records[role]["sha256"] != digest:
            raise ValueError(f"approved production identity mismatch: {role}")
    logical_receipt = _approved_logical_manifest_receipt(paths)
    if contract.get("approved_logical_manifests") != logical_receipt:
        raise ValueError("production approved logical manifest receipt changed")
    receipt = contract.get("probe_validation")
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != (
        "if-benchmark-v3-probe-validation/1"
    ):
        raise ValueError("production typed probe validation receipt is missing")
    if replay_probes:
        observed_receipt = (probe_validator or _default_probe_validators)(paths)
        if receipt != observed_receipt:
            raise ValueError("production typed probe validation receipt changed")
    return records, paths


def _stage_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    shard_counts = contract.get("array_shards") or {}
    resources = contract.get("resources") or {}
    return [
        {
            "name": stage.name,
            "dependencies": list(stage.dependencies),
            "execution": stage.execution,
            "task_count": 1 if stage.shard_key is None else int(shard_counts[stage.shard_key]),
            "resource_class": stage.resource_class,
            "resources": dict(resources[stage.resource_class]),
        }
        for stage in STAGES
    ]


def _validate_stage_parameters(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"nmp", "head"}:
        raise ValueError("production stage parameter classes are incomplete")
    nmp = value["nmp"]
    head = value["head"]
    if (
        not isinstance(nmp, Mapping)
        or set(nmp) != {
            "batch_size", "subprocess_timeout_s", "max_lengths_per_call", "n_workers"
        }
        or any(int(nmp[key]) < 1 for key in nmp)
        or not isinstance(head, Mapping)
        or set(head) != {"window_batch_size", "device"}
        or int(head["window_batch_size"]) < 1
        or head["device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("production stage parameter contract is invalid")


def _validate_failed_rung_snapshot_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    root = Path(str(payload.get("candidate_root") or "")).resolve()
    if (
        payload.get("schema_version") != "if-benchmark-v3-failed-rung-snapshot/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or path.parent != root or int(payload.get("target", -1)) < 1
    ):
        raise ValueError("failed-rung snapshot manifest identity mismatch")
    observed = [
        {
            "path": str(item.relative_to(root)), "size_bytes": item.stat().st_size,
            "sha256": _sha(item),
        }
        for item in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)))
        if item.is_file() and item != path
    ]
    if payload.get("artifacts") != observed:
        raise ValueError("failed-rung snapshot artifact table changed")
    return payload


def _validate_prior_rung_snapshots(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = int(execution.get("current_c5_rung", -1))
    ladder = [5000, 7500, 10000]
    if current not in ladder:
        raise ValueError("current C5 rung is outside the preregistered ladder")
    raw = execution.get("prior_rung_snapshots")
    if not isinstance(raw, list):
        raise ValueError("prior C5 rung snapshots must be a list")
    expected_targets = ladder[:ladder.index(current)]
    if [int(record.get("target", -1)) for record in raw] != expected_targets:
        raise ValueError("prior C5 rung snapshots do not form the exact ladder prefix")
    receipt = []
    for record, target in zip(raw, expected_targets, strict=True):
        if set(record) != {"target", "snapshot_manifest"}:
            raise ValueError("prior C5 rung snapshot descriptor fields are invalid")
        manifest_path = Path(str(record["snapshot_manifest"])).resolve()
        payload = _validate_failed_rung_snapshot_manifest(manifest_path)
        if int(payload["target"]) != target:
            raise ValueError("prior C5 rung snapshot target mismatch")
        receipt.append({
            "target": target, "candidate_root": str(Path(payload["candidate_root"]).resolve()),
            "snapshot_manifest": _file_identity(manifest_path),
        })
    return receipt


def _dag_unsigned_payload(
    *, contract_path: Path, contract: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "if-benchmark-v3-production-dag/1",
        "protocol_id": PROTOCOL_ID,
        "build_id": str(contract["build_id"]),
        "allele": ALLELE,
        "allele_tag": ALLELE_TAG,
        "candidate_root": str(Path(str(contract["candidate_root"])).resolve()),
        "contract": _file_identity(contract_path),
        "identity_profile": str(contract["identity_profile"]),
        "targets": contract["targets"],
        "c5_ladder": contract["c5_ladder"],
        "seed": int(contract["seed"]),
        "stage_parameters": contract["stage_parameters"],
        "current_c5_rung": int(contract["current_c5_rung"]),
        "prior_rung_snapshots": contract["prior_rung_snapshots"],
        "immutable_inputs": {role: dict(record) for role, record in input_records.items()},
        "stages": _stage_rows(contract),
    }


def _normalize_task_descriptor(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    allowed = {
        "task_index", "producer_code_path", "code_paths", "command_argv",
        "input_paths", "output_roots", "required_outputs",
    }
    if set(raw) != allowed:
        raise ValueError("production task descriptor field set is invalid")
    command = raw.get("command_argv")
    producer = Path(str(raw.get("producer_code_path") or "")).resolve()
    code_paths = [str(Path(path).resolve()) for path in raw.get("code_paths") or []]
    inputs = [str(Path(path).resolve()) for path in raw.get("input_paths") or []]
    roots = [str(value) for value in raw.get("output_roots") or []]
    required = [str(value) for value in raw.get("required_outputs") or []]
    relative_paths = [Path(value) for value in roots + required]
    if (
        int(raw.get("task_index", -1)) != index
        or not isinstance(command, list) or not command
        or not all(isinstance(value, str) and value for value in command)
        or not producer.is_file() or str(producer) not in command
        or not code_paths or str(producer) not in code_paths
        or len(code_paths) != len(set(code_paths))
        or any(not Path(path).is_file() for path in code_paths)
        or not inputs or len(inputs) != len(set(inputs))
        or any(not Path(path).is_file() for path in inputs)
        or not roots or not required
        or len(roots) != len(set(roots)) or len(required) != len(set(required))
        or any(path.is_absolute() or ".." in path.parts or not path.parts for path in relative_paths)
        or any(not any(Path(root) == path or Path(root) in path.parents for root in roots)
               for path in map(Path, required))
    ):
        raise ValueError("invalid production task descriptor")
    return {
        "task_index": index,
        "producer_code_path": str(producer),
        "code_paths": code_paths,
        "command_argv": list(command),
        "input_paths": inputs,
        "output_roots": roots,
        "required_outputs": required,
    }


def _production_task_descriptors(
    *, dag_path: Path, dag: Mapping[str, Any], stage: str,
) -> list[dict[str, Any]]:
    """Generate the only legal production argv/path contract for one complete stage array."""

    if dag.get("identity_profile") != "production":
        raise ValueError("automatic production task generation requires production identity")
    from .stage_execution import PRODUCERS
    if stage not in PRODUCERS:
        raise ValueError(f"production scientific producer is not implemented: {stage}")
    inputs = dag["immutable_inputs"]
    runner = _validate_file_identity(inputs["runner_code"], "production task runner")
    python = _validate_file_identity(inputs["python_executable"], "production task Python")
    expected_runner = Path(__file__).parents[3] / "scripts" / "build_if_benchmark_v3.py"
    if runner != expected_runner.resolve():
        raise ValueError("production task runner is not the registered v3 CLI")
    package_root = Path(__file__).resolve().parent
    code_paths = [runner, *sorted(package_root.glob("*.py"))]
    code_strings = list(dict.fromkeys(str(path.resolve()) for path in code_paths))
    row = next(item for item in dag["stages"] if item["name"] == stage)
    task_count = int(row["task_count"])
    input_paths = [
        str(_validate_file_identity(inputs[role], f"production task {stage} input {role}"))
        for role in STAGE_INPUT_ROLES[stage]
    ]
    descriptors = []
    for index in range(task_count):
        descriptors.append(_normalize_task_descriptor({
            "task_index": index,
            "producer_code_path": str(runner),
            "code_paths": code_strings,
            "command_argv": [
                str(python), str(runner), "execute-production-stage",
                "--dag", str(Path(dag_path).resolve()), "--stage", stage,
                "--task-index", str(index), "--task-count", str(task_count),
                "--output-dir", "{task_dir}/payload",
            ],
            "input_paths": input_paths,
            "output_roots": ["payload"],
            "required_outputs": ["payload/stage_result.json"],
        }, index=index))
    return descriptors


def write_production_contract(
    *, build_id: str, candidate_root: Path, input_registry_path: Path,
    execution_config_path: Path, output_path: Path,
    identity_profile: str = "production", allow_fixture: bool = False,
    probe_validator: Callable[[Mapping[str, Path]], Mapping[str, Any]] | None = None,
) -> Path:
    """Resolve every registered input to bytes and freeze targets/resources before scoring."""

    if identity_profile != "production" and not (
        allow_fixture and identity_profile == "fixture"
    ):
        raise ValueError("production contract producer forbids fixture identity")
    candidate_root = Path(candidate_root).resolve()
    if (
        not SAFE_ID.fullmatch(str(build_id)) or candidate_root.name != ALLELE_TAG
        or candidate_root.parent.name != str(build_id)
        or "if_ready/main" in candidate_root.as_posix()
    ):
        raise ValueError("invalid production build/candidate identity")
    input_registry_path = Path(input_registry_path).resolve()
    execution_config_path = Path(execution_config_path).resolve()
    registry = json.loads(input_registry_path.read_text())
    execution = json.loads(execution_config_path.read_text())
    if not isinstance(registry, Mapping) or not registry:
        raise ValueError("production input registry must be a non-empty role/path mapping")
    roles = set(str(role) for role in registry)
    if identity_profile == "production" and roles != REQUIRED_INPUT_ROLES:
        raise ValueError("production input registry role set is incomplete or has extras")
    inputs = []
    for role in sorted(roles):
        identity = _file_identity(Path(str(registry[role])))
        inputs.append({"role": role, **identity})
    if identity_profile == "production":
        by_role = {record["role"]: record for record in inputs}
        for role, digest in APPROVED_HASHES.items():
            if by_role[role]["sha256"] != digest:
                raise ValueError(f"approved production identity mismatch: {role}")
    if set(execution) != {
        "array_shards", "resources", "stage_parameters", "current_c5_rung",
        "prior_rung_snapshots",
    }:
        raise ValueError(
            "production execution config requires shards/resources/stage_parameters"
        )
    _validate_stage_parameters(execution["stage_parameters"])
    prior_rung_snapshots = _validate_prior_rung_snapshots(execution)
    input_paths = {record["role"]: Path(record["path"]) for record in inputs}
    probe_validation = (
        (probe_validator or _default_probe_validators)(input_paths)
        if identity_profile == "production"
        else {
            "schema_version": "if-benchmark-v3-probe-validation/1",
            "status": "fixture_not_validated",
        }
    )
    approved_logical_manifests = (
        _approved_logical_manifest_receipt(input_paths)
        if identity_profile == "production" else {}
    )
    output_path = Path(output_path).resolve()
    if candidate_root not in output_path.parents or output_path.exists():
        raise ValueError("production contract output path is not a fresh candidate artifact")
    payload = {
        "schema_version": "if-benchmark-v3-production-contract/1",
        "identity_profile": identity_profile, "protocol_id": PROTOCOL_ID,
        "build_id": str(build_id), "allele": ALLELE, "allele_tag": ALLELE_TAG,
        "candidate_root": str(candidate_root),
        "targets": {"tier1": 15, "tier2": 3000},
        "c5_ladder": [5000, 7500, 10000], "seed": 42,
        "input_registry": _file_identity(input_registry_path),
        "execution_config": _file_identity(execution_config_path),
        "immutable_inputs": inputs,
        "array_shards": execution["array_shards"],
        "resources": execution["resources"],
        "stage_parameters": execution["stage_parameters"],
        "current_c5_rung": int(execution["current_c5_rung"]),
        "prior_rung_snapshots": prior_rung_snapshots,
        "probe_validation": probe_validation,
        "approved_logical_manifests": approved_logical_manifests,
    }
    payload["contract_sha256"] = _self_hash(payload, "contract_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def write_production_dag(
    *, contract_path: Path, output_path: Path,
    probe_validator: Callable[[Mapping[str, Path]], Mapping[str, Any]] | None = None,
    allow_fixture: bool = False,
) -> Path:
    """Validate a frozen build contract and materialize the only legal stage DAG."""

    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    mode = str(contract.get("identity_profile") or "production")
    if mode != "production" and not (allow_fixture and mode == "fixture"):
        raise ValueError("production DAG forbids nonproduction identity profiles")
    if mode == "production":
        _validate_self_hash(contract, "contract_sha256", "production contract")
        _validate_file_identity(contract.get("input_registry") or {}, "input registry")
        _validate_file_identity(contract.get("execution_config") or {}, "execution config")
    elif contract.get("contract_sha256") is not None:
        _validate_self_hash(contract, "contract_sha256", "fixture production contract")
        if contract.get("input_registry") is not None:
            _validate_file_identity(contract["input_registry"], "fixture input registry")
        if contract.get("execution_config") is not None:
            _validate_file_identity(contract["execution_config"], "fixture execution config")
    build_id = str(contract.get("build_id") or "")
    candidate_root = Path(str(contract.get("candidate_root") or "")).resolve()
    if (
        contract.get("schema_version") != "if-benchmark-v3-production-contract/1"
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("allele") != ALLELE or contract.get("allele_tag") != ALLELE_TAG
        or not SAFE_ID.fullmatch(build_id) or candidate_root.name != ALLELE_TAG
        or candidate_root.parent.name != build_id
        or "if_ready/main" in candidate_root.as_posix()
        or contract.get("targets") != {"tier1": 15, "tier2": 3000}
        or contract.get("c5_ladder") != [5000, 7500, 10000]
        or int(contract.get("seed", -1)) != 42
        or int(contract.get("current_c5_rung", -1)) not in [5000, 7500, 10000]
    ):
        raise ValueError("production build contract differs from the frozen v3 release contract")
    raw_inputs = contract.get("immutable_inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("production build contract lacks immutable inputs")
    input_records: dict[str, dict[str, Any]] = {}
    input_paths: dict[str, Path] = {}
    for raw in raw_inputs:
        role = str(raw.get("role") or "")
        if not role or role in input_records:
            raise ValueError("production immutable input roles are missing/duplicated")
        path = _validate_file_identity(raw, f"production input {role}")
        input_records[role] = dict(raw)
        input_paths[role] = path
    if mode == "production":
        if set(input_records) != REQUIRED_INPUT_ROLES:
            raise ValueError("production immutable input role set is incomplete or has extras")
        for role, expected in APPROVED_HASHES.items():
            if input_records[role]["sha256"] != expected:
                raise ValueError(f"approved production identity mismatch: {role}")
        rebound_records, rebound_paths = _validate_contract_bound_sources(
            contract, probe_validator=probe_validator, replay_probes=False,
        )
        if rebound_records != input_records or rebound_paths != input_paths:
            raise ValueError("production contract input reconstruction mismatch")
    shard_counts = contract.get("array_shards") or {}
    required_shards = {
        stage.shard_key for stage in STAGES if stage.shard_key is not None
    }
    if set(shard_counts) != required_shards or any(
        not isinstance(shard_counts[key], int) or not 1 <= int(shard_counts[key]) <= 4096
        for key in required_shards
    ):
        raise ValueError("production array shard contract is incomplete/invalid")
    resources = contract.get("resources") or {}
    if set(resources) != {"network", "cpu", "gpu"}:
        raise ValueError("production resources require network/cpu/gpu classes")
    for resource_class, record in resources.items():
        required = {
            "partition", "qos", "cpus", "memory_gb", "time", "max_parallel", "gres"
        }
        if (
            set(record) != required or not str(record["partition"])
            or not str(record["qos"]) or int(record["cpus"]) < 1
            or int(record["memory_gb"]) < 1 or not str(record["time"])
            or int(record["max_parallel"]) < 1 or not isinstance(record["gres"], str)
        ):
            raise ValueError(f"invalid production resource class: {resource_class}")
    _validate_stage_parameters(contract.get("stage_parameters"))
    output_path = Path(output_path).resolve()
    if candidate_root not in output_path.parents:
        raise ValueError("production DAG must live inside its candidate root")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = _dag_unsigned_payload(
        contract_path=contract_path, contract=contract, input_records=input_records,
    )
    payload["dag_sha256"] = _self_hash(payload, "dag_sha256")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def validate_production_dag(
    path: Path, *,
    probe_validator: Callable[[Mapping[str, Path]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    _validate_self_hash(payload, "dag_sha256", "production DAG")
    if (
        payload.get("schema_version") != "if-benchmark-v3-production-dag/1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("allele") != ALLELE or payload.get("allele_tag") != ALLELE_TAG
        or [row.get("name") for row in payload.get("stages") or []]
        != [stage.name for stage in STAGES]
    ):
        raise ValueError("production DAG schema/stage order mismatch")
    contract_path = _validate_file_identity(payload["contract"], "production DAG contract")
    contract = json.loads(contract_path.read_text())
    if contract.get("contract_sha256") is not None:
        _validate_self_hash(contract, "contract_sha256", "production DAG contract")
    if contract.get("input_registry") is not None:
        _validate_file_identity(contract["input_registry"], "production DAG input registry")
    if contract.get("execution_config") is not None:
        _validate_file_identity(contract["execution_config"], "production DAG execution config")
    mode = str(contract.get("identity_profile") or "production")
    contract_inputs: dict[str, dict[str, Any]] = {}
    for raw in contract.get("immutable_inputs") or []:
        role = str(raw.get("role") or "")
        if not role or role in contract_inputs:
            raise ValueError("production DAG contract input roles are invalid")
        _validate_file_identity(raw, f"production DAG contract input {role}")
        contract_inputs[role] = dict(raw)
    if mode == "production":
        rebound, _paths = _validate_contract_bound_sources(
            contract, probe_validator=probe_validator, replay_probes=False,
        )
        if rebound != contract_inputs:
            raise ValueError("production DAG contract input reconstruction mismatch")
    for role, record in payload.get("immutable_inputs", {}).items():
        _validate_file_identity(record, f"production DAG input {role}")
    candidate_root = Path(payload["candidate_root"]).resolve()
    if candidate_root not in path.parents or "if_ready/main" in candidate_root.as_posix():
        raise ValueError("production DAG candidate root/path mismatch")
    expected = _dag_unsigned_payload(
        contract_path=contract_path, contract=contract, input_records=contract_inputs,
    )
    observed = dict(payload)
    observed.pop("dag_sha256", None)
    if observed != expected:
        raise ValueError("production DAG differs from canonical contract reconstruction")
    return payload


def write_production_task_table(
    *, dag_path: Path, stage: str, output_path: Path,
    tasks: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    dag = validate_production_dag(dag_path)
    if stage not in STAGE_BY_NAME:
        raise ValueError("unknown production stage")
    stage_row = next(row for row in dag["stages"] if row["name"] == stage)
    if dag.get("identity_profile") == "production":
        generated = _production_task_descriptors(
            dag_path=Path(dag_path).resolve(), dag=dag, stage=stage,
        )
        if tasks is not None:
            supplied = [
                _normalize_task_descriptor(raw, index=index)
                for index, raw in enumerate(tasks)
            ]
            if supplied != generated:
                raise ValueError("manual production task descriptors differ from stage registry")
        tasks = generated
    elif tasks is None:
        raise ValueError("fixture task tables require explicit descriptors")
    if len(tasks) != int(stage_row["task_count"]):
        raise ValueError("production task table cardinality differs from DAG")
    normalized = []
    for index, raw in enumerate(tasks):
        record = _normalize_task_descriptor(raw, index=index)
        record["task_sha256"] = _self_hash(record, "task_sha256")
        normalized.append(record)
    payload = {
        "schema_version": "if-benchmark-v3-production-task-table/1",
        "dag_path": str(Path(dag_path).resolve()),
        "dag_sha256": dag["dag_sha256"], "stage": stage, "tasks": normalized,
    }
    payload["table_sha256"] = _self_hash(payload, "table_sha256")
    output_path = Path(output_path).resolve()
    if Path(dag["candidate_root"]).resolve() not in output_path.parents:
        raise ValueError("production task table escapes candidate root")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def validate_production_task_table(path: Path, *, dag_path: Path, stage: str) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    _validate_self_hash(payload, "table_sha256", "production task table")
    dag = validate_production_dag(dag_path)
    stage_row = next(row for row in dag["stages"] if row["name"] == stage)
    if (
        payload.get("schema_version") != "if-benchmark-v3-production-task-table/1"
        or payload.get("dag_path") != str(Path(dag_path).resolve())
        or payload.get("dag_sha256") != dag["dag_sha256"]
        or payload.get("stage") != stage
        or len(payload.get("tasks") or []) != int(stage_row["task_count"])
    ):
        raise ValueError("production task table DAG/stage identity mismatch")
    for index, record in enumerate(payload["tasks"]):
        _validate_self_hash(record, "task_sha256", "production task")
        unsigned = dict(record)
        unsigned.pop("task_sha256", None)
        normalized = _normalize_task_descriptor(unsigned, index=index)
        if unsigned != normalized:
            raise ValueError("production task descriptor differs from normalized contract")
    if dag.get("identity_profile") == "production":
        expected = _production_task_descriptors(
            dag_path=Path(dag_path).resolve(), dag=dag, stage=stage,
        )
        expected_with_hash = []
        for record in expected:
            record = dict(record)
            record["task_sha256"] = _self_hash(record, "task_sha256")
            expected_with_hash.append(record)
        if payload["tasks"] != expected_with_hash:
            raise ValueError("production task table differs from registered stage contract")
    return payload


def _stage_root(dag: Mapping[str, Any], stage: str) -> Path:
    return Path(dag["candidate_root"]) / "audit" / "orchestrator" / "stages" / stage


def _validate_stage_seal(
    path: Path, *, dag_path: Path, dag: Mapping[str, Any], stage: str,
    deep: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    _validate_self_hash(payload, "seal_sha256", "production stage seal")
    if (
        payload.get("schema_version") != "if-benchmark-v3-production-stage-seal/1"
        or payload.get("dag_sha256") != dag["dag_sha256"]
        or payload.get("stage") != stage or payload.get("status") != "complete"
    ):
        raise ValueError("production stage seal identity mismatch")
    task_table_path = _validate_file_identity(
        payload.get("task_table") or {}, "production stage task table"
    )
    table = validate_production_task_table(
        task_table_path, dag_path=dag_path, stage=stage,
    )
    journals = payload.get("task_journals") or []
    if len(journals) != len(table["tasks"]):
        raise ValueError("production stage seal task count mismatch")
    for task, journal_record in zip(table["tasks"], journals, strict=True):
        journal_path = _validate_file_identity(
            journal_record, "production stage task journal"
        )
        expected_path = (
            _stage_root(dag, stage) / f"{int(task['task_index']):05d}" /
            "stage_identity.json"
        )
        if journal_path != expected_path.resolve():
            raise ValueError("production stage seal journal path mismatch")
        if deep:
            identity = _task_identity(
                dag_path=dag_path, table_path=task_table_path, dag=dag,
                stage=stage, record=task,
            )
            StageJournal(journal_path.parent).run(
                identity, lambda _path: (), resume=True,
            )
    return payload


def _validate_stage_failure(
    path: Path, *, dag: Mapping[str, Any], stage: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    _validate_self_hash(payload, "failure_sha256", "production stage failure")
    if (
        payload.get("schema_version") != "if-benchmark-v3-production-stage-failure/1"
        or payload.get("dag_sha256") != dag["dag_sha256"]
        or payload.get("stage") != stage
        or payload.get("status") != "scientific_failed"
        or payload.get("reason") != "insufficient_capacity"
    ):
        raise ValueError("production stage failure identity mismatch")
    result = _validate_file_identity(
        payload.get("stage_result") or {}, "production failed stage result"
    )
    from .stage_execution import validate_stage_result

    task_index = int(payload.get("task_index", -1))
    observed = validate_stage_result(
        result, dag=dag, expected_stage=stage, expected_index=task_index,
    )
    if observed.get("summary", {}).get("scientific_status") != "insufficient_capacity":
        raise ValueError("production stage failure lacks measured capacity evidence")
    return payload


def _task_identity(
    *, dag_path: Path, table_path: Path, dag: Mapping[str, Any], stage: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies = STAGE_BY_NAME[stage].dependencies
    input_paths = [Path(path).resolve() for path in record["input_paths"]]
    seals = []
    for dependency in dependencies:
        seal_path = _stage_root(dag, dependency) / "stage_complete.json"
        _validate_stage_seal(
            seal_path, dag_path=dag_path, dag=dag, stage=dependency, deep=False,
        )
        seals.append(seal_path)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "code": _file_identity(Path(record["producer_code_path"])),
        "config": _file_identity(Path(dag_path)),
        "inputs": [
            _file_identity(Path(table_path)),
            *[
                _file_identity(Path(path))
                for path in record["code_paths"]
                if Path(path).resolve() != Path(record["producer_code_path"]).resolve()
            ],
            *[_file_identity(path) for path in input_paths],
            *[_file_identity(path) for path in seals],
        ],
    }


def run_production_task(
    *, dag_path: Path, task_table_path: Path, stage: str, task_index: int,
    resume: bool,
) -> dict[str, Any]:
    dag_path, task_table_path = Path(dag_path).resolve(), Path(task_table_path).resolve()
    dag = validate_production_dag(dag_path)
    table = validate_production_task_table(task_table_path, dag_path=dag_path, stage=stage)
    if not 0 <= int(task_index) < len(table["tasks"]):
        raise ValueError("production task index outside array cardinality")
    record = table["tasks"][int(task_index)]
    task_dir = _stage_root(dag, stage) / f"{int(task_index):05d}"
    identity = _task_identity(
        dag_path=dag_path, table_path=task_table_path, dag=dag,
        stage=stage, record=record,
    )

    def producer(stage_dir: Path) -> Sequence[Path]:
        command = [
            value.replace("{task_dir}", str(stage_dir.resolve()))
            for value in record["command_argv"]
        ]
        completed = subprocess.run(
            command, cwd=stage_dir, text=True, capture_output=True, check=False,
            env={**os.environ, "PYTHONHASHSEED": "42", "LC_ALL": "C"},
        )
        stdout_path = stage_dir / "command.stdout.txt"
        stderr_path = stage_dir / "command.stderr.txt"
        command_path = stage_dir / "command.json"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        command_payload = {
            "argv": command, "returncode": int(completed.returncode),
            "task_sha256": record["task_sha256"],
        }
        command_path.write_text(json.dumps(command_payload, indent=2, sort_keys=True) + "\n")
        if completed.returncode != 0:
            raise RuntimeError(f"production stage command failed rc={completed.returncode}")
        for required in record["required_outputs"]:
            if not (stage_dir / required).is_file():
                raise ValueError(f"production task required output missing: {required}")
        files = [stdout_path, stderr_path, command_path]
        for output_root in record["output_roots"]:
            root = stage_dir / output_root
            if not root.exists():
                raise ValueError(f"production task output root missing: {output_root}")
            files.extend(path for path in root.rglob("*") if path.is_file())
        if len({path.resolve() for path in files}) != len(files):
            raise ValueError("production task output roots overlap")
        return files

    return StageJournal(task_dir).run(identity, producer, resume=bool(resume))


def seal_production_stage(
    *, dag_path: Path, task_table_path: Path, stage: str,
) -> Path:
    dag_path, task_table_path = Path(dag_path).resolve(), Path(task_table_path).resolve()
    dag = validate_production_dag(dag_path)
    table = validate_production_task_table(task_table_path, dag_path=dag_path, stage=stage)
    identities = []
    for record in table["tasks"]:
        index = int(record["task_index"])
        task_dir = _stage_root(dag, stage) / f"{index:05d}"
        identity = _task_identity(
            dag_path=dag_path, table_path=task_table_path, dag=dag,
            stage=stage, record=record,
        )
        StageJournal(task_dir).run(identity, lambda _path: (), resume=True)
        identities.append(_file_identity(task_dir / "stage_identity.json"))
    if stage == "c7_select":
        from .stage_execution import validate_stage_result

        failed_results = []
        for record in table["tasks"]:
            index = int(record["task_index"])
            result_path = (
                _stage_root(dag, stage) / f"{index:05d}" / "payload" /
                "stage_result.json"
            )
            result = validate_stage_result(
                result_path, dag=dag, expected_stage=stage, expected_index=index,
            )
            if result.get("summary", {}).get("scientific_status") == "insufficient_capacity":
                failed_results.append((index, result_path))
        if failed_results:
            if len(failed_results) != 1 or len(table["tasks"]) != 1:
                raise ValueError("C7 capacity failure must be one complete reducer task")
            failure_path = _stage_root(dag, stage) / "stage_failed.json"
            if failure_path.exists():
                _validate_stage_failure(failure_path, dag=dag, stage=stage)
                return failure_path
            index, result_path = failed_results[0]
            payload = {
                "schema_version": "if-benchmark-v3-production-stage-failure/1",
                "dag_sha256": dag["dag_sha256"], "stage": stage,
                "status": "scientific_failed", "reason": "insufficient_capacity",
                "task_index": index, "stage_result": _file_identity(result_path),
                "task_table": _file_identity(task_table_path),
                "task_journals": identities,
            }
            payload["failure_sha256"] = _self_hash(payload, "failure_sha256")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = failure_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, failure_path)
            return failure_path
    seal_path = _stage_root(dag, stage) / "stage_complete.json"
    if seal_path.exists():
        _validate_stage_seal(seal_path, dag_path=dag_path, dag=dag, stage=stage)
        return seal_path
    payload = {
        "schema_version": "if-benchmark-v3-production-stage-seal/1",
        "dag_sha256": dag["dag_sha256"], "stage": stage, "status": "complete",
        "task_table": _file_identity(task_table_path), "task_journals": identities,
    }
    payload["seal_sha256"] = _self_hash(payload, "seal_sha256")
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = seal_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, seal_path)
    return seal_path


def production_status(*, dag_path: Path, deep: bool = True) -> dict[str, str]:
    dag_path = Path(dag_path).resolve()
    dag = validate_production_dag(dag_path)
    status = {}
    for stage in STAGES:
        seal = _stage_root(dag, stage.name) / "stage_complete.json"
        failure = _stage_root(dag, stage.name) / "stage_failed.json"
        if failure.is_file():
            _validate_stage_failure(failure, dag=dag, stage=stage.name)
            status[stage.name] = "failed"
            continue
        if seal.is_file():
            _validate_stage_seal(
                seal, dag_path=dag_path, dag=dag, stage=stage.name, deep=deep,
            )
            status[stage.name] = "complete"
            continue
        dependencies_complete = all(status.get(dep) == "complete" for dep in stage.dependencies)
        status[stage.name] = "ready" if dependencies_complete else "blocked"
    return status


def render_production_stage_commands(
    *, dag_path: Path, task_table_path: Path, stage: str, resume: bool = False,
) -> dict[str, Any]:
    """Render (never submit) the exact login or SLURM-array commands for one ready stage."""

    dag_path, task_table_path = Path(dag_path).resolve(), Path(task_table_path).resolve()
    dag = validate_production_dag(dag_path)
    table = validate_production_task_table(task_table_path, dag_path=dag_path, stage=stage)
    status = production_status(dag_path=dag_path)
    if status.get(stage) != "ready":
        raise ValueError(f"production stage is not ready: {stage}={status.get(stage)}")
    row = next(item for item in dag["stages"] if item["name"] == stage)
    inputs = dag["immutable_inputs"]
    for role in ("runner_code", "slurm_launcher", "python_executable"):
        if role not in inputs:
            raise ValueError(f"production command rendering lacks {role}")
        _validate_file_identity(inputs[role], f"production command {role}")
    runner = str(Path(inputs["runner_code"]["path"]).resolve())
    launcher = str(Path(inputs["slurm_launcher"]["path"]).resolve())
    python = str(Path(inputs["python_executable"]["path"]).resolve())
    resume_flag = "1" if resume else "0"
    if row["execution"] == "networked_login":
        commands = [
            [
                python, runner, "run-production-task", "--dag", str(dag_path),
                "--task-table", str(task_table_path), "--stage", stage,
                "--task-index", str(index), *( ["--resume"] if resume else [] ),
            ]
            for index in range(int(row["task_count"]))
        ]
        commands.append([
            python, runner, "seal-production-stage", "--dag", str(dag_path),
            "--task-table", str(task_table_path), "--stage", stage,
        ])
        return {"stage": stage, "execution": "networked_login", "commands": commands}
    resource = row["resources"]
    exports = ",".join([
        "ALL", "MODE=production_task", f"DAG_MANIFEST={dag_path}",
        f"TASK_TABLE={task_table_path}", f"STAGE={stage}", f"RESUME={resume_flag}",
        f"PROJECT_ROOT={Path(runner).parents[1]}", "CONDA_ENV=immune-design",
    ])
    array = f"0-{int(row['task_count']) - 1}%{int(resource['max_parallel'])}"
    common = [
        "sbatch", "--parsable", f"--job-name=ifD-{stage[:20]}",
        f"--partition={resource['partition']}", f"--qos={resource['qos']}",
        f"--cpus-per-task={int(resource['cpus'])}",
        f"--mem={int(resource['memory_gb'])}G", f"--time={resource['time']}",
    ]
    if resource["gres"]:
        common.append(f"--gres={resource['gres']}")
    array_command = [*common, f"--array={array}", f"--export={exports}", launcher]
    seal_exports = ",".join([
        "ALL", "MODE=production_seal", f"DAG_MANIFEST={dag_path}",
        f"TASK_TABLE={task_table_path}", f"STAGE={stage}",
        f"PROJECT_ROOT={Path(runner).parents[1]}", "CONDA_ENV=immune-design",
    ])
    seal_resource = next(
        item["resources"] for item in dag["stages"]
        if item["resource_class"] == "cpu"
    )
    seal_common = [
        "sbatch", "--parsable", f"--job-name=ifD-{stage[:20]}-seal",
        f"--partition={seal_resource['partition']}",
        f"--qos={seal_resource['qos']}", "--cpus-per-task=1",
        f"--mem={min(4, int(seal_resource['memory_gb']))}G", "--time=00:20:00",
    ]
    seal_command = [
        *seal_common, "--dependency=afterok:${ARRAY_JOB_ID}",
        f"--export={seal_exports}", launcher,
    ]
    return {
        "stage": stage, "execution": "slurm", "array_job_assignment": "ARRAY_JOB_ID",
        "array_command": array_command, "seal_command": seal_command,
        "task_table_sha256": table["table_sha256"],
    }
