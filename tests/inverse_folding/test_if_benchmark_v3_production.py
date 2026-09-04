"""Production DAG, SLURM-array task, and resume contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

import inverse_folding.evaluation.if_benchmark_v3.production as production
import inverse_folding.evaluation.if_benchmark_v3.stage_execution as stage_execution
from inverse_folding.evaluation.if_benchmark_v3.production import (
    STAGES,
    production_status,
    render_production_stage_commands,
    run_production_task,
    seal_production_stage,
    validate_production_dag,
    write_production_contract,
    write_production_dag,
    write_production_task_table,
)


def _self_hash(payload, field):
    unsigned = dict(payload)
    unsigned.pop(field, None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity(path, role):
    return {
        "role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_approved_fixture(path, role, monkeypatch):
    if role in production.APPROVED_LOGICAL_MANIFESTS:
        unsigned = {"schema_version": "fixture-logical-manifest/1"}
        logical = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode()).hexdigest()
        path.write_text(json.dumps({**unsigned, "manifest_sha256": logical}))
        monkeypatch.setitem(production.APPROVED_LOGICAL_MANIFESTS, role, logical)
    else:
        path.write_text(f"{role}\n")


def _fixture_contract(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    build_id = "fixture-build"
    candidate = tmp_path / "builds" / build_id / "HLA-DRB1_15_01"
    source = tmp_path / "source.txt"
    source.write_text("source")
    runner = tmp_path / "build_if_benchmark_v3.py"
    launcher = tmp_path / "submit_if_benchmark_v3.slurm"
    runner.write_text("# runner\n")
    launcher.write_text("#!/bin/bash\n")
    contract = {
        "schema_version": "if-benchmark-v3-production-contract/1",
        "identity_profile": "fixture",
        "protocol_id": "if-benchmark-test-set/3", "build_id": build_id,
        "allele": "HLA-DRB1*15:01", "allele_tag": "HLA-DRB1_15_01",
        "candidate_root": str(candidate),
        "targets": {"tier1": 15, "tier2": 3000},
        "c5_ladder": [5000, 7500, 10000], "seed": 42,
        "immutable_inputs": [
            _identity(source, "fixture_source"), _identity(runner, "runner_code"),
            _identity(launcher, "slurm_launcher"),
            _identity(Path("/usr/bin/python3"), "python_executable"),
        ],
        "array_shards": {
            "c5_nmp_shards": 2, "c6_tier2_shards": 3,
            "c6_tier1_shards": 1, "c7_nmp_shards": 2,
        },
        "resources": {
            "network": {
                "partition": "login", "qos": "none", "cpus": 1,
                "memory_gb": 4, "time": "02:00:00", "max_parallel": 1, "gres": "",
            },
            "cpu": {
                "partition": "cpu", "qos": "normal", "cpus": 4,
                "memory_gb": 16, "time": "04:00:00", "max_parallel": 8, "gres": "",
            },
            "gpu": {
                "partition": "rtx6000", "qos": "gpu-short", "cpus": 4,
                "memory_gb": 16, "time": "00:30:00", "max_parallel": 1,
                "gres": "gpu:rtx_pro_6000:1",
            },
        },
        "stage_parameters": {
            "nmp": {
                "batch_size": 32, "subprocess_timeout_s": 600,
                "max_lengths_per_call": 4, "n_workers": 1,
            },
            "head": {"window_batch_size": 1024, "device": "cuda"},
        },
        "current_c5_rung": 5000,
        "prior_rung_snapshots": [],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    dag_path = candidate / "audit" / "orchestrator" / "production_dag.json"
    return contract, contract_path, dag_path, source


def _real_profile_dag(
    tmp_path, monkeypatch, *, current_c5_rung=5000, prior_rung_snapshots=None,
):
    fixture, _old_contract, dag_path, _source = _fixture_contract(tmp_path)
    project = Path(__file__).parents[2]
    special = {
        "runner_code": project / "scripts" / "build_if_benchmark_v3.py",
        "slurm_launcher": project / "scripts" / "submit_if_benchmark_v3.slurm",
        "python_executable": Path("/home/zc1519/.conda/envs/immune-design/bin/python"),
    }
    registry = {}
    for role in sorted(production.REQUIRED_INPUT_ROLES):
        path = special.get(role, tmp_path / "inputs" / role)
        if role not in special:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_approved_fixture(path, role, monkeypatch)
        registry[role] = str(path)
        if role in production.APPROVED_HASHES:
            monkeypatch.setitem(
                production.APPROVED_HASHES, role,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    receipt = {
        "schema_version": "if-benchmark-v3-probe-validation/1",
        "receipt": "typed-probes-pass",
    }
    monkeypatch.setattr(production, "_default_probe_validators", lambda _paths: receipt)
    registry_path = tmp_path / "input_registry.json"
    execution_path = tmp_path / "execution.json"
    registry_path.write_text(json.dumps(registry))
    execution_path.write_text(json.dumps({
        "array_shards": fixture["array_shards"], "resources": fixture["resources"],
        "stage_parameters": fixture["stage_parameters"],
        "current_c5_rung": current_c5_rung,
        "prior_rung_snapshots": (
            [] if prior_rung_snapshots is None else prior_rung_snapshots
        ),
    }))
    contract_path = dag_path.parent / "build_contract.json"
    write_production_contract(
        build_id=fixture["build_id"], candidate_root=fixture["candidate_root"],
        input_registry_path=registry_path, execution_config_path=execution_path,
        output_path=contract_path,
    )
    write_production_dag(contract_path=contract_path, output_path=dag_path)
    return dag_path


def test_production_dag_freezes_stage_order_dependencies_arrays_and_sources(tmp_path):
    _contract, contract_path, dag_path, _source = _fixture_contract(tmp_path)
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    dag = validate_production_dag(dag_path)
    assert [row["name"] for row in dag["stages"]] == [stage.name for stage in STAGES]
    counts = {row["name"]: row["task_count"] for row in dag["stages"]}
    assert counts["c5_nmp"] == 2
    assert counts["c6_tier2"] == 3
    assert counts["c7_nmp"] == 2
    assert next(row for row in dag["stages"] if row["name"] == "c6_coordinate_source")[
        "execution"
    ] == "networked_login"


def test_contract_producer_hashes_input_registry_and_execution_config(tmp_path):
    contract, _old_contract_path, dag_path, source = _fixture_contract(tmp_path)
    registry_path = tmp_path / "input_registry.json"
    execution_path = tmp_path / "execution.json"
    registry_path.write_text(json.dumps({"fixture_source": str(source)}))
    execution_path.write_text(json.dumps({
        "array_shards": contract["array_shards"], "resources": contract["resources"],
        "stage_parameters": contract["stage_parameters"],
        "current_c5_rung": contract["current_c5_rung"],
        "prior_rung_snapshots": contract["prior_rung_snapshots"],
    }))
    contract_path = dag_path.parent / "build_contract.json"
    write_production_contract(
        build_id=contract["build_id"], candidate_root=contract["candidate_root"],
        input_registry_path=registry_path, execution_config_path=execution_path,
        output_path=contract_path, identity_profile="fixture", allow_fixture=True,
    )
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    registry_path.write_text("{}")
    with pytest.raises(ValueError, match="input registry"):
        validate_production_dag(dag_path)


def test_production_contract_executes_typed_probes_once_and_dag_reuses_receipt(
    tmp_path, monkeypatch
):
    contract, _old_contract_path, dag_path, _source = _fixture_contract(tmp_path)
    registry = {}
    for role in sorted(production.REQUIRED_INPUT_ROLES):
        path = tmp_path / "inputs" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_approved_fixture(path, role, monkeypatch)
        registry[role] = str(path)
        if role in production.APPROVED_HASHES:
            monkeypatch.setitem(
                production.APPROVED_HASHES, role,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    registry_path = tmp_path / "input_registry.json"
    execution_path = tmp_path / "execution.json"
    registry_path.write_text(json.dumps(registry))
    execution_path.write_text(json.dumps({
        "array_shards": contract["array_shards"], "resources": contract["resources"],
        "stage_parameters": contract["stage_parameters"],
        "current_c5_rung": contract["current_c5_rung"],
        "prior_rung_snapshots": contract["prior_rung_snapshots"],
    }))
    calls = []

    def validator(paths):
        assert set(paths) == production.REQUIRED_INPUT_ROLES
        calls.append({role: str(path) for role, path in paths.items()})
        return {
            "schema_version": "if-benchmark-v3-probe-validation/1",
            "receipt": "typed-probes-pass",
        }

    contract_path = dag_path.parent / "build_contract.json"
    write_production_contract(
        build_id=contract["build_id"], candidate_root=contract["candidate_root"],
        input_registry_path=registry_path, execution_config_path=execution_path,
        output_path=contract_path, probe_validator=validator,
    )
    assert len(calls) == 1
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, probe_validator=validator,
    )
    assert len(calls) == 1
    validate_production_dag(dag_path, probe_validator=validator)
    # Per-task DAG validation rehashes the signed probe receipt; it does not rerun
    # every expensive scientific probe for every array shard.
    assert len(calls) == 1


def test_real_tier1_preflight_binds_raw_bytes_and_logical_manifest_separately(
    tmp_path, monkeypatch
):
    contract, _old_contract_path, dag_path, _source = _fixture_contract(tmp_path)
    real = Path(
        "/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/builds/"
        "ifbench-v3-drb1501-full-20260829T054839Z-g0b353ac/"
        "HLA-DRB1_15_01/audit/preflight/preflight_manifest.json"
    )
    copied = tmp_path / "inputs" / "tier1_preflight_manifest"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(real, copied)
    registry = {}
    for role in sorted(production.REQUIRED_INPUT_ROLES):
        if role == "tier1_preflight_manifest":
            path = copied
        else:
            path = tmp_path / "inputs" / role
            path.write_text(f"{role}\n")
        registry[role] = str(path)
        if role in production.APPROVED_HASHES and role != "tier1_preflight_manifest":
            monkeypatch.setitem(
                production.APPROVED_HASHES, role,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    registry_path = tmp_path / "registry.json"
    execution_path = tmp_path / "execution.json"
    registry_path.write_text(json.dumps(registry))
    execution_path.write_text(json.dumps({
        "array_shards": contract["array_shards"], "resources": contract["resources"],
        "stage_parameters": contract["stage_parameters"],
        "current_c5_rung": contract["current_c5_rung"],
        "prior_rung_snapshots": contract["prior_rung_snapshots"],
    }))
    output = dag_path.parent / "build_contract.json"
    write_production_contract(
        build_id=contract["build_id"], candidate_root=contract["candidate_root"],
        input_registry_path=registry_path, execution_config_path=execution_path,
        output_path=output,
        probe_validator=lambda _paths: {
            "schema_version": "if-benchmark-v3-probe-validation/1", "pass": True,
        },
    )
    payload = json.loads(output.read_text())
    record = next(
        item for item in payload["immutable_inputs"]
        if item["role"] == "tier1_preflight_manifest"
    )
    assert record["sha256"] == "b3fa9206f7351b092cde3d0633305ab0e513973da35d2b30ee0500838c15d109"
    assert payload["approved_logical_manifests"]["tier1_preflight_manifest"] == (
        "f9c7875d2497f7122b1e72992d0768702b0d4ca62874e21e60229f421b96fc40"
    )


@pytest.mark.parametrize("tamper", ["resources", "task_count", "immutable_inputs"])
def test_dag_validation_canonical_rebuild_rejects_self_rehashed_tamper(tmp_path, tamper):
    _contract, contract_path, dag_path, _source = _fixture_contract(tmp_path)
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    payload = json.loads(dag_path.read_text())
    if tamper == "resources":
        payload["stages"][2]["resources"]["cpus"] = 999
    elif tamper == "task_count":
        payload["stages"][4]["task_count"] = 999
    else:
        payload["immutable_inputs"].pop("fixture_source")
    payload["dag_sha256"] = _self_hash(payload, "dag_sha256")
    dag_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical contract"):
        validate_production_dag(dag_path)


def test_production_task_table_is_generated_from_registered_stage_contract(
    tmp_path, monkeypatch
):
    dag_path = _real_profile_dag(tmp_path, monkeypatch)
    table_path = dag_path.parent / "tasks" / "c1_rcsb_source.json"
    write_production_task_table(
        dag_path=dag_path, stage="c1_rcsb_source", output_path=table_path,
    )
    payload = production.validate_production_task_table(
        table_path, dag_path=dag_path, stage="c1_rcsb_source",
    )
    task = payload["tasks"][0]
    assert task["command_argv"][2:5] == [
        "execute-production-stage", "--dag", str(dag_path.resolve())
    ]
    assert task["output_roots"] == ["payload"]
    assert task["required_outputs"] == ["payload/stage_result.json"]


@pytest.mark.parametrize("tamper", ["command", "input", "output"])
def test_production_task_validation_rejects_self_rehashed_arbitrary_descriptor(
    tmp_path, monkeypatch, tamper
):
    dag_path = _real_profile_dag(tmp_path, monkeypatch)
    table_path = dag_path.parent / "tasks" / "c1_rcsb_source.json"
    write_production_task_table(
        dag_path=dag_path, stage="c1_rcsb_source", output_path=table_path,
    )
    payload = json.loads(table_path.read_text())
    task = payload["tasks"][0]
    if tamper == "command":
        task["command_argv"].insert(2, "arbitrary-command")
    elif tamper == "input":
        extra = tmp_path / "unregistered.txt"
        extra.write_text("unregistered")
        task["input_paths"].append(str(extra.resolve()))
    else:
        task["required_outputs"] = ["payload/forged.txt"]
    task["task_sha256"] = _self_hash(task, "task_sha256")
    payload["table_sha256"] = _self_hash(payload, "table_sha256")
    table_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="registered stage contract"):
        production.validate_production_task_table(
            table_path, dag_path=dag_path, stage="c1_rcsb_source",
        )


def test_registered_stage_dispatcher_executes_and_replays_complete_payload(
    tmp_path, monkeypatch
):
    dag_path = _real_profile_dag(tmp_path, monkeypatch)
    dag = validate_production_dag(dag_path)

    def producer(_dag, output_dir, index):
        assert index == 0
        artifact = output_dir / "artifact.txt"
        artifact.write_text("scientific-evidence\n")
        return {"artifact": artifact.name}

    monkeypatch.setitem(stage_execution.PRODUCERS, "c1_rcsb_source", producer)
    output_dir = (
        Path(dag["candidate_root"]) / "audit" / "orchestrator" / "stages" /
        "c1_rcsb_source" / "00000" / "payload"
    )
    result_path = stage_execution.execute_production_stage(
        dag_path=dag_path, stage="c1_rcsb_source", task_index=0, task_count=1,
        output_dir=output_dir,
    )
    stage_execution.validate_stage_result(
        result_path, dag=dag, expected_stage="c1_rcsb_source", expected_index=0,
    )
    (output_dir / "artifact.txt").write_text("tamper\n")
    with pytest.raises(ValueError, match="file table changed"):
        stage_execution.validate_stage_result(
            result_path, dag=dag, expected_stage="c1_rcsb_source", expected_index=0,
        )


def test_insufficient_5000_rung_seals_failure_snapshot_and_binds_7500_retry(
    tmp_path, monkeypatch
):
    dag_path = _real_profile_dag(tmp_path / "first", monkeypatch)
    dag = validate_production_dag(dag_path)
    table_path = dag_path.parent / "tasks" / "c7_select.json"
    write_production_task_table(
        dag_path=dag_path, stage="c7_select", output_path=table_path,
    )
    task_dir = (
        Path(dag["candidate_root"]) / "audit" / "orchestrator" / "stages" /
        "c7_select" / "00000"
    )
    payload_dir = task_dir / "payload"
    payload_dir.mkdir(parents=True)
    (payload_dir / "failed_attempt.json").write_text("{}")
    stage_execution._write_result(
        payload_dir, dag=dag, stage="c7_select", task_index=0,
        summary={
            "scientific_status": "insufficient_capacity", "eligible_count": 2999,
            "required_tier2_count": 3000,
        },
    )
    (task_dir / "stage_identity.json").write_text("{}")
    monkeypatch.setattr(production, "_task_identity", lambda **_kwargs: {})
    monkeypatch.setattr(production.StageJournal, "run", lambda *_args, **_kwargs: {})
    failure = seal_production_stage(
        dag_path=dag_path, task_table_path=table_path, stage="c7_select",
    )
    assert failure.name == "stage_failed.json"
    assert production_status(dag_path=dag_path)["c7_select"] == "failed"
    snapshot = stage_execution.package_failed_rung_snapshot(dag_path=dag_path)
    assert json.loads(snapshot.read_text())["target"] == 5000

    retry_dag_path = _real_profile_dag(
        tmp_path / "retry", monkeypatch, current_c5_rung=7500,
        prior_rung_snapshots=[{
            "target": 5000, "snapshot_manifest": str(snapshot),
        }],
    )
    retry = validate_production_dag(retry_dag_path)
    assert retry["current_c5_rung"] == 7500
    assert [row["target"] for row in retry["prior_rung_snapshots"]] == [5000]


def test_post_seal_candidate_packaging_assigns_distinct_semantic_release_id(
    tmp_path, monkeypatch
):
    warehouse = tmp_path / "if_test_set"
    candidate = warehouse / "builds" / "build-001" / "HLA-DRB1_15_01"
    (candidate / "audit" / "orchestrator").mkdir(parents=True)
    assemble = tmp_path / "stages" / "assemble"
    head_stage = tmp_path / "stages" / "head"
    homology = tmp_path / "stages" / "homology"
    release_stage = tmp_path / "stages" / "release"
    for path in (assemble, head_stage, homology, release_stage):
        path.mkdir(parents=True)
    primary = pd.DataFrame([{
        "dataset_release_id": "candidate-build-001", "protein_id": "P1",
        "sequence": "A", "sequence_sha256": hashlib.sha256(b"A").hexdigest(),
    }])
    diagnostic = primary.iloc[0:0].copy()
    primary.to_parquet(assemble / "cohort_if_ready.parquet", index=False)
    diagnostic.to_parquet(
        assemble / "tier1_overlap_diagnostic_if_ready.parquet", index=False
    )
    (assemble / "pdbs_if_ready").mkdir()
    (assemble / "pdbs_if_ready" / "P1.cif").write_text("fixture\n")
    for name in ("release_cath_sidecar.parquet", "load_coords.parquet"):
        primary.to_parquet(assemble / name, index=False)
    head = primary.assign(global_risk=1.0)
    head.to_parquet(head_stage / "head.parquet", index=False)
    (head_stage / "head.manifest.json").write_text(json.dumps({
        "schema_version": "if-benchmark-v3-head-annotation/1",
        "cohort_inputs": [], "output_path": "head.parquet", "row_count": 1,
    }))
    primary.to_parquet(homology / "head_homology.parquet", index=False)
    (release_stage / "release_gates.json").write_text(json.dumps({
        str(index): "pass" for index in range(1, 14)
    }))
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("dummy\n")
    mapping = {
        "assemble_family": assemble, "c9_head": head_stage,
        "c9_homology": homology, "release_validate": release_stage,
    }
    for stage in (
        "c1_rcsb_source", "c4_source_cath", "c5_select",
        "c8_sifts_residue_source", "c4_final_cath", "c8_tier1_finalize",
        "collision_dedup", "c7_select", "references",
    ):
        mapping[stage] = tmp_path / "stages" / stage
        mapping[stage].mkdir(parents=True)
    dag = {
        "build_id": "build-001", "candidate_root": str(candidate),
        "dag_sha256": "d" * 64,
        "targets": {"tier1": 15, "tier2": 3000},
    }
    monkeypatch.setattr(stage_execution, "validate_production_dag", lambda _path: dag)
    monkeypatch.setattr(
        stage_execution, "production_status",
        lambda **_kwargs: {stage: "complete" for stage in production.STAGE_BY_NAME},
    )
    monkeypatch.setattr(stage_execution, "_stage_payloads", lambda _dag, stage: [mapping[stage]])
    monkeypatch.setattr(stage_execution, "_input_path", lambda _dag, _role: dummy)

    def fake_approved(_dag, path):
        path.write_text("{}")
        return path

    def fake_publication(_dag, output_dir, *, release_id):
        inventory = output_dir / "inventory.json"
        context = output_dir / "context.json"
        inventory.write_text("{}")
        context.write_text(json.dumps({"release_id": release_id}))
        return inventory, context

    def fake_evidence(path, **_kwargs):
        path.write_text("{}")
        return {}

    monkeypatch.setattr(stage_execution, "_approved_sources", fake_approved)
    monkeypatch.setattr(stage_execution, "_publication_artifacts", fake_publication)
    monkeypatch.setattr(stage_execution, "write_release_evidence_bundle", fake_evidence)
    monkeypatch.setattr(
        stage_execution, "validate_release_tables",
        lambda primary, diagnostic, *_args, release_id, **_kwargs: (
            {str(index): "pass" for index in range(1, 14)}
            if set(primary["dataset_release_id"]) == {release_id}
            and diagnostic.empty else pytest.fail("semantic ID rewrite failed")
        ),
    )
    main_alias = warehouse / "if_ready" / "main" / "HLA-DRB1_15_01"
    manifest_path = stage_execution.package_candidate_release(
        dag_path=tmp_path / "dag.json", dataset_release_id="semantic-release-v1",
        main_alias=main_alias,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_release_id"] == "semantic-release-v1"
    assert manifest["build_id"] == "build-001"
    assert (candidate / "cohort_if_ready.parquet").is_file()
    assert (candidate / "annotations" / "head.parquet").is_file()
    assert not os.path.lexists(main_alias)


@pytest.mark.parametrize(
    "field,value", [
        ("targets", {"tier1": 14, "tier2": 3000}),
        ("c5_ladder", [5000, 10000]),
        ("seed", 7),
    ],
)
def test_production_dag_rejects_relaxed_frozen_release_contract(tmp_path, field, value):
    contract, contract_path, dag_path, _source = _fixture_contract(tmp_path)
    contract[field] = value
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="frozen v3"):
        write_production_dag(
            contract_path=contract_path, output_path=dag_path, allow_fixture=True,
        )


def test_array_task_resume_seal_and_dependency_status_are_byte_bound(tmp_path):
    _contract, contract_path, dag_path, source = _fixture_contract(tmp_path)
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    producer = tmp_path / "producer.py"
    producer.write_text(
        "from pathlib import Path\nimport sys\n"
        "out=Path(sys.argv[1]); out.mkdir(parents=True)\n"
        "(out/'result.txt').write_text(Path(sys.argv[2]).read_text()+'-produced')\n"
    )
    table_path = dag_path.parent / "tasks" / "c1_rcsb_source.json"
    write_production_task_table(
        dag_path=dag_path, stage="c1_rcsb_source", output_path=table_path,
        tasks=[{
            "task_index": 0, "producer_code_path": str(producer),
            "code_paths": [str(producer)],
            "command_argv": [
                "/usr/bin/python3", str(producer), "{task_dir}/payload", str(source),
            ],
            "input_paths": [str(source)], "output_roots": ["payload"],
            "required_outputs": ["payload/result.txt"],
        }],
    )
    rendered = render_production_stage_commands(
        dag_path=dag_path, task_table_path=table_path, stage="c1_rcsb_source",
    )
    assert rendered["execution"] == "networked_login"
    assert rendered["commands"][0][-2:] == ["--task-index", "0"]
    first = run_production_task(
        dag_path=dag_path, task_table_path=table_path,
        stage="c1_rcsb_source", task_index=0, resume=False,
    )
    second = run_production_task(
        dag_path=dag_path, task_table_path=table_path,
        stage="c1_rcsb_source", task_index=0, resume=True,
    )
    assert first == second
    seal = seal_production_stage(
        dag_path=dag_path, task_table_path=table_path, stage="c1_rcsb_source",
    )
    assert seal.is_file()
    status = production_status(dag_path=dag_path)
    assert status["c1_rcsb_source"] == "complete"
    assert status["c4_source_cath"] == "blocked"  # references is not sealed.
    result = seal.parent / "00000" / "payload" / "result.txt"
    result.write_text("tamper")
    with pytest.raises(ValueError, match="output digest"):
        production_status(dag_path=dag_path)
    with pytest.raises(ValueError, match="output digest"):
        run_production_task(
            dag_path=dag_path, task_table_path=table_path,
            stage="c1_rcsb_source", task_index=0, resume=True,
        )


def test_task_cannot_run_before_every_dependency_is_sealed(tmp_path):
    _contract, contract_path, dag_path, source = _fixture_contract(tmp_path)
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    producer = tmp_path / "producer.py"
    producer.write_text("from pathlib import Path\nimport sys\nPath(sys.argv[1]).mkdir()\n")
    table = dag_path.parent / "tasks" / "c4_source_cath.json"
    write_production_task_table(
        dag_path=dag_path, stage="c4_source_cath", output_path=table,
        tasks=[{
            "task_index": 0, "producer_code_path": str(producer),
            "code_paths": [str(producer)],
            "command_argv": ["/usr/bin/python3", str(producer), "{task_dir}/payload"],
            "input_paths": [str(source)], "output_roots": ["payload"],
            "required_outputs": ["payload/result.txt"],
        }],
    )
    with pytest.raises(FileNotFoundError):
        run_production_task(
            dag_path=dag_path, task_table_path=table,
            stage="c4_source_cath", task_index=0, resume=False,
        )


def test_ready_slurm_stage_renders_bounded_array_and_afterok_seal(tmp_path):
    _contract, contract_path, dag_path, source = _fixture_contract(tmp_path)
    write_production_dag(
        contract_path=contract_path, output_path=dag_path, allow_fixture=True,
    )
    producer = tmp_path / "producer.py"
    producer.write_text("# producer\n")
    table = dag_path.parent / "tasks" / "references.json"
    write_production_task_table(
        dag_path=dag_path, stage="references", output_path=table,
        tasks=[{
            "task_index": 0, "producer_code_path": str(producer),
            "code_paths": [str(producer)],
            "command_argv": ["/usr/bin/python3", str(producer), "{task_dir}/payload"],
            "input_paths": [str(source)], "output_roots": ["payload"],
            "required_outputs": ["payload/result.txt"],
        }],
    )
    rendered = render_production_stage_commands(
        dag_path=dag_path, task_table_path=table, stage="references",
    )
    assert rendered["execution"] == "slurm"
    assert "--array=0-0%8" in rendered["array_command"]
    assert not any(value.startswith("--gres=") for value in rendered["array_command"])
    assert "--dependency=afterok:${ARRAY_JOB_ID}" in rendered["seal_command"]
