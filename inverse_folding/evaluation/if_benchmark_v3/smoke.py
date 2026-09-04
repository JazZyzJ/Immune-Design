"""Short, content-bound production probes used before expensive v3 jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .entities import build_entity_source_pool, parse_graphql_entity_page
from .evidence import _validate_nmp_pool, validate_nmp_install_evidence
from .leakage import (
    build_mmseqs_sidecar,
    run_mmseqs_easy_search,
    validate_mmseqs_run_manifest,
    validate_mmseqs_sidecar,
)
from .head_annotation import annotate_fixed_epoch_head, validate_head_annotation
from .mapping import AA3_TO_1, materialize_mmcif_chain, read_mmcif_atom_records
from .nmp import (
    make_standalone_score_shard_fn,
    nmp_parameter_payload,
    write_nmp_install_manifest,
    write_nmp_sharded_evidence,
)
from .preflight import SiftsHttpResponse
from .rcsb_snapshot import (
    GRAPHQL_QUERY,
    GRAPHQL_URL,
    RcsbRequestError,
    _default_transport,
    _request,
)
from .structures import acquire_rcsb_mmcif, validate_cached_rcsb_mmcif


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _file_identity(path: Path, *, root: Path) -> dict[str, Any]:
    path, root = Path(path).resolve(), Path(root).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("smoke evidence file escapes its source root")
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _resolve_file_identity(root: Path, identity: dict[str, Any]) -> Path:
    raw = Path(str(identity.get("path") or ""))
    root = Path(root).resolve()
    path = (root / raw).resolve()
    if (
        raw.is_absolute() or ".." in raw.parts or root not in path.parents
        or not path.is_file() or path.stat().st_size != int(identity.get("size_bytes", -1))
        or _sha(path) != identity.get("sha256")
    ):
        raise ValueError("smoke source file identity mismatch")
    return path


def probe_netmhciipan(
    *, output_dir: Path, binary_path: Path, install_root: Path,
    allele: str = "HLA-DRB1*15:01", sequence: str | None = None,
) -> Path:
    """Merkle the real install and score one sequence at C5 and C7 lengths."""

    output_dir = Path(output_dir)
    binary_path = Path(binary_path).resolve()
    install_root = Path(install_root).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frozen_probe_sequence = "ACDEFGHIKLMNPQRSTVWY" * 5
    sequence = sequence or frozen_probe_sequence
    if sequence != frozen_probe_sequence:
        raise ValueError("NetMHCIIpan production probe sequence is frozen at 100 residues")
    sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
    version_path = install_root / "data" / "version"
    if not version_path.is_file():
        raise FileNotFoundError(version_path)
    install_manifest = output_dir / "install_manifest.json"
    write_nmp_install_manifest(
        tool_path=binary_path, data_root=install_root,
        tool_version=version_path.read_text().strip(), manifest_path=install_manifest,
        effective_environment={
            "LC_ALL": "C", "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        }, identity_mode="official_install_merkle",
    )
    install_paths = {
        "tool": binary_path, "data_manifest": install_manifest,
        "contract_producer": Path(__file__).with_name("nmp.py"),
        "runner_producer": Path(__file__).parents[3] / "epitope_head" / "data" /
        "netmhciipan_runner.py",
    }
    install_identity = validate_nmp_install_evidence(install_paths)
    help_result = subprocess.run(
        [str(binary_path), "-h"], capture_output=True, check=False, timeout=60,
    )
    help_stdout = output_dir / "netmhciipan_help.stdout.txt"
    help_stderr = output_dir / "netmhciipan_help.stderr.txt"
    help_stdout.write_bytes(help_result.stdout)
    help_stderr.write_bytes(help_result.stderr)
    help_bytes = bytes(help_result.stdout) + b"\n" + bytes(help_result.stderr)
    if (
        help_result.returncode != 0
        or b"[-filter]" not in help_bytes
        or b"Toggle filtering of output" not in help_bytes
        or b"[-context]" not in help_bytes
    ):
        raise ValueError("NetMHCIIpan 4.3 CLI filter/context contract is not verified")
    stages = {}
    for stage, lengths in (("c5", [15]), ("c7", list(range(12, 26)))):
        stage_dir = output_dir / stage
        stage_dir.mkdir()
        parameters = stage_dir / "parameters.json"
        parameters.write_text(json.dumps(nmp_parameter_payload(
            stage=stage, allele=allele, peptide_lengths=lengths,
            install_manifest_sha256=install_identity["manifest_sha256"],
            batch_size=1, subprocess_timeout_s=600, max_lengths_per_call=4, n_workers=1,
        ), indent=2, sort_keys=True) + "\n")
        if stage == "c5":
            queries = pd.DataFrame([{
                "selection_unit_id": "probe-sequence", "source_sequence": sequence,
                "source_sequence_sha256": sequence_sha,
            }])
            key, seq_col, sha_col = (
                "selection_unit_id", "source_sequence", "source_sequence_sha256"
            )
        else:
            queries = pd.DataFrame([{
                "protein_id": "probe-sequence", "sequence": sequence,
                "sequence_sha256": sequence_sha,
            }])
            key, seq_col, sha_col = "protein_id", "sequence", "sequence_sha256"
        scorer = make_standalone_score_shard_fn(
            binary_path=binary_path, allele=allele, peptide_lengths=lengths,
            parameters_path=parameters, key_column=key, sequence_column=seq_col,
        )
        shard_manifest = write_nmp_sharded_evidence(
            output_dir=stage_dir / "evidence", queries=queries, key_column=key,
            sequence_column=seq_col, sequence_sha_column=sha_col, stage=stage,
            peptide_lengths=lengths, tool_path=binary_path, parameters_path=parameters,
            score_shard_fn=scorer, shard_query_count=1,
        )
        coverage, raw_by_id = _validate_nmp_pool(
            queries=queries, shard_manifest_path=shard_manifest, tool_path=binary_path,
            parameters_path=parameters, allele=allele, lengths=lengths, stage=stage,
            key_column=key, sequence_column=seq_col, sequence_sha_column=sha_col,
            install_identity=install_identity,
        )
        shard_payload = json.loads(shard_manifest.read_text())
        expected_window_count = sum(max(0, len(sequence) - length + 1) for length in lengths)
        if int(shard_payload["window_count"]) != expected_window_count:
            raise ValueError(
                f"NetMHCIIpan {stage} probe window count differs from complete enumeration"
            )
        stages[stage] = {
            "shard_manifest": str(shard_manifest.relative_to(output_dir)),
            "shard_manifest_sha256": _sha(shard_manifest),
            "window_count": int(shard_payload["window_count"]),
            "expected_window_count": expected_window_count,
            "coverage_fraction": coverage["probe-sequence"],
            "raw_evidence_sha256": raw_by_id["probe-sequence"],
        }
    manifest = {
        "schema_version": "if-benchmark-v3-nmp-probe/1", "allele": allele,
        "sequence_sha256": sequence_sha, "sequence_length": len(sequence),
        "install_manifest_sha256": install_identity["manifest_sha256"], "stages": stages,
        "cli_contract": {
            "argv": [str(binary_path), "-h"],
            "returncode": int(help_result.returncode),
            "stdout": _file_identity(help_stdout, root=output_dir),
            "stderr": _file_identity(help_stderr, root=output_dir),
            "filter_mode": "off_by_omission_verified_netmhciipan_4_3",
            "context_mode": "on_by_presence_verified_netmhciipan_4_3",
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output_dir / "probe_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_nmp_probe_manifest(
        path, binary_path=binary_path, install_root=install_root,
    )
    return path


def validate_nmp_probe_manifest(
    manifest_path: Path, *, binary_path: Path, install_root: Path,
) -> dict[str, Any]:
    """Rehash and replay the complete real-install C5/C7 mini probe."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 5
    binary_path, install_root = Path(binary_path).resolve(), Path(install_root).resolve()
    cli = payload.get("cli_contract") or {}
    if (
        payload.get("schema_version") != "if-benchmark-v3-nmp-probe/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or payload.get("allele") != "HLA-DRB1*15:01"
        or payload.get("sequence_length") != len(sequence)
        or payload.get("sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest()
        or set(payload.get("stages") or {}) != {"c5", "c7"}
        or cli.get("argv") != [str(binary_path), "-h"]
        or cli.get("returncode") != 0
        or cli.get("filter_mode") != "off_by_omission_verified_netmhciipan_4_3"
        or cli.get("context_mode") != "on_by_presence_verified_netmhciipan_4_3"
    ):
        raise ValueError("NetMHCIIpan probe manifest identity mismatch")
    help_stdout = _resolve_file_identity(root, cli["stdout"])
    help_stderr = _resolve_file_identity(root, cli["stderr"])
    help_bytes = help_stdout.read_bytes() + b"\n" + help_stderr.read_bytes()
    if (
        b"[-filter]" not in help_bytes
        or b"Toggle filtering of output" not in help_bytes
        or b"[-context]" not in help_bytes
    ):
        raise ValueError("NetMHCIIpan probe help evidence changed")
    install_manifest = root / "install_manifest.json"
    install_payload = json.loads(install_manifest.read_text())
    raw_data_root = Path(str(install_payload.get("data_root") or ""))
    manifest_data_root = (
        raw_data_root.resolve() if raw_data_root.is_absolute()
        else (install_manifest.parent / raw_data_root).resolve()
    )
    if manifest_data_root != install_root:
        raise ValueError("NetMHCIIpan probe install root differs from registered data root")
    install_identity = validate_nmp_install_evidence({
        "tool": binary_path,
        "data_manifest": install_manifest,
        "contract_producer": Path(__file__).with_name("nmp.py"),
        "runner_producer": Path(__file__).parents[3] / "epitope_head" / "data" /
        "netmhciipan_runner.py",
    })
    if install_identity["manifest_sha256"] != payload["install_manifest_sha256"]:
        raise ValueError("NetMHCIIpan probe install manifest changed")
    for stage, lengths in (("c5", [15]), ("c7", list(range(12, 26)))):
        if stage == "c5":
            queries = pd.DataFrame([{
                "selection_unit_id": "probe-sequence", "source_sequence": sequence,
                "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            }])
            key, sequence_column, sha_column = (
                "selection_unit_id", "source_sequence", "source_sequence_sha256"
            )
        else:
            queries = pd.DataFrame([{
                "protein_id": "probe-sequence", "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            }])
            key, sequence_column, sha_column = "protein_id", "sequence", "sequence_sha256"
        parameters = root / stage / "parameters.json"
        stage_record = payload["stages"][stage]
        shard_manifest = (root / str(stage_record["shard_manifest"])).resolve()
        if (
            root not in shard_manifest.parents or not shard_manifest.is_file()
            or _sha(shard_manifest) != stage_record["shard_manifest_sha256"]
        ):
            raise ValueError("NetMHCIIpan probe shard manifest identity mismatch")
        coverage, raw_by_id = _validate_nmp_pool(
            queries=queries, shard_manifest_path=shard_manifest, tool_path=binary_path,
            parameters_path=parameters, allele="HLA-DRB1*15:01", lengths=lengths,
            stage=stage, key_column=key, sequence_column=sequence_column,
            sequence_sha_column=sha_column, install_identity=install_identity,
        )
        expected_count = sum(len(sequence) - length + 1 for length in lengths)
        shard_payload = json.loads(shard_manifest.read_text())
        if (
            int(stage_record["window_count"]) != expected_count
            or int(stage_record["expected_window_count"]) != expected_count
            or int(shard_payload["window_count"]) != expected_count
            or float(stage_record["coverage_fraction"]) != coverage["probe-sequence"]
            or stage_record["raw_evidence_sha256"] != raw_by_id["probe-sequence"]
        ):
            raise ValueError("NetMHCIIpan probe stage replay mismatch")
    return payload


def probe_mmseqs(
    *, output_dir: Path, tool_path: Path, threads: int = 2,
) -> Path:
    """Run both frozen MMseqs coverage modes on a self-contained two-query fixture."""

    output_dir = Path(output_dir)
    tool_path = Path(tool_path).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not tool_path.is_file() or int(threads) < 1:
        raise ValueError("MMseqs probe tool/thread contract is invalid")
    output_dir.mkdir(parents=True)
    first = "ACDEFGHIKLMNPQRSTVWY" * 5
    second = "YWVTSRQPNMLKIHGFEDCA" * 5
    queries = pd.DataFrame([
        {"protein_id": "probe-q1", "sequence": first},
        {"protein_id": "probe-q2", "sequence": second},
    ])
    query_table = output_dir / "queries.parquet"
    query_fasta = output_dir / "queries.fasta"
    reference_fasta = output_dir / "reference.fasta"
    queries.to_parquet(query_table, index=False)
    query_fasta.write_text(f">probe-q1\n{first}\n>probe-q2\n{second}\n")
    reference_fasta.write_text(f">probe-r1\n{first}\n>probe-r2\n{second}\n")
    reference_sha = _sha(reference_fasta)
    stages = {}
    for search_kind, cov_mode in (("cath", 0), ("head", 2)):
        run_dir = output_dir / search_kind
        paths = run_mmseqs_easy_search(
            tool_path=tool_path, query_fasta_path=query_fasta,
            reference_fasta_path=reference_fasta, output_dir=run_dir,
            cov_mode=cov_mode, coverage=0.8, min_seq_id=0.3,
            threads=int(threads), timeout_s=600,
        )
        run_manifest = validate_mmseqs_run_manifest(paths["manifest"], replay=True)
        sidecar = build_mmseqs_sidecar(
            queries, raw_tsv_path=paths["raw_output"], query_fasta_path=query_fasta,
            search_kind=search_kind, reference_sha256=reference_sha,
            tool_sha256=str(run_manifest["tool_sha256"]),
            tool_version=str(run_manifest["tool_version"]), cov_mode=cov_mode,
            coverage=0.8, min_seq_id=0.3, path_root=output_dir,
        )
        sidecar_path = run_dir / "sidecar.parquet"
        sidecar.to_parquet(sidecar_path, index=False)
        validate_mmseqs_sidecar(
            queries, sidecar, raw_tsv_path=paths["raw_output"],
            query_fasta_path=query_fasta, search_kind=search_kind,
            reference_sha256=reference_sha, tool_sha256=str(run_manifest["tool_sha256"]),
            tool_version=str(run_manifest["tool_version"]), cov_mode=cov_mode,
            coverage=0.8, min_seq_id=0.3, path_root=output_dir,
        )
        stages[search_kind] = {
            "cov_mode": cov_mode,
            "run_manifest": str(paths["manifest"].relative_to(output_dir)),
            "run_manifest_sha256": _sha(paths["manifest"]),
            "sidecar": str(sidecar_path.relative_to(output_dir)),
            "sidecar_sha256": _sha(sidecar_path),
            "hit_count": int((sidecar["search_status"] == "complete_hit").sum()),
        }
    manifest = {
        "schema_version": "if-benchmark-v3-mmseqs-probe/1",
        "tool_path": str(tool_path), "tool_sha256": _sha(tool_path),
        "threads": int(threads),
        "queries": _file_identity(query_table, root=output_dir),
        "query_fasta": _file_identity(query_fasta, root=output_dir),
        "reference_fasta": _file_identity(reference_fasta, root=output_dir),
        "stages": stages,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output_dir / "probe_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_mmseqs_probe_manifest(path, replay=False)
    return path


def validate_mmseqs_probe_manifest(
    manifest_path: Path, *, replay: bool = False, tool_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    tool = Path(str(payload.get("tool_path") or "")).resolve()
    registered_tool = None if tool_path is None else Path(tool_path).resolve()
    if (
        payload.get("schema_version") != "if-benchmark-v3-mmseqs-probe/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not tool.is_file() or _sha(tool) != payload.get("tool_sha256")
        or (registered_tool is not None and tool != registered_tool)
        or int(payload.get("threads", 0)) < 1
        or set(payload.get("stages") or {}) != {"cath", "head"}
    ):
        raise ValueError("MMseqs probe manifest identity mismatch")
    query_table = _resolve_file_identity(root, payload["queries"])
    query_fasta = _resolve_file_identity(root, payload["query_fasta"])
    reference_fasta = _resolve_file_identity(root, payload["reference_fasta"])
    queries = pd.read_parquet(query_table)
    reference_sha = _sha(reference_fasta)
    for search_kind, cov_mode in (("cath", 0), ("head", 2)):
        record = payload["stages"][search_kind]
        run_manifest = (root / str(record["run_manifest"])).resolve()
        sidecar_path = (root / str(record["sidecar"])).resolve()
        if (
            root not in run_manifest.parents or root not in sidecar_path.parents
            or _sha(run_manifest) != record["run_manifest_sha256"]
            or _sha(sidecar_path) != record["sidecar_sha256"]
            or int(record["cov_mode"]) != cov_mode
        ):
            raise ValueError("MMseqs probe stage file identity mismatch")
        run = validate_mmseqs_run_manifest(run_manifest, replay=replay)
        sidecar = pd.read_parquet(sidecar_path)
        raw_path = run_manifest.parent / run["outputs"]["raw_tsv"]["path"]
        validate_mmseqs_sidecar(
            queries, sidecar, raw_tsv_path=raw_path, query_fasta_path=query_fasta,
            search_kind=search_kind, reference_sha256=reference_sha,
            tool_sha256=str(run["tool_sha256"]), tool_version=str(run["tool_version"]),
            cov_mode=cov_mode, coverage=0.8, min_seq_id=0.3, path_root=root,
        )
        if int(record["hit_count"]) != int(
            (sidecar["search_status"] == "complete_hit").sum()
        ):
            raise ValueError("MMseqs probe hit count mismatch")
    return payload


def probe_head(
    *, output_dir: Path, checkpoint_path: Path, resolved_config_path: Path,
    run_summary_path: Path, model_config_dir: Path, device: str = "cuda",
    window_batch_size: int = 128,
    predictor_factory: Callable[..., Any] | None = None,
) -> Path:
    """Score a fixed two-sequence cohort with the real epoch-24 Head."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY" * 5,
        "YWVTSRQPNMLKIHGFEDCA" * 5,
    ]
    cohort = pd.DataFrame([
        {
            "dataset_release_id": "ifbench-v3-drb1501-head-smoke",
            "protein_id": f"probe-head-{index + 1}", "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        }
        for index, sequence in enumerate(sequences)
    ])
    cohort_path = output_dir / "cohort.parquet"
    output_path = output_dir / "head.parquet"
    manifest_path = output_dir / "head.manifest.json"
    cohort.to_parquet(cohort_path, index=False)
    annotate_fixed_epoch_head(
        cohort=cohort, checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path, run_summary_path=run_summary_path,
        model_config_dir=model_config_dir, output_path=output_path,
        manifest_path=manifest_path, device=device,
        window_batch_size=int(window_batch_size), cohort_input_paths=[cohort_path],
        predictor_factory=predictor_factory,
    )
    validate_head_annotation(
        manifest_path=manifest_path, checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path, run_summary_path=run_summary_path,
        model_config_dir=model_config_dir, cohort_input_paths=[cohort_path], replay=False,
        predictor_factory=predictor_factory,
    )
    return manifest_path


def _write_graphql_smoke_failure(
    *, output_dir: Path, rcsb_entity_id: str, query_path: Path,
    error: RcsbRequestError,
) -> Path:
    response_identity = None
    if error.response is not None and bytes(error.response.body):
        response_path = output_dir / "graphql_failure_response.body"
        response_path.write_bytes(bytes(error.response.body))
        response_identity = _file_identity(response_path, root=output_dir)
    payload = {
        "schema_version": "if-benchmark-v3-rcsb-structure-source-failure/1",
        "rcsb_entity_id": rcsb_entity_id,
        "query": _file_identity(query_path, root=output_dir),
        "attempts": error.attempts,
        "response": response_identity,
        "failure": f"{type(error).__name__}: {error}",
    }
    payload["failure_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = output_dir / "source_failure.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def freeze_rcsb_structure_smoke_source(
    *, output_dir: Path, rcsb_entity_id: str,
    transport: Callable[..., Any] = _default_transport,
    coordinate_request_fn: Callable[..., SiftsHttpResponse] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    api_delay_s: float = 0.1,
) -> Path:
    """Freeze instance and revision-bound coordinate bytes on a networked host."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if api_delay_s < 0:
        raise ValueError("RCSB structure source API delay must be nonnegative")
    output_dir.mkdir(parents=True)
    rcsb_entity_id = str(rcsb_entity_id).upper()
    body = _canonical({"query": GRAPHQL_QUERY, "variables": {"ids": [rcsb_entity_id]}})
    query_path = output_dir / "graphql_query.json"
    query_path.write_bytes(body)
    try:
        response, attempts = _request(
            url=GRAPHQL_URL, body=body, transport=transport, sleep_fn=sleep_fn,
            timeout_s=60.0, max_attempts=4,
        )
    except RcsbRequestError as exc:
        failure_path = _write_graphql_smoke_failure(
            output_dir=output_dir, rcsb_entity_id=rcsb_entity_id,
            query_path=query_path, error=exc,
        )
        raise RuntimeError(
            f"RCSB structure source freeze failed; evidence={failure_path}"
        ) from exc
    if api_delay_s:
        sleep_fn(api_delay_s)
    response_path = output_dir / "graphql_response.json"
    response_path.write_bytes(bytes(response.body))
    entities, _edges = parse_graphql_entity_page(bytes(response.body))
    if len(entities) != 1 or entities[0]["rcsb_entity_id"] != rcsb_entity_id:
        raise ValueError("RCSB smoke GraphQL entity identity mismatch")
    entity = entities[0]
    _groups, _fallbacks, rejected = build_entity_source_pool([entity], _edges)
    if rejected:
        raise ValueError("RCSB smoke entity is outside v3 C1 eligibility")
    acquire_kwargs: dict[str, Any] = {
        "pdb_id": str(entity["pdb_id"]),
        "expected_revision_date": str(entity["structure_revision_date"]),
        "cache_dir": output_dir / "coordinate_cache",
        "api_delay_s": api_delay_s,
        "sleep_fn": sleep_fn,
    }
    if coordinate_request_fn is not None:
        acquire_kwargs["request_fn"] = coordinate_request_fn
    cif_path, _cache_manifest = acquire_rcsb_mmcif(**acquire_kwargs)
    cache_manifest_path = cif_path.parent / "manifest.json"
    manifest = {
        "schema_version": "if-benchmark-v3-rcsb-structure-source/1",
        "rcsb_entity_id": rcsb_entity_id,
        "graphql_url": GRAPHQL_URL,
        "api_delay_s": float(api_delay_s),
        "request_attempts": attempts,
        "files": {
            "graphql_query": _file_identity(query_path, root=output_dir),
            "graphql_response": _file_identity(response_path, root=output_dir),
            "coordinate_cif": _file_identity(cif_path, root=output_dir),
            "coordinate_manifest": _file_identity(cache_manifest_path, root=output_dir),
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output_dir / "source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_rcsb_structure_smoke_source(path)
    return path


def validate_rcsb_structure_smoke_source(
    source_manifest: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    source_manifest = Path(source_manifest).resolve()
    root = source_manifest.parent
    payload = json.loads(source_manifest.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    if (
        payload.get("schema_version") != "if-benchmark-v3-rcsb-structure-source/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or payload.get("graphql_url") != GRAPHQL_URL
        or float(payload.get("api_delay_s", -1)) < 0.0
        or not isinstance(payload.get("request_attempts"), list)
        or not payload["request_attempts"]
        or set(payload.get("files") or {}) != {
            "graphql_query", "graphql_response", "coordinate_cif", "coordinate_manifest"
        }
    ):
        raise ValueError("RCSB structure source manifest identity mismatch")
    files = {
        key: _resolve_file_identity(root, value)
        for key, value in payload["files"].items()
    }
    rcsb_entity_id = str(payload["rcsb_entity_id"])
    expected_query = _canonical({
        "query": GRAPHQL_QUERY, "variables": {"ids": [rcsb_entity_id]},
    })
    if files["graphql_query"].read_bytes() != expected_query:
        raise ValueError("RCSB structure source GraphQL query differs from frozen contract")
    entities, edges = parse_graphql_entity_page(files["graphql_response"].read_bytes())
    if len(entities) != 1 or entities[0]["rcsb_entity_id"] != rcsb_entity_id or not edges:
        raise ValueError("RCSB structure source response entity/instance identity mismatch")
    entity = entities[0]
    if any(edge["rcsb_entity_id"] != rcsb_entity_id for edge in edges):
        raise ValueError("RCSB structure source contains an unrelated instance edge")
    _groups, _fallbacks, rejected = build_entity_source_pool([entity], edges)
    if rejected:
        raise ValueError("RCSB structure source is outside v3 C1 eligibility")
    validate_cached_rcsb_mmcif(
        cif_path=files["coordinate_cif"], manifest_path=files["coordinate_manifest"],
        pdb_id=str(entity["pdb_id"]),
        expected_revision_date=str(entity["structure_revision_date"]),
    )
    return entity, edges, files["coordinate_cif"]


def smoke_rcsb_multicopy_structure(
    *, output_dir: Path, source_manifest: Path, min_coverage: float = 0.8,
    require_multiple_instances: bool = True, require_repeated_auth_id: bool = False,
    load_coords_fn: Callable[..., Any] | None = None,
) -> Path:
    """Offline replay of frozen source bytes through clean-mmCIF and DPLM load_coords."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    entity, edges, cif_path = validate_rcsb_structure_smoke_source(source_manifest)
    if require_multiple_instances and len(edges) < 2:
        raise ValueError("RCSB smoke entity does not have multiple polymer instances")
    auth_ids = [str(edge["auth_asym_id"]) for edge in edges]
    if require_repeated_auth_id and len(auth_ids) == len(set(auth_ids)):
        raise ValueError("RCSB smoke entity does not have a repeated auth chain ID")
    if load_coords_fn is None:
        try:
            from byprot.utils.io import load_coords as load_coords_fn
        except ImportError as exc:
            raise RuntimeError(
                "byprot is unavailable; include inverse_folding/dplm/src in PYTHONPATH"
            ) from exc
    output_dir.mkdir(parents=True)
    sequence = str(entity["entity_sequence"])
    rcsb_entity_id = str(entity["rcsb_entity_id"])
    results = []
    clean_dir = output_dir / "clean"
    clean_dir.mkdir()
    observed_to_source = {index + 1: index for index in range(len(sequence))}
    for edge in edges:
        expected_edge = {
            "rcsb_entity_id": rcsb_entity_id,
            "label_asym_id": str(edge["label_asym_id"]),
            "auth_asym_id": str(edge["auth_asym_id"]),
        }
        clean_path = clean_dir / f"{edge['label_asym_id']}.cif"
        try:
            materialized = materialize_mmcif_chain(
                source_mmcif=cif_path, expected_edge=expected_edge,
                source_sequence=sequence, observed_to_source=observed_to_source,
                denominator_positions=range(len(sequence)), output_path=clean_path,
                output_chain_id="A", load_coords_fn=load_coords_fn,
                min_coverage=min_coverage,
            )
            results.append({
                "label_asym_id": edge["label_asym_id"],
                "auth_asym_id": edge["auth_asym_id"], "status": "pass",
                "sequence_sha256": hashlib.sha256(
                    materialized.mapping.sequence.encode()
                ).hexdigest(),
                "sequence_length": len(materialized.mapping.sequence),
                "if_sequence_coverage": materialized.mapping.if_sequence_coverage,
                "clean_path": str(clean_path.relative_to(output_dir)),
                "clean_sha256": materialized.structure_sha256,
                "load_coords_evidence": materialized.load_coords_evidence,
            })
        except (RuntimeError, ValueError) as exc:
            results.append({
                "label_asym_id": edge["label_asym_id"],
                "auth_asym_id": edge["auth_asym_id"], "status": "failure",
                "failure": f"{type(exc).__name__}: {exc}",
            })
    if sum(item["status"] == "pass" for item in results) < (2 if require_multiple_instances else 1):
        raise ValueError("RCSB smoke did not materialize the required instance count")
    manifest = {
        "schema_version": "if-benchmark-v3-rcsb-structure-probe/1",
        "rcsb_entity_id": rcsb_entity_id,
        "source_manifest_path": str(
            Path(source_manifest).resolve().relative_to(output_dir.resolve().parent)
        ),
        "source_manifest_sha256": _sha(Path(source_manifest)),
        "coordinate_cif_sha256": _sha(cif_path),
        "parameters": {
            "min_coverage": float(min_coverage),
            "require_multiple_instances": bool(require_multiple_instances),
            "require_repeated_auth_id": bool(require_repeated_auth_id),
        },
        "results": results,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output_dir / "probe_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_rcsb_structure_probe_manifest(path, load_coords_fn=load_coords_fn)
    return path


def validate_rcsb_structure_probe_manifest(
    manifest_path: Path, *, load_coords_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Rehash the frozen source and every clean structure in an offline smoke result."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    parameters = payload.get("parameters") or {}
    if (
        payload.get("schema_version") != "if-benchmark-v3-rcsb-structure-probe/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or not isinstance(payload.get("results"), list)
        or not payload["results"]
        or not 0.0 <= float(parameters.get("min_coverage", -1.0)) <= 1.0
        or not isinstance(parameters.get("require_multiple_instances"), bool)
        or not isinstance(parameters.get("require_repeated_auth_id"), bool)
    ):
        raise ValueError("RCSB structure probe manifest identity mismatch")
    raw_source = Path(str(payload.get("source_manifest_path") or ""))
    source_manifest = (root.parent / raw_source).resolve()
    if (
        raw_source.is_absolute() or ".." in raw_source.parts
        or source_manifest.parent.parent != root.parent
        or not source_manifest.is_file()
        or _sha(source_manifest) != payload.get("source_manifest_sha256")
    ):
        raise ValueError("RCSB structure probe source-manifest identity mismatch")
    entity, edges, cif_path = validate_rcsb_structure_smoke_source(source_manifest)
    if (
        payload.get("rcsb_entity_id") != entity["rcsb_entity_id"]
        or payload.get("coordinate_cif_sha256") != _sha(cif_path)
    ):
        raise ValueError("RCSB structure probe source identity differs from frozen bytes")
    expected_keys = {
        (str(edge["label_asym_id"]), str(edge["auth_asym_id"])) for edge in edges
    }
    rows = payload["results"]
    observed_keys = {
        (str(row.get("label_asym_id")), str(row.get("auth_asym_id"))) for row in rows
    }
    if len(observed_keys) != len(rows) or observed_keys != expected_keys:
        raise ValueError("RCSB structure probe results do not exactly cover frozen instances")
    if load_coords_fn is None:
        try:
            from byprot.utils.io import load_coords as load_coords_fn
        except ImportError as exc:
            raise RuntimeError(
                "byprot is unavailable; include inverse_folding/dplm/src in PYTHONPATH"
            ) from exc
    passed = 0
    for row in rows:
        if row.get("status") == "failure":
            if not str(row.get("failure") or ""):
                raise ValueError("RCSB structure probe failure lacks evidence")
            continue
        if row.get("status") != "pass":
            raise ValueError("RCSB structure probe result status is invalid")
        raw_clean = Path(str(row.get("clean_path") or ""))
        clean_path = (root / raw_clean).resolve()
        if (
            raw_clean.is_absolute() or ".." in raw_clean.parts
            or root not in clean_path.parents or not clean_path.is_file()
            or _sha(clean_path) != row.get("clean_sha256")
        ):
            raise ValueError("RCSB structure probe clean-file identity mismatch")
        records = read_mmcif_atom_records(clean_path)
        residues: dict[int, str] = {}
        atom_names: dict[int, set[str]] = {}
        for atom in records:
            position = int(atom["label_seq_id"])
            residue = AA3_TO_1.get(str(atom["resname"]).upper())
            if residue is None or (position in residues and residues[position] != residue):
                raise ValueError("RCSB structure probe clean sequence is ambiguous")
            residues[position] = residue
            atom_names.setdefault(position, set()).add(str(atom["atom_name"]))
        if sorted(residues) != list(range(1, len(residues) + 1)) or any(
            names != {"N", "CA", "C", "O"} for names in atom_names.values()
        ):
            raise ValueError("RCSB structure probe clean backbone is incomplete")
        sequence = "".join(residues[index] for index in sorted(residues))
        expected_coverage = len(sequence) / len(str(entity["entity_sequence"]))
        if (
            len(sequence) != int(row.get("sequence_length", -1))
            or hashlib.sha256(sequence.encode()).hexdigest() != row.get("sequence_sha256")
            or abs(float(row.get("if_sequence_coverage", -1.0)) - expected_coverage) > 1e-12
            or expected_coverage < float(parameters["min_coverage"])
        ):
            raise ValueError("RCSB structure probe clean sequence/coverage mismatch")
        loaded = load_coords_fn(str(clean_path), chain="A")
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("RCSB structure probe DPLM load_coords result is invalid")
        coords, loaded_sequence = loaded
        load_evidence = {
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "structure_sha256": _sha(clean_path),
            "loaded_sequence_sha256": hashlib.sha256(str(loaded_sequence).encode()).hexdigest(),
            "loaded_length": len(coords),
            "status": "pass",
        }
        if (
            len(coords) != len(sequence) or str(loaded_sequence) != sequence
            or row.get("load_coords_evidence") != load_evidence
        ):
            raise ValueError("RCSB structure probe DPLM load_coords replay mismatch")
        passed += 1
    required = 2 if parameters["require_multiple_instances"] else 1
    if passed < required:
        raise ValueError("RCSB structure probe pass count is below its frozen requirement")
    auth_ids = [str(edge["auth_asym_id"]) for edge in edges]
    if parameters["require_repeated_auth_id"] and len(auth_ids) == len(set(auth_ids)):
        raise ValueError("RCSB structure probe repeated-auth requirement is not met")
    return payload
