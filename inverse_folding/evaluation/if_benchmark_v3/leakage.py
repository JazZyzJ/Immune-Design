"""Content-bound MMseqs evidence for CATH isolation and Head homology."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _params_digest(
    *, search_kind: str, min_seq_id: float, cov_mode: int, coverage: float
) -> str:
    payload = {
        "search_kind": search_kind,
        "min_seq_id": float(min_seq_id),
        "cov_mode": int(cov_mode),
        "coverage": float(coverage),
        "format": "query,target,fident,qcov,tcov",
    }
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _canonicalize_mmseqs_help(text: str) -> str:
    """Normalize only the host-CPU-derived default that the argv explicitly overrides."""

    lines = []
    for line in text.splitlines():
        if re.match(r"^\s+--threads INT\s", line):
            line = re.sub(r"\[\d+\]$", "[EXPLICIT_THREADS]", line)
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _read_query_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current is not None:
                records[current] = "".join(chunks)
            current = line[1:].split()[0]
            if not current or current in records:
                raise ValueError("query FASTA has missing/duplicate identifier")
            chunks = []
        elif current is None:
            if line.strip():
                raise ValueError("query FASTA sequence precedes identifier")
        else:
            chunks.append(line.strip())
    if current is not None:
        records[current] = "".join(chunks)
    return records


def build_mmseqs_sidecar(
    queries: pd.DataFrame,
    *,
    raw_tsv_path: Path,
    query_fasta_path: Path,
    search_kind: str,
    reference_sha256: str,
    tool_sha256: str,
    tool_version: str,
    cov_mode: int,
    coverage: float,
    min_seq_id: float,
    path_root: Path | None = None,
) -> pd.DataFrame:
    """Parse persisted MMseqs output and emit one measured row per final sequence."""

    if search_kind not in {"cath", "head"}:
        raise ValueError("search_kind must be cath or head")
    if cov_mode not in {0, 2} or not 0.0 <= float(coverage) <= 1.0:
        raise ValueError("unsupported MMseqs coverage contract")
    if not 0.0 <= float(min_seq_id) <= 1.0:
        raise ValueError("MMseqs minimum identity is outside [0,1]")
    required = {"protein_id", "sequence"}
    if not required <= set(queries.columns):
        raise ValueError("queries require protein_id and sequence")
    if queries["protein_id"].isna().any() or not queries["protein_id"].is_unique:
        raise ValueError("queries require unique protein_id")
    raw_tsv_path = Path(raw_tsv_path).resolve()
    query_fasta_path = Path(query_fasta_path).resolve()
    if path_root is not None:
        path_root = Path(path_root).resolve()
        try:
            stored_raw_path = str(raw_tsv_path.relative_to(path_root))
            stored_query_path = str(query_fasta_path.relative_to(path_root))
        except ValueError as exc:
            raise ValueError("MMseqs stage output escapes path_root") from exc
    else:
        stored_raw_path = str(raw_tsv_path)
        stored_query_path = str(query_fasta_path)
    if not raw_tsv_path.is_file() or not query_fasta_path.is_file():
        raise FileNotFoundError("persisted MMseqs TSV/query FASTA is missing")
    expected_query_sequences = {
        str(row.protein_id): str(row.sequence) for row in queries.itertuples(index=False)
    }
    if _read_query_fasta(query_fasta_path) != expected_query_sequences:
        raise ValueError("query FASTA does not exactly match final query table")
    raw_tsv = raw_tsv_path.read_text()
    query_ids = set(expected_query_sequences)
    hits: dict[str, tuple[float, str, float, float]] = {}
    for line in raw_tsv.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 5:
            raise ValueError("MMseqs TSV must have query,target,fident,qcov,tcov")
        query, target = parts[0], parts[1]
        if query not in query_ids:
            raise ValueError(f"MMseqs output contains unknown query: {query}")
        identity, qcov, tcov = map(float, parts[2:])
        if not all(math.isfinite(value) for value in (identity, qcov, tcov)):
            raise ValueError("MMseqs output has non-finite metric")
        if not all(0.0 <= value <= 1.0 for value in (identity, qcov, tcov)):
            raise ValueError("MMseqs identity/qcov/tcov outside [0,1]")
        coverage_pass = (
            qcov >= coverage and tcov >= coverage if cov_mode == 0 else qcov >= coverage
        )
        if not coverage_pass:
            raise ValueError("MMseqs raw hit violates the frozen coverage filter")
        candidate = (identity, target, qcov, tcov)
        previous = hits.get(query)
        if previous is None or (-candidate[0], candidate[1]) < (-previous[0], previous[1]):
            hits[query] = candidate

    raw_sha = _sha(raw_tsv.encode())
    params_sha = _params_digest(
        search_kind=search_kind, min_seq_id=min_seq_id,
        cov_mode=cov_mode, coverage=coverage,
    )
    rows: list[dict[str, Any]] = []
    for query in queries.sort_values("protein_id", kind="stable").itertuples(index=False):
        protein_id = str(query.protein_id)
        sequence = str(query.sequence)
        hit = hits.get(protein_id)
        if hit is None:
            best_identity, best_target, qcov, tcov = None, None, None, None
            overlap = False
            status = "complete_no_hit"
        else:
            best_identity, best_target, qcov, tcov = hit
            overlap = (
                best_identity > min_seq_id
                if search_kind == "cath"
                else best_identity >= min_seq_id
            )
            status = "complete_hit"
        rows.append(
            {
                "protein_id": protein_id,
                "query_sequence_sha256": _sha(sequence.encode()),
                "search_kind": search_kind,
                "search_status": status,
                "best_target": best_target,
                "best_identity": best_identity,
                "query_coverage": qcov,
                "target_coverage": tcov,
                "overlap_flag": overlap,
                "reference_sha256": reference_sha256,
                "tool_sha256": tool_sha256,
                "tool_version": tool_version,
                "min_seq_id": float(min_seq_id),
                "cov_mode": int(cov_mode),
                "coverage": float(coverage),
                "parameters_sha256": params_sha,
                "raw_output_sha256": raw_sha,
                "raw_output_path": stored_raw_path,
                "query_fasta_sha256": _sha(query_fasta_path.read_bytes()),
                "query_fasta_path": stored_query_path,
                "evidence_status": "measured_mmseqs",
            }
        )
    return pd.DataFrame(rows)


def validate_mmseqs_sidecar(
    queries: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    raw_tsv_path: Path,
    query_fasta_path: Path,
    search_kind: str,
    reference_sha256: str,
    tool_sha256: str,
    tool_version: str,
    cov_mode: int,
    coverage: float,
    min_seq_id: float,
    path_root: Path | None = None,
) -> None:
    """Validate query and evaluator identities; no flag may be absent/defaulted."""

    raw_tsv_path = Path(raw_tsv_path).resolve()
    query_fasta_path = Path(query_fasta_path).resolve()
    if path_root is not None:
        root = Path(path_root).resolve()
        expected_raw_path = str(raw_tsv_path.relative_to(root))
        expected_query_path = str(query_fasta_path.relative_to(root))
    else:
        expected_raw_path = str(raw_tsv_path)
        expected_query_path = str(query_fasta_path)
    if not sidecar["protein_id"].is_unique or len(sidecar) != len(queries):
        raise ValueError("MMseqs sidecar row/key count mismatch")
    expected_ids = set(queries["protein_id"].astype(str))
    if set(sidecar["protein_id"].astype(str)) != expected_ids:
        raise ValueError("MMseqs sidecar query IDs mismatch")
    if not all(isinstance(value, (bool, np.bool_)) for value in sidecar["overlap_flag"]):
        raise ValueError("MMseqs overlap flags must be measured booleans")
    expected_params = _params_digest(
        search_kind=search_kind, min_seq_id=min_seq_id,
        cov_mode=cov_mode, coverage=coverage,
    )
    expected_sequences = {
        str(row.protein_id): _sha(str(row.sequence).encode())
        for row in queries.itertuples(index=False)
    }
    for row in sidecar.itertuples(index=False):
        if row.query_sequence_sha256 != expected_sequences[str(row.protein_id)]:
            raise ValueError("MMseqs query sequence digest mismatch")
        expected_identity = (
            row.search_kind == search_kind
            and row.reference_sha256 == reference_sha256
            and row.tool_sha256 == tool_sha256
            and row.tool_version == tool_version
            and int(row.cov_mode) == int(cov_mode)
            and float(row.coverage) == float(coverage)
            and float(row.min_seq_id) == float(min_seq_id)
            and row.parameters_sha256 == expected_params
            and row.evidence_status == "measured_mmseqs"
            and isinstance(row.raw_output_sha256, str)
            and len(row.raw_output_sha256) == 64
            and row.raw_output_path == expected_raw_path
            and row.query_fasta_path == expected_query_path
            and row.raw_output_sha256 == _sha(raw_tsv_path.read_bytes())
            and row.query_fasta_sha256 == _sha(query_fasta_path.read_bytes())
        )
        if not expected_identity:
            if row.raw_output_sha256 != _sha(raw_tsv_path.read_bytes()):
                raise ValueError("MMseqs persisted raw TSV digest mismatch")
            raise ValueError("MMseqs sidecar evaluator identity mismatch")
    rebuilt = build_mmseqs_sidecar(
        queries,
        raw_tsv_path=raw_tsv_path,
        query_fasta_path=query_fasta_path,
        search_kind=search_kind,
        reference_sha256=reference_sha256,
        tool_sha256=tool_sha256,
        tool_version=tool_version,
        cov_mode=cov_mode,
        coverage=coverage,
        min_seq_id=min_seq_id,
        path_root=path_root,
    )
    extra_columns = set(sidecar) - set(rebuilt)
    allowed_extra = (
        {"dataset_release_id", "sequence_sha256", "head_checkpoint_sha256"}
        if search_kind == "head" else set()
    )
    if extra_columns != (extra_columns & allowed_extra):
        raise ValueError("MMseqs sidecar contains unsupported annotation columns")
    try:
        pd.testing.assert_frame_equal(
            sidecar[list(rebuilt.columns)].sort_values("protein_id").reset_index(drop=True),
            rebuilt.sort_values("protein_id").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("MMseqs sidecar does not replay from persisted evidence") from exc


def run_mmseqs_easy_search(
    *, tool_path: Path, query_fasta_path: Path, reference_fasta_path: Path,
    output_dir: Path, cov_mode: int, coverage: float, min_seq_id: float,
    threads: int = 1, timeout_s: int = 3600,
) -> dict[str, Path]:
    """Run one exact MMseqs search and retain command/stdout/stderr/result identities."""

    tool_path = Path(tool_path).resolve()
    query_fasta_path = Path(query_fasta_path).resolve()
    reference_fasta_path = Path(reference_fasta_path).resolve()
    output_dir = Path(output_dir)
    if (
        not tool_path.is_file() or not query_fasta_path.is_file()
        or not reference_fasta_path.is_file() or cov_mode not in {0, 2}
        or not 0.0 <= float(coverage) <= 1.0
        or not 0.0 <= float(min_seq_id) <= 1.0 or int(threads) < 1
        or int(timeout_s) < 1
    ):
        raise ValueError("invalid MMseqs easy-search contract")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "hits.tsv"
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    with tempfile.TemporaryDirectory(prefix="ifbench-v3-mmseqs-") as temporary:
        command = [
            str(tool_path), "easy-search", str(query_fasta_path),
            str(reference_fasta_path), str(raw_path), str(Path(temporary) / "tmp"),
            "--min-seq-id", str(float(min_seq_id)), "--cov-mode", str(int(cov_mode)),
            "-c", str(float(coverage)), "--format-output",
            "query,target,fident,qcov,tcov", "--threads", str(int(threads)),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=int(timeout_s),
            env={**os.environ, "LC_ALL": "C"},
        )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0 or not raw_path.is_file():
        raise RuntimeError(f"MMseqs easy-search failed with rc={completed.returncode}")
    version = subprocess.run(
        [str(tool_path), "version"], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise RuntimeError("MMseqs version probe failed")
    version_path = output_dir / "version.txt"
    version_path.write_text(version.stdout)
    help_result = subprocess.run(
        [str(tool_path), "easy-search", "-h"], text=True, capture_output=True,
        check=False, timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    help_path = output_dir / "easy_search_help.txt"
    help_path.write_text(help_result.stdout)
    if (
        help_result.returncode != 0
        or not all(token in help_result.stdout for token in (
            "--min-seq-id", "--cov-mode", "--format-output", "--threads",
        ))
    ):
        raise RuntimeError("MMseqs easy-search help/default contract probe failed")
    manifest = {
        "schema_version": "if-benchmark-v3-mmseqs-run/1",
        "command_argv": command, "returncode": int(completed.returncode),
        "tool_path": str(tool_path), "tool_sha256": _sha(tool_path.read_bytes()),
        "tool_version": version.stdout.strip(),
        "query_fasta_path": str(query_fasta_path),
        "query_fasta_sha256": _sha(query_fasta_path.read_bytes()),
        "reference_fasta_path": str(reference_fasta_path),
        "reference_fasta_sha256": _sha(reference_fasta_path.read_bytes()),
        "parameters": {
            "min_seq_id": float(min_seq_id), "cov_mode": int(cov_mode),
            "coverage": float(coverage), "threads": int(threads),
        },
        "outputs": {
            name: {"path": path.name, "size_bytes": path.stat().st_size,
                   "sha256": _sha(path.read_bytes())}
            for name, path in (
                ("raw_tsv", raw_path), ("stdout", stdout_path),
                ("stderr", stderr_path), ("version", version_path),
                ("help", help_path),
            )
        },
    }
    manifest["manifest_sha256"] = _sha(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_mmseqs_run_manifest(manifest_path)
    return {
        "raw_output": raw_path, "stdout": stdout_path, "stderr": stderr_path,
        "version": version_path, "manifest": manifest_path,
        "help": help_path,
    }


def validate_mmseqs_run_manifest(
    manifest_path: Path, *, replay: bool = False,
) -> dict[str, Any]:
    """Rehash one exact easy-search invocation and optionally rerun its native TSV."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != "if-benchmark-v3-mmseqs-run/1"
        or digest != _sha(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")
        ).encode())
        or int(manifest.get("returncode", -1)) != 0
        or set(manifest.get("outputs") or {}) != {
            "raw_tsv", "stdout", "stderr", "version", "help"
        }
    ):
        raise ValueError("MMseqs run manifest identity mismatch")
    tool = Path(str(manifest.get("tool_path") or "")).resolve()
    query = Path(str(manifest.get("query_fasta_path") or "")).resolve()
    reference = Path(str(manifest.get("reference_fasta_path") or "")).resolve()
    if (
        not tool.is_file() or _sha(tool.read_bytes()) != manifest.get("tool_sha256")
        or not query.is_file() or _sha(query.read_bytes()) != manifest.get("query_fasta_sha256")
        or not reference.is_file()
        or _sha(reference.read_bytes()) != manifest.get("reference_fasta_sha256")
    ):
        raise ValueError("MMseqs run tool/query/reference bytes changed")
    outputs: dict[str, Path] = {}
    for label, identity in manifest["outputs"].items():
        raw = Path(str(identity.get("path") or ""))
        path = (root / raw).resolve()
        if (
            raw.is_absolute() or ".." in raw.parts or path.parent != root
            or not path.is_file() or path.stat().st_size != int(identity.get("size_bytes", -1))
            or _sha(path.read_bytes()) != identity.get("sha256")
        ):
            raise ValueError(f"MMseqs run {label} output identity mismatch")
        outputs[label] = path
    params = manifest.get("parameters") or {}
    if (
        int(params.get("cov_mode", -1)) not in {0, 2}
        or not 0.0 <= float(params.get("coverage", -1.0)) <= 1.0
        or not 0.0 <= float(params.get("min_seq_id", -1.0)) <= 1.0
        or int(params.get("threads", 0)) < 1
    ):
        raise ValueError("MMseqs run parameter contract mismatch")
    command = manifest.get("command_argv")
    if not isinstance(command, list) or len(command) != 16:
        raise ValueError("MMseqs run command argv is malformed")
    expected = [
        str(tool), "easy-search", str(query), str(reference), str(outputs["raw_tsv"]),
        str(command[5]), "--min-seq-id", str(float(params["min_seq_id"])),
        "--cov-mode", str(int(params["cov_mode"])), "-c", str(float(params["coverage"])),
        "--format-output", "query,target,fident,qcov,tcov",
        "--threads", str(int(params["threads"])),
    ]
    if command != expected or not str(command[5]):
        raise ValueError("MMseqs run command differs from frozen easy-search contract")
    version = subprocess.run(
        [str(tool), "version"], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    if (
        version.returncode != 0 or version.stdout.strip() != manifest.get("tool_version")
        or outputs["version"].read_text() != version.stdout
    ):
        raise ValueError("MMseqs run version identity mismatch")
    help_result = subprocess.run(
        [str(tool), "easy-search", "-h"], text=True, capture_output=True, check=False,
        timeout=60, env={**os.environ, "LC_ALL": "C"},
    )
    if (
        help_result.returncode != 0
        or _canonicalize_mmseqs_help(outputs["help"].read_text())
        != _canonicalize_mmseqs_help(help_result.stdout)
        or not all(token in help_result.stdout for token in (
            "--min-seq-id", "--cov-mode", "--format-output", "--threads",
        ))
    ):
        raise ValueError("MMseqs run help/effective-default identity mismatch")
    raw_lines = sorted(line for line in outputs["raw_tsv"].read_text().splitlines() if line)
    for line in raw_lines:
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError("MMseqs run TSV schema mismatch")
        try:
            identity, qcov, tcov = map(float, fields[2:])
        except ValueError as exc:
            raise ValueError("MMseqs run TSV has nonnumeric metrics") from exc
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (
            identity, qcov, tcov,
        )):
            raise ValueError("MMseqs run TSV metric range mismatch")
        coverage_pass = (
            qcov >= float(params["coverage"]) and tcov >= float(params["coverage"])
            if int(params["cov_mode"]) == 0 else qcov >= float(params["coverage"])
        )
        if not coverage_pass:
            raise ValueError("MMseqs run TSV violates the frozen coverage filter")
    if replay:
        with tempfile.TemporaryDirectory(prefix="ifbench-v3-mmseqs-replay-") as temporary:
            temporary = Path(temporary)
            replay_raw = temporary / "hits.tsv"
            replay_command = [
                str(tool), "easy-search", str(query), str(reference), str(replay_raw),
                str(temporary / "tmp"), *expected[6:],
            ]
            completed = subprocess.run(
                replay_command, text=True, capture_output=True, check=False, timeout=3600,
                env={**os.environ, "LC_ALL": "C"},
            )
            replay_lines = sorted(
                line for line in replay_raw.read_text().splitlines() if line
            ) if replay_raw.is_file() else []
        if completed.returncode != 0 or replay_lines != raw_lines:
            raise ValueError("MMseqs native easy-search replay differs from persisted TSV")
    return manifest
