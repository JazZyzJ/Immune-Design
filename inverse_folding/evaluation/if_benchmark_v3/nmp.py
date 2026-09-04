"""Content-bound NetMHCIIpan install and invocation contracts for benchmark v3."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
import shutil
import threading
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


NMP_WINDOW_COLUMNS = [
    "protein_id", "peptide_length", "start_0b", "end_0b", "peptide", "rank_el",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def data_directory_file_table(data_root: Path) -> list[dict[str, Any]]:
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    return [
        {"path": str(path.relative_to(data_root)), "size_bytes": path.stat().st_size,
         "sha256": _sha(path)}
        for path in sorted(data_root.rglob("*"), key=lambda item: str(item.relative_to(data_root)))
        if path.is_file()
    ]


def write_nmp_install_manifest(
    *, tool_path: Path, data_root: Path, tool_version: str,
    manifest_path: Path, effective_environment: Mapping[str, str],
    identity_mode: str = "official_install_merkle",
) -> dict[str, Any]:
    tool_path, data_root = Path(tool_path).resolve(), Path(data_root).resolve()
    if not tool_path.is_file() or not tool_version:
        raise ValueError("NetMHCIIpan tool/version identity is incomplete")
    files = data_directory_file_table(data_root)
    if not files:
        raise ValueError("NetMHCIIpan data/model directory is empty")
    if identity_mode not in {"official_install_merkle", "fixture"}:
        raise ValueError("invalid NetMHCIIpan install identity mode")
    if identity_mode == "official_install_merkle":
        names = {row["path"] for row in files}
        if "data/version" not in names or not any(
            name.startswith("Linux_") and "NetMHCIIpan" in Path(name).name for name in names
        ):
            raise ValueError("official NetMHCIIpan install lacks data/version or compiled model binary")
    manifest_root = Path(manifest_path).resolve().parent
    try:
        rendered_tool = str(tool_path.relative_to(manifest_root))
    except ValueError:
        rendered_tool = str(tool_path)
    try:
        rendered_data_root = str(data_root.relative_to(manifest_root))
    except ValueError:
        rendered_data_root = str(data_root)
    payload = {
        "schema_version": "if-benchmark-v3-nmp-install/1",
        "identity_mode": identity_mode,
        "tool_path": rendered_tool, "tool_sha256": _sha(tool_path),
        "tool_version": str(tool_version), "data_root": rendered_data_root,
        "data_files": files,
        "data_merkle_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "effective_environment": dict(sorted(effective_environment.items())),
    }
    Path(manifest_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def nmp_parameter_payload(
    *, stage: str, allele: str, peptide_lengths: Sequence[int],
    install_manifest_sha256: str, batch_size: int, subprocess_timeout_s: int,
    max_lengths_per_call: int, n_workers: int,
) -> dict[str, Any]:
    lengths = [int(value) for value in peptide_lengths]
    if stage not in {"c5", "c7"} or not lengths:
        raise ValueError("invalid NMP stage/length contract")
    allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")
    return {
        "schema_version": "if-benchmark-v3-nmp-parameters/2",
        "stage": stage, "allele": allele, "allele_argument": allele_fmt,
        "peptide_lengths": lengths, "rank_el_threshold": 2.0,
        # NetMHCIIpan 4.3i implements ``-filter`` as a presence toggle: even
        # ``-filter 0`` enables filtering.  Complete-window production therefore
        # omits the flag and records the empirically verified 4.3 contract.
        "filter_mode": "off_by_omission_verified_netmhciipan_4_3",
        "context_mode": "on_by_presence_verified_netmhciipan_4_3",
        "install_manifest_sha256": install_manifest_sha256,
        "command_argv_template": [
            "{netmhciipan_binary}", "-f", "{input_fasta}", "-a", allele_fmt,
            "-length", "{comma_separated_lengths}", "-context",
        ],
        "runner": {
            "batch_size": int(batch_size),
            "subprocess_timeout_s": int(subprocess_timeout_s),
            "max_lengths_per_call": int(max_lengths_per_call),
            "n_workers": int(n_workers),
        },
    }


def peptide_scores_to_native_percent_rows(
    *, protein_id: str, sequence: str, scores_by_length: Mapping[int, Sequence[Any]],
    requested_lengths: Sequence[int],
) -> list[dict[str, Any]]:
    """Convert runner fractions to native 0--100 %Rank_EL and prove window completeness."""

    rows: list[dict[str, Any]] = []
    for length in sorted(int(value) for value in requested_lengths):
        expected_positions = list(range(max(0, len(sequence) - length + 1)))
        scores = list(scores_by_length.get(length, []))
        observed_positions = [int(score.pos) for score in scores]
        if observed_positions != expected_positions:
            raise ValueError(
                f"NetMHCIIpan incomplete/noncanonical positions for {protein_id} length {length}"
            )
        for score in scores:
            start = int(score.pos)
            peptide = str(score.peptide)
            if len(peptide) != length or peptide != sequence[start:start + length]:
                raise ValueError("NetMHCIIpan peptide/position differs from input sequence")
            rank_percent = float(score.el_rank) * 100.0
            if not 0.0 <= rank_percent <= 100.0:
                raise ValueError("NetMHCIIpan native percentage rank is outside [0,100]")
            rows.append({
                "protein_id": protein_id, "peptide_length": length,
                "start_0b": start, "end_0b": start + length,
                "peptide": peptide, "rank_el": rank_percent,
            })
    return rows


def _query_identity(
    queries: pd.DataFrame, *, key_column: str, sequence_column: str,
    sequence_sha_column: str,
) -> str:
    records = [
        {
            "protein_id": str(row[key_column]), "sequence": str(row[sequence_column]),
            "sequence_sha256": str(row[sequence_sha_column]),
        }
        for row in queries.sort_values(key_column, kind="stable").to_dict("records")
    ]
    return hashlib.sha256(_canonical(records)).hexdigest()


def _coverage(sequence: str, rows: pd.DataFrame) -> float:
    covered: set[int] = set()
    for item in rows.itertuples(index=False):
        start, end = int(item.start_0b), int(item.end_0b)
        if str(item.peptide) != sequence[start:end]:
            raise ValueError("NMP shard peptide does not match query sequence")
        rank = float(item.rank_el)
        if not math.isfinite(rank) or not 0.0 <= rank <= 100.0:
            raise ValueError("NMP shard rank is outside native [0,100] percentage")
        if rank < 2.0:
            covered.update(range(start, end))
    return len(covered) / len(sequence)


def write_nmp_sharded_evidence(
    *, output_dir: Path, queries: pd.DataFrame, key_column: str,
    sequence_column: str, sequence_sha_column: str, stage: str,
    peptide_lengths: Sequence[int], tool_path: Path, parameters_path: Path,
    score_shard_fn: Callable[[pd.DataFrame, Path], pd.DataFrame], shard_query_count: int,
) -> Path:
    """Produce bounded-memory canonical NMP shards and a content-Merkle manifest.

    ``score_shard_fn`` may invoke the real binary or a deterministic test fixture.  It receives
    only one lexically contiguous query shard and must return native 0--100 percentage ranks.
    """

    lengths = sorted(int(value) for value in peptide_lengths)
    required = {key_column, sequence_column, sequence_sha_column}
    if (
        stage not in {"c5", "c7"} or not lengths or len(lengths) != len(set(lengths))
        or not required <= set(queries) or int(shard_query_count) < 1
        or not Path(tool_path).is_file() or not Path(parameters_path).is_file()
    ):
        raise ValueError("invalid NMP shard producer contract")
    ordered = queries.sort_values(key_column, kind="stable").reset_index(drop=True)
    if ordered[key_column].isna().any() or ordered[key_column].duplicated().any():
        raise ValueError("NMP shard query keys are missing or duplicated")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shards_root = output_dir / "shards"
    shards_root.mkdir(parents=True)
    tool_sha = _sha(Path(tool_path))
    parameters_sha = _sha(Path(parameters_path))
    shard_records: list[dict[str, Any]] = []
    total_windows = 0
    for shard_index, start in enumerate(range(0, len(ordered), int(shard_query_count))):
        shard_queries = ordered.iloc[start:start + int(shard_query_count)].copy()
        query_ids = shard_queries[key_column].astype(str).tolist()
        shard_dir = shards_root / f"{shard_index:05d}"
        shard_dir.mkdir()
        produced = score_shard_fn(shard_queries.copy(), shard_dir)
        if list(produced.columns) != NMP_WINDOW_COLUMNS:
            raise ValueError("NMP shard scorer returned a noncanonical window schema")
        windows = produced.copy()
        windows["protein_id"] = windows["protein_id"].astype(str)
        windows = windows.sort_values(
            ["protein_id", "peptide_length", "start_0b"], kind="stable"
        ).reset_index(drop=True)
        if set(windows["protein_id"]) != set(query_ids):
            raise ValueError("NMP shard scorer query IDs differ from shard assignment")
        raw_path = shard_dir / "raw_windows.tsv"
        windows_path = shard_dir / "windows.parquet"
        summary_path = shard_dir / "summary.parquet"
        windows.to_csv(raw_path, sep="\t", header=False, index=False)
        windows.to_parquet(windows_path, index=False)
        raw_sha = _sha(raw_path)
        summary_rows = []
        by_id = {pid: group for pid, group in windows.groupby("protein_id", sort=False)}
        for query in shard_queries.to_dict("records"):
            pid = str(query[key_column])
            sequence = str(query[sequence_column])
            sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
            if sequence_sha != str(query[sequence_sha_column]):
                raise ValueError("NMP shard query sequence digest mismatch")
            observed = by_id[pid]
            expected_count = sum(max(0, len(sequence) - length + 1) for length in lengths)
            if len(observed) != expected_count:
                raise ValueError("NMP shard scorer returned incomplete window count")
            expected_keys = {
                (length, position, position + length)
                for length in lengths for position in range(max(0, len(sequence) - length + 1))
            }
            observed_keys = {
                (int(item.peptide_length), int(item.start_0b), int(item.end_0b))
                for item in observed.itertuples(index=False)
            }
            if observed_keys != expected_keys:
                raise ValueError("NMP shard scorer returned incomplete/duplicate coordinates")
            summary_rows.append({
                "protein_id": pid, "sequence_sha256": sequence_sha, "status": "complete",
                "coverage_fraction": _coverage(sequence, observed),
                "raw_evidence_sha256": raw_sha, "tool_sha256": tool_sha,
                "parameters_sha256": parameters_sha,
                "scored_lengths_json": json.dumps(lengths, separators=(",", ":")),
            })
        pd.DataFrame(summary_rows).to_parquet(summary_path, index=False)
        identities = {
            label: {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size, "sha256": _sha(path),
            }
            for label, path in (
                ("raw_output", raw_path), ("windows", windows_path),
                ("summary", summary_path),
            )
        }
        shard_records.append({
            "shard_index": shard_index, "query_count": len(shard_queries),
            "window_count": len(windows), "first_query_id": query_ids[0],
            "last_query_id": query_ids[-1],
            "query_ids_sha256": hashlib.sha256(_canonical(query_ids)).hexdigest(),
            "files": identities,
            "native_files": [
                {"path": str(path.relative_to(output_dir)),
                 "size_bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in sorted(
                    (shard_dir / "native").rglob("*"),
                    key=lambda item: str(item.relative_to(shard_dir / "native")),
                ) if path.is_file()
            ] if (shard_dir / "native").is_dir() else [],
        })
        total_windows += len(windows)
    payload = {
        "schema_version": "if-benchmark-v3-nmp-shards/1", "stage": stage,
        "peptide_lengths": lengths, "key_column": key_column,
        "sequence_column": sequence_column, "sequence_sha_column": sequence_sha_column,
        "shard_query_count": int(shard_query_count), "query_count": len(ordered),
        "window_count": total_windows,
        "query_identity_sha256": _query_identity(
            ordered, key_column=key_column, sequence_column=sequence_column,
            sequence_sha_column=sequence_sha_column,
        ),
        "tool_sha256": tool_sha, "parameters_sha256": parameters_sha,
        "shards": shard_records,
    }
    payload["shard_merkle_sha256"] = hashlib.sha256(_canonical(shard_records)).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_nmp_shard_manifest(
    path: Path, *, queries: pd.DataFrame, key_column: str, sequence_column: str,
    sequence_sha_column: str, stage: str, peptide_lengths: Sequence[int],
    tool_path: Path, parameters_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Path]]]:
    """Verify manifest/file Merkle and exact deterministic query-to-shard partition."""

    path = Path(path).resolve()
    root = path.parent
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    lengths = sorted(int(value) for value in peptide_lengths)
    ordered = queries.sort_values(key_column, kind="stable").reset_index(drop=True)
    if (
        payload.get("schema_version") != "if-benchmark-v3-nmp-shards/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or payload.get("stage") != stage or payload.get("peptide_lengths") != lengths
        or payload.get("key_column") != key_column
        or payload.get("sequence_column") != sequence_column
        or payload.get("sequence_sha_column") != sequence_sha_column
        or int(payload.get("query_count", -1)) != len(ordered)
        or payload.get("query_identity_sha256") != _query_identity(
            ordered, key_column=key_column, sequence_column=sequence_column,
            sequence_sha_column=sequence_sha_column,
        )
        or payload.get("tool_sha256") != _sha(Path(tool_path))
        or payload.get("parameters_sha256") != _sha(Path(parameters_path))
        or payload.get("shard_merkle_sha256")
        != hashlib.sha256(_canonical(payload.get("shards") or [])).hexdigest()
    ):
        raise ValueError("NMP shard manifest/query/tool identity mismatch")
    shard_size = int(payload.get("shard_query_count", 0))
    if shard_size < 1:
        raise ValueError("NMP shard size is invalid")
    resolved_shards: list[dict[str, Path]] = []
    seen_paths: set[Path] = set()
    total_windows = 0
    for index, shard in enumerate(payload.get("shards") or []):
        expected_queries = ordered.iloc[index * shard_size:(index + 1) * shard_size]
        expected_ids = expected_queries[key_column].astype(str).tolist()
        if (
            int(shard.get("shard_index", -1)) != index
            or int(shard.get("query_count", -1)) != len(expected_ids)
            or not expected_ids
            or shard.get("first_query_id") != expected_ids[0]
            or shard.get("last_query_id") != expected_ids[-1]
            or shard.get("query_ids_sha256")
            != hashlib.sha256(_canonical(expected_ids)).hexdigest()
        ):
            raise ValueError("NMP shard query partition is not deterministic/complete")
        resolved: dict[str, Path] = {}
        if set(shard.get("files") or {}) != {"raw_output", "windows", "summary"}:
            raise ValueError("NMP shard file identity set is incomplete")
        for label, identity in shard["files"].items():
            raw = Path(str(identity.get("path") or ""))
            file_path = (root / raw).resolve()
            if (
                raw.is_absolute() or ".." in raw.parts or root not in file_path.parents
                or not file_path.is_file() or file_path in seen_paths
                or file_path.stat().st_size != int(identity["size_bytes"])
                or _sha(file_path) != identity["sha256"]
            ):
                raise ValueError("NMP shard file path/bytes identity mismatch")
            seen_paths.add(file_path)
            resolved[label] = file_path
        native_files = shard.get("native_files")
        if not isinstance(native_files, list):
            raise ValueError("NMP shard native evidence table is missing")
        for identity in native_files:
            raw = Path(str(identity.get("path") or ""))
            file_path = (root / raw).resolve()
            if (
                raw.is_absolute() or ".." in raw.parts or root not in file_path.parents
                or not file_path.is_file() or file_path in seen_paths
                or file_path.stat().st_size != int(identity["size_bytes"])
                or _sha(file_path) != identity["sha256"]
            ):
                raise ValueError("NMP shard native file path/bytes identity mismatch")
            seen_paths.add(file_path)
        resolved_shards.append(resolved)
        total_windows += int(shard["window_count"])
    observed_files = {
        item.resolve() for item in (root / "shards").rglob("*") if item.is_file()
    }
    if observed_files != seen_paths:
        raise ValueError("NMP shard directory contains unregistered/missing files")
    if (
        sum(int(item["query_count"]) for item in payload.get("shards") or []) != len(ordered)
        or total_windows != int(payload.get("window_count", -1))
    ):
        raise ValueError("NMP shard manifest aggregate counts mismatch")
    return payload, resolved_shards


def merge_nmp_sharded_evidence(
    *, source_manifests: Sequence[Path], output_dir: Path, queries: pd.DataFrame,
    key_column: str, sequence_column: str, sequence_sha_column: str,
    stage: str, peptide_lengths: Sequence[int], tool_path: Path, parameters_path: Path,
) -> Path:
    """Merge contiguous one-shard array outputs into one replayable canonical manifest."""

    manifests = [Path(path).resolve() for path in source_manifests]
    output_dir = Path(output_dir)
    if not manifests or output_dir.exists():
        raise ValueError("invalid NMP array merge destination/source set")
    payloads = [json.loads(path.read_text()) for path in manifests]
    invariant_fields = (
        "schema_version", "stage", "peptide_lengths", "key_column", "sequence_column",
        "sequence_sha_column", "tool_sha256", "parameters_sha256",
    )
    first = payloads[0]
    for payload in payloads:
        unsigned = dict(payload)
        digest = unsigned.pop("manifest_sha256", None)
        if digest != hashlib.sha256(_canonical(unsigned)).hexdigest():
            raise ValueError("NMP array source manifest self-hash mismatch")
    if any(
        any(payload.get(field) != first.get(field) for field in invariant_fields)
        for payload in payloads[1:]
    ):
        raise ValueError("NMP array source manifests disagree on scoring identity")
    if any(len(payload.get("shards") or []) != 1 for payload in payloads):
        raise ValueError("NMP array merge requires exactly one native shard per task")
    ordered = queries.sort_values(key_column, kind="stable").reset_index(drop=True)
    chunk_size = (len(ordered) + len(payloads) - 1) // len(payloads)
    if chunk_size < 1:
        raise ValueError("NMP array merge query table is empty")
    output_dir.mkdir(parents=True)
    shards_root = output_dir / "shards"
    shards_root.mkdir()
    merged_shards = []
    query_cursor = 0
    total_windows = 0
    for index, (manifest_path, payload) in enumerate(zip(manifests, payloads, strict=True)):
        source_root = manifest_path.parent
        source_record = payload["shards"][0]
        expected = ordered.iloc[index * chunk_size:(index + 1) * chunk_size]
        expected_ids = expected[key_column].astype(str).tolist()
        if (
            not expected_ids
            or int(source_record["query_count"]) != len(expected_ids)
            or source_record["first_query_id"] != expected_ids[0]
            or source_record["last_query_id"] != expected_ids[-1]
            or source_record["query_ids_sha256"] != hashlib.sha256(
                _canonical(expected_ids)
            ).hexdigest()
        ):
            raise ValueError("NMP array shard does not match canonical contiguous assignment")
        destination = shards_root / f"{index:05d}"
        source_shard = source_root / "shards" / "00000"
        if not source_shard.is_dir():
            raise ValueError("NMP array source shard directory is missing")
        shutil.copytree(source_shard, destination)

        def rewrite(identity: Mapping[str, Any]) -> dict[str, Any]:
            raw = (source_root / str(identity["path"])).resolve()
            relative_inside = raw.relative_to(source_shard.resolve())
            copied = destination / relative_inside
            if (
                not copied.is_file() or copied.stat().st_size != int(identity["size_bytes"])
                or _sha(copied) != identity["sha256"]
            ):
                raise ValueError("NMP array copied shard file identity mismatch")
            return {
                "path": str(copied.relative_to(output_dir)),
                "size_bytes": copied.stat().st_size, "sha256": _sha(copied),
            }

        files = {name: rewrite(identity) for name, identity in source_record["files"].items()}
        native_files = [rewrite(identity) for identity in source_record.get("native_files") or []]
        merged_shards.append({
            **{
                key: value for key, value in source_record.items()
                if key not in {"shard_index", "files", "native_files"}
            },
            "shard_index": index, "files": files, "native_files": native_files,
        })
        query_cursor += len(expected_ids)
        total_windows += int(source_record["window_count"])
    if query_cursor != len(ordered):
        raise ValueError("NMP array merged shards do not cover complete query table")
    merged = {
        **{field: first[field] for field in invariant_fields},
        "shard_query_count": chunk_size,
        "query_count": len(ordered), "window_count": total_windows,
        "query_identity_sha256": _query_identity(
            ordered, key_column=key_column, sequence_column=sequence_column,
            sequence_sha_column=sequence_sha_column,
        ),
        "shards": merged_shards,
    }
    merged["shard_merkle_sha256"] = hashlib.sha256(_canonical(merged_shards)).hexdigest()
    merged["manifest_sha256"] = hashlib.sha256(_canonical(merged)).hexdigest()
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    load_nmp_shard_manifest(
        path, queries=ordered, key_column=key_column, sequence_column=sequence_column,
        sequence_sha_column=sequence_sha_column, stage=stage,
        peptide_lengths=peptide_lengths, tool_path=tool_path,
        parameters_path=parameters_path,
    )
    return path


def _read_fasta_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current is not None:
                records[current] = "".join(chunks)
            current = line[1:].split()[0]
            if not current or current in records:
                raise ValueError("native NMP FASTA has missing/duplicate ID")
            chunks = []
        elif current is None:
            if line.strip():
                raise ValueError("native NMP FASTA sequence precedes ID")
        else:
            chunks.append(line.strip())
    if current is not None:
        records[current] = "".join(chunks)
    return records


def replay_native_nmp_shard(
    *, shard_record: Mapping[str, Any], manifest_root: Path,
    expected_queries: pd.DataFrame, key_column: str, sequence_column: str,
    tool_path: Path, parameters_path: Path,
) -> pd.DataFrame:
    """Reparse saved native stdout and prove it is exactly the canonical shard table."""

    from epitope_head.data.netmhciipan_runner import StandaloneRunner

    parameters = json.loads(Path(parameters_path).read_text())
    expected_lengths = sorted(int(value) for value in parameters["peptide_lengths"])
    allele_argument = str(parameters["allele_argument"])
    query_sequences = {
        str(row[key_column]): str(row[sequence_column])
        for row in expected_queries.to_dict("records")
    }
    native_table = shard_record.get("native_files")
    if not isinstance(native_table, list) or not native_table:
        raise ValueError("production NMP shard lacks native call evidence")
    native_paths = {
        (Path(manifest_root) / str(item["path"])).resolve(): item
        for item in native_table
    }
    record_paths = sorted(path for path in native_paths if path.suffix == ".json")
    if not record_paths:
        raise ValueError("production NMP shard has no native call records")
    contributed: list[dict[str, Any]] = []
    successful_pairs: set[tuple[str, int]] = set()
    for record_path in record_paths:
        record = json.loads(record_path.read_text())
        unsigned = dict(record)
        digest = unsigned.pop("record_sha256", None)
        if (
            record.get("schema_version") != "if-benchmark-v3-nmp-native-call/2"
            or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
            or not str(record.get("started_at_utc") or "")
            or not str(record.get("ended_at_utc") or "")
            or record.get("canonical_working_directory") != "native"
        ):
            raise ValueError("native NMP call record self/temporal identity mismatch")
        parent = record_path.parent
        referenced = {}
        for label, sha_label in (
            ("input_fasta", "input_fasta_sha256"),
            ("stdout", "stdout_sha256"), ("stderr", "stderr_sha256"),
        ):
            raw = Path(str(record.get(label) or ""))
            path = (parent / raw).resolve()
            if (
                raw.is_absolute() or ".." in raw.parts or path.parent != parent
                or path not in native_paths or not path.is_file()
                or _sha(path) != record.get(sha_label)
            ):
                raise ValueError(f"native NMP {label} bytes/record identity mismatch")
            referenced[label] = path
        fasta = _read_fasta_records(referenced["input_fasta"])
        native_map = record.get("native_id_map")
        if (
            not isinstance(native_map, Mapping) or set(native_map) != set(fasta)
            or any(str(value) not in query_sequences for value in native_map.values())
            or any(fasta[native_id] != query_sequences[str(original)]
                   for native_id, original in native_map.items())
        ):
            raise ValueError("native NMP FASTA/native-ID/original-ID mapping mismatch")
        argv = record.get("canonical_argv")
        if not isinstance(argv, list):
            raise ValueError("native NMP canonical argv is missing")
        try:
            length_values = sorted(int(value) for value in argv[argv.index("-length") + 1].split(","))
            expected_argv = [
                str(Path(tool_path).resolve()), "-f", referenced["input_fasta"].name,
                "-a", allele_argument, "-length", ",".join(str(value) for value in length_values),
                "-context",
            ]
        except (ValueError, IndexError, AttributeError) as exc:
            raise ValueError("native NMP canonical argv is malformed") from exc
        if (
            argv != expected_argv or not set(length_values) <= set(expected_lengths)
            or not length_values
        ):
            raise ValueError("native NMP binary/allele/length/filter-off/context argv mismatch")
        status = str(record.get("status"))
        if status == "success":
            if (
                int(record.get("returncode", -1)) != 0
                or record.get("exception_type") is not None
                or record.get("exception_message") is not None
            ):
                raise ValueError("native NMP successful call has forged status/returncode")
            native_stdout = referenced["stdout"].read_text()
            if "# Prediction Mode: EL with Context" not in native_stdout:
                raise ValueError("native NMP stdout does not prove context mode is enabled")
            parsed = StandaloneRunner._parse_batch_output(None, native_stdout)
            if set(parsed) != set(native_map):
                raise ValueError("native NMP stdout ID set differs from explicit call mapping")
            for native_id, original_raw in native_map.items():
                original = str(original_raw)
                rows = peptide_scores_to_native_percent_rows(
                    protein_id=original, sequence=query_sequences[original],
                    scores_by_length=parsed[native_id], requested_lengths=length_values,
                )
                for length in length_values:
                    pair = (original, length)
                    if pair in successful_pairs:
                        raise ValueError("native NMP query/length has duplicate successful producer")
                    successful_pairs.add(pair)
                contributed.extend(rows)
        elif status in {"nonzero", "timeout", "exception"}:
            if status == "nonzero" and (
                record.get("returncode") is None or int(record["returncode"]) == 0
            ):
                raise ValueError("native NMP nonzero call has forged returncode")
            if status in {"timeout", "exception"} and not record.get("exception_type"):
                raise ValueError("native NMP failed call lacks exception evidence")
        else:
            raise ValueError("native NMP call status is invalid")
    expected_pairs = {
        (protein_id, length) for protein_id in query_sequences for length in expected_lengths
    }
    if successful_pairs != expected_pairs:
        raise ValueError("native NMP calls have silent missing query/length outputs")
    frame = pd.DataFrame(contributed, columns=NMP_WINDOW_COLUMNS)
    if frame.duplicated(["protein_id", "peptide_length", "start_0b"]).any():
        raise ValueError("native NMP replay has duplicate canonical window")
    return frame.sort_values(
        ["protein_id", "peptide_length", "start_0b"], kind="stable"
    ).reset_index(drop=True)


class _AuditedStandaloneRunner:
    """Thin composition wrapper that preserves every native subprocess input/output."""

    def __init__(self, *, audit_dir: Path, binary_path: Path, runner_parameters: Mapping[str, Any]):
        from epitope_head.data.netmhciipan_runner import StandaloneRunner

        audit_dir.mkdir(parents=True, exist_ok=False)
        self.audit_dir = audit_dir
        self._counter = 0
        self._lock = threading.Lock()
        outer = self

        class AuditedRunner(StandaloneRunner):
            def _run_audited(self, cmd, *, input_fasta, native_id_map):
                with outer._lock:
                    call_index = outer._counter
                    outer._counter += 1
                prefix = outer.audit_dir / f"call_{call_index:05d}"
                fasta_index = cmd.index("-f") + 1
                copied_fasta = prefix.with_suffix(".fasta")
                shutil.copyfile(input_fasta, copied_fasta)
                stdout_path = prefix.with_suffix(".stdout.txt")
                stderr_path = prefix.with_suffix(".stderr.txt")
                started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                result = None
                status = "exception"
                exception_type = None
                exception_message = None
                try:
                    result = super()._run_netmhciipan(cmd)
                    status = "success" if result.returncode == 0 else "nonzero"
                    return result
                except subprocess.TimeoutExpired as exc:
                    status = "timeout"
                    exception_type = type(exc).__name__
                    exception_message = str(exc)
                    raise
                except Exception as exc:
                    exception_type = type(exc).__name__
                    exception_message = str(exc)
                    raise
                finally:
                    stdout_path.write_text("" if result is None else result.stdout)
                    stderr_path.write_text("" if result is None else result.stderr)
                    canonical_argv = [str(value) for value in cmd]
                    canonical_argv[fasta_index] = copied_fasta.name
                    command_payload = {
                        "schema_version": "if-benchmark-v3-nmp-native-call/2",
                        "started_at_utc": started,
                        "ended_at_utc": datetime.now(timezone.utc).isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "status": status,
                        "exception_type": exception_type,
                        "exception_message": exception_message,
                        "original_argv": [str(value) for value in cmd],
                        "canonical_argv": canonical_argv,
                        "canonical_working_directory": "native",
                        "returncode": None if result is None else int(result.returncode),
                        "native_id_map": dict(sorted(native_id_map.items())),
                        "input_fasta": copied_fasta.name,
                        "input_fasta_sha256": _sha(copied_fasta),
                        "stdout": stdout_path.name, "stdout_sha256": _sha(stdout_path),
                        "stderr": stderr_path.name, "stderr_sha256": _sha(stderr_path),
                    }
                    command_payload["record_sha256"] = hashlib.sha256(
                        _canonical(command_payload)
                    ).hexdigest()
                    prefix.with_suffix(".json").write_text(
                        json.dumps(command_payload, indent=2, sort_keys=True) + "\n"
                    )

            def _run_chunk_lengths(
                self, chunk, allele_fmt, length_group, chunk_offset,
            ):  # type: ignore[override]
                short_to_orig = {}
                length_str = ",".join(str(value) for value in sorted(length_group))
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".fasta", delete=False
                ) as handle:
                    for index, (protein_id, sequence) in enumerate(chunk):
                        short_id = f"S{chunk_offset + index:06d}"
                        short_to_orig[short_id] = str(protein_id)
                        handle.write(f">{short_id}\n{sequence}\n")
                    fasta_path = Path(handle.name)
                command = [
                    str(self.binary_path), "-f", str(fasta_path), "-a", allele_fmt,
                    "-length", length_str, "-context",
                ]
                try:
                    result = self._run_audited(
                        command, input_fasta=fasta_path, native_id_map=short_to_orig
                    )
                    if result.returncode != 0:
                        raise StandaloneRunner.ChunkFailedError(
                            f"rc={result.returncode}, stderr={result.stderr[:300]}"
                        )
                    parsed = self._parse_batch_output(result.stdout)
                    remapped = {}
                    for short_id, by_length in parsed.items():
                        original = short_to_orig.get(short_id)
                        if original is None:
                            raise ValueError(
                                f"NetMHCIIpan emitted unknown native ID {short_id}"
                            )
                        remapped[original] = by_length
                    return remapped
                finally:
                    fasta_path.unlink(missing_ok=True)

        self.runner = AuditedRunner(
            binary_path=binary_path,
            batch_size=int(runner_parameters["batch_size"]),
            subprocess_timeout=int(runner_parameters["subprocess_timeout_s"]),
            max_lengths_per_call=int(runner_parameters["max_lengths_per_call"]),
            n_workers=int(runner_parameters["n_workers"]),
        )


def make_standalone_score_shard_fn(
    *, binary_path: Path, allele: str, peptide_lengths: Sequence[int],
    parameters_path: Path, key_column: str, sequence_column: str,
) -> Callable[[pd.DataFrame, Path], pd.DataFrame]:
    """Create the production shard scorer with complete native stdout/command evidence."""

    parameters = json.loads(Path(parameters_path).read_text())
    lengths = sorted(int(value) for value in peptide_lengths)
    if (
        parameters.get("allele") != allele
        or parameters.get("peptide_lengths") != lengths
        or not isinstance(parameters.get("runner"), Mapping)
    ):
        raise ValueError("standalone shard scorer differs from NMP parameter contract")

    def score(shard: pd.DataFrame, shard_dir: Path) -> pd.DataFrame:
        audited = _AuditedStandaloneRunner(
            audit_dir=shard_dir / "native", binary_path=Path(binary_path),
            runner_parameters=parameters["runner"],
        )
        entries = [
            (str(row[key_column]), str(row[sequence_column]))
            for row in shard.to_dict("records")
        ]
        results = audited.runner.score_batch(entries, allele, lengths)
        rows: list[dict[str, Any]] = []
        for protein_id, sequence in entries:
            if protein_id not in results:
                raise ValueError(f"NetMHCIIpan shard lacks result for {protein_id}")
            rows.extend(peptide_scores_to_native_percent_rows(
                protein_id=protein_id, sequence=sequence,
                scores_by_length=results[protein_id], requested_lengths=lengths,
            ))
        return pd.DataFrame(rows, columns=NMP_WINDOW_COLUMNS)

    return score
