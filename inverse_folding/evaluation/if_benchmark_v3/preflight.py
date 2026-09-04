"""Content-bound Tier-1 source preflight for ``if-benchmark-test-set/3``.

This module only establishes source-frame capacity. It does not select final Tier-1 rows:
structure materialization, final-byte CATH isolation, SIFTS-to-IF projection, and backfill happen
later. Every HTTP response and every source/chain rejection is retained so an apparent shortfall
cannot be confused with an API outage or an early top-N truncation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .source_identity import resolve_canonical_source_units
from .uniprot_snapshot import snapshot_uniprot_identity


SIFTS_BEST_STRUCTURES_URL = (
    "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{source_id}"
)


CHAIN_ATTEMPT_COLUMNS = [
    "source_id", "pdb_id", "auth_chain_id", "unp_start", "unp_end", "pdb_start",
    "pdb_end", "mapped_range_length", "resolution", "experimental_method",
    "sifts_coverage", "method_xray_pass", "resolution_pass",
    "mapped_range_length_pass", "mapped_range_sequence_pass", "verified_spans_pass",
    "source_epitope_coverage_pass", "verified_span_count",
    "verified_covered_residue_count", "source_epitope_coverage",
    "span_outside_range_count", "span_peptide_mismatch_count", "verified_spans_json",
    "failure_reasons_json", "source_valid", "sifts_item_index", "source_sequence_sha256",
]
VALID_SOURCE_UNIT_COLUMNS = [
    "source_id", "canonical_sequence_length", "canonical_sequence_sha256",
    "raw_source_ids_json", "viable_sifts_mapping_count",
]
SIFTS_REQUEST_COLUMNS = [
    "source_id", "url", "final_status", "network_failure", "attempt_count",
    "attempts_json", "response_path", "response_size_bytes", "response_sha256",
    "response_headers_json", "parse_status", "parse_error", "parsed_mapping_count",
]
SOURCE_STAGE_COLUMNS = [
    "source_id", "request_final_status", "parsed_mapping_count",
    "with_parsed_sifts_mapping", "with_any_xray", "with_any_xray_resolution",
    "with_any_xray_resolution_length", "with_any_xray_resolution_length_sequence",
    "with_any_xray_resolution_length_sequence_2_spans", "source_frame_valid",
    "source_terminal_status", "observed_chain_failure_reasons_json",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass(frozen=True)
class SiftsHttpResponse:
    """Small transport-neutral HTTP response used by the retrier and tests."""

    status: int
    body: bytes
    headers: Mapping[str, str]


@dataclass(frozen=True)
class SiftsRequestResult:
    source_id: str
    url: str
    final_status: str
    response_body: bytes
    response_sha256: str | None
    response_headers: Mapping[str, str]
    attempts: tuple[dict[str, Any], ...]
    network_failure: bool


def _default_request(url: str, timeout_s: float) -> SiftsHttpResponse:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "immune-design-if-benchmark-v3/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return SiftsHttpResponse(
            status=int(response.status),
            body=response.read(),
            headers={str(k).lower(): str(v) for k, v in response.headers.items()},
        )


def request_sifts_with_retry(
    source_id: str,
    *,
    request_fn: Callable[[str, float], SiftsHttpResponse] = _default_request,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_attempts: int = 4,
    timeout_s: float = 30.0,
    backoff_s: float = 1.0,
) -> SiftsRequestResult:
    """Fetch one PDBe SIFTS response with audited transient retries.

    HTTP 404 is a terminal, scientifically meaningful ``no endpoint mapping`` result. HTTP 429,
    5xx responses, URL errors, and timeouts are transient and retried. Exhausted transient errors
    remain ``network_failure`` and must block the preflight rather than masquerade as no mapping.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if not source_id or "/" in source_id:
        raise ValueError(f"invalid source_id: {source_id!r}")

    url = SIFTS_BEST_STRUCTURES_URL.format(
        source_id=urllib.parse.quote(source_id, safe="._-")
    )
    attempts: list[dict[str, Any]] = []

    for attempt_idx in range(1, max_attempts + 1):
        started = _utc_now()
        try:
            response = request_fn(url, timeout_s)
            status = int(response.status)
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "queried_at_utc": started,
                    "outcome": f"http_{status}",
                    "error": None,
                }
            )
            if status == 429 or 500 <= status <= 599:
                if attempt_idx < max_attempts:
                    sleep_fn(backoff_s * (2 ** (attempt_idx - 1)))
                    continue
                return SiftsRequestResult(
                    source_id, url, "network_failure", response.body,
                    _sha256_bytes(response.body), response.headers, tuple(attempts), True,
                )
            return SiftsRequestResult(
                source_id, url, f"http_{status}", response.body,
                _sha256_bytes(response.body), response.headers, tuple(attempts), False,
            )
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = exc.read() if getattr(exc, "fp", None) is not None else b""
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "queried_at_utc": started,
                    "outcome": f"http_{status}",
                    "error": str(exc),
                }
            )
            if status == 404:
                return SiftsRequestResult(
                    source_id, url, "http_404", body, _sha256_bytes(body), {},
                    tuple(attempts), False,
                )
            if status == 429 or 500 <= status <= 599:
                if attempt_idx < max_attempts:
                    sleep_fn(backoff_s * (2 ** (attempt_idx - 1)))
                    continue
                return SiftsRequestResult(
                    source_id, url, "network_failure", body, _sha256_bytes(body), {},
                    tuple(attempts), True,
                )
            return SiftsRequestResult(
                source_id, url, f"http_{status}", body, _sha256_bytes(body), {},
                tuple(attempts), False,
            )
        except (
            # IncompleteRead/BadStatusLine are HTTPExceptions, not OSErrors: a truncated
            # chunked response must be retried, not abort a multi-hour network stage.
            urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException,
        ) as exc:
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "queried_at_utc": started,
                    "outcome": "transport_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt_idx < max_attempts:
                sleep_fn(backoff_s * (2 ** (attempt_idx - 1)))
                continue
            return SiftsRequestResult(
                source_id, url, "network_failure", b"", None, {}, tuple(attempts), True,
            )

    raise AssertionError("retry loop terminated without a result")


def parse_sifts_mappings(payload_bytes: bytes, source_id: str) -> list[dict[str, Any]]:
    """Parse the persisted PDBe ``best_structures`` payload without ranking/truncation."""

    try:
        payload = json.loads(payload_bytes or b"{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SIFTS JSON for {source_id}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"SIFTS payload for {source_id} must be an object")
    raw_items = payload.get(source_id) or []
    if not isinstance(raw_items, list):
        raise ValueError(f"SIFTS payload entry for {source_id} must be a list")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item_idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        try:
            row = {
                "sifts_item_index": item_idx,
                "pdb_id": str(item["pdb_id"]).upper(),
                "chain_id": str(item["chain_id"]),
                "unp_start": int(item["unp_start"]),
                "unp_end": int(item["unp_end"]),
                "pdb_start": int(item["start"]),
                "pdb_end": int(item["end"]),
                "resolution": (
                    None if item.get("resolution") is None else float(item["resolution"])
                ),
                "experimental_method": str(item.get("experimental_method") or ""),
                "coverage": None if item.get("coverage") is None else float(item["coverage"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        key = (
            row["pdb_id"], row["chain_id"], row["unp_start"], row["unp_end"],
            row["pdb_start"], row["pdb_end"], row["resolution"],
            row["experimental_method"],
        )
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _is_xray(method: str) -> bool:
    value = method.casefold()
    return "x-ray" in value or "crystal" in value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate_sifts_mapping(
    *,
    source_id: str,
    mapping: Mapping[str, Any],
    full_sequence: str,
    spans: Sequence[Mapping[str, Any]],
    min_length: int = 100,
    max_length: int = 500,
    max_resolution: float = 2.5,
    min_spans: int = 2,
    min_epitope_coverage: float = 0.10,
    max_epitope_coverage: float = 0.50,
) -> dict[str, Any]:
    """Evaluate one SIFTS mapping in the frozen UniProt source frame."""

    method = str(mapping.get("experimental_method") or mapping.get("method") or "")
    resolution = mapping.get("resolution")
    try:
        unp_start = int(mapping["unp_start"])
        unp_end = int(mapping["unp_end"])
    except (KeyError, TypeError, ValueError):
        unp_start, unp_end = 0, -1
    mapped_length = unp_end - unp_start + 1

    method_ok = _is_xray(method)
    resolution_ok = _finite_number(resolution) and float(resolution) <= max_resolution
    length_ok = min_length <= mapped_length <= max_length
    sequence_ok = bool(full_sequence) and 1 <= unp_start <= unp_end <= len(full_sequence)

    verified: list[dict[str, Any]] = []
    mismatch_count = 0
    outside_count = 0
    if sequence_ok:
        range_start_0b = unp_start - 1
        range_end_0b = unp_end
        for span in spans:
            try:
                start = int(span["start_0b"])
                end = int(span["end_0b"])
            except (KeyError, TypeError, ValueError):
                mismatch_count += 1
                continue
            peptide = str(span.get("peptide") or span.get("peptide_seq") or "")
            if start < range_start_0b or end > range_end_0b or end <= start:
                outside_count += 1
                continue
            if not peptide or full_sequence[start:end] != peptide:
                mismatch_count += 1
                continue
            verified.append(
                {
                    "start_0b": start,
                    "end_0b": end,
                    "peptide": peptide,
                    "source": str(span.get("source") or "iedb"),
                }
            )

    unique_verified: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for span in verified:
        key = (span["start_0b"], span["end_0b"], span["peptide"])
        if key not in seen_spans:
            seen_spans.add(key)
            unique_verified.append(span)
    covered: set[int] = set()
    for span in unique_verified:
        covered.update(range(span["start_0b"], span["end_0b"]))
    epitope_coverage = (len(covered) / mapped_length) if mapped_length > 0 else 0.0
    spans_ok = len(unique_verified) >= min_spans
    epitope_coverage_ok = (
        min_epitope_coverage <= epitope_coverage <= max_epitope_coverage
    )

    failures: list[str] = []
    if not method_ok:
        failures.append("non_xray")
    if not resolution_ok:
        failures.append(
            "resolution_missing" if not _finite_number(resolution) else "resolution_above_2_5"
        )
    if not length_ok:
        failures.append("mapped_range_length_outside_100_500")
    if not sequence_ok:
        failures.append("mapped_range_outside_sequence")
    if not spans_ok:
        failures.append("fewer_than_2_verified_spans")
    if spans_ok and not epitope_coverage_ok:
        failures.append("source_epitope_coverage_outside_0_10_0_50")

    valid = not failures
    return {
        "source_id": source_id,
        "pdb_id": str(mapping.get("pdb_id") or "").upper(),
        "auth_chain_id": str(mapping.get("chain_id") or ""),
        "unp_start": unp_start,
        "unp_end": unp_end,
        "pdb_start": mapping.get("pdb_start"),
        "pdb_end": mapping.get("pdb_end"),
        "mapped_range_length": mapped_length,
        "resolution": None if resolution is None else float(resolution),
        "experimental_method": method,
        "sifts_coverage": mapping.get("coverage"),
        "method_xray_pass": method_ok,
        "resolution_pass": resolution_ok,
        "mapped_range_length_pass": length_ok,
        "mapped_range_sequence_pass": sequence_ok,
        "verified_spans_pass": spans_ok,
        "source_epitope_coverage_pass": epitope_coverage_ok,
        "verified_span_count": len(unique_verified),
        "verified_covered_residue_count": len(covered),
        "source_epitope_coverage": epitope_coverage,
        "span_outside_range_count": outside_count,
        "span_peptide_mismatch_count": mismatch_count,
        "verified_spans_json": json.dumps(
            unique_verified, sort_keys=True, separators=(",", ":")
        ),
        "failure_reasons_json": json.dumps(failures, separators=(",", ":")),
        "source_valid": valid,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _input_file_identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": (
            str(path.resolve().relative_to(Path(root).resolve()))
            if root is not None else str(path.resolve())
        ),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _safe_source_filename(source_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_id)
    return safe or hashlib.sha256(source_id.encode()).hexdigest()


def _source_stage_row(
    source_id: str,
    request_row: Mapping[str, Any],
    chain_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def any_stage(*columns: str) -> bool:
        return any(all(bool(row[col]) for col in columns) for row in chain_rows)

    has_mapping = bool(chain_rows)
    xray = any_stage("method_xray_pass")
    resolution = any_stage("method_xray_pass", "resolution_pass")
    length = any_stage(
        "method_xray_pass", "resolution_pass", "mapped_range_length_pass"
    )
    sequence = any_stage(
        "method_xray_pass", "resolution_pass", "mapped_range_length_pass",
        "mapped_range_sequence_pass",
    )
    spans = any_stage(
        "method_xray_pass", "resolution_pass", "mapped_range_length_pass",
        "mapped_range_sequence_pass", "verified_spans_pass",
    )
    valid = any(bool(row["source_valid"]) for row in chain_rows)

    final_status = str(request_row["final_status"])
    if final_status == "network_failure":
        terminal = "network_failure"
    elif final_status == "http_404":
        terminal = "http_404"
    elif final_status != "http_200":
        terminal = "http_error"
    elif not has_mapping:
        terminal = "no_mapping"
    elif not xray:
        terminal = "no_xray_mapping"
    elif not resolution:
        terminal = "no_xray_resolution_pass"
    elif not length:
        terminal = "no_range_length_pass"
    elif not sequence:
        terminal = "no_sequence_range_pass"
    elif not spans:
        terminal = "no_two_verified_spans"
    elif not valid:
        terminal = "no_source_coverage_band_pass"
    else:
        terminal = "source_frame_valid"

    failure_reasons = sorted(
        {
            reason
            for row in chain_rows
            for reason in json.loads(str(row["failure_reasons_json"]))
        }
    )
    return {
        "source_id": source_id,
        "request_final_status": final_status,
        "parsed_mapping_count": len(chain_rows),
        "with_parsed_sifts_mapping": has_mapping,
        "with_any_xray": xray,
        "with_any_xray_resolution": resolution,
        "with_any_xray_resolution_length": length,
        "with_any_xray_resolution_length_sequence": sequence,
        "with_any_xray_resolution_length_sequence_2_spans": spans,
        "source_frame_valid": valid,
        "source_terminal_status": terminal,
        "observed_chain_failure_reasons_json": json.dumps(
            failure_reasons, separators=(",", ":")
        ),
    }


def validate_preflight_raw_file_bindings(
    manifest: Mapping[str, Any], root: Path
) -> None:
    """Rehash every raw API body and its sorted file-table Merkle root."""

    root = Path(root).resolve()
    for field in ("uniprot_response_files", "sifts_response_files"):
        binding = manifest.get(field)
        if not isinstance(binding, Mapping) or not isinstance(binding.get("files"), list):
            raise ValueError(f"preflight manifest missing {field} file table")
        files = binding["files"]
        if files != sorted(files, key=lambda row: row["path"]):
            raise ValueError(f"{field} file table is not sorted")
        expected_merkle = _sha256_bytes(_canonical_json_bytes(files))
        if binding.get("merkle_sha256") != expected_merkle:
            raise ValueError(f"{field} Merkle mismatch")
        for entry in files:
            path = (root / str(entry["path"])).resolve()
            if root not in path.parents:
                raise ValueError(f"{field} path escapes preflight root: {entry['path']}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"{field} size mismatch: {entry['path']}")
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{field} SHA-256 mismatch: {entry['path']}")


def run_tier1_source_preflight(
    *,
    allele: str,
    test_ids_path: Path,
    protein_samples_path: Path,
    span_records_path: Path,
    output_dir: Path,
    tier1_target: int,
    request_fn: Callable[[str, float], SiftsHttpResponse] = _default_request,
    uniprot_transport: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_attempts: int = 4,
    timeout_s: float = 30.0,
    backoff_s: float = 1.0,
    api_delay_s: float = 0.1,
    command_argv: Sequence[str] | None = None,
    prior_failed_build_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a fresh content-bound Tier-1 capacity preflight directory.

    ``output_dir`` is creation-only. A shortfall is recorded in the returned summary rather than
    suppressing artifacts; the CLI converts it to a non-zero, fail-closed exit status.
    """

    test_ids_path = Path(test_ids_path)
    protein_samples_path = Path(protein_samples_path)
    span_records_path = Path(span_records_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"preflight output already exists: {output_dir}")
    if tier1_target < 1:
        raise ValueError("tier1_target must be positive")
    if api_delay_s < 0:
        raise ValueError("api_delay_s must be non-negative")

    for source in (test_ids_path, protein_samples_path, span_records_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir.mkdir(parents=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir()
    frozen_inputs = {
        "test_ids": input_dir / "source_ids.txt",
        "protein_samples": input_dir / "protein_samples.parquet",
        "span_records": input_dir / "span_records.parquet",
    }
    for source, destination in zip(
        (test_ids_path, protein_samples_path, span_records_path),
        frozen_inputs.values(), strict=True,
    ):
        shutil.copyfile(source, destination)
    test_ids_path = frozen_inputs["test_ids"]
    protein_samples_path = frozen_inputs["protein_samples"]
    span_records_path = frozen_inputs["span_records"]
    input_files = {
        label: _input_file_identity(path, root=output_dir)
        for label, path in frozen_inputs.items()
    }
    source_ids = [line.strip() for line in test_ids_path.read_text().splitlines() if line.strip()]
    if not source_ids:
        raise ValueError("Tier-1 source ID list is empty")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Tier-1 source ID list contains duplicates")

    proteins = pd.read_parquet(protein_samples_path)
    spans = pd.read_parquet(span_records_path)
    _require_columns(
        proteins, {"protein_id", "allele", "protein_seq", "sequence_length"},
        "protein_samples",
    )
    _require_columns(
        spans,
        {"protein_id", "allele", "start_0b", "end_0b", "peptide_seq", "source"},
        "span_records",
    )
    proteins = proteins[
        proteins["protein_id"].astype(str).isin(source_ids) & (proteins["allele"] == allele)
    ].copy()
    spans = spans[
        spans["protein_id"].astype(str).isin(source_ids) & (spans["allele"] == allele)
    ].copy()

    raw_sequence_by_id: dict[str, str] = {}
    for source_id, group in proteins.groupby("protein_id", sort=False):
        values = {str(value) for value in group["protein_seq"]}
        if len(values) != 1:
            raise ValueError(f"source {source_id} has conflicting sequences")
        sequence = values.pop()
        declared_lengths = {int(value) for value in group["sequence_length"]}
        if declared_lengths != {len(sequence)}:
            raise ValueError(f"source {source_id} has a sequence_length mismatch")
        raw_sequence_by_id[str(source_id)] = sequence
    missing_sequences = sorted(set(source_ids) - set(raw_sequence_by_id))
    if missing_sequences:
        raise ValueError(f"Tier-1 sources missing allele-matched sequences: {missing_sequences}")

    raw_spans_by_id: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in source_ids}
    for row in spans.itertuples(index=False):
        raw_spans_by_id[str(row.protein_id)].append(
            {
                "start_0b": int(row.start_0b),
                "end_0b": int(row.end_0b),
                "peptide": str(row.peptide_seq),
                "source": str(row.source),
            }
        )
    missing_spans = sorted(source_id for source_id in source_ids if not raw_spans_by_id[source_id])
    if missing_spans:
        raise ValueError(f"Tier-1 sources missing allele-matched spans: {missing_spans}")

    uniprot_kwargs: dict[str, Any] = {}
    if uniprot_transport is not None:
        uniprot_kwargs["transport"] = uniprot_transport
    uniprot_snapshot = snapshot_uniprot_identity(
        raw_source_ids=source_ids,
        output_dir=output_dir / "uniprot",
        sleep_fn=sleep_fn,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
        **uniprot_kwargs,
    )
    canonical_units, identity_ledger = resolve_canonical_source_units(
        raw_sequences=raw_sequence_by_id,
        raw_spans=raw_spans_by_id,
        external_mappings=uniprot_snapshot.external_mappings,
        canonical_sequences=uniprot_snapshot.canonical_sequences,
    )
    identity_ledger_df = pd.DataFrame(identity_ledger).sort_values(
        "raw_source_id", kind="stable"
    ).reset_index(drop=True)
    canonical_unit_rows = [
        {
            "source_id": source_id,
            "canonical_sequence": unit["canonical_sequence"],
            "canonical_sequence_length": unit["canonical_sequence_length"],
            "canonical_sequence_sha256": unit["canonical_sequence_sha256"],
            "raw_source_ids_json": json.dumps(unit["raw_source_ids"], separators=(",", ":")),
            "raw_source_id_count": unit["raw_source_id_count"],
            "span_count": unit["span_count"],
            "spans_json": json.dumps(unit["spans"], sort_keys=True, separators=(",", ":")),
        }
        for source_id, unit in sorted(canonical_units.items())
    ]
    canonical_units_df = pd.DataFrame(
        canonical_unit_rows,
        columns=[
            "source_id", "canonical_sequence", "canonical_sequence_length",
            "canonical_sequence_sha256", "raw_source_ids_json", "raw_source_id_count",
            "span_count", "spans_json",
        ],
    )

    response_dir = output_dir / "sifts" / "responses"
    response_dir.mkdir(parents=True)
    request_rows: list[dict[str, Any]] = []
    chain_attempts: list[dict[str, Any]] = []
    source_stage_rows: list[dict[str, Any]] = []
    used_response_names: set[str] = set()

    for source_id, unit in sorted(canonical_units.items()):
        result = request_sifts_with_retry(
            source_id,
            request_fn=request_fn,
            sleep_fn=sleep_fn,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            backoff_s=backoff_s,
        )
        if api_delay_s:
            sleep_fn(api_delay_s)
        basename = _safe_source_filename(source_id) + ".body"
        if basename in used_response_names:
            raise ValueError(f"SIFTS response filename collision for {source_id}")
        used_response_names.add(basename)
        response_path = response_dir / basename
        response_path.write_bytes(result.response_body)

        request_row = {
            "source_id": source_id,
            "url": result.url,
            "final_status": result.final_status,
            "network_failure": result.network_failure,
            "attempt_count": len(result.attempts),
            "attempts_json": json.dumps(
                result.attempts, sort_keys=True, separators=(",", ":")
            ),
            "response_path": str(response_path.relative_to(output_dir)),
            "response_size_bytes": len(result.response_body),
            "response_sha256": result.response_sha256,
            "response_headers_json": json.dumps(
                dict(result.response_headers), sort_keys=True, separators=(",", ":")
            ),
        }
        mappings: list[dict[str, Any]] = []
        if result.final_status == "http_200":
            try:
                mappings = parse_sifts_mappings(result.response_body, source_id)
                request_row["parse_status"] = "complete"
            except ValueError as exc:
                request_row["parse_status"] = "malformed_json"
                request_row["parse_error"] = str(exc)
        else:
            request_row["parse_status"] = "not_applicable"
        request_row.setdefault("parse_error", None)
        request_row["parsed_mapping_count"] = len(mappings)
        request_rows.append(request_row)

        current_attempts: list[dict[str, Any]] = []
        for mapping in mappings:
            evaluated = evaluate_sifts_mapping(
                source_id=source_id,
                mapping=mapping,
                full_sequence=unit["canonical_sequence"],
                spans=unit["spans"],
            )
            evaluated["sifts_item_index"] = mapping["sifts_item_index"]
            evaluated["source_sequence_sha256"] = _sha256_bytes(
                unit["canonical_sequence"].encode()
            )
            current_attempts.append(evaluated)
            chain_attempts.append(evaluated)
        source_stage_rows.append(_source_stage_row(source_id, request_row, current_attempts))

    requests_df = pd.DataFrame(request_rows, columns=SIFTS_REQUEST_COLUMNS)
    if len(requests_df):
        requests_df = requests_df.sort_values("source_id", kind="stable").reset_index(drop=True)
    attempts_df = pd.DataFrame(chain_attempts, columns=CHAIN_ATTEMPT_COLUMNS)
    if len(attempts_df):
        attempts_df = attempts_df.sort_values(
            ["source_id", "sifts_item_index"], kind="stable"
        ).reset_index(drop=True)
    stages_df = pd.DataFrame(source_stage_rows, columns=SOURCE_STAGE_COLUMNS)
    if len(stages_df):
        stages_df = stages_df.sort_values("source_id", kind="stable").reset_index(drop=True)

    valid_unit_rows: list[dict[str, Any]] = []
    if len(attempts_df):
        valid_counts = (
            attempts_df[attempts_df["source_valid"]]
            .groupby("source_id", sort=True).size().to_dict()
        )
        for source_id, count in sorted(valid_counts.items()):
            unit = canonical_units[str(source_id)]
            valid_unit_rows.append(
                {
                    "source_id": source_id,
                    "canonical_sequence_length": unit["canonical_sequence_length"],
                    "canonical_sequence_sha256": unit["canonical_sequence_sha256"],
                    "raw_source_ids_json": json.dumps(
                        unit["raw_source_ids"], separators=(",", ":")
                    ),
                    "viable_sifts_mapping_count": int(count),
                }
            )
    valid_units_df = pd.DataFrame(valid_unit_rows, columns=VALID_SOURCE_UNIT_COLUMNS)

    funnel = {
        "input_split_ids": len(source_ids),
        "resolved_raw_ids": int((identity_ledger_df["resolution_status"] == "resolved").sum()),
        "identity_excluded_raw_ids": int((identity_ledger_df["resolution_status"] != "resolved").sum()),
        "canonical_source_units": len(canonical_units),
        "input_source_units": len(canonical_units),
        "http_200": int((requests_df["final_status"] == "http_200").sum()) if len(requests_df) else 0,
        "http_404": int((requests_df["final_status"] == "http_404").sum()) if len(requests_df) else 0,
        "network_failure": int(requests_df["network_failure"].sum()) if len(requests_df) else 0,
        "malformed_response": int((requests_df["parse_status"] == "malformed_json").sum()) if len(requests_df) else 0,
        "with_parsed_sifts_mapping": int(stages_df["with_parsed_sifts_mapping"].sum()) if len(stages_df) else 0,
        "with_any_xray": int(stages_df["with_any_xray"].sum()) if len(stages_df) else 0,
        "with_any_xray_resolution": int(stages_df["with_any_xray_resolution"].sum()) if len(stages_df) else 0,
        "with_any_xray_resolution_length": int(
            stages_df["with_any_xray_resolution_length"].sum() if len(stages_df) else 0
        ),
        "with_any_xray_resolution_length_sequence": int(
            stages_df["with_any_xray_resolution_length_sequence"].sum() if len(stages_df) else 0
        ),
        "with_any_xray_resolution_length_sequence_2_spans": int(
            stages_df["with_any_xray_resolution_length_sequence_2_spans"].sum() if len(stages_df) else 0
        ),
        "source_frame_valid": int(stages_df["source_frame_valid"].sum()) if len(stages_df) else 0,
    }
    terminal_counts = {
        str(key): int(value)
        for key, value in (
            stages_df["source_terminal_status"].value_counts().sort_index().items()
            if len(stages_df) else []
        )
    }
    chain_failure_counts: dict[str, int] = {}
    for value in attempts_df.get("failure_reasons_json", pd.Series(dtype=str)):
        for reason in json.loads(str(value)):
            chain_failure_counts[reason] = chain_failure_counts.get(reason, 0) + 1
    chain_failure_counts = dict(sorted(chain_failure_counts.items()))
    identity_status_counts = {
        str(key): int(value)
        for key, value in identity_ledger_df["resolution_status"].value_counts().sort_index().items()
    }
    release_capacity_pass = (
        funnel["network_failure"] == 0
        and funnel["malformed_response"] == 0
        and all(status in {"http_200", "http_404"} for status in requests_df["final_status"])
        and funnel["source_frame_valid"] >= tier1_target
    )

    pd.DataFrame(uniprot_snapshot.requests).to_parquet(
        output_dir / "uniprot_requests.parquet", index=False
    )
    identity_ledger_df.to_parquet(
        output_dir / "tier1_accession_resolution_ledger.parquet", index=False
    )
    canonical_units_df.to_parquet(
        output_dir / "tier1_canonical_source_units.parquet", index=False
    )
    requests_df.to_parquet(output_dir / "sifts_requests.parquet", index=False)
    attempts_df.to_parquet(output_dir / "tier1_sifts_chain_attempts.parquet", index=False)
    stages_df.to_parquet(output_dir / "tier1_source_funnel.parquet", index=False)
    valid_units_df.to_parquet(
        output_dir / "tier1_source_frame_valid_units.parquet", index=False
    )
    funnel_payload = {
        "allele": allele,
        "tier1_target": tier1_target,
        "release_capacity_pass": release_capacity_pass,
        "funnel": funnel,
        "source_terminal_status_counts": terminal_counts,
        "identity_resolution_status_counts": identity_status_counts,
        "chain_failure_reason_counts": chain_failure_counts,
    }
    (output_dir / "preflight_funnel.json").write_text(
        json.dumps(funnel_payload, indent=2, sort_keys=True) + "\n"
    )

    output_names = [
        "uniprot_requests.parquet",
        "tier1_accession_resolution_ledger.parquet",
        "tier1_canonical_source_units.parquet",
        "sifts_requests.parquet",
        "tier1_sifts_chain_attempts.parquet",
        "tier1_source_funnel.parquet",
        "tier1_source_frame_valid_units.parquet",
        "preflight_funnel.json",
    ]
    output_files = {
        name: {
            "path": name,
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": _sha256_file(output_dir / name),
        }
        for name in output_names
    }
    sifts_response_files = []
    for row in requests_df.itertuples(index=False):
        response_path = output_dir / str(row.response_path)
        sifts_response_files.append(
            {
                "path": str(row.response_path),
                "size_bytes": response_path.stat().st_size,
                "sha256": _sha256_file(response_path),
            }
        )
    sifts_response_files.sort(key=lambda row: row["path"])
    sifts_response_binding = {
        "files": sifts_response_files,
        "merkle_sha256": _sha256_bytes(_canonical_json_bytes(sifts_response_files)),
    }
    uniprot_response_files = [
        {**entry, "path": f"uniprot/{entry['path']}"}
        for entry in uniprot_snapshot.raw_files
    ]
    uniprot_response_files.sort(key=lambda row: row["path"])
    uniprot_response_binding = {
        "files": uniprot_response_files,
        "merkle_sha256": _sha256_bytes(_canonical_json_bytes(uniprot_response_files)),
    }
    manifest: dict[str, Any] = {
        "schema_version": "if-benchmark-v3-tier1-preflight/2",
        "protocol_id": "if-benchmark-test-set/3",
        "created_at_utc": _utc_now(),
        "allele": allele,
        "tier1_target": tier1_target,
        "command_argv": list(command_argv or []),
        "prior_failed_build_ids": list(prior_failed_build_ids),
        "input_files": input_files,
        "source_identity": {
            "unit": "allele_plus_canonical_uniprot",
            "canonical_sequence_authority": "UniProtKB REST snapshot",
            "accession_resolution_ledger": "tier1_accession_resolution_ledger.parquet",
        },
        "uniprot_response_files": uniprot_response_binding,
        "sifts_response_files": sifts_response_binding,
        "api": {
            "endpoint_template": SIFTS_BEST_STRUCTURES_URL,
            "max_attempts": max_attempts,
            "timeout_s": timeout_s,
            "backoff_s": backoff_s,
            "api_delay_s": api_delay_s,
        },
        "frozen_filters": {
            "method": "X-ray",
            "max_resolution_angstrom": 2.5,
            "mapped_range_length": [100, 500],
            "minimum_distinct_verified_spans": 2,
            "source_epitope_coverage_inclusive": [0.10, 0.50],
        },
        "funnel": funnel,
        "source_terminal_status_counts": terminal_counts,
        "identity_resolution_status_counts": identity_status_counts,
        "chain_failure_reason_counts": chain_failure_counts,
        "release_capacity_pass": release_capacity_pass,
        "output_files": output_files,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    (output_dir / "preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_preflight_raw_file_bindings(manifest, output_dir)
    return funnel_payload
