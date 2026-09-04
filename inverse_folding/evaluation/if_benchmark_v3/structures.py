"""Content-bound RCSB mmCIF cache acquisition and revision revalidation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import http.client
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .preflight import SiftsHttpResponse


RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
STRUCTURE_DOWNLOAD_KEY_COLUMNS = [
    "source_tier", "selection_unit_id", "rcsb_entity_id", "label_asym_id",
    "auth_chain_id",
]
STRUCTURE_DOWNLOAD_COLUMNS = [
    *STRUCTURE_DOWNLOAD_KEY_COLUMNS, "attempt_status", "source_revision_date",
    "evidence_kind", "coordinate_cif_path", "coordinate_manifest_path",
    "request_failure_path", "download_sha256", "evidence_sha256",
]


class RcsbMmcifAcquisitionError(RuntimeError):
    """Terminal content-bound acquisition failure with a persisted request ledger."""

    def __init__(self, message: str, *, ledger_path: Path):
        super().__init__(message)
        self.ledger_path = Path(ledger_path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _write_failure_ledger(
    *, revision_dir: Path, pdb_id: str, expected_revision_date: str, url: str,
    attempts: list[dict[str, Any]], terminal_http_status: int | None,
    response: SiftsHttpResponse | None, failure_reason: str, api_delay_s: float,
) -> Path:
    body_identity = None
    if response is not None and response.body:
        body_sha = hashlib.sha256(bytes(response.body)).hexdigest()
        body_path = revision_dir / f"failure_response_{body_sha}.bin"
        body_path.write_bytes(bytes(response.body))
        body_identity = {
            "filename": body_path.name, "size_bytes": body_path.stat().st_size,
            "sha256": body_sha,
        }
    payload = {
        "schema_version": "if-benchmark-v3-rcsb-mmcif-failure/1",
        "pdb_id": pdb_id, "expected_revision_date": expected_revision_date,
        "url": url, "attempts": attempts, "terminal_http_status": terminal_http_status,
        "failure_reason": failure_reason, "response_body": body_identity,
        "api_delay_s": float(api_delay_s),
    }
    payload["ledger_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    ledger_path = revision_dir / "request_failure.json"
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ledger_path


def validate_rcsb_failure_ledger(
    path: Path, *, pdb_id: str, expected_revision_date: str,
) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("ledger_sha256", None)
    if (
        payload.get("schema_version") != "if-benchmark-v3-rcsb-mmcif-failure/1"
        or digest != hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        or payload.get("pdb_id") != str(pdb_id).upper()
        or payload.get("expected_revision_date") != expected_revision_date
        or not isinstance(payload.get("attempts"), list) or not payload["attempts"]
        or not str(payload.get("failure_reason") or "")
        or not math.isfinite(float(payload.get("api_delay_s", -1.0)))
        or float(payload.get("api_delay_s", -1.0)) < 0.0
    ):
        raise ValueError("RCSB mmCIF failure ledger identity mismatch")
    body = payload.get("response_body")
    if body is not None:
        body_path = path.parent / str(body.get("filename") or "")
        if (
            not body_path.is_file() or body_path.parent != path.parent
            or body_path.stat().st_size != int(body["size_bytes"])
            or _sha(body_path) != body["sha256"]
        ):
            raise ValueError("RCSB mmCIF failure response body identity mismatch")
    return payload


def _inside_candidate(path: Path, *, candidate_root: Path) -> tuple[Path, str]:
    root = Path(candidate_root).resolve()
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("structure download evidence escapes candidate root")
    return resolved, str(resolved.relative_to(root))


def _attempt_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in STRUCTURE_DOWNLOAD_KEY_COLUMNS)


def write_structure_download_registry(
    path: Path, *, candidate_root: Path, chain_attempts: pd.DataFrame,
    evidence_rows: list[Mapping[str, Any]],
) -> Path:
    """Bind each C6 attempt to immutable source-CIF bytes or a failed request ledger."""

    required = set(STRUCTURE_DOWNLOAD_KEY_COLUMNS) | {
        "attempt_status", "source_revision_date", "download_sha256",
    }
    if not required <= set(chain_attempts):
        raise ValueError("chain attempt table lacks structure-download identity columns")
    if chain_attempts.duplicated(STRUCTURE_DOWNLOAD_KEY_COLUMNS).any():
        raise ValueError("chain attempts have duplicate structure-download keys")
    evidence_by_key: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for evidence in evidence_rows:
        key = _attempt_key(evidence)
        if key in evidence_by_key:
            raise ValueError("duplicate structure-download evidence key")
        evidence_by_key[key] = evidence
    expected_keys = {_attempt_key(row) for row in chain_attempts.to_dict("records")}
    if set(evidence_by_key) != expected_keys:
        raise ValueError("structure-download evidence does not exactly cover chain attempts")
    normalized: list[dict[str, Any]] = []
    for attempt in chain_attempts.to_dict("records"):
        key = _attempt_key(attempt)
        evidence = evidence_by_key[key]
        status = str(attempt["attempt_status"])
        revision = str(attempt["source_revision_date"] or "")
        pdb_id = str(attempt["rcsb_entity_id"]).split("_", 1)[0].upper()
        if not revision:
            raise ValueError("structure attempt lacks frozen source revision")
        base = {
            **{column: str(attempt[column]) for column in STRUCTURE_DOWNLOAD_KEY_COLUMNS},
            "attempt_status": status, "source_revision_date": revision,
            "coordinate_cif_path": None, "coordinate_manifest_path": None,
            "request_failure_path": None,
        }
        if status == "download_failure":
            failure_path, failure_relative = _inside_candidate(
                Path(evidence["request_failure_path"]), candidate_root=candidate_root
            )
            validate_rcsb_failure_ledger(
                failure_path, pdb_id=pdb_id, expected_revision_date=revision
            )
            if not pd.isna(attempt["download_sha256"]):
                raise ValueError("download-failure attempt unexpectedly has source CIF SHA")
            normalized.append({
                **base, "evidence_kind": "request_failure",
                "request_failure_path": failure_relative, "download_sha256": None,
                "evidence_sha256": _sha(failure_path),
            })
            continue
        cif_path, cif_relative = _inside_candidate(
            Path(evidence["coordinate_cif_path"]), candidate_root=candidate_root
        )
        manifest_path, manifest_relative = _inside_candidate(
            Path(evidence["coordinate_manifest_path"]), candidate_root=candidate_root
        )
        cache_manifest = json.loads(manifest_path.read_text())
        validate_cached_rcsb_mmcif(
            cif_path=cif_path, manifest_path=manifest_path, pdb_id=pdb_id,
            expected_revision_date=str(cache_manifest["expected_revision_date"]),
        )
        cif_sha = _sha(cif_path)
        if (
            str(cache_manifest["observed_revision_date"]) != _validated_revision_date(revision)
            or str(attempt["download_sha256"]) != cif_sha
        ):
            raise ValueError("chain attempt revision/download SHA differs from source cache")
        normalized.append({
            **base, "evidence_kind": "coordinate_cif",
            "coordinate_cif_path": cif_relative,
            "coordinate_manifest_path": manifest_relative,
            "download_sha256": cif_sha, "evidence_sha256": _sha(manifest_path),
        })
    frame = pd.DataFrame(normalized, columns=STRUCTURE_DOWNLOAD_COLUMNS).sort_values(
        STRUCTURE_DOWNLOAD_KEY_COLUMNS, kind="stable"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def validate_structure_download_registry(
    path: Path, *, candidate_root: Path, chain_attempt_path: Path,
) -> None:
    registry = pd.read_parquet(path)
    attempts = pd.read_parquet(chain_attempt_path)
    if list(registry.columns) != STRUCTURE_DOWNLOAD_COLUMNS:
        raise ValueError("structure download registry schema mismatch")
    if registry.duplicated(STRUCTURE_DOWNLOAD_KEY_COLUMNS).any():
        raise ValueError("structure download registry has duplicate attempt keys")
    if {_attempt_key(row) for row in registry.to_dict("records")} != {
        _attempt_key(row) for row in attempts.to_dict("records")
    }:
        raise ValueError("structure download registry does not exactly cover chain attempts")
    attempts_by_key = {_attempt_key(row): row for row in attempts.to_dict("records")}
    for evidence in registry.to_dict("records"):
        attempt = attempts_by_key[_attempt_key(evidence)]
        status = str(attempt["attempt_status"])
        revision = str(attempt["source_revision_date"] or "")
        pdb_id = str(attempt["rcsb_entity_id"]).split("_", 1)[0].upper()
        if (
            str(evidence["attempt_status"]) != status
            or str(evidence["source_revision_date"]) != revision
        ):
            raise ValueError("structure download registry/attempt identity mismatch")
        if status == "download_failure":
            if evidence["evidence_kind"] != "request_failure":
                raise ValueError("download failure lacks request-ledger evidence")
            failure_path, _relative = _inside_candidate(
                Path(str(evidence["request_failure_path"])), candidate_root=candidate_root
            )
            validate_rcsb_failure_ledger(
                failure_path, pdb_id=pdb_id, expected_revision_date=revision
            )
            if (
                _sha(failure_path) != str(evidence["evidence_sha256"])
                or not pd.isna(evidence["download_sha256"])
                or not pd.isna(attempt["download_sha256"])
            ):
                raise ValueError("failed structure request evidence digest mismatch")
            continue
        if evidence["evidence_kind"] != "coordinate_cif":
            raise ValueError("successful structure attempt lacks coordinate-CIF evidence")
        cif_path, _relative = _inside_candidate(
            Path(str(evidence["coordinate_cif_path"])), candidate_root=candidate_root
        )
        manifest_path, _manifest_relative = _inside_candidate(
            Path(str(evidence["coordinate_manifest_path"])), candidate_root=candidate_root
        )
        manifest = json.loads(manifest_path.read_text())
        validate_cached_rcsb_mmcif(
            cif_path=cif_path, manifest_path=manifest_path, pdb_id=pdb_id,
            expected_revision_date=str(manifest["expected_revision_date"]),
        )
        cif_sha = _sha(cif_path)
        if (
            str(manifest["observed_revision_date"]) != _validated_revision_date(revision)
            or cif_sha != str(attempt["download_sha256"])
            or cif_sha != str(evidence["download_sha256"])
            or _sha(manifest_path) != str(evidence["evidence_sha256"])
        ):
            raise ValueError("structure download SHA/revision registry mismatch")


def _validated_revision_date(value: str) -> str:
    raw = str(value or "")
    expected_date = raw.split("T", 1)[0]
    try:
        parsed = datetime.strptime(expected_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("invalid RCSB PDB/cache request revision date") from exc
    if parsed.strftime("%Y-%m-%d") != expected_date:
        raise ValueError("invalid RCSB PDB/cache request revision date")
    return expected_date


def _default_get(url: str, timeout_s: float) -> SiftsHttpResponse:
    request = urllib.request.Request(url, headers={"Accept": "chemical/x-mmcif"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return SiftsHttpResponse(
            int(response.status), response.read(),
            {str(key).lower(): str(value) for key, value in response.headers.items()},
        )


def _mmcif_identity(path: Path) -> tuple[str, str | None]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    payload = MMCIF2Dict(str(path))
    raw_entry = payload.get("_entry.id")
    entry = str(raw_entry[0] if isinstance(raw_entry, list) else raw_entry or "").upper()
    raw_revisions = payload.get("_pdbx_audit_revision_history.revision_date")
    if raw_revisions is None:
        revision = None
    else:
        values = raw_revisions if isinstance(raw_revisions, list) else [raw_revisions]
        revision = max(str(value) for value in values)
    return entry, revision


def validate_cached_rcsb_mmcif(
    *, cif_path: Path, manifest_path: Path, pdb_id: str, expected_revision_date: str,
) -> dict[str, Any]:
    expected_date = _validated_revision_date(expected_revision_date)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != "if-benchmark-v3-rcsb-mmcif-cache/1":
        raise ValueError("RCSB mmCIF cache manifest schema mismatch")
    if (
        manifest.get("pdb_id") != pdb_id.upper()
        or manifest.get("expected_revision_date") != expected_revision_date
        or not math.isfinite(float(manifest.get("api_delay_s", -1.0)))
        or float(manifest.get("api_delay_s", -1.0)) < 0.0
    ):
        raise ValueError("RCSB mmCIF cache source identity mismatch")
    cif_path = Path(cif_path)
    if (
        manifest.get("cif_filename") != cif_path.name
        or cif_path.parent.name != expected_date
        or cif_path.parent.parent.name != pdb_id.upper()
        or not cif_path.is_file() or cif_path.stat().st_size != int(manifest["size_bytes"])
        or _sha(cif_path) != manifest["sha256"]
    ):
        raise ValueError("RCSB mmCIF cache bytes changed")
    entry, observed_revision = _mmcif_identity(cif_path)
    if entry != pdb_id.upper():
        raise ValueError("RCSB mmCIF _entry.id mismatch")
    if observed_revision != expected_date:
        raise ValueError("RCSB mmCIF revision differs from GraphQL source snapshot")
    return manifest


def acquire_rcsb_mmcif(
    *, pdb_id: str, expected_revision_date: str, cache_dir: Path,
    request_fn: Callable[[str, float], SiftsHttpResponse] = _default_get,
    sleep_fn: Callable[[float], None] = time.sleep, timeout_s: float = 60.0,
    max_attempts: int = 4, api_delay_s: float = 0.1,
) -> tuple[Path, dict[str, Any]]:
    """Reuse only an exactly validated cache entry; otherwise create a fresh immutable entry."""

    pdb_id = str(pdb_id).upper()
    if (
        len(pdb_id) != 4 or not pdb_id.isalnum() or max_attempts < 1
        or not math.isfinite(float(api_delay_s)) or float(api_delay_s) < 0.0
    ):
        raise ValueError("invalid RCSB PDB/cache request")
    try:
        expected_date = _validated_revision_date(expected_revision_date)
    except ValueError as exc:
        raise ValueError("invalid RCSB PDB/cache request") from exc
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    revision_dir = cache_dir / pdb_id / expected_date
    manifest_path = revision_dir / "manifest.json"
    if revision_dir.exists():
        if not revision_dir.is_dir() or not manifest_path.is_file():
            raise ValueError("partial/legacy RCSB cache entry cannot be reused")
        cached_manifest = json.loads(manifest_path.read_text())
        cif_filename = str(cached_manifest.get("cif_filename") or "")
        cif_path = revision_dir / cif_filename
        return cif_path, validate_cached_rcsb_mmcif(
            cif_path=cif_path, manifest_path=manifest_path, pdb_id=pdb_id,
            expected_revision_date=expected_revision_date,
        )
    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    revision_dir.mkdir(parents=True)
    attempts: list[dict[str, Any]] = []
    response: SiftsHttpResponse | None = None
    failure_reason: str | None = None
    terminal_http_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        queried = _utc()
        try:
            candidate = request_fn(url, timeout_s)
            if api_delay_s > 0.0:
                sleep_fn(float(api_delay_s))
            status = int(candidate.status)
            attempts.append({"attempt": attempt, "queried_at_utc": queried,
                             "http_status": status, "error": None})
            if status == 429 or 500 <= status <= 599:
                if attempt < max_attempts:
                    sleep_fn(float(2 ** (attempt - 1)))
                    continue
                response = candidate
                terminal_http_status = status
                failure_reason = f"transient_http_exhausted:{status}"
                break
            if status != 200:
                response = candidate
                terminal_http_status = status
                failure_reason = f"terminal_http:{status}"
                break
            response = candidate
            break
        except (
            # IncompleteRead/BadStatusLine are HTTPExceptions, not OSErrors: a truncated
            # chunked response must be retried, not abort a multi-hour network stage.
            urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException,
        ) as exc:
            attempts.append({"attempt": attempt, "queried_at_utc": queried,
                             "http_status": None, "error": f"{type(exc).__name__}: {exc}"})
            if attempt < max_attempts:
                sleep_fn(float(2 ** (attempt - 1)))
                continue
            failure_reason = f"transport_exhausted:{type(exc).__name__}"
            break
    if failure_reason is not None or response is None:
        failure_reason = failure_reason or "no_response"
        ledger_path = _write_failure_ledger(
            revision_dir=revision_dir, pdb_id=pdb_id,
            expected_revision_date=expected_revision_date, url=url, attempts=attempts,
            terminal_http_status=terminal_http_status, response=response,
            failure_reason=failure_reason,
            api_delay_s=api_delay_s,
        )
        raise RcsbMmcifAcquisitionError(
            f"RCSB mmCIF acquisition failed: {failure_reason}", ledger_path=ledger_path,
        )
    temporary = revision_dir / f".{pdb_id}.cif.tmp"
    temporary.write_bytes(bytes(response.body))
    entry, observed_revision = _mmcif_identity(temporary)
    if entry != pdb_id or observed_revision != expected_date:
        rejected_sha = _sha(temporary)
        rejected_path = revision_dir / f"rejected_{rejected_sha}.cif"
        os.replace(temporary, rejected_path)
        ledger_path = _write_failure_ledger(
            revision_dir=revision_dir, pdb_id=pdb_id,
            expected_revision_date=expected_revision_date, url=url, attempts=attempts,
            terminal_http_status=200, response=None,
            failure_reason=(
                f"entry_or_revision_mismatch:entry={entry}:revision={observed_revision}"
            ),
            api_delay_s=api_delay_s,
        )
        raise RcsbMmcifAcquisitionError(
            "downloaded RCSB mmCIF entry/revision identity mismatch",
            ledger_path=ledger_path,
        )
    content_sha = _sha(temporary)
    cif_path = revision_dir / f"{content_sha}.cif"
    if cif_path.exists():
        temporary.unlink(missing_ok=True)
        raise ValueError("content-addressed RCSB cache target already exists without manifest")
    os.replace(temporary, cif_path)
    manifest = {
        "schema_version": "if-benchmark-v3-rcsb-mmcif-cache/1",
        "pdb_id": pdb_id, "expected_revision_date": expected_revision_date,
        "observed_revision_date": observed_revision, "url": url,
        "response_headers": dict(response.headers), "attempts": attempts,
        "api_delay_s": float(api_delay_s),
        "cif_filename": cif_path.name,
        "size_bytes": cif_path.stat().st_size, "sha256": content_sha,
    }
    temporary_manifest = revision_dir / ".manifest.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return cif_path, manifest
