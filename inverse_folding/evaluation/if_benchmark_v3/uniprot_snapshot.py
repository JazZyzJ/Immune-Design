"""Freeze accession mappings and canonical UniProt sequences for Tier-1 sources."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .source_identity import classify_source_accession


UNIPROT_API = "https://rest.uniprot.org"


@dataclass(frozen=True)
class _HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


@dataclass(frozen=True)
class UniProtIdentitySnapshot:
    external_mappings: dict[str, list[str]]
    canonical_sequences: dict[str, str]
    requests: tuple[dict[str, Any], ...]
    raw_files: tuple[dict[str, Any], ...]
    raw_file_merkle_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _default_transport(
    method: str,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout_s: float,
) -> _HttpResponse:
    request = urllib.request.Request(url, data=(body or None), headers=dict(headers), method=method)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return _HttpResponse(
            int(response.status), response.read(),
            {str(k).lower(): str(v) for k, v in response.headers.items()},
        )


def _write_raw(
    root: Path,
    relative: str,
    body: bytes,
    raw_files: list[dict[str, Any]],
) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    entry = {"path": relative, "size_bytes": len(body), "sha256": _sha(body)}
    raw_files.append(entry)
    return entry


def _audited_request(
    *,
    method: str,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    relative_stem: str,
    output_dir: Path,
    requests: list[dict[str, Any]],
    raw_files: list[dict[str, Any]],
    transport: Callable[[str, str, bytes, Mapping[str, str], float], Any],
    sleep_fn: Callable[[float], None],
    timeout_s: float,
    max_attempts: int,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        queried_at = _utc_now()
        try:
            response = transport(method, url, body, headers, timeout_s)
            raw = _write_raw(
                output_dir, f"raw/{relative_stem}.attempt{attempt:02d}.body",
                bytes(response.body), raw_files,
            )
            status = int(response.status)
            requests.append(
                {
                    "method": method,
                    "url": url,
                    "request_body_sha256": _sha(body),
                    "queried_at_utc": queried_at,
                    "attempt": attempt,
                    "http_status": status,
                    "response_path": raw["path"],
                    "response_sha256": raw["sha256"],
                    "response_headers_json": json.dumps(
                        dict(response.headers), sort_keys=True, separators=(",", ":")
                    ),
                    "error": None,
                }
            )
            if status == 429 or 500 <= status <= 599:
                if attempt < max_attempts:
                    sleep_fn(float(2 ** (attempt - 1)))
                    continue
                raise RuntimeError(f"UniProt request exhausted transient HTTP {status}: {url}")
            if not 200 <= status < 300:
                raise RuntimeError(f"UniProt request failed HTTP {status}: {url}")
            return response
        except (
            # IncompleteRead/BadStatusLine are HTTPExceptions, not OSErrors: a truncated
            # chunked response must be retried, not abort a multi-hour network stage.
            urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException,
        ) as exc:
            requests.append(
                {
                    "method": method,
                    "url": url,
                    "request_body_sha256": _sha(body),
                    "queried_at_utc": queried_at,
                    "attempt": attempt,
                    "http_status": None,
                    "response_path": None,
                    "response_sha256": None,
                    "response_headers_json": "{}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt < max_attempts:
                sleep_fn(float(2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"UniProt request exhausted transport retries: {url}") from exc
    raise AssertionError("request retry loop terminated unexpectedly")


def _parse_mapping_tsv(body: bytes) -> tuple[dict[str, list[str]], set[str]]:
    reader = csv.DictReader(io.StringIO(body.decode()), delimiter="\t")
    if not reader.fieldnames or not {"From", "Entry"} <= set(reader.fieldnames):
        raise ValueError("UniProt mapping TSV lacks From/Entry columns")
    mappings: dict[str, set[str]] = {}
    targets: set[str] = set()
    for row in reader:
        source = str(row.get("From") or "")
        target = str(row.get("Entry") or "")
        if not source or not target:
            continue
        mappings.setdefault(source, set()).add(target)
        targets.add(target)
    return {key: sorted(value) for key, value in mappings.items()}, targets


def _parse_sequence_tsv(body: bytes) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(body.decode()), delimiter="\t")
    if not reader.fieldnames or not {"Entry", "Sequence", "Length"} <= set(reader.fieldnames):
        raise ValueError("UniProt sequence TSV lacks Entry/Sequence/Length columns")
    out: dict[str, str] = {}
    for row in reader:
        accession = str(row.get("Entry") or "")
        sequence = str(row.get("Sequence") or "").replace(" ", "").upper()
        if not accession or not sequence:
            continue
        if int(row["Length"]) != len(sequence):
            raise ValueError(f"UniProt sequence length mismatch for {accession}")
        previous = out.get(accession)
        if previous is not None and previous != sequence:
            raise ValueError(f"UniProt returned conflicting sequences for {accession}")
        out[accession] = sequence
    return out


def snapshot_uniprot_identity(
    *,
    raw_source_ids: Sequence[str],
    output_dir: Path,
    transport: Callable[[str, str, bytes, Mapping[str, str], float], Any] = _default_transport,
    sleep_fn: Callable[[float], None] = time.sleep,
    timeout_s: float = 30.0,
    max_attempts: int = 4,
    poll_attempts: int = 30,
    poll_interval_s: float = 2.0,
    sequence_batch_size: int = 40,
) -> UniProtIdentitySnapshot:
    """Snapshot external accession mappings and all referenced canonical sequences."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(raw_source_ids) != len(set(raw_source_ids)):
        raise ValueError("raw_source_ids contains duplicates")
    if sequence_batch_size < 1:
        raise ValueError("sequence_batch_size must be positive")
    requests: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []

    _audited_request(
        method="GET", url=f"{UNIPROT_API}/configure/idmapping/fields", body=b"",
        headers={"Accept": "application/json"}, relative_stem="configure_idmapping_fields",
        output_dir=output_dir, requests=requests, raw_files=raw_files, transport=transport,
        sleep_fn=sleep_fn, timeout_s=timeout_s, max_attempts=max_attempts,
    )

    direct: set[str] = set()
    external_groups: dict[str, list[str]] = {"RefSeq_Protein": [], "EMBL-GenBank-DDBJ_CDS": []}
    for raw_id in raw_source_ids:
        kind, canonical = classify_source_accession(raw_id)
        if kind == "uniprot" and canonical:
            direct.add(canonical)
        elif kind == "refseq":
            external_groups["RefSeq_Protein"].append(raw_id)
        elif kind == "genbank_cds":
            external_groups["EMBL-GenBank-DDBJ_CDS"].append(raw_id)

    external_mappings: dict[str, list[str]] = {}
    external_targets: set[str] = set()
    for from_db, source_ids in external_groups.items():
        if not source_ids:
            continue
        source_ids = sorted(source_ids)
        body = urllib.parse.urlencode(
            {"from": from_db, "to": "UniProtKB", "ids": ",".join(source_ids)}
        ).encode()
        stem = "idmapping_" + from_db.lower().replace("-", "_")
        submitted = _audited_request(
            method="POST", url=f"{UNIPROT_API}/idmapping/run", body=body,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            relative_stem=f"{stem}/submit", output_dir=output_dir, requests=requests,
            raw_files=raw_files, transport=transport, sleep_fn=sleep_fn, timeout_s=timeout_s,
            max_attempts=max_attempts,
        )
        try:
            job_id = str(json.loads(bytes(submitted.body))["jobId"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid UniProt ID-mapping submission response for {from_db}") from exc

        finished = False
        for poll_idx in range(1, poll_attempts + 1):
            status = _audited_request(
                method="GET", url=f"{UNIPROT_API}/idmapping/status/{job_id}", body=b"",
                headers={"Accept": "application/json"},
                relative_stem=f"{stem}/status_{poll_idx:02d}", output_dir=output_dir,
                requests=requests, raw_files=raw_files, transport=transport,
                sleep_fn=sleep_fn, timeout_s=timeout_s, max_attempts=max_attempts,
            )
            try:
                payload = json.loads(bytes(status.body))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid UniProt mapping status JSON for {from_db}") from exc
            if payload.get("jobStatus") == "RUNNING":
                sleep_fn(poll_interval_s)
                continue
            if payload.get("jobStatus") in {"NEW", "UNDEFINED"}:
                sleep_fn(poll_interval_s)
                continue
            if payload.get("jobStatus") in {"ERROR", "FAILED"}:
                raise RuntimeError(f"UniProt ID-mapping job failed for {from_db}: {payload}")
            # Finished responses can contain jobStatus=FINISHED or inline results/failedIds.
            finished = True
            break
        if not finished:
            raise RuntimeError(f"UniProt ID-mapping job did not finish for {from_db}")

        query = urllib.parse.urlencode(
            {"format": "tsv", "fields": "accession,sequence,length"}
        )
        mapped = _audited_request(
            method="GET",
            url=f"{UNIPROT_API}/idmapping/uniprotkb/results/stream/{job_id}?{query}",
            body=b"", headers={"Accept": "text/tab-separated-values"},
            relative_stem=f"{stem}/results", output_dir=output_dir, requests=requests,
            raw_files=raw_files, transport=transport, sleep_fn=sleep_fn, timeout_s=timeout_s,
            max_attempts=max_attempts,
        )
        mappings, targets = _parse_mapping_tsv(bytes(mapped.body))
        for source_id, values in mappings.items():
            external_mappings[source_id] = values
        external_targets.update(targets)

    requested_canonical = sorted(direct | external_targets)
    canonical_sequences: dict[str, str] = {}
    for batch_idx, start in enumerate(range(0, len(requested_canonical), sequence_batch_size)):
        batch = requested_canonical[start:start + sequence_batch_size]
        expression = "(" + " OR ".join(f"accession:{item}" for item in batch) + ")"
        query = urllib.parse.urlencode(
            {"query": expression, "format": "tsv", "fields": "accession,sequence,length", "size": 500}
        )
        response = _audited_request(
            method="GET", url=f"{UNIPROT_API}/uniprotkb/search?{query}", body=b"",
            headers={"Accept": "text/tab-separated-values"},
            relative_stem=f"sequence_search/batch_{batch_idx:03d}", output_dir=output_dir,
            requests=requests, raw_files=raw_files, transport=transport, sleep_fn=sleep_fn,
            timeout_s=timeout_s, max_attempts=max_attempts,
        )
        for accession, sequence in _parse_sequence_tsv(bytes(response.body)).items():
            previous = canonical_sequences.get(accession)
            if previous is not None and previous != sequence:
                raise ValueError(f"conflicting canonical UniProt sequence for {accession}")
            canonical_sequences[accession] = sequence

    raw_files.sort(key=lambda row: row["path"])
    merkle = _sha(_canonical(raw_files))
    return UniProtIdentitySnapshot(
        external_mappings={key: external_mappings[key] for key in sorted(external_mappings)},
        canonical_sequences={key: canonical_sequences[key] for key in sorted(canonical_sequences)},
        requests=tuple(requests),
        raw_files=tuple(raw_files),
        raw_file_merkle_sha256=merkle,
    )
