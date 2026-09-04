"""Content-bound RCSB Search+GraphQL polymer-entity snapshot producer."""

from __future__ import annotations

import hashlib
import json
import os
import time
import http.client
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .entities import (
    build_entity_source_pool,
    build_rcsb_entity_query,
    parse_graphql_entity_page,
    parse_search_pages,
    validate_search_graphql_identity,
)
from .preflight import SiftsHttpResponse


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
GRAPHQL_QUERY = """
query EntitySnapshot($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    entity_poly {
      pdbx_seq_one_letter_code_can
      rcsb_sample_sequence_length
      rcsb_entity_polymer_type
    }
    rcsb_polymer_entity_container_identifiers {
      entry_id entity_id
    }
    polymer_entity_instances {
      rcsb_id
      rcsb_polymer_entity_instance_container_identifiers {
        entry_id entity_id asym_id auth_asym_id
      }
    }
    entry {
      exptl { method }
      rcsb_entry_info { resolution_combined }
      rcsb_accession_info { initial_release_date revision_date }
    }
  }
}
""".strip()


class RcsbRequestError(RuntimeError):
    """Terminal RCSB request failure retaining every attempted request."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]], response: Any = None):
        super().__init__(message)
        self.attempts = list(attempts)
        self.response = response


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _default_transport(
    method: str, url: str, body: bytes, headers: Mapping[str, str], timeout_s: float
) -> SiftsHttpResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return SiftsHttpResponse(
            int(response.status), response.read(),
            {str(k).lower(): str(v) for k, v in response.headers.items()},
        )


def _request(
    *, url: str, body: bytes, transport: Callable[..., Any], sleep_fn: Callable[[float], None],
    timeout_s: float, max_attempts: int,
) -> tuple[Any, list[dict[str, Any]]]:
    attempts = []
    for attempt in range(1, max_attempts + 1):
        queried = _utc()
        try:
            response = transport(
                "POST", url, body,
                {"Content-Type": "application/json", "Accept": "application/json"},
                timeout_s,
            )
            status = int(response.status)
            attempts.append({"attempt": attempt, "queried_at_utc": queried, "http_status": status})
            if status == 429 or 500 <= status <= 599:
                if attempt < max_attempts:
                    sleep_fn(float(2 ** (attempt - 1)))
                    continue
                raise RcsbRequestError(
                    f"RCSB request exhausted HTTP {status}: {url}",
                    attempts=attempts, response=response,
                )
            if status != 200:
                raise RcsbRequestError(
                    f"RCSB request HTTP {status}: {url}",
                    attempts=attempts, response=response,
                )
            return response, attempts
        except (
            # IncompleteRead/BadStatusLine are HTTPExceptions, not OSErrors: a truncated
            # chunked response must be retried, not abort a multi-hour network stage.
            urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException,
        ) as exc:
            attempts.append(
                {"attempt": attempt, "queried_at_utc": queried, "http_status": None,
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            if attempt < max_attempts:
                sleep_fn(float(2 ** (attempt - 1)))
                continue
            raise RcsbRequestError(
                f"RCSB request exhausted transport retries: {url}",
                attempts=attempts,
            ) from exc
    raise AssertionError("RCSB retry loop terminated")


def snapshot_rcsb_entities(
    *,
    output_dir: Path,
    transport: Callable[..., Any] = _default_transport,
    sleep_fn: Callable[[float], None] = time.sleep,
    search_page_rows: int = 10000,
    graphql_batch_size: int = 250,
    timeout_s: float = 60.0,
    max_attempts: int = 4,
    api_delay_s: float = 0.1,
) -> dict[str, Any]:
    """Fetch, revalidate, and persist the complete C1/C2 entity source."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if search_page_rows < 1 or graphql_batch_size < 1 or api_delay_s < 0:
        raise ValueError("invalid RCSB batch dimensions")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    query_dir = output_dir / "queries"
    raw_dir.mkdir()
    query_dir.mkdir()
    request_rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []

    search_pages = []
    start = 0
    total: int | None = None
    page_idx = 0
    while total is None or start < total:
        query = build_rcsb_entity_query(page_start=start, page_rows=search_page_rows)
        body = _canonical(query)
        query_path = query_dir / f"search_{page_idx:04d}.json"
        query_path.write_bytes(body)
        response, attempts = _request(
            url=SEARCH_URL, body=body, transport=transport, sleep_fn=sleep_fn,
            timeout_s=timeout_s, max_attempts=max_attempts,
        )
        raw_path = raw_dir / f"search_{page_idx:04d}.body"
        raw_path.write_bytes(bytes(response.body))
        payload = json.loads(bytes(response.body))
        search_pages.append(payload)
        current_total = int(payload["total_count"])
        if total is None:
            total = current_total
        elif total != current_total:
            raise ValueError("RCSB total_count changed during snapshot")
        request_rows.append(
            {
                "request_kind": "search", "page_or_batch": page_idx, "url": SEARCH_URL,
                "query_path": str(query_path.relative_to(output_dir)),
                "query_sha256": _sha_file(query_path),
                "response_path": str(raw_path.relative_to(output_dir)),
                "response_sha256": _sha_file(raw_path),
                "response_headers_json": json.dumps(dict(response.headers), sort_keys=True),
                "attempts_json": json.dumps(attempts, sort_keys=True),
            }
        )
        raw_files.append(
            {"path": str(raw_path.relative_to(output_dir)),
             "size_bytes": raw_path.stat().st_size, "sha256": _sha_file(raw_path)}
        )
        page_count = len(payload.get("result_set") or [])
        start += page_count
        page_idx += 1
        if page_count == 0 and start < total:
            raise ValueError("RCSB search pagination stopped before total_count")

    entity_ids = parse_search_pages(search_pages)
    all_entities: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    for batch_idx, offset in enumerate(range(0, len(entity_ids), graphql_batch_size)):
        ids = entity_ids[offset:offset + graphql_batch_size]
        request_payload = {"query": GRAPHQL_QUERY, "variables": {"ids": ids}}
        body = _canonical(request_payload)
        query_path = query_dir / f"graphql_{batch_idx:04d}.json"
        query_path.write_bytes(body)
        response, attempts = _request(
            url=GRAPHQL_URL, body=body, transport=transport, sleep_fn=sleep_fn,
            timeout_s=timeout_s, max_attempts=max_attempts,
        )
        raw_path = raw_dir / f"graphql_{batch_idx:04d}.body"
        raw_path.write_bytes(bytes(response.body))
        if api_delay_s:
            sleep_fn(api_delay_s)
        entities, edges = parse_graphql_entity_page(bytes(response.body))
        all_entities.extend(entities)
        all_edges.extend(edges)
        request_rows.append(
            {
                "request_kind": "graphql_entity_instances", "page_or_batch": batch_idx,
                "url": GRAPHQL_URL,
                "query_path": str(query_path.relative_to(output_dir)),
                "query_sha256": _sha_file(query_path),
                "response_path": str(raw_path.relative_to(output_dir)),
                "response_sha256": _sha_file(raw_path),
                "response_headers_json": json.dumps(dict(response.headers), sort_keys=True),
                "attempts_json": json.dumps(attempts, sort_keys=True),
            }
        )
        raw_files.append(
            {"path": str(raw_path.relative_to(output_dir)),
             "size_bytes": raw_path.stat().st_size, "sha256": _sha_file(raw_path)}
        )

    validate_search_graphql_identity(entity_ids, all_entities)
    groups, fallback_rows, rejected = build_entity_source_pool(all_entities, all_edges)
    groups_frame = pd.DataFrame(groups)
    if len(groups_frame):
        groups_frame["ordered_entity_fallbacks_json"] = groups_frame[
            "ordered_entity_fallbacks"
        ].map(lambda value: json.dumps(value, separators=(",", ":")))
        groups_frame = groups_frame.drop(columns=["ordered_entity_fallbacks"])
    fallback_frame = pd.DataFrame(fallback_rows)
    if len(fallback_frame):
        fallback_frame["entity_chain_edges_json"] = fallback_frame["entity_chain_edges"].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        fallback_frame = fallback_frame.drop(columns=["entity_chain_edges"])

    pd.DataFrame(request_rows).to_parquet(output_dir / "request_ledger.parquet", index=False)
    pd.DataFrame(all_entities).to_parquet(output_dir / "entities_all.parquet", index=False)
    pd.DataFrame(all_edges).to_parquet(output_dir / "entity_chain_edges_all.parquet", index=False)
    pd.DataFrame(rejected, columns=["rcsb_entity_id", "rejection_reasons_json"]).to_parquet(
        output_dir / "entities_rejected.parquet", index=False
    )
    groups_frame.to_parquet(output_dir / "sequence_groups.parquet", index=False)
    fallback_frame.to_parquet(output_dir / "ordered_entity_fallbacks.parquet", index=False)
    output_names = [
        "request_ledger.parquet", "entities_all.parquet", "entity_chain_edges_all.parquet",
        "entities_rejected.parquet", "sequence_groups.parquet",
        "ordered_entity_fallbacks.parquet",
    ]
    outputs = {
        name: {"path": name, "size_bytes": (output_dir / name).stat().st_size,
               "sha256": _sha_file(output_dir / name)}
        for name in output_names
    }
    raw_files.sort(key=lambda row: row["path"])
    summary = {
        "searched_entity_count": len(entity_ids),
        "eligible_entity_count": len(fallback_rows),
        "rejected_entity_count": len(rejected),
        "exact_sequence_group_count": len(groups),
        "entity_chain_edge_count": len(all_edges),
    }
    manifest = {
        "schema_version": "if-benchmark-v3-rcsb-source/1",
        "protocol_id": "if-benchmark-test-set/3",
        "queried_at_utc": _utc(),
        "api_identity": {
            "search_url": SEARCH_URL, "graphql_url": GRAPHQL_URL,
            "graphql_query_sha256": _sha(GRAPHQL_QUERY.encode()),
            "search_query_contract_sha256": _sha(
                _canonical(build_rcsb_entity_query(page_start=0, page_rows=search_page_rows))
            ),
        },
        "parameters": {
            "search_page_rows": search_page_rows,
            "graphql_batch_size": graphql_batch_size,
            "timeout_s": timeout_s,
            "max_attempts": max_attempts,
            "api_delay_s": api_delay_s,
        },
        "summary": summary,
        "raw_response_files": {
            "files": raw_files,
            "merkle_sha256": _sha(_canonical(raw_files)),
        },
        "output_files": outputs,
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    temporary = output_dir / ".source_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "source_manifest.json")
    return summary
