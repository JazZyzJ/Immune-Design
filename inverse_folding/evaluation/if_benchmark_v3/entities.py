"""RCSB polymer-entity source contracts for benchmark v3."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .selection import AA20, build_exact_sequence_groups


def build_rcsb_entity_query(*, page_start: int, page_rows: int) -> dict[str, Any]:
    """Return the canonical C1 Search API request for one deterministic page."""

    if page_start < 0 or page_rows < 1:
        raise ValueError("invalid RCSB pagination")
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match", "value": "Protein",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "exptl.method", "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal", "value": 2.5,
                    },
                },
                {
                    "type": "terminal", "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {
                            "from": 100, "to": 500,
                            "include_lower": True, "include_upper": True,
                        },
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": page_start, "rows": page_rows},
            "sort": [{"sort_by": "rcsb_id", "direction": "asc"}],
            "results_verbosity": "minimal",
        },
    }


def parse_search_pages(pages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Join persisted Search API pages and verify their count/order identity."""

    if not pages:
        raise ValueError("RCSB search snapshot has no pages")
    totals = {int(page.get("total_count", -1)) for page in pages}
    if len(totals) != 1:
        raise ValueError("RCSB search pages disagree on total_count")
    total = totals.pop()
    identifiers: list[str] = []
    for page in pages:
        result_set = page.get("result_set")
        if not isinstance(result_set, list):
            raise ValueError("RCSB search page lacks result_set")
        for row in result_set:
            identifier = str(row.get("identifier") or "")
            if not identifier:
                raise ValueError("RCSB search result lacks identifier")
            identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RCSB search snapshot contains duplicate entity IDs")
    if len(identifiers) != total:
        raise ValueError(
            f"RCSB search result count {len(identifiers)} != total_count {total}"
        )
    if identifiers != sorted(identifiers):
        raise ValueError("RCSB search pages are not in canonical lexical order")
    return identifiers


def parse_graphql_entity_page(
    payload_bytes: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse one content-bound GraphQL page without discarding entity/chain identity."""

    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid RCSB GraphQL JSON") from exc
    if payload.get("errors"):
        raise ValueError(f"RCSB GraphQL returned errors: {payload['errors']}")
    raw_entities = (payload.get("data") or {}).get("polymer_entities")
    if not isinstance(raw_entities, list):
        raise ValueError("RCSB GraphQL payload lacks polymer_entities")
    entities: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_entities:
        rcsb_id = str(raw.get("rcsb_id") or "")
        identifiers = raw.get("rcsb_polymer_entity_container_identifiers") or {}
        entry_id = str(identifiers.get("entry_id") or "").upper()
        entity_id = str(identifiers.get("entity_id") or "")
        expected = f"{entry_id}_{entity_id}"
        if not rcsb_id or rcsb_id.upper() != expected:
            raise ValueError(f"RCSB entity identity mismatch: {rcsb_id!r} vs {expected!r}")
        rcsb_id = rcsb_id.upper()
        if rcsb_id in seen:
            raise ValueError(f"duplicate GraphQL entity: {rcsb_id}")
        seen.add(rcsb_id)

        entity_poly = raw.get("entity_poly") or {}
        sequence = "".join(
            str(entity_poly.get("pdbx_seq_one_letter_code_can") or "").split()
        ).upper()
        declared_length = int(entity_poly.get("rcsb_sample_sequence_length", -1))
        polymer_type = str(entity_poly.get("rcsb_entity_polymer_type") or "")

        entry = raw.get("entry") or {}
        methods = sorted({str(item.get("method") or "") for item in entry.get("exptl") or []})
        resolutions = [
            float(value)
            for value in (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        accession = entry.get("rcsb_accession_info") or {}
        entities.append(
            {
                "rcsb_entity_id": rcsb_id,
                "pdb_id": entry_id,
                "entity_id": entity_id,
                "entity_sequence": sequence,
                "sequence_length": len(sequence),
                "declared_sequence_length": declared_length,
                "entity_polymer_type": polymer_type,
                "experimental_method": ";".join(methods),
                "resolution": min(resolutions) if resolutions else None,
                "initial_release_date": accession.get("initial_release_date"),
                "structure_revision_date": accession.get("revision_date"),
            }
        )

        raw_instances = raw.get("polymer_entity_instances")
        if not isinstance(raw_instances, list) or not raw_instances:
            raise ValueError(f"RCSB entity lacks authoritative polymer instances: {rcsb_id}")
        seen_instances: set[str] = set()
        for instance in raw_instances:
            container = (
                instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}
            )
            label_asym_id = str(container.get("asym_id") or "")
            auth_asym_id = str(container.get("auth_asym_id") or "")
            instance_entry = str(container.get("entry_id") or "").upper()
            instance_entity = str(container.get("entity_id") or "")
            instance_id = str(instance.get("rcsb_id") or "").upper()
            expected_instance = f"{entry_id}.{label_asym_id}".upper()
            if (
                not label_asym_id or not auth_asym_id
                or instance_entry != entry_id or instance_entity != entity_id
                or instance_id != expected_instance or instance_id in seen_instances
            ):
                raise ValueError(f"RCSB polymer instance identity mismatch: {instance_id}")
            seen_instances.add(instance_id)
            edges.append(
                {
                    "rcsb_instance_id": instance_id,
                    "rcsb_entity_id": rcsb_id,
                    "pdb_id": entry_id,
                    "entity_id": entity_id,
                    "label_asym_id": label_asym_id,
                    "auth_asym_id": auth_asym_id,
                }
            )
    entities.sort(key=lambda row: row["rcsb_entity_id"])
    edges.sort(
        key=lambda row: (
            row["rcsb_entity_id"], row["label_asym_id"], row["auth_asym_id"]
        )
    )
    return entities, edges


def validate_search_graphql_identity(
    search_entity_ids: Sequence[str], graphql_entities: Sequence[Mapping[str, Any]]
) -> None:
    search = [str(value).upper() for value in search_entity_ids]
    graphql = [str(row.get("rcsb_entity_id") or "").upper() for row in graphql_entities]
    if len(graphql) != len(set(graphql)):
        raise ValueError("GraphQL entity IDs are not unique")
    if set(search) != set(graphql) or len(search) != len(graphql):
        missing = sorted(set(search) - set(graphql))
        extra = sorted(set(graphql) - set(search))
        raise ValueError(f"Search/GraphQL ID-set mismatch: missing={missing}, extra={extra}")


def build_entity_source_pool(
    entities: Sequence[Mapping[str, Any]],
    chain_edges: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Revalidate C1 from fetched fields, ledger rejections, then apply C2 grouping."""

    entity_ids = [str(row.get("rcsb_entity_id") or "") for row in entities]
    if not all(entity_ids) or len(entity_ids) != len(set(entity_ids)):
        raise ValueError("entity source table has duplicate/missing IDs")
    entity_id_set = set(entity_ids)
    edges_by_entity: dict[str, list[dict[str, Any]]] = {}
    for edge in chain_edges:
        entity_id = str(edge.get("rcsb_entity_id") or "")
        if entity_id not in entity_id_set:
            raise ValueError(f"chain edge references unknown entity: {entity_id}")
        edges_by_entity.setdefault(entity_id, []).append(dict(edge))

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in entities:
        row = dict(source)
        entity_id = str(row["rcsb_entity_id"])
        sequence = "".join(str(row.get("entity_sequence") or "").split()).upper()
        reasons: list[str] = []
        if str(row.get("entity_polymer_type") or "Protein") != "Protein":
            reasons.append("non_protein_polymer")
        if not sequence or not set(sequence) <= AA20:
            reasons.append("non_aa20_sequence")
        declared = int(row.get("declared_sequence_length", row.get("sequence_length", -1)))
        if declared != len(sequence):
            reasons.append("declared_length_mismatch")
        if not 100 <= len(sequence) <= 500:
            reasons.append("sequence_length_outside_100_500")
        methods = {part.strip().upper() for part in str(row.get("experimental_method") or "").split(";")}
        if "X-RAY DIFFRACTION" not in methods:
            reasons.append("non_xray_method")
        resolution = row.get("resolution")
        if not isinstance(resolution, (int, float)) or isinstance(resolution, bool) or not math.isfinite(float(resolution)):
            reasons.append("resolution_missing_or_nonfinite")
        elif float(resolution) > 2.5:
            reasons.append("resolution_above_2_5")
        if not edges_by_entity.get(entity_id):
            reasons.append("no_entity_chain_edges")
        if reasons:
            rejected.append(
                {
                    "rcsb_entity_id": entity_id,
                    "rejection_reasons_json": json.dumps(reasons, separators=(",", ":")),
                }
            )
            continue
        row["entity_sequence"] = sequence
        row["sequence_length"] = len(sequence)
        row["resolution"] = float(resolution)
        eligible.append(row)

    groups, fallback_rows = build_exact_sequence_groups(eligible)
    expanded: list[dict[str, Any]] = []
    for fallback in fallback_rows:
        entity_chain_edges = sorted(
            [dict(edge) for edge in edges_by_entity[fallback["rcsb_entity_id"]]],
            key=lambda edge: (edge["label_asym_id"], edge["auth_asym_id"]),
        )
        expanded.append(
            {
                **fallback,
                "entity_chain_edges": entity_chain_edges,
            }
        )
    return groups, expanded, rejected


def validate_observed_chain_entity(
    expected_edge: Mapping[str, Any],
    *,
    observed_entity_id: str,
    observed_label_asym_id: str,
    observed_auth_asym_id: str,
) -> None:
    """Fail when a cached/downloaded chain does not implement the entity edge."""

    expected_entity = str(expected_edge["rcsb_entity_id"]).rsplit("_", 1)[1]
    expected = (
        expected_entity,
        str(expected_edge["label_asym_id"]),
        str(expected_edge["auth_asym_id"]),
    )
    observed = (
        str(observed_entity_id), str(observed_label_asym_id), str(observed_auth_asym_id)
    )
    if observed != expected:
        raise ValueError(f"wrong entity-chain mapping: observed={observed}, expected={expected}")
