"""Entity-level RCSB source producer contracts."""

from __future__ import annotations

import json

import pytest

from inverse_folding.evaluation.if_benchmark_v3.entities import (
    build_entity_source_pool,
    build_rcsb_entity_query,
    parse_graphql_entity_page,
    parse_search_pages,
    validate_search_graphql_identity,
    validate_observed_chain_entity,
)
from inverse_folding.evaluation.if_benchmark_v3.rcsb_snapshot import snapshot_rcsb_entities
from inverse_folding.evaluation.if_benchmark_v3.preflight import SiftsHttpResponse


def test_rcsb_query_is_polymer_entity_level_and_freezes_all_c1_filters():
    query = build_rcsb_entity_query(page_start=0, page_rows=10000)
    assert query["return_type"] == "polymer_entity"
    nodes = query["query"]["nodes"]
    predicates = {(node["parameters"]["attribute"], node["parameters"]["operator"])
                  for node in nodes}
    assert ("entity_poly.rcsb_entity_polymer_type", "exact_match") in predicates
    assert ("exptl.method", "exact_match") in predicates
    assert ("rcsb_entry_info.resolution_combined", "less_or_equal") in predicates
    assert ("entity_poly.rcsb_sample_sequence_length", "range") in predicates
    assert query["request_options"]["sort"] == [{"sort_by": "rcsb_id", "direction": "asc"}]


def test_search_pages_require_exact_total_and_unique_entity_ids():
    pages = [
        {"total_count": 3, "result_set": [{"identifier": "1AAA_1"}, {"identifier": "1AAA_2"}]},
        {"total_count": 3, "result_set": [{"identifier": "2BBB_1"}]},
    ]
    assert parse_search_pages(pages) == ["1AAA_1", "1AAA_2", "2BBB_1"]
    with pytest.raises(ValueError, match="duplicate"):
        parse_search_pages([{"total_count": 2, "result_set": [
            {"identifier": "1AAA_1"}, {"identifier": "1AAA_1"}]}])
    with pytest.raises(ValueError, match="total_count"):
        parse_search_pages([{"total_count": 3, "result_set": [{"identifier": "1AAA_1"}]}])


def test_graphql_page_retains_entity_sequence_revision_and_all_chain_edges():
    payload = {
        "data": {"polymer_entities": [{
            "rcsb_id": "1AAA_2",
            "entity_poly": {
                "pdbx_seq_one_letter_code_can": "ACDE",
                "rcsb_sample_sequence_length": 4,
                "rcsb_entity_polymer_type": "Protein",
            },
            "rcsb_polymer_entity_container_identifiers": {
                "entry_id": "1AAA", "entity_id": "2",
            },
            "polymer_entity_instances": [
                {"rcsb_id": "1AAA.B",
                 "rcsb_polymer_entity_instance_container_identifiers": {
                     "entry_id": "1AAA", "entity_id": "2", "asym_id": "B",
                     "auth_asym_id": "X"}},
                {"rcsb_id": "1AAA.C",
                 "rcsb_polymer_entity_instance_container_identifiers": {
                     "entry_id": "1AAA", "entity_id": "2", "asym_id": "C",
                     "auth_asym_id": "X"}},
            ],
            "entry": {
                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                "rcsb_entry_info": {"resolution_combined": [1.8]},
                "rcsb_accession_info": {
                    "initial_release_date": "2020-01-01T00:00:00Z",
                    "revision_date": "2026-01-02T00:00:00Z",
                },
            },
        }]},
    }
    entities, edges = parse_graphql_entity_page(json.dumps(payload).encode())
    assert entities[0]["rcsb_entity_id"] == "1AAA_2"
    assert entities[0]["entity_sequence"] == "ACDE"
    assert entities[0]["structure_revision_date"] == "2026-01-02T00:00:00Z"
    assert [(e["label_asym_id"], e["auth_asym_id"]) for e in edges] == [("B", "X"), ("C", "X")]


def test_graphql_page_rejects_wrong_entity_identity_and_chain_array_mismatch():
    base = {
        "data": {"polymer_entities": [{
            "rcsb_id": "1AAA_2",
            "entity_poly": {"pdbx_seq_one_letter_code_can": "ACDE",
                            "rcsb_sample_sequence_length": 4,
                            "rcsb_entity_polymer_type": "Protein"},
            "rcsb_polymer_entity_container_identifiers": {
                "entry_id": "1AAA", "entity_id": "1",
            },
            "polymer_entity_instances": [{
                "rcsb_id": "1AAA.A",
                "rcsb_polymer_entity_instance_container_identifiers": {
                    "entry_id": "1AAA", "entity_id": "1", "asym_id": "A",
                    "auth_asym_id": "A"},
            }],
            "entry": {"exptl": [{"method": "X-RAY DIFFRACTION"}],
                      "rcsb_entry_info": {"resolution_combined": [1.0]},
                      "rcsb_accession_info": {"revision_date": "2026-01-01"}},
        }]},
    }
    with pytest.raises(ValueError, match="entity identity"):
        parse_graphql_entity_page(json.dumps(base).encode())
    entity = base["data"]["polymer_entities"][0]
    entity["rcsb_polymer_entity_container_identifiers"]["entity_id"] = "2"
    entity["polymer_entity_instances"][0][
        "rcsb_polymer_entity_instance_container_identifiers"
    ]["entity_id"] = "1"
    with pytest.raises(ValueError, match="instance identity"):
        parse_graphql_entity_page(json.dumps(base).encode())


def test_source_revalidation_ledgers_ineligible_rows_without_aborting_pool():
    entities = [
        {"rcsb_entity_id": "1AAA_1", "pdb_id": "1AAA", "entity_id": "1",
         "entity_sequence": "A" * 100, "sequence_length": 100,
         "declared_sequence_length": 100, "experimental_method": "X-RAY DIFFRACTION",
         "resolution": 1.5},
        {"rcsb_entity_id": "2BBB_1", "pdb_id": "2BBB", "entity_id": "1",
         "entity_sequence": "A" * 99 + "X", "sequence_length": 100,
         "declared_sequence_length": 100, "experimental_method": "X-RAY DIFFRACTION",
         "resolution": 1.5},
        {"rcsb_entity_id": "3CCC_1", "pdb_id": "3CCC", "entity_id": "1",
         "entity_sequence": "C" * 100, "sequence_length": 100,
         "declared_sequence_length": 100, "experimental_method": "ELECTRON MICROSCOPY",
         "resolution": 1.5},
        {"rcsb_entity_id": "4DDD_1", "pdb_id": "4DDD", "entity_id": "1",
         "entity_sequence": "D" * 100, "sequence_length": 100,
         "declared_sequence_length": 101, "experimental_method": "X-RAY DIFFRACTION",
         "resolution": 3.0},
    ]
    edges = [
        {"rcsb_entity_id": row["rcsb_entity_id"], "pdb_id": row["pdb_id"],
         "entity_id": "1", "label_asym_id": "A", "auth_asym_id": "A"}
        for row in entities
    ]
    groups, fallback_edges, rejected = build_entity_source_pool(entities, edges)
    assert len(groups) == 1 and len(fallback_edges) == 1
    assert groups[0]["ordered_entity_fallbacks"] == ["1AAA_1"]
    reasons = {row["rcsb_entity_id"]: json.loads(row["rejection_reasons_json"])
               for row in rejected}
    assert reasons["2BBB_1"] == ["non_aa20_sequence"]
    assert reasons["3CCC_1"] == ["non_xray_method"]
    assert set(reasons["4DDD_1"]) == {"declared_length_mismatch", "resolution_above_2_5"}


def test_large_entity_edge_pool_uses_indexed_fallback_expansion():
    n = 10_000
    entities = [
        {"rcsb_entity_id": f"{index:04X}_1", "pdb_id": f"{index:04X}", "entity_id": "1",
         "entity_sequence": "A" * 100, "sequence_length": 100,
         "declared_sequence_length": 100, "entity_polymer_type": "Protein",
         "experimental_method": "X-RAY DIFFRACTION", "resolution": 2.0}
        for index in range(n)
    ]
    edges = [
        {"rcsb_entity_id": row["rcsb_entity_id"], "pdb_id": row["pdb_id"],
         "entity_id": "1", "label_asym_id": "A", "auth_asym_id": "A"}
        for row in entities
    ]
    groups, fallbacks, rejected = build_entity_source_pool(entities, edges)
    assert len(groups) == 1 and len(fallbacks) == n and rejected == []
    assert all(len(row["entity_chain_edges"]) == 1 for row in fallbacks)


def test_search_and_graphql_complete_entity_sets_must_match():
    entities = [{"rcsb_entity_id": "1AAA_1"}, {"rcsb_entity_id": "2BBB_1"}]
    validate_search_graphql_identity(["1AAA_1", "2BBB_1"], entities)
    with pytest.raises(ValueError, match="ID-set mismatch"):
        validate_search_graphql_identity(["1AAA_1", "3CCC_1"], entities)


def test_observed_chain_must_belong_to_expected_entity_edge():
    expected = {"rcsb_entity_id": "1AAA_2", "label_asym_id": "B", "auth_asym_id": "X"}
    validate_observed_chain_entity(
        expected, observed_entity_id="2", observed_label_asym_id="B", observed_auth_asym_id="X"
    )
    with pytest.raises(ValueError, match="wrong entity-chain mapping"):
        validate_observed_chain_entity(
            expected, observed_entity_id="1", observed_label_asym_id="B",
            observed_auth_asym_id="X",
        )


def test_content_bound_rcsb_snapshot_producer_tiny_fixture(tmp_path):
    entity_data = {
        "1AAA_1": ("A" * 100, "X-RAY DIFFRACTION", 1.5),
        "2BBB_1": ("A" * 100, "X-RAY DIFFRACTION", 2.0),
        "3CCC_1": ("C" * 99 + "X", "X-RAY DIFFRACTION", 1.0),
    }

    def transport(_method, url, body, _headers, _timeout):
        request = json.loads(body)
        if "search.rcsb.org" in url:
            start = request["request_options"]["paginate"]["start"]
            ids = ["1AAA_1", "2BBB_1", "3CCC_1"][start:start + 2]
            payload = {"total_count": 3, "result_set": [{"identifier": x} for x in ids]}
        else:
            ids = request["variables"]["ids"]
            rows = []
            for entity_id in ids:
                pdb, eid = entity_id.split("_")
                sequence, method, resolution = entity_data[entity_id]
                rows.append({
                    "rcsb_id": entity_id,
                    "entity_poly": {"pdbx_seq_one_letter_code_can": sequence,
                                    "rcsb_sample_sequence_length": len(sequence),
                                    "rcsb_entity_polymer_type": "Protein"},
                    "rcsb_polymer_entity_container_identifiers": {
                        "entry_id": pdb, "entity_id": eid},
                    "polymer_entity_instances": [{
                        "rcsb_id": f"{pdb}.A",
                        "rcsb_polymer_entity_instance_container_identifiers": {
                            "entry_id": pdb, "entity_id": eid, "asym_id": "A",
                            "auth_asym_id": "A"},
                    }],
                    "entry": {"exptl": [{"method": method}],
                              "rcsb_entry_info": {"resolution_combined": [resolution]},
                              "rcsb_accession_info": {"initial_release_date": "2020-01-01",
                                                      "revision_date": "2026-01-01"}},
                })
            payload = {"data": {"polymer_entities": rows}}
        return SiftsHttpResponse(200, json.dumps(payload).encode(), {"server": "fixture"})

    out = tmp_path / "rcsb"
    delays = []
    summary = snapshot_rcsb_entities(
        output_dir=out, transport=transport, sleep_fn=delays.append,
        search_page_rows=2, graphql_batch_size=2,
    )
    assert summary["searched_entity_count"] == 3
    assert summary["eligible_entity_count"] == 2
    assert summary["exact_sequence_group_count"] == 1
    rejected = json.loads((out / "source_manifest.json").read_text())
    assert rejected["parameters"]["api_delay_s"] == 0.1
    assert delays == [0.1, 0.1]
    assert rejected["raw_response_files"]["merkle_sha256"]
    assert len(rejected["raw_response_files"]["files"]) == 4
