"""Contract tests for the IF benchmark v3 Tier-1 source preflight."""

from __future__ import annotations

import hashlib
import json
import urllib.error

import pytest

from inverse_folding.evaluation.if_benchmark_v3.preflight import (
    CHAIN_ATTEMPT_COLUMNS,
    SiftsHttpResponse,
    VALID_SOURCE_UNIT_COLUMNS,
    evaluate_sifts_mapping,
    parse_sifts_mappings,
    request_sifts_with_retry,
    run_tier1_source_preflight,
    validate_preflight_raw_file_bindings,
)
from inverse_folding.evaluation.if_benchmark_v3.source_identity import (
    classify_source_accession,
    resolve_canonical_source_units,
)
from inverse_folding.evaluation.if_benchmark_v3.uniprot_snapshot import (
    snapshot_uniprot_identity,
)


def _mapping(**overrides):
    row = {
        "pdb_id": "1ABC",
        "chain_id": "A",
        "unp_start": 1,
        "unp_end": 120,
        "pdb_start": 1,
        "pdb_end": 120,
        "resolution": 2.0,
        "experimental_method": "X-ray diffraction",
        "coverage": 1.0,
    }
    row.update(overrides)
    return row


def _spans():
    return [
        {"start_0b": 0, "end_0b": 15, "peptide": "A" * 15},
        {"start_0b": 20, "end_0b": 35, "peptide": "A" * 15},
    ]


def test_sifts_retry_records_transient_attempts_and_raw_hash():
    calls = []

    def request(_url, _timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(_url, 503, "busy", {}, None)
        return SiftsHttpResponse(200, b'{"P1": []}', {"content-type": "application/json"})

    result = request_sifts_with_retry(
        "P1", request_fn=request, sleep_fn=lambda _seconds: None, max_attempts=3,
    )

    assert [a["outcome"] for a in result.attempts] == ["http_503", "http_200"]
    assert result.final_status == "http_200"
    assert result.response_sha256 == hashlib.sha256(b'{"P1": []}').hexdigest()
    assert result.network_failure is False


def test_sifts_404_and_exhausted_network_failure_are_distinct():
    def not_found(url, _timeout):
        raise urllib.error.HTTPError(url, 404, "missing", {}, None)

    missing = request_sifts_with_retry(
        "P1", request_fn=not_found, sleep_fn=lambda _seconds: None, max_attempts=3,
    )
    assert missing.final_status == "http_404"
    assert missing.network_failure is False
    assert len(missing.attempts) == 1

    def unavailable(_url, _timeout):
        raise urllib.error.URLError("offline")

    failed = request_sifts_with_retry(
        "P1", request_fn=unavailable, sleep_fn=lambda _seconds: None, max_attempts=3,
    )
    assert failed.final_status == "network_failure"
    assert failed.network_failure is True
    assert len(failed.attempts) == 3


def test_parse_sifts_empty_payload_is_no_mapping_not_network_failure():
    assert parse_sifts_mappings(b"{}", "P1") == []
    assert parse_sifts_mappings(b'{"P1": []}', "P1") == []


def test_source_frame_candidate_passes_with_exact_peptides():
    result = evaluate_sifts_mapping(
        source_id="P1", mapping=_mapping(), full_sequence="A" * 120, spans=_spans(),
    )
    assert result["source_valid"] is True
    assert result["verified_span_count"] == 2
    assert result["verified_covered_residue_count"] == 30
    assert result["source_epitope_coverage"] == pytest.approx(0.25)
    assert json.loads(result["failure_reasons_json"]) == []


@pytest.mark.parametrize(
    ("mapping_overrides", "sequence", "spans", "reason"),
    [
        ({"experimental_method": "Electron microscopy"}, "A" * 120, _spans(), "non_xray"),
        ({"resolution": 3.0}, "A" * 120, _spans(), "resolution_above_2_5"),
        ({"unp_end": 99}, "A" * 120, _spans(), "mapped_range_length_outside_100_500"),
        ({"unp_end": 130}, "A" * 120, _spans(), "mapped_range_outside_sequence"),
        ({}, "A" * 120, _spans()[:1], "fewer_than_2_verified_spans"),
        ({}, "A" * 120, [{**_spans()[0], "peptide": "C" * 15}, _spans()[1]],
         "fewer_than_2_verified_spans"),
        ({}, "A" * 120,
         [{"start_0b": 0, "end_0b": 6, "peptide": "A" * 6},
          {"start_0b": 10, "end_0b": 15, "peptide": "A" * 5}],
         "source_epitope_coverage_outside_0_10_0_50"),
    ],
)
def test_source_frame_candidate_fails_closed(mapping_overrides, sequence, spans, reason):
    result = evaluate_sifts_mapping(
        source_id="P1", mapping=_mapping(**mapping_overrides),
        full_sequence=sequence, spans=spans,
    )
    assert result["source_valid"] is False
    assert reason in json.loads(result["failure_reasons_json"])


def test_versioned_uniprot_aliases_collapse_to_one_canonical_source_unit():
    assert classify_source_accession("P34925.3") == ("uniprot", "P34925")
    sequence = "A" * 120
    units, ledger = resolve_canonical_source_units(
        raw_sequences={"P34925": sequence, "P34925.3": sequence},
        raw_spans={"P34925": _spans(), "P34925.3": _spans()},
        external_mappings={},
        canonical_sequences={"P34925": sequence},
    )
    assert list(units) == ["P34925"]
    assert units["P34925"]["raw_source_ids"] == ["P34925", "P34925.3"]
    assert units["P34925"]["canonical_sequence_sha256"] == hashlib.sha256(
        sequence.encode()
    ).hexdigest()
    assert {row["resolution_status"] for row in ledger} == {"resolved"}


def test_same_uniprot_alias_sequence_conflict_fails_source_unit():
    units, ledger = resolve_canonical_source_units(
        raw_sequences={"P01023": "A" * 120, "P01023.1": "C" * 120},
        raw_spans={"P01023": _spans(), "P01023.1": _spans()},
        external_mappings={},
        canonical_sequences={"P01023": "A" * 120},
    )
    assert units == {}
    assert {row["resolution_status"] for row in ledger} == {"alias_sequence_conflict"}


def test_unmapped_external_accession_has_separate_fail_closed_reason():
    assert classify_source_accession("NP_000001.1") == ("refseq", None)
    units, ledger = resolve_canonical_source_units(
        raw_sequences={"NP_000001.1": "A" * 120},
        raw_spans={"NP_000001.1": _spans()},
        external_mappings={},
        canonical_sequences={},
    )
    assert units == {}
    assert ledger[0]["resolution_status"] == "unmapped_external_accession"


def test_external_mapping_must_match_exact_canonical_sequence():
    sequence = "A" * 120
    units, ledger = resolve_canonical_source_units(
        raw_sequences={"NP_000001.1": sequence},
        raw_spans={"NP_000001.1": _spans()},
        external_mappings={"NP_000001.1": ["P12345", "Q12345"]},
        canonical_sequences={"P12345": "C" * 120, "Q12345": sequence},
    )
    assert list(units) == ["Q12345"]
    assert ledger[0]["resolution_status"] == "resolved"
    assert ledger[0]["mapping_resolution"] == "external_exact_sequence"


def test_external_alias_sequence_conflict_invalidates_same_direct_uniprot_unit():
    canonical = "A" * 120
    units, ledger = resolve_canonical_source_units(
        raw_sequences={"P12345": canonical, "AAA00001.1": "C" * 120},
        raw_spans={"P12345": _spans(), "AAA00001.1": _spans()},
        external_mappings={"AAA00001.1": ["P12345"]},
        canonical_sequences={"P12345": canonical},
    )
    assert units == {}
    assert {row["resolution_status"] for row in ledger} == {"alias_sequence_conflict"}
    assert {row["canonical_uniprot_id"] for row in ledger} == {"P12345"}
    assert {row["alias_resolution_status"] for row in ledger} == {
        "resolved", "alias_sequence_conflict"
    }
    assert {row["unit_invalidation_reason"] for row in ledger} == {
        "alias_sequence_conflict"
    }


def test_uniprot_snapshot_persists_mapping_and_canonical_sequence_bodies(tmp_path):
    responses = {
        "configure": b'{"groups":[],"rules":[]}',
        "submit": b'{"jobId":"job1"}',
        "status": b'{"jobStatus":"FINISHED"}',
        "mapping": (
            b"From\tEntry\tSequence\tLength\n"
            b"NP_000001.1\tQ12345\t" + b"A" * 120 + b"\t120\n"
        ),
        "search": (
            b"Entry\tSequence\tLength\n"
            b"P34925\t" + b"C" * 120 + b"\t120\n"
            b"Q12345\t" + b"A" * 120 + b"\t120\n"
        ),
    }

    def transport(method, url, body, headers, timeout):
        assert timeout > 0
        if url.endswith("/configure/idmapping/fields"):
            key = "configure"
        elif url.endswith("/idmapping/run"):
            assert method == "POST" and b"RefSeq_Protein" in body
            key = "submit"
        elif "/idmapping/status/" in url:
            key = "status"
        elif "/idmapping/uniprotkb/results/stream/" in url:
            key = "mapping"
        else:
            assert "/uniprotkb/search?" in url
            key = "search"
        return SiftsHttpResponse(200, responses[key], {"x-uniprot-release": "2026_02"})

    snapshot = snapshot_uniprot_identity(
        raw_source_ids=["P34925.3", "NP_000001.1"],
        output_dir=tmp_path / "uniprot",
        transport=transport,
        sleep_fn=lambda _seconds: None,
        sequence_batch_size=10,
    )
    assert snapshot.external_mappings == {"NP_000001.1": ["Q12345"]}
    assert snapshot.canonical_sequences == {"P34925": "C" * 120, "Q12345": "A" * 120}
    assert snapshot.raw_file_merkle_sha256
    for entry in snapshot.raw_files:
        path = tmp_path / "uniprot" / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_preflight_persists_content_bound_funnel_and_shortfall(tmp_path):
    import pandas as pd

    ids = tmp_path / "test_ids.txt"
    ids.write_text("P12345\nQ12345\n")
    proteins = tmp_path / "proteins.parquet"
    spans = tmp_path / "spans.parquet"
    pd.DataFrame(
        [
            {"protein_id": "P12345", "allele": "HLA-DRB1*15:01",
             "protein_seq": "A" * 120, "sequence_length": 120},
            {"protein_id": "Q12345", "allele": "HLA-DRB1*15:01",
             "protein_seq": "A" * 120, "sequence_length": 120},
        ]
    ).to_parquet(proteins, index=False)
    span_rows = []
    for pid in ("P12345", "Q12345"):
        for span in _spans():
            span_rows.append(
                {"protein_id": pid, "allele": "HLA-DRB1*15:01",
                 "start_0b": span["start_0b"], "end_0b": span["end_0b"],
                 "peptide_seq": span["peptide"], "source": "iedb"}
            )
    pd.DataFrame(span_rows).to_parquet(spans, index=False)

    payload = json.dumps(
        {"P12345": [{"pdb_id": "1abc", "chain_id": "A", "unp_start": 1,
                  "unp_end": 120, "start": 1, "end": 120, "resolution": 2.0,
                  "experimental_method": "X-ray diffraction", "coverage": 1.0}]}
    ).encode()

    def request(url, _timeout):
        if url.endswith("/P12345"):
            return SiftsHttpResponse(200, payload, {"content-type": "application/json"})
        return SiftsHttpResponse(200, b'{"Q12345": []}', {"content-type": "application/json"})

    def uniprot_transport(_method, url, _body, _headers, _timeout):
        if url.endswith("/configure/idmapping/fields"):
            return SiftsHttpResponse(200, b'{"groups":[],"rules":[]}', {})
        return SiftsHttpResponse(
            200,
            b"Entry\tSequence\tLength\nP12345\t" + b"A" * 120 + b"\t120\n"
            b"Q12345\t" + b"A" * 120 + b"\t120\n",
            {},
        )

    output = tmp_path / "audit" / "preflight"
    delays = []
    summary = run_tier1_source_preflight(
        allele="HLA-DRB1*15:01",
        test_ids_path=ids,
        protein_samples_path=proteins,
        span_records_path=spans,
        output_dir=output,
        tier1_target=2,
        request_fn=request,
        uniprot_transport=uniprot_transport,
        sleep_fn=delays.append,
        api_delay_s=0.25,
        command_argv=["fixture"],
    )

    assert summary["release_capacity_pass"] is False
    assert delays == [0.25, 0.25]
    assert summary["funnel"]["input_source_units"] == 2
    assert summary["funnel"]["with_parsed_sifts_mapping"] == 1
    assert summary["funnel"]["source_frame_valid"] == 1
    assert (output / "sifts" / "responses" / "P12345.body").read_bytes() == payload
    requests = pd.read_parquet(output / "sifts_requests.parquet")
    assert requests.loc[requests.source_id == "P12345", "response_sha256"].item() == hashlib.sha256(payload).hexdigest()
    valid = pd.read_parquet(output / "tier1_source_frame_valid_units.parquet")
    assert valid[["source_id", "viable_sifts_mapping_count"]].to_dict("records") == [
        {"source_id": "P12345", "viable_sifts_mapping_count": 1}
    ]
    assert "preflight_source_rank" not in valid.columns
    manifest = json.loads((output / "preflight_manifest.json").read_text())
    assert manifest["api"]["api_delay_s"] == 0.25
    assert manifest["input_files"]["test_ids"]["sha256"] == hashlib.sha256(ids.read_bytes()).hexdigest()
    assert manifest["output_files"]["tier1_source_frame_valid_units.parquet"]["sha256"]
    assert manifest["sifts_response_files"]["merkle_sha256"]
    for entry in manifest["sifts_response_files"]["files"]:
        body_path = output / entry["path"]
        assert hashlib.sha256(body_path.read_bytes()).hexdigest() == entry["sha256"]
    validate_preflight_raw_file_bindings(manifest, output)
    tampered = output / manifest["sifts_response_files"]["files"][0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_preflight_raw_file_bindings(manifest, output)


def test_empty_sifts_mapping_uses_fixed_typed_artifact_schemas(tmp_path):
    import pandas as pd

    ids = tmp_path / "ids.txt"
    ids.write_text("P12345\n")
    proteins = tmp_path / "proteins.parquet"
    spans = tmp_path / "spans.parquet"
    pd.DataFrame([{"protein_id": "P12345", "allele": "HLA-DRB1*15:01",
                   "protein_seq": "A" * 120, "sequence_length": 120}]).to_parquet(proteins)
    pd.DataFrame([{"protein_id": "P12345", "allele": "HLA-DRB1*15:01",
                   "start_0b": 0, "end_0b": 15, "peptide_seq": "A" * 15,
                   "source": "iedb"}]).to_parquet(spans)

    def uniprot_transport(_method, url, _body, _headers, _timeout):
        if url.endswith("/configure/idmapping/fields"):
            return SiftsHttpResponse(200, b"{}", {})
        return SiftsHttpResponse(
            200, b"Entry\tSequence\tLength\nP12345\t" + b"A" * 120 + b"\t120\n", {}
        )

    out = tmp_path / "preflight"
    run_tier1_source_preflight(
        allele="HLA-DRB1*15:01", test_ids_path=ids, protein_samples_path=proteins,
        span_records_path=spans, output_dir=out, tier1_target=15,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(200, b'{"P12345": []}', {}),
        uniprot_transport=uniprot_transport, sleep_fn=lambda _seconds: None,
    )
    assert list(pd.read_parquet(out / "tier1_sifts_chain_attempts.parquet").columns) == CHAIN_ATTEMPT_COLUMNS
    assert list(pd.read_parquet(out / "tier1_source_frame_valid_units.parquet").columns) == VALID_SOURCE_UNIT_COLUMNS
