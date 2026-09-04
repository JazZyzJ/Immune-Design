"""Content-bound RCSB mmCIF acquisition and cache revalidation."""

from __future__ import annotations

from collections import deque
import hashlib
import json

import pandas as pd
import pytest

from inverse_folding.evaluation.if_benchmark_v3.preflight import SiftsHttpResponse
from inverse_folding.evaluation.if_benchmark_v3.mapping import read_mmcif_atom_records
from inverse_folding.evaluation.if_benchmark_v3.smoke import (
    freeze_rcsb_structure_smoke_source,
    smoke_rcsb_multicopy_structure,
    validate_rcsb_structure_probe_manifest,
    validate_rcsb_structure_smoke_source,
)
from inverse_folding.evaluation.if_benchmark_v3.structures import (
    RcsbMmcifAcquisitionError,
    acquire_rcsb_mmcif,
    validate_structure_download_registry,
    validate_rcsb_failure_ledger,
    validate_cached_rcsb_mmcif,
    write_structure_download_registry,
)


def _mmcif(*, pdb_id: str = "1AAA", revision: str = "2026-01-02") -> bytes:
    return (
        f"data_{pdb_id}\n"
        f"_entry.id {pdb_id}\n"
        "loop_\n"
        "_pdbx_audit_revision_history.ordinal\n"
        "_pdbx_audit_revision_history.revision_date\n"
        f"1 {revision}\n"
        "#\n"
    ).encode()


def _multicopy_graphql() -> bytes:
    sequence = "A" * 100
    payload = {"data": {"polymer_entities": [{
        "rcsb_id": "1AAA_1",
        "entity_poly": {
            "pdbx_seq_one_letter_code_can": sequence,
            "rcsb_sample_sequence_length": len(sequence),
            "rcsb_entity_polymer_type": "Protein",
        },
        "rcsb_polymer_entity_container_identifiers": {
            "entry_id": "1AAA", "entity_id": "1",
        },
        "polymer_entity_instances": [
            {
                "rcsb_id": f"1AAA.{label}",
                "rcsb_polymer_entity_instance_container_identifiers": {
                    "entry_id": "1AAA", "entity_id": "1",
                    "asym_id": label, "auth_asym_id": auth,
                },
            }
            for label, auth in (("A", "X"), ("C", "Y"))
        ],
        "entry": {
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": [1.8]},
            "rcsb_accession_info": {
                "initial_release_date": "2020-01-01T00:00:00Z",
                "revision_date": "2026-01-02T00:00:00Z",
            },
        },
    }]}}
    return json.dumps(payload).encode()


def _multicopy_coordinate_mmcif() -> bytes:
    fields = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines = [
        "data_1AAA", "_entry.id 1AAA", "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.revision_date", "1 2026-01-02", "#",
        "loop_", *fields,
    ]
    serial = 1
    for label, auth in (("A", "X"), ("C", "Y")):
        for residue in range(1, 101):
            for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                element = "C" if atom in {"CA", "C"} else atom
                lines.append(
                    f"ATOM {serial} {element} {atom} . ALA {label} 1 {residue} ? "
                    f"{residue:.1f} {atom_index:.1f} 0.0 1.0 10.0 ? {residue} "
                    f"ALA {auth} {atom} 1"
                )
                serial += 1
    lines.append("#")
    return ("\n".join(lines) + "\n").encode()


def test_rcsb_mmcif_download_binds_revision_bytes_and_retry_ledger(tmp_path):
    responses = deque([
        SiftsHttpResponse(503, b"busy", {"retry-after": "1"}),
        SiftsHttpResponse(200, _mmcif(), {"etag": "fixture-etag"}),
    ])
    sleeps = []

    def request(url, timeout):
        assert url.endswith("/1AAA.cif")
        assert timeout == 7.0
        return responses.popleft()

    cif_path, manifest = acquire_rcsb_mmcif(
        pdb_id="1aaa", expected_revision_date="2026-01-02T00:00:00Z",
        cache_dir=tmp_path, request_fn=request, sleep_fn=sleeps.append,
        timeout_s=7.0,
    )
    assert cif_path.read_bytes() == _mmcif()
    assert manifest["observed_revision_date"] == "2026-01-02"
    assert [item["http_status"] for item in manifest["attempts"]] == [503, 200]
    assert sleeps == [0.1, 1.0, 0.1]
    assert manifest["api_delay_s"] == 0.1
    assert manifest["response_headers"] == {"etag": "fixture-etag"}
    assert cif_path == tmp_path / "1AAA" / "2026-01-02" / f"{manifest['sha256']}.cif"
    validate_cached_rcsb_mmcif(
        cif_path=cif_path, manifest_path=cif_path.parent / "manifest.json",
        pdb_id="1AAA", expected_revision_date="2026-01-02T00:00:00Z",
    )


def test_rcsb_mmcif_cache_reuse_performs_no_network_request(tmp_path):
    calls = []

    def first_request(_url, _timeout):
        calls.append("download")
        return SiftsHttpResponse(200, _mmcif(), {})

    first_path, first_manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02",
        cache_dir=tmp_path, request_fn=first_request,
    )

    def forbidden_request(_url, _timeout):  # pragma: no cover - failure assertion
        raise AssertionError("validated immutable cache must not hit the network")

    second_path, second_manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02",
        cache_dir=tmp_path, request_fn=forbidden_request,
    )
    assert calls == ["download"]
    assert second_path == first_path
    assert second_manifest == first_manifest


@pytest.mark.parametrize("tamper", ["bytes", "revision", "entry", "partial"])
def test_rcsb_mmcif_cache_or_source_identity_tampering_fails_closed(tmp_path, tamper):
    cif_path, _manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02",
        cache_dir=tmp_path,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(200, _mmcif(), {}),
    )
    manifest_path = cif_path.parent / "manifest.json"
    if tamper == "bytes":
        cif_path.write_bytes(cif_path.read_bytes() + b"# tamper\n")
    elif tamper == "revision":
        with pytest.raises(ValueError, match="source identity"):
            validate_cached_rcsb_mmcif(
                cif_path=cif_path, manifest_path=manifest_path, pdb_id="1AAA",
                expected_revision_date="2026-01-03",
            )
        return
    elif tamper == "entry":
        cif_path.write_bytes(_mmcif(pdb_id="2BBB"))
    else:
        manifest_path.unlink()
        with pytest.raises(ValueError, match="partial/legacy"):
            acquire_rcsb_mmcif(
                pdb_id="1AAA", expected_revision_date="2026-01-02",
                cache_dir=tmp_path,
                request_fn=lambda *_args: pytest.fail("must not replace partial cache"),
            )
        return
    with pytest.raises(ValueError, match="cache bytes changed"):
        validate_cached_rcsb_mmcif(
            cif_path=cif_path, manifest_path=manifest_path, pdb_id="1AAA",
            expected_revision_date="2026-01-02",
        )


def test_downloaded_wrong_entry_or_revision_leaves_no_reusable_partial_cache(tmp_path):
    for body in (_mmcif(pdb_id="2BBB"), _mmcif(revision="2026-01-03")):
        cache = tmp_path / str(len(list(tmp_path.iterdir())))
        with pytest.raises(RcsbMmcifAcquisitionError, match="identity mismatch"):
            acquire_rcsb_mmcif(
                pdb_id="1AAA", expected_revision_date="2026-01-02", cache_dir=cache,
                request_fn=lambda _url, _timeout, body=body: SiftsHttpResponse(200, body, {}),
            )
        assert len(list(cache.rglob("rejected_*.cif"))) == 1
        assert not list(cache.rglob("manifest.json"))
        assert len(list(cache.rglob("request_failure.json"))) == 1


def test_rcsb_cache_keeps_old_revision_immutable_when_new_revision_is_acquired(tmp_path):
    first_path, first_manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02", cache_dir=tmp_path,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(
            200, _mmcif(revision="2026-01-02"), {}
        ),
    )
    first_bytes = first_path.read_bytes()
    second_path, second_manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-02-03", cache_dir=tmp_path,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(
            200, _mmcif(revision="2026-02-03"), {}
        ),
    )
    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert first_manifest["sha256"] != second_manifest["sha256"]
    assert first_path.parent.name == "2026-01-02"
    assert second_path.parent.name == "2026-02-03"


@pytest.mark.parametrize("pdb_id,revision", [("1AA", "2026-01-02"), ("1AAA", "")])
def test_rcsb_mmcif_request_requires_frozen_entry_and_revision(pdb_id, revision, tmp_path):
    with pytest.raises(ValueError, match="invalid RCSB PDB/cache request"):
        acquire_rcsb_mmcif(
            pdb_id=pdb_id, expected_revision_date=revision, cache_dir=tmp_path,
            request_fn=lambda *_args: pytest.fail("invalid request must fail before network"),
        )


@pytest.mark.parametrize("status", [404, 503])
def test_rcsb_download_failure_persists_replayable_request_ledger(tmp_path, status):
    with pytest.raises(RcsbMmcifAcquisitionError) as captured:
        acquire_rcsb_mmcif(
            pdb_id="1AAA", expected_revision_date="2026-01-02", cache_dir=tmp_path,
            max_attempts=2,
            request_fn=lambda _url, _timeout: SiftsHttpResponse(status, b"failure-body", {}),
            sleep_fn=lambda _seconds: None,
        )
    ledger = captured.value.ledger_path
    assert ledger == tmp_path / "1AAA" / "2026-01-02" / "request_failure.json"
    payload = validate_rcsb_failure_ledger(
        ledger, pdb_id="1AAA", expected_revision_date="2026-01-02"
    )
    assert payload["terminal_http_status"] == status
    assert len(payload["attempts"]) == (1 if status == 404 else 2)
    ledger.write_bytes(ledger.read_bytes() + b"tamper")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        validate_rcsb_failure_ledger(
            ledger, pdb_id="1AAA", expected_revision_date="2026-01-02"
        )


def test_structure_download_registry_binds_every_success_and_failure_attempt(tmp_path):
    candidate = tmp_path / "candidate"
    cache = candidate / "audit" / "coordinates"
    success_path, _manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02", cache_dir=cache,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(200, _mmcif(), {}),
        sleep_fn=lambda _seconds: None,
    )
    with pytest.raises(RcsbMmcifAcquisitionError) as captured:
        acquire_rcsb_mmcif(
            pdb_id="2BBB", expected_revision_date="2026-01-02", cache_dir=cache,
            request_fn=lambda _url, _timeout: SiftsHttpResponse(404, b"missing", {}),
            sleep_fn=lambda _seconds: None,
        )
    attempts = pd.DataFrame([
        {
            "source_tier": "tier2", "selection_unit_id": "u1",
            "rcsb_entity_id": "1AAA_1", "label_asym_id": "A",
            "auth_chain_id": "A", "attempt_status": "viable",
            "source_revision_date": "2026-01-02T00:00:00Z",
            "download_sha256": hashlib.sha256(success_path.read_bytes()).hexdigest(),
        },
        {
            "source_tier": "tier2", "selection_unit_id": "u2",
            "rcsb_entity_id": "2BBB_1", "label_asym_id": "B",
            "auth_chain_id": "B", "attempt_status": "download_failure",
            "source_revision_date": "2026-01-02", "download_sha256": None,
        },
    ])
    attempts_path = candidate / "audit" / "chain_attempts.parquet"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts.to_parquet(attempts_path, index=False)
    registry = candidate / "audit" / "structure_downloads.parquet"
    write_structure_download_registry(
        registry, candidate_root=candidate, chain_attempts=attempts,
        evidence_rows=[
            {
                **attempts.iloc[0].to_dict(), "coordinate_cif_path": success_path,
                "coordinate_manifest_path": success_path.parent / "manifest.json",
            },
            {
                **attempts.iloc[1].to_dict(),
                "request_failure_path": captured.value.ledger_path,
            },
        ],
    )
    validate_structure_download_registry(
        registry, candidate_root=candidate, chain_attempt_path=attempts_path
    )
    success_path.write_bytes(success_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="(bytes changed|download SHA)"):
        validate_structure_download_registry(
            registry, candidate_root=candidate, chain_attempt_path=attempts_path
        )


def test_structure_smoke_source_transport_failure_retains_all_attempts(tmp_path):
    sleeps = []

    def offline_transport(*_args):
        raise ConnectionError("compute-node egress unavailable")

    output = tmp_path / "failed_source"
    with pytest.raises(RuntimeError, match="evidence="):
        freeze_rcsb_structure_smoke_source(
            output_dir=output, rcsb_entity_id="1AAA_1",
            transport=offline_transport, sleep_fn=sleeps.append,
        )
    failure_path = output / "source_failure.json"
    failure = json.loads(failure_path.read_text())
    unsigned = dict(failure)
    digest = unsigned.pop("failure_sha256")
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert len(failure["attempts"]) == 4
    assert all(item["http_status"] is None for item in failure["attempts"])
    assert all("ConnectionError" in item["error"] for item in failure["attempts"])
    assert sleeps == [1.0, 2.0, 4.0]


def test_structure_smoke_freezes_online_bytes_then_materializes_strictly_offline(tmp_path):
    sleeps = []

    def graphql_transport(_method, url, _body, _headers, _timeout):
        assert url.endswith("/graphql")
        return SiftsHttpResponse(200, _multicopy_graphql(), {"etag": "graphql-fixture"})

    def coordinate_request(url, _timeout):
        assert url.endswith("/1AAA.cif")
        return SiftsHttpResponse(200, _multicopy_coordinate_mmcif(), {"etag": "cif-fixture"})

    source_manifest = freeze_rcsb_structure_smoke_source(
        output_dir=tmp_path / "source", rcsb_entity_id="1AAA_1",
        transport=graphql_transport, coordinate_request_fn=coordinate_request,
        sleep_fn=sleeps.append,
    )
    assert sleeps == [0.1, 0.1]
    entity, edges, cif_path = validate_rcsb_structure_smoke_source(source_manifest)
    assert entity["rcsb_entity_id"] == "1AAA_1"
    assert [(row["label_asym_id"], row["auth_asym_id"]) for row in edges] == [
        ("A", "X"), ("C", "Y"),
    ]
    assert cif_path.read_bytes() == _multicopy_coordinate_mmcif()

    def load_coords(path, chain):
        assert chain == "A"
        records = read_mmcif_atom_records(path)
        positions = sorted({int(row["label_seq_id"]) for row in records})
        return [[[0.0, 0.0, 0.0]] * 4 for _ in positions], "A" * len(positions)

    probe_manifest = smoke_rcsb_multicopy_structure(
        output_dir=tmp_path / "probe", source_manifest=source_manifest,
        load_coords_fn=load_coords,
    )
    probe = json.loads(probe_manifest.read_text())
    assert [row["status"] for row in probe["results"]] == ["pass", "pass"]
    assert [row["sequence_length"] for row in probe["results"]] == [100, 100]

    original_probe = probe_manifest.read_bytes()
    probe["results"][0]["load_coords_evidence"]["loaded_length"] = 99
    unsigned = dict(probe)
    unsigned.pop("manifest_sha256")
    probe["manifest_sha256"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode()).hexdigest()
    probe_manifest.write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="load_coords replay"):
        validate_rcsb_structure_probe_manifest(
            probe_manifest, load_coords_fn=load_coords,
        )
    probe_manifest.write_bytes(original_probe)

    probe = json.loads(probe_manifest.read_text())
    clean = probe_manifest.parent / probe["results"][0]["clean_path"]
    original_clean = clean.read_bytes()
    clean.write_bytes(original_clean + b"# tamper\n")
    with pytest.raises(ValueError, match="clean-file identity"):
        validate_rcsb_structure_probe_manifest(
            probe_manifest, load_coords_fn=load_coords,
        )
    clean.write_bytes(original_clean)

    response = source_manifest.parent / "graphql_response.json"
    response.write_bytes(response.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file identity"):
        validate_rcsb_structure_smoke_source(source_manifest)


def test_truncated_chunked_response_is_retried_not_fatal(tmp_path):
    """`http.client.IncompleteRead` is an HTTPException, NOT an OSError, so the transport retry
    tuple missed it: one truncated chunked response out of the ~9,200 coordinate downloads a C6
    stage makes would abort the whole multi-hour stage. Over that many requests, at least one is
    effectively certain."""
    import http.client

    calls = []

    def request(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"partial-body")
        return SiftsHttpResponse(200, _mmcif(), {})

    cif_path, manifest = acquire_rcsb_mmcif(
        pdb_id="1AAA", expected_revision_date="2026-01-02", cache_dir=tmp_path,
        request_fn=request, sleep_fn=lambda _s: None,
    )
    assert len(calls) == 2
    assert cif_path.read_bytes() == _mmcif()
    ledger = manifest["attempts"]
    assert ledger[0]["http_status"] is None
    assert "IncompleteRead" in ledger[0]["error"]
    assert ledger[1]["http_status"] == 200
