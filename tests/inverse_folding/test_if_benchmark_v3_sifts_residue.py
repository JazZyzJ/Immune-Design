"""Residue-level PDBe SIFTS snapshot and RCSB coordinate cross-checks."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from inverse_folding.evaluation.if_benchmark_v3.preflight import SiftsHttpResponse
from inverse_folding.evaluation.if_benchmark_v3.sifts_residue import (
    Tier1CapacityError,
    frozen_revision_map,
    parse_pdbe_enriched_residue_maps,
    snapshot_tier1_residue_maps,
    validate_tier1_residue_map_snapshot,
)
from inverse_folding.evaluation.if_benchmark_v3.structures import acquire_rcsb_mmcif


CANONICAL = "G" * 10 + "ACD" + "G" * 20


FROZEN_REVISION = "2026-01-02"


def _identity_prefix(*, revision: str | None = None) -> list[str]:
    """`revision=None` reproduces the PRODUCTION PDBe enriched file, which carries no
    `_pdbx_audit_revision_history` loop at all. Only RCSB coordinate files carry one."""
    head = ["data_1AAA", "_entry.id 1AAA"]
    if revision is not None:
        head += [
            "loop_",
            "_pdbx_audit_revision_history.ordinal",
            "_pdbx_audit_revision_history.revision_date",
            f"1 {revision}", "#",
        ]
    return head + [
        "loop_", "_struct_asym.id",
        "_struct_asym.entity_id", "B 2", "C 2", "#", "loop_",
        "_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.entity_id",
        "_pdbx_poly_seq_scheme.seq_id", "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.auth_seq_num", "_pdbx_poly_seq_scheme.pdb_ins_code",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
        "B 2 1 ALA 100 ? X", "B 2 2 CYS 100 A X", "B 2 3 ASP 102 ? X",
        "B 2 4 GLY ? ? X",
        "C 2 1 ALA 100 ? X", "C 2 2 CYS 100 A X", "C 2 3 ASP 102 ? X",
        "C 2 4 GLY ? ? X", "#",
    ]


def _enriched_cif(*, source_id: str = "P12345", omit_unp_12: bool = False) -> bytes:
    lines = _identity_prefix() + [
        "loop_", "_pdbx_sifts_xref_db.entity_id", "_pdbx_sifts_xref_db.asym_id",
        "_pdbx_sifts_xref_db.seq_id", "_pdbx_sifts_xref_db.mon_id_one_letter_code",
        "_pdbx_sifts_xref_db.unp_res", "_pdbx_sifts_xref_db.unp_num",
        "_pdbx_sifts_xref_db.unp_acc", "_pdbx_sifts_xref_db.observed",
    ]
    for asym in ("B", "C"):
        for seq_id, aa, unp_num in ((1, "A", 11), (2, "C", 12), (3, "D", 13)):
            if omit_unp_12 and unp_num == 12:
                continue
            lines.append(f"2 {asym} {seq_id} {aa} {aa} {unp_num} {source_id} Y")
        lines.append(f"2 {asym} 4 G G 14 {source_id} N")
    lines.append("#")
    return ("\n".join(lines) + "\n").encode()


def _coordinate_cif(*, auth_chain: str = "X", atom_auth_chain: str | None = None,
                    revision: str = FROZEN_REVISION) -> bytes:
    prefix = _identity_prefix(revision=revision)
    if auth_chain != "X":
        prefix = [line.replace(" ? X", f" ? {auth_chain}").replace(" A X", f" A {auth_chain}")
                  for line in prefix]
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
    lines = prefix + ["loop_", *fields]
    atom_auth_chain = auth_chain if atom_auth_chain is None else atom_auth_chain
    serial = 1
    for asym in ("B", "C"):
        for seq_id, resname, auth_num, ins in (
            (1, "ALA", 100, "?"), (2, "CYS", 100, "A"), (3, "ASP", 102, "?"),
        ):
            for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                element = "C" if atom in {"CA", "C"} else atom
                lines.append(
                    f"ATOM {serial} {element} {atom} . {resname} {asym} 2 {seq_id} {ins} "
                    f"{seq_id}.0 {atom_index}.0 0.0 1.0 10.0 ? {auth_num} {resname} "
                    f"{atom_auth_chain} {atom} 1"
                )
                serial += 1
    lines.append("#")
    return ("\n".join(lines) + "\n").encode()


def test_enriched_sifts_parser_preserves_label_auth_insertion_and_multicopy(tmp_path):
    path = tmp_path / "1aaa_updated.cif"
    path.write_bytes(_enriched_cif())
    candidates = parse_pdbe_enriched_residue_maps(
        path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
        canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
        source_revision_date=FROZEN_REVISION,
    )
    assert [(row["label_asym_id"], row["rcsb_entity_id"]) for row in candidates] == [
        ("B", "1AAA_2"), ("C", "1AAA_2"),
    ]
    residue_map = candidates[0]["residue_map"]
    assert residue_map[10]["chain_local_index_0b"] == 0
    assert residue_map[11] == {
        "label_seq_id": 2, "auth_seq_id": 100, "insertion_code": "A",
        "chain_local_index_0b": 1, "amino_acid": "C", "observed": True,
    }


def test_enriched_sifts_parser_does_not_interpolate_internal_mapping_gap(tmp_path):
    path = tmp_path / "1aaa_updated.cif"
    path.write_bytes(_enriched_cif(omit_unp_12=True))
    candidates = parse_pdbe_enriched_residue_maps(
        path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
        canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
        source_revision_date=FROZEN_REVISION,
    )
    assert set(candidates[0]["residue_map"]) == {10, 12}


def test_unobserved_sifts_residue_keeps_label_frame_without_fake_author_number(tmp_path):
    path = tmp_path / "1aaa_updated.cif"
    path.write_bytes(_enriched_cif())
    candidates = parse_pdbe_enriched_residue_maps(
        path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
        canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 14),
        source_revision_date=FROZEN_REVISION,
    )
    assert candidates[0]["residue_map"][13] == {
        "label_seq_id": 4, "auth_seq_id": None, "insertion_code": "",
        "chain_local_index_0b": 3, "amino_acid": "G", "observed": False,
    }


@pytest.mark.parametrize("tamper", ["uniprot", "amino_acid", "auth_chain"])
def test_enriched_sifts_parser_rejects_wrong_source_or_residue_identity(tmp_path, tamper):
    path = tmp_path / "1aaa_updated.cif"
    body = _enriched_cif(source_id="Q99999" if tamper == "uniprot" else "P12345")
    if tamper == "amino_acid":
        body = body.replace(b"2 B 2 C C 12", b"2 B 2 C A 12")
    if tamper == "auth_chain":
        body = body.replace(b" ? X", b" ? Y").replace(b" A X", b" A Y")
    path.write_bytes(body)
    with pytest.raises(ValueError, match="(no exact|amino-acid|auth chain)"):
        parse_pdbe_enriched_residue_maps(
            path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
            canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
            source_revision_date=FROZEN_REVISION,
        )


def _snapshot(tmp_path, *, coordinate_auth="X", coordinate_atom_chain=None,
              frozen_revisions=None, coordinate_revision=FROZEN_REVISION,
              min_tier1_source_units=None):
    attempts = pd.DataFrame([{
        "source_id": "P12345", "pdb_id": "1AAA", "auth_chain_id": "X",
        "unp_start": 11, "unp_end": 13, "sifts_item_index": 0,
        "source_valid": True, "source_sequence_sha256": hashlib.sha256(
            CANONICAL.encode()
        ).hexdigest(),
    }])
    sources = pd.DataFrame([{
        "source_id": "P12345", "canonical_sequence": CANONICAL,
        "canonical_sequence_sha256": hashlib.sha256(CANONICAL.encode()).hexdigest(),
    }])

    def coordinate_provider(pdb_id, revision):
        path, _manifest = acquire_rcsb_mmcif(
            pdb_id=pdb_id, expected_revision_date=revision,
            cache_dir=tmp_path / "candidate" / "audit" / "coordinates",
            request_fn=lambda _url, _timeout: SiftsHttpResponse(
                200, _coordinate_cif(
                    auth_chain=coordinate_auth, atom_auth_chain=coordinate_atom_chain,
                    revision=coordinate_revision,
                ), {}
            ),
        )
        return path, path.parent / "manifest.json"

    manifest = snapshot_tier1_residue_maps(
        source_attempts=attempts, canonical_sources=sources,
        output_dir=tmp_path / "candidate" / "audit" / "tier1_residue_maps",
        candidate_root=tmp_path / "candidate",
        request_fn=lambda _url, _timeout: SiftsHttpResponse(200, _enriched_cif(), {}),
        sleep_fn=lambda _seconds: None, coordinate_provider=coordinate_provider,
        frozen_revisions=(
            {"1AAA": FROZEN_REVISION} if frozen_revisions is None else frozen_revisions
        ),
        min_tier1_source_units=min_tier1_source_units,
        command_argv=["fixture-residue-producer"],
    )
    return manifest


def test_residue_snapshot_replays_raw_enriched_and_rcsb_coordinate_bytes(tmp_path):
    manifest = _snapshot(tmp_path)
    result = validate_tier1_residue_map_snapshot(manifest)
    assert result["candidate_count"] == 2
    root = manifest.parent
    candidates = pd.read_parquet(root / "tier1_residue_map_candidates.parquet")
    assert set(candidates["label_asym_id"]) == {"B", "C"}
    payload = json.loads(manifest.read_text())
    assert payload["raw_file_merkle_sha256"]
    assert payload["coordinate_file_merkle_sha256"]


def test_residue_snapshot_rejects_raw_or_coordinate_tamper_and_chain_mismatch(tmp_path):
    manifest = _snapshot(tmp_path)
    payload = json.loads(manifest.read_text())
    raw_path = manifest.parent / payload["raw_files"][0]["path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"#tamper\n")
    with pytest.raises(ValueError, match="raw file table"):
        validate_tier1_residue_map_snapshot(manifest)


def _coordinate_disagreement_failure(manifest):
    payload = json.loads(manifest.read_text())
    failures = pd.read_parquet(manifest.parent / "tier1_residue_map_failures.parquet")
    assert payload["counts"]["residue_map_candidates"] == 0
    assert list(failures["failure_code"]) == ["coordinate_identity_mismatch"]
    # A content rejection, not a fetch failure: the request succeeded and its bytes are on disk,
    # so `release_blocked` (unresolved SOURCE REQUESTS) must stay False -- the same semantics the
    # amino-acid mismatch already has.
    assert payload["release_blocked"] is False
    assert payload["counts"]["request_success"] == 1
    assert payload["counts"]["malformed_or_coordinate_failure"] == 0
    return str(failures.loc[0, "failure_detail"])


@pytest.mark.parametrize("kwargs,expected", [
    ({"coordinate_auth": "Y"}, "coordinate chain identity"),
    ({"coordinate_atom_chain": "Y"}, "coordinate atom identity"),
])
def test_pdbe_rcsb_coordinate_disagreement_is_typed_per_attempt_not_a_stage_crash(
    tmp_path, kwargs, expected,
):
    """PDBe and RCSB disagreeing about one chain is a database disagreement, in the same class as
    the amino-acid mismatch that is already typed. Raising here would discard a multi-hour login
    stage's completed downloads because a single entry of ~1,650 disagrees."""
    manifest = _snapshot(tmp_path, **kwargs)
    assert expected in _coordinate_disagreement_failure(manifest)


def test_a_coordinate_disagreement_rejects_the_whole_attempt_not_just_one_chain_copy(tmp_path):
    """The failure table is keyed per attempt, so an attempt is admitted all-or-nothing."""
    # the fixture's enriched file carries two chain copies (B and C) of one entity
    ok = _snapshot(tmp_path / "clean")
    assert json.loads(ok.read_text())["counts"]["residue_map_candidates"] == 2
    bad = _snapshot(tmp_path / "one-bad", coordinate_atom_chain="Y")
    assert json.loads(bad.read_text())["counts"]["residue_map_candidates"] == 0


def test_residue_snapshot_persists_terminal_http_failure_before_blocking_release(tmp_path):
    attempts = pd.DataFrame([{
        "source_id": "P12345", "pdb_id": "1AAA", "auth_chain_id": "X",
        "unp_start": 11, "unp_end": 13, "sifts_item_index": 0,
        "source_valid": True,
        "source_sequence_sha256": hashlib.sha256(CANONICAL.encode()).hexdigest(),
    }])
    sources = pd.DataFrame([{
        "source_id": "P12345", "canonical_sequence": CANONICAL,
        "canonical_sequence_sha256": hashlib.sha256(CANONICAL.encode()).hexdigest(),
    }])
    manifest = snapshot_tier1_residue_maps(
        source_attempts=attempts, canonical_sources=sources,
        output_dir=tmp_path / "candidate" / "audit" / "tier1_residue_maps",
        candidate_root=tmp_path / "candidate",
        request_fn=lambda _url, _timeout: SiftsHttpResponse(404, b"not-found", {}),
        sleep_fn=lambda _seconds: None,
        coordinate_provider=lambda *_args: pytest.fail("404 must not request coordinates"),
        frozen_revisions={"1AAA": FROZEN_REVISION},
        command_argv=["fixture-failure"],
    )
    payload = json.loads(manifest.read_text())
    assert payload["counts"]["terminal_http_failure"] == 1
    assert payload["release_blocked"] is True
    requests = pd.read_parquet(manifest.parent / "requests.parquet")
    assert requests.loc[0, "http_status"] == 404
    assert requests.loc[0, "response_sha256"] == hashlib.sha256(b"not-found").hexdigest()
    assert len(pd.read_parquet(manifest.parent / "tier1_residue_map_failures.parquet")) == 1
    assert validate_tier1_residue_map_snapshot(
        manifest, require_complete=False
    )["failure_count"] == 1
    with pytest.raises(ValueError, match="unresolved source requests"):
        validate_tier1_residue_map_snapshot(manifest)


def test_residue_snapshot_throttles_two_unique_pdbe_requests_and_records_delay(tmp_path):
    attempts = pd.DataFrame([
        {
            "source_id": "P12345", "pdb_id": pdb_id, "auth_chain_id": "X",
            "unp_start": 11, "unp_end": 13, "sifts_item_index": index,
            "source_valid": True,
            "source_sequence_sha256": hashlib.sha256(CANONICAL.encode()).hexdigest(),
        }
        for index, pdb_id in enumerate(("1AAA", "1AAB"))
    ])
    sources = pd.DataFrame([{
        "source_id": "P12345", "canonical_sequence": CANONICAL,
        "canonical_sequence_sha256": hashlib.sha256(CANONICAL.encode()).hexdigest(),
    }])
    sleeps = []

    def body_for(pdb_id, *, coordinate):
        body = _coordinate_cif() if coordinate else _enriched_cif()
        return body.replace(b"1AAA", pdb_id.encode())

    def coordinate_provider(pdb_id, revision):
        path, _manifest = acquire_rcsb_mmcif(
            pdb_id=pdb_id, expected_revision_date=revision,
            cache_dir=tmp_path / "candidate" / "audit" / "coordinates",
            request_fn=lambda _url, _timeout: SiftsHttpResponse(
                200, body_for(pdb_id, coordinate=True), {}
            ),
            sleep_fn=lambda _seconds: None, api_delay_s=0.0,
        )
        return path, path.parent / "manifest.json"

    manifest = snapshot_tier1_residue_maps(
        source_attempts=attempts, canonical_sources=sources,
        output_dir=tmp_path / "candidate" / "audit" / "tier1_residue_maps",
        candidate_root=tmp_path / "candidate", coordinate_provider=coordinate_provider,
        request_fn=lambda url, _timeout: SiftsHttpResponse(
            200, body_for("1AAA" if "1aaa_" in url else "1AAB", coordinate=False), {}
        ),
        sleep_fn=sleeps.append, api_delay_s=0.1,
        frozen_revisions={"1AAA": FROZEN_REVISION, "1AAB": FROZEN_REVISION},
        command_argv=["fixture-throttle"],
    )
    assert sleeps == [0.1]
    assert json.loads(manifest.read_text())["api_delay_s"] == 0.1


# ------------------------------------------------------------ frozen-revision contract (C8)
#
# Production PDBe enriched mmCIF carries no `_pdbx_audit_revision_history`, so deriving the
# coordinate revision from it rejected all 2,864 DRB1*15:01 Tier 1 attempts and sealed an empty
# C8. The revision authority is the frozen RCSB source snapshot; PDBe supplies residue mapping
# and entry identity only.


def test_parser_accepts_production_pdbe_file_that_has_no_revision_history(tmp_path):
    path = tmp_path / "1aaa_updated.cif"
    path.write_bytes(_enriched_cif())
    assert b"_pdbx_audit_revision_history" not in path.read_bytes()
    candidates = parse_pdbe_enriched_residue_maps(
        path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
        canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
        source_revision_date=FROZEN_REVISION,
    )
    assert candidates
    assert {row["source_revision_date"] for row in candidates} == {FROZEN_REVISION}


def test_parser_still_rejects_a_pdbe_file_for_the_wrong_entry(tmp_path):
    path = tmp_path / "other.cif"
    path.write_bytes(_enriched_cif().replace(b"_entry.id 1AAA", b"_entry.id 9ZZZ"))
    with pytest.raises(ValueError, match="entry identity"):
        parse_pdbe_enriched_residue_maps(
            path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
            canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
            source_revision_date=FROZEN_REVISION,
        )


def test_parser_rejects_a_malformed_frozen_revision(tmp_path):
    path = tmp_path / "1aaa_updated.cif"
    path.write_bytes(_enriched_cif())
    with pytest.raises(ValueError, match="revision date"):
        parse_pdbe_enriched_residue_maps(
            path, pdb_id="1AAA", source_uniprot_id="P12345", auth_chain_id="X",
            canonical_sequence=CANONICAL, frozen_uniprot_range=(11, 13),
            source_revision_date="not-a-date",
        )


def test_snapshot_fails_closed_when_rcsb_coordinates_disagree_with_the_frozen_revision(tmp_path):
    """The frozen source is the authority; a coordinate file at another revision is a failure."""
    manifest = _snapshot(tmp_path, coordinate_revision="2020-05-05")
    payload = json.loads(manifest.read_text())
    assert payload["counts"]["residue_map_candidates"] == 0
    assert payload["release_blocked"] is True
    failures = pd.read_parquet(manifest.parent / "tier1_residue_map_failures.parquet")
    assert len(failures) == 1
    assert "revision" in failures.loc[0, "failure_detail"]


def test_snapshot_records_a_pdb_with_no_frozen_revision_as_a_typed_failure(tmp_path):
    manifest = _snapshot(tmp_path, frozen_revisions={})
    payload = json.loads(manifest.read_text())
    assert payload["counts"]["residue_map_candidates"] == 0
    assert payload["release_blocked"] is True
    failures = pd.read_parquet(manifest.parent / "tier1_residue_map_failures.parquet")
    assert "frozen source revision" in failures.loc[0, "failure_detail"]


def test_frozen_revision_map_normalizes_and_rejects_mixed_revisions_for_one_pdb():
    frame = pd.DataFrame([
        {"pdb_id": "1aaa", "structure_revision_date": "2026-01-02"},
        {"pdb_id": "1AAA", "structure_revision_date": "2026-01-02"},
        {"pdb_id": "1bbb", "structure_revision_date": "2025-03-04"},
    ])
    assert frozen_revision_map(frame) == {"1AAA": "2026-01-02", "1BBB": "2025-03-04"}

    mixed = pd.DataFrame([
        {"pdb_id": "1AAA", "structure_revision_date": "2026-01-02"},
        {"pdb_id": "1AAA", "structure_revision_date": "2024-09-09"},
    ])
    with pytest.raises(ValueError, match="conflicting"):
        frozen_revision_map(mixed)


def test_frozen_revision_map_normalizes_the_iso_timestamp_rcsb_actually_stores():
    """C1 `entities_all.parquet` stores `2026-08-12T00:00:00Z`, not a bare date."""
    frame = pd.DataFrame([{"pdb_id": "1AAA", "structure_revision_date": "2026-08-12T00:00:00Z"}])
    assert frozen_revision_map(frame) == {"1AAA": "2026-08-12"}


def test_frozen_revision_map_rejects_a_missing_or_invalid_date():
    with pytest.raises(ValueError, match="revision date"):
        frozen_revision_map(pd.DataFrame([
            {"pdb_id": "1AAA", "structure_revision_date": ""},
        ]))
    with pytest.raises(ValueError, match="revision date"):
        frozen_revision_map(pd.DataFrame([
            {"pdb_id": "1AAA", "structure_revision_date": "2026-13-45"},
        ]))


def test_snapshot_writes_its_audit_trail_then_fails_the_stage_below_tier1_capacity(tmp_path):
    """A capacity failure must not be silent AND must not destroy the diagnostic artifacts."""
    with pytest.raises(Tier1CapacityError, match="Tier 1 capacity"):
        _snapshot(tmp_path, min_tier1_source_units=15)
    manifest = (
        tmp_path / "candidate" / "audit" / "tier1_residue_maps" / "manifest.json"
    )
    assert manifest.is_file()
    payload = json.loads(manifest.read_text())
    assert payload["counts"]["viable_tier1_source_units"] == 1
    assert payload["min_tier1_source_units"] == 15
    assert payload["release_blocked"] is True


def test_snapshot_passes_the_capacity_gate_when_the_target_is_met(tmp_path):
    manifest = _snapshot(tmp_path, min_tier1_source_units=1)
    payload = json.loads(manifest.read_text())
    assert payload["counts"]["viable_tier1_source_units"] == 1
    assert payload["counts"]["residue_map_candidates"] == 2
    assert payload["release_blocked"] is False
