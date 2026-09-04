"""C6 entity/chain and final-sequence mapping gates."""

from __future__ import annotations

import pytest

from inverse_folding.evaluation.if_benchmark_v3.mapping import (
    materialize_mmcif_chain,
    materialize_atom_records,
    read_mmcif_atom_records,
    validate_load_coords_result,
)


def _atoms(position, aa3="ALA", *, entity="2", label_chain="B", auth_chain="X"):
    return [
        {"entity_id": entity, "label_asym_id": label_chain, "auth_asym_id": auth_chain,
         "label_seq_id": position, "resname": aa3, "atom_name": atom,
         "x": float(position), "y": float(i), "z": 0.0, "occupancy": 1.0,
         "alt_id": "."}
        for i, atom in enumerate(("N", "CA", "C", "O"))
    ]


def _edge():
    return {"rcsb_entity_id": "1AAA_2", "label_asym_id": "B", "auth_asym_id": "X"}


def test_t2_coverage_denominator_is_full_entity_and_unresolved_deletion_can_pass():
    source = "A" * 10
    rows = sum((_atoms(i) for i in range(1, 9)), [])
    result = materialize_atom_records(
        rows, expected_edge=_edge(), source_sequence=source,
        observed_to_source={i: i - 1 for i in range(1, 11)}, denominator_positions=range(10),
        min_coverage=0.8,
    )
    assert result.sequence == "A" * 8
    assert result.if_sequence_coverage == pytest.approx(0.8)
    assert result.final_to_source == tuple(range(8))
    assert result.mapping_status == "complete"


def test_mse_to_m_is_only_nonstandard_residue_alias():
    rows = _atoms(1, "MSE")
    result = materialize_atom_records(
        rows, expected_edge=_edge(), source_sequence="M",
        observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
    )
    assert result.sequence == "M"
    with pytest.raises(ValueError, match="unsupported residue"):
        materialize_atom_records(
            _atoms(1, "SEC"), expected_edge=_edge(), source_sequence="C",
            observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
        )


def test_substitution_tag_and_wrong_entity_fail_closed():
    with pytest.raises(ValueError, match="substitution"):
        materialize_atom_records(
            _atoms(1, "GLY"), expected_edge=_edge(), source_sequence="A",
            observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
        )
    with pytest.raises(ValueError, match="unmapped observed residue"):
        materialize_atom_records(
            _atoms(1) + _atoms(2), expected_edge=_edge(), source_sequence="A",
            observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
        )
    with pytest.raises(ValueError, match="wrong entity-chain mapping"):
        materialize_atom_records(
            _atoms(1, entity="1"), expected_edge=_edge(), source_sequence="A",
            observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
        )


def test_incomplete_backbone_is_unresolved_and_low_coverage_fails():
    rows = _atoms(1)[:-1]
    with pytest.raises(ValueError, match="coverage"):
        materialize_atom_records(
            rows, expected_edge=_edge(), source_sequence="A",
            observed_to_source={1: 0}, denominator_positions=[0], min_coverage=0.8,
        )


def test_tier1_uses_explicit_sifts_position_map_and_frozen_range_denominator():
    # Observed positions 10..13 map onto canonical UniProt 100..103; one unresolved deletion.
    rows = _atoms(10) + _atoms(11) + _atoms(13)
    result = materialize_atom_records(
        rows, expected_edge=_edge(), source_sequence="A" * 200,
        observed_to_source={10: 100, 11: 101, 12: 102, 13: 103},
        denominator_positions=[100, 101, 102, 103], min_coverage=0.7,
    )
    assert result.sequence == "AAA"
    assert result.final_to_source == (100, 101, 103)
    assert result.if_sequence_coverage == pytest.approx(0.75)


def test_load_coords_must_return_exact_sequence_and_length():
    validate_load_coords_result(expected_sequence="ACD", loaded_sequence="ACD", loaded_length=3)
    with pytest.raises(ValueError, match="load_coords sequence"):
        validate_load_coords_result(expected_sequence="ACD", loaded_sequence="ACE", loaded_length=3)
    with pytest.raises(ValueError, match="load_coords length"):
        validate_load_coords_result(expected_sequence="ACD", loaded_sequence="ACD", loaded_length=2)


def _write_mmcif(path):
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
    lines = ["data_fixture", "#", "loop_", *fields]
    serial = 1
    for residue, resname in ((1, "ALA"), (2, "MSE")):
        for atom_idx, atom in enumerate(("N", "CA", "C", "O")):
            element = "C" if atom in {"CA", "C"} else atom
            lines.append(
                f"ATOM {serial} {element} {atom} . {resname} B 2 {residue} ? "
                f"{residue:.1f} {atom_idx:.1f} 0.0 1.0 10.0 ? {residue} "
                f"{resname} X {atom} 1"
            )
            serial += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def test_mmcif_adapter_preserves_entity_chain_and_load_coords_identity(tmp_path):
    source = tmp_path / "1AAA.cif"
    output = tmp_path / "clean.cif"
    _write_mmcif(source)
    records = read_mmcif_atom_records(source)
    assert {row["entity_id"] for row in records} == {"2"}
    assert {row["auth_asym_id"] for row in records} == {"X"}
    assert {row["auth_seq_id"] for row in records} == {1, 2}
    assert {row["insertion_code"] for row in records} == {""}

    def load_coords(path, chain):
        assert chain == "OUT"
        loaded = read_mmcif_atom_records(path)
        residues = {}
        for row in loaded:
            residues[int(row["label_seq_id"])] = row["resname"]
        sequence = "".join("M" if residues[index] == "MET" else "A" for index in sorted(residues))
        return [[[0.0, 0.0, 0.0]] * 4 for _ in residues], sequence

    result = materialize_mmcif_chain(
        source_mmcif=source, expected_edge=_edge(), source_sequence="AM",
        observed_to_source={1: 0, 2: 1}, denominator_positions=[0, 1],
        output_path=output, output_chain_id="OUT", load_coords_fn=load_coords,
    )
    assert result.mapping.sequence == "AM"
    assert result.load_coords_evidence["status"] == "pass"
    assert result.structure_sha256
    assert output.is_file()


def test_structure_adapter_forbids_identity_losing_pdb_input(tmp_path):
    path = tmp_path / "legacy.pdb"
    path.write_text("END\n")
    with pytest.raises(ValueError, match="requires mmCIF"):
        read_mmcif_atom_records(path)
