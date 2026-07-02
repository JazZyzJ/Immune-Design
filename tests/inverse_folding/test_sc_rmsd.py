from pathlib import Path

import pytest

from inverse_folding.evaluation.sc_rmsd import (
    compute_ca_self_consistency,
    parse_ca_trace,
)


def _write_pdb(path: Path, coords: list[tuple[float, float, float]]) -> None:
    names = ["ALA", "CYS", "ASP", "GLU"]
    lines = []
    for idx, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {idx:5d}  CA  {names[idx - 1]:>3} A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_cif(path: Path, coords: list[tuple[float, float, float]]) -> None:
    names = ["ALA", "CYS", "ASP", "GLU"]
    lines = [
        "data_test\n",
        "#\n",
        "loop_\n",
        "_atom_site.group_PDB\n",
        "_atom_site.id\n",
        "_atom_site.type_symbol\n",
        "_atom_site.label_atom_id\n",
        "_atom_site.label_alt_id\n",
        "_atom_site.label_comp_id\n",
        "_atom_site.label_asym_id\n",
        "_atom_site.label_entity_id\n",
        "_atom_site.label_seq_id\n",
        "_atom_site.pdbx_PDB_ins_code\n",
        "_atom_site.Cartn_x\n",
        "_atom_site.Cartn_y\n",
        "_atom_site.Cartn_z\n",
        "_atom_site.occupancy\n",
        "_atom_site.B_iso_or_equiv\n",
        "_atom_site.auth_seq_id\n",
        "_atom_site.auth_asym_id\n",
    ]
    for idx, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM {idx} C CA . {names[idx - 1]} A 1 {idx} ? "
            f"{x:.3f} {y:.3f} {z:.3f} 1.00 50.00 {idx} A\n"
        )
    lines.append("#\n")
    path.write_text("".join(lines))


def test_parse_ca_trace_supports_pdb_and_mmcif(tmp_path: Path):
    coords = [(0.0, 0.0, 0.0), (1.5, 0.2, 0.0), (3.0, 0.1, 0.7)]
    pdb_path = tmp_path / "trace.pdb"
    cif_path = tmp_path / "trace.cif"
    _write_pdb(pdb_path, coords)
    _write_cif(cif_path, coords)

    pdb_trace = parse_ca_trace(pdb_path)
    cif_trace = parse_ca_trace(cif_path)

    assert [res.aa for res in pdb_trace] == ["A", "C", "D"]
    assert [res.aa for res in cif_trace] == ["A", "C", "D"]
    assert cif_trace[1].resseq == "2"
    assert cif_trace[1].chain_id == "A"


def test_compute_ca_self_consistency_emits_indexable_residue_rows(tmp_path: Path):
    ref_coords = [
        (0.0, 0.0, 0.0),
        (1.4, 0.4, 0.0),
        (2.6, -0.2, 0.8),
        (3.9, 0.3, 1.1),
    ]
    pred_coords = [(x + 10.0, y - 2.0, z + 5.0) for x, y, z in ref_coords]
    ref_path = tmp_path / "ref.cif"
    pred_path = tmp_path / "pred.pdb"
    _write_cif(ref_path, ref_coords)
    _write_pdb(pred_path, pred_coords)

    sc_rmsd, rows = compute_ca_self_consistency(
        pred_pdb=pred_path,
        ref_pdb=ref_path,
        protein_id="p1",
        design_id="design_0001",
        design_idx=1,
        design_sequence="ACEE",
        ref_sequence="ACDE",
        refold_backend="esmfold",
    )

    assert sc_rmsd == pytest.approx(0.0, abs=1e-6)
    assert len(rows) == 4
    assert rows[2]["residue_idx"] == 2
    assert rows[2]["residue_idx_1based"] == 3
    assert rows[2]["ref_aa"] == "D"
    assert rows[2]["design_aa"] == "E"
    assert rows[2]["sc_ca_distance"] == pytest.approx(0.0, abs=1e-6)
