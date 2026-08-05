"""CLI integration tests for homolog-specific tetramer contact consensus."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.eval_tetramer_reference import main


def _write_tetramer(path):
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    rows = []
    for chain, y in zip("ABCD", (0.0, 3.0, 30.0, 33.0)):
        for res_id, res_name, x in ((10, "ALA", 0.0), (20, "CYS", 6.0)):
            rows.extend([
                ((x, y, 0.0), chain, res_id, res_name, "N", "N"),
                ((x + 1.0, y, 0.0), chain, res_id, res_name, "CA", "C"),
                ((x + 2.0, y, 0.0), chain, res_id, res_name, "C", "C"),
                ((x + 2.5, y, 0.0), chain, res_id, res_name, "O", "O"),
                ((x + 1.0, y + 1.0, 0.0), chain, res_id, res_name, "CB", "C"),
            ])
    arr = struc.AtomArray(len(rows))
    arr.coord = np.asarray([row[0] for row in rows], dtype=float)
    arr.chain_id = np.asarray([row[1] for row in rows])
    arr.res_id = np.asarray([row[2] for row in rows])
    arr.res_name = np.asarray([row[3] for row in rows])
    arr.atom_name = np.asarray([row[4] for row in rows])
    arr.element = np.asarray([row[5] for row in rows])
    arr.hetero = np.zeros(len(rows), dtype=bool)
    pdb = PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))


def test_consensus_cli_writes_exact_mapping_pairs_candidates_and_unresolved_masks(tmp_path):
    structure = tmp_path / "sample_0.pdb"
    _write_tetramer(structure)
    out_dir = tmp_path / "out"

    assert main([
        "--structure", str(structure),
        "--name", "P1_wt",
        "--protein-id", "P1",
        "--canonical-sequence", "MAC",
        "--out-dir", str(out_dir),
        "--sasa-point-number", "50",
    ]) == 0

    mapping = pd.read_parquet(out_dir / "P1_wt_residue_mapping.parquet")
    pairs = pd.read_parquet(out_dir / "P1_wt_contact_chain_pairs.parquet")
    candidates = pd.read_parquet(out_dir / "P1_wt_contact_candidates.parquet")
    contacts = pd.read_parquet(out_dir / "P1_wt_residue_pair_contacts.parquet")
    consensus = pd.read_parquet(out_dir / "P1_wt_contact_consensus.parquet")
    masks = json.loads((out_dir / "P1_wt_contact_masks.json").read_text())
    metadata = json.loads((out_dir / "P1_wt_contact_metadata.json").read_text())

    assert set(mapping["index_0b"]) == {1, 2}
    assert len(pairs) == 6
    assert set(pairs["bsa_rank_class"]) == {"interface_1", "interface_2", "diagonal"}
    assert set(pairs["mechanistic_class"]) == {"unresolved"}
    assert not candidates.empty
    assert {"within_chain", "tetramer_only"}.issubset(set(contacts["contact_kind"]))
    assert len(consensus) == 3 * len("MAC")
    assert masks["mechanism_status"] == "unresolved"
    assert masks["mask_semantics"]["symmetry_copy_requirement"] == "both copies"
    assert "is_biological_interface" not in pairs
    assert pairs["mechanistic_is_biological_interface"].isna().all()
    assert metadata["canonical_source"]["kind"] == "inline_sequence"
    assert len(metadata["implementation"]["entrypoint_sha256"]) == 64
    assert len(metadata["implementation"]["contact_library_sha256"]) == 64
    assert "--canonical-sequence MAC" in metadata["command"]


def test_consensus_cli_accepts_repeated_structures_and_external_annotations(tmp_path):
    structures = [tmp_path / f"sample_{sample}.pdb" for sample in range(2)]
    for structure in structures:
        _write_tetramer(structure)
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame([
        {"index_0b": 1, "aa": "A", "functional_role": "relay"},
        {"index_0b": 2, "aa": "C", "ddg_reu": 2.5},
    ]).to_csv(annotations, index=False)
    out_dir = tmp_path / "out"

    argv = [
        "--name", "P1_wt", "--protein-id", "P1", "--canonical-sequence", "MAC",
        "--out-dir", str(out_dir), "--sasa-point-number", "50",
        "--position-annotations", str(annotations),
        "--catalytic-pair", "A:B",
    ]
    for structure in structures:
        argv.extend(["--structure", str(structure)])
    assert main(argv) == 0

    consensus = pd.read_parquet(out_dir / "P1_wt_contact_consensus.parquet")
    pairs = pd.read_parquet(out_dir / "P1_wt_contact_chain_pairs.parquet")
    assert set(consensus["n_samples"]) == {2}
    assert "annotation_functional_role" in consensus
    assert "annotation_ddg_reu" in consensus
    # Annotations describe positions but cannot change a coordinate-derived mask.
    assert not consensus["mask_4ofN"].any()
    assert set(pairs["mechanistic_class"]) == {"catalytic", "assembly", "diagonal"}
    assert pairs["mechanistic_is_biological_interface"].notna().all()
    assert pairs.loc[
        pairs["mechanistic_class"] == "diagonal",
        "mechanistic_is_biological_interface",
    ].eq(False).all()

