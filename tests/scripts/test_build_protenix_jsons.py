"""TDD for the Protenix JSON sequences-block builder (multi-ligand holo support)."""

from __future__ import annotations

from scripts.build_protenix_jsons import LIGAND_SMILES, build_sequences_block

ZN = "[Zn+2]"
ADO = "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N"


def test_apo_has_no_ligand_entity():
    b = build_sequences_block("AAAA", 1, apo=True)
    assert b == [{"proteinChain": {"sequence": "AAAA", "count": 1}}]


def test_default_single_ligand_is_uric_acid_backward_compatible():
    b = build_sequences_block("AAAA", 4)
    assert b[1] == {"ligand": {"ligand": LIGAND_SMILES, "count": 4}}
    assert len(b) == 2


def test_multiple_ligands_become_separate_entities_in_order():
    # cofactor + substrate must be two DISTINCT entities, not one concatenated string
    b = build_sequences_block("AAAA", 1, ligands=[ZN, ADO])
    assert len(b) == 3
    assert [s["ligand"]["ligand"] for s in b[1:]] == [ZN, ADO]
    assert all(s["ligand"]["count"] == 1 for s in b[1:])


def test_ligand_copies_overrides_n_copies():
    b = build_sequences_block("AAAA", 4, ligands=[ZN], ligand_copies=1)
    assert b[0]["proteinChain"]["count"] == 4
    assert b[1]["ligand"]["count"] == 1


def test_msa_paths_attach_to_protein_chain_only():
    b = build_sequences_block("AAAA", 1, ligands=[ZN], paired="p.a3m", unpaired="np.a3m")
    assert b[0]["proteinChain"]["pairedMsaPath"] == "p.a3m"
    assert b[0]["proteinChain"]["unpairedMsaPath"] == "np.a3m"
    assert "pairedMsaPath" not in b[1]["ligand"]
