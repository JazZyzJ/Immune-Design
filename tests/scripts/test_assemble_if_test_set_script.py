import json

import pandas as pd

from scripts.assemble_if_test_set import (
    _load_tier2_from_checkpoints,
    _load_tier3,
    _resolve_pdb_relpath,
    _safe_file_id,
)


def test_load_tier2_from_checkpoints_reconstructs_entries(tmp_path):
    candidate_fasta = tmp_path / "tier2_candidates_merged.fasta"
    candidate_fasta.write_text(
        ">p1\nAAAA\n"
        ">p2\nBBBB\n"
        ">p3\nCCCC\n"
    )

    head_ckpt = tmp_path / "_prescreen_head_results.parquet"
    pd.DataFrame(
        [
            {"protein_id": "p1", "global_risk": 0.1, "n_hotspot_positions": 1},
            {"protein_id": "p2", "global_risk": 0.8, "n_hotspot_positions": 4},
            {"protein_id": "p3", "global_risk": 0.9, "n_hotspot_positions": 5},
        ]
    ).to_parquet(head_ckpt, index=False)

    nmp_ckpt = tmp_path / "_prescreen_nmp_results.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "p1",
                "n_strong_windows": 8,
                "mean_best_rank": 0.5,
                "nmp_status": "complete",
            },
            {
                "protein_id": "p2",
                "n_strong_windows": 7,
                "mean_best_rank": 0.6,
                "nmp_status": "partial",
            },
            {
                "protein_id": "p3",
                "n_strong_windows": 6,
                "mean_best_rank": 0.7,
                "nmp_status": "complete",
            },
        ]
    ).to_parquet(nmp_ckpt, index=False)

    entries = _load_tier2_from_checkpoints(
        candidate_fasta_path=str(candidate_fasta),
        head_ckpt_path=str(head_ckpt),
        nmp_ckpt_path=str(nmp_ckpt),
        min_strong_windows=5,
    )

    assert [e["protein_id"] for e in entries] == ["p3"]
    assert entries[0]["sequence"] == "CCCC"
    assert entries[0]["netmhciipan_n_strong"] == 6
    assert entries[0]["head_global_risk"] == 0.9
    assert entries[0]["cath_topology"] is None


def test_load_tier2_from_checkpoints_accepts_legacy_complete_rows(tmp_path):
    candidate_fasta = tmp_path / "tier2_candidates_merged.fasta"
    candidate_fasta.write_text(">p1\nAAAA\n")

    head_ckpt = tmp_path / "_prescreen_head_results.parquet"
    pd.DataFrame(
        [{"protein_id": "p1", "global_risk": 0.7, "n_hotspot_positions": 3}]
    ).to_parquet(head_ckpt, index=False)

    nmp_ckpt = tmp_path / "_prescreen_nmp_results.parquet"
    pd.DataFrame(
        [{"protein_id": "p1", "n_strong_windows": 5, "mean_best_rank": 0.4}]
    ).to_parquet(nmp_ckpt, index=False)

    entries = _load_tier2_from_checkpoints(
        candidate_fasta_path=str(candidate_fasta),
        head_ckpt_path=str(head_ckpt),
        nmp_ckpt_path=str(nmp_ckpt),
        min_strong_windows=5,
    )

    assert len(entries) == 1
    assert entries[0]["protein_id"] == "p1"


def test_load_tier3_accepts_uricase_prescreen_parquet(tmp_path):
    parquet_path = tmp_path / "uricase_dual_pass_HLA-DRB1_07_01.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "uox_1",
                "sequence": "AAAA",
                "sequence_length": 4,
                "netmhciipan_n_strong": 7,
                "netmhciipan_mean_best_rank": 0.8,
                "head_global_risk": 0.6,
                "head_n_hotspot": 3,
            }
        ]
    ).to_parquet(parquet_path, index=False)

    entries = _load_tier3(str(parquet_path), pdb_dir=str(tmp_path))

    assert len(entries) == 1
    assert entries[0]["protein_id"] == "uox_1"
    assert entries[0]["tier"] == 3
    assert entries[0]["sequence"] == "AAAA"
    assert entries[0]["netmhciipan_n_strong"] == 7
    assert entries[0]["head_global_risk"] == 0.6


def test_safe_file_id_replaces_path_unsafe_characters():
    raw = "D3BGR1|Heterostelium_pallidum_(strain_ATCC_26659_/_Pp_5_/_PN500)|Uricase|taxid:670386|len:222"
    safe = _safe_file_id(raw)

    assert "/" not in safe
    assert "|" not in safe
    assert ":" not in safe
    assert safe.startswith("D3BGR1_")


def test_load_tier3_uses_safe_pdb_path_for_parquet_input(tmp_path):
    parquet_path = tmp_path / "uricase_dual_pass_HLA-DRB1_07_01.parquet"
    raw_id = "D3BGR1|Heterostelium_pallidum_(strain_ATCC_26659_/_Pp_5_/_PN500)|Uricase|taxid:670386|len:222"
    pd.DataFrame(
        [
            {
                "protein_id": raw_id,
                "sequence": "AAAA",
                "sequence_length": 4,
                "netmhciipan_n_strong": 7,
                "netmhciipan_mean_best_rank": 0.8,
                "head_global_risk": 0.6,
                "head_n_hotspot": 3,
            }
        ]
    ).to_parquet(parquet_path, index=False)

    entries = _load_tier3(str(parquet_path), pdb_dir=str(tmp_path))

    assert entries[0]["protein_id"] == raw_id
    assert entries[0]["pdb_path"].startswith("pdbs/")
    assert "|" not in entries[0]["pdb_path"]
    assert ":" not in entries[0]["pdb_path"]
    assert "_Pp_5_" in entries[0]["pdb_path"]


# --- pdb_path must name a file that exists, not a guessed <id>.pdb -----------------


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ATOM\n")
    return p


def test_resolve_pdb_relpath_finds_flat_pdb(tmp_path):
    _touch(tmp_path, "1ABC_A.pdb")
    assert _resolve_pdb_relpath("1ABC_A", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/1ABC_A.pdb"


def test_resolve_pdb_relpath_finds_short_allele_subdir(tmp_path):
    """Structures added after the bulk download live in a per-allele subdir."""
    _touch(tmp_path, "0401/1ABC_A.pdb")
    assert _resolve_pdb_relpath("1ABC_A", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/0401/1ABC_A.pdb"


def test_resolve_pdb_relpath_finds_cif_in_allele_subdir(tmp_path):
    """The 6TN1_AAA case: the structure only ever existed as .cif."""
    _touch(tmp_path, "0401/6TN1_AAA.cif")
    assert _resolve_pdb_relpath("6TN1_AAA", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/0401/6TN1_AAA.cif"


def test_resolve_pdb_relpath_finds_long_allele_subdir(tmp_path):
    _touch(tmp_path, "HLA-DRB1_15_01/2XYZ_B.cif")
    assert (
        _resolve_pdb_relpath("2XYZ_B", str(tmp_path), "HLA-DRB1_15_01")
        == "pdbs/HLA-DRB1_15_01/2XYZ_B.cif"
    )


def test_resolve_pdb_relpath_prefers_flat_pdb_over_subdir_cif(tmp_path):
    _touch(tmp_path, "1ABC_A.pdb")
    _touch(tmp_path, "0401/1ABC_A.cif")
    assert _resolve_pdb_relpath("1ABC_A", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/1ABC_A.pdb"


def test_resolve_pdb_relpath_falls_back_to_legacy_guess_when_absent(tmp_path):
    """The assembler must still run before structures are downloaded."""
    assert _resolve_pdb_relpath("1ABC_A", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/1ABC_A.pdb"


def test_resolve_pdb_relpath_without_pdb_dir_keeps_legacy_behaviour(tmp_path):
    assert _resolve_pdb_relpath("weird/id", None, None) == f"pdbs/{_safe_file_id('weird/id')}.pdb"


def test_resolve_pdb_relpath_sanitizes_protein_id(tmp_path):
    _touch(tmp_path, "0401/weird_id.cif")
    assert _resolve_pdb_relpath("weird/id", str(tmp_path), "HLA-DRB1_04_01") == "pdbs/0401/weird_id.cif"


def test_load_tier3_emits_resolved_cif_path(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    _touch(pdb_dir, "0401/6TN1_AAA.cif")
    src = tmp_path / "tier3.json"
    src.write_text(json.dumps([{"protein_id": "6TN1_AAA", "sequence": "AAAA"}]))
    entries = _load_tier3(str(src), str(pdb_dir), allele_tag="HLA-DRB1_04_01")
    assert [e["pdb_path"] for e in entries] == ["pdbs/0401/6TN1_AAA.cif"]
