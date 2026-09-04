from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import freeze_uricase_ev_structure_inputs as freeze


SEQUENCE_1 = "ACDEFGHIKLMNPQRSTVWY"
SEQUENCE_2 = "ACDEFGHIKLMNPQRSTVWA"


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    cohort = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "sequence": [SEQUENCE_1, SEQUENCE_2],
            "sequence_length": [20, 20],
            "if_ready": [True, True],
            "pdb_path": ["/pdb/P1.pdb", "/pdb/P2.pdb"],
            "structure_source": ["AFDB", "ESMFold2"],
            "characterized": [False, True],
        }
    )
    summary = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "design_id": ["design_0000", "design_0000"],
            "design_idx": [0, 0],
            "n_strong_binders": [2, 0],
            "n_weak_binders": [2, 1],
            "mean_best_rank": [0.5, 4.0],
            "n_windows_scored": [3, 1],
        }
    )
    peptides = pd.DataFrame(
        {
            "protein_id": ["P1", "P1", "P1", "P2"],
            "design_id": ["design_0000"] * 4,
            "design_idx": [0] * 4,
            "pep_length": [15, 15, 15, 15],
            "pos": [0, 1, 0, 0],
            "peptide": [
                SEQUENCE_1[:15],
                SEQUENCE_1[1:16],
                SEQUENCE_1[:15],
                SEQUENCE_2[:15],
            ],
            "core": ["EFGHIKLMN", "EFGHIKLMN", "EFGHIKLMN", "ACDEFGHIK"],
            "rank_EL": [0.5, 1.5, 2.0, 8.0],
            "el_score": [0.9, 0.8, 0.7, 0.1],
        }
    )
    role_positions = {
        "Lys11": 8,
        "Thr58": 16,
        "Asp59": 2,
        "Phe160": 4,
        "Arg177": 14,
        "Gln229": 13,
        "Asn255": 11,
        "His257": 6,
        "Val228": 17,
    }
    projection_rows = []
    for protein_id, sequence in [("P1", SEQUENCE_1), ("P2", SEQUENCE_2)]:
        for role in freeze.EXPECTED_Q00511_ROLES:
            target_index = role_positions[role.label]
            target_aa = sequence[target_index]
            projection_rows.append(
                {
                    "protein_id": protein_id,
                    "kind": role.kind,
                    "ref_index_0b": role.ref_index_0b,
                    "label": role.label,
                    "expected_aa": role.expected_aa,
                    "target_index_0b": float(target_index),
                    "target_aa": target_aa,
                    "matched": target_aa == role.expected_aa,
                }
            )

    paths = {
        "cohort": tmp_path / "cohort.parquet",
        "summary": tmp_path / "imm_nmp.parquet",
        "peptides": tmp_path / "imm_nmp_peptides.parquet",
        "projection": tmp_path / "projection.parquet",
        "failures": tmp_path / "failures.json",
    }
    cohort.to_parquet(paths["cohort"], index=False)
    summary.to_parquet(paths["summary"], index=False)
    peptides.to_parquet(paths["peptides"], index=False)
    pd.DataFrame(projection_rows).to_parquet(paths["projection"], index=False)
    paths["failures"].write_text(json.dumps({"failures": []}))
    return paths


def _args(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--cohort-parquet",
        str(paths["cohort"]),
        "--nmp-summary",
        str(paths["summary"]),
        "--nmp-peptides",
        str(paths["peptides"]),
        "--nmp-failures",
        str(paths["failures"]),
        "--q00511-projection-audit",
        str(paths["projection"]),
        "--allele",
        "HLA-DRB1*15:01",
        "--output-dir",
        str(output),
        "--n-shards",
        "2",
        "--expected-parent-count",
        "2",
        "--expected-clean-count",
        "1",
        "--expected-nonclean-count",
        "1",
        "--expected-all-hard-mapped-count",
        "2",
    ]


def test_freeze_outputs_exact_core_and_keeps_nonidentity_prior(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out = tmp_path / "frozen"

    assert freeze.main(_args(paths, out)) == 0

    cohort = pd.read_parquet(out / "cohort.parquet")
    assert cohort.protein_id.tolist() == ["P1", "P2"]
    assert cohort.set_index("protein_id").wt_nmp_clean.to_dict() == {
        "P1": False,
        "P2": True,
    }
    assert cohort.set_index("protein_id").q00511_all_hard_roles_identity.to_dict() == {
        "P1": True,
        "P2": True,
    }

    cores = pd.read_parquet(out / "wt_strong_cores.parquet")
    assert cores[["protein_id", "core_start_0b", "core_seq"]].to_dict("records") == [
        {"protein_id": "P1", "core_start_0b": 3, "core_seq": "EFGHIKLMN"}
    ]
    assert cores.iloc[0].n_supporting_windows == 2
    assert cores.iloc[0].best_rank_EL == pytest.approx(0.5)
    assert (out / "clean.ids").read_text() == "P2\n"
    assert (out / "nonclean.ids").read_text() == "P1\n"
    assert (out / "wt_fastas" / "P1.fasta").read_text() == ">P1\n" + SEQUENCE_1 + "\n"
    assert len(list((out / "shards").glob("shard_*.fasta"))) == 2

    prior = pd.read_parquet(out / "q00511_projection_prior.parquet")
    assert set(prior.mapping_status) == {"mapped_identity"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["parents"] == 2
    assert manifest["counts"]["wt_clean"] == 1
    assert manifest["counts"]["distinct_strong_cores"] == 1
    assert manifest["release_semantics"] == "all_safe_P1_P4_P6_P9_per_actionable_core"


def test_nonidentity_projection_is_retained_as_prior_not_cohort_gate(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    projection = pd.read_parquet(paths["projection"])
    row = projection.index[
        (projection.protein_id == "P2") & (projection.label == "His257")
    ][0]
    projection.loc[row, "target_index_0b"] = 5.0
    projection.loc[row, "target_aa"] = SEQUENCE_2[5]
    projection.loc[row, "matched"] = False
    projection.to_parquet(paths["projection"], index=False)
    args = _args(paths, tmp_path / "out")

    assert freeze.main(args) == 0
    cohort = pd.read_parquet(tmp_path / "out" / "cohort.parquet")
    assert cohort.protein_id.tolist() == ["P1", "P2"]
    assert not bool(
        cohort.set_index("protein_id").loc["P2", "q00511_all_hard_roles_identity"]
    )
    prior = pd.read_parquet(tmp_path / "out" / "q00511_projection_prior.parquet")
    row = prior[(prior.protein_id == "P2") & (prior.label == "His257")].iloc[0]
    assert row.mapping_status == "mapped_nonidentity"


def test_summary_strong_window_mismatch_fails_before_publication(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    summary = pd.read_parquet(paths["summary"])
    summary.loc[summary.protein_id == "P1", "n_strong_binders"] = 3
    summary.to_parquet(paths["summary"], index=False)
    out = tmp_path / "out"

    with pytest.raises(freeze.FreezeContractError, match="strong-window count"):
        freeze.main(_args(paths, out))
    assert not out.exists()


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(freeze.FreezeContractError, match="already exists"):
        freeze.main(_args(paths, out))
