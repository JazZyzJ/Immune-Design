"""Contract tests for the Active-15 production evidence-manifest builder."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.aggregate_uricase_active15_evidence import (
    JoinContractError,
    MANIFEST_COLUMNS,
)
from scripts.build_uricase_active15_evidence_manifest import main


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    protein_id = "P1"
    sequence = "ACDEFGHIK"
    fasta = tmp_path / "parents.fasta"
    fasta.write_text(f">{protein_id}\n{sequence}\n")

    evolution_dir = tmp_path / "evolution" / protein_id
    evolution_dir.mkdir(parents=True)
    for name in (
        "per_position_evolution.tsv",
        "wt_lock_masks.json",
        "complete_ec_table.tsv",
    ):
        (evolution_dir / name).write_text("fixture\n")
    (evolution_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protein_id": protein_id,
            }
        )
        + "\n"
    )

    source_hashes = {
        f"sample_{index}": _sha256(f"{protein_id}:sample_{index}")
        for index in range(5)
    }
    contact_dir = tmp_path / "contact" / protein_id
    contact_dir.mkdir(parents=True)
    prefix = f"{protein_id}_wt5"
    for suffix in (
        "contact_consensus.parquet",
        "contact_masks.json",
        "residue_pair_contacts.parquet",
        "residue_mapping.parquet",
    ):
        (contact_dir / f"{prefix}_{suffix}").write_text("fixture\n")
    (contact_dir / f"{prefix}_contact_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "tetramer_contact_consensus_v1",
                "protein_id": protein_id,
                "canonical_length": len(sequence),
                "canonical_sequence_sha256": _sha256(sequence),
                "mechanism": {"status": "resolved"},
                "sources": {
                    sample_id: {"sha256": sha256}
                    for sample_id, sha256 in source_hashes.items()
                },
                "parameters": {
                    "residue_pair_contact_cutoff_a": 5.0,
                    "mask_symmetry_copy_requirement": "both copies",
                    "mask_threshold_counts": [5, 4, 1],
                },
            }
        )
        + "\n"
    )

    energy_dir = tmp_path / "energy" / protein_id
    energy_dir.mkdir(parents=True)
    energy_table = (
        energy_dir
        / f"{protein_id}_sample0_energy_alanine_scan_by_position.parquet"
    )
    energy_table.write_text("fixture\n")
    (energy_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "tetramer_reference_metrics_v2",
                "name": f"{protein_id}_sample0_energy",
                "source_sha256": source_hashes["sample_0"],
            }
        )
        + "\n"
    )
    return {
        "fasta": fasta,
        "evolution_root": tmp_path / "evolution",
        "contact_root": tmp_path / "contact",
        "energy_root": tmp_path / "energy",
        "output": tmp_path / "evidence_manifest.tsv",
        "energy_metadata": energy_dir / "metadata.json",
        "energy_table": energy_table,
    }


def _argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--parent-fasta",
        str(paths["fasta"]),
        "--evolution-root",
        str(paths["evolution_root"]),
        "--contact-root",
        str(paths["contact_root"]),
        "--energy-root",
        str(paths["energy_root"]),
        "--expected-parent-count",
        "1",
        "--output",
        str(paths["output"]),
    ]


def test_builds_exact_available_manifest_atomically(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    assert main(_argv(paths)) == 0

    with paths["output"].open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == MANIFEST_COLUMNS
    assert len(rows) == 1
    row = rows[0]
    assert row["protein_id"] == "P1"
    assert row["evolution_status"] == "available"
    assert row["structure_status"] == "qualified"
    assert row["energy_status"] == "available"
    assert Path(row["energy_per_position_path"]) == paths["energy_table"].resolve()
    assert all(
        Path(row[column]).is_absolute()
        for column in MANIFEST_COLUMNS
        if column.endswith("_path")
    )
    assert Path(row["evolution_dir"]).is_absolute()


def test_accepts_metadata_only_evolution_panel_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    panel = tmp_path / "evolution_panel.parquet"
    pd.DataFrame(
        [{"protein_id": "P1", "output_dir": str(paths["evolution_root"] / "P1")}]
    ).to_parquet(panel, index=False)
    argv = _argv(paths)
    root_index = argv.index("--evolution-root")
    argv[root_index : root_index + 2] = ["--evolution-panel-manifest", str(panel)]

    assert main(argv) == 0
    with paths["output"].open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert Path(row["evolution_dir"]) == (paths["evolution_root"] / "P1").resolve()


def test_selects_one_complete_energy_root_across_base_and_repair(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    base = paths["energy_root"]
    repair = tmp_path / "repair_energy"
    (repair / "P1").mkdir(parents=True)
    argv = _argv(paths)
    root_index = argv.index("--energy-root")
    argv[root_index : root_index + 2] = [
        "--energy-root", str(repair), "--energy-root", str(base)
    ]

    assert main(argv) == 0
    with paths["output"].open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert Path(row["energy_metadata_path"]).parent == (base / "P1").resolve()


def test_missing_required_artifact_fails_without_partial_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["energy_table"].unlink()

    with pytest.raises(JoinContractError, match="missing production artifact"):
        main(_argv(paths))
    assert not paths["output"].exists()


def test_energy_source_must_belong_to_five_sample_contact_ensemble(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    metadata = json.loads(paths["energy_metadata"].read_text())
    metadata["source_sha256"] = "f" * 64
    paths["energy_metadata"].write_text(json.dumps(metadata) + "\n")

    with pytest.raises(JoinContractError, match="energy source SHA256"):
        main(_argv(paths))
    assert not paths["output"].exists()


def test_refuses_to_overwrite_existing_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].write_text("sentinel\n")

    with pytest.raises(JoinContractError, match="already exists"):
        main(_argv(paths))
    assert paths["output"].read_text() == "sentinel\n"
