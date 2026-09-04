from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_merge_evolution_plmc_panel as audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parent(root: Path, protein_id: str, sequence: str, alignment: Path) -> None:
    parent = root / protein_id
    parent.mkdir(parents=True)
    model = parent / "plmc.model"
    ecs = parent / "plmc_ECs.txt"
    iterations = parent / "plmc_iterations.tsv"
    model.write_bytes(b"model")
    ecs.write_text("i j\n")
    iterations.write_text("iteration\n")
    summary = {
        "schema_version": 1,
        "query_id": protein_id,
        "query_sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "L": len(sequence),
        "alignment_path": str(alignment.resolve()),
        "alignment_sha256": _sha(alignment),
        "model_path": str(model.resolve()),
        "model_sha256": _sha(model),
        "ecs_path": str(ecs.resolve()),
        "ecs_sha256": _sha(ecs),
        "iterations_path": str(iterations.resolve()),
        "iterations_sha256": _sha(iterations),
        "theta_identity": 0.8,
        "plmc_theta_distance": 0.2,
        "lambda_h": 0.01,
        "lambda_J_base": 0.01,
        "lambda_J_scaled": 0.01 * 20 * (len(sequence) - 1),
        "iterations": 500,
        "focus_seq_index": 1,
        "region_start": 1,
        "model_validation": {
            "length_matches_query": True,
            "indices_1_to_L": True,
            "target_sequence_exact": True,
            "all_sites_valid": True,
        },
        "optimization_status": "ROUNDING_ERROR",
        "runtime_sec": 12.5,
        "overwrite_enabled": True,
    }
    (parent / "plmc_summary.json").write_text(json.dumps(summary))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cohort = tmp_path / "cohort.parquet"
    manifest = tmp_path / "plmc_manifest.tsv"
    plmc_root = tmp_path / "run" / "plmc"
    repair = tmp_path / "repair"
    sequences = {"P1": "ACDEF", "P2": "ACDEY"}
    pd.DataFrame(
        {
            "protein_id": list(sequences),
            "sequence": list(sequences.values()),
            "sequence_length": [len(value) for value in sequences.values()],
        }
    ).to_parquet(cohort, index=False)
    rows = []
    for protein_id, sequence in sequences.items():
        alignment = tmp_path / "msa" / protein_id / "focus_cov60.fasta"
        alignment.parent.mkdir(parents=True)
        alignment.write_text(f">{protein_id}\n{sequence}\n")
        _write_parent(plmc_root, protein_id, sequence, alignment)
        rows.append(
            {
                "protein_id": protein_id,
                "query_id": protein_id,
                "alignment_path": str(alignment.resolve()),
                "covariance_gate": "qualified" if protein_id == "P1" else "unqualified",
                "neff_exact": "50",
                "neff_per_length": "10" if protein_id == "P1" else "1",
            }
        )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    repair.mkdir()
    (repair / "post_repair_audit_summary.json").write_text(
        json.dumps(
            {
                "invalid_parent_rows": 0,
                "repair_parent_rows": 1,
                "repair_hash_revalidated": ["P2"],
            }
        )
    )
    return cohort, manifest, plmc_root, repair


def test_merge_publishes_exact_nonduplicating_canonical_view(tmp_path: Path) -> None:
    cohort, manifest, plmc_root, repair = _fixture(tmp_path)
    out = tmp_path / "work" / "merged"

    assert (
        audit.main(
            [
                "--cohort",
                str(cohort),
                "--plmc-manifest",
                str(manifest),
                "--plmc-root",
                str(plmc_root),
                "--repair-audit-dir",
                str(repair),
                "--expected-parents",
                "2",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )
    table = pd.read_parquet(out / "plmc_panel_manifest.parquet")
    assert table.protein_id.tolist() == ["P1", "P2"]
    assert table.set_index("protein_id").repair_hash_revalidated.to_dict() == {
        "P1": False,
        "P2": True,
    }
    assert all(str(path).startswith(str(plmc_root.resolve())) for path in table.model_path)
    summary = json.loads((out / "merge_audit_summary.json").read_text())
    assert summary["counts"]["valid_parents"] == 2
    assert summary["storage"]["models_duplicated"] is False
    assert (out / "plmc_manifest.tsv").read_bytes() == manifest.read_bytes()


def test_sequence_identity_failure_is_atomic(tmp_path: Path) -> None:
    cohort, manifest, plmc_root, repair = _fixture(tmp_path)
    summary_path = plmc_root / "P1" / "plmc_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["query_sequence_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    out = tmp_path / "out"

    with pytest.raises(audit.PlmcAuditError, match="P1"):
        audit.main(
            [
                "--cohort",
                str(cohort),
                "--plmc-manifest",
                str(manifest),
                "--plmc-root",
                str(plmc_root),
                "--repair-audit-dir",
                str(repair),
                "--expected-parents",
                "2",
                "--output-dir",
                str(out),
            ]
        )
    assert not out.exists()
