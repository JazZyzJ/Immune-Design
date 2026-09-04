"""Contract tests for canonical evolution-evidence shard merge."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_evolution_evidence_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_evolution_evidence_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, protein_id: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "protein_id\tquery_id\talignment_path\tcovariance_gate\t"
        "neff_exact\tneff_per_length\n"
        f"{protein_id}\t{protein_id}\t/a/{protein_id}/focus_cov60.fasta\t"
        "qualified\t40\t10\n"
    )


def _write_parent(out_dir: Path, protein_id: str) -> None:
    out_dir.mkdir()
    metadata = {
        "schema_version": 1,
        "protein_id": protein_id,
        "sequence": {"length": 4},
        "model": {"covariance_status": "qualified"},
        "outputs": {
            "contact_precision": None,
            "contact_pairs": None,
            "contact_comparison": None,
        },
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata) + "\n")
    (out_dir / "potts_calibration_summary.json").write_text(
        json.dumps(
            {
                "protein_id": protein_id,
                "comparison_scope": f"within_model_only:{protein_id}",
                "cross_parent_thresholds_emitted": False,
            }
        )
        + "\n"
    )
    (out_dir / "wt_lock_masks.json").write_text(
        json.dumps(
            {
                "protein_id": protein_id,
                "masks": {name: [] for name in merge.EXPECTED_MASKS},
            }
        )
        + "\n"
    )
    for name in merge.MANDATORY_OUTPUT_FILES - {
        "analysis_metadata.json",
        "potts_calibration_summary.json",
        "wt_lock_masks.json",
    }:
        (out_dir / name).write_text("nonempty\n")


def _write_shard(
    normalized_root: Path,
    evolution_root: Path,
    tag: str,
    protein_id: str,
) -> None:
    manifest = normalized_root / tag / "plmc_manifest.tsv"
    _write_manifest(manifest, protein_id)
    shard_out = evolution_root / tag
    shard_out.mkdir(parents=True)
    _write_parent(shard_out / protein_id, protein_id)
    validation = pd.DataFrame(
        [
            {
                "protein_id": protein_id,
                "length": 4,
                "covariance_status": "qualified",
                "ec_pairs": 6,
                "potts_rows": 2,
                "potts_unique_imputed_sequences": 1,
                "potts_calibration_status": "descriptive_gate_not_supplied",
                "n_C70_stable": 2,
                "n_C80_stable": 1,
                "n_C90_stable": 1,
                "n_sigma80_L": 2,
                "n_sigma80_robust": 1,
                "n_sigma90_L": 2,
                "n_sigma90_robust": 1,
                "runtime_sec": 0.5,
            }
        ],
        columns=merge.VALIDATION_COLUMNS,
    )
    validation.to_csv(shard_out / "panel_validation.tsv", sep="\t", index=False)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "output_root": str(shard_out.resolve()),
        "parent_count": 1,
        "protein_ids": [protein_id],
        "preflight": [
            {
                "protein_id": protein_id,
                "length": 4,
                "covariance_status": "qualified",
                "ec_pairs": 6,
            }
        ],
        "validation_table": "panel_validation.tsv",
    }
    (shard_out / "panel_summary.json").write_text(json.dumps(summary) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    normalized = tmp_path / "normalized"
    evolution = tmp_path / "evolution"
    _write_shard(normalized, evolution, "000", "P1")
    _write_shard(normalized, evolution, "001", "P2")
    canonical = tmp_path / "canonical.tsv"
    canonical.write_text(
        "protein_id\tquery_id\talignment_path\tcovariance_gate\t"
        "neff_exact\tneff_per_length\n"
        "P2\tP2\t/a/P2/focus_cov60.fasta\tqualified\t40\t10\n"
        "P1\tP1\t/a/P1/focus_cov60.fasta\tqualified\t40\t10\n"
    )
    return canonical, normalized, evolution


def test_merge_restores_canonical_order_without_copying_parent_outputs(
    tmp_path: Path,
) -> None:
    canonical, normalized, evolution = _fixture(tmp_path)
    output = tmp_path / "merged"

    assert (
        merge.main(
            [
                "--canonical-plmc-manifest",
                str(canonical),
                "--normalized-shard-root",
                str(normalized),
                "--evolution-shard-root",
                str(evolution),
                "--expected-shards",
                "2",
                "--expected-parents",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    table = pd.read_parquet(output / "evolution_panel_manifest.parquet")
    assert table["protein_id"].tolist() == ["P2", "P1"]
    assert table["shard"].tolist() == ["001", "000"]
    assert not (output / "P1").exists()
    summary = json.loads((output / "evolution_merge_summary.json").read_text())
    assert summary["counts"]["valid_parents"] == 2
    assert summary["counts"]["covariance_status"] == {"qualified": 2}
    assert summary["storage"]["parent_outputs_duplicated"] is False


def test_merge_fails_atomically_when_parent_artifact_is_missing(tmp_path: Path) -> None:
    canonical, normalized, evolution = _fixture(tmp_path)
    (evolution / "001" / "P2" / "complete_ec_table.tsv").unlink()
    output = tmp_path / "merged"

    with pytest.raises(merge.MergeContractError, match="output file set mismatch"):
        merge.main(
            [
                "--canonical-plmc-manifest",
                str(canonical),
                "--normalized-shard-root",
                str(normalized),
                "--evolution-shard-root",
                str(evolution),
                "--expected-shards",
                "2",
                "--expected-parents",
                "2",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_merge_accepts_explicit_full_shard_replacement(tmp_path: Path) -> None:
    canonical, normalized, evolution = _fixture(tmp_path)
    replacement = tmp_path / "repair" / "001"
    replacement.parent.mkdir()
    (evolution / "001").rename(replacement)
    summary_path = replacement / "panel_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["output_root"] = str(replacement.resolve())
    summary_path.write_text(json.dumps(summary) + "\n")
    output = tmp_path / "merged"

    assert (
        merge.main(
            [
                "--canonical-plmc-manifest",
                str(canonical),
                "--normalized-shard-root",
                str(normalized),
                "--evolution-shard-root",
                str(evolution),
                "--shard-override",
                f"001={replacement}",
                "--expected-shards",
                "2",
                "--expected-parents",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    table = pd.read_parquet(output / "evolution_panel_manifest.parquet")
    assert table.loc[table["protein_id"].eq("P2"), "output_dir"].item().startswith(
        str(replacement)
    )
    summary = json.loads((output / "evolution_merge_summary.json").read_text())
    source = {row["shard"]: row for row in summary["inputs"]["shards"]}
    assert source["001"]["source_kind"] == "override"
