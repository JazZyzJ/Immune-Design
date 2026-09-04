from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import merge_query_centered_msa_shards as merge


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _write_shard(root: Path, tag: str, protein_id: str, sequence: str) -> None:
    shard = root / tag
    parent = shard / protein_id
    parent.mkdir(parents=True)
    alignment = parent / "focus_cov60.fasta"
    alignment.write_text(f">{protein_id}\n{sequence}\n")
    (parent / "focus_cov70.fasta").write_text(f">{protein_id}\n{sequence}\n")
    (parent / "focus_cov80.fasta").write_text(f">{protein_id}\n{sequence}\n")
    qc_path = parent / "msa_qc.json"
    qc = {
        "protein_id": protein_id,
        "query_sha256": _sha(sequence),
        "covariance_gate": "unqualified",
    }
    qc_path.write_text(json.dumps(qc))
    with (shard / "plmc_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merge.PLMC_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "protein_id": protein_id,
                "query_id": protein_id,
                "alignment_path": str(alignment.resolve()),
                "covariance_gate": "unqualified",
                "neff_exact": "1.0",
                "neff_per_length": "0.2",
            }
        )
    with (shard / "msa_qc.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merge.QC_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "protein_id": protein_id,
                "query_length": len(sequence),
                "n_raw_rows": 1,
                "n_normalized_rows": 1,
                "plmc_coverage": 0.6,
                "plmc_n_rows": 1,
                "plmc_neff_exact": 1.0,
                "plmc_neff_per_length": 0.2,
                "covariance_gate": "unqualified",
                "query_sha256": _sha(sequence),
                "a3m_sha256": "a" * 64,
                "qc_path": str(qc_path.resolve()),
            }
        )
    (shard / "msa_qc.json").write_text(json.dumps([qc]))


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    frozen = tmp_path / "frozen"
    (frozen / "shards").mkdir(parents=True)
    (frozen / "all_parents.ids").write_text("P1\nP2\n")
    (frozen / "shards" / "shard_000.ids").write_text("P1\n")
    (frozen / "shards" / "shard_001.ids").write_text("P2\n")
    pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "sequence": ["ACDEF", "ACDEY"],
            "sequence_sha256": [_sha("ACDEF"), _sha("ACDEY")],
        }
    ).to_parquet(frozen / "cohort.parquet", index=False)
    normalized = tmp_path / "normalized_shards"
    _write_shard(normalized, "000", "P1", "ACDEF")
    _write_shard(normalized, "001", "P2", "ACDEY")
    return frozen, normalized


def test_merge_exact_shard_union_in_frozen_order(tmp_path: Path) -> None:
    frozen, normalized = _fixtures(tmp_path)
    out = tmp_path / "merged"

    assert (
        merge.main(
            [
                "--frozen-root",
                str(frozen),
                "--normalized-shard-root",
                str(normalized),
                "--output-dir",
                str(out),
                "--expected-shards",
                "2",
                "--expected-parents",
                "2",
            ]
        )
        == 0
    )
    with (out / "plmc_manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["protein_id"] for row in rows] == ["P1", "P2"]
    qc = json.loads((out / "msa_qc.json").read_text())
    assert [row["protein_id"] for row in qc] == ["P1", "P2"]
    manifest = json.loads((out / "merge_manifest.json").read_text())
    assert manifest["counts"] == {"parents": 2, "shards": 2}


def test_duplicate_parent_across_frozen_shards_fails_atomically(tmp_path: Path) -> None:
    frozen, normalized = _fixtures(tmp_path)
    (frozen / "shards" / "shard_001.ids").write_text("P1\n")
    out = tmp_path / "out"

    with pytest.raises(merge.MergeContractError, match="duplicate parent"):
        merge.main(
            [
                "--frozen-root",
                str(frozen),
                "--normalized-shard-root",
                str(normalized),
                "--output-dir",
                str(out),
                "--expected-shards",
                "2",
                "--expected-parents",
                "2",
            ]
        )
    assert not out.exists()
