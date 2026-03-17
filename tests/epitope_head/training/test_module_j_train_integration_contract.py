"""Module J contract tests — training integration.

TDD gate for PLAN.md Task J7: augmented entries in train, excluded from val/test.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from epitope_head.training.datamodule import (
    load_augmented_proteins,
    load_split_proteins,
)


@pytest.fixture
def tmp_data(tmp_path):
    """Create minimal parquet and split files for testing."""
    # Base samples
    base_df = pd.DataFrame([
        {
            "protein_id": "TRAIN_P1",
            "allele": "HLA-DRB1*07:01",
            "protein_seq": "A" * 50,
            "positives_json": json.dumps([
                {"start_0b": 0, "end_0b": 15, "pep_len": 15, "support_n": 5},
            ]),
            "sequence_length": 50,
        },
        {
            "protein_id": "VAL_P1",
            "allele": "HLA-DRB1*07:01",
            "protein_seq": "G" * 50,
            "positives_json": json.dumps([
                {"start_0b": 0, "end_0b": 15, "pep_len": 15, "support_n": 3},
            ]),
            "sequence_length": 50,
        },
    ])

    base_path = tmp_path / "base.parquet"
    base_df.to_parquet(base_path, index=False)

    train_ids_path = tmp_path / "train_ids.txt"
    train_ids_path.write_text("TRAIN_P1\n")
    val_ids_path = tmp_path / "val_ids.txt"
    val_ids_path.write_text("VAL_P1\n")

    # Augmented samples (train-only)
    aug_df = pd.DataFrame([
        {
            "protein_id": "AUG::TRAIN_P1::5A>G",
            "allele": "HLA-DRB1*07:01",
            "protein_seq": "A" * 5 + "G" + "A" * 44,
            "positives_json": json.dumps([]),
            "sequence_length": 50,
        },
        {
            "protein_id": "AUG::TRAIN_P1::10A>K",
            "allele": "HLA-DRB1*07:01",
            "protein_seq": "A" * 10 + "K" + "A" * 39,
            "positives_json": json.dumps([]),
            "sequence_length": 50,
        },
    ])
    aug_path = tmp_path / "aug_train.parquet"
    aug_df.to_parquet(aug_path, index=False)

    return {
        "base_path": base_path,
        "train_ids_path": train_ids_path,
        "val_ids_path": val_ids_path,
        "aug_path": aug_path,
    }


class TestJ7TrainIntegration:
    """TDD gate J7: train includes augmented, val/test exclude."""

    def test_train_loads_base_entries(self, tmp_data):
        entries = load_split_proteins(
            tmp_data["base_path"], tmp_data["train_ids_path"],
        )
        assert len(entries) == 1
        assert entries[0].protein_id == "TRAIN_P1"

    def test_augmented_loader_returns_entries(self, tmp_data):
        aug_entries = load_augmented_proteins(tmp_data["aug_path"])
        assert len(aug_entries) == 2
        assert all(e.protein_id.startswith("AUG::") for e in aug_entries)

    def test_train_with_augmented_union(self, tmp_data):
        base = load_split_proteins(
            tmp_data["base_path"], tmp_data["train_ids_path"],
        )
        aug = load_augmented_proteins(tmp_data["aug_path"])
        combined = base + aug
        assert len(combined) == 3
        ids = [e.protein_id for e in combined]
        assert "TRAIN_P1" in ids
        assert "AUG::TRAIN_P1::5A>G" in ids
        assert "AUG::TRAIN_P1::10A>K" in ids

    def test_val_excludes_augmented(self, tmp_data):
        val_entries = load_split_proteins(
            tmp_data["base_path"], tmp_data["val_ids_path"],
        )
        assert len(val_entries) == 1
        assert val_entries[0].protein_id == "VAL_P1"
        # No augmented entries
        assert all(not e.protein_id.startswith("AUG::") for e in val_entries)

    def test_missing_aug_file_returns_empty(self, tmp_path):
        entries = load_augmented_proteins(tmp_path / "nonexistent.parquet")
        assert entries == []

    def test_augmented_entries_have_correct_schema(self, tmp_data):
        aug = load_augmented_proteins(tmp_data["aug_path"])
        for e in aug:
            assert hasattr(e, "protein_id")
            assert hasattr(e, "protein_seq")
            assert hasattr(e, "allele")
            assert hasattr(e, "positives")
            assert hasattr(e, "sequence_length")
            assert isinstance(e.positives, list)
