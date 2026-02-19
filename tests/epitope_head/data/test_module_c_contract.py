"""Module C contract tests — dedup, group isolation, metadata invariants."""

import json

import pandas as pd
import pytest

from epitope_head.data.make_protein_samples import (
    AggStats,
    aggregate_protein_samples,
    validate_protein_samples,
)


def _make_span_df(rows: list[dict]) -> pd.DataFrame:
    """Build a Stage B-like DataFrame from simplified row dicts."""
    defaults = {
        "allele": "HLA-DRB1*07:01",
        "protein_seq": "A" * 100,
        "start_1b": 1, "end_1b": 15,
        "source": "test", "dataset_source": "test_v1",
    }
    full_rows = []
    for r in rows:
        row = {**defaults, **r}
        row.setdefault("start_0b", row["start_1b"] - 1)
        row.setdefault("end_0b", row["end_1b"])
        row.setdefault("pep_len", row["end_0b"] - row["start_0b"])
        row.setdefault("peptide_seq", "A" * row["pep_len"])
        full_rows.append(row)
    return pd.DataFrame(full_rows)


# ── C1: Deduplication ────────────────────────────────────────────────────────

class TestC1Dedup:

    def test_duplicate_spans_collapsed_with_support(self):
        """Two identical spans → one unique positive with support_n=2."""
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        assert len(out) == 1
        assert out.iloc[0]["positive_count"] == 1
        assert out.iloc[0]["duplicate_span_count"] == 1
        assert out.iloc[0]["input_span_count"] == 2
        positives = json.loads(out.iloc[0]["positives_json"])
        assert positives[0]["support_n"] == 2

    def test_different_spans_kept_with_support_1(self):
        """Two different spans → two unique positives each with support_n=1."""
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 10, "end_1b": 25},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        assert out.iloc[0]["positive_count"] == 2
        assert out.iloc[0]["duplicate_span_count"] == 0
        positives = json.loads(out.iloc[0]["positives_json"])
        assert all(p["support_n"] == 1 for p in positives)

    def test_positives_json_is_sorted(self):
        """Positives are sorted by (start_0b, end_0b, pep_len)."""
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 10, "end_1b": 25},
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        positives = json.loads(out.iloc[0]["positives_json"])
        starts = [p["start_0b"] for p in positives]
        assert starts == sorted(starts)


# ── C2: Group isolation ──────────────────────────────────────────────────────

class TestC2Isolation:

    def test_different_proteins_not_merged(self):
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P002", "start_1b": 1, "end_1b": 15},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        assert len(out) == 2
        assert set(out["protein_id"]) == {"P001", "P002"}

    def test_inconsistent_seq_within_group_dropped(self):
        """Same protein_id but different protein_seq → dropped."""
        df = _make_span_df([
            {"protein_id": "P001", "protein_seq": "A" * 100, "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "protein_seq": "B" * 100, "start_1b": 1, "end_1b": 15},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_inconsistent_seq", 0) == 2


# ── C3: Metadata invariants ─────────────────────────────────────────────────

class TestC3Metadata:

    def test_invariants_pass_on_valid_data(self):
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},  # duplicate
            {"protein_id": "P001", "start_1b": 10, "end_1b": 25},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        # Should not raise
        validate_protein_samples(out)
        row = out.iloc[0]
        assert row["sequence_length"] == 100
        assert row["input_span_count"] == 3
        assert row["positive_count"] == 2
        assert row["duplicate_span_count"] == 1

    def test_metadata_has_source_counts(self):
        df = _make_span_df([
            {"protein_id": "P001", "source": "iedb", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "source": "atlas", "start_1b": 10, "end_1b": 25},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        meta = json.loads(out.iloc[0]["metadata_json"])
        assert meta["source_counts"] == {"atlas": 1, "iedb": 1}

    def test_support_n_sum_equals_input_span_count(self):
        """sum(support_n) across all positives must equal input_span_count."""
        df = _make_span_df([
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 1, "end_1b": 15},
            {"protein_id": "P001", "start_1b": 10, "end_1b": 25},
        ])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        validate_protein_samples(out)
        row = out.iloc[0]
        positives = json.loads(row["positives_json"])
        assert sum(p["support_n"] for p in positives) == row["input_span_count"]
        # First span seen 3 times, second seen 1 time
        support_map = {p["start_0b"]: p["support_n"] for p in positives}
        assert support_map[0] == 3
        assert support_map[9] == 1

    def test_empty_input_returns_empty_with_schema(self):
        df = _make_span_df([])
        stats = AggStats()
        out = aggregate_protein_samples(df, stats)
        assert len(out) == 0
        assert "protein_id" in out.columns
        assert "positives_json" in out.columns
