"""Module J contract tests — materialization and registry validation.

TDD gates for PLAN.md Tasks J3, J4, J5, J6.
"""

import json
import pytest
import pandas as pd

from epitope_head.data.netmhciipan_mutation import (
    MutationCandidate,
    ScoredMutation,
    enumerate_mutations,
    select_top_mutations,
    validate_augmentation_config,
)
from epitope_head.data.materialize_augmented_samples import (
    materialize_augmented_samples,
    validate_augmented_samples,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

PROTEIN_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 AA

SOURCE_DF = pd.DataFrame([{
    "protein_id": "TEST001",
    "allele": "HLA-DRB1*07:01",
    "protein_seq": PROTEIN_SEQ,
    "positives_json": json.dumps([
        {"start_0b": 10, "end_0b": 25, "pep_len": 15, "support_n": 5},
        {"start_0b": 12, "end_0b": 27, "pep_len": 15, "support_n": 3},
        {"start_0b": 50, "end_0b": 65, "pep_len": 15, "support_n": 8},
    ]),
    "sequence_length": 100,
    "input_span_count": 10,
    "positive_count": 3,
    "duplicate_span_count": 7,
    "metadata_json": json.dumps({"source_counts": {"iedb": 10}}),
}])


REGISTRY_DF = pd.DataFrame([{
    "source_protein_id": "TEST001",
    "mut_pos_0b": 15,
    "wt_aa": "H",
    "mut_aa": "A",
    "mutant_protein_id": "AUG::TEST001::15H>A",
    "seed_spans_json": json.dumps([{"start_0b": 10, "end_0b": 25, "pep_len": 15}]),
    "affected_spans_json": json.dumps([
        {"start_0b": 10, "end_0b": 25, "pep_len": 15},
        {"start_0b": 12, "end_0b": 27, "pep_len": 15},
    ]),
    "wt_ranks_json": json.dumps({"15": 0.01}),
    "mut_ranks_json": json.dumps({"15": 0.35}),
    "n_affected_spans": 2,
    "max_delta_rank": 0.34,
}])


# ── J3: Mutation Enumeration ──────────────────────────────────────────────

class TestJ3Enumeration:
    """TDD gate J3: dedup of overlapping-span mutations."""

    def test_dedup_overlapping_spans(self):
        """Two spans overlapping at position 15 produce one candidate per (pos, mut_aa)."""
        eligible = [
            {"start_0b": 10, "end_0b": 25, "pep_len": 15},
            {"start_0b": 12, "end_0b": 27, "pep_len": 15},
        ]
        candidates = enumerate_mutations("TEST001", PROTEIN_SEQ, eligible)

        # Position 15 is in both spans → should appear once per mut_aa
        pos15_candidates = [c for c in candidates if c.mut_pos_0b == 15]
        assert len(pos15_candidates) == 19  # 20 AA - 1 wt

        # Each should have 2 seed spans (from both eligible spans)
        for c in pos15_candidates:
            assert len(c.seed_spans) == 2

    def test_non_overlapping_spans_independent(self):
        eligible = [
            {"start_0b": 10, "end_0b": 15, "pep_len": 5},
            {"start_0b": 50, "end_0b": 55, "pep_len": 5},
        ]
        candidates = enumerate_mutations("TEST001", PROTEIN_SEQ, eligible)

        # No overlap → each position has 1 seed span
        for c in candidates:
            assert len(c.seed_spans) == 1

    def test_correct_wt_aa(self):
        eligible = [{"start_0b": 0, "end_0b": 3, "pep_len": 3}]
        candidates = enumerate_mutations("TEST001", PROTEIN_SEQ, eligible)

        # Position 0 = 'A', should have 19 candidates (not including A)
        pos0 = [c for c in candidates if c.mut_pos_0b == 0]
        assert len(pos0) == 19
        assert all(c.wt_aa == "A" for c in pos0)
        assert "A" not in [c.mut_aa for c in pos0]

    def test_sorted_output(self):
        eligible = [{"start_0b": 5, "end_0b": 10, "pep_len": 5}]
        candidates = enumerate_mutations("TEST001", PROTEIN_SEQ, eligible)
        # Should be sorted by (pos, mut_aa)
        keys = [(c.mut_pos_0b, c.mut_aa) for c in candidates]
        assert keys == sorted(keys)


# ── J5: Top-N Selection ───────────────────────────────────────────────────

class TestJ5TopN:
    def test_top_n_selects_highest_delta(self):
        mutations = [
            ScoredMutation("P1", 10, "A", "G", [], [{}],
                           wt_ranks={15: 0.01}, mut_ranks={15: 0.30}),
            ScoredMutation("P1", 20, "L", "K", [], [{}],
                           wt_ranks={15: 0.01}, mut_ranks={15: 0.50}),
            ScoredMutation("P1", 30, "M", "D", [], [{}],
                           wt_ranks={15: 0.01}, mut_ranks={15: 0.40}),
        ]
        selected = select_top_mutations(mutations, top_n=2)
        assert len(selected) == 2
        # Highest delta first
        assert selected[0].mut_pos_0b == 20  # delta = 0.49
        assert selected[1].mut_pos_0b == 30  # delta = 0.39

    def test_top_n_returns_all_if_fewer(self):
        mutations = [
            ScoredMutation("P1", 10, "A", "G", [], [{}],
                           wt_ranks={15: 0.01}, mut_ranks={15: 0.30}),
        ]
        selected = select_top_mutations(mutations, top_n=64)
        assert len(selected) == 1


# ── J6: Materialization ───────────────────────────────────────────────────

class TestJ6Materialization:
    def test_basic_materialization(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        assert len(aug_df) == 1
        row = aug_df.iloc[0]
        assert row["protein_id"] == "AUG::TEST001::15H>A"
        assert row["allele"] == "HLA-DRB1*07:01"
        assert row["sequence_length"] == 100

    def test_mutant_sequence_correct(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        mut_seq = aug_df.iloc[0]["protein_seq"]
        # Position 15 should be 'A' instead of original
        assert mut_seq[15] == "A"
        # Rest unchanged
        assert mut_seq[:15] == PROTEIN_SEQ[:15]
        assert mut_seq[16:] == PROTEIN_SEQ[16:]

    def test_affected_spans_removed(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        remaining = json.loads(aug_df.iloc[0]["positives_json"])
        # 2 of 3 positives were affected → 1 remaining
        assert len(remaining) == 1
        assert remaining[0]["start_0b"] == 50  # the non-overlapping one

    def test_no_reuse_of_source_id(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        source_ids = set(SOURCE_DF["protein_id"])
        aug_ids = set(aug_df["protein_id"])
        assert len(source_ids & aug_ids) == 0

    def test_provenance_metadata(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        meta = json.loads(aug_df.iloc[0]["metadata_json"])
        assert "augmentation" in meta
        assert meta["augmentation"]["source_protein_id"] == "TEST001"
        assert meta["augmentation"]["mut_pos_0b"] == 15

    def test_validation_passes_clean_data(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        errors = validate_augmented_samples(
            aug_df, SOURCE_DF, val_ids={"VAL001"}, test_ids={"TEST_ID"}
        )
        assert errors == []

    def test_validation_catches_source_id_reuse(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        # Tamper: change protein_id to match source
        aug_df.loc[0, "protein_id"] = "TEST001"
        errors = validate_augmented_samples(
            aug_df, SOURCE_DF, val_ids=set(), test_ids=set()
        )
        assert any("reuse source" in e for e in errors)

    def test_validation_catches_val_leak(self):
        aug_df = materialize_augmented_samples(REGISTRY_DF, SOURCE_DF)
        # Pretend augmented ID is in val
        errors = validate_augmented_samples(
            aug_df, SOURCE_DF,
            val_ids={"AUG::TEST001::15H>A"},
            test_ids=set(),
        )
        assert any("leaked into val" in e for e in errors)

    def test_empty_registry(self):
        empty_reg = pd.DataFrame(columns=REGISTRY_DF.columns)
        aug_df = materialize_augmented_samples(empty_reg, SOURCE_DF)
        assert len(aug_df) == 0
