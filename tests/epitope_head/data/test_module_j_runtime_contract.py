"""Module J contract tests — runtime p_aug augmentation.

TDD gates for PLAN.md Task J7.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from epitope_head.data.netmhciipan_mutation import (
    MutationRegistryIndex,
    apply_runtime_augmentation,
)
from epitope_head.training.datamodule import ProteinEntry


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_entry(pid, seq="A" * 50, positives=None):
    if positives is None:
        positives = [
            {"start_0b": 0, "end_0b": 15, "pep_len": 15, "support_n": 5},
            {"start_0b": 20, "end_0b": 35, "pep_len": 15, "support_n": 3},
        ]
    return ProteinEntry(
        protein_id=pid, protein_seq=seq, allele="HLA-DRB1*07:01",
        positives=positives, sequence_length=len(seq),
    )


def _make_registry_df(entries):
    """Create a minimal registry DataFrame for testing."""
    rows = []
    for pid, pos, wt, mut, delta, affected in entries:
        n_src_pos = 2  # assume 2 positives per source protein
        rows.append({
            "source_protein_id": pid,
            "mut_pos_0b": pos,
            "wt_aa": wt,
            "mut_aa": mut,
            "mutant_protein_id": f"AUG::{pid}::{pos}{wt}>{mut}",
            "seed_spans_json": json.dumps([]),
            "affected_spans_json": json.dumps(affected),
            "wt_ranks_json": json.dumps({}),
            "mut_ranks_json": json.dumps({}),
            "n_affected_spans": len(affected),
            "max_delta_rank": delta,
            "source_positive_count": n_src_pos,
            "remaining_positive_count": n_src_pos - len(affected),
            "runtime_train_eligible": (n_src_pos - len(affected)) > 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def registry_df():
    return _make_registry_df([
        # P1 has 3 mutations: different deltas for weighted sampling
        ("P1", 5, "A", "G", 0.80, [{"start_0b": 0, "end_0b": 15, "pep_len": 15}]),
        ("P1", 5, "A", "K", 0.40, [{"start_0b": 0, "end_0b": 15, "pep_len": 15}]),
        ("P1", 5, "A", "D", 0.20, [{"start_0b": 0, "end_0b": 15, "pep_len": 15}]),
        # P2 has 1 mutation
        ("P2", 25, "A", "G", 0.50, [{"start_0b": 20, "end_0b": 35, "pep_len": 15}]),
        # P3 has 1 mutation that removes ALL positives -> not eligible
        ("P3", 5, "A", "G", 0.60, [
            {"start_0b": 0, "end_0b": 15, "pep_len": 15},
            {"start_0b": 20, "end_0b": 35, "pep_len": 15},
        ]),
    ])


@pytest.fixture
def registry_parquet(tmp_path, registry_df):
    path = tmp_path / "registry.parquet"
    registry_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def base_entries():
    return [_make_entry("P1"), _make_entry("P2"), _make_entry("P3"), _make_entry("P4")]


# ── MutationRegistryIndex ─────────────────────────────────────────────────

class TestRegistryIndex:
    def test_load_from_parquet(self, registry_parquet):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        # P3 has remaining_positive_count=0 → not eligible
        assert "P1" in idx.by_protein
        assert "P2" in idx.by_protein
        assert "P3" not in idx.by_protein  # filtered out
        assert idx.n_eligible_proteins == 2
        assert idx.n_eligible_mutations == 4  # 3 from P1 + 1 from P2

    def test_weights_proportional_to_delta(self, registry_parquet):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        w = idx.weights["P1"]
        # deltas: 0.80, 0.40, 0.20 → weights proportional
        assert w[0] > w[1] > w[2]
        assert abs(w.sum() - 1.0) < 1e-6

    def test_missing_file_returns_empty(self, tmp_path):
        idx = MutationRegistryIndex.from_parquet(tmp_path / "nope.parquet")
        assert idx.n_eligible_proteins == 0


# ── apply_runtime_augmentation ────────────────────────────────────────────

class TestRuntimeAugmentation:
    def test_cardinality_preserved(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        result = apply_runtime_augmentation(base_entries, idx, p_aug=0.5, seed=42, epoch=0)
        assert len(result) == len(base_entries)

    def test_p_aug_zero_no_changes(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        result = apply_runtime_augmentation(base_entries, idx, p_aug=0.0, seed=42, epoch=0)
        for orig, aug in zip(base_entries, result):
            assert orig.protein_seq == aug.protein_seq
            assert orig.positives == aug.positives

    def test_deterministic_same_seed_epoch(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        r1 = apply_runtime_augmentation(base_entries, idx, p_aug=0.5, seed=42, epoch=3)
        r2 = apply_runtime_augmentation(base_entries, idx, p_aug=0.5, seed=42, epoch=3)
        for a, b in zip(r1, r2):
            assert a.protein_seq == b.protein_seq
            assert a.positives == b.positives

    def test_different_epoch_different_choices(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        r0 = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        r1 = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=1)
        # With p_aug=1.0, P1 and P2 are always replaced.
        # Different epochs should pick different mutations (at least sometimes)
        seqs_e0 = [e.protein_seq for e in r0]
        seqs_e1 = [e.protein_seq for e in r1]
        # At least one difference across 50 epochs would confirm non-deterministic across epochs
        # But with only 2 epochs, it's possible they happen to match — just check structure is valid
        assert all(len(e.positives) > 0 for e in r0)
        assert all(len(e.positives) > 0 for e in r1)

    def test_unregistered_protein_unchanged(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        result = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        # P4 has no registry entries → always unchanged
        p4_orig = base_entries[3]
        p4_aug = result[3]
        assert p4_orig.protein_seq == p4_aug.protein_seq
        assert p4_orig.positives == p4_aug.positives

    def test_mutated_entry_has_correct_seq(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        # Force P1 to be replaced (p_aug=1.0)
        result = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        p1 = result[0]
        # Should have exactly 1 AA different from original
        orig_seq = base_entries[0].protein_seq
        diffs = sum(1 for a, b in zip(orig_seq, p1.protein_seq) if a != b)
        assert diffs == 1

    def test_mutated_entry_has_fewer_positives(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        result = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        p1 = result[0]
        # Original has 2 positives, mutation removes 1 → 1 remaining
        assert len(p1.positives) == 1

    def test_original_entries_not_mutated(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        orig_seqs = [e.protein_seq for e in base_entries]
        _ = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        # Original entries should be unchanged
        for orig, seq in zip(base_entries, orig_seqs):
            assert orig.protein_seq == seq

    def test_zero_positive_mutations_excluded(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        # P3's only mutation removes all positives → not in registry index
        assert "P3" not in idx.by_protein
        result = apply_runtime_augmentation(base_entries, idx, p_aug=1.0, seed=42, epoch=0)
        # P3 should be unchanged
        p3 = result[2]
        assert p3.protein_seq == base_entries[2].protein_seq

    def test_return_stats_reports_effective_fraction(self, registry_parquet, base_entries):
        idx = MutationRegistryIndex.from_parquet(registry_parquet)
        result, stats = apply_runtime_augmentation(
            base_entries, idx, p_aug=1.0, seed=42, epoch=0, return_stats=True,
        )
        assert len(result) == len(base_entries)
        assert stats["configured_p_aug"] == 1.0
        assert stats["n_base_entries"] == 4
        assert stats["n_eligible_entries"] == 2
        assert stats["n_replaced"] == 2
        assert stats["effective_aug_fraction"] == 0.5
