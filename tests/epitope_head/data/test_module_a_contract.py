"""Module A contract tests — targeted tests for A2 (explode), A3 (coords), A4 (allele filter)."""

import numpy as np
import pandas as pd
import pytest

from epitope_head.data.parse_mhc_if_v2 import (
    RejectionLedger,
    _classify_allele_row,
    apply_allele_filter,
    explode_positions,
    normalize_coordinates,
)


# ── A2: Explode position info ────────────────────────────────────────────────

class TestA2Explode:

    def _make_df(self, positions_list, **overrides):
        """Helper to build a single-row DF with parsed columns."""
        row = {
            "peptide_seq": "ACDEFGHIKLMN",
            "_alleles_parsed": ["HLA-DRB1*07:01"],
            "_positions_parsed": positions_list,
            "resolution": "high_res_single",
            "source": "test",
            "dataset_source": "test_v1",
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_multi_position_creates_multiple_records(self):
        positions = [
            {"protein_id": "P001", "start": 10.0, "end": 21.0},
            {"protein_id": "P002", "start": 50.0, "end": 61.0},
        ]
        df = self._make_df(positions)
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert len(out) == 2
        assert set(out["protein_id"]) == {"P001", "P002"}

    def test_single_position_creates_one_record(self):
        positions = [{"protein_id": "P001", "start": 10.0, "end": 21.0}]
        df = self._make_df(positions)
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert len(out) == 1

    def test_empty_positions_rejected(self):
        df = self._make_df([])
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert len(out) == 0
        assert ledger.counts.get("empty_positions", 0) == 1

    def test_missing_fields_in_position_rejected(self):
        positions = [{"protein_id": "P001"}]  # no start/end
        df = self._make_df(positions)
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert len(out) == 0
        assert ledger.counts.get("missing_position_fields", 0) == 1

    def test_float_coordinates_converted_to_int(self):
        positions = [{"protein_id": "P001", "start": 10.0, "end": 21.0}]
        df = self._make_df(positions)
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert out.iloc[0]["start_1b"] == 10
        assert out.iloc[0]["end_1b"] == 21
        assert isinstance(out.iloc[0]["start_1b"], (int, np.integer))

    def test_non_integral_float_coordinates_rejected(self):
        positions = [{"protein_id": "P001", "start": 10.9, "end": 21.0}]
        df = self._make_df(positions)
        ledger = RejectionLedger()
        out = explode_positions(df, ledger)
        assert len(out) == 0
        assert ledger.counts.get("non_integral_coordinates", 0) == 1


# ── A3: Coordinate normalization + length filter ─────────────────────────────

class TestA3Coords:

    def _make_records(self, start_1b_list, end_1b_list):
        return pd.DataFrame({
            "protein_id": [f"P{i}" for i in range(len(start_1b_list))],
            "start_1b": start_1b_list,
            "end_1b": end_1b_list,
            "peptide_seq": ["A" * (e - s + 1) for s, e in zip(start_1b_list, end_1b_list)],
            "alleles": [["HLA-DRB1*07:01"]] * len(start_1b_list),
            "resolution": ["high_res_single"] * len(start_1b_list),
            "source": ["test"] * len(start_1b_list),
            "dataset_source": ["test_v1"] * len(start_1b_list),
        })

    def test_conversion_invariant(self):
        """start_0b = start_1b - 1, end_0b = end_1b, pep_len = end_1b - start_1b + 1."""
        df = self._make_records([1, 100], [15, 120])
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        for _, row in out.iterrows():
            assert row["start_0b"] == row["start_1b"] - 1
            assert row["end_0b"] == row["end_1b"]
            assert row["pep_len"] == row["end_0b"] - row["start_0b"]

    def test_boundary_start_equals_1(self):
        """start_1b=1 should give start_0b=0."""
        df = self._make_records([1], [15])
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        assert out.iloc[0]["start_0b"] == 0

    def test_short_peptide_filtered(self):
        df = self._make_records([1], [10])  # length 10 < 12
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        assert len(out) == 0
        assert ledger.counts["pep_too_short"] == 1

    def test_long_peptide_filtered(self):
        df = self._make_records([1], [30])  # length 30 > 25
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        assert len(out) == 0
        assert ledger.counts["pep_too_long"] == 1

    def test_exact_boundary_lengths_kept(self):
        """Length 12 and 25 should both pass."""
        df = self._make_records([1, 1], [12, 25])  # len=12, len=25
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        assert len(out) == 2

    def test_invalid_1b_coordinates_rejected(self):
        df = self._make_records([0, 10], [20, 5])
        ledger = RejectionLedger()
        out = normalize_coordinates(df, 12, 25, ledger)
        assert len(out) == 0
        assert ledger.counts.get("invalid_1b_coordinates", 0) == 2


# ── A4: Allele filter policy ────────────────────────────────────────────────

class TestA4AlleleFilter:

    TARGET = "HLA-DRB1*07:01"

    def test_classify_strict(self):
        assert _classify_allele_row([self.TARGET], "high_res_single", self.TARGET) == "strict"

    def test_classify_multi(self):
        assert _classify_allele_row(
            [self.TARGET, "HLA-DRB1*01:01"], "multi", self.TARGET
        ) == "multi"

    def test_classify_reject_wrong_allele(self):
        assert _classify_allele_row(["HLA-DRB1*01:01"], "high_res_single", self.TARGET) == "reject"

    def test_classify_reject_wrong_resolution(self):
        """Single allele match but resolution is not high_res_single → reject."""
        assert _classify_allele_row([self.TARGET], "high_res_dimer", self.TARGET) == "reject"

    def test_classify_reject_multi_without_target(self):
        assert _classify_allele_row(
            ["HLA-DRB1*01:01", "HLA-DRB1*03:01"], "multi", self.TARGET
        ) == "reject"

    def test_balanced_includes_strict_plus_multi_sample(self):
        """Balanced = all strict + 50% of multi."""
        rows = []
        for i in range(100):
            rows.append({
                "protein_id": f"P{i}", "start_1b": 1, "end_1b": 15,
                "start_0b": 0, "end_0b": 15, "pep_len": 15,
                "peptide_seq": "A" * 15,
                "alleles": [self.TARGET],
                "resolution": "high_res_single",
                "source": "test", "dataset_source": "test_v1",
            })
        for i in range(100, 200):
            rows.append({
                "protein_id": f"P{i}", "start_1b": 1, "end_1b": 15,
                "start_0b": 0, "end_0b": 15, "pep_len": 15,
                "peptide_seq": "A" * 15,
                "alleles": [self.TARGET, "HLA-DRB1*01:01"],
                "resolution": "multi",
                "source": "test", "dataset_source": "test_v1",
            })
        df = pd.DataFrame(rows)
        ledger = RejectionLedger()
        strict_df, balanced_df = apply_allele_filter(
            df, self.TARGET, multi_ratio=0.5, multi_seed=42, ledger=ledger
        )
        assert len(strict_df) == 100
        assert len(balanced_df) == 150  # 100 strict + 50 multi

    def test_deterministic_multi_sampling(self):
        """Same seed → same sample."""
        rows = []
        for i in range(200):
            rows.append({
                "protein_id": f"P{i}", "start_1b": 1, "end_1b": 15,
                "start_0b": 0, "end_0b": 15, "pep_len": 15,
                "peptide_seq": "A" * 15,
                "alleles": [self.TARGET, "HLA-DRB1*01:01"],
                "resolution": "multi",
                "source": "test", "dataset_source": "test_v1",
            })
        df = pd.DataFrame(rows)
        ledger1 = RejectionLedger()
        ledger2 = RejectionLedger()
        _, b1 = apply_allele_filter(df, self.TARGET, 0.5, 42, ledger1)
        _, b2 = apply_allele_filter(df, self.TARGET, 0.5, 42, ledger2)
        pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))
