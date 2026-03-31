"""L2 contract tests: Tier 1 candidate validation script.

RED targets:
  1. Rejects candidate missing experimental epitopes.
  2. Rejects candidate with sequence length outside 100-500.
  3. Rejects candidate with resolution > 2.5.
  4. Rejects candidate with < 2 experimental epitopes.
  5. Accepts valid candidate.
  6. Flags CATH overlap correctly (via mock).
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from inverse_folding.evaluation.tier_validators import (
    validate_tier1_candidate,
    validate_tier1_candidates,
)
from inverse_folding.evaluation.overlap import OverlapResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_candidate(**overrides) -> dict:
    base = {
        "protein_id": "1ABC_A",
        "uniprot_id": "P12345",
        "pdb_id": "1ABC",
        "chain": "A",
        "sequence_length": 250,
        "resolution": 1.8,
        "experimental_epitopes": [
            {"start_0b": 45, "end_0b": 60, "assay_type": "T-cell", "iedb_ref": "REF1"},
            {"start_0b": 120, "end_0b": 135, "assay_type": "EL", "iedb_ref": "REF2"},
        ],
        "selection_reason": "well-characterized",
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSingleCandidate:
    def test_valid_candidate_accepted(self):
        c = _make_candidate()
        errors = validate_tier1_candidate(c)
        assert len(errors) == 0

    def test_missing_protein_id_rejected(self):
        c = _make_candidate()
        del c["protein_id"]
        errors = validate_tier1_candidate(c)
        assert any("protein_id" in e for e in errors)

    def test_missing_epitopes_rejected(self):
        c = _make_candidate()
        del c["experimental_epitopes"]
        errors = validate_tier1_candidate(c)
        assert any("experimental_epitopes" in e for e in errors)

    def test_too_short_rejected(self):
        c = _make_candidate(sequence_length=50)
        errors = validate_tier1_candidate(c)
        assert any("sequence_length" in e for e in errors)

    def test_too_long_rejected(self):
        c = _make_candidate(sequence_length=600)
        errors = validate_tier1_candidate(c)
        assert any("sequence_length" in e for e in errors)

    def test_low_resolution_rejected(self):
        c = _make_candidate(resolution=3.0)
        errors = validate_tier1_candidate(c)
        assert any("resolution" in e for e in errors)

    def test_fewer_than_2_epitopes_rejected(self):
        c = _make_candidate(experimental_epitopes=[
            {"start_0b": 45, "end_0b": 60, "assay_type": "T-cell", "iedb_ref": "REF1"},
        ])
        errors = validate_tier1_candidate(c)
        assert any("epitope" in e.lower() for e in errors)

    def test_boundary_100_length_accepted(self):
        c = _make_candidate(sequence_length=100)
        errors = validate_tier1_candidate(c)
        assert len(errors) == 0

    def test_boundary_500_length_accepted(self):
        c = _make_candidate(sequence_length=500)
        errors = validate_tier1_candidate(c)
        assert len(errors) == 0

    def test_boundary_2_5_resolution_accepted(self):
        c = _make_candidate(resolution=2.5)
        errors = validate_tier1_candidate(c)
        assert len(errors) == 0


class TestBatchValidation:
    def test_all_valid(self, tmp_path):
        candidates = [_make_candidate(protein_id=f"P{i}_A") for i in range(3)]
        json_path = tmp_path / "tier1_candidates.json"
        json_path.write_text(json.dumps(candidates))

        result = validate_tier1_candidates(str(json_path))
        assert result["n_valid"] == 3
        assert result["n_invalid"] == 0

    def test_mixed_valid_invalid(self, tmp_path):
        candidates = [
            _make_candidate(protein_id="GOOD_A"),
            _make_candidate(protein_id="BAD_A", sequence_length=10),
        ]
        json_path = tmp_path / "tier1_candidates.json"
        json_path.write_text(json.dumps(candidates))

        result = validate_tier1_candidates(str(json_path))
        assert result["n_valid"] == 1
        assert result["n_invalid"] == 1
        assert "BAD_A" in str(result["errors"])


class TestOverlapIntegration:
    """Verify that MMseqs2 overlap results exclude CATH-overlapping candidates."""

    def test_overlap_candidate_excluded(self, tmp_path):
        candidates = [
            _make_candidate(protein_id="OVERLAP_A"),
            _make_candidate(protein_id="CLEAN_A"),
        ]
        json_path = tmp_path / "tier1.json"
        json_path.write_text(json.dumps(candidates))

        overlap_results = {
            "OVERLAP_A": OverlapResult(
                query_id="OVERLAP_A",
                best_target="CATH_T1",
                best_identity=0.45,
                exclude=True,
            ),
        }

        result = validate_tier1_candidates(str(json_path), overlap_results)
        assert result["n_valid"] == 1
        assert result["n_overlap_excluded"] == 1
        assert "OVERLAP_A" in result["overlap_excluded"]
        assert "CLEAN_A" in result["valid"]

    def test_no_overlap_results_all_pass(self, tmp_path):
        candidates = [_make_candidate(protein_id="P1_A")]
        json_path = tmp_path / "tier1.json"
        json_path.write_text(json.dumps(candidates))

        result = validate_tier1_candidates(str(json_path), overlap_results=None)
        assert result["n_valid"] == 1
        assert result.get("n_overlap_excluded", 0) == 0
