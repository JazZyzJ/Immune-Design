"""L4 contract tests: Tier 3 candidate validation.

RED targets:
  1. Rejects candidate without literature evidence.
  2. Rejects candidate with empty literature evidence.
  3. Rejects candidate with sequence length outside 100-500.
  4. Accepts valid candidate.
  5. Batch validation counts correctly.
"""

import json

import pytest

from inverse_folding.evaluation.tier_validators import (
    validate_tier3_candidate,
    validate_tier3_candidates,
)
# _TIER3_MIN_NMP_WINDOWS = 3 in tier_validators.py


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_candidate(**overrides) -> dict:
    base = {
        "protein_id": "1D6R_A",
        "name": "adalimumab Fab",
        "pdb_id": "1D6R",
        "chain": "A",
        "sequence_length": 214,
        "resolution": 2.0,
        "structure_source": "xray",
        "literature_evidence": "Clinical ADA incidence 5-12%, Bartelds et al. 2011",
        "selection_reason": "well-characterized therapeutic",
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSingleCandidate:
    def test_valid_candidate_accepted(self):
        errors = validate_tier3_candidate(_make_candidate())
        assert len(errors) == 0

    def test_missing_literature_evidence_rejected(self):
        c = _make_candidate()
        del c["literature_evidence"]
        errors = validate_tier3_candidate(c)
        assert any("literature_evidence" in e for e in errors)

    def test_empty_literature_evidence_rejected(self):
        errors = validate_tier3_candidate(_make_candidate(literature_evidence=""))
        assert any("literature_evidence" in e for e in errors)

    def test_whitespace_literature_evidence_rejected(self):
        errors = validate_tier3_candidate(_make_candidate(literature_evidence="   "))
        assert any("literature_evidence" in e for e in errors)

    def test_too_short_rejected(self):
        errors = validate_tier3_candidate(_make_candidate(sequence_length=50))
        assert any("sequence_length" in e for e in errors)

    def test_too_long_rejected(self):
        errors = validate_tier3_candidate(_make_candidate(sequence_length=600))
        assert any("sequence_length" in e for e in errors)

    def test_missing_name_rejected(self):
        c = _make_candidate()
        del c["name"]
        errors = validate_tier3_candidate(c)
        assert any("name" in str(e) for e in errors)

    def test_boundary_length_accepted(self):
        assert len(validate_tier3_candidate(_make_candidate(sequence_length=100))) == 0
        assert len(validate_tier3_candidate(_make_candidate(sequence_length=500))) == 0


class TestNMPSignalCheck:
    """Tier 3 requires >= 3 NMP windows at %Rank < 5%."""

    def test_sufficient_nmp_accepted(self):
        c = _make_candidate(protein_id="NMP_OK")
        nmp = {"NMP_OK": {"n_strong_binders": 5}}
        errors = validate_tier3_candidate(c, nmp_scores=nmp)
        assert len(errors) == 0

    def test_insufficient_nmp_rejected(self):
        c = _make_candidate(protein_id="NMP_BAD")
        nmp = {"NMP_BAD": {"n_strong_binders": 1}}
        errors = validate_tier3_candidate(c, nmp_scores=nmp)
        assert any("NMP signal" in e for e in errors)

    def test_missing_nmp_scores_rejected(self):
        c = _make_candidate(protein_id="NO_NMP")
        nmp = {}  # protein not in scores dict
        errors = validate_tier3_candidate(c, nmp_scores=nmp)
        assert any("nmp_scores missing" in e for e in errors)

    def test_no_nmp_scores_skips_check(self):
        """When nmp_scores=None, NMP check is skipped."""
        c = _make_candidate(protein_id="SKIP_NMP")
        errors = validate_tier3_candidate(c, nmp_scores=None)
        assert len(errors) == 0


class TestBatchValidation:
    def test_all_valid(self, tmp_path):
        candidates = [_make_candidate(protein_id=f"P{i}_A") for i in range(3)]
        path = tmp_path / "tier3.json"
        path.write_text(json.dumps(candidates))

        result = validate_tier3_candidates(str(path))
        assert result["n_valid"] == 3
        assert result["n_invalid"] == 0

    def test_mixed(self, tmp_path):
        candidates = [
            _make_candidate(protein_id="GOOD_A"),
            _make_candidate(protein_id="BAD_A", literature_evidence=""),
        ]
        path = tmp_path / "tier3.json"
        path.write_text(json.dumps(candidates))

        result = validate_tier3_candidates(str(path))
        assert result["n_valid"] == 1
        assert result["n_invalid"] == 1

    def test_batch_with_nmp_scores(self, tmp_path):
        candidates = [
            _make_candidate(protein_id="OK_A"),
            _make_candidate(protein_id="LOW_A"),
        ]
        path = tmp_path / "tier3.json"
        path.write_text(json.dumps(candidates))

        nmp = {
            "OK_A": {"n_strong_binders": 5},
            "LOW_A": {"n_strong_binders": 1},
        }
        result = validate_tier3_candidates(str(path), nmp_scores=nmp)
        assert result["n_valid"] == 1
        assert result["n_invalid"] == 1
