"""Tests for Module N1: Level 1 post-hoc filter baseline.

Tests the argmin-risk selection logic and M3 candidate FASTA parsing.
"""

import json
import os
import textwrap

import numpy as np
import pytest

from inverse_folding.baselines.level1_filter import (
    parse_candidates_fasta,
    select_argmin_candidate,
    filter_protein_set,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def m3_candidates_dir(tmp_path):
    """Create a mock M3 all_candidates/ directory."""
    cand_dir = tmp_path / "all_candidates"
    cand_dir.mkdir()

    # Protein A: 4 candidates, candidate 2 has lowest risk
    (cand_dir / "protA.fasta").write_text(textwrap.dedent("""\
        >protA_cand0 | risk=0.7500 | hash=aaa111aaa111
        ACDEFGH
        >protA_cand1 | risk=0.5200 | hash=bbb222bbb222
        ACDEYGH
        >protA_cand2 | risk=0.3100 | hash=ccc333ccc333
        ACDEYTH
        >protA_cand3 | risk=0.6800 | hash=ddd444ddd444
        ACDXFGH
    """))

    # Protein B: 3 candidates, candidate 0 has lowest risk
    (cand_dir / "protB.fasta").write_text(textwrap.dedent("""\
        >protB_cand0 | risk=0.1200 | hash=eee555eee555
        MKTLLILAVL
        >protB_cand1 | risk=0.4500 | hash=fff666fff666
        MKTLXILAVL
        >protB_cand2 | risk=0.3300 | hash=ggg777ggg777
        MKTLLIXXVL
    """))

    return tmp_path


# ── parse_candidates_fasta ────────────────────────────────────────────────

class TestParseCandidatesFasta:

    def test_parses_sequences_and_risks(self, m3_candidates_dir):
        fasta_path = m3_candidates_dir / "all_candidates" / "protA.fasta"
        result = parse_candidates_fasta(str(fasta_path))

        assert len(result["sequences"]) == 4
        assert len(result["risks"]) == 4
        assert result["sequences"][0] == "ACDEFGH"
        assert result["sequences"][2] == "ACDEYTH"
        np.testing.assert_allclose(
            result["risks"], [0.75, 0.52, 0.31, 0.68], atol=1e-4,
        )

    def test_preserves_hashes(self, m3_candidates_dir):
        fasta_path = m3_candidates_dir / "all_candidates" / "protA.fasta"
        result = parse_candidates_fasta(str(fasta_path))

        assert result["hashes"][0] == "aaa111aaa111"
        assert result["hashes"][2] == "ccc333ccc333"

    def test_raises_on_empty_fasta(self, tmp_path):
        empty = tmp_path / "empty.fasta"
        empty.write_text("")
        with pytest.raises(ValueError, match="No candidates"):
            parse_candidates_fasta(str(empty))


# ── select_argmin_candidate ───────────────────────────────────────────────

class TestSelectArgminCandidate:

    def test_selects_minimum_risk(self):
        risks = np.array([0.75, 0.52, 0.31, 0.68])
        sequences = ["ACDEFGH", "ACDEYGH", "ACDEYTH", "ACDXFGH"]

        result = select_argmin_candidate(risks, sequences)
        assert result["selected_index"] == 2
        assert result["selected_risk"] == pytest.approx(0.31)
        assert result["selected_sequence"] == "ACDEYTH"
        assert result["selection_rule"] == "argmin_risk"

    def test_returns_candidate_risks(self):
        risks = np.array([0.5, 0.3, 0.8])
        sequences = ["A", "B", "C"]

        result = select_argmin_candidate(risks, sequences)
        np.testing.assert_allclose(result["candidate_risks"], [0.5, 0.3, 0.8])

    def test_ties_select_first(self):
        """When multiple candidates tie at min risk, select the first."""
        risks = np.array([0.5, 0.2, 0.2, 0.8])
        sequences = ["A", "B", "C", "D"]

        result = select_argmin_candidate(risks, sequences)
        assert result["selected_index"] == 1

    def test_single_candidate(self):
        risks = np.array([0.42])
        sequences = ["ONLY"]

        result = select_argmin_candidate(risks, sequences)
        assert result["selected_index"] == 0
        assert result["selected_sequence"] == "ONLY"

    def test_returns_seq_hash(self):
        risks = np.array([0.9, 0.1])
        sequences = ["AAA", "BBB"]

        result = select_argmin_candidate(risks, sequences)
        assert "selected_seq_hash" in result
        assert isinstance(result["selected_seq_hash"], str)
        assert len(result["selected_seq_hash"]) == 12


# ── filter_protein_set ────────────────────────────────────────────────────

class TestFilterProteinSet:

    def test_processes_all_proteins(self, m3_candidates_dir):
        cand_dir = str(m3_candidates_dir / "all_candidates")
        results = filter_protein_set(cand_dir)

        assert set(results.keys()) == {"protA", "protB"}

    def test_selects_correct_candidates(self, m3_candidates_dir):
        cand_dir = str(m3_candidates_dir / "all_candidates")
        results = filter_protein_set(cand_dir)

        # protA: candidate 2 (risk=0.31)
        assert results["protA"]["selected_risk"] == pytest.approx(0.31, abs=1e-4)
        assert results["protA"]["selected_sequence"] == "ACDEYTH"

        # protB: candidate 0 (risk=0.12)
        assert results["protB"]["selected_risk"] == pytest.approx(0.12, abs=1e-4)
        assert results["protB"]["selected_sequence"] == "MKTLLILAVL"

    def test_empty_dir_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        results = filter_protein_set(str(empty_dir))
        assert results == {}
