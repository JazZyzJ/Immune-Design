"""L1 contract tests: MMseqs2-based sequence identity overlap detection.

RED targets:
  1. Overlap checker catches a 35% identity match (should exclude).
  2. Boundary values: 29% passes, 30% passes, 31% excluded.
  3. Candidate with no CATH hits passes.
  4. Multiple matches → report highest identity.
  5. MMseqs2 binary not found → fail fast.
  6. CLI script run_overlap_filter.py produces correct JSON output.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from inverse_folding.evaluation.overlap import (
    MMseqs2NotFoundError,
    OverlapResult,
    parse_mmseqs_results,
    run_mmseqs_overlap,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


FIXTURE_MMSEQS_OUTPUT_35 = (
    "query1\ttarget_cath1\t0.350\t100\t5\t0\t1\t100\t1\t100\t1e-20\t200\n"
)

FIXTURE_MMSEQS_OUTPUT_BOUNDARY = (
    "query_29\ttarget_a\t0.290\t100\t5\t0\t1\t100\t1\t100\t1e-10\t150\n"
    "query_30\ttarget_b\t0.300\t100\t5\t0\t1\t100\t1\t100\t1e-10\t150\n"
    "query_31\ttarget_c\t0.310\t100\t5\t0\t1\t100\t1\t100\t1e-10\t150\n"
)

FIXTURE_MMSEQS_OUTPUT_MULTI = (
    "query_multi\ttarget_x\t0.250\t100\t5\t0\t1\t100\t1\t100\t1e-5\t80\n"
    "query_multi\ttarget_y\t0.400\t100\t5\t0\t1\t100\t1\t100\t1e-25\t250\n"
    "query_multi\ttarget_z\t0.320\t100\t5\t0\t1\t100\t1\t100\t1e-15\t180\n"
)

FIXTURE_MMSEQS_OUTPUT_EMPTY = ""


# ── L1-RED-1: Parser catches 35% identity ───────────────────────────────────


class TestParseResults:
    def test_35pct_excluded(self):
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_35, threshold=0.3)
        assert "query1" in results
        assert results["query1"].exclude is True
        assert results["query1"].best_identity == pytest.approx(0.35)
        assert results["query1"].best_target == "target_cath1"

    def test_boundary_29_passes(self):
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_BOUNDARY, threshold=0.3)
        assert results["query_29"].exclude is False

    def test_boundary_30_passes(self):
        """30% identity is at the threshold boundary; > 30% excludes, so 30% passes."""
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_BOUNDARY, threshold=0.3)
        assert results["query_30"].exclude is False

    def test_boundary_31_excluded(self):
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_BOUNDARY, threshold=0.3)
        assert results["query_31"].exclude is True

    def test_no_hits_candidate_passes(self):
        """Candidate not appearing in results at all → passes."""
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_EMPTY, threshold=0.3)
        assert len(results) == 0

    def test_multiple_matches_reports_highest(self):
        results = parse_mmseqs_results(FIXTURE_MMSEQS_OUTPUT_MULTI, threshold=0.3)
        r = results["query_multi"]
        assert r.best_identity == pytest.approx(0.40)
        assert r.best_target == "target_y"
        assert r.exclude is True


# ── L1-RED-2: MMseqs2 runner ────────────────────────────────────────────────


class TestRunMmseqsOverlap:
    def test_mmseqs2_not_found_raises(self, tmp_path):
        candidate_fasta = tmp_path / "candidates.fasta"
        cath_fasta = tmp_path / "cath_train.fasta"
        candidate_fasta.write_text(">q1\nMKTLLI\n")
        cath_fasta.write_text(">t1\nMKTLLI\n")

        with pytest.raises(MMseqs2NotFoundError):
            run_mmseqs_overlap(
                candidate_fasta=str(candidate_fasta),
                cath_train_fasta=str(cath_fasta),
                mmseqs_bin="/nonexistent/mmseqs",
                threshold=0.3,
            )

    def test_result_type(self, tmp_path):
        """When mmseqs runs successfully, results are dict of OverlapResult."""
        candidate_fasta = tmp_path / "candidates.fasta"
        cath_fasta = tmp_path / "cath_train.fasta"
        candidate_fasta.write_text(">q1\nMKTLLI\n")
        cath_fasta.write_text(">t1\nMKTLLI\n")

        mock_output = "q1\tt1\t0.500\t6\t0\t0\t1\t6\t1\t6\t1e-5\t50\n"

        with patch(
            "inverse_folding.evaluation.overlap._run_mmseqs_easy_search",
            return_value=mock_output,
        ), patch(
            "shutil.which",
            return_value="/usr/bin/mmseqs",
        ):
            results = run_mmseqs_overlap(
                candidate_fasta=str(candidate_fasta),
                cath_train_fasta=str(cath_fasta),
                mmseqs_bin="mmseqs",
                threshold=0.3,
            )
        assert isinstance(results, dict)
        assert "q1" in results
        assert results["q1"].exclude is True


# ── L1-RED-3: OverlapResult dataclass ────────────────────────────────────────


class TestOverlapResult:
    def test_fields(self):
        r = OverlapResult(
            query_id="q1",
            best_target="t1",
            best_identity=0.35,
            exclude=True,
        )
        assert r.query_id == "q1"
        assert r.best_target == "t1"
        assert r.best_identity == 0.35
        assert r.exclude is True

    def test_to_dict(self):
        r = OverlapResult(
            query_id="q1",
            best_target="t1",
            best_identity=0.35,
            exclude=True,
        )
        d = r.to_dict()
        assert d["query_id"] == "q1"
        assert d["exclude"] is True
