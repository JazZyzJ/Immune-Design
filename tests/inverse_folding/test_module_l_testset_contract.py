"""Module L1 contract tests — test set curation and overlap exclusion.

TDD gates from PLAN_IF.md Task L1:
  RED:   train/test overlap slips through the curation filter.
  GREEN: test-set builder deterministically flags overlaps and writes
         explicit rejection reasons.
"""

import json

import pytest

from inverse_folding.evaluation.test_set import (
    CurationConfig,
    CurationLedger,
    check_overlap,
    filter_candidate,
    load_cath_chain_names,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def default_cfg():
    return CurationConfig()


@pytest.fixture
def ledger():
    return CurationLedger()


@pytest.fixture
def cath_names():
    return {"1ABCA00", "2XYZB01", "3PQRC02", "4DEFG00"}


@pytest.fixture
def cath_splits_file(tmp_path):
    splits = {
        "train": ["1ABCA00", "2XYZB01"],
        "validation": ["3PQRC02"],
        "test": ["4DEFG00"],
    }
    path = tmp_path / "chain_set_splits.json"
    path.write_text(json.dumps(splits))
    return str(path)


# ── L1.1: Overlap detection ─────────────────────────────────────────────────

class TestOverlapDetection:

    def test_exact_match_detected(self, cath_names):
        assert check_overlap("1ABCA00", cath_names) == "1ABCA00"

    def test_pdb_code_match_detected(self, cath_names):
        # "1ABC_X" should match "1ABCA00" by PDB code
        result = check_overlap("1ABC_X", cath_names)
        assert result is not None
        assert result.startswith("1ABC")

    def test_no_overlap_returns_none(self, cath_names):
        assert check_overlap("9ZZZA00", cath_names) is None

    def test_case_insensitive_pdb_code(self, cath_names):
        result = check_overlap("1abc_x", cath_names)
        assert result is not None


# ── L1.2: Candidate filtering ───────────────────────────────────────────────

class TestCandidateFiltering:

    def test_valid_candidate_accepted(self, default_cfg, ledger):
        assert filter_candidate("test_001", 200, 1.8, 1, default_cfg, ledger)
        assert len(ledger.rejected) == 0

    def test_too_short_rejected(self, default_cfg, ledger):
        assert not filter_candidate("test_001", 50, 1.8, 1, default_cfg, ledger)
        assert ledger.rejected[0]["reason"].startswith("too_short")

    def test_too_long_rejected(self, default_cfg, ledger):
        assert not filter_candidate("test_001", 600, 1.8, 1, default_cfg, ledger)
        assert ledger.rejected[0]["reason"].startswith("too_long")

    def test_low_resolution_rejected(self, default_cfg, ledger):
        assert not filter_candidate("test_001", 200, 3.0, 1, default_cfg, ledger)
        assert ledger.rejected[0]["reason"].startswith("low_resolution")

    def test_multi_chain_rejected(self, default_cfg, ledger):
        assert not filter_candidate("test_001", 200, 1.8, 2, default_cfg, ledger)
        assert ledger.rejected[0]["reason"].startswith("multi_chain")

    def test_boundary_length_min_accepted(self, default_cfg, ledger):
        assert filter_candidate("test_001", 100, 1.8, 1, default_cfg, ledger)

    def test_boundary_length_max_accepted(self, default_cfg, ledger):
        assert filter_candidate("test_001", 500, 1.8, 1, default_cfg, ledger)

    def test_none_resolution_accepted(self, default_cfg, ledger):
        """Missing resolution should not reject (NMR structures)."""
        assert filter_candidate("test_001", 200, None, 1, default_cfg, ledger)


# ── L1.3: CATH name loading ─────────────────────────────────────────────────

class TestCATHNameLoading:

    def test_default_loads_train_only(self, cath_splits_file):
        names = load_cath_chain_names(cath_splits_file)
        assert len(names) == 2  # only train split
        assert "1ABCA00" in names
        assert "2XYZB01" in names
        # val/test should NOT be included
        assert "3PQRC02" not in names
        assert "4DEFG00" not in names

    def test_explicit_train_only(self, cath_splits_file):
        names = load_cath_chain_names(cath_splits_file, split_keys=["train"])
        assert len(names) == 2

    def test_all_splits_when_requested(self, cath_splits_file):
        names = load_cath_chain_names(
            cath_splits_file, split_keys=["train", "validation", "test"]
        )
        assert len(names) == 4


# ── L1.4: Ledger provenance ─────────────────────────────────────────────────

class TestLedgerProvenance:

    def test_ledger_summary_counts(self, ledger):
        ledger.accept("p1")
        ledger.reject("p2", "too_short")
        ledger.reject("p3", "too_long")
        ledger.exclude_overlap("p4", "1ABCA00")

        s = ledger.summary()
        assert s["n_accepted"] == 1
        assert s["n_rejected"] == 2
        assert s["n_overlap_excluded"] == 1

    def test_ledger_to_json(self, ledger, tmp_path):
        ledger.accept("p1")
        ledger.reject("p2", "too_short (50 < 100)")
        path = str(tmp_path / "ledger.json")
        ledger.to_json(path)

        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["n_accepted"] == 1
        assert data["summary"]["n_rejected"] == 1
        assert "too_short" in data["summary"]["rejection_reasons"]

    def test_rejection_reason_grouping(self, ledger):
        ledger.reject("p1", "too_short (50 < 100)")
        ledger.reject("p2", "too_short (80 < 100)")
        ledger.reject("p3", "too_long (600 > 500)")

        s = ledger.summary()
        assert s["rejection_reasons"]["too_short"] == 2
        assert s["rejection_reasons"]["too_long"] == 1
