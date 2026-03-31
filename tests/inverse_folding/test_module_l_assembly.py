"""L5 contract tests: test set assembly and validation.

RED targets:
  1. Assembly rejects protein with missing WT hotspot map.
  2. Assembly rejects duplicate protein_id.
  3. Assembler produces schema-valid parquet.
  4. Summary counts match input tier sizes.
  5. Validate_test_set catches CATH overlap flags that should have been excluded.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from inverse_folding.evaluation.assembly import (
    assemble_test_set,
    validate_test_set,
    AssemblyError,
)
from inverse_folding.evaluation.schema import validate_test_protein_entry


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_tier1_entry(pid: str = "T1_A") -> dict:
    return {
        "protein_id": pid,
        "tier": 1,
        "sequence": "MKTLLILAVL" * 10,
        "sequence_length": 100,
        "pdb_path": f"pdbs/{pid}.pdb",
        "resolution": 1.8,
        "cath_overlap_flag": False,
        "cath_overlap_id": None,
        "head_train_overlap_flag": False,
        "netmhciipan_n_strong": 6,
        "netmhciipan_mean_best_rank": 1.1,
        "head_global_risk": 0.55,
        "head_n_hotspot": 4,
        "cath_topology": None,
        "experimental_epitopes_json": '[{"start_0b": 5, "end_0b": 20}]',
        "literature_evidence": None,
        "selection_reason": "gold standard",
    }


def _make_tier2_entry(pid: str = "T2_A") -> dict:
    return {
        "protein_id": pid,
        "tier": 2,
        "sequence": "ACDEFGHIKL" * 15,
        "sequence_length": 150,
        "pdb_path": f"pdbs/{pid}.pdb",
        "resolution": 2.0,
        "cath_overlap_flag": False,
        "cath_overlap_id": None,
        "head_train_overlap_flag": False,
        "netmhciipan_n_strong": 8,
        "netmhciipan_mean_best_rank": 0.9,
        "head_global_risk": 0.65,
        "head_n_hotspot": 5,
        "cath_topology": "1.10.490",
        "experimental_epitopes_json": None,
        "literature_evidence": None,
        "selection_reason": "pre-screened",
    }


def _make_tier3_entry(pid: str = "T3_A") -> dict:
    return {
        "protein_id": pid,
        "tier": 3,
        "sequence": "MNOPQRSTUV" * 20,
        "sequence_length": 200,
        "pdb_path": f"pdbs/{pid}.pdb",
        "resolution": 2.2,
        "cath_overlap_flag": False,
        "cath_overlap_id": None,
        "head_train_overlap_flag": False,
        "netmhciipan_n_strong": 4,
        "netmhciipan_mean_best_rank": 2.5,
        "head_global_risk": 0.40,
        "head_n_hotspot": 2,
        "cath_topology": None,
        "experimental_epitopes_json": None,
        "literature_evidence": "Bartelds et al. 2011",
        "selection_reason": "therapeutic",
    }


# ── L5-RED-1: Duplicate protein_id rejected ─────────────────────────────────


class TestAssemblyRejectsDuplicates:
    def test_duplicate_within_tier(self):
        entries = [_make_tier1_entry("DUP"), _make_tier1_entry("DUP")]
        with pytest.raises(AssemblyError, match="[Dd]uplicate"):
            assemble_test_set(entries)

    def test_duplicate_across_tiers(self):
        e1 = _make_tier1_entry("CROSS")
        e2 = _make_tier2_entry("CROSS")
        with pytest.raises(AssemblyError, match="[Dd]uplicate"):
            assemble_test_set([e1, e2])


# ── L5-RED-2: Schema-valid output ───────────────────────────────────────────


class TestAssemblyOutput:
    def test_produces_dataframe(self):
        entries = [
            _make_tier1_entry("A"),
            _make_tier2_entry("B"),
            _make_tier3_entry("C"),
        ]
        df = assemble_test_set(entries)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_all_rows_schema_valid(self):
        entries = [
            _make_tier1_entry("A"),
            _make_tier2_entry("B"),
            _make_tier3_entry("C"),
        ]
        df = assemble_test_set(entries)
        for _, row in df.iterrows():
            validate_test_protein_entry(row.to_dict())

    def test_tier_counts_in_summary(self):
        entries = [
            _make_tier1_entry("A"),
            _make_tier1_entry("B"),
            _make_tier2_entry("C"),
            _make_tier3_entry("D"),
        ]
        df = assemble_test_set(entries)
        summary = {
            "tier_1": int((df["tier"] == 1).sum()),
            "tier_2": int((df["tier"] == 2).sum()),
            "tier_3": int((df["tier"] == 3).sum()),
        }
        assert summary == {"tier_1": 2, "tier_2": 1, "tier_3": 1}


# ── L5-RED-3: Validate test set catches issues ──────────────────────────────


class TestValidateTestSet:
    def test_cath_overlap_flag_rejected(self, tmp_path):
        entries = [_make_tier1_entry("A")]
        entries[0]["cath_overlap_flag"] = True
        df = pd.DataFrame(entries)
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)

        errors = validate_test_set(str(path))
        assert any("cath_overlap" in e.lower() for e in errors)

    def test_duplicate_protein_id_detected(self, tmp_path):
        entries = [_make_tier1_entry("DUP"), _make_tier1_entry("DUP")]
        df = pd.DataFrame(entries)
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)

        errors = validate_test_set(str(path))
        assert any("duplicate" in e.lower() for e in errors)

    def test_clean_set_passes(self, tmp_path):
        entries = [
            _make_tier1_entry("A"),
            _make_tier2_entry("B"),
            _make_tier3_entry("C"),
        ]
        df = assemble_test_set(entries)
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)

        errors = validate_test_set(str(path))
        assert len(errors) == 0
