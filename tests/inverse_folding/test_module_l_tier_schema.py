"""L0 contract tests: per-protein tier schema and validation.

RED targets:
  1. Schema validator rejects entries missing required fields.
  2. Tier-specific nullable rules are enforced:
     - Tier 1 requires experimental_epitopes_json
     - Tier 2 requires cath_topology
     - Tier 3 requires literature_evidence
  3. Validator accepts fully populated entries.
  4. Invalid tier value is rejected.
"""

import pytest

from inverse_folding.evaluation.schema import (
    EvalSchemaError,
    TEST_PROTEIN_COLUMNS,
    TIER_REQUIRED_FIELDS,
    validate_test_protein_entry,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_entry(tier: int, **overrides) -> dict:
    """Build a minimal valid test protein entry for the given tier."""
    base = {
        "protein_id": "1ABC_A",
        "tier": tier,
        "sequence": "MKTLLILAVL",
        "sequence_length": 10,
        "pdb_path": "pdbs/1ABC_A.pdb",
        "resolution": 1.8,
        "cath_overlap_flag": False,
        "cath_overlap_id": None,
        "head_train_overlap_flag": False,
        "netmhciipan_n_strong": 5,
        "netmhciipan_mean_best_rank": 1.2,
        "head_global_risk": 0.45,
        "head_n_hotspot": 3,
        "cath_topology": None,
        "experimental_epitopes_json": None,
        "literature_evidence": None,
        "selection_reason": "test fixture",
    }
    # Set tier-specific required fields to non-null
    if tier == 1:
        base["experimental_epitopes_json"] = '[{"start_0b": 5, "end_0b": 10}]'
    elif tier == 2:
        base["cath_topology"] = "1.10.490"
    elif tier == 3:
        base["literature_evidence"] = "Bartelds et al. 2011"
    base.update(overrides)
    return base


# ── L0-RED-1: Missing required fields ───────────────────────────────────────


class TestMissingFields:
    """Validator must reject entries missing any universally required field."""

    @pytest.mark.parametrize("drop_field", [
        "protein_id",
        "tier",
        "sequence",
        "sequence_length",
        "pdb_path",
        "cath_overlap_flag",
        "head_train_overlap_flag",
        "netmhciipan_n_strong",
        "netmhciipan_mean_best_rank",
        "head_global_risk",
        "head_n_hotspot",
        "selection_reason",
    ])
    def test_missing_universal_field_rejected(self, drop_field: str):
        entry = _make_entry(tier=1)
        del entry[drop_field]
        with pytest.raises(EvalSchemaError, match=drop_field):
            validate_test_protein_entry(entry)

    def test_completely_empty_entry_rejected(self):
        with pytest.raises(EvalSchemaError):
            validate_test_protein_entry({})


# ── L0-RED-2: Tier-specific nullable rules ───────────────────────────────────


class TestTierSpecificNullable:
    """Tier-specific fields must be non-null for their tier, nullable for others."""

    def test_tier1_requires_experimental_epitopes(self):
        entry = _make_entry(tier=1, experimental_epitopes_json=None)
        with pytest.raises(EvalSchemaError, match="experimental_epitopes_json"):
            validate_test_protein_entry(entry)

    def test_tier2_requires_cath_topology(self):
        entry = _make_entry(tier=2, cath_topology=None)
        with pytest.raises(EvalSchemaError, match="cath_topology"):
            validate_test_protein_entry(entry)

    def test_tier2_skip_diversity_allows_null_cath_topology(self):
        entry = _make_entry(
            tier=2,
            cath_topology=None,
            selection_reason="pre-screened-no-diversity",
        )
        validate_test_protein_entry(entry)  # should NOT raise

    def test_tier3_requires_literature_evidence(self):
        entry = _make_entry(tier=3, literature_evidence=None)
        with pytest.raises(EvalSchemaError, match="literature_evidence"):
            validate_test_protein_entry(entry)

    def test_tier1_allows_null_cath_topology(self):
        entry = _make_entry(tier=1)
        assert entry["cath_topology"] is None
        validate_test_protein_entry(entry)  # should NOT raise

    def test_tier2_allows_null_experimental_epitopes(self):
        entry = _make_entry(tier=2)
        assert entry["experimental_epitopes_json"] is None
        validate_test_protein_entry(entry)  # should NOT raise

    def test_tier3_allows_null_cath_topology(self):
        entry = _make_entry(tier=3)
        assert entry["cath_topology"] is None
        validate_test_protein_entry(entry)  # should NOT raise


# ── L0-RED-3: Valid entries accepted ─────────────────────────────────────────


class TestValidEntries:
    """Fully populated entries for each tier must pass validation."""

    @pytest.mark.parametrize("tier", [1, 2, 3])
    def test_valid_entry_accepted(self, tier: int):
        entry = _make_entry(tier=tier)
        result = validate_test_protein_entry(entry)
        assert result == entry  # passthrough

    def test_extra_fields_preserved(self):
        entry = _make_entry(tier=1, custom_note="extra info")
        result = validate_test_protein_entry(entry)
        assert result["custom_note"] == "extra info"


# ── L0-RED-4: Invalid tier values ───────────────────────────────────────────


class TestInvalidTier:
    """Tier must be 1, 2, or 3."""

    @pytest.mark.parametrize("bad_tier", [0, 4, -1, "high", None])
    def test_invalid_tier_rejected(self, bad_tier):
        entry = _make_entry(tier=1)
        entry["tier"] = bad_tier
        with pytest.raises(EvalSchemaError, match="tier"):
            validate_test_protein_entry(entry)


# ── L0-RED-5: Frozen column set completeness ─────────────────────────────────


class TestFrozenColumnSet:
    """TEST_PROTEIN_COLUMNS must contain exactly the 17 planned fields."""

    EXPECTED_FIELDS = {
        "protein_id", "tier", "sequence", "sequence_length", "pdb_path",
        "resolution", "cath_overlap_flag", "cath_overlap_id",
        "head_train_overlap_flag", "netmhciipan_n_strong",
        "netmhciipan_mean_best_rank", "head_global_risk", "head_n_hotspot",
        "cath_topology", "experimental_epitopes_json",
        "literature_evidence", "selection_reason",
    }

    def test_column_set_matches_plan(self):
        assert TEST_PROTEIN_COLUMNS == self.EXPECTED_FIELDS

    def test_tier_required_fields_mapping(self):
        assert 1 in TIER_REQUIRED_FIELDS
        assert 2 in TIER_REQUIRED_FIELDS
        assert 3 in TIER_REQUIRED_FIELDS
        assert "experimental_epitopes_json" in TIER_REQUIRED_FIELDS[1]
        assert "cath_topology" in TIER_REQUIRED_FIELDS[2]
        assert "literature_evidence" in TIER_REQUIRED_FIELDS[3]
