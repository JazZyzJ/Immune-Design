"""Tests for Module M: Classifier Guidance Contract (M0) and Reweighting (M2).

TDD RED → GREEN for:
  M0: guidance config validation (eta grid, K, selection rule, provenance)
  M2: reweighting algorithm (exp(-eta*R), numerical stability, degenerate cases)
"""

import math

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# M0: Guidance Config Contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuidanceConfigContract:
    """M0 TDD gate: guidance validator accepts only the frozen v1 contract
    and rejects incomplete configs."""

    def test_valid_config_accepted(self):
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(
            eta=1.0,
            num_candidates=8,
            seed=42,
        )
        # Should not raise
        validate_guidance_config(cfg)

    def test_eta_zero_accepted(self):
        """eta=0 means no guidance (baseline), should be valid."""
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(eta=0.0, num_candidates=8, seed=42)
        validate_guidance_config(cfg)

    def test_eta_from_frozen_grid_accepted(self):
        """All values in the frozen eta grid must be accepted."""
        from inverse_folding.guidance.config import (
            FROZEN_ETA_GRID,
            GuidanceConfig,
            validate_guidance_config,
        )
        for eta in FROZEN_ETA_GRID:
            cfg = GuidanceConfig(eta=eta, num_candidates=8, seed=42)
            validate_guidance_config(cfg)

    def test_negative_eta_rejected(self):
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            GuidanceConfigError,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(eta=-1.0, num_candidates=8, seed=42)
        with pytest.raises(GuidanceConfigError, match="eta"):
            validate_guidance_config(cfg)

    def test_zero_candidates_rejected(self):
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            GuidanceConfigError,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(eta=1.0, num_candidates=0, seed=42)
        with pytest.raises(GuidanceConfigError, match="num_candidates"):
            validate_guidance_config(cfg)

    def test_selection_rule_frozen_to_resampling(self):
        """v1 selection rule is risk-weighted resampling only."""
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            GuidanceConfigError,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(
            eta=1.0, num_candidates=8, seed=42,
            selection_rule="weighted_logit_avg",  # Explicitly out of scope
        )
        with pytest.raises(GuidanceConfigError, match="selection_rule"):
            validate_guidance_config(cfg)

    def test_default_selection_rule_is_resampling(self):
        from inverse_folding.guidance.config import GuidanceConfig
        cfg = GuidanceConfig(eta=1.0, num_candidates=8, seed=42)
        assert cfg.selection_rule == "risk_weighted_resampling"

    def test_frozen_eta_grid_values(self):
        """Verify the frozen eta grid matches PLAN_IF.md §M."""
        from inverse_folding.guidance.config import FROZEN_ETA_GRID
        assert FROZEN_ETA_GRID == (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

    def test_frozen_default_num_candidates(self):
        from inverse_folding.guidance.config import DEFAULT_NUM_CANDIDATES
        assert DEFAULT_NUM_CANDIDATES == 8

    def test_off_grid_eta_rejected_in_strict_mode(self):
        """eta=3.7 is not in the frozen grid → rejected with enforce_frozen_grid."""
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            GuidanceConfigError,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(eta=3.7, num_candidates=8, seed=42)
        with pytest.raises(GuidanceConfigError, match="frozen eta grid"):
            validate_guidance_config(cfg, enforce_frozen_grid=True)

    def test_on_grid_eta_accepted_in_strict_mode(self):
        from inverse_folding.guidance.config import (
            FROZEN_ETA_GRID,
            GuidanceConfig,
            validate_guidance_config,
        )
        for eta in FROZEN_ETA_GRID:
            cfg = GuidanceConfig(eta=eta, num_candidates=8, seed=42)
            validate_guidance_config(cfg, enforce_frozen_grid=True)

    def test_off_grid_eta_accepted_without_strict(self):
        """Without enforce_frozen_grid, any non-negative eta passes."""
        from inverse_folding.guidance.config import (
            GuidanceConfig,
            validate_guidance_config,
        )
        cfg = GuidanceConfig(eta=3.7, num_candidates=8, seed=42)
        validate_guidance_config(cfg)  # Should not raise


class TestGuidanceProvenance:
    """M0 TDD gate: provenance per design must include required fields."""

    def test_provenance_has_required_fields(self):
        from inverse_folding.guidance.config import REQUIRED_PROVENANCE_FIELDS
        expected = {
            "eta", "seed", "num_candidates", "selection_rule",
            "selected_risk", "selected_seq_hash",
        }
        assert REQUIRED_PROVENANCE_FIELDS == expected

    def test_validate_provenance_accepts_complete(self):
        from inverse_folding.guidance.config import validate_provenance
        prov = {
            "eta": 1.0,
            "seed": 42,
            "num_candidates": 8,
            "selection_rule": "risk_weighted_resampling",
            "selected_risk": 0.35,
            "selected_seq_hash": "abc123def456",
        }
        # Should not raise
        validate_provenance(prov)

    def test_validate_provenance_rejects_missing_field(self):
        from inverse_folding.guidance.config import (
            GuidanceConfigError,
            validate_provenance,
        )
        prov = {
            "eta": 1.0,
            "seed": 42,
            # missing num_candidates and others
        }
        with pytest.raises(GuidanceConfigError, match="Missing provenance"):
            validate_provenance(prov)


# ═══════════════════════════════════════════════════════════════════════════════
# M2: Reweighting Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeWeights:
    """M2 TDD gate: weight computation is numerically stable and correct."""

    def test_uniform_weights_when_eta_zero(self):
        """eta=0 → all candidates equally weighted."""
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.1, 0.5, 0.9])
        weights = compute_weights(risks, eta=0.0)
        np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_lower_risk_gets_higher_weight(self):
        """Higher eta should increasingly prefer low-risk candidates."""
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.1, 0.5, 0.9])
        weights = compute_weights(risks, eta=5.0)
        assert weights[0] > weights[1] > weights[2]
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_weights_sum_to_one(self):
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.2, 0.4, 0.6, 0.8])
        for eta in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            weights = compute_weights(risks, eta=eta)
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_single_candidate_gets_weight_one(self):
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.5])
        weights = compute_weights(risks, eta=5.0)
        np.testing.assert_allclose(weights, [1.0])

    def test_equal_risks_give_uniform_weights(self):
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.3, 0.3, 0.3, 0.3])
        weights = compute_weights(risks, eta=10.0)
        np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25])

    def test_extreme_eta_concentrates_on_best(self):
        """Very large eta should put nearly all weight on the lowest-risk candidate."""
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.1, 0.5, 0.9])
        weights = compute_weights(risks, eta=100.0)
        assert weights[0] > 0.99
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_numerical_stability_large_eta(self):
        """Should not produce NaN or Inf with very large eta."""
        from inverse_folding.guidance.reweighting import compute_weights
        risks = np.array([0.01, 0.5, 0.99])
        weights = compute_weights(risks, eta=1000.0)
        assert np.all(np.isfinite(weights))
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_negative_eta_rejected(self):
        from inverse_folding.guidance.reweighting import compute_weights
        with pytest.raises(ValueError, match="eta"):
            compute_weights(np.array([0.1, 0.5]), eta=-1.0)

    def test_empty_risks_rejected(self):
        from inverse_folding.guidance.reweighting import compute_weights
        with pytest.raises(ValueError, match="empty"):
            compute_weights(np.array([]), eta=1.0)


class TestSelectCandidate:
    """M2 TDD gate: candidate selection is deterministic with seed and
    produces valid provenance."""

    def test_deterministic_with_seed(self):
        """Same risks, eta, seed → same selection."""
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.1, 0.3, 0.5, 0.7])
        sequences = ["ACDE", "FGHI", "KLMN", "PQRS"]

        r1 = select_candidate(risks, sequences, eta=2.0, seed=42)
        r2 = select_candidate(risks, sequences, eta=2.0, seed=42)
        assert r1["selected_index"] == r2["selected_index"]
        assert r1["selected_seq_hash"] == r2["selected_seq_hash"]

    def test_different_seed_can_differ(self):
        """Different seeds may (though not guaranteed) produce different selections."""
        from inverse_folding.guidance.reweighting import select_candidate
        # Use risks that spread the weight enough to allow variation
        risks = np.array([0.3, 0.3, 0.3, 0.3])
        sequences = ["AAAA", "BBBB", "CCCC", "DDDD"]

        results = set()
        for seed in range(100):
            r = select_candidate(risks, sequences, eta=1.0, seed=seed)
            results.add(r["selected_index"])
        # With uniform risks, we should see at least 2 different selections
        assert len(results) >= 2

    def test_returns_provenance_fields(self):
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.2, 0.8])
        sequences = ["ACDE", "FGHI"]
        result = select_candidate(risks, sequences, eta=1.0, seed=42)

        assert "selected_index" in result
        assert "selected_risk" in result
        assert "selected_seq_hash" in result
        assert "weights" in result
        assert "candidate_risks" in result
        # Full provenance: per-candidate hashes and duplicate groups
        assert "candidate_seq_hashes" in result
        assert len(result["candidate_seq_hashes"]) == 2
        assert "duplicate_groups" in result

    def test_duplicate_groups_detected(self):
        """Duplicate sequences should be grouped in provenance."""
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.2, 0.3, 0.5])
        sequences = ["SAME", "SAME", "DIFF"]
        result = select_candidate(risks, sequences, eta=2.0, seed=42)
        # The two "SAME" sequences share a hash → one duplicate group
        groups = result["duplicate_groups"]
        assert len(groups) >= 1
        # The group should contain indices [0, 1]
        same_hash = result["candidate_seq_hashes"][0]
        assert same_hash == result["candidate_seq_hashes"][1]
        assert same_hash != result["candidate_seq_hashes"][2]

    def test_no_duplicates_means_empty_groups(self):
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.2, 0.8])
        sequences = ["AAAA", "BBBB"]
        result = select_candidate(risks, sequences, eta=1.0, seed=42)
        assert result["duplicate_groups"] == {}

    def test_selected_risk_matches_index(self):
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.1, 0.5, 0.9])
        sequences = ["AAA", "BBB", "CCC"]
        result = select_candidate(risks, sequences, eta=2.0, seed=42)
        idx = result["selected_index"]
        assert result["selected_risk"] == risks[idx]

    def test_eta_zero_is_uniform_sampling(self):
        """eta=0 should sample uniformly regardless of risk."""
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.01, 0.99])
        sequences = ["LOW", "HIGH"]
        # Run many trials — both should be selected
        selected = set()
        for seed in range(200):
            r = select_candidate(risks, sequences, eta=0.0, seed=seed)
            selected.add(r["selected_index"])
        assert selected == {0, 1}

    def test_duplicate_sequences_handled(self):
        """Duplicate sequences should not cause errors."""
        from inverse_folding.guidance.reweighting import select_candidate
        risks = np.array([0.3, 0.3, 0.7])
        sequences = ["SAME", "SAME", "DIFF"]
        # Should not raise
        result = select_candidate(risks, sequences, eta=2.0, seed=42)
        assert result["selected_index"] in {0, 1, 2}


class TestSelectCandidateBatch:
    """M2: batch selection for multi-protein sweep."""

    def test_batch_returns_per_protein_results(self):
        from inverse_folding.guidance.reweighting import select_candidates_batch
        protein_risks = {
            "prot_A": np.array([0.1, 0.5, 0.9]),
            "prot_B": np.array([0.2, 0.4]),
        }
        protein_sequences = {
            "prot_A": ["A1", "A2", "A3"],
            "prot_B": ["B1", "B2"],
        }
        results = select_candidates_batch(
            protein_risks, protein_sequences, eta=1.0, seed=42,
        )
        assert set(results.keys()) == {"prot_A", "prot_B"}
        assert "selected_index" in results["prot_A"]
        assert "selected_index" in results["prot_B"]
