"""Tests for Module M1: Epitope-Head Scoring Bridge.

TDD RED → GREEN for:
  - Bridge scores match direct predictor.predict_protein() calls
  - Micro-batching produces consistent results
  - Interface stability with provenance
"""

import numpy as np
import pytest
import torch


# ── Mock predictor (matches InferencePredictor.predict_protein interface) ────

class MockPredictor:
    """Minimal mock matching InferencePredictor.predict_protein output contract."""

    def __init__(self, risk_map: dict[str, float] | None = None):
        self._risk_map = risk_map or {}
        self._call_count = 0

    def predict_protein(self, seq: str, allele_idx: int = 0) -> dict:
        self._call_count += 1
        risk = self._risk_map.get(seq, hash(seq) % 100 / 100.0)
        protein_len = len(seq)
        return {
            "global_risk": risk,
            "residue_hotspot": torch.rand(protein_len),
            "window_logits": [{"start_0b": 0, "end_0b": min(12, protein_len), "k": 12, "z": risk}],
            "meta": {"protein_len": protein_len, "n_windows": 1, "min_k": 12, "max_k": 25},
            "debug": {},
        }


class TestScoreCandidates:
    """M1 TDD gate: bridge scores match direct predictor calls."""

    def test_returns_array_of_risks(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor({"AAA": 0.1, "BBB": 0.5, "CCC": 0.9})
        risks = score_candidates(predictor, ["AAA", "BBB", "CCC"])
        assert isinstance(risks, np.ndarray)
        assert risks.shape == (3,)
        np.testing.assert_allclose(risks, [0.1, 0.5, 0.9])

    def test_matches_direct_predictor_calls(self):
        """Bridge output must exactly match calling predict_protein directly."""
        from inverse_folding.guidance.scoring_bridge import score_candidates
        sequences = ["ACDEFGH", "KLMNPQR", "STVWYAC"]
        predictor = MockPredictor()

        # Direct calls
        direct_risks = []
        for seq in sequences:
            result = predictor.predict_protein(seq)
            direct_risks.append(result["global_risk"])

        # Bridge call (reset call count)
        predictor._call_count = 0
        bridge_risks = score_candidates(predictor, sequences)

        np.testing.assert_allclose(bridge_risks, direct_risks)

    def test_single_candidate(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor({"ACDE": 0.42})
        risks = score_candidates(predictor, ["ACDE"])
        assert risks.shape == (1,)
        np.testing.assert_allclose(risks, [0.42])

    def test_empty_candidates_rejected(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor()
        with pytest.raises(ValueError, match="empty"):
            score_candidates(predictor, [])


class TestScoreCandidatesBatching:
    """M1 TDD gate: micro-batching produces same results as sequential."""

    def test_batch_size_respected(self):
        """Predictor should be called in batches, not all at once."""
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5})
        sequences = ["A", "B", "C", "D", "E"]
        risks = score_candidates(predictor, sequences, batch_size=2)
        np.testing.assert_allclose(risks, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_batch_results_match_sequential(self):
        """Batched scoring must match sequential scoring exactly."""
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor()
        sequences = ["ACDE", "FGHI", "KLMN", "PQRS", "STVW"]

        sequential = score_candidates(predictor, sequences, batch_size=1)
        batched = score_candidates(predictor, sequences, batch_size=3)
        np.testing.assert_allclose(sequential, batched)

    def test_batch_size_larger_than_candidates(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor({"AA": 0.5})
        risks = score_candidates(predictor, ["AA"], batch_size=100)
        np.testing.assert_allclose(risks, [0.5])

    def test_default_batch_size_works(self):
        """Default batch_size should work without explicit argument."""
        from inverse_folding.guidance.scoring_bridge import score_candidates
        predictor = MockPredictor({"X": 0.42})
        risks = score_candidates(predictor, ["X"])
        np.testing.assert_allclose(risks, [0.42])


class TestScoreCandidatesDetailed:
    """M1 TDD gate: detailed output with per-residue hotspot scores."""

    def test_returns_detailed_results(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates_detailed
        predictor = MockPredictor({"AAAA": 0.3, "BBBB": 0.7})
        results = score_candidates_detailed(predictor, ["AAAA", "BBBB"])
        assert len(results) == 2
        assert results[0]["global_risk"] == 0.3
        assert "residue_hotspot" in results[0]
        assert results[0]["residue_hotspot"].shape == (4,)

    def test_detailed_risks_match_simple(self):
        from inverse_folding.guidance.scoring_bridge import (
            score_candidates,
            score_candidates_detailed,
        )
        predictor = MockPredictor()
        sequences = ["ACDEFGH", "KLMNPQR"]

        simple_risks = score_candidates(predictor, sequences)
        detailed = score_candidates_detailed(predictor, sequences)
        detailed_risks = np.array([d["global_risk"] for d in detailed])

        np.testing.assert_allclose(simple_risks, detailed_risks)


class TestGuidedSelection:
    """M1+M2 integration: score candidates then select via reweighting."""

    def test_end_to_end_guidance_step(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        from inverse_folding.guidance.reweighting import select_candidate

        predictor = MockPredictor({
            "LOWRISK": 0.1, "MEDRISK": 0.5,
            "HIRISK1": 0.9, "HIRISK2": 0.95,
        })
        sequences = ["LOWRISK", "MEDRISK", "HIRISK1", "HIRISK2"]
        risks = score_candidates(predictor, sequences)
        result = select_candidate(risks, sequences, eta=5.0, seed=42)

        assert result["selected_index"] in range(4)
        assert result["selected_risk"] == risks[result["selected_index"]]
        assert len(result["weights"]) == 4

    def test_eta_zero_ignores_risks(self):
        from inverse_folding.guidance.scoring_bridge import score_candidates
        from inverse_folding.guidance.reweighting import select_candidate

        predictor = MockPredictor({"LOW": 0.01, "HIGH": 0.99})
        sequences = ["LOW", "HIGH"]
        risks = score_candidates(predictor, sequences)

        # With eta=0, selection should be uniform
        selected = set()
        for seed in range(200):
            r = select_candidate(risks, sequences, eta=0.0, seed=seed)
            selected.add(r["selected_index"])
        assert selected == {0, 1}
