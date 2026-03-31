"""L3 contract tests: Tier 2 pre-screening and CATH topology sampling.

RED targets:
  1. Pre-screener rejects protein with only 3 strong NMP windows (threshold 5).
  2. Pre-screener passes protein meeting both NMP and head thresholds.
  3. Topology sampler respects max_per_topology.
  4. Topology sampler respects target_total.
  5. CATH topology assignment returns code or None.
"""

import pandas as pd
import pytest

from inverse_folding.evaluation.prescreen import (
    apply_tier2_filters,
    PrescreenResult,
)
from inverse_folding.evaluation.cath_topology import (
    assign_topology,
    sample_diverse,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _nmp_result(n_strong: int, mean_best_rank: float) -> dict:
    return {"n_strong_windows": n_strong, "mean_best_rank": mean_best_rank}


def _head_result(global_risk: float, n_hotspot: int) -> dict:
    return {"global_risk": global_risk, "n_hotspot_positions": n_hotspot}


# ── L3-RED-1: NMP threshold enforcement ─────────────────────────────────────


class TestTier2Filters:
    def test_rejects_below_nmp_threshold(self):
        """3 strong windows < 5 threshold → reject."""
        result = apply_tier2_filters(
            nmp_result=_nmp_result(n_strong=3, mean_best_rank=1.0),
            head_result=_head_result(global_risk=0.8, n_hotspot=5),
            min_strong_windows=5,
            risk_median_threshold=0.5,
        )
        assert result.passed is False
        assert "nmp" in result.reason.lower()

    def test_rejects_below_risk_threshold(self):
        """Head risk below median threshold → reject."""
        result = apply_tier2_filters(
            nmp_result=_nmp_result(n_strong=10, mean_best_rank=0.8),
            head_result=_head_result(global_risk=0.2, n_hotspot=1),
            min_strong_windows=5,
            risk_median_threshold=0.5,
        )
        assert result.passed is False
        assert "risk" in result.reason.lower()

    def test_passes_both_thresholds(self):
        """Both conditions met → pass."""
        result = apply_tier2_filters(
            nmp_result=_nmp_result(n_strong=8, mean_best_rank=0.9),
            head_result=_head_result(global_risk=0.6, n_hotspot=4),
            min_strong_windows=5,
            risk_median_threshold=0.5,
        )
        assert result.passed is True

    def test_boundary_exactly_5_windows_passes(self):
        """Exactly 5 windows at threshold → pass."""
        result = apply_tier2_filters(
            nmp_result=_nmp_result(n_strong=5, mean_best_rank=1.0),
            head_result=_head_result(global_risk=0.6, n_hotspot=3),
            min_strong_windows=5,
            risk_median_threshold=0.5,
        )
        assert result.passed is True

    def test_boundary_exactly_median_risk_passes(self):
        """Risk exactly at threshold → pass."""
        result = apply_tier2_filters(
            nmp_result=_nmp_result(n_strong=6, mean_best_rank=1.0),
            head_result=_head_result(global_risk=0.5, n_hotspot=2),
            min_strong_windows=5,
            risk_median_threshold=0.5,
        )
        assert result.passed is True


# ── L3-RED-2: CATH topology assignment ──────────────────────────────────────


class TestCATHTopology:
    def test_known_domain_returns_topology(self, tmp_path):
        # Minimal CATH domain list fixture (CathDomainList format):
        # domain_name class arch topology homology ...
        cath_file = tmp_path / "cath-domain-list.txt"
        cath_file.write_text(
            "# CATH domain list\n"
            "1abcA00    1    10    490    10    ...\n"
            "2xyzB00    2    60    40     10    ...\n"
        )
        result = assign_topology("1ABC_A", str(cath_file))
        assert result == "1.10.490"

    def test_unknown_domain_returns_none(self, tmp_path):
        cath_file = tmp_path / "cath-domain-list.txt"
        cath_file.write_text(
            "# CATH domain list\n"
            "1abcA00    1    10    490    10    ...\n"
        )
        result = assign_topology("9ZZZ_X", str(cath_file))
        assert result is None


# ── L3-RED-3: Diversity sampling ────────────────────────────────────────────


class TestDiversitySampling:
    def _make_df(self, n: int, topologies: list) -> pd.DataFrame:
        """Create a candidates DataFrame with assigned topologies."""
        return pd.DataFrame({
            "protein_id": [f"P{i}" for i in range(n)],
            "cath_topology": topologies,
            "netmhciipan_n_strong": list(range(n, 0, -1)),  # descending
        })

    def test_max_per_topology_respected(self):
        df = self._make_df(6, ["1.10.490"] * 4 + ["2.60.40"] * 2)
        result = sample_diverse(df, max_per_topology=2, target_total=50)
        topo_counts = result["cath_topology"].value_counts()
        assert topo_counts.max() <= 2

    def test_target_total_respected(self):
        df = self._make_df(100, [f"{i}.10.20" for i in range(100)])
        result = sample_diverse(df, max_per_topology=2, target_total=10)
        assert len(result) <= 10

    def test_prioritizes_higher_nmp_signal(self):
        df = self._make_df(4, ["1.10.490"] * 4)
        result = sample_diverse(df, max_per_topology=2, target_total=50)
        # Should keep the two with highest netmhciipan_n_strong
        assert set(result["protein_id"]) == {"P0", "P1"}

    def test_null_topology_grouped_separately(self):
        df = self._make_df(4, [None, None, "1.10.490", "1.10.490"])
        result = sample_diverse(df, max_per_topology=1, target_total=50)
        # 1 from null group, 1 from 1.10.490
        assert len(result) == 2

    def test_architecture_class_coverage(self):
        """Ensure at least one candidate from each CATH class when available."""
        # 6 candidates: 3 from class 1, 2 from class 2, 1 from class 3
        # All class 1 have the highest NMP scores
        df = pd.DataFrame({
            "protein_id": ["A", "B", "C", "D", "E", "F"],
            "cath_topology": [
                "1.10.10", "1.20.30", "1.30.40",  # class 1 (alpha)
                "2.10.20", "2.60.40",              # class 2 (beta)
                "3.40.50",                          # class 3 (alpha/beta)
            ],
            "netmhciipan_n_strong": [100, 90, 80, 10, 5, 1],
        })
        # With max_per_topology=1 and target_total=4, a naive sampler
        # would pick A, B, C, D (all high NMP) = no class 3.
        result = sample_diverse(df, max_per_topology=1, target_total=4)
        classes_present = set()
        for topo in result["cath_topology"]:
            classes_present.add(topo.split(".")[0])
        # Must have class 3 represented despite low NMP score
        assert "3" in classes_present
