"""Synthetic tests for scripts/analysis/scgr_residue_accuracy.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.scgr_residue_accuracy import (
    compute_residue_accuracy,
    explode_probe_residue_map,
    oracle_residue_map,
    per_protein_residue_spearman,
    region_spearman,
    residue_tail_metrics,
)

_L = 12
_PROTEINS = ["P1", "P2"]
_ARMS = ["fresh", "self_conditioned"]


def _make_samples(*, monotone: bool = True) -> pd.DataFrame:
    rows = []
    for pid in _PROTEINS:
        for didx in (0, 1):
            for arm in _ARMS:
                for k in (0, 1):
                    re = (
                        [float(i) for i in range(_L)]
                        if monotone
                        else [float(_L - i) for i in range(_L)]
                    )
                    rows.append(
                        {
                            "protein_id": pid,
                            "design_idx": didx,
                            "refresh_step": 0,
                            "arm": arm,
                            "sample_idx": k,
                            "residue_excess": re,
                        }
                    )
    return pd.DataFrame(rows)


def _make_oracle() -> pd.DataFrame:
    # oracle hotspot increases with residue index; 3 NoD designs per protein
    rows = []
    for pid in _PROTEINS:
        for didx in (0, 1, 2):
            for i in range(_L):
                rows.append(
                    {"protein_id": pid, "design_idx": didx, "residue_idx": i, "hotspot": float(i)}
                )
    return pd.DataFrame(rows)


def test_explode_reduces_k_and_designs_to_per_protein_residue():
    probe_map = explode_probe_residue_map(_make_samples())
    # 2 proteins x 2 arms x 1 refresh x 12 residues
    assert len(probe_map) == 2 * 2 * 1 * _L
    assert set(probe_map.columns) == {"protein_id", "refresh_step", "arm", "residue_idx", "r_i"}
    g = probe_map[(probe_map.protein_id == "P1") & (probe_map.arm == "fresh")]
    assert sorted(g.residue_idx) == list(range(_L))


def test_oracle_residue_map_aggregates_over_designs():
    oracle = oracle_residue_map(_make_oracle())
    assert set(oracle.columns) == {"protein_id", "residue_idx", "oracle"}
    assert len(oracle) == len(_PROTEINS) * _L
    p1 = oracle[oracle.protein_id == "P1"].sort_values("residue_idx")
    np.testing.assert_allclose(p1["oracle"].to_numpy(), np.arange(_L, dtype=float))


def test_perfect_monotone_probe_gives_spearman_one():
    probe = explode_probe_residue_map(_make_samples(monotone=True))
    oracle = oracle_residue_map(_make_oracle())
    pp = per_protein_residue_spearman(probe, oracle)
    assert (pp["n_residues"] == _L).all()
    np.testing.assert_allclose(pp["spearman"].to_numpy(), 1.0)


def test_anticorrelated_probe_gives_spearman_minus_one():
    probe = explode_probe_residue_map(_make_samples(monotone=False))
    oracle = oracle_residue_map(_make_oracle())
    pp = per_protein_residue_spearman(probe, oracle)
    np.testing.assert_allclose(pp["spearman"].to_numpy(), -1.0)


def test_region_spearman_and_tail_metrics_on_monotone():
    probe = explode_probe_residue_map(_make_samples(monotone=True))
    oracle = oracle_residue_map(_make_oracle())
    reg = region_spearman(probe, oracle, region_size=4, min_regions=3)
    # 3 regions, monotone -> region spearman 1.0
    np.testing.assert_allclose(reg["region_spearman_mean"].dropna().to_numpy(), 1.0)
    tail = residue_tail_metrics(probe, oracle, min_residues=9)
    # perfect ranking -> high tercile recovered, no high-as-low
    np.testing.assert_allclose(tail["recall_at_high_residue"].to_numpy(), 1.0)
    np.testing.assert_allclose(tail["p_low_given_high_residue"].to_numpy(), 0.0)


def test_compute_residue_accuracy_summary_shape():
    summary, per_protein = compute_residue_accuracy(_make_samples(), _make_oracle())
    cells = summary["by_arm_refresh"]
    assert {c["arm"] for c in cells} == set(_ARMS)
    for c in cells:
        assert c["residue_spearman_median"] == pytest.approx(1.0)
        assert "region_spearman_median" in c
        assert c["n_proteins"] == len(_PROTEINS)


def test_missing_residue_excess_column_fails_fast():
    bad = _make_samples().drop(columns=["residue_excess"])
    with pytest.raises(ValueError, match="residue_excess"):
        explode_probe_residue_map(bad)
