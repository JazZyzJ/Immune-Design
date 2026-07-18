"""TDD for the NoD-difficulty-uniform pilot selector (pure parts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_pilot_difficulty import (
    per_protein_strong_frac,
    per_protein_head,
    uniform_bin_sample,
)


def _nmp(n_prot=4, n_design=8):
    rows = []
    for pi in range(n_prot):
        for di in range(n_design):
            rows.append({"protein_id": f"p{pi}", "design_idx": di,
                         # strong_frac = di/10 so half-A (0..3) median differs from half-B (4..7)
                         "n_strong_binders": di, "n_windows_scored": 10})
    return pd.DataFrame(rows)


def test_per_protein_strong_frac_uses_only_requested_designs():
    nmp = _nmp()
    a = per_protein_strong_frac(nmp, [0, 1, 2, 3])
    b = per_protein_strong_frac(nmp, [4, 5, 6, 7])
    # half-A median of {0,1,2,3}/10 = 0.15 ; half-B median of {4,5,6,7}/10 = 0.55
    assert a["p0"] == pytest.approx(0.15) and b["p0"] == pytest.approx(0.55)
    # disjoint halves give different difficulty vs baseline (the whole point of the split)
    assert (a < b).all()


def test_per_protein_head_medians_over_designs():
    head = pd.DataFrame([{"protein_id": "p0", "design_idx": d, "global_risk": float(d)} for d in range(8)])
    assert per_protein_head(head, [0, 1, 2, 3])["p0"] == 1.5


def test_uniform_bin_sample_equal_per_bin_and_monotone():
    diff = pd.Series({f"p{i}": float(i) for i in range(100)})  # 100 candidates, difficulty 0..99
    sel = uniform_bin_sample(diff, n_total=20, n_bins=10, seed=42)
    assert len(sel) == 20
    # exactly 2 per bin (20/10), bins 0..9 all present
    assert sel["bin"].value_counts().to_dict() == {b: 2 for b in range(10)}
    # bins are difficulty-monotone: max difficulty of bin b < min of bin b+1
    for b in range(9):
        assert sel.loc[sel.bin == b, "difficulty"].max() < sel.loc[sel.bin == b + 1, "difficulty"].min()


def test_uniform_bin_sample_remainder_goes_to_hardest_bins():
    diff = pd.Series({f"p{i}": float(i) for i in range(100)})
    sel = uniform_bin_sample(diff, n_total=25, n_bins=10, seed=0)
    assert len(sel) == 25
    counts = sel["bin"].value_counts().sort_index().to_dict()
    # base 2/bin + 5 remainder -> hardest bins 5..9 get 3, easiest 0..4 get 2
    assert [counts[b] for b in range(10)] == [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]


def test_uniform_bin_sample_is_seed_deterministic():
    diff = pd.Series({f"p{i}": float(i % 7) for i in range(60)})  # ties -> stable first-rank bins
    s1 = uniform_bin_sample(diff, 20, 10, seed=7)
    s2 = uniform_bin_sample(diff, 20, 10, seed=7)
    assert list(s1.index) == list(s2.index)
