"""Tests for Tier 2 v2 NMP-only density-stratified selection (PLAN_DATA_SEL §12).

Covers:
  - compute_coverage_fraction: residue-coverage density from NMP windows
  - sample_uniform_bins / sample_gaussian_bins: density-stratified subsampling
"""

import numpy as np
import pandas as pd
import pytest

from inverse_folding.evaluation.immunogenicity import compute_coverage_fraction
from inverse_folding.evaluation.sampling import (
    sample_uniform_bins,
    sample_gaussian_bins,
)


# ── compute_coverage_fraction ────────────────────────────────────────────────

def _windows(rows):
    """rows: list of (pos, peptide, rank_EL) → window DataFrame."""
    return pd.DataFrame(
        [{"pos": p, "peptide": pep, "rank_EL": r} for p, pep, r in rows]
    )


def test_coverage_fraction_single_strong_window():
    # one strong 5-mer at pos 0 in a length-10 protein → 5/10
    df = _windows([(0, "ABCDE", 1.0)])
    assert compute_coverage_fraction(df, seq_len=10) == pytest.approx(0.5)


def test_coverage_fraction_unions_overlapping_windows():
    # [0,5) ∪ [3,8) = [0,8) = 8 residues / 10
    df = _windows([(0, "ABCDE", 1.0), (3, "FGHIJ", 0.5)])
    assert compute_coverage_fraction(df, seq_len=10) == pytest.approx(0.8)


def test_coverage_fraction_ignores_weak_windows():
    # second window is weak (rank >= 2.0) → only [0,5) counts
    df = _windows([(0, "ABCDE", 1.0), (5, "FGHIJ", 9.0)])
    assert compute_coverage_fraction(df, seq_len=10) == pytest.approx(0.5)


def test_coverage_fraction_clips_to_sequence_length():
    # window runs past the end; covered residues clipped to [0, seq_len)
    df = _windows([(8, "ABCDE", 1.0)])  # [8,13) clipped to [8,10) = 2
    assert compute_coverage_fraction(df, seq_len=10) == pytest.approx(0.2)


def test_coverage_fraction_empty_or_no_strong_is_zero():
    assert compute_coverage_fraction(_windows([]), seq_len=10) == 0.0
    assert compute_coverage_fraction(_windows([(0, "ABCDE", 5.0)]), seq_len=10) == 0.0


def test_coverage_fraction_custom_threshold():
    df = _windows([(0, "ABCDE", 4.0)])
    assert compute_coverage_fraction(df, seq_len=10, strong_threshold=5.0) == pytest.approx(0.5)
    assert compute_coverage_fraction(df, seq_len=10, strong_threshold=2.0) == 0.0


# ── stratified sampling ──────────────────────────────────────────────────────

def _pool(values):
    return pd.DataFrame({
        "protein_id": [f"P{i}" for i in range(len(values))],
        "cov": list(values),
    })


def test_uniform_bins_equal_counts_when_rich():
    # 4 bins over [0,1), 250 per bin available, want 400 total → ~100/bin
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.uniform(lo, lo + 0.25, 250) for lo in (0.0, 0.25, 0.5, 0.75)])
    df = _pool(vals)
    sel, hist = sample_uniform_bins(df, "cov", n_target=400, n_bins=4, seed=1)
    assert len(sel) == 400
    counts = hist["n_selected"].to_numpy()
    # near-uniform: every bin within ±1 of 100
    assert counts.max() - counts.min() <= 1


def test_uniform_bins_caps_thin_bin_and_redistributes():
    # bin 3 (high) has only 5 members; uniform must take all 5 and still reach n_target
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        rng.uniform(0.0, 0.25, 200),
        rng.uniform(0.25, 0.5, 200),
        rng.uniform(0.5, 0.75, 200),
        rng.uniform(0.75, 1.0, 5),
    ])
    df = _pool(vals)
    sel, hist = sample_uniform_bins(df, "cov", n_target=300, n_bins=4, seed=1)
    assert len(sel) == 300                                  # deficit redistributed
    assert (hist["n_selected"] <= hist["n_available"]).all()  # never over-draw
    # the thinnest bin (< uniform target of 75) is fully consumed, not exceeded
    thin = hist.loc[hist["n_available"].idxmin()]
    assert thin["n_available"] < 75
    assert thin["n_selected"] == thin["n_available"]


def test_uniform_bins_deterministic():
    rng = np.random.default_rng(0)
    df = _pool(rng.uniform(0, 1, 1000))
    a, _ = sample_uniform_bins(df, "cov", n_target=200, n_bins=10, seed=7)
    b, _ = sample_uniform_bins(df, "cov", n_target=200, n_bins=10, seed=7)
    assert list(a) == list(b)


def test_gaussian_bins_unimodal_center_heavier_than_tails():
    rng = np.random.default_rng(0)
    df = _pool(rng.uniform(0, 1, 5000))
    sel, hist = sample_gaussian_bins(
        df, "cov", n_target=1000, n_bins=10, peak_to_tail=3.0, min_per_bin=20, seed=3,
    )
    counts = hist.sort_values("bin_lo")["n_selected"].to_numpy()
    center = counts[len(counts) // 2 - 1: len(counts) // 2 + 1].mean()
    tails = (counts[0] + counts[-1]) / 2
    assert center > tails                         # unimodal hump
    assert counts.min() >= 20                     # min-per-bin floor honored
    assert abs(len(sel) - 1000) <= 10             # hits target within rounding


def test_gaussian_bins_respect_pool_availability():
    # high-density bin is thin; gaussian must not request more than available
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.uniform(0.0, 0.8, 4000), rng.uniform(0.8, 1.0, 10)])
    df = _pool(vals)
    sel, hist = sample_gaussian_bins(
        df, "cov", n_target=1000, n_bins=10, min_per_bin=5, seed=3,
    )
    for _, row in hist.iterrows():
        assert row["n_selected"] <= row["n_available"]
