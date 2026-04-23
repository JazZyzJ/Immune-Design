"""Phase C1 amplification contract tests."""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.amplification import (
    amplification_factor,
    normalize_h_values,
    shuffle_h_values,
)


def test_amplification_is_always_ge_one():
    h = np.linspace(-10.0, 10.0, 200, dtype=np.float32)
    configs = [
        {"form": "constant_one", "g_max_cap": 20.0},
        {"form": "linear_clamp", "c": 2.0, "mu": 0.0, "g_max_cap": 20.0},
        {"form": "sigmoid", "c": 3.0, "mu": 0.0, "kappa": 2.0, "g_max_cap": 20.0},
        {"form": "power", "c": 1.5, "mu": 0.0, "p": 2.0, "g_max_cap": 20.0},
    ]
    for cfg in configs:
        g = amplification_factor(h, cfg)
        assert np.all(g >= 1.0), cfg["form"]


def test_constant_one_is_exactly_one():
    h = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    g = amplification_factor(h, {"form": "constant_one", "g_max_cap": 20.0})
    assert np.all(g == 1.0)


def test_sigmoid_has_expected_asymptotes():
    cfg = {"form": "sigmoid", "c": 4.0, "mu": 0.0, "kappa": 5.0, "g_max_cap": 20.0}
    low = amplification_factor(np.array([-100.0], dtype=np.float32), cfg)[0]
    high = amplification_factor(np.array([100.0], dtype=np.float32), cfg)[0]
    assert np.isclose(low, 1.0, atol=1e-4)
    assert np.isclose(high, 5.0, atol=1e-4)


def test_power_respects_threshold():
    cfg = {"form": "power", "c": 2.0, "mu": 1.0, "p": 2.0, "g_max_cap": 20.0}
    h = np.array([-1.0, 0.5, 1.0, 2.0], dtype=np.float32)
    g = amplification_factor(h, cfg)
    assert np.allclose(g[:3], 1.0)
    assert g[3] > 1.0


def test_normalize_and_shuffle_h_values():
    h_raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stats = {"corpus_h_raw_mean": 2.5, "corpus_h_raw_std": 0.5}
    normalized = normalize_h_values(h_raw, stats)
    assert np.allclose(normalized, np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32))

    shuffled_1 = shuffle_h_values(h_raw, seed=7)
    shuffled_2 = shuffle_h_values(h_raw, seed=7)
    assert np.array_equal(shuffled_1, shuffled_2)
    assert sorted(shuffled_1.tolist()) == sorted(h_raw.tolist())
