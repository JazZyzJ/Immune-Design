"""Phase C1 schedule contract tests."""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.schedule import (
    base_schedule_derivative,
    base_schedule_value,
)


def test_schedule_endpoints_and_monotone_derivative():
    grid = np.linspace(0.001, 0.999, 200)
    for form in ("linear", "cosine", "cubic"):
        assert np.isclose(base_schedule_value(0.0, form), 0.0)
        assert np.isclose(base_schedule_value(1.0, form), 1.0)
        deriv = base_schedule_derivative(grid, form)
        assert np.all(deriv > 0.0), form


def test_schedule_derivative_matches_finite_difference():
    grid = np.linspace(0.01, 0.99, 100)
    eps = 1e-5
    for form in ("linear", "cosine", "cubic"):
        analytic = base_schedule_derivative(grid, form)
        numeric = (
            base_schedule_value(grid + eps, form)
            - base_schedule_value(grid - eps, form)
        ) / (2 * eps)
        assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-4), form
