"""Stage B typed-actionability pure operators (PLAN_RF_UNI_CTRL.md Task B1/B3).

These cover the residue-level ``A_i(t)`` field math in isolation from the
controller: max-covering projection, the anchored SoftOR union, the
proposal-envelope peak*consistency aggregation, the fresh-evidence /
memory firewall, and the target-vs-pressure split. Controller integration
is exercised separately in the D1/D2-D3 controller tests.
"""

from __future__ import annotations

import numpy as np

from inverse_folding.reference_flow.actionability import (
    max_covering_window_projection,
    soft_or,
)


def test_max_covering_window_projection_preserves_focal_peak():
    windows = [
        {"start": 0, "end": 3, "score": 0.2},
        {"start": 2, "end": 5, "score": 1.7},
    ]
    out = max_covering_window_projection(length=5, windows=windows)
    np.testing.assert_allclose(out, [0.2, 0.2, 1.7, 1.7, 1.7])


def test_soft_or_zero_single_and_multi_source_properties():
    assert soft_or([0.0, 0.0, 0.0], tau=0.5) == 0.0
    assert np.isclose(soft_or([1.2, 0.0], tau=0.5), 1.2)
    both = soft_or([1.2, 0.8], tau=0.5)
    assert both >= 1.2
    assert both < 2.0
