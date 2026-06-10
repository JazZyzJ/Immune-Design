"""Stage B typed-actionability pure operators (PLAN_RF_UNI_CTRL.md §"Mathematical
Definitions", Tasks B1/B3).

This module holds the residue-level ``A_i(t)`` field math as pure functions:
max-covering window projection, the anchored SoftOR union, the
proposal-envelope ``peak * consistency`` aggregation, the fresh-evidence /
memory firewall, the target-vs-pressure split, and the global-pressure mass.

Design constraints carried from the PLAN:

* Projection is **max-covering window** (not the head's internal LME), to keep
  focal hotspots visible to D2 targeting.
* ``SoftOR`` is anchored so ``SoftOR(0,...) = 0`` and ``SoftOR(x, 0,...) = x``;
  any single positive channel can make a residue actionable, but the union
  grows sublinearly versus the arithmetic sum.
* The memory firewall is enforced by signature: ``update_memory`` accepts only
  ``previous_b_mem``, ``e_fresh``, and ``half_life_refreshes``. It never accepts
  ``v_target``, ``u_pressure``, or D3 ``m_i``.

All residue arrays are shape ``(L,)`` with ``L`` the protein length.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Projection + SoftOR (Task B1)
# ---------------------------------------------------------------------------


def max_covering_window_projection(
    *,
    length: int,
    windows: Sequence[Mapping[str, float | int]],
) -> np.ndarray:
    """Per residue, the max ``score`` over windows whose ``[start, end)`` covers it.

    Residues covered by no window are 0.0. This is the PLAN's
    ``Proj_i(z_w) = max_{w: i in span(w)} z_w`` used for ``b_cur`` / ``b_env``;
    it deliberately differs from the head's window→residue LME projection so
    that a single focal high-risk window is not averaged away.
    """
    values = np.full(int(length), -np.inf, dtype=float)
    for window in windows:
        start = int(window["start"])
        end = int(window["end"])
        score = float(window["score"])
        values[start:end] = np.maximum(values[start:end], score)
    values[~np.isfinite(values)] = 0.0
    return values


def soft_or(values: Iterable[float], *, tau: float) -> float:
    """Anchored soft union ``tau * log(1 + sum_k (exp(x_k/tau) - 1))``.

    Properties (asserted by tests):
    ``SoftOR(0,...) = 0``; ``SoftOR(x, 0,...) = x``; ``SoftOR(x, y) >= max(x, y)``;
    sublinear versus ``x + y``. Expects nonnegative evidence.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < -1e-12):
        raise ValueError("soft_or expects nonnegative evidence values")
    if float(np.max(arr)) == 0.0:
        return 0.0
    return float(tau * np.log1p(np.sum(np.expm1(arr / tau))))
