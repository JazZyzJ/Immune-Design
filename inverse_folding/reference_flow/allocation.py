"""Allocation layer — per-residue editability mass ``Φ_i`` (PLAN_PLANNER_SC_GR.md §2).

DISTINCT from the typed-actionability field ``A_i(t)`` in ``actionability.py``:
``Φ_i`` is a mean-normalized editability mass derived from the SC-GR prospective
risk ``r_i``; it only *reweights* the actionability field's ``v_target`` at the
selection seams. It never replaces ``A_i(t)``, never touches pressure/memory, and
never enters the denoising schedule. All arrays are shape ``(L,)``.
"""
from __future__ import annotations

import hashlib

import numpy as np


def smooth_window_max(values: np.ndarray, *, half_width: int) -> np.ndarray:
    """Per residue, max over ``[i - half_width, i + half_width]`` (register grain,
    RAR 0020 9-residue window ⇒ ``half_width = 4``)."""
    v = np.asarray(values, dtype=float)
    L = int(v.shape[0])
    r = int(half_width)
    out = np.zeros(L, dtype=float)
    for i in range(L):
        lo, hi = max(0, i - r), min(L, i + r + 1)
        out[i] = float(v[lo:hi].max()) if hi > lo else 0.0
    return out


def allocation_mass(r_i: np.ndarray, *, half_width: int) -> np.ndarray:
    """Mean-normalized editability mass ``Φ_i`` from per-residue risk ``r_i``.

    Smooth to register grain, normalize over the protein to **mean 1** (high-risk
    regions ⇒ ``Φ_i > 1``). All-zero ``r_i`` ⇒ uniform ``Φ_i ≡ 1`` (degrade to the
    ``v_target`` baseline).
    """
    smoothed = smooth_window_max(np.asarray(r_i, dtype=float), half_width=half_width)
    L = int(smoothed.shape[0])
    total = float(smoothed.sum())
    if L == 0:
        return smoothed
    if total <= 0.0:
        return np.ones(L, dtype=float)
    return smoothed * (L / total)


def reweight_by_allocation(
    v_target: np.ndarray, allocation: np.ndarray, *, c: float, eps: float
) -> np.ndarray:
    """Selection field ``s_alloc_i = v_target_i * (eps + c * Φ_i)``.

    Uniform ``Φ_i ≡ 1`` ⇒ constant rescale of ``v_target`` (preserves baseline
    selection order, the H1 uniform control); tilted ``Φ_i`` reorders toward
    high prospective-immune regions. Reweights the SELECTION field only — never
    ``v_target`` itself (pressure stays on raw ``v_target``) and never memory.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    return v * (float(eps) + float(c) * phi)


def stable_seed(*parts: object) -> int:
    """Process-stable 64-bit seed from arbitrary parts via sha256.

    Reproducible across Della jobs — unlike the built-in ``hash()`` (salted by
    ``PYTHONHASHSEED``). Mirrors the ``_rng_for`` idiom (self_conditioned_gr.py).
    """
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


# ---------------------------------------------------------------------------
# §A1 — revised r_i consumption: triage / tie-break within the eligible set
# (PLAN_PLANNER_SC_GR.md §A1; doc/Self-Cond_GR.md §8.4 Path A)
# ---------------------------------------------------------------------------


def _norm_rank(x: np.ndarray) -> np.ndarray:
    """Ranks in (0, 1), stable ties. Empty ⇒ empty."""
    n = x.shape[0]
    if n == 0:
        return x.astype(float)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (order + 0.5) / n


def triage_field(
    v_target: np.ndarray,
    allocation: np.ndarray,
    *,
    eligible_quantile: float,
    triage_lambda: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Path-A triage (doc §8.4): within the v_target-eligible set, rank by
    ``norm_rank(v_target) + triage_lambda * norm_rank(Φ_i)``; ineligible ⇒ 0.

    ``v_target`` keeps coefficient 1 and dominates; ``Φ_i`` is a bounded
    (``triage_lambda``) tie-break/nudge.

    **Strict eligibility (doc §8.4 — load-bearing).** Eligibility is the
    top-(1−eligible_quantile) AMONG positions with **strictly-positive
    actionability** ``v_target > floor`` (``floor`` = the pipeline's
    ``active_window_min_excess``, default 0). The quantile is taken over the
    POSITIVE values only, so a sparse ``v_target`` whose median is 0 (or any
    non-positive site) can NEVER become eligible — this would otherwise re-admit
    exactly the low-actionability sites whose edits did not convert (RAR 0019 M4)
    and re-create the ``v_target·Φ`` failure (the multiplicative reweight's harm).
    All-``v_target``-non-positive ⇒ nothing eligible ⇒ all-zero field. ``Φ_i ≡ 1``
    (or ``triage_lambda`` 0) ⇒ pure ``v_target`` order within eligible.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    L = v.shape[0]
    if L == 0:
        return v
    out = np.zeros(L, dtype=float)
    positive = v > float(floor)
    if not positive.any():
        return out  # no genuinely-actionable site ⇒ select nothing (no τ_v widening)
    thr = float(np.quantile(v[positive], eligible_quantile))
    eligible = positive & (v >= thr)
    idx = np.where(eligible)[0]
    if idx.size:
        out[idx] = _norm_rank(v[idx]) + float(triage_lambda) * _norm_rank(phi[idx])
    return out
