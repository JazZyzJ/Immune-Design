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
