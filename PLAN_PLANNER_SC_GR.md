# SC-GR Position-Dependent Allocation (v1) Implementation Plan

> **STATUS — direct-allocation v1 (Tasks 1–8) SUPERSEDED (2026-06-28).** The direct-allocation H1 (`r_i → Φ → v_target` reweight) was **NOT supported** (RAR 0019 M4; mechanism + authority ceiling in `doc/Self-Cond_GR.md` §8.2). Tasks 1–8 are retained as the build/run record — **do not execute as-is**. The **live plan is §A1 + §A2 appended at the end of this file** (revised `r_i` consumption + terminal-probe ablation, doc §8.4 Path A). They **reuse** the `Φ_i` field (Tasks 1–4), telemetry (Task 6), and the 3-arm harness/eval (Tasks 7–8). Path C is still under discussion (doc §8.4), not yet planned.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an inference-only **allocation layer** that reweights D2 position-*selection* by the SC-GR per-residue prospective-immune field `r_i`, and run the pre-registered H1 test that immune-driven allocation beats uniform allocation at matched scTM.

**Architecture:** Budget → Allocation → Value stack (`doc/Self-Cond_GR.md` §8.3). The **Budget** layer (`B_sc → g`, incl. amplify `g_max>1`) and the `pressure_source` runtime seam already exist (SC1/SC2 code, LOG L0118/L0119) and are reused unchanged. This plan builds **only the Allocation layer**: freeze a per-residue editability mass `Φ_i` (parallel to the frozen `B_sc`), and reweight the typed selection field `v_target` by it at the two selection seams. The **Value** layer (`P_cheap` token correction) is explicitly deferred to v1.1.

**Tech Stack:** Python 3.12, NumPy, pytest. Reference-flow controller (`inverse_folding/reference_flow/`), DPLM masked-diffusion generator, NetMHCIIpan + ESMFold evaluation, Della SLURM.

> **NAMING — read first.** The new editability mass is **`Φ_i`** (Greek phi), held in a **new** module `inverse_folding/reference_flow/allocation.py`. It is **distinct from `A_i(t)`**, the established **typed-actionability field** (`actionability.py`, `controller.py:251`, `controller_config.py:144`, `PLAN_RF_UNI_CTRL.md`, `RF_Controller_Architecture.md`). Do **not** name anything `A_i` / `A_i(t)` / reuse the actionability module for `Φ_i`. `Φ_i` reweights the actionability field's `v_target`; it never *is* `A_i(t)`.

> **CONFIG SCHEMA — read first.** All controller config lives under the top-level `controller:` block. `materialize_controller_config(payload)` reads `payload["controller"]` then `controller_payload.get("targeting" / "self_conditioned_gr" / "global_pressure" / "allocation")` (controller_config.py:333-454). Every config example and test payload below nests under `controller:`. **There is no top-level `self_conditioned_gr:` / `allocation:` / `targeting:`.**

> **TESTS — read first.** Reference-flow tests live in **`tests/inverse_folding/`** (e.g. `tests/inverse_folding/test_reference_flow_d1_controller.py`); analysis-script tests in `tests/scripts/`. Controller state is read in tests via the existing accessors `controller.actionability_states()` (controller.py:506) and `controller.refresh_records()` (controller.py:483) — there is no `tiny_controller_factory`/`last_active_windows` API; reuse the controller-construction fixture already in `test_reference_flow_d1_controller.py`.

---

## 0. Context & cross-references

- Scientific spec: `doc/Self-Cond_GR.md` §8 (esp. §8.3 + the H1 spine). Read it first.
- Prior stage plan: `PLAN_RF_SC_GR.md` (SC0/SC1/SC2). The **standalone SC2 amplify experiment is superseded by this plan** — amplify (`g_max>1`) is folded in here as the Budget layer rather than swept alone (§6.2 of that file is annotated).
- Validation the field rests on: RAR 0020 — the per-residue `r_i` ranks the NoD per-residue oracle at region ρ 0.578 / Recall@High 0.645, **median-over-K, fresh arm, 9-residue region grain** (frozen in §2). RAR 0010 (`B_sc` ρ0.74), RAR 0011 (SC1 actuation).
- Frozen architecture decisions (do not violate): NetMHCIIpan is the **independent external validator** — never a guidance signal (it is the gate metric here precisely because it is not in the loop). WT baselines must be real data, fail-fast, no placeholders. One config per experiment; never mutate existing config defaults. Print every new hyperparameter to stdout. Cluster paths via CLI only. Check `doc/SCRIPTS.md` before adding scripts.

## 1. Scope

**In scope (v1 = Allocation, WHERE-only):**
1. A frozen per-residue editability mass `Φ_i` derived from the SC-GR probe's `residue_excess`, computed and frozen parallel to the existing frozen `B_sc` (firewall-clean, no new head calls).
2. A selection-field reweight `s_alloc = v_target · (eps + c·Φ_i)` applied **only at the two selection seams** (`controller.py:644` active-window rank, `controller.py:733` within-block residue rank).
3. Three selection modes for the experiment: `flat` / `v_target` / `v_target_x_alloc`.
4. Allocation telemetry (`phi_alloc`, `selection_field`, mode, frozen flag) for failure diagnosis.
5. The pre-registered 3-arm H1 experiment on pilot50 DRB1\*07:01 + its gate.

**Out of scope (deferred — do NOT implement here):**
- **Value layer `P_cheap`** (`ΔR_B^eff = ΔR_B + λ_P·q_B·P_cheap`): a separate v1.1 increment; bundling it would confound H1's attribution. Needs new inline `_score_cheap_probe` + `λ_P` calibration + a units-frozen Ω(B)-LME aggregator.
- **"How long editable" / time-axis online update.** The remask cadence (`sampler.py:632`) and freeze (`commit.py:320`) are global; the only per-position lever is D3 `m_i`, which the SC-GR firewall forbids the probe from feeding (`PLAN_RF_SC_GR.md:29,771`). v1 uses the **existing schedule + early-freeze** only; no new time knob.
- Ensemble instability `σ_i`; content-dependent noise schedules (hyperschedules, training-bound).
- Any change to D2 candidate scoring / resampling, D3 memory, or `A_i(t)` / `v_target` construction. Allocation touches the *selection field only*, downstream of memory.

## 2. Frozen definitions (no TBD — fixed before coding)

All arrays are shape `(L,)`. The field reuses the existing `residue_excess = excess_over_tau(max_covering_window_projection(windows), tau_ref_B)` (`self_conditioned_gr.py:293-294`), full-sequence `b_cur` scope. `Φ_i` is **distinct from** the typed-actionability field `A_i(t)`; it only *reweights* `A_i(t)`'s output `v_target`.

**D1 — per-residue prospective risk `r_i`:** median over the `K` completions of `residue_excess`, **fresh arm** (SC1 pin, `actuation_arm`), at the frozen horizon (first reliable refresh, `freeze_after_reliable_refreshes=1`). Exactly the form validated in RAR 0020.

**D2 — editability mass `Φ_i` (mean-normalized):**
- Smooth to register grain: `Φ̃_i = max_{|j-i| ≤ R} r_j`, `R = 4` (9-residue window = RAR 0020 `region_size=9`).
- Normalize over the protein to **mean 1**: `Φ_i = Φ̃_i · L / Σ_j Φ̃_j`.
- Degenerate guard: if `Σ Φ̃ = 0`, `Φ_i ≡ 1` (controller degrades to the `v_target` baseline, never divides by zero).

**D3 — selection reweight:** `s_alloc_i = v_target_i · (eps + c·Φ_i)`, `eps = 0.05`, `c = 1.0` (v1 defaults, config-exposed + printed). Uniform `Φ_i ≡ 1` ⇒ constant rescale of `v_target` ⇒ **identical selection order to the `v_target` arm** (so the `v_target` arm is the exact uniform-allocation control). Applied to the selection field only; `u_pressure`/`G`/β stay on raw `v_target`; memory untouched.

**D4 — experiment arms (`controller.targeting.selection_field_mode`):**
| mode | selection field | role |
|---|---|---|
| `flat` | per-protein/design fixed pseudo-random vector (sha256-seeded — process-stable); D2's structural safe-candidate support still applies at the token level | C2 — tests "does targeting help at all" |
| `v_target` | raw `v_target` (current behavior) | C1 — uniform-allocation control (= `Φ_i≡1`) |
| `v_target_x_alloc` | `reweight_by_allocation(v_target, Φ, c, eps)` | Treatment — immune-tilted allocation |

**D5 — Budget (reused, unchanged):** `pressure_source="self_conditioned_probe"` + amplify `g_min=0.5`, `g_max ∈ {1.5, 2.0, 2.5}` swept (SC2 calibration). All three arms run at each `g_max`.

**D6 — behavior cadence & remask (held constant across all arms):**
- **`N_STEPS=100`** — a **run/CLI/SLURM** parameter (NOT in the YAML; the YAML only has `*_ttl_steps`). The SC monitor's `N_STEPS=20` was diagnostic-only ("10-step exposes only one post-start refresh"). At `t_start=0.50` / `refresh_interval=5`, 100 steps yields **~10 post-`t_start` actuation refreshes** (vs ~2–3 at 20). This matters because the global **Budget** is frozen and refresh-count-insensitive, but **Allocation is a per-refresh selection tilt** that needs many refreshes to accumulate — budget-only (SC1) works at 20 steps; allocation does not.
- **DPLM remask (reparam) stays ON** at the base-config setting, **identical across the `flat`/`v_target`/`alloc` arms**. It is the mechanism that keeps positions editable into the reliable mid-late trajectory where the tilt is most effective; turning it off would starve late-trajectory allocation. Held constant ⇒ not a confound.
- Remask stays **D3-`m_i`-driven**; **`Φ_i` is never fed to remask/D3** (firewall; immune-aware remask is the deferred v2 "how long editable" lever).
- Because `N_STEPS` changes the freeze-point trajectory, the amplify calibration bands (`B_low`/`B_high`/`B_median`) must be **re-emitted at the 100-step cadence**, not inherited from the 20-step monitor (the freeze still anchors at `t_start≈0.5`, so it likely transfers — but verify, don't assume).

## 3. File structure

| File | Responsibility | Change |
|---|---|---|
| `inverse_folding/reference_flow/self_conditioned_gr.py` | per-residue K reduction `r_i` | **Add** `reduce_residue_excess_over_k` |
| `inverse_folding/reference_flow/allocation.py` | **new module** — `Φ_i` math + stable seed (distinct from `A_i(t)`) | **Create** `smooth_window_max`, `allocation_mass`, `reweight_by_allocation`, `stable_seed` |
| `inverse_folding/reference_flow/controller_config.py` | config | **Add** `AllocationConfig` (under `controller.allocation`) + `selection_field_mode` on `TargetingConfig` |
| `inverse_folding/reference_flow/controller.py` | runtime wiring | **Modify**: compute+freeze `Φ_i` (parallel to `B_sc`); build mode-selected selection field at `:644`/`:733`; stash telemetry |
| `scripts/run_if_phase_c1.py` | actionability telemetry writer | **Modify**: add `phi_alloc`/`selection_field` residue columns + `selection_field_mode`/`allocation_frozen_flag` summary fields |
| `inverse_folding/reference_flow/configs/` | experiment configs | **Add** 9 YAMLs (3 modes × 3 `g_max`), based on `scgr_betaonly` |
| `scripts/analysis/planner_h1_pareto.py` | H1 Pareto + gate | **Add** (register in `doc/SCRIPTS.md`) |
| `tests/inverse_folding/…`, `tests/scripts/…` | unit + integration | **Add** per task |

`actionability.py` is **not modified** — `Φ_i` math gets its own module.

---

## 4. Tasks

### Task 1: Per-residue K reduction `r_i`

**Files:** Modify `inverse_folding/reference_flow/self_conditioned_gr.py` (after `summarize_probe_refresh`, ~line 394). Test `tests/inverse_folding/test_reference_flow_self_conditioned_gr.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_reduce_residue_excess_over_k_medians_fresh_arm():
    from inverse_folding.reference_flow.self_conditioned_gr import (
        reduce_residue_excess_over_k, SCGRProbeSample, SCGRRiskAggregates)
    import numpy as np
    def mk(arm, idx, excess):
        s = SCGRProbeSample(arm=arm, sample_idx=idx, tokens=np.zeros(1),
                            num_masked=0, num_reused_from_prev=0, reuse_fraction=0.0,
                            mean_prev_confidence_reused=0.0, mean_sample_entropy=0.0,
                            state_bootstrap_flag=False)
        a = SCGRRiskAggregates(G_mean_excess=0.0, G_topm_lse=0.0, supra_masses={},
                               head_risk_LME=0.0, head_risk_max=0.0,
                               residue_excess=np.asarray(excess, dtype=float))
        return (s, a)
    per_sample = [mk("fresh", 0, [0.0, 2.0, 4.0]), mk("fresh", 1, [0.0, 6.0, 0.0]),
                  mk("fresh", 2, [0.0, 4.0, 2.0]), mk("self_conditioned", 0, [9.0, 9.0, 9.0])]
    r_i = reduce_residue_excess_over_k(per_sample, arm="fresh")
    np.testing.assert_allclose(r_i, [0.0, 4.0, 2.0])

def test_reduce_residue_excess_over_k_empty_arm_returns_empty():
    from inverse_folding.reference_flow.self_conditioned_gr import reduce_residue_excess_over_k
    assert reduce_residue_excess_over_k([], arm="fresh").size == 0
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/inverse_folding/test_reference_flow_self_conditioned_gr.py -k reduce_residue_excess -v` → `ImportError`.
- [ ] **Step 3: Write minimal implementation**

```python
def reduce_residue_excess_over_k(
    per_sample: "Sequence[tuple[SCGRProbeSample, SCGRRiskAggregates]]", *, arm: str,
) -> np.ndarray:
    """Median over the K completions of one arm's per-residue ``residue_excess``.

    Returns the per-residue prospective-risk map ``r_i`` (RAR 0020 form:
    median-over-K, ``fresh`` arm). Empty completions skipped; no match ⇒ empty.
    """
    arrays = [a.residue_excess for s, a in per_sample
              if s.arm == arm and a.residue_excess.size]
    if not arrays:
        return np.empty(0, dtype=float)
    return np.median(np.stack(arrays, axis=0), axis=0)
```

- [ ] **Step 4: Run to verify it passes** — same `-k` → PASS (2).
- [ ] **Step 5: Commit** — `git commit -m "feat(scgr): per-residue r_i = median-over-K residue_excess (fresh arm)"`

### Task 2: Allocation module (`Φ_i` math + stable seed)

**Files:** Create `inverse_folding/reference_flow/allocation.py`. Test `tests/inverse_folding/test_reference_flow_allocation.py`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from inverse_folding.reference_flow.allocation import (
    smooth_window_max, allocation_mass, reweight_by_allocation, stable_seed)

def test_smooth_window_max_register_grain():
    v = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(smooth_window_max(v, half_width=1), [0.0, 5.0, 5.0, 5.0, 0.0])

def test_allocation_mass_is_mean_one():
    phi = allocation_mass(np.array([0.0, 0.0, 3.0, 1.0]), half_width=0)
    assert abs(float(phi.mean()) - 1.0) < 1e-9
    assert phi[2] > phi[0]

def test_allocation_mass_all_zero_is_uniform():
    np.testing.assert_allclose(allocation_mass(np.zeros(6), half_width=4), np.ones(6))

def test_reweight_uniform_alloc_preserves_vtarget_order():
    v = np.array([0.2, 0.9, 0.5, 0.1])
    s = reweight_by_allocation(v, np.ones(4), c=1.0, eps=0.05)
    assert list(np.argsort(s)) == list(np.argsort(v))

def test_reweight_tilts_toward_high_alloc():
    s = reweight_by_allocation(np.array([1.0, 1.0]), np.array([0.2, 1.8]), c=1.0, eps=0.05)
    assert s[1] > s[0]

def test_stable_seed_is_process_stable():
    # value is fixed (NOT salted like builtin hash); recompute must match this literal
    assert stable_seed("flat", "P123", 0) == stable_seed("flat", "P123", 0)
    assert stable_seed("flat", "P123", 0) != stable_seed("flat", "P123", 1)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/inverse_folding/test_reference_flow_allocation.py -v` → `ModuleNotFoundError`.
- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run to verify they pass** — `pytest tests/inverse_folding/test_reference_flow_allocation.py -v` → PASS (6).
- [ ] **Step 5: Commit** — `git commit -m "feat(allocation): Phi_i mass operators + stable seed (new module, distinct from A_i(t))"`

### Task 3: Allocation config + selection_field_mode (under `controller.*`)

**Files:** Modify `inverse_folding/reference_flow/controller_config.py` — add `AllocationConfig`; add `selection_field_mode` to `TargetingConfig` (built by `_materialize_targeting`, :759); wire `_materialize_allocation(controller_payload.get("allocation"))` next to the other sub-configs (:449-454). Test `tests/inverse_folding/test_reference_flow_controller_config.py`.

- [ ] **Step 1: Write the failing test** (note: all blocks nested under `controller`)

```python
def test_allocation_config_defaults_and_materialize():
    from inverse_folding.reference_flow.controller_config import materialize_controller_config
    cfg = materialize_controller_config({"controller": {
        "enabled": True,
        "self_conditioned_gr": {"enabled": True, "write_residue_telemetry": True},
        "allocation": {"enabled": True, "reweight_c": 0.5, "smooth_half_width": 4},
        "targeting": {"selection_field_mode": "v_target_x_alloc"},
    }})
    assert cfg.allocation.enabled is True
    assert cfg.allocation.reweight_c == 0.5
    assert cfg.allocation.smooth_half_width == 4
    assert cfg.allocation.reweight_eps == 0.05      # default
    assert cfg.allocation.arm == "fresh"            # default
    assert cfg.targeting.selection_field_mode == "v_target_x_alloc"

def test_selection_field_mode_rejects_unknown():
    import pytest
    from inverse_folding.reference_flow.controller_config import materialize_controller_config
    with pytest.raises(ValueError, match="selection_field_mode"):
        materialize_controller_config({"controller": {
            "enabled": True, "targeting": {"selection_field_mode": "bogus"}}})

def test_selection_field_mode_default_is_v_target():
    from inverse_folding.reference_flow.controller_config import materialize_controller_config
    cfg = materialize_controller_config({"controller": {"enabled": True}})
    assert cfg.targeting.selection_field_mode == "v_target"
    assert cfg.allocation.enabled is False
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/inverse_folding/test_reference_flow_controller_config.py -k "allocation_config or selection_field_mode" -v` → FAIL.
- [ ] **Step 3: Write minimal implementation**

Add the dataclass:
```python
@dataclass(frozen=True)
class AllocationConfig:
    """Allocation layer (PLAN_PLANNER_SC_GR.md §2). Reweights the selection
    field v_target by a frozen per-residue editability mass Phi_i. WHERE-only;
    never touches v_target construction, pressure, memory, or the schedule.
    Phi_i is distinct from the typed-actionability field A_i(t)."""
    enabled: bool = False
    arm: str = "fresh"
    smooth_half_width: int = 4
    reweight_c: float = 1.0
    reweight_eps: float = 0.05
```
Add `allocation: AllocationConfig = field(default_factory=AllocationConfig)` to `ControllerConfig`. Add `selection_field_mode: str = "v_target"` to `TargetingConfig`. In `_materialize_targeting` (:759), parse + validate:
```python
_SELECTION_FIELD_MODES = ("v_target", "v_target_x_alloc", "flat")
mode = str((payload or {}).get("selection_field_mode", "v_target"))
if mode not in _SELECTION_FIELD_MODES:
    raise ValueError(f"selection_field_mode must be one of {_SELECTION_FIELD_MODES}, got {mode!r}")
# include selection_field_mode=mode in the returned TargetingConfig
```
Add the allocation materializer + wire it (next to `targeting = _materialize_targeting(...)` at :449):
```python
_ALLOC_ARMS = ("fresh", "self_conditioned")
def _materialize_allocation(payload: Any) -> AllocationConfig:
    p = payload if isinstance(payload, dict) else {}
    d = AllocationConfig()
    arm = str(p.get("arm", d.arm))
    if arm not in _ALLOC_ARMS:
        raise ValueError(f"allocation.arm must be one of {_ALLOC_ARMS}, got {arm!r}")
    return AllocationConfig(
        enabled=bool(p.get("enabled", d.enabled)), arm=arm,
        smooth_half_width=int(p.get("smooth_half_width", d.smooth_half_width)),
        reweight_c=float(p.get("reweight_c", d.reweight_c)),
        reweight_eps=float(p.get("reweight_eps", d.reweight_eps)))
# in materialize_controller_config, next to targeting/global_pressure:
allocation = _materialize_allocation(controller_payload.get("allocation"))
# ...pass allocation=allocation into the ControllerConfig(...) construction
```
**Print** at config load (print-hyperparams rule): add to the controller stdout summary
`f"[allocation] enabled={cfg.allocation.enabled} mode={cfg.targeting.selection_field_mode} c={cfg.allocation.reweight_c} eps={cfg.allocation.reweight_eps} half_width={cfg.allocation.smooth_half_width} arm={cfg.allocation.arm}"`.

- [ ] **Step 4: Run to verify it passes** — same `-k` → PASS (3).
- [ ] **Step 5: Commit** — `git commit -m "feat(config): AllocationConfig + TargetingConfig.selection_field_mode under controller.*"`

### Task 4: Compute & freeze `Φ_i` in the controller (parallel to `B_sc`)

**Files:** Modify `inverse_folding/reference_flow/controller.py` — instance init (~:466-479); `_run_sc_gr_probe` store (~:1914); `_scgr_frozen_pressure` freeze block (after the `B_sc` freeze at :1707-1717). Test `tests/inverse_folding/test_reference_flow_d1_controller.py`.

**Seam spec (concrete — note `self.config`, `scfg.freeze_after_reliable_refreshes`, `reliable` gate):**
1. Instance init (next to `self._scgr_frozen_B_sc`):
   `self._scgr_actuation_residue_excess: "np.ndarray | None" = None`;
   `self._scgr_residue_window: list = []`;
   `self._scgr_frozen_allocation: "np.ndarray | None" = None`  (holds `Φ_i`).
2. In `_run_sc_gr_probe`, after `self._scgr_actuation_B_sc` is set (~:1914):
   ```python
   if self.config.allocation.enabled:
       r_i = reduce_residue_excess_over_k(per_sample, arm=self.config.allocation.arm)
       self._scgr_actuation_residue_excess = r_i if r_i.size else None
   ```
   (Import `reduce_residue_excess_over_k` from `self_conditioned_gr`.) Runs unconditionally of telemetry.
3. In `_scgr_frozen_pressure(self, reliable)` — **after** the existing `B_sc` freeze block (:1714-1717), where `scfg = self.config.self_conditioned_gr` is already in scope:
   ```python
   if (self.config.allocation.enabled and reliable
           and self._scgr_frozen_allocation is None
           and self._scgr_actuation_residue_excess is not None):
       self._scgr_residue_window.append(self._scgr_actuation_residue_excess)
       if len(self._scgr_residue_window) >= int(scfg.freeze_after_reliable_refreshes):
           frozen_r = np.median(np.stack(self._scgr_residue_window, axis=0), axis=0)
           self._scgr_frozen_allocation = allocation_mass(
               frozen_r, half_width=self.config.allocation.smooth_half_width)
   ```
   (Import `allocation_mass` from `allocation`.) `reliable`-gated + freeze-once (`is None`) ⇒ cannot self-reinforce. No new head calls; firewall preserved.

- [ ] **Step 1: Write the failing integration test** (reuse the existing controller fixture in this file that runs a guided trajectory with the SC-GR probe; read state via the real accessors)

```python
def test_frozen_allocation_persisted_when_enabled(build_guided_controller):
    # build_guided_controller: the existing fixture/helper in this module that
    # constructs a ReferenceFlowController with the SC-GR probe enabled and runs
    # it past the first reliable refresh. Pass the extra config blocks through.
    ctrl = build_guided_controller(controller_overrides={
        "self_conditioned_gr": {"enabled": True, "write_residue_telemetry": True},
        "allocation": {"enabled": True},
        "global_pressure": {"pressure_source": "self_conditioned_probe",
                            "freeze_after_reliable_refreshes": 1},
    })
    phi = ctrl._scgr_frozen_allocation
    assert phi is not None
    assert phi.shape[0] == len(ctrl.refresh_records()[-1].__dict__.get("sequence", phi))  # L
    assert abs(float(phi.mean()) - 1.0) < 1e-6
    assert (phi >= 0).all()

def test_frozen_allocation_absent_when_disabled(build_guided_controller):
    ctrl = build_guided_controller(controller_overrides={
        "self_conditioned_gr": {"enabled": True},
        "allocation": {"enabled": False},
        "global_pressure": {"pressure_source": "self_conditioned_probe"},
    })
    assert ctrl._scgr_frozen_allocation is None
```

> If the module has no `build_guided_controller` helper, factor the controller-construction + run loop out of the existing `write_residue_telemetry` test into a local helper and call it here — do **not** invent a new public API. The length `L` assertion: use whatever length handle the existing tests use (probe length / masked array length).

- [ ] **Step 2: Run to verify it fails** — `pytest tests/inverse_folding/test_reference_flow_d1_controller.py -k frozen_allocation -v` → FAIL.
- [ ] **Step 3: Implement the three seam edits.**
- [ ] **Step 4: Run to verify it passes + no regressions** — `-k frozen_allocation` → PASS (2); then `pytest tests/inverse_folding/test_reference_flow_d1_controller.py -q` → no regressions.
- [ ] **Step 5: Commit** — `git commit -m "feat(controller): freeze per-design Phi_i parallel to B_sc (reliable-gated, firewall-clean)"`

### Task 5: Apply the selection field at the two seams

**Files:** Modify `inverse_folding/reference_flow/controller.py` — add `_selection_field(...)` helper; use at `:644` and `:733`. Test `tests/inverse_folding/test_reference_flow_d1_controller.py`.

**Seam spec (concrete):** add a helper computed where `actionability.v_target` is available (just before `:644`):
```python
def _selection_field(self, v_target, protein_id, design_idx):
    mode = self.config.targeting.selection_field_mode
    if mode == "flat":
        rng = np.random.default_rng(stable_seed("flat", protein_id, design_idx))
        return rng.random(v_target.shape[0])   # content-blind, process-stable
    if mode == "v_target_x_alloc" and self._scgr_frozen_allocation is not None:
        return reweight_by_allocation(
            v_target, self._scgr_frozen_allocation,
            c=self.config.allocation.reweight_c, eps=self.config.allocation.reweight_eps)
    return v_target  # "v_target", or alloc-enabled-but-not-yet-frozen ⇒ baseline
```
(Import `reweight_by_allocation`, `stable_seed` from `allocation`.) Then: at `:644` compute `sel = self._selection_field(actionability.v_target, protein_id, design_idx)` and use `sel` as the active-window field; at `:733` pass `typed_field=sel`. **Do not** touch `u_pressure`/`G`/β (raw `v_target` at `:2064-2079`). Stash for telemetry (Task 6): `self._last_selection_field = sel`.

- [ ] **Step 1: Write the failing integration tests** (assert on `controller.refresh_records()[-1]` active-window content + `actionability_states()`; reuse the guided fixture; inject `_scgr_frozen_allocation` directly to isolate the seam)

```python
def test_uniform_phi_equals_v_target_order(build_guided_controller):
    import numpy as np
    base = build_guided_controller(controller_overrides={"targeting": {"selection_field_mode": "v_target"}})
    alloc = build_guided_controller(controller_overrides={
        "targeting": {"selection_field_mode": "v_target_x_alloc"},
        "allocation": {"enabled": True}}, frozen_allocation=lambda L: np.ones(L))
    # uniform Phi -> identical active-window selection to the v_target arm
    assert (alloc.refresh_records()[-1].num_active_windows
            == base.refresh_records()[-1].num_active_windows)
    assert _active_window_spans(alloc) == _active_window_spans(base)

def test_alloc_arm_reorders_toward_high_phi(build_guided_controller):
    import numpy as np
    ctrl = build_guided_controller(controller_overrides={
        "targeting": {"selection_field_mode": "v_target_x_alloc"},
        "allocation": {"enabled": True, "reweight_c": 5.0}},
        frozen_allocation=lambda L, low=_low_vtarget_region: _mass_peaked_at(L, low))
    assert _region_selected(ctrl, _low_vtarget_region)        # tilt took effect

def test_flat_arm_is_content_blind_and_reproducible(build_guided_controller):
    a = build_guided_controller(controller_overrides={"targeting": {"selection_field_mode": "flat"}})
    b = build_guided_controller(controller_overrides={"targeting": {"selection_field_mode": "flat"}})
    assert _active_window_spans(a) == _active_window_spans(b)  # sha256-seeded, stable
```

> `_active_window_spans(ctrl)`, `_region_selected`, `_mass_peaked_at` are tiny local helpers reading `ctrl.refresh_records()` / `ctrl.actionability_states()` — add them in the test file (not production). `build_guided_controller(..., frozen_allocation=fn)` sets `ctrl._scgr_frozen_allocation = fn(L)` before the steering refresh so the seam is tested independently of Task 4's freeze timing. If the existing fixture can't inject mid-run, set the attribute and call one more `run`-refresh.

- [ ] **Step 2: Run to verify they fail** — `pytest tests/inverse_folding/test_reference_flow_d1_controller.py -k "phi or flat or selection" -v` → FAIL.
- [ ] **Step 3: Implement the `_selection_field` helper + the two seam swaps.**
- [ ] **Step 4: Run to verify they pass + full suite** — `-k "phi or flat or selection"` → PASS (3); then `pytest tests/inverse_folding -q` → green.
- [ ] **Step 5: Commit** — `git commit -m "feat(controller): mode-selected selection field (flat/v_target/x_alloc)"`

### Task 6: Allocation telemetry (failure diagnosis)

**Files:** Modify `inverse_folding/reference_flow/controller.py` (attach per-refresh telemetry) + `scripts/run_if_phase_c1.py` (write columns). Test `tests/inverse_folding/test_reference_flow_d1_controller.py`.

**Why:** if H1 fails we must distinguish "`Φ_i` never formed" vs "selection cap too small to express the tilt" vs "high-`Φ_i` region structurally locked". The current `_ACTIONABILITY_RESIDUE_COLUMNS` (run_if_phase_c1.py:832) has no allocation signal.

**Seam spec:**
1. Controller: when building the selection field (Task 5), record per-refresh arrays the writer can read alongside `state.v_target`: `phi_alloc = self._scgr_frozen_allocation if (mode=="v_target_x_alloc" and self._scgr_frozen_allocation is not None) else np.ones(L)`; `selection_field = sel`. Attach both to the `UnifiedActionabilityState` for that refresh (add two `(L,)` fields with default `None`/empty) so they flow to the writer next to `v_target`/`u_pressure`.
2. Writer `scripts/run_if_phase_c1.py`: add `"phi_alloc"` and `"selection_field"` to `_ACTIONABILITY_RESIDUE_COLUMNS` (:832) and to the row dict (:891-892): `"phi_alloc": float(state.phi_alloc[i])`, `"selection_field": float(state.selection_field[i])`. In the refresh summary writer (:947 `actionability_refresh_summary.jsonl`) add `selection_field_mode` (from config) and `allocation_frozen_flag` (`controller._scgr_frozen_allocation is not None`).

- [ ] **Step 1: Write the failing test**

```python
def test_actionability_telemetry_has_phi_and_selection(build_guided_controller):
    ctrl = build_guided_controller(controller_overrides={
        "self_conditioned_gr": {"enabled": True, "write_residue_telemetry": True},
        "allocation": {"enabled": True},
        "targeting": {"selection_field_mode": "v_target_x_alloc"},
        "global_pressure": {"pressure_source": "self_conditioned_probe"}})
    st = ctrl.actionability_states()[-1]
    assert hasattr(st, "phi_alloc") and st.phi_alloc.shape == st.v_target.shape
    assert hasattr(st, "selection_field") and st.selection_field.shape == st.v_target.shape
```

(A second test asserting the written parquet has the two columns may be added against the existing actionability-writer test, if one exists.)

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the state fields + writer columns + summary fields.
- [ ] **Step 4: Run to verify it passes** — `-k telemetry` → PASS; `pytest tests/inverse_folding -q` green.
- [ ] **Step 5: Commit** — `git commit -m "feat(telemetry): persist phi_alloc/selection_field + mode/frozen flag"`

### Task 7: Experiment configs (one per arm × g_max) + calibration

**Files:** Create 9 YAMLs + 3 amplify calibration JSONs. Test `tests/inverse_folding/test_reference_flow_controller_config.py`.

**Base config = `inverse_folding/reference_flow/configs/d2_d3_full_stageC_scgr_betaonly_aopen.yaml`** (the only config wiring `controller.global_pressure.pressure_source="self_conditioned_probe"` + `controller.self_conditioned_gr.enabled=true`). **Do not** use `d2_d3_full_stageC_pressure_aopen.yaml` (that is the old `trajectory_thresholded_G`). **Copy the full base file** per new config (YAML is not merged — the whole file loads), then set, all under `controller:`:
```yaml
controller:
  self_conditioned_gr:
    write_residue_telemetry: true
  allocation:
    enabled: true            # flat/v_target arms ignore the field; mode is authoritative
    arm: fresh
    smooth_half_width: 4
    reweight_c: 1.0
    reweight_eps: 0.05
  targeting:
    selection_field_mode: <MODE>
  global_pressure:
    pressure_source: self_conditioned_probe
    amplify: true
    g_min: 0.5            # pinned (SC2 learning); do NOT rely on default 0.0
    g_max: <GMAX>
```

**Calibration contract (P1-1):** `_load_global_pressure_calibration` (run_if_phase_c1.py:181-220) **requires**, when `pressure_source=self_conditioned_probe` AND `amplify=true`, a calibration JSON carrying `B_median`, `g_min`, `g_max` that **match the YAML** (`pressure_source` and `B_high>B_low` validated). So per `g_max` emit **one** amplify calibration JSON (shared by that g_max's 3 arm configs) using the SC2 emitter (LOG L0119; `sc_gr_monitor_probe.py --emit-calibration --amplify --g-min 0.5 --g-max <GMAX>` — confirm exact flags from L0119), built on the **`N_STEPS=100`** `refresh_step=0 / fresh / topm_lse` `B_sc` distribution (re-emit at the behavior cadence per §2 D6 — do **not** inherit the 20-step monitor bands). Pass it at run via `--global-pressure-calibration-json amplify_calib_gmax<GMAX>.json`. All runs use **`N_STEPS=100`** (set in the run command / SLURM, not the YAML); remask stays at the base-config setting, identical across all 9 runs.

| file | MODE | GMAX | calibration JSON |
|---|---|---|---|
| `planner_flat_gmax15_aopen.yaml` | flat | 1.5 | `amplify_calib_gmax15.json` |
| `planner_vtarget_gmax15_aopen.yaml` | v_target | 1.5 | `amplify_calib_gmax15.json` |
| `planner_alloc_gmax15_aopen.yaml` | v_target_x_alloc | 1.5 | `amplify_calib_gmax15.json` |
| `planner_flat_gmax20_aopen.yaml` | flat | 2.0 | `amplify_calib_gmax20.json` |
| `planner_vtarget_gmax20_aopen.yaml` | v_target | 2.0 | `amplify_calib_gmax20.json` |
| `planner_alloc_gmax20_aopen.yaml` | v_target_x_alloc | 2.0 | `amplify_calib_gmax20.json` |
| `planner_flat_gmax25_aopen.yaml` | flat | 2.5 | `amplify_calib_gmax25.json` |
| `planner_vtarget_gmax25_aopen.yaml` | v_target | 2.5 | `amplify_calib_gmax25.json` |
| `planner_alloc_gmax25_aopen.yaml` | v_target_x_alloc | 2.5 | `amplify_calib_gmax25.json` |

- [ ] **Step 1: Write the failing smoke test**

```python
import glob, pytest
from inverse_folding.reference_flow.controller_config import load_controller_config_file

@pytest.mark.parametrize("path", sorted(glob.glob(
    "inverse_folding/reference_flow/configs/planner_*_aopen.yaml")))
def test_planner_configs_load(path):
    cfg = load_controller_config_file(path)   # existing loader
    assert cfg.controller.enabled
    assert cfg.global_pressure.pressure_source == "self_conditioned_probe"
    assert cfg.global_pressure.amplify is True
    assert cfg.global_pressure.g_min == 0.5
    assert cfg.targeting.selection_field_mode in ("flat", "v_target", "v_target_x_alloc")
```

- [ ] **Step 2: Run to verify it fails** (no files). Create the 9 YAMLs; emit the 3 calibration JSONs (cluster step — document the exact emitter command in a sibling `README` or the run SLURM, paths via CLI).
- [ ] **Step 3: Run to verify it passes** — `pytest tests/inverse_folding/test_reference_flow_controller_config.py -k planner_configs_load -v` → PASS (9).
- [ ] **Step 4: Commit** — `git commit -m "feat(configs): planner v1 grid (3 modes x 3 g_max) on scgr_betaonly + amplify calibration"`

### Task 8: H1 evaluation + Pareto gate (design→protein aggregation)

**Files:** Create `scripts/analysis/planner_h1_pareto.py`. Modify `doc/SCRIPTS.md`. Test `tests/scripts/test_planner_h1_pareto.py`.

**Reuse-first:** runs reuse `scripts/run_if_phase_c1.py` (9 configs + the 3 calibration JSONs); eval reuses the existing `evaluate_phase_c.py` immune(NetMHCIIpan)+scTM path (RAR 0011/0020) — confirm per-design `immune_nmp` + `sctm` columns; if a comparison script already exists in `doc/SCRIPTS.md`, extend it. **Stats: n=8 designs/protein ⇒ collapse to one value per (g_max, arm_mode, protein_id) by design-median FIRST, then paired Wilcoxon over the 50 proteins** (never `set_index("protein_id")` on the raw design rows — duplicate index).

- [ ] **Step 1: Write the failing test** (multiple designs/protein, to exercise the aggregation)

```python
import numpy as np, pandas as pd
from scripts.analysis.planner_h1_pareto import compute_h1

def test_compute_h1_aggregates_designs_then_pairs():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(50):
        base = rng.normal(0, 1)
        for d in range(8):                                  # 8 designs/protein
            jit = rng.normal(0, 0.3)
            for mode, imm in [("flat", base + 1.0 + jit), ("v_target", base + jit),
                              ("v_target_x_alloc", base - 0.6 + jit)]:
                rows.append(dict(protein_id=f"p{p}", design_idx=d, g_max=2.0,
                                 arm_mode=mode, immune_nmp=imm, sctm=0.95))
    out = compute_h1(pd.DataFrame(rows))
    cell = [c for c in out["by_g_max"] if c["g_max"] == 2.0][0]
    assert cell["n_proteins_paired"] == 50                 # paired over proteins, not 400 rows
    assert cell["alloc_minus_vtarget_median"] < 0
    assert cell["wilcoxon_alloc_vs_vtarget_p"] < 0.05
    assert cell["vtarget_minus_flat_median"] < 0
    assert cell["sctm_noninferior"] is True
    assert cell["h1_pass"] is True
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/scripts/test_planner_h1_pareto.py -v` → `ImportError`.
- [ ] **Step 3: Implement `compute_h1` + thin CLI**

```python
"""H1 gate for SC-GR position-dependent allocation (PLAN_PLANNER_SC_GR.md §5).
Aggregates n designs/protein to a protein-median, then pairs over proteins.
No cluster paths hardcoded (all via --eval-glob)."""
import argparse, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

SCTM_TOL = 0.01

def _protein_level(df, col):
    # collapse n designs/protein -> one row per (arm_mode, protein_id)
    return df.groupby(["arm_mode", "protein_id"], as_index=False)[col].median()

def _paired(df, a, b, col):
    m = _protein_level(df, col)
    pa = m[m.arm_mode == a].set_index("protein_id")[col]
    pb = m[m.arm_mode == b].set_index("protein_id")[col]
    idx = pa.index.intersection(pb.index)
    da, db = pa.loc[idx], pb.loc[idx]
    diff = da - db
    p = float(wilcoxon(da, db).pvalue) if (diff.abs() > 0).any() else 1.0
    return float(diff.median()), p, int(len(idx))

def compute_h1(df: pd.DataFrame) -> dict:
    cells = []
    for g, sub in df.groupby("g_max"):
        amv_med, amv_p, n_pair = _paired(sub, "v_target_x_alloc", "v_target", "immune_nmp")
        vmf_med, _, _ = _paired(sub, "v_target", "flat", "immune_nmp")
        sctm = _protein_level(sub, "sctm").groupby("arm_mode")["sctm"].median()
        noninf = bool(sctm.get("v_target_x_alloc", np.nan)
                      >= sctm.get("v_target", np.nan) - SCTM_TOL)
        cells.append(dict(
            g_max=float(g), n_proteins_paired=n_pair,
            alloc_minus_vtarget_median=amv_med, wilcoxon_alloc_vs_vtarget_p=amv_p,
            vtarget_minus_flat_median=vmf_med,
            sctm_median_by_arm={k: float(v) for k, v in sctm.items()},
            sctm_noninferior=noninf,
            h1_pass=bool(amv_med < 0 and amv_p < 0.05 and noninf)))
    return {"by_g_max": cells, "n_proteins": int(df.protein_id.nunique())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pd.concat([pd.read_parquet(p) for p in glob.glob(args.eval_glob)], ignore_index=True)
    out = compute_h1(df)
    print(f"[planner-h1] n_proteins={out['n_proteins']}")
    for c in out["by_g_max"]:
        print(f"[planner-h1] g_max={c['g_max']} n={c['n_proteins_paired']} "
              f"alloc-vtarget dmed={c['alloc_minus_vtarget_median']:.3f} "
              f"p={c['wilcoxon_alloc_vs_vtarget_p']:.3f} sctm_noninf={c['sctm_noninferior']} "
              f"PASS={c['h1_pass']}")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(args.output.replace(".json", ".per_design.csv"), index=False)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes + register** — `pytest tests/scripts/test_planner_h1_pareto.py -v` → PASS; add to `doc/SCRIPTS.md`: `scripts/analysis/planner_h1_pareto.py — H1 gate for SC-GR position-dependent allocation (design→protein median, paired Wilcoxon + Pareto)`.
- [ ] **Step 5: Commit** — `git commit -m "feat(analysis): planner H1 Pareto gate (design->protein aggregation)"`

---

## 5. Pre-registered H1 gate (fixed before results)

Cohort: 50 pilot50r2 DRB1\*07:01 proteins, n=8 designs/protein, seed=42 (matches RAR 0020). **Statistics are protein-level**: collapse the 8 designs to a per-protein median, then pair across the 50 proteins. Primary immune metric = **NetMHCIIpan** on final sequences (external validator, never in the loop — non-circular). Structure = **ESMFold scTM**. Frozen horizon, per `g_max ∈ {1.5, 2.0, 2.5}`.

- **H1 (primary, SC-GR contribution):** protein-median `immune_nmp(v_target_x_alloc) − immune_nmp(v_target) < 0`, paired Wilcoxon p < 0.05, **and** scTM non-inferior (`median scTM(alloc) ≥ median scTM(v_target) − 0.01`) — at ≥1 `g_max`, robustly.
- **Macro thesis (secondary):** `immune_nmp(v_target) < immune_nmp(flat)` and `immune_nmp(v_target_x_alloc) < immune_nmp(flat)`.
- **FAIL:** alloc not below v_target at any `g_max`, or scTM median drop > 0.02, or the macro leg fails (flat ≈ v_target ⇒ selection not actuating).
- Report the immune–scTM **Pareto front** across `g_max` for all three arms (alloc front vs v_target front). Head-burden is a mechanism check, not a gate.
- **On FAIL, use the Task-6 telemetry** to localize: `allocation_frozen_flag=false` ⇒ `Φ_i` never formed; `phi_alloc` tilted but selection unchanged ⇒ cap too small; selection tilted but immune flat ⇒ high-`Φ_i` regions structurally locked (the immune/structure overlap question).

On PASS: register a RAR, then proceed to v1.1 Value layer (`P_cheap`).

## 6. Reuse, registration, hygiene

- **Reuse-First Gate:** runs reuse `scripts/run_if_phase_c1.py` at **`N_STEPS=100`** (behavior cadence, §2 D6) with **remask held at the base setting across all arms**; eval reuses `evaluate_phase_c.py` immune+scTM; only `planner_h1_pareto.py` is new (register it). No new SLURM template — follow `scripts/submit_cnn_enhance.slurm`, `--output/--error → logs/`. Calibration JSONs via the SC2 emitter (LOG L0119), re-emitted at the 100-step cadence.
- **Budget layer reused, not rebuilt:** `pressure_source="self_conditioned_probe"` + amplify (SC2 code) done; this plan adds no pressure/β code. Base config = `d2_d3_full_stageC_scgr_betaonly_aopen.yaml`.
- **Cluster paths** only via CLI; **WT/real-data** only, fail-fast.
- **LOG.md:** one CODEMAP entry when Tasks 1–6 land (behavior-changing actuator + telemetry); experiment runs/returns need no LOG entry.

## 7. Self-review checklist (verified against source)

- Config schema: all blocks under `controller.*`; `allocation` materialized from `controller_payload.get("allocation")`; `selection_field_mode` on `TargetingConfig`. ✓ (controller_config.py:333-454,759)
- Controller API: `self.config` (not `_config`); freeze uses `scfg.freeze_after_reliable_refreshes` + `reliable` gate (controller.py:1705-1717). ✓
- Reproducibility: flat seed via `stable_seed` (sha256), not built-in `hash()`. ✓
- Base config + calibration: `scgr_betaonly` + per-`g_max` amplify JSON with matching `B_median/g_min/g_max` (run_if_phase_c1.py:181-220); `g_min=0.5` pinned. ✓
- Stats: design→protein median before Wilcoxon (n=8/protein). ✓
- Test layout: `tests/inverse_folding/` + `tests/scripts/`; real accessors `actionability_states()`/`refresh_records()`. ✓
- Telemetry: `phi_alloc`/`selection_field`/`selection_field_mode`/`allocation_frozen_flag` persisted. ✓
- Naming firewall: new mass `Φ_i` in `allocation.py`; nothing reuses `A_i(t)`. ✓
- Firewall: `Φ_i` from `residue_excess` only; injected at selection only; pressure/memory untouched. ✓
- Spec coverage: §8.3 Budget=reused, Allocation=Tasks 1–6, Value=deferred, H1=Tasks 7–8 + §5. ✓

---

# §A — Revised v1.1 (LIVE): `r_i` as triage + terminal-probe ablation (doc §8.4 Path A)

Direct-allocation (Tasks 1–8) is falsified (§8.2). This is the live plan. **Two independent ablations**, both reusing the Tasks 1–4/6 `Φ_i`/telemetry code and the Tasks 7–8 harness/eval:

- **A1 (WHERE layer):** `r_i` reorders **within** the `v_target`-eligible set, additive + capped-rank, **no widening of `τ_v`** — fixes the §8.2.1 mechanism (alloc dragged fire onto low-`v_target` sites).
- **A2 (VALUE layer):** swap the D2 candidate score from local `ΔR_B` to a **terminal-completion probe** (`P_cheap`), on the **B1** baseline. Orthogonal to A1 (score layer vs where layer; Agent-verified separable from the β/pressure layer).

**Tempered expectation (pre-registered, both):** the per-edit authority ceiling (RAR 0019 M5 / 0013: `delta_R_B`≈−0.2 flat, `ESS_candidates`≈3) bounds both. Success criterion for A1/A2 is **"recover to neutral / small positive vs the `v_target`/`local` baseline at matched scTM"**, NOT a large immune drop. A large drop would be surprising and should trigger a re-check, not celebration. The decisive lever remains Path C (doc §8.4), out of scope here.

## Task A1: `v_target_triage` selection mode

**Files:** `allocation.py` (+`triage_field`); `controller_config.py` (mode enum + `AllocationConfig` fields); `controller.py` (`_selection_field` branch, :1818-1844); test `tests/inverse_folding/test_reference_flow_allocation.py` + `..._d1_controller.py`. Reuses frozen `Φ_i` (Task 4).

- [ ] **Step 1 — failing unit tests** (`test_reference_flow_allocation.py`):

```python
import numpy as np
from inverse_folding.reference_flow.allocation import triage_field

def test_triage_lambda0_is_v_target_order_within_eligible():
    v = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    s = triage_field(v, np.array([5.,5.,5.,5.,5.]), eligible_quantile=0.5, triage_lambda=0.0)
    elig = v >= np.quantile(v, 0.5)
    # ineligible get 0; eligible ordered exactly by v_target
    assert (s[~elig] == 0).all()
    e = np.where(elig)[0]
    assert list(e[np.argsort(s[e])]) == list(e[np.argsort(v[e])])

def test_triage_reorders_toward_high_phi_within_eligible():
    v = np.array([0.8, 0.82])          # both eligible, nearly tied
    s = triage_field(v, np.array([0.2, 1.8]), eligible_quantile=0.0, triage_lambda=0.5)
    assert s[1] > s[0]                 # high-phi wins the tie-break

def test_triage_never_selects_ineligible_over_eligible():
    v = np.array([0.05, 0.9])          # idx0 ineligible
    s = triage_field(v, np.array([9.0, 0.1]), eligible_quantile=0.5, triage_lambda=1.0)
    assert s[1] > s[0]                 # huge phi at idx0 cannot lift it past eligible idx1
```

- [ ] **Step 2 — run, expect fail** (`ImportError`).
- [ ] **Step 3 — implement** in `allocation.py`:

```python
def _norm_rank(x: np.ndarray) -> np.ndarray:
    """Ranks in (0,1), stable ties. Empty ⇒ empty."""
    n = x.shape[0]
    if n == 0:
        return x.astype(float)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (order + 0.5) / n


def triage_field(
    v_target: np.ndarray, allocation: np.ndarray, *, eligible_quantile: float, triage_lambda: float
) -> np.ndarray:
    """Path-A triage (doc §8.4): within the v_target-eligible set, rank by
    ``norm_rank(v_target) + triage_lambda * norm_rank(Φ_i)``; ineligible ⇒ 0.

    v_target keeps coefficient 1 and dominates; Φ_i is a bounded (triage_lambda)
    tie-break/nudge. Eligibility is top-(1−eligible_quantile) by v_target — it does
    NOT widen v_target (§8.2.1): ineligible (low-v_target) positions score 0 and are
    never selected over eligible ones, so Φ_i cannot drag fire onto low-actionability
    sites. Φ_i ≡ 1 (or triage_lambda 0) ⇒ pure v_target order within eligible.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    L = v.shape[0]
    if L == 0:
        return v
    eligible = v >= float(np.quantile(v, eligible_quantile))
    out = np.zeros(L, dtype=float)
    idx = np.where(eligible)[0]
    if idx.size:
        out[idx] = _norm_rank(v[idx]) + float(triage_lambda) * _norm_rank(phi[idx])
    return out
```

- [ ] **Step 4 — config** (`controller_config.py`): add `"v_target_triage"` to `_SELECTION_FIELD_MODES`; add to `AllocationConfig`: `eligible_quantile: float = 0.5`, `triage_lambda: float = 0.3` (parse + print in `_materialize_allocation`). Add a config test mirroring the Task-3 pattern.
- [ ] **Step 5 — controller branch** (`controller.py` `_selection_field`, :1818-1844), before the `return v_target` default:

```python
if mode == "v_target_triage" and self._scgr_frozen_allocation is not None:
    return triage_field(
        v_target, self._scgr_frozen_allocation,
        eligible_quantile=self.config.allocation.eligible_quantile,
        triage_lambda=self.config.allocation.triage_lambda)
```

(import `triage_field` from `allocation`.) Integration test: uniform `Φ` ⇒ triage active-window set ⊆ v_target set with identical order on the eligible top; injected peaked `Φ` reorders **within** eligible only (mirror Task-5 fixtures).
- [ ] **Step 6 — run + suite green; commit** `feat(allocation): v_target_triage mode (eligible-gated additive capped-rank)`.

## Task A2: terminal candidate-probe (`P_cheap`) ablation on B1

**Files:** `counterfactual.py` (+`_score_completion_terminal`); `controller_config.py` (`D2Config.candidate_score_source` + relax the `completion_ensemble_scope` guard :635-639); `controller.py` (wire); test. Independent of A1.

**Seam (Agent-verified):** candidate scoring is hardcoded local `ΔR_B` (`counterfactual.py:923-930`, on `_build_hard_completion` argmax-fill — local, not terminal). No pluggable seam exists. Add a new method paralleling `_score_completion_ensemble` (`counterfactual.py:707-799`) that fills **all** masked positions (reuse `self_conditioned_gr.build_probe_samples`) `K_P` times with **paired seeds across candidates**, head-scores, and returns Ω(B)-LME `risk − r_current` in the **same head-logit units** as local `ΔR_B` (so the resampling weight `exp(−β·ΔR)` is unit-consistent — do NOT use `G_topm_lse`).

- [ ] **Step 1 — failing test** (`tests/inverse_folding/test_reference_flow_counterfactual.py`): with `candidate_score_source="terminal"`, `_process_block` ranks a hand-built candidate set where the terminal-immune-best tuple differs from the local-best, and the terminal scorer picks the terminal-best; with `="local"` (default) behaviour is byte-identical to today (regression guard).
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** `_score_completion_terminal(self, candidates_effective, context, omega, r_current, *, K_P, seed)` (~70–130 LOC): for each candidate, commit its tokens, `build_probe_samples` over all remaining masked positions (`K_P` paired completions), `_score_window_risk_matrix`, `compute_local_risk_batch(..., omega)`, mean over `K_P`, minus `r_current`. Add `D2Config.candidate_score_source: str = "local"` (validate `in {"local","terminal"}`) + `candidate_terminal_K_P: int = 4`; relax the `completion_ensemble_scope` guard so `"terminal"` is permitted. Branch in `_process_block` to call the terminal scorer when configured. Add a `D2BlockOutcome` telemetry field (`candidate_score_source`).
- [ ] **Step 4 — run test + full suite green; commit** `feat(d2): terminal candidate-probe scorer (P_cheap), default-off`.

## Task A3: configs + runs (reuse Tasks 7–8 eval)

- [ ] **A1 configs** — 3 on the `scgr_betaonly` base at `g_max=2.0` only (authority ceiling ⇒ no g_max sweep): `planner_flat_gmax20_aopen.yaml` + `planner_vtarget_gmax20_aopen.yaml` (exist) + **new** `planner_triage_gmax20_aopen.yaml` (`selection_field_mode: v_target_triage`, `allocation.enabled: true`, `eligible_quantile: 0.5`, `triage_lambda: 0.3`). Cohort pilot47, `N_STEPS=100`, reuse `amplify_calib_gmax20.json`. Smoke-test loads.
- [ ] **A2 configs** — 2 on the **B1** base (`d2_d3_full_stageB_aopen.yaml`, `global_pressure.enabled=false`, β=3.0): `b1_local` (existing B1, unchanged) + **new** `b1_terminal_aopen.yaml` (`d2.candidate_score_source: terminal`, `candidate_terminal_K_P: 4`). High-risk cohort (the RAR 0013 NoD-worst-100 set; cohort path via CLI). `N_STEPS=100`.
- [ ] **Eval/gate** — reuse `scripts/analysis/planner_h1_pareto.py`: A1 = triage vs v_target vs flat (the gate's `alloc_minus_vtarget` slot now reads the triage arm — pass `--arm-treatment v_target_triage`, add that thin flag); A2 = `b1_terminal` vs `b1_local` paired Wilcoxon on `immune_nmp` + scTM non-inferior. Pre-registered success = **neutral-or-better at matched scTM** (not a large drop), per the tempered expectation above.
- [ ] **Commit** `feat(configs): A1 triage + A2 terminal-probe ablation configs`.

**On A1/A2 outcomes:** register a RAR (objective measurements only). If A1 recovers to neutral and A2 ≈ neutral → confirms the authority ceiling is the binding constraint and **green-lights Path C as the main line** (the only lever left). If A2 shows a real drop → the value layer has more headroom than RAR 0019 M3 implied; re-examine before C.
