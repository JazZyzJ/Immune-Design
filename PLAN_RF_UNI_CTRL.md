# Reference Flow Unified Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage B/C unified Reference Flow controller: typed actionability targeting and gated global-risk pressure, developed and validated **together**.

**Architecture:** Stage B replaces static active-block targeting with a typed residue actionability field `A_i(t)` while keeping the Stage A actuation **mechanism** unchanged (the actuation *operating point* is reset — see "Empirical update" below). Stage C consumes the same field's `G(t)` to produce `g_GR(t)` pressure.

**Decision update (post-A_open feasibility, RAR 0005 / 0001):** Stage B is **demoted to a mechanism check**, not an outcome run, and Stage B and Stage C are **run together**. Evidence: opening the candidate space (A_open: δ=3, β=3, 5× more persisted beneficial edits, ~0 structural cost) did **not** improve the aggregate immune outcome; the aggregate is masked by a **low-burden over-intervention that is controller-wide** (present in D3-only too, RAR 0001) and is the **dominant, first-order** aggregate effect — not the second-order optimization the original §2.9 assumed. Typed targeting alone (B without C) cannot fix it, because targeting still fires on each protein's top positions regardless of absolute burden; only `g_GR` trajectory-level pressure (Stage C) suppresses it. The GR-idealized counterfactual (zero low-burden steering) recovers a ~5× larger aggregate improvement (RAR 0005 Measurement 6). Therefore B confirms the typed mechanism runs; C is the outcome lever; both run on a non-empty (>A0) operating point.

**Branch update (post-RAR 0008):** the thresholded `trajectory_thresholded_G` Stage C.1 path in this file is retained as the implemented historical baseline and as reusable actuator plumbing. New GR burden-estimator work is moved to `PLAN_RF_SC_GR.md`, starting with the monitor-only Self-Conditioned Proposal-Envelope GR probe from `doc/Self-Cond_GR.md`.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pandas/pyarrow telemetry, existing `inverse_folding/reference_flow/*` controller stack, existing `scripts/run_if_phase_c1.py` and `scripts/submit_if_phase_c.slurm`.

---

## Scope And Gates

**Design sources**

- Primary design contract: `doc/RF_Controller_Architecture.md` §2.8-§2.9.
- Implementation-stage boundary: `doc/RF_Controller_Architecture.md` §3.
- Historical consolidation source: `doc/Steering_tradeoff.md` §2.9-§2.10.

**Deployment gate**

This plan can be written and reviewed now, but implementation is gated on Stage A local gates in `PLAN_RF.md` passing:

- `final_persistence_rate >= 10%`
- structure not worse than matched D3-only baseline
- `realized_benefit_rate > 0`

If Stage A fails these local gates, do not implement this plan. Diagnose Stage A actuation first.

**Status:** Stage A local gates passed (RAR 0003). The A_open feasibility run (RAR 0005) further established that the candidate space can be opened at ~0 structural cost. **Operating-point reset:** Stage B/C experiments run on the A_open operating point (δ=3, top_k=8, max_pos=4, max_candidates=4096, β=3, min_ess_fraction=0, max_abs_logit_shift=10), **not** the v1 frozen corner. Rationale: A0's tiny action space (~64 disagreements) is too small to make the static-vs-typed targeting contrast measurable; A_open gives ~405 disagreements as the test substrate. β over-steering does not bias the typed-vs-static contrast (same β both arms) and is handled by Stage C `g_GR`.

**Ownership boundary**

| File | Owns |
|---|---|
| `PLAN_RF.md` | Phase C/D status, C0/C1/D1, and Stage A actuation contract |
| `PLAN_RF_UNI_CTRL.md` | Stage B typed targeting, historical thresholded-pressure C.1 baseline, and deferred Phase C schedule controller |
| `PLAN_RF_SC_GR.md` | SC-GR monitor-only burden-estimator branch and any later SC-GR pressure handoff |

**Out of scope**

- No model retraining.
- No NetMHC guidance signal.
- No WT/static position-specific runtime subtraction.
- No new SLURM script unless later execution discovers an unavoidable CLI gap. Use `scripts/run_if_phase_c1.py` and `scripts/submit_if_phase_c.slurm`.
- No Stage C parameter or actuator changes inside the Stage B-only mechanism arm. Stage C pressure runs use separate configs and remain the outcome layer.

---

## Core Contract

Stage B introduces a typed local actionability state:

```text
A_i(t) = {
  b_cur_i(t),
  b_env_i(t),
  b_mem_i(t),
  r_ctx_i(t),
  e_fresh_i(t),
  v_target_i(t),
  u_pressure_i(t),
  G(t),
  g_GR(t)
}
```

The controller must preserve provenance:

| Channel | Meaning | Runtime role |
|---|---|---|
| `b_cur` | current observed residue burden | visible hotspot evidence |
| `b_env` | proposal/completion envelope burden | accessible-risk evidence |
| `b_mem` | decayed cross-refresh trajectory memory | dynamic replacement for static prior |
| `r_ctx` | context reliability | gates only positive current evidence |
| `e_fresh` | fresh non-memory evidence | updates `b_mem` and Stage B full-mode D3 memory |
| `v_target` | unclustered actionability | active-block discovery and D2 targeting |
| `u_pressure` | cluster-supported actionability | global pressure aggregation |

Static prior is not used as a position-specific runtime target in this plan. Its old role, "remember where this protein tends to be risky", is replaced by `b_mem`, which is sequence-conditioned and refresh-local.

---

## Mathematical Definitions

### Background Reference

Stage B uses a provisional per-protein scalar background:

$$
\tau_{\mathrm{ref},B}(p)
=
\mathrm{median}_{w}
\left(
z^{\mathrm{static}}_{w}(p)
\right)
$$

This is a scalar normalization anchor, not a biological safety threshold and not a position-specific static prior. It is allowed because it does not encode residue/window ordering. If the static window cache is missing, Stage B v1 must fail fast; do not silently fall back to a dynamic threshold. Because `active_window_min_excess` defaults to `0.0`, this scalar controls how many windows carry positive excess; log the positive-excess window count in the refresh summary, and if it routinely saturates `max_windows`, raise `tau_ref_quantile` rather than relying on the top-N cap alone.

Default:

```yaml
controller:
  targeting:
    tau_ref_source: static_median
    tau_ref_quantile: 0.50
```

### Projection

Stage B v1 uses max-covering-window projection to preserve focal hotspots:

$$
\mathrm{Proj}_i(z_w)
=
\max_{w: i \in \mathrm{span}(w)}
z_w
$$

This projection is used for `b_cur` and `b_env`. Note this differs from the head's internal window-to-residue LME projection (`doc/Reference_Flow_Derivation.md` §D1): targeting deliberately uses max-covering to keep focal hotspots visible, so `b_cur` is not numerically identical to the head's LME residue risk.

### Current Burden

$$
h_i^{\mathrm{cur}}(t)
=
\mathrm{Proj}_i
\left(
z_w^{\mathrm{cur}}(t)
\right)
$$

$$
b_i^{\mathrm{cur}}(t)
=
\max
\left(
0,
h_i^{\mathrm{cur}}(t) - \tau_{\mathrm{ref},B}
\right)
$$

### Context Reliability

Stage B v1 uses a **discovery-before-D2** reliability. Do not read `D2BlockOutcome.g_pnll` or any D2 handler output here: D2 runs after active-block discovery, while `v_target` is the signal that discovers those blocks.

$$
r_B^{\mathrm{ctx}}(t)
=
g_B^{\mathrm{time}}(t)
\cdot
g_B^{\mathrm{comp}}(t)
\cdot
g_B^{\mathrm{ent}}(t)
\cdot
g_B^{\mathrm{pnll}}(t)
\cdot
g_B^{\mathrm{stability}}(t)
$$

Definitions:

```text
g_time, g_comp, g_ent:
  computed from the same pre-D2 quantities already used to build ActiveBlock
  records: t, span completion fraction, and mean structural entropy.

g_pnll:
  computed pre-D2 by calling the existing free function
  compute_context_pnll(...) on the candidate span/window source, then
  exp(-pnll / d2.context_pnll_h0).

g_stability:
  1.0 in Stage B v1 when previous-refresh matched logits are unavailable.
```

If `compute_context_pnll(...)` returns `None` because the span has no committed context, set `g_pnll = 0.0` and record `context_pnll = null`. This does not erase positive current evidence because `w_pos(r_ctx)` has a floor.

If `g_stability_B` is unavailable, use `1.0` and record `stability_available = false` in telemetry.

Residue reliability is block-broadcast:

```text
r_ctx_i(t) = r_ctx_B(t) for the pre-D2 source span/window that contributed max current risk to residue i
r_ctx_i(t) = 0 when no committed context exists
```

For `d_monitor_stageB`, this reliability is still computed. It is independent of `D2Handler`, so monitor-only and full modes see the same `r_ctx` field before D2 is applied.

`r_ctx` gates only `b_cur`. It does not gate `b_env` or `b_mem`.

Positive current evidence uses a floor:

$$
w_{\mathrm{pos}}(r)
=
r_{\mathrm{floor}}
+
\left(1-r_{\mathrm{floor}}\right)r
$$

Default:

```yaml
controller:
  targeting:
    r_ctx_floor: 0.25
```

This prevents unreliable context from erasing early positive evidence. Negative/freeze evidence remains stricter: low `b_cur` only supports freeze when `r_ctx` is high.

### Proposal-Envelope Burden

`b_env` is computed before D2 active-block selection. It must not be computed from the blocks it is meant to justify.

Seed windows:

```text
seed_windows =
  top env_seed_top_current windows by z_cur
  union
  top env_seed_top_uncertain windows by masked_fraction
```

Defaults:

```yaml
controller:
  targeting:
    env_seed_top_current: 16
    env_seed_top_uncertain: 16
    env_seed_max_windows: 32
    env_ensemble_size: 3
```

Stage B v1 uses **global envelope samples over the seed-window union**, not per-seed-window completions. For each envelope sample `s`:

1. Clone the current hard completion.
2. Resample only masked residues inside the union of all selected seed-window spans.
3. Use the same canonical-token, structural-temperature, and deterministic-seed conventions as the Stage A completion ensemble.
4. Score the whole completed sequence once with the head.
5. Consume only windows that overlap the seed-window union when building `b_env`.

This costs `K_env` head calls per refresh, not `num_seed_windows * K_env` calls.

For each envelope sample `s`, project the consumed local window scores to residue excess:

$$
e_{i,s}^{\mathrm{env}}(t)
=
\max
\left(
0,
\mathrm{Proj}_i
\left(
z_{w,s}^{\mathrm{env}}(t)
\right)
-
\tau_{\mathrm{ref},B}
\right)
$$

With `K_env = 3`, CVaR at 0.8 degenerates into `max`, so Stage B v1 uses max times consistency:

$$
\mathrm{peak}_i(t)
=
\max_s
e_{i,s}^{\mathrm{env}}(t)
$$

$$
\mathrm{cons}_i(t)
=
\frac{1}{K_{\mathrm{env}}}
\sum_s
\mathbf{1}
\left[
e_{i,s}^{\mathrm{env}}(t) > 0
\right]
$$

$$
b_i^{\mathrm{env}}(t)
=
\mathrm{peak}_i(t)
\left(
c_{\mathrm{floor}}
+
\left(1-c_{\mathrm{floor}}\right)
\mathrm{cons}_i(t)
\right)
$$

Default:

```yaml
controller:
  targeting:
    env_consistency_floor: 0.3333333333
```

This preserves rare-but-accessible hotspots while downweighting a single noisy completion. Only switch to CVaR when a later config raises `env_ensemble_size >= 10`.

### SoftOR

Use one anchored soft union operator:

$$
\mathrm{SoftOR}_{\tau}(x_1,\ldots,x_m)
=
\tau
\log
\left(
1
+
\sum_k
\left(
\exp(x_k / \tau) - 1
\right)
\right)
$$

Properties required by tests:

```text
SoftOR(0, 0, 0) = 0
SoftOR(x, 0, 0) = x for x > 0 up to numerical tolerance
SoftOR(x, y) >= max(x, y)
SoftOR(x, y) grows sublinearly versus x + y
```

Default:

```yaml
controller:
  targeting:
    softor_tau: 0.5
```

### Fresh Evidence And Memory Firewall

Fresh evidence excludes memory:

$$
e_i^{\mathrm{fresh}}(t)
=
\mathrm{SoftOR}_{\tau}
\left(
w_{\mathrm{pos}}
\left(
r_i^{\mathrm{ctx}}(t)
\right)
b_i^{\mathrm{cur}}(t),
b_i^{\mathrm{env}}(t)
\right)
$$

Memory updates only from fresh evidence:

$$
b_i^{\mathrm{mem}}(t)
=
\gamma_{\mathrm{mem}}
b_i^{\mathrm{mem}}(t-K)
+
\left(1-\gamma_{\mathrm{mem}}\right)
e_i^{\mathrm{fresh}}(t)
$$

$$
\gamma_{\mathrm{mem}}
=
2^{-1/H_{\mathrm{mem}}}
$$

Defaults:

```yaml
controller:
  targeting:
    mem_half_life_refreshes: 3
    mem_cold_start: zero
```

Firewall:

```text
Allowed:
  b_cur, b_env -> e_fresh -> b_mem

Forbidden:
  v_target -> b_mem
  u_pressure -> b_mem
  b_mem -> e_fresh
  D3 m_i -> b_mem
  WT/static residue prior -> b_mem initialization
```

### Targeting Versus Pressure

First compute unclustered targeting evidence:

$$
v_i^{\mathrm{target}}(t)
=
\mathrm{SoftOR}_{\tau}
\left(
w_{\mathrm{pos}}
\left(
r_i^{\mathrm{ctx}}(t)
\right)
b_i^{\mathrm{cur}}(t),
b_i^{\mathrm{env}}(t),
b_i^{\mathrm{mem}}(t)
\right)
$$

Use `v_target` for active-block discovery. Do not apply cluster support before targeting; focal hotspots must remain visible to D2.

Cluster support is only for global pressure:

$$
a_i(t)
=
\frac{
v_i^{\mathrm{target}}(t)
}{
v_i^{\mathrm{target}}(t) + \tau
}
$$

$$
\mathrm{support}_i(t)
=
\min
\left(
1,
\frac{
\sum_{|j-i| \le r_{\mathrm{cluster}}}
a_j(t)
}{
m_{\mathrm{min}}}
\right)
$$

$$
u_i^{\mathrm{pressure}}(t)
=
v_i^{\mathrm{target}}(t)
\left(
s_{\mathrm{floor}}
+
\left(1-s_{\mathrm{floor}}\right)
\mathrm{support}_i(t)
\right)
$$

Defaults:

```yaml
controller:
  targeting:
    cluster_radius: 4
    cluster_min_mass: 3.0
    cluster_floor: 0.25
    use_cluster_for_active_blocks: false
```

### Global Pressure

Because `u_pressure` is already positive excess above `tau_ref_B`, Stage B telemetry computes:

$$
G(t)
=
\frac{1}{L}
\sum_i
u_i^{\mathrm{pressure}}(t)
$$

Stage B logs `G(t)` and a diagnostic `g_GR(t)`, but does not use them to scale beta/lambda. The Stage B diagnostic scalar may use the legacy raw-sigmoid defaults below because it is telemetry only:

$$
g_{\mathrm{GR}}(t)
=
g_{\min}
+
\left(
g_{\max} - g_{\min}
\right)
\sigma
\left(
\frac{G(t)-G_0}{s_G}
\right)
$$

```yaml
controller:
  global_pressure:
    g_min: 0.25
    g_max: 1.0
    G0_source: stageB_pilot_median
    s_G_source: stageB_pilot_iqr_half
```

Stage C.1 **does not** use this diagnostic mapping. It replaces it with the trajectory-level smoothstep pressure mapping defined in the Stage C Plan (`g_min = 0.0`, calibrated `B_low/B_high`).

---

## Stage B File Structure

**Create**

- `inverse_folding/reference_flow/actionability.py`  
  Pure functions and dataclasses for `A_i(t)`: projection, `tau_ref_B`, `b_cur`, `b_env`, `e_fresh`, `b_mem`, `v_target`, `u_pressure`, `G(t)`, `g_GR(t)`.

- `tests/inverse_folding/test_reference_flow_actionability.py`  
  Unit tests for every pure operator and firewall rule.

- `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen.yaml`
  Primary Stage B mechanism-check config. It copies A_open and changes only the active-block targeting source (`static_excess` → `typed_actionability`); D3 evidence remains `legacy_window_excess` to avoid a targeting-vs-D3-memory confound.

- `inverse_folding/reference_flow/configs/d2_d3_full_stageB_unified_aopen.yaml`
  Full typed-wiring sanity config. It copies the Stage B mechanism-check config and additionally sets `d3.evidence_source = typed_fresh`. This is not the primary static-vs-typed targeting comparator.

- `inverse_folding/reference_flow/configs/d_monitor_stageB.yaml`  
  Monitor-only Stage B config for telemetry smoke runs without D2/D3 actuation.

**Modify**

- `inverse_folding/reference_flow/controller_config.py`  
  Add typed-targeting config dataclasses and defaults.

- `inverse_folding/reference_flow/controller.py`  
  Compute typed actionability at refresh boundaries and use `v_target` for active-block discovery in Stage B modes.

- `inverse_folding/reference_flow/commit.py`  
  Add explicit D3 evidence-source switch. Preserve legacy `d3_revisit` behavior.

- `inverse_folding/reference_flow/runtime.py`  
  Carry typed actionability telemetry paths and config hashes into manifests.

- `scripts/run_if_phase_c1.py`  
  Emit typed actionability artifacts and print resolved targeting/global-pressure configs at startup.

- `tests/inverse_folding/test_reference_flow_d1_controller.py`  
  Add regression tests proving legacy static active-block selection is unchanged when `targeting.mode = "static_excess"`.

- `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`  
  Add integration tests for Stage B full mode and D3 evidence-source firewall.

- `tests/scripts/test_run_if_phase_c1_d2_d3.py`  
  Add script-level smoke checks for artifact creation and manifest fields.

---

## Stage B Config Contract

Add this config section:

```yaml
controller:
  targeting:
    mode: static_excess              # static_excess | typed_actionability
    tau_ref_source: static_median    # v1 implements static_median only
    tau_ref_quantile: 0.50
    projection: max_covering_window

    env_seed_top_current: 16
    env_seed_top_uncertain: 16
    env_seed_max_windows: 32
    env_ensemble_size: 3
    env_consistency_floor: 0.3333333333

    softor_tau: 0.5
    r_ctx_floor: 0.25
    mem_half_life_refreshes: 3
    mem_cold_start: zero

    cluster_radius: 4
    cluster_min_mass: 3.0
    cluster_floor: 0.25
    use_cluster_for_active_blocks: false

    active_window_source: v_target
    active_window_min_excess: 0.0
    write_actionability_telemetry: true

  global_pressure:
    enabled: false
    # Stage B diagnostic-only default. Stage C.1 overrides to g_min: 0.0.
    g_min: 0.25
    g_max: 1.0
    G0_source: stageB_pilot_median
    s_G_source: stageB_pilot_iqr_half

  d3:
    evidence_source: legacy_window_excess   # legacy_window_excess | typed_fresh
```

All fields are nested under the existing `controller:` YAML wrapper. Do not add a second top-level config namespace.

`targeting.tau_ref_source = "first_reliable_refresh_median"` is reserved but not implemented in Stage B v1. If a config sets it, fail fast with a clear error. The only implemented source is `static_median`.

Stage B configs must keep:

```yaml
controller:
  reliability:
    min_completion_fraction: 0.0
```

`g_comp` may contribute to `r_ctx`, but completion is not allowed to become a hard disable gate in Stage B. The positive-evidence floor in `w_pos(r_ctx)` is what prevents the old completion trap from returning.

Mode defaults:

| Mode/config | `targeting.mode` | `d3.evidence_source` | Uses `g_GR` for beta/lambda |
|---|---|---|---|
| existing `d1_monitor` | `static_excess` | none | no |
| existing `d2_d3_full` | `static_excess` | `legacy_window_excess` | no |
| existing `d3_revisit` | `static_excess` | `legacy_window_excess` | no |
| new `d_monitor_stageB` | `typed_actionability` | none | no |
| new `d2_d3_full_stageB_aopen` | `typed_actionability` | `legacy_window_excess` | no |
| new `d2_d3_full_stageB_unified_aopen` | `typed_actionability` | `typed_fresh` | no |
| Stage C.1 pressure Aopen | `typed_actionability` | `legacy_window_excess` | yes |
| future Stage C unified | `typed_actionability` | `typed_fresh` | yes, after the typed-field floor is fixed |

---

## Stage B Telemetry Contract

Write a sidecar artifact rather than stuffing long arrays into `refresh_log.jsonl`.

**Artifact:** `actionability_residues.parquet`

One row per `(protein_id, design_idx, seed, refresh_step, residue_index_0b)`.

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `protein_id` | string | protein identifier |
| `design_idx` | int | generated design index |
| `seed` | int | sampler seed |
| `refresh_step` | int | controller refresh step |
| `t` | float | normalized denoise time |
| `residue_index_0b` | int | residue index |
| `tau_ref_B` | float | scalar background reference |
| `h_cur` | float | projected current risk before excess |
| `b_cur` | float | current excess |
| `b_env` | float | proposal-envelope excess |
| `env_peak` | float | max envelope excess |
| `env_consistency` | float | fraction of envelope samples above zero |
| `env_coverage_flag` | bool | residue was covered by the seed-window union |
| `b_mem` | float | memory excess after update |
| `e_fresh` | float | fresh non-memory evidence |
| `r_ctx` | float | residue reliability |
| `context_pnll` | float or null | pre-D2 pseudo-NLL for the source span/window |
| `g_time_pre` | float | pre-D2 time reliability |
| `g_comp_pre` | float | pre-D2 completion reliability |
| `g_ent_pre` | float | pre-D2 entropy reliability |
| `g_pnll_pre` | float | pre-D2 pseudo-NLL reliability |
| `g_stability_pre` | float | pre-D2 stability reliability |
| `v_target` | float | active-block targeting field |
| `u_pressure` | float | global-pressure field |
| `active_target_flag` | bool | residue belongs to selected active windows/blocks |
| `cluster_support` | float | soft support multiplier before floor |
| `d3_fresh_input_flag` | bool | this row is eligible as D3 typed fresh evidence |

**Artifact:** `actionability_refresh_summary.jsonl`

One JSON object per refresh:

```json
{
  "protein_id": "example",
  "design_idx": 0,
  "seed": 42,
  "refresh_step": 6,
  "t": 0.60,
  "targeting_mode": "typed_actionability",
  "tau_ref_source": "static_median",
  "tau_ref_B": 0.37,
  "G": 0.041,
  "g_GR_diagnostic": 0.52,
  "num_seed_windows": 24,
  "num_env_head_calls": 3,
  "num_active_windows": 8,
  "num_active_blocks": 3,
  "mean_b_cur": 0.019,
  "mean_b_env": 0.024,
  "mean_b_mem": 0.017,
  "max_v_target": 1.21,
  "max_u_pressure": 0.74,
  "stability_available": false,
  "d3_evidence_source": "typed_fresh"
}
```

Extend existing `refresh_log.jsonl` with compact pointers only:

```json
{
  "actionability_residues_path": "actionability_residues.parquet",
  "actionability_refresh_summary_path": "actionability_refresh_summary.jsonl",
  "targeting_mode": "typed_actionability",
  "G": 0.041,
  "g_GR_diagnostic": 0.52
}
```

---

## Stage B Tasks

### Task B1: Add Pure Actionability Operators

**Files**

- Create: `inverse_folding/reference_flow/actionability.py`
- Test: `tests/inverse_folding/test_reference_flow_actionability.py`

- [ ] **Step B1.1: Write failing tests for projection and SoftOR**

Add tests:

```python
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
```

- [ ] **Step B1.2: Run tests and verify failure**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py -v
```

Expected: FAIL because `actionability.py` does not exist.

- [ ] **Step B1.3: Implement projection and SoftOR**

Create `actionability.py` with these public functions:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def max_covering_window_projection(
    *,
    length: int,
    windows: Sequence[Mapping[str, float | int]],
) -> np.ndarray:
    values = np.full(length, -np.inf, dtype=float)
    for window in windows:
        start = int(window["start"])
        end = int(window["end"])
        score = float(window["score"])
        values[start:end] = np.maximum(values[start:end], score)
    values[~np.isfinite(values)] = 0.0
    return values


def soft_or(values: Iterable[float], *, tau: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < -1e-12):
        raise ValueError("soft_or expects nonnegative evidence values")
    if float(np.max(arr)) == 0.0:
        return 0.0
    return float(tau * np.log1p(np.sum(np.expm1(arr / tau))))
```

- [ ] **Step B1.4: Run tests and verify pass**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py -v
```

Expected: PASS for the new tests.

### Task B2: Add Config Dataclasses And Defaults

**Files**

- Modify: `inverse_folding/reference_flow/controller_config.py`
- Test: `tests/inverse_folding/test_reference_flow_controller_config.py`

- [ ] **Step B2.1: Write config parsing tests**

Add tests that assert:

```python
from inverse_folding.reference_flow.controller_config import (
    materialize_controller_config,
)


def _base_enabled_payload():
    return {
        "controller": {
            "enabled": True,
            "mode": "monitor_only",
            "t_start": 0.5,
            "refresh_interval": 5,
            "completion": {"method": "argmax"},
            "head": {
                "score_scale": "raw_logit",
                "static_cache_policy": "lazy_write",
                "local_risk_aggregation": "LME",
            },
            "active_windows": {
                "excess_threshold": 0.0,
                "max_windows": 16,
                "selection": "threshold_then_top_n",
                "merge_overlapping_scoring_windows": True,
            },
            "reliability": {
                "time_k": 20.0,
                "entropy_h0": 1.5,
                "min_completion_fraction": 0.0,
                "min_rho_to_emit_event": 0.0,
            },
            "d2": {"enabled": False},
            "d3": {"enabled": False},
            "attribution": {
                "write_paired_counterfactual": False,
                "write_independent_delta_R_i": False,
                "productive_delta_logp": 0.5,
            },
            "controls": {
                "allow_wrong_allele_head": True,
                "allow_shuffled_head": True,
            },
            "telemetry": {
                "write_refresh_log": True,
                "write_controller_events": True,
                "write_per_protein_summary": True,
            },
        }
    }


def test_targeting_config_defaults_to_static_excess():
    config = materialize_controller_config(_base_enabled_payload())
    assert config.targeting.mode == "static_excess"
    assert config.targeting.use_cluster_for_active_blocks is False
    assert config.global_pressure.enabled is False
    assert config.d3.evidence_source == "legacy_window_excess"


def test_typed_targeting_config_validates_modes():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {
        "enabled": False,
        "evidence_source": "typed_fresh",
    }
    config = materialize_controller_config(payload)
    assert config.targeting.mode == "typed_actionability"
    assert config.d3.evidence_source == "typed_fresh"
```

- [ ] **Step B2.2: Run tests and verify failure**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -v
```

Expected: FAIL because the new fields do not exist.

- [ ] **Step B2.3: Add config dataclasses**

Add dataclasses equivalent to:

```python
@dataclass(frozen=True)
class TargetingConfig:
    mode: str = "static_excess"
    tau_ref_source: str = "static_median"
    tau_ref_quantile: float = 0.50
    projection: str = "max_covering_window"
    env_seed_top_current: int = 16
    env_seed_top_uncertain: int = 16
    env_seed_max_windows: int = 32
    env_ensemble_size: int = 3
    env_consistency_floor: float = 1.0 / 3.0
    softor_tau: float = 0.5
    r_ctx_floor: float = 0.25
    mem_half_life_refreshes: float = 3.0
    mem_cold_start: str = "zero"
    cluster_radius: int = 4
    cluster_min_mass: float = 3.0
    cluster_floor: float = 0.25
    use_cluster_for_active_blocks: bool = False
    active_window_source: str = "v_target"
    active_window_min_excess: float = 0.0
    write_actionability_telemetry: bool = True


@dataclass(frozen=True)
class GlobalPressureConfig:
    enabled: bool = False
    g_min: float = 0.25
    g_max: float = 1.0
    G0_source: str = "stageB_pilot_median"
    s_G_source: str = "stageB_pilot_iqr_half"
```

Validation rules:

```text
targeting.mode in {"static_excess", "typed_actionability"}
d3.evidence_source in {"legacy_window_excess", "typed_fresh"}
targeting.tau_ref_source in {"static_median", "first_reliable_refresh_median"}
targeting.tau_ref_source == "first_reliable_refresh_median" raises NotImplementedError in Stage B v1
targeting.env_ensemble_size >= 1
targeting.mem_half_life_refreshes > 0
targeting.r_ctx_floor in [0, 1]
global_pressure.g_min >= 0
global_pressure.g_max >= global_pressure.g_min
```

- [ ] **Step B2.4: Run config tests**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -v
```

Expected: PASS.

### Task B3: Compute Stage B Fields With Firewall

**Files**

- Modify: `inverse_folding/reference_flow/actionability.py`
- Test: `tests/inverse_folding/test_reference_flow_actionability.py`

- [ ] **Step B3.1: Write tests for `b_env`, `e_fresh`, `b_mem`, and target/pressure split**

Add tests:

```python
def test_env_burden_uses_peak_times_consistency_for_small_k():
    samples = np.array([
        [1.0, 0.0],
        [0.0, 0.5],
        [0.0, 0.7],
    ])
    out = envelope_burden_from_excess_samples(samples, consistency_floor=1.0 / 3.0)
    np.testing.assert_allclose(out, [1.0 * (1/3 + 2/3 * 1/3), 0.7 * (1/3 + 2/3 * 2/3)])


def test_fresh_evidence_excludes_memory():
    fresh = compute_fresh_evidence(
        b_cur=np.array([1.0]),
        b_env=np.array([0.0]),
        r_ctx=np.array([0.0]),
        r_ctx_floor=0.25,
        tau=0.5,
    )
    target = compute_target_evidence(
        fresh_evidence=fresh,
        b_mem=np.array([10.0]),
        tau=0.5,
    )
    assert fresh[0] < 1.0
    assert target[0] > fresh[0]


def test_cluster_support_does_not_feed_active_targeting():
    v = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
    pressure = cluster_supported_pressure(
        v_target=v,
        tau=0.5,
        radius=1,
        min_mass=3.0,
        floor=0.25,
    )
    assert v[1] == 2.0
    assert pressure[1] < v[1]
```

- [ ] **Step B3.2: Run tests and verify failure**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py -v
```

Expected: FAIL for missing functions.

- [ ] **Step B3.3: Implement field functions**

Implement public functions:

```python
def positive_evidence_weight(r_ctx: np.ndarray, *, floor: float) -> np.ndarray: ...
def excess_over_tau(values: np.ndarray, *, tau_ref: float) -> np.ndarray: ...
def envelope_burden_from_excess_samples(samples: np.ndarray, *, consistency_floor: float) -> np.ndarray: ...
def compute_fresh_evidence(...): ...
def update_memory(...): ...
def compute_target_evidence(...): ...
def cluster_supported_pressure(...): ...
def global_pressure_mass(u_pressure: np.ndarray) -> float: ...
```

The `update_memory` function must accept `previous_b_mem`, `e_fresh`, and `half_life_refreshes`; it must never accept `v_target`, `u_pressure`, or D3 `m_i`.

- [ ] **Step B3.4: Run tests and verify pass**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py -v
```

Expected: PASS.

### Task B4: Integrate Typed Targeting Into Controller Refresh

**Files**

- Modify: `inverse_folding/reference_flow/controller.py`
- Modify: `inverse_folding/reference_flow/runtime.py`
- Test: `tests/inverse_folding/test_reference_flow_d1_controller.py`
- Test: `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`

- [ ] **Step B4.1: Write legacy bit-equivalence test**

Add a test that constructs the same synthetic refresh context twice:

```python
baseline_config = _make_config(mode="monitor_only", d2_enabled=False, d3_enabled=False)
legacy_config = dataclasses.replace(
    baseline_config,
    targeting=TargetingConfig(mode="static_excess"),
)
baseline = ReferenceFlowController(
    protein_id="p",
    design_idx=0,
    seed=42,
    static_sequence="AAAAAAAA",
    scorer=_StubScorer(),
    config=baseline_config,
    decode_tokens=_decode,
    canonical_token_ids=_CANONICAL,
)
legacy = ReferenceFlowController(
    protein_id="p",
    design_idx=0,
    seed=42,
    static_sequence="AAAAAAAA",
    scorer=_StubScorer(),
    config=legacy_config,
    decode_tokens=_decode,
    canonical_token_ids=_CANONICAL,
)
baseline_result = baseline.step(ctx)
legacy_result = legacy.step(ctx)
assert baseline_result.refresh_record.active_blocks == legacy_result.refresh_record.active_blocks
assert torch.allclose(baseline_result.logits, legacy_result.logits)
```

Use the existing local test helpers in `tests/inverse_folding/test_reference_flow_d2_d3_controller.py` or equivalent file-local helpers. Do not add a new public `from_dict` or `controller_with_config` API just for tests.

- [ ] **Step B4.2: Write typed-targeting active-block test**

Use a synthetic `v_target` field with one focal peak and one broad cluster:

```python
v_target = np.array([0, 0, 3.0, 0, 0, 1.0, 1.1, 1.2, 0], dtype=float)
u_pressure = np.array([0, 0, 0.8, 0, 0, 1.0, 1.1, 1.2, 0], dtype=float)
```

Assert:

```text
active-block discovery sees residue 2 from v_target
G(t) uses u_pressure
the focal residue is not removed by cluster support before active-window selection
```

Also assert the Stage B envelope path uses exactly `env_ensemble_size` head calls per refresh for envelope probing, not `num_seed_windows * env_ensemble_size`. A stub scorer with a call counter is sufficient.

- [ ] **Step B4.3: Implement typed refresh state**

Add a controller-owned refresh state field:

```python
@dataclass
class UnifiedActionabilityState:
    tau_ref_B: float
    h_cur: np.ndarray
    b_cur: np.ndarray
    b_env: np.ndarray
    env_peak: np.ndarray
    env_consistency: np.ndarray
    e_fresh: np.ndarray
    b_mem: np.ndarray
    r_ctx: np.ndarray
    v_target: np.ndarray
    u_pressure: np.ndarray
    G: float
    g_GR_diagnostic: float
```

Store the latest state on the controller. Keep legacy fields intact.

- [ ] **Step B4.4: Route active windows by mode**

Mode behavior:

```text
targeting.mode = static_excess:
  active windows come from legacy z_dyn - z_static path

targeting.mode = typed_actionability:
  window_actionability_w = max_i_in_window v_target_i
  select windows with window_actionability_w > active_window_min_excess
  sort descending by window_actionability_w
  cap by existing max_windows
  use existing span/block merge logic
```

Typed mode Stage B.0 has a narrow action boundary:

```text
Changed:
  active-window selection source = v_target
  D3 m_i evidence source in d2_d3_full_stageB = e_fresh

Unchanged:
  legacy window_excess is still computed
  new_hotspot_count still uses legacy window_excess/static windows
  D1RefreshRecord.window_excess still records the legacy value
  D2 select_editable_positions(...) still receives legacy residue_excess
  D2 candidate enumeration/scoring remains Stage A behavior
```

This is intentional. Stage B.0 tests whether changing the **block targeting source** is enough. If typed-selected blocks often contain high `v_target` residues but D2 edits different low-`v_target` positions because legacy `residue_excess` is zero, record that as a Stage B.1 diagnostic and then consider changing D2's within-block `residue_excess` input to `v_target` or `e_fresh`. Do not make that change in Stage B.0.

- [ ] **Step B4.5: Run controller tests**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_d1_controller.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py -v
```

Expected: PASS. Legacy tests must remain unchanged in behavior.

### Task B5: Add D3 Evidence-Source Firewall

**Files**

- Modify: `inverse_folding/reference_flow/commit.py`
- Modify: `inverse_folding/reference_flow/controller.py`
- Test: `tests/inverse_folding/test_reference_flow_commit.py`
- Test: `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`

- [ ] **Step B5.1: Write D3 mode tests**

Add tests:

```python
def test_d3_revisit_keeps_legacy_window_excess_input():
    config = dataclasses.replace(
        _make_config(mode="d3_revisit", d2_enabled=False, d3_enabled=True).d3,
        evidence_source="legacy_window_excess",
    )
    result = compute_d3_evidence_input(
        d3_config=config,
        legacy_e_i=np.array([1.0]),
        typed_fresh_i=np.array([9.0]),
    )
    np.testing.assert_allclose(result, [1.0])


def test_stageB_full_uses_typed_fresh_input():
    config = dataclasses.replace(
        _make_config(mode="d2_d3_full", d2_enabled=True, d3_enabled=True).d3,
        evidence_source="typed_fresh",
    )
    result = compute_d3_evidence_input(
        d3_config=config,
        legacy_e_i=np.array([1.0]),
        typed_fresh_i=np.array([9.0]),
    )
    np.testing.assert_allclose(result, [9.0])
```

`compute_d3_evidence_input(...)` can be a small private helper in `commit.py` or `controller.py`; it exists to make the mode firewall directly testable. Do not add a run-level test-only API.

- [ ] **Step B5.2: Implement evidence switch**

Rules:

```text
d3_revisit and all existing D3-only configs:
  D3 m_i input = legacy window-excess projection

d2_d3_full_stageB:
  D3 m_i input = e_fresh

Never:
  D3 m_i -> b_mem
  b_mem-inflated v_target -> D3 m_i update
```

- [ ] **Step B5.3: Run D3 tests**

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_commit.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py -v
```

Expected: PASS.

### Task B6: Add Stage B Configs And Telemetry Artifacts

**Files**

- Create/modify: `inverse_folding/reference_flow/configs/d2_d3_full_stageB.yaml`
- Create/modify: `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen.yaml`
- Create: `inverse_folding/reference_flow/configs/d2_d3_full_stageB_unified_aopen.yaml`
- Create: `inverse_folding/reference_flow/configs/d_monitor_stageB.yaml`
- Modify: `scripts/run_if_phase_c1.py`
- Test: `tests/scripts/test_run_if_phase_c1_d2_d3.py`

- [ ] **Step B6.1: Add YAML configs**

`d2_d3_full_stageB.yaml` and `d2_d3_full_stageB_unified_aopen.yaml` must set the full typed D3 evidence source:

```yaml
controller:
  reliability:
    min_completion_fraction: 0.0

  targeting:
    mode: typed_actionability
    tau_ref_source: static_median
    tau_ref_quantile: 0.50
    projection: max_covering_window
    env_seed_top_current: 16
    env_seed_top_uncertain: 16
    env_seed_max_windows: 32
    env_ensemble_size: 3
    env_consistency_floor: 0.3333333333
    softor_tau: 0.5
    r_ctx_floor: 0.25
    mem_half_life_refreshes: 3
    cluster_radius: 4
    cluster_min_mass: 3.0
    cluster_floor: 0.25
    use_cluster_for_active_blocks: false
    active_window_source: v_target
    active_window_min_excess: 0.0
    write_actionability_telemetry: true

  global_pressure:
    enabled: false

  d3:
    evidence_source: typed_fresh
```

`d2_d3_full_stageB_aopen.yaml` is different by design: it must copy the A_open D2/D3 operating point and set `targeting.mode = typed_actionability`, but keep `d3.evidence_source = legacy_window_excess`. This makes the primary Stage B mechanism run a pure static-vs-typed targeting comparison.

`d_monitor_stageB.yaml` must set the same `targeting` block and keep D2/D3 actuation disabled.

- [ ] **Step B6.2: Add script-level telemetry test**

Assert a tiny scripted run writes:

```text
actionability_residues.parquet
actionability_refresh_summary.jsonl
manifest.json with targeting_config_hash
refresh_log.jsonl with actionability sidecar paths
```

- [ ] **Step B6.3: Implement artifact writing**

In `scripts/run_if_phase_c1.py`:

```text
print resolved targeting config at startup, one key per line
print resolved global_pressure config at startup, one key per line
write actionability_residues.parquet when targeting.write_actionability_telemetry is true
write actionability_refresh_summary.jsonl when targeting.write_actionability_telemetry is true
record file paths in manifest and refresh_log
```

- [ ] **Step B6.4: Run script tests**

Run:

```bash
pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -v
```

Expected: PASS.

---

## Stage B Mechanism-Check Experiment

Two matched 50-protein primary arms on the A_open operating point; the only difference is the targeting source. The static arm is **already run** — reuse RAR 0005.

| arm | config | targeting | D3 evidence | status |
|---|---|---|---|---|
| static comparator | `stageA_d2_d3_full_A_open.yaml` (RAR 0005 bundle `meta/`) | `static_excess` | `legacy_window_excess` | done (RAR 0005) |
| typed-targeting treatment | `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen.yaml` | `typed_actionability` | `legacy_window_excess` | to run |
| full typed-wiring sanity | `inverse_folding/reference_flow/configs/d2_d3_full_stageB_unified_aopen.yaml` | `typed_actionability` | `typed_fresh` | optional sanity, not the primary comparator |

`d2_d3_full_stageB_aopen.yaml` copies the A_open `d2` operating point verbatim (β=3, δ_struct=3, top_k=8, max_pos=4, max_candidates=4096, min_ess_fraction=0, max_abs_logit_shift=10) and adds the typed `targeting` block while keeping `d3.evidence_source=legacy_window_excess`; `global_pressure.enabled=false` (G(t)/g_GR diagnostic-only). This is the only run used for the pure static-vs-typed targeting claim.

`d2_d3_full_stageB_unified_aopen.yaml` changes one additional field, `d3.evidence_source=typed_fresh`, to verify the full typed D3 memory route. Do not use it to claim that targeting alone changed the result.

Run (typed arm only; static arm is RAR 0005). Use the same standard Phase C1 inputs and the same 50-protein input parquet as A_open; the controller-specific flags are:

```bash
python scripts/run_if_phase_c1.py [standard Phase C1 args] \
  --controller-config inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen.yaml \
  --n-designs-per-protein 4 \
  --seed 42
```

Read-out is the Stage B gate below (mechanism, not outcome). Spatial redistribution (gate condition 3) joins typed `actionability_residues.parquet` against the static A_open `controller_events.parquet` on matched `(protein_id, design_idx, refresh_step)`. Do not compute aggregate Pareto or NMP here — those are Stage C.

---

## Stage B Acceptance

Local verification:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py \
       tests/inverse_folding/test_reference_flow_controller_config.py \
       tests/inverse_folding/test_reference_flow_d1_controller.py \
       tests/inverse_folding/test_reference_flow_d2_d3_controller.py \
       tests/inverse_folding/test_reference_flow_commit.py \
       tests/scripts/test_run_if_phase_c1_d2_d3.py -v
```

Expected: all pass.

Real-head smoke:

```bash
python scripts/run_if_phase_c1.py [standard Phase C1 args with a 2-protein smoke input parquet] \
  --controller-config inverse_folding/reference_flow/configs/d_monitor_stageB.yaml \
  --n-designs-per-protein 1 \
  --seed 42
```

Expected artifacts:

```text
generated.parquet
refresh_log.jsonl
actionability_residues.parquet
actionability_refresh_summary.jsonl
manifest.json
```

50-protein Stage B gate (mechanism check, not outcome — see "Decision update"):

Stage B's gate confirms the typed targeting **mechanism runs and redistributes** D2's fire on the A_open operating point. It does **not** gate on the aggregate immune outcome or the D2-vs-D3-only Pareto: RAR 0005 showed those are masked by controller-wide low-burden over-intervention until Stage C `g_GR` is active. Comparator for all primary contrasts is the already-run static-targeting A_open arm (RAR 0005) — identical operating point and `d3.evidence_source=legacy_window_excess`, with only `targeting.mode` changed.

1. Stage A local gates remain satisfied with typed targeting at the A_open operating point (`final_persistence_rate >= 10%`, `realized_benefit_rate > 0`, structure not worse than matched D3-only).
2. **Typed field populated**: `b_cur`, `b_env`, `b_mem`, `r_ctx`, `v_target` are non-degenerate across the pilot, the firewall and `r_ctx`-gating behave per `actionability_residues.parquet`, and `b_env` is computed pre-D2 (no leakage from the active blocks it justifies).
3. **Redistribution runs and is directional**: typed `v_target` selects measurably different active blocks / editable positions than the static comparator on matched proteins, and the typed-selected positions carry higher `v_target` / proposal-envelope `b_env` than the static-selected ones. This is a spatial redistribution metric over `(protein, refresh, residue)`, **not** an aggregate immune delta.
4. Focal diagnostic passes: at least one refresh shows a residue with high `v_target` but reduced `u_pressure`, proving focal hotspots can be targeted without inflating global pressure.
5. Memory firewall passes: when `e_fresh` falls to zero across consecutive refreshes, `b_mem` decays by the configured half-life and cannot grow.

**RAR 0006 verdict (typed Aopen mechanism arm, already run).** Gates 1–3 (the *measured* typed-mechanism gates) passed: the field is populated, the firewall/`r_ctx`-gating behave as specified, and `v_target` redistributes D2's fire vs the static comparator. **Gates 4 and 5 are reclassified `dense-field N/A`, not failures.** The typed pilot's background floor (`tau_ref_B` median ≈ −14.35 + max-covering projection) makes `u_pressure > 0` for ~100% of residues and `e_fresh > 0` for ~100%, so neither "high `v_target` with *reduced* `u_pressure`" (gate 4) nor "`e_fresh` falls to zero across refreshes" (gate 5) can be cleanly demonstrated on this field. They do **not** block Stage C and must not be treated as a re-run target on the current field; they become testable only after the field-floor fix (`tau_ref_quantile` / max-covering projection sharpening — G10, deferred). The dense floor is also the root cause behind the G3 driver change (`mean(u_pressure)` floor-dominated) and the G10 `typed_fresh` deferral.

Telemetry note: burden-stratified and typed-vs-static **attribution** telemetry (separating the targeting contribution from the GR contribution, and reporting the high-burden D2 marginal) is implemented in **Stage C**, not Stage B — see Stage C Plan.

If Stage B fails condition 2 or 3 while Stage A gates remain good, the typed-field **wiring** is the problem — fix `actionability.py` / controller integration; do **not** fall back to revisit-only. A flat aggregate Pareto at this stage is **expected** (it is GR's job, Stage C), not a falsification of D2; the §5.3 / §6.D Pareto verdict is deferred to after Stage C.

---

## Stage B.1: Within-Block Typed Position Selection

**Motivation (RAR 0006 M7).** Stage B.0 routed the *block* source to `v_target`, but `select_editable_positions` still ranks *within-block* positions by legacy `residue_excess`. M7 shows the typed signal is not reaching the position layer: `edited_in_high_v_top4_frac = 0.448` (< 0.5), within-block Spearman(`v_target`, edited) only 0.31, and legacy excess (0.52) is in fact slightly better aligned. The typed field's within-block value is unrealized. B.1 fixes this and runs **together with C** on the typed_target A_open operating point.

**Change.** `select_editable_positions` takes a configurable score source:

```yaml
controller:
  targeting:
    within_block_source: legacy_excess   # legacy_excess | v_target
```

Mode-default-OFF: existing/static configs default `legacy_excess` (bit-equivalent to today). `d2_d3_full_stageB_aopen.yaml` and the C.1 config set `within_block_source: v_target`. The low-entropy tiebreak is retained as a secondary key.

**Files**
- `inverse_folding/reference_flow/counterfactual.py`: `select_editable_positions(..., score_source, typed_field)` — rank by the chosen source, fall back to legacy when the typed field is absent.
- `inverse_folding/reference_flow/controller_config.py`: add `within_block_source` to `TargetingConfig` (default `legacy_excess`), validate the enum.
- `inverse_folding/reference_flow/controller.py`: pass the per-residue `v_target` of the active block into `select_editable_positions` in typed modes.
- Tests: `tests/inverse_folding/test_reference_flow_counterfactual.py` (or the editable-position test file).

`e_fresh` is deliberately not a v1 within-block score source. RAR 0006 showed the same typed-field floor that made `typed_fresh` harmful for D3; keep `e_fresh` reserved until the field-floor fix makes it sharp enough to act on.

**Tests (TDD)**
- `within_block_source=legacy_excess` reproduces today's editable-position selection bit-for-bit.
- `within_block_source=v_target` on a synthetic block with one high-`v_target` low-`legacy_excess` residue selects that residue.
- typed field absent → falls back to legacy without error.

**Acceptance (mechanism, re-run M7 on the B.1 typed run).**
- `edited_in_high_v_top4_frac` rises markedly above the 0.448 B.0 baseline, and Spearman(`v_target`, edited) > the 0.31 B.0 value.
- Stage A local gates still hold (persistence, realized_benefit, structure no worse than matched D3-only).
- This is a mechanism/redistribution gain; aggregate immune/Pareto remains a Stage C (GR) verdict.

---

## Stage C Plan

**Current GR-estimator branch note:** this section documents the thresholded `trajectory_thresholded_G` C.1 route that has already been implemented and evaluated as a baseline. Post-RAR 0008 SC-GR estimator work should be planned from `PLAN_RF_SC_GR.md`; do not extend this section with the self-conditioned monitor or behavior handoff.

**Priority update (RAR 0005 / 0001):** Stage C is **not** deferred second-order tuning. `g_GR` trajectory-level pressure is the first-order lever for the aggregate immune outcome, because controller-wide low-burden over-intervention masks the real high-burden D2 gain. A_open showed that opening the D2 candidate space gives many more persisted beneficial edits at near-zero structural cost, but fixed-β steering still regresses the low-burden tercile (`Aopen_dhead_median = +2.080`) while preserving a strong high-burden gain (`Aopen_dhead_median = -2.567`). The GR-idealized counterfactual zeroing low-burden steering moves aggregate mean head delta from `-0.244` to `-1.211`. C.1 therefore must make low-burden pressure approach **zero**, not merely shrink to `g_min=0.25`.

Stage C is split into two sub-stages:

| Sub-stage | Scope | Status |
|---|---|---|
| **C.1 trajectory-level global pressure** | Scale D2 β and D3 λ by a stable per-design trajectory burden scalar. No schedule changes. | implement first |
| **C.2 Phase C editability allocation** | Reallocate unmask opportunity using `u_hold/u_exec/u_freeze`. | gated after C.1; needs new scheduler API |

### Stage C Decisions

These decisions resolve the C-stage gaps before coding.

| Gap | Decision |
|---|---|
| G1 `g_min` | Primary C.1 config uses `g_min = 0.0`. Optional sensitivity uses `g_min = 0.05`. `g_min = 0.25` remains Stage B diagnostic-only and is not valid for the C.1 outcome run. |
| G2 pressure time scale | Actuation reads a stable per-design trajectory pressure state `B_GR`, not raw per-refresh `G_step`. `G_step` is still logged at every refresh. |
| G3 `G(t)` formula | **Revised (RAR 0006 M8/M11).** `mean(u_pressure)` is dominated by the dense-field floor and does **not** discriminate burden (typed `tau_ref_B` median ≈ −14.35 + max-covering projection ⇒ `u_pressure>0` for ~100% of residues; per-design `median_G` separates only low-vs-non-low). Use the doc §2.8 prominence-thresholded mass `G_step(t) = (1/L) Σ_i ReLU(u_pressure_i(t) − τ_prom)`, which M11 showed correlates markedly better with d3 burden (medium, not strong). **`τ_prom` is a second prominence cut, NOT a re-subtraction of `tau_ref_B`** (`u_pressure` is already excess over `tau_ref_B`; `τ_prom` selects which excess counts as prominent burden), calibrated from the typed pilot. Fallback form if ReLU-mass underperforms on the C run: count `(1/L) Σ_i I(u_pressure_i ≥ τ_prom)`. |
| G4 C.2 fields | `c_ready_i` and `f_rank_i` are required before C.2. They are not required for C.1. |
| G5 scheduler correctness | C.2 intentionally changes positionwise unmask ordering. It preserves total unmask budget but does **not** preserve Theorem 1's positionwise ordering guarantee. Keep it off in C.1. |
| G6 burden proxy | Runtime burden bins use the final per-design `B_GR` from the same trajectory-level proxy that drives `g_GR`; do not use offline `d3_t050` at runtime. |
| G7 acceptance | Use burden-stratified A_open targets, but **do not require low-bin → 0**: the GR burden driver is only medium-correlated with true burden (RAR 0006 M11), so require **substantial** low-bin reduction (low-bin median head delta improves ≥50% vs A_open's `+2.080`), high bin retains about the `-2.5` median head gain, structure no worse than matched D3-only. Sharper discrimination requires fixing the field floor (G10), deferred. |
| G8 sticky TTL | C.1 freezes `β_eff` into each D2 correction at refresh creation. Sticky re-delivery within TTL reuses the stored delta logits; it does not recompute β. |
| G9 β/λ sharing | v1 uses the same `g_GR` for D2 β and D3 λ. Split them only if high-burden D3 revisit is visibly over-suppressed. |
| G10 base config + field floor | C.1 base config is **`d2_d3_full_stageB_aopen.yaml` (typed_target, `d3.evidence_source=legacy_window_excess`)**, NOT the unified arm: RAR 0006 M9 shows `typed_fresh` (unified) is harmful (dense `e_fresh` inflates the D3 EMA → more low-burden over-intervention). `typed_fresh` is **deferred, not abandoned** — it is the eventual unified config and should become viable once the field floor (`tau_ref_quantile` / max-covering projection) is sharpened. That field-floor fix is the deeper lever behind both GR discrimination and `typed_fresh` viability; deferred to a follow-up, after C.1 with the thresholded driver. |

### Stage C.1: Global Pressure Scaling

**Goal:** suppress controller effort on intrinsically low-actionability trajectories while preserving the high-burden D2/D3 gain. C.1 is an actuator-only change over the Stage B typed controller: active-block discovery, D2 candidate generation, D3 evidence source, reliability gates, sticky TTL, and Bernoulli schedule stay unchanged.

#### C.1 Runtime Definitions

At every typed refresh, the controller aggregates the typed pressure field into a scalar. **Per G3 / RAR 0006 M11, use the prominence-thresholded mass, not the mean** (the mean is floor-dominated and does not discriminate burden):

$$
G_{\mathrm{step}}(t)
=
\frac{1}{L}
\sum_i
\mathrm{ReLU}\!\left(u_i^{\mathrm{pressure}}(t) - \tau_{\mathrm{prom}}\right)
$$

`τ_prom` is the prominence cut calibrated from the typed pilot (see C.1 Calibration). The Stage-B `mean(u_pressure)` is retained only as a `G_step_mean` diagnostic column.

`G_step` is a refresh-level observation, not the actuator input. The actuator input is the stable per-design trajectory pressure burden:

$$
B_{\mathrm{GR}}(t)
=
\mathrm{median}
\left(
\{G_{\mathrm{step}}(r): r \le t,\ r\ \mathrm{reliable}\}
\right)
$$

For v1, a refresh is reliable if the typed actionability state exists and all of `G_step`, `mean_v_target`, and `mean_u_pressure` are finite. This intentionally reuses Stage B's typed-field reliability rather than adding a second head-variance gate.

Map `B_GR` to pressure with a clipped smoothstep, not a sigmoid:

$$
q(t)
=
\mathrm{clip}
\left(
\frac{
B_{\mathrm{GR}}(t)-B_{\mathrm{low}}
}{
B_{\mathrm{high}}-B_{\mathrm{low}}
},
0,
1
\right)
$$

$$
s(q)
=
3q^2 - 2q^3
$$

$$
g_{\mathrm{GR}}(t)
=
g_{\min}
+
\left(
g_{\max}-g_{\min}
\right)
s(q(t))
$$

Primary C.1 settings:

```yaml
controller:
  global_pressure:
    enabled: true
    pressure_source: trajectory_thresholded_G   # median_t of (1/L)Σ ReLU(u_pressure - tau_prom); G3
    tau_prom_source: calibration_json
    mapping: smoothstep
    g_min: 0.0
    g_max: 1.0
    B_low_source: calibration_json
    B_high_source: calibration_json
    min_reliable_refreshes: 1
    unready_g: 0.0
    scale_beta: true
    scale_lambda: true
```

`B_low` and `B_high` come from the Stage B/Aopen typed pilot, using the same runtime proxy. The calibration unit is one `(protein_id, design_idx, seed)` trajectory, not a protein-wide guarantee:

```text
unit = median G_step per (protein_id, design_idx, seed)
B_low  = quantile(unit, 1/3)
B_high = quantile(unit, 2/3)
```

`min_reliable_refreshes=1` is intentional for the primary config because short 10-step runs may expose only one post-`t_start` refresh. The first `B_GR` is therefore a one-sample median and can be noisy; this is accepted in C.1 because early actuation is already limited by committed-context and D2 feasibility. If the run has at least two post-start refreshes, a sensitivity config may set `min_reliable_refreshes=2` to reduce first-refresh spikes.

The calibration artifact is a JSON file passed by CLI, never hardcoded in Python modules:

```json
{
  "schema_version": "stageC1_global_pressure_calibration.v2",
  "pressure_source": "trajectory_thresholded_G",
  "tau_prom": 11.75,
  "low_quantile": 0.3333333333,
  "high_quantile": 0.6666666667,
  "B_low": 1.3291509778635653,
  "B_high": 1.9728705118798107,
  "source_artifact": "typed_actionability_residue_rows.parquet"
}
```

These are the **actual calibrated anchors from the RAR 0006 typed_target pilot** (`tau_prom=11.75`, Spearman(B_GR, d3 burden)=0.46). The ready-to-use calibration artifact for the current C.1 pilot is:

```text
Results/Analysis/0006-stage-b-typed-targeting-mechanism-check/data/stageC1_global_pressure_calibration_typed_target_thresholded_tauprom11p75.json
```

The artifact is passed by CLI, never hardcoded in Python; a different operating point must recompute its own via the C.1 Calibration command. If `global_pressure.enabled=true` and `B_high <= B_low`, config loading must fail fast.

Actuator:

$$
\beta_{\mathrm{eff}}(t)
=
\beta
g_{\mathrm{GR}}(t)
$$

$$
\lambda_{\mathrm{eff}}(t)
=
\lambda
g_{\mathrm{GR}}(t)
$$

Do not change active-block threshold, `max_windows`, reliability gates, or unmask Bernoulli schedule in C.1.

#### C.1 File Structure

| File | Change |
|---|---|
| `inverse_folding/reference_flow/actionability.py` | Add pure helpers for smoothstep pressure mapping and trajectory-level `B_GR` aggregation. |
| `inverse_folding/reference_flow/controller_config.py` | Extend `GlobalPressureConfig`; remove the Stage-B-only fail-fast that rejects `enabled=true`; validate Stage C fields. |
| `inverse_folding/reference_flow/controller.py` | Maintain pressure state, compute `beta_eff/lambda_eff`, pass `beta_eff` into D2 scoring, pass `lambda_eff` into D3/rank scoring, log pressure fields. |
| `inverse_folding/reference_flow/counterfactual.py` | Allow `D2Handler.correct_logits(..., beta_override=...)` and use it for candidate weights / telemetry. |
| `inverse_folding/reference_flow/commit.py` | No new scoring formula; tests must verify existing `lambda_commit` argument receives the effective value. |
| `scripts/run_if_phase_c1.py` | Load optional `--global-pressure-calibration-json`, stamp values into the controller setup, print resolved pressure config, write manifest fields. |
| `scripts/submit_if_phase_c.slurm` | Add optional `GLOBAL_PRESSURE_CALIBRATION_JSON` env forwarded to `run_if_phase_c1.py`. |
| `inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml` | New C.1 primary config; copies `d2_d3_full_stageB_aopen.yaml` (typed_target / legacy D3 — G10) and enables thresholded global pressure. |
| `tests/inverse_folding/test_reference_flow_actionability.py` | Pure mapping / aggregation tests. |
| `tests/inverse_folding/test_reference_flow_controller_config.py` | Config validation and calibration JSON loading tests. |
| `tests/inverse_folding/test_reference_flow_d2_d3_controller.py` | Controller-level beta/lambda scaling and bit-equivalence tests. |
| `tests/scripts/test_run_if_phase_c1_d2_d3.py` | CLI/manifest tests for calibration JSON. |

#### C.1 Task List

> **Implementation status:** an earlier C1.x implementation exists for `pressure_source: trajectory_median_G` over the unified `typed_fresh` base. The RAR 0006 revision supersedes that route. Treat the C1.x tasks below as open until the code/config/tests pass with `pressure_source: trajectory_thresholded_G`, `tau_prom`, and the `d2_d3_full_stageB_aopen.yaml` typed-target / legacy-D3 base.

- [ ] **Task C1.1: Add pure pressure functions**

Required functions:

```python
def smoothstep_pressure(
    B_GR: float,
    *,
    B_low: float,
    B_high: float,
    g_min: float,
    g_max: float,
) -> float: ...

def protein_pressure_burden(G_values: list[float]) -> float: ...
```

Tests:

```python
def test_smoothstep_pressure_hits_zero_for_low_burden():
    assert smoothstep_pressure(0.1, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0) == 0.0

def test_smoothstep_pressure_hits_one_for_high_burden():
    assert smoothstep_pressure(0.9, B_low=0.2, B_high=0.8, g_min=0.0, g_max=1.0) == 1.0

def test_protein_pressure_burden_uses_median_not_latest_spike():
    assert protein_pressure_burden([0.02, 0.03, 1.20]) == 0.03
```

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_actionability.py -q
```

- [ ] **Task C1.2: Extend global-pressure config**

Add fields:

```python
pressure_source: str = "trajectory_thresholded_G"   # G3; median_t of (1/L)Σ ReLU(u_pressure - tau_prom)
tau_prom_source: str | None = None                  # "calibration_json" for the primary C.1 YAML
tau_prom: float | None = None                        # prominence cut, from calibration JSON
mapping: str = "smoothstep"
g_min: float = 0.0
g_max: float = 1.0
B_low: float | None = None
B_high: float | None = None
min_reliable_refreshes: int = 1
unready_g: float = 0.0
scale_beta: bool = True
scale_lambda: bool = True
```

Validation:

- `mapping` must be `"smoothstep"` in C.1.
- `pressure_source` must be `"trajectory_thresholded_G"` in C.1 (G3). The legacy `"trajectory_median_G"` is rejected for the C.1 outcome run.
- if `tau_prom_source` is present, it must be `"calibration_json"` in C.1.
- if `enabled=true`, `tau_prom`, `B_low`, and `B_high` must be present and `B_high > B_low`.
- if `enabled=true`, `targeting.mode` must be `typed_actionability`.
- if `enabled=true`, `d3.evidence_source` must be `legacy_window_excess` for the primary C.1 config (G10; `typed_fresh` is harmful per RAR 0006 M9 and deferred).
- remove the existing Stage-B-only error that rejects `global_pressure.enabled=true`; keep all existing Stage B configs disabled.

Run:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -q
```

- [ ] **Task C1.3: Load calibration JSON at run setup**

Add CLI flag to `scripts/run_if_phase_c1.py`:

```bash
--global-pressure-calibration-json PATH
```

Behavior:

- If controller config has `global_pressure.enabled=false`, ignore the flag except for manifest provenance.
- If `enabled=true`, the flag is required.
- Load `pressure_source`, `tau_prom`, `B_low`, and `B_high` from the JSON and materialize them into `ControllerConfig.global_pressure` via `dataclasses.replace`.
- Reject a mismatch between the YAML `global_pressure.pressure_source` and the JSON `pressure_source`.
- Print `global_pressure.pressure_source`, `global_pressure.tau_prom`, `global_pressure.B_low`, `global_pressure.B_high`, `global_pressure.g_min`, `global_pressure.g_max`, and `global_pressure.mapping` at startup.
- Manifest gains `global_pressure_calibration_path`, `global_pressure_calibration_hash`, `tau_prom`, `B_low`, `B_high`, and `pressure_source`.

Test:

```bash
pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -q
```

- [ ] **Task C1.4: Maintain trajectory-level pressure state**

Controller state:

```python
_pressure_G_values: list[float]
_pressure_B_GR: float | None
_pressure_g_GR: float
```

At each typed refresh:

1. append finite `actionability.G` to `_pressure_G_values`;
2. if fewer than `min_reliable_refreshes`, set `g_GR = unready_g`;
3. else compute `B_GR = median(_pressure_G_values)` and `g_GR = smoothstep_pressure(...)`;
4. store both in refresh addendum and actionability summary.

Telemetry fields:

```text
G_step
B_GR
g_GR_effective
beta_base
beta_eff
lambda_base
lambda_eff
pressure_burden_bin
```

Burden bin uses the same anchors:

```text
low:  B_GR <= B_low
mid:  B_low < B_GR < B_high
high: B_GR >= B_high
```

- [ ] **Task C1.5: Scale D2 β**

`D2Handler.correct_logits()` receives `beta_override: float | None = None`. Candidate weights use:

```python
beta_used = self.config.beta if beta_override is None else float(beta_override)
```

The D2 block/event telemetry must record:

```text
beta_base
beta_eff
g_GR_effective
```

Sticky rule: delta logits are computed once with `beta_eff` at refresh creation. Pending sticky corrections store the resulting delta logits; re-delivery within TTL does not recompute β.

Tests:

- with `g_GR=0.0`, D2 candidate weights become the `beta=0` posterior;
- with `g_GR=1.0`, outputs match Stage B for the same random seed;
- sticky delivery at `age > 0` reuses the original delta logits.

Compute optimization is optional: if cluster time later becomes limiting, the controller may short-circuit D2 candidate scoring when `g_GR_effective <= 1e-6` and emit explicit `skipped_reason="global_pressure_zero"`. Do not add this optimization in the first C.1 implementation unless profiling requires it; correctness and telemetry attribution are higher priority.

- [ ] **Task C1.6: Scale D3 λ**

Compute:

```python
lambda_eff = config.d3.lambda_commit * g_GR_effective
```

Use `lambda_eff` in both places:

1. `D3Handler.run_refresh(..., lambda_override=lambda_eff)` or equivalent;
2. `_build_stage_a_rank_scores()` for `d2_logits` / `d2_d3_full` rank scores.

Telemetry must record `lambda_base`, `lambda_eff`, and `g_GR_effective` in refresh log and D3/D2 event rows when available.

Tests:

- with `g_GR=0.0`, immune EMA penalty drops out of commit rank;
- with `g_GR=1.0`, rank scores match Stage B;
- `d3_revisit` without typed/global pressure remains legacy.

- [ ] **Task C1.7: Add C.1 config**

Create:

```text
inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml
```

It must copy **`d2_d3_full_stageB_aopen.yaml` (typed_target / `legacy_window_excess` D3 — see G10; NOT the unified arm)** and change only:

```yaml
controller:
  global_pressure:
    enabled: true
    pressure_source: trajectory_thresholded_G   # median over refreshes of (1/L)Σ ReLU(u_pressure - tau_prom); G3
    tau_prom_source: calibration_json           # prominence cut from typed pilot (C.1 Calibration)
    mapping: smoothstep
    g_min: 0.0
    g_max: 1.0
    min_reliable_refreshes: 1
    unready_g: 0.0
    scale_beta: true
    scale_lambda: true
```

Do not set C.2 schedule fields in this config.

> **Revision note (RAR 0006):** Tasks C1.1–C1.8 above were first implemented against the unified base with `pressure_source: trajectory_median_G`. Both are superseded: rebase the config on `d2_d3_full_stageB_aopen.yaml` and switch the driver/actuator to the thresholded `G_step` (G3). Re-run the C1.x bit-equivalence tests after the change.

- [ ] **Task C1.8: Register docs and launcher**

Update `doc/SCRIPTS.md`:

- `run_if_phase_c1.py` supports Stage C `--global-pressure-calibration-json`;
- `submit_if_phase_c.slurm` forwards `GLOBAL_PRESSURE_CALIBRATION_JSON`;
- Stage C config path is `d2_d3_full_stageC_pressure_aopen.yaml`.

No new SLURM script.

#### C.1 Calibration Artifact

The RAR 0006 calibration task has already produced the primary C.1 JSON:

```bash
CALIBRATION_JSON=/Users/jerry/Project/MHC-IF/Results/Analysis/0006-stage-b-typed-targeting-mechanism-check/data/stageC1_global_pressure_calibration_typed_target_thresholded_tauprom11p75.json
```

Required loader behavior:

- read `B_low`, `B_high`, `tau_prom`, and `pressure_source`;
- reject the file unless `pressure_source == "trajectory_thresholded_G"`;
- reject missing `tau_prom` when `global_pressure.enabled=true`;
- stamp the loaded values into `ControllerConfig.global_pressure` before hashing and manifest writing;
- preserve `calibration_source.burden_reference_usage = offline_calibration_only_not_runtime_guidance` in manifest provenance if practical.

Only regenerate this artifact for a different operating point. The reference regeneration command is:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

run = Path(os.environ["STAGE_B_RUN"])            # typed_target arm
out_path = run / "meta/global_pressure_calibration.json"
res = pd.read_parquet(run / "generation/actionability_residues.parquet",
                      columns=["protein_id", "design_idx", "seed", "refresh_step", "u_pressure"])
# Offline burden anchor for tau selection ONLY (calibration-time; never read at runtime, per G6).
burden_path = Path(os.environ["BURDEN_CSV"])
burden = pd.read_csv(burden_path)   # columns: protein_id, design_idx, global_risk_d3_t050

def per_design_median_G(tau):
    m = np.clip(res["u_pressure"].to_numpy() - tau, 0.0, None)
    g_step = pd.Series(m).groupby(
        [res["protein_id"], res["design_idx"], res["seed"], res["refresh_step"]]).mean()
    return g_step.groupby(level=[0, 1, 2]).median().rename("median_G").reset_index()

# Pick tau_prom maximizing Spearman(median_G, d3 burden) over a u_pressure-quantile grid.
best_tau, best_rho = None, 0.0
for tau in np.quantile(res["u_pressure"], np.linspace(0.30, 0.95, 14)):
    mg = per_design_median_G(tau).merge(burden, on=["protein_id", "design_idx"])
    rho = mg["median_G"].corr(mg["global_risk_d3_t050"], method="spearman")
    if rho is not None and rho > best_rho:        # higher burden -> higher median_G
        best_tau, best_rho = float(tau), float(rho)
if best_tau is None:
    raise SystemExit("no positive-correlation tau found; field/driver needs rework (G10)")
unit = per_design_median_G(best_tau)["median_G"].dropna()
B_low, B_high = float(unit.quantile(1/3)), float(unit.quantile(2/3))
if not B_high > B_low:
    raise SystemExit(f"invalid anchors: B_low={B_low}, B_high={B_high}")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "schema_version": "stageC1_global_pressure_calibration.v2",
    "pressure_source": "trajectory_thresholded_G",
    "tau_prom": best_tau,
    "tau_selection_spearman_vs_d3_burden": best_rho,
    "low_quantile": 1/3, "high_quantile": 2/3,
    "B_low": B_low, "B_high": B_high,
    "n_units": int(len(unit)),
    "source_artifact": str(run / "generation/actionability_residues.parquet"),
}, indent=2, sort_keys=True) + "\n")
print(out_path, "tau_prom=", best_tau, "rho=", round(best_rho, 3))
PY
```

`STAGE_B_RUN` is the returned run directory for the typed Aopen mechanism arm. The coder must not bake this path into Python code.

#### C.1 Experiment Arm

Run against the same 50-protein input parquet, same seed, same design count, same Aopen operating point. Use the normal `run_if_phase_c1.py` required inputs (`--checkpoint`, `--test-set-parquet`, `--pdb-root`, `--allele`, `--config`, `--output-root`; `--h-maps-parquet` is now optional and only required when `amplification.form` ∈ {`linear_clamp`, `sigmoid`, `power`}) or the existing `submit_if_phase_c.slurm` wrapper. The C-specific additions are:

```bash
python scripts/run_if_phase_c1.py [standard Phase C1 args] \
  --controller-config inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml \
  --global-pressure-calibration-json /Users/jerry/Project/MHC-IF/Results/Analysis/0006-stage-b-typed-targeting-mechanism-check/data/stageC1_global_pressure_calibration_typed_target_thresholded_tauprom11p75.json \
  --n-designs-per-protein 4 \
  --seed 42
```

The absolute calibration path above is allowed in experiment commands and SLURM environment variables. It must not be hardcoded inside Python modules.

Primary comparator:

```text
static Aopen arm from RAR 0005
```

Secondary comparator:

```text
Stage B typed Aopen mechanism arm
```

#### C.1 Acceptance

Use internal-head metrics first; NetMHCIIpan remains confirmation.

1. **Low-burden suppression:** using final per-design `B_GR` bins, the low bin median head delta must move from Aopen's `+2.080` regression toward no-steering. Pass if low-bin median `dhead <= +1.04` (at least 50% reduction versus Aopen); `dhead <= +0.5` is the stretch target.
2. **High-burden retention:** high-bin median head delta must remain close to Aopen's high-burden gain. Pass if high-bin median `dhead <= -2.0` and high-bin improved fraction remains `>= 0.60` (Aopen reference: median `-2.567`, improved fraction `0.687`).
3. **Aggregate movement toward GR ideal:** aggregate mean head delta must improve over Aopen `-0.244` and move toward the idealized `-1.211`. Pass if aggregate mean `dhead <= -0.8`. The `-1.211` anchor is approximate: RAR 0005's idealized counterfactual zeroed low-burden D2 relative to the matched D3 run, while C.1 with `scale_lambda=true` also attenuates D3 revisit on low-burden trajectories. If this difference appears to matter, use the `scale_lambda=false` sensitivity below.
4. **Structure:** median scTM must be no worse than matched D3-only by more than `0.01`, and the `immune_better_AND_struct_bad_lt_-0.15` count must not exceed the Aopen count.
5. **Attribution telemetry:** report all three tables:
   - burden-bin summary by final per-design `B_GR`;
   - high-burden D2 marginal versus D3-only;
   - typed-vs-static targeting contribution using the Stage B mechanism arm.

If low-bin suppression passes but high-bin retention fails, do not tune C.2. First test `scale_lambda=false` while keeping `scale_beta=true`; this isolates whether D3 revisit was over-suppressed by shared `g_GR`.

### Stage C.2: Phase C Editability Allocation

**Goal:** use `A_i(t)` projections to allocate unmask opportunity. C.2 is not part of the first C.1 experiment. It is only allowed after C.1 shows that trajectory-level pressure improves the low/high-burden tradeoff.

#### C.2 Preconditions

C.2 cannot start until two new fields exist:

```text
c_ready_i(t)
f_rank_i(t)
```

Definitions for v1:

```text
c_ready_i(t)
  = live sticky D2 correction at position i
    AND originating block was feasible
    AND best_delta_R_B < 0
    AND ESS_candidates_fraction passed its configured floor
    AND ensemble_sign_consistency is positive when available

f_rank_i(t)
  = committed-position rank percentile from the most-recent available
    Stage A/D3 rank face before the current Bernoulli draw
    with masked positions set to 0
```

`c_ready` is pre-sampling readiness. It must not use realized benefit from the future same step. Realized benefit remains telemetry and D2 evidence after sampling.
`f_rank` is also causal: C.2 schedule modulation runs before same-step sampling/remasking, so it must use the previous post-step rank snapshot, not this-step's rank scores.

Derived fields:

```text
u_hold_i = v_target_i * (1 - c_ready_i)
u_exec_i = c_ready_i
u_freeze_i = low b_cur, low b_env, low b_mem, high r_ctx, high f_rank
```

#### C.2 Scheduler Contract

C.2 modifies the positionwise Bernoulli probabilities before the sampler draws `rng.random(L) < probs`. It therefore does **not** preserve Theorem 1's positionwise ordering guarantee. The only invariant it preserves is the total per-step unmask budget over currently masked positions:

$$
\sum_i
\tilde{p}_i^{\mathrm{unmask}}(t)
=
\sum_i
p_i^{\mathrm{base}}(t)
$$

This is intentional and must be isolated from C.1.

Schedule form:

$$
\tilde{p}_i^{\mathrm{unmask}}(t)
\propto
p_i^{\mathrm{base}}(t)
\exp
\left(
-\alpha_{\mathrm{hold}} u_i^{\mathrm{hold}}(t)
\right)
\left(
1+\alpha_{\mathrm{exec}}u_i^{\mathrm{exec}}(t)
\right)
+
p_i^{\mathrm{catch}}(t)
$$

Normalize so:

$$
\sum_i
\tilde{p}_i^{\mathrm{unmask}}(t)
=
\sum_i
p_i^{\mathrm{base}}(t)
$$

Defaults for the first Stage C.2 run:

```yaml
controller:
  schedule_editability:
    enabled: true
    alpha_hold: 0.5
    alpha_exec: 1.0
    catchup_start_t: 0.80
    catchup_strength: 1.0
    normalize_budget: true
```

#### C.2 File/API Boundary

| File | Change |
|---|---|
| `inverse_folding/reference_flow/controller_config.py` | Add `ScheduleEditabilityConfig`, default disabled. |
| `inverse_folding/reference_flow/controller.py` | Add `adjust_unmask_probabilities(context)` hook consuming `u_hold/u_exec/u_freeze`. |
| `inverse_folding/reference_flow/sampler.py` | Call the hook after base `positionwise_unmask_probabilities(...)` and before Bernoulli draws. |
| `inverse_folding/reference_flow/schedule.py` | Add pure budget-normalized modulation helper with probability cap handling. |
| `tests/inverse_folding/test_reference_flow_schedule_editability.py` | Unit tests for modulation and cap redistribution. |

Tests:

- total unmask budget is preserved after modulation;
- probabilities stay in `[0, 1]` after cap redistribution;
- a live sticky D2 correction increases target-position unmask probability within TTL;
- a risky region without correction readiness is held early but not permanently starved because catch-up activates near the end;
- with `schedule_editability.enabled=false`, C.1 is bit-equivalent.

#### C.2 Acceptance

- `selected_after_d2_rate` increases relative to Stage B at the same sticky TTL.
- `final_persistence_rate` does not drop below the Stage B value.
- Structure metrics remain no worse than matched D3-only.
- Low-burden proteins show lower unnecessary D3 revisit/churn than Stage B.

Do not run C.2 until C.1 passes at least the low-burden suppression and structure gates. If C.1 fails, C.2 will confound pressure calibration with schedule changes.

---

## Self-Review Checklist

- [ ] The Stage B mechanism-check arm does not use Stage C pressure actuation.
- [ ] Stage B computes and logs `G(t)` / `g_GR(t)` but does not scale beta/lambda.
- [ ] `b_env` uses pre-D2 seed regions, not already-selected active blocks.
- [ ] `b_mem` updates only from `e_fresh`.
- [ ] `r_ctx` gates only positive `b_cur`, with a nonzero floor.
- [ ] Active-block discovery uses `v_target`, not `u_pressure`.
- [ ] Cluster support affects global pressure only.
- [ ] Existing `d3_revisit` comparator keeps legacy D3 input.
- [ ] The Stage B mechanism-check Aopen config keeps `legacy_window_excess` as D3 input.
- [ ] The full typed-wiring sanity config uses `typed_fresh` as D3 input.
- [ ] Existing configs default to legacy/static behavior.
- [ ] No new SLURM script is introduced.
- [ ] Stage C pressure scaling is separate from Stage C schedule editability.
- [ ] Stage C.1 uses `g_min = 0.0` or an explicit `0.05` sensitivity, never the Stage B diagnostic `0.25`.
- [ ] Stage C.1 actuation reads trajectory-level `B_GR`, not raw per-refresh `G_step`.
- [ ] Stage C.1 uses the thresholded `G_step = (1/L)Σ ReLU(u_pressure - tau_prom)` (G3, RAR 0006), **not** `mean(u_pressure)`; `tau_prom` comes from the calibration JSON; base config is `d2_d3_full_stageB_aopen.yaml` (typed_target), not the unified arm.
- [ ] Stage C.2 is not implemented or enabled before `c_ready_i` and `f_rank_i` exist.
- [ ] Stage C.2 documentation states that schedule modulation preserves total budget but not Theorem 1 positionwise ordering.
