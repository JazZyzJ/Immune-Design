# Reference Flow Unified Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage B/C unified Reference Flow controller: dynamic typed actionability targeting first, then gated global-risk pressure and Phase C editability control.

**Architecture:** Stage B replaces static active-block targeting with a typed residue actionability field `A_i(t)` while keeping Stage A actuation unchanged. Stage C consumes the same field to produce `g_GR(t)` pressure and schedule editability, but only after Stage B proves dynamic targeting beats the old static source.

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

**Ownership boundary**

| File | Owns |
|---|---|
| `PLAN_RF.md` | Phase C/D status, C0/C1/D1, and Stage A actuation contract |
| `PLAN_RF_UNI_CTRL.md` | Stage B typed targeting and Stage C GR x Phase C schedule controller |

**Out of scope**

- No model retraining.
- No NetMHC guidance signal.
- No WT/static position-specific runtime subtraction.
- No new SLURM script unless later execution discovers an unavoidable CLI gap. Use `scripts/run_if_phase_c1.py` and `scripts/submit_if_phase_c.slurm`.
- No Stage C actuation until Stage B gate passes.

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

Stage B logs `G(t)` and a diagnostic `g_GR(t)`, but does not use them to scale beta/lambda.

Stage C maps `G(t)` to a bounded pressure scalar:

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

Stage C calibration values are derived from the accepted Stage B 50-protein pilot:

```yaml
controller:
  global_pressure:
    g_min: 0.25
    g_max: 1.0
    G0_source: stageB_pilot_median
    s_G_source: stageB_pilot_iqr_half
```

---

## Stage B File Structure

**Create**

- `inverse_folding/reference_flow/actionability.py`  
  Pure functions and dataclasses for `A_i(t)`: projection, `tau_ref_B`, `b_cur`, `b_env`, `e_fresh`, `b_mem`, `v_target`, `u_pressure`, `G(t)`, `g_GR(t)`.

- `tests/inverse_folding/test_reference_flow_actionability.py`  
  Unit tests for every pure operator and firewall rule.

- `inverse_folding/reference_flow/configs/d2_d3_full_stageB.yaml`  
  Primary Stage B pilot config. Extends the Stage A full mode by setting typed targeting and typed D3 fresh-evidence input.

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
| new `d2_d3_full_stageB` | `typed_actionability` | `typed_fresh` | no |
| Stage C full | `typed_actionability` | `typed_fresh` | yes |

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

- Create: `inverse_folding/reference_flow/configs/d2_d3_full_stageB.yaml`
- Create: `inverse_folding/reference_flow/configs/d_monitor_stageB.yaml`
- Modify: `scripts/run_if_phase_c1.py`
- Test: `tests/scripts/test_run_if_phase_c1_d2_d3.py`

- [ ] **Step B6.1: Add YAML configs**

`d2_d3_full_stageB.yaml` must set:

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
python scripts/run_if_phase_c1.py \
  --config inverse_folding/reference_flow/configs/d_monitor_stageB.yaml \
  --limit-proteins 2 \
  --n-designs-per-protein 1 \
  --seed 42 \
  --output-root <run-output-root>
```

Expected artifacts:

```text
generated.parquet
refresh_log.jsonl
actionability_residues.parquet
actionability_refresh_summary.jsonl
manifest.json
```

50-protein Stage B gate:

1. Stage A local gates remain satisfied with typed targeting.
2. Dynamic `A_i(t)` targeting beats static `z_dyn - z_static` on the same protein/seed set by active-block usefulness. Primary criterion: higher `realized_benefit_rate` among D2 disagreements. Supporting evidence (report alongside; do not substitute for the primary): higher fraction of D2 events landing inside eventually resolved immune hotspots; lower fraction of active blocks with no measurable head-risk benefit after paired counterfactual scoring. ("Resolved" must be pinned in the analysis script: a window whose risk drops below `tau_ref_B` by the final sequence.)
3. Full pipeline passes end-to-end falsification:
   - D2 adds at least 5 percentage points Pareto rate over matched D3-only.
4. Focal diagnostic passes:
   - at least one refresh shows a residue with high `v_target` but reduced `u_pressure`, proving focal hotspots can be targeted without inflating global pressure.
5. Memory firewall passes:
   - when `e_fresh` falls to zero across consecutive refreshes, `b_mem` decays by the configured half-life and cannot grow.

If Stage B fails condition 2 while Stage A gates remain good, stop at Stage A. Static targeting was not the bottleneck under the tested setup. If Stage B passes condition 2 but fails condition 3, D2 directional information is not adding enough global value; use the revisit-only fallback discussion in `doc/RF_Controller_Architecture.md` §6.D before adding Stage C complexity.

---

## Stage C Plan

Stage C is entered only after Stage B acceptance. It has two sub-stages to keep attribution readable.

### Stage C.1: Global Pressure Scaling

**Goal:** Use `g_GR(t)` only to scale global immune pressure.

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

**Files**

- Modify: `inverse_folding/reference_flow/controller.py`
- Modify: `inverse_folding/reference_flow/counterfactual.py`
- Modify: `inverse_folding/reference_flow/commit.py`
- Modify: `inverse_folding/reference_flow/configs/d2_d3_full_stageB.yaml` or create `d2_d3_full_stageC_pressure.yaml`
- Test: `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`

**Tests**

- `global_pressure.enabled = false` is bit-equivalent to Stage B.
- `global_pressure.enabled = true` scales only beta/lambda.
- Low `G(t)` produces weaker D2 posterior tilt and weaker D3 revisit pressure.
- High `G(t)` recovers baseline beta/lambda within configured `g_max`.

**Acceptance**

- Low-burden over-intervention decreases relative to Stage B without losing the Stage B immune benefit on high-burden proteins.
- `g_GR(t)` distribution is reported by protein and by refresh.

### Stage C.2: Phase C Editability Allocation

**Goal:** Use `A_i(t)` projections to allocate unmask opportunity without changing total denoising budget.

Derived fields:

```text
u_hold_i = v_target_i * (1 - c_ready_i)
u_exec_i = c_ready_i
u_freeze_i = low b_cur, low b_env, low b_mem, high r_ctx, high f_rank
```

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
schedule_editability:
  enabled: true
  alpha_hold: 0.5
  alpha_exec: 1.0
  catchup_start_t: 0.80
  catchup_strength: 1.0
  normalize_budget: true
```

**Tests**

- Total unmask budget is preserved after schedule modulation.
- A live sticky D2 correction increases target-position unmask probability within TTL.
- A risky region without correction readiness is held but not permanently starved because catch-up activates near the end.
- With `schedule_editability.enabled = false`, Stage C.1 remains bit-equivalent.

**Acceptance**

- `selected_after_d2_rate` increases relative to Stage B at the same sticky TTL.
- `final_persistence_rate` does not drop below the Stage B value.
- Structure metrics remain no worse than matched D3-only.
- Low-burden proteins show lower unnecessary D3 revisit/churn than Stage B.

---

## Self-Review Checklist

- [ ] Stage B does not depend on Stage C.
- [ ] Stage B computes and logs `G(t)` / `g_GR(t)` but does not scale beta/lambda.
- [ ] `b_env` uses pre-D2 seed regions, not already-selected active blocks.
- [ ] `b_mem` updates only from `e_fresh`.
- [ ] `r_ctx` gates only positive `b_cur`, with a nonzero floor.
- [ ] Active-block discovery uses `v_target`, not `u_pressure`.
- [ ] Cluster support affects global pressure only.
- [ ] Existing `d3_revisit` comparator keeps legacy D3 input.
- [ ] `d2_d3_full_stageB` uses `typed_fresh` as D3 input.
- [ ] Existing configs default to legacy/static behavior.
- [ ] No new SLURM script is introduced.
- [ ] Stage C pressure scaling is separate from Stage C schedule editability.
