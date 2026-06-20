# Self-Conditioned GR Monitor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the monitor-only Self-Conditioned Proposal-Envelope GR probe from `doc/Self-Cond_GR.md`.

**Architecture:** SC-GR is a sensing layer. It runs pre-D2 at refresh steps, builds fresh and self-conditioned pseudo-terminal completions from structural logits, scores them with the frozen head, and writes burden-estimator telemetry. It must not change logits, D2/D3 state, remask ranking, `global_pressure`, or the Phase C schedule.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pandas/pyarrow, existing `inverse_folding/reference_flow/*`, existing `scripts/run_if_phase_c1.py`, existing `scripts/submit_if_phase_c.slurm`.

---

## 0. Scope And Continuity

**Design source:** `doc/Self-Cond_GR.md`.

**Previous plan boundary:** `PLAN_RF_UNI_CTRL.md` remains the source for Stage B typed targeting, Stage B.1 within-block `v_target`, the implemented thresholded-pressure C.1 baseline, and deferred Phase C schedule work.

**This plan owns:** the SC-GR monitor-only burden-estimator branch. If the monitor passes, a later plan may connect a selected `B_sc` estimator to the existing `global_pressure` actuator.

**Discard / freeze for this stage:**

- Do not extend `trajectory_thresholded_G` as the main GR estimator.
- Do not enable `global_pressure.enabled`.
- Do not scale `beta` or `lambda`.
- Do not implement `g_GR > 1`.
- Do not feed probe evidence into D3 `m_i`, typed `b_mem`, active-block discovery, or schedule editability.

**Carry over:**

- Stage B.1 Aopen controller is the base operating point.
- `within_block_source=v_target` stays enabled in the monitor preset.
- `d3.evidence_source=legacy_window_excess` stays unchanged.
- Existing run and SLURM entrypoints are reused.

---

## 1. Monitor Contract

At each eligible refresh, SC-GR runs before D2 and produces telemetry only.

Allowed inputs:

```text
x_t
structural_logits
mask_token_id
canonical_token_ids
completed argmax tokens
frozen OnlineHeadScorer
protein_id / design_idx / seed / refresh_step
```

Forbidden inputs:

```text
active_blocks
D2 sticky cache
D2 candidates
corrected logits
D3 m_i / commit state
typed b_mem
g_GR_effective
```

Probe arms:

```text
fresh:
  committed positions fixed
  masked positions sampled from structural softmax over canonical AA tokens

self_conditioned:
  committed positions fixed
  if prev_confidence[i] >= confidence_threshold, masked position i reuses prev_x1_hat[i]
  otherwise sampled from structural softmax over canonical AA tokens
```

State update:

```text
prev_x1_hat, prev_confidence
  <- current structural argmax terminal estimate and canonical-softmax confidence
```

The update must not use head score, active blocks, D2 output, corrected logits, or downstream samples.

Important experimental caveat: current 10-step Aopen runs usually expose only one post-start refresh (`t_start=0.50`, `refresh_interval=5`), so they can test fresh ensemble vs argmax but cannot test real cross-refresh self-conditioning. The monitor experiment must include a multi-refresh setting, e.g. `N_STEPS=20`, before judging the self-conditioned arm.

---

## 2. Data Products

When enabled, SC-GR writes two sidecars.

### `sc_gr_probe_samples.parquet`

One row per completion sample.

Required columns:

```text
protein_id, design_idx, seed, refresh_step, step, t
arm, sample_idx, sequence_md5
num_masked, num_reused_from_prev, reuse_fraction
mean_prev_confidence_reused, mean_sample_entropy, state_bootstrap_flag
G_mean_excess, G_topm_lse, G_supra_mass_tau_11p75
head_risk_LME, head_risk_max
```

### `sc_gr_probe_refresh.parquet`

One row per refresh and arm.

Required columns:

```text
protein_id, design_idx, seed, refresh_step, step, t
arm, ensemble_size_effective
B_sc_mean_excess_median
B_sc_topm_lse_median
B_sc_supra_mass_tau_11p75_median
G_mean_excess_max, G_mean_excess_std
G_topm_lse_max, G_topm_lse_std
G_supra_mass_tau_11p75_max, G_supra_mass_tau_11p75_std
num_masked, reuse_fraction_mean, state_bootstrap_flag
old_argmax_G_mean_excess
old_argmax_G_topm_lse
old_argmax_G_supra_mass_tau_11p75
```

Manifest additions:

```text
self_conditioned_gr_config
sc_gr_probe_samples_path
sc_gr_probe_refresh_path
```

Do not merge SC-GR telemetry into `controller_events.parquet`.

---

## 3. Burden Aggregators

For each pseudo-terminal completion:

```text
window z -> max-covering residue projection
residue_excess_i = max(0, projected_z_i - tau_ref_B)
```

`tau_ref_B` reuses the Stage B scalar background convention: median static window `z` under `targeting.tau_ref_quantile`. This is only a scalar normalization anchor, not a position-specific static prior.

**Scope is global, not seed-local (critical).** Project over the **full sequence**: pass *every* window the frozen head returns for the completion into `max_covering_window_projection`. Do **not** pre-filter to seed / active / overlap windows — that local restriction is the typed `b_env` path (`_compute_envelope_burden`, `controller.py:1889`), which forces every non-seed residue to `0` and would miss burden spread thinly across many positions (exactly the distributed-burden failure SC-GR exists to catch). The correct template is the full-sequence `b_cur` projection (`controller.py:1698-1704`): all windows → `max_covering_window_projection` → per-residue `z`. SC-GR differs from `b_cur` only in (a) running over the `K`-completion ensemble instead of one argmax completion and (b) reducing with the amplify-aligned aggregators above.

Monitor these candidate trajectory burden aggregators:

```text
G_mean_excess = mean_i(residue_excess_i)

G_topm_lse = logmeanexp(top_m(residue_excess), temperature=lse_temperature)

G_supra_mass_tau_11p75 = mean_i(max(0, residue_excess_i - 11.75))
```

`11.75` is a YAML/config value (`supra_tau_values: [11.75]`), never hardcoded in Python.

SC-0 does not select a behavior aggregator. The analysis script chooses the best monitored candidate by high-burden recall first, Spearman second.

---

## 4. Implementation Tasks

### Task SC0.1: Config And Preset

**Files:**

- Modify: `inverse_folding/reference_flow/controller_config.py`
- Create: `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor.yaml`
- Test: `tests/inverse_folding/test_reference_flow_controller_config.py`

- [ ] Add frozen `SelfConditionedGRConfig` with fields:

```python
enabled: bool = False
mode: str = "monitor_only"
arms: tuple[str, ...] = ("fresh", "self_conditioned")
ensemble_size: int = 3
struct_temperature: float = 1.0
confidence_source: str = "canonical_softmax_max"
confidence_threshold: float = 0.70
update_state_from: str = "structural_argmax"
probe_refresh_policy: str = "all_typed_refreshes"
top_m: int = 16
lse_temperature: float = 1.0
supra_tau_values: tuple[float, ...] = (11.75,)
write_probe_telemetry: bool = True
```

- [ ] Add `self_conditioned_gr` to `ControllerConfig` and `controller_config_to_dict`.
- [ ] Validate:

```text
mode == "monitor_only"
arms subset of {"fresh", "self_conditioned"}
ensemble_size >= 1
struct_temperature > 0
confidence_source == "canonical_softmax_max"
0 <= confidence_threshold <= 1
update_state_from == "structural_argmax"
probe_refresh_policy == "all_typed_refreshes"
top_m >= 1
lse_temperature > 0
supra_tau_values finite and >= 0
if mode == "monitor_only" and self_conditioned_gr.enabled, global_pressure.enabled must be false
  (SC1 supersedes this: mode=="beta_pressure" instead REQUIRES global_pressure.enabled==true -- Task SC1.1)
```

- [ ] Create monitor preset by copying `d2_d3_full_stageB_aopen.yaml`, keeping the Stage B.1 Aopen operating point, and adding:

```yaml
controller:
  global_pressure:
    enabled: false
  self_conditioned_gr:
    enabled: true
    mode: monitor_only
    arms: [fresh, self_conditioned]
    ensemble_size: 3
    struct_temperature: 1.0
    confidence_source: canonical_softmax_max
    confidence_threshold: 0.70
    update_state_from: structural_argmax
    probe_refresh_policy: all_typed_refreshes
    top_m: 16
    lse_temperature: 1.0
    supra_tau_values: [11.75]
    write_probe_telemetry: true
```

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -q
```

Acceptance:

```text
config loads with SC-GR enabled
global_pressure.enabled remains false
invalid arms / invalid ensemble size / SC-GR + global pressure fail fast
```

### Task SC0.2: Pure Probe Helpers

**Files:**

- Create: `inverse_folding/reference_flow/self_conditioned_gr.py`
- Test: `tests/inverse_folding/test_reference_flow_self_conditioned_gr.py`

- [ ] Add dataclasses:

```python
SCGRState
SCGRProbeSample
SCGRRiskAggregates
```

- [ ] Implement deterministic helpers:

```python
build_probe_samples(...)
update_state_from_structural_argmax(...)
compute_risk_aggregates(...)
summarize_probe_refresh(...)
```

- [ ] Requirements:

```text
use a local np.random.default_rng seeded from protein/design/seed/refresh/arm/sample
never use sampler RNG
sample only canonical amino acid tokens
fresh arm samples every masked position
self_conditioned arm reuses previous token only above confidence threshold
no previous state -> self_conditioned falls back to fresh and marks state_bootstrap_flag
state update uses only structural argmax and canonical-softmax confidence
confidence is canonical max-softmax (renormalized over canonical AA tokens), DELIBERATELY not full-vocab
  entropy: it matches the canonical-AA sampling distribution and avoids special/non-canonical-token bias
aggregators are length-normalized
empty arrays return 0.0
```

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_self_conditioned_gr.py -q
```

Acceptance:

```text
committed positions stay fixed
low-confidence previous tokens are not reused
state update is independent of head scores
top-m LSE is more tail-sensitive than mean on a focal spike
```

### Task SC0.3: Controller Integration

**Files:**

- Modify: `inverse_folding/reference_flow/controller.py`
- Test: `tests/inverse_folding/test_reference_flow_d1_controller.py`

- [ ] Add controller state:

```python
_scgr_state
_scgr_sample_rows
_scgr_refresh_rows
```

- [ ] `_scgr_state` (prev_x1_hat / prev_confidence) is read **only** by the self_conditioned arm; the fresh arm never reads or writes it. The state is the deterministic structural-argmax observation, **never** sampled from either arm's completions — so the Fresh/SC contrast stays uncontaminated.
- [ ] Add public accessors:

```python
self_conditioned_gr_sample_rows()
self_conditioned_gr_refresh_rows()
```

- [ ] Run SC-GR after `dyn_score` / `static_score` are available and before active-window selection / D2 correction.
- [ ] Score all probe completions in one `score_batch_same_protein` call per refresh. **Project full-sequence:** feed *all* returned windows into `max_covering_window_projection` (the `b_cur` scope, `controller.py:1698-1704`) — never the seed-union `b_env` scope (`_compute_envelope_burden`, `controller.py:1889`); see §3.
- [ ] Build rows using the telemetry contract in §2.
- [ ] Update SC-GR state after the refresh probe.
- [ ] Do not mutate:

```text
context.logits
context.x_t
completed_tokens
window_excess
self._b_mem_prev
self._pressure_G_values
self._pending_d2_corrections
```

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_d1_controller.py -q
```

Acceptance:

```text
SC-GR enabled vs disabled returns identical logits for the same synthetic controller state
active-block counts are unchanged
B_GR/g_GR_effective remain None when global_pressure.enabled=false
fresh and self_conditioned rows are collected
```

### Task SC0.4: Driver Telemetry And Manifest

**Files:**

- Modify: `scripts/run_if_phase_c1.py`
- Test: `tests/scripts/test_run_if_phase_c1_d2_d3.py`

- [ ] Add constants:

```python
SC_GR_PROBE_SAMPLES_FILE = "sc_gr_probe_samples.parquet"
SC_GR_PROBE_REFRESH_FILE = "sc_gr_probe_refresh.parquet"
```

- [ ] Add `write_sc_gr_probe_artifacts(run_dir, sample_rows, refresh_rows)`.
- [ ] Collect rows from each controller instance after each design.
- [ ] Write sidecars when SC-GR telemetry is enabled and rows exist.
- [ ] Stamp manifest with `self_conditioned_gr_config` and sidecar paths.
- [ ] Print resolved `self_conditioned_gr` config at startup, one key per line.
- [ ] Required tests:

```bash
pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -q
```

Acceptance:

```text
SC-GR run writes both sidecars and manifest paths
SC-GR disabled run writes no SC-GR sidecars
empty row lists still produce schema-compatible parquet when writer is called
```

### Task SC0.5: Monitor Analysis CLI

**Files:**

- Create: `scripts/analysis/sc_gr_monitor_probe.py`
- Modify: `doc/SCRIPTS.md`
- Test: `tests/scripts/test_sc_gr_monitor_probe.py`

- [ ] CLI:

```bash
python scripts/analysis/sc_gr_monitor_probe.py \
  --run-dir /path/to/scgr/run \
  --oracle-parquet /path/to/oracle_imm_head.parquet \
  --oracle-risk-column global_risk \
  --oracle-aggregate median \
  --output /path/to/sc_gr_monitor_summary.json
```

- [ ] Analysis behavior:

```text
read sc_gr_probe_refresh.parquet
aggregate oracle burden per protein by median unless already per-protein
compute Spearman by arm / refresh_step / metric
bin oracle burden into terciles
bin probe burden into terciles per arm / refresh_step / metric
report crosswalk diagonal
report P(probe low | oracle high)
report Recall@High = P(probe high | oracle high)
choose best metric by Recall@High, tie-break by Spearman
```

- [ ] Register script in `doc/SCRIPTS.md` under "Analysis (Optional)".
- [ ] Required tests:

```bash
pytest tests/scripts/test_sc_gr_monitor_probe.py -q
```

Acceptance:

```text
synthetic monotone probe gets positive Spearman
true-high-as-low is computed correctly
best metric selection prefers Recall@High over mean correlation
script registration exists
```

---

## 5. Experiment Handoff

### Unit Gate

Run:

```bash
pytest \
  tests/inverse_folding/test_reference_flow_self_conditioned_gr.py \
  tests/inverse_folding/test_reference_flow_controller_config.py \
  tests/inverse_folding/test_reference_flow_d1_controller.py \
  tests/scripts/test_run_if_phase_c1_d2_d3.py \
  tests/scripts/test_sc_gr_monitor_probe.py \
  -q
```

Expected: PASS.

### Cluster Monitor Run

Use existing SLURM:

```bash
MODE=reference_flow \
CONTROLLER_CONFIG=/home/zc1519/src/Immune-Design/inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor.yaml \
N_STEPS=20 \
N_DESIGNS_PER_PROTEIN=4 \
SEED=42 \
GLOBAL_PRESSURE_CALIBRATION_JSON="" \
sbatch scripts/submit_if_phase_c.slurm
```

Expected run properties:

```text
self_conditioned_gr.enabled=true
global_pressure.enabled=false
SC-GR sidecars written
generated sequences match the same controller config with SC-GR disabled, up to deterministic head-call side effects only
```

### Monitor Analysis

After immune eval:

```bash
python scripts/analysis/sc_gr_monitor_probe.py \
  --run-dir /path/to/scgr_monitor_run/generation \
  --oracle-parquet /path/to/nod_or_d3_oracle/imm_head.parquet \
  --oracle-risk-column global_risk \
  --oracle-aggregate median \
  --output /path/to/scgr_monitor_run/analysis/sc_gr_monitor_summary.json
```

Pass criteria:

```text
best Spearman(B_sc, oracle burden) >= 0.60
Recall@High improves over old C.1 B_GR
P(probe low | oracle high) drops versus old C.1 B_GR
self_conditioned arm beats or matches fresh arm at a multi-refresh step
```

Failure interpretation:

```text
fresh good, self_conditioned bad -> recycling is too conservative; keep Fresh-K as estimator candidate
both weak -> probe time or aggregator is wrong; inspect refresh-index curves
Spearman okay, Recall@High weak -> tail-sensitive aggregator or instability trigger is needed
```

---

## 6. Post-Monitor Behavior Stages

These stages connect a **monitor-validated** `B_sc` estimator to the **existing** `global_pressure` actuator. The actuator plumbing is reused unchanged — `smoothstep_pressure → g_GR → beta_eff` (`controller.py:1567-1606`), `_effective_beta_override` (`controller.py:1606`). Only the **burden source** feeding `g_GR` changes; D2/D3 token actuation is untouched. Each stage is **gated** on the prior one.

**Entry gate (SC0 monitor → SC1):**

```text
best Spearman(B_sc, oracle burden) >= 0.60        (from sc_gr_monitor_summary.json)
Recall@High beats old C.1 B_GR; P(probe low | oracle high) drops
best_metric pinned in {mean_excess, topm_lse, supra_mass}
actuation_arm decided in {fresh, self_conditioned}
```

If the gate fails, **stop**: SC-GR is not a usable estimator, these stages do not run, and the legacy `trajectory_thresholded_G` configs stay the only GR baseline.

---

### 6.1 Stage SC1 — Beta-Only SC Pressure

Goal: gate D2 `beta` **only** by `g_GR = smoothstep(B_sc)`, where `B_sc` is the early-frozen per-design probe burden. `scale_lambda=false` (locked, RAR 0008). `g in [g_min, 1]` (protect-low); amplify-high (`g > 1`) is SC2.

#### Task SC1.1: Pressure-source dispatch seam + actuation config

**Files:**

- Modify: `inverse_folding/reference_flow/controller_config.py`
- Test: `tests/inverse_folding/test_reference_flow_controller_config.py`

- [ ] Add `"self_conditioned_probe"` to `_ALLOWED_PRESSURE_SOURCES` (`controller_config.py:35-38`).
- [ ] Relax the **two existing hard gates** that currently assume `trajectory_thresholded_G` (otherwise an enabled SC pressure run is rejected before it starts):
  - `_materialize_global_pressure` (`controller_config.py:887-900`): when `enabled`, allow `pressure_source in {trajectory_thresholded_G, self_conditioned_probe}` instead of forcing `== _REQUIRED_C1_PRESSURE_SOURCE`.
  - `validate_global_pressure_runtime` (`controller_config.py:959-985`): require `tau_prom` **only** for `trajectory_thresholded_G`; require `B_low`/`B_high` for **both** sources.
- [ ] This task supersedes SC0.1's `mode == "monitor_only"` enum restriction (mode becomes the two-value enum below) and SC0.1's `enabled → global_pressure.enabled=false` rule (now mode-conditional).
- [ ] Extend `SelfConditionedGRConfig` (from Task SC0.1) with actuation fields:

```python
mode: str = "monitor_only"              # monitor_only | beta_pressure
actuation_aggregator: str = "topm_lse"  # mean_excess | topm_lse | supra_mass (= SC0.5 best_metric)
actuation_arm: str = "fresh"            # fresh | self_conditioned (= SC0.5 recycling decision)
freeze_after_reliable_refreshes: int = 2
actuation_reduce: str = "median"        # median | ema  (over the frozen window)
```

- [ ] Extend the SC0.1 validator:

```text
mode in {"monitor_only", "beta_pressure"}
if mode == "beta_pressure":
    global_pressure.enabled == true
    global_pressure.pressure_source == "self_conditioned_probe"
    global_pressure.scale_lambda == false
    global_pressure.g_max == 1.0            # amplify (g>1) is SC2, not here
actuation_aggregator in {"mean_excess", "topm_lse", "supra_mass"}
  (names map to the SC0.2 telemetry stems G_mean_excess / G_topm_lse / G_supra_mass_tau_*;
   if "supra_mass", supra_tau_values must have exactly one element so the tau is unambiguous)
actuation_arm in {"fresh", "self_conditioned"}
freeze_after_reliable_refreshes >= 1
actuation_reduce in {"median", "ema"}
```

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -q
```

Acceptance:

```text
pressure_source=self_conditioned_probe + mode=beta_pressure loads
mode=beta_pressure with scale_lambda=true fails fast
mode=beta_pressure with g_max != 1.0 fails fast
mode=beta_pressure with global_pressure.enabled=false fails fast
```

#### Task SC1.2: Early-frozen per-design `B_sc` in the actuator

**Files:**

- Modify: `inverse_folding/reference_flow/controller.py` (`_update_pressure_state` ~1540-1606; SC-GR integration point from SC0.3)
- Test: `tests/inverse_folding/test_reference_flow_d1_controller.py`

- [ ] Add controller state `_scgr_B_sc_window: list[float]` and `_scgr_frozen_B_sc: float | None = None`.
- [ ] **Ordering:** the SC-GR probe (SC0.3) must run **before** `_update_pressure_state` (`controller.py:561`) so the current refresh's `B_sc` is available for actuation. (Monitor mode: order irrelevant. `beta_pressure` mode: required.)
- [ ] Each **reliable** refresh (`_pressure_refresh_reliable`, `controller.py:2233`), append the probe `B_sc` for `(actuation_arm, actuation_aggregator)` to `_scgr_B_sc_window`. When `len(_scgr_B_sc_window) >= freeze_after_reliable_refreshes` and `_scgr_frozen_B_sc is None`, set `_scgr_frozen_B_sc = median|ema(_scgr_B_sc_window)` per `actuation_reduce`, and never update it again.
- [ ] In `_update_pressure_state`, branch on `gp.pressure_source`:
  - `"self_conditioned_probe"`: `B = _scgr_frozen_B_sc`; if still `None`, keep `g_GR = unready_g` (today's path, `controller.py:1559-1564`). Otherwise feed `B` into the existing `smoothstep_pressure` call (`controller.py:1567-1573`) unchanged.
  - else: the legacy `trajectory_thresholded_G` path (`prominence_thresholded_mass` + median) is untouched.
- [ ] `scale_lambda=false` already routes `_effective_lambda_commit → None` (`controller.py:1615`); assert lambda is never scaled.

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_d1_controller.py -q
```

Acceptance:

```text
before freeze (< N reliable refreshes): g_GR == unready_g
after freeze: _scgr_frozen_B_sc is constant for the rest of the design; telemetry B_sc(t) may drift
lambda_commit override is None for all steps (scale_lambda=false)
a pressure_source=trajectory_thresholded_G run is byte-identical to pre-SC1 behavior
```

#### Task SC1.3: `B_sc` calibration JSON + loader

**Files:**

- Modify: `scripts/run_if_phase_c1.py` (`_load_global_pressure_calibration` ~157-253)
- Modify: `scripts/analysis/sc_gr_monitor_probe.py` (emit calibration)
- Test: `tests/scripts/test_run_if_phase_c1_d2_d3.py`

- [ ] Extend the loader: when `payload["pressure_source"] == "self_conditioned_probe"`, require keys `aggregator`, `arm`, `B_low`, `B_high` (`B_high > B_low`); `tau_prom` is not required for this source. Stamp `B_low`/`B_high` via the existing `dataclasses.replace` path (`run_if_phase_c1.py:244-253`).
- [ ] **Option A (YAML authoritative).** `aggregator` / `arm` live in `SelfConditionedGRConfig.actuation_aggregator` / `actuation_arm` (Task SC1.1), **not** in `GlobalPressureConfig`. The YAML must already set them; the loader only **validates consistency** — reject if JSON `aggregator`/`arm` != `self_conditioned_gr.actuation_aggregator`/`actuation_arm`, and reject `pressure_source` mismatch (mirror `run_if_phase_c1.py:236-243`). The loader stamps **only** `B_low`/`B_high` into `global_pressure`; it never mutates `self_conditioned_gr` (one config block touched, not two).
- [ ] `sc_gr_monitor_probe.py --emit-calibration PATH` writes `{schema_version, pressure_source:"self_conditioned_probe", aggregator:<best_metric>, arm:<actuation_arm>, B_low:q_low(B_sc), B_high:q_high(B_sc)}` from the monitor `B_sc` distribution at the selected refresh horizon (`--low-quantile`/`--high-quantile`, default 0.25/0.75).

- [ ] Required tests:

```bash
pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -q
```

Acceptance:

```text
SC calibration JSON loads and stamps B_low/B_high
missing aggregator/arm key fails fast
B_high <= B_low fails fast
YAML aggregator != JSON aggregator fails fast
```

#### Task SC1.4: Behavior config + run/analysis

**Files:**

- Create: `inverse_folding/reference_flow/configs/d2_d3_full_stageC_scgr_betaonly_aopen.yaml`
- Test: `tests/inverse_folding/test_reference_flow_controller_config.py`

- [ ] Copy the SC0 monitor preset (`d2_d3_full_stageB_aopen_scgr_monitor.yaml`); change only:

```yaml
controller:
  global_pressure:
    enabled: true
    pressure_source: self_conditioned_probe
    mapping: smoothstep
    g_min: 0.0
    g_max: 1.0
    scale_beta: true
    scale_lambda: false
    min_reliable_refreshes: 1
    unready_g: 1.0          # hold BASE beta until B_sc is frozen (do NOT zero beta pre-freeze)
  self_conditioned_gr:
    enabled: true
    mode: beta_pressure
    actuation_aggregator: <sc_gr_monitor_summary.json best_metric>
    actuation_arm: <sc_gr_monitor_summary.json recycling decision>
    freeze_after_reliable_refreshes: 2
    actuation_reduce: median
```

- [ ] Print resolved `self_conditioned_gr` + `global_pressure` at startup, one key per line.
- [ ] Run on the 50-protein pilot (SLURM reuse, `N_STEPS=20`) with `--global-pressure-calibration-json` from SC1.3.
- [ ] Analyze vs B1 (the Stage B.1 typed arm), binning by **NoD** burden terciles.

Pass criteria (`doc/Self-Cond_GR.md` §6 Stage 2):

```text
true-low / NoD-low over-intervention drops vs B1
true-high / NoD-high gain preserved vs B1
structure (scTM / recovery) no worse than B1
pressure-bin vs NoD-burden-bin diagonal markedly above old C.1
```

**Note on `unready_g`:** the monitor preset uses `unready_g: 0.0` (telemetry, never actuates). SC1 sets `unready_g: 1.0` so beta stays at **base** (B1 behavior) before the early-freeze completes — `0.0` would run the first 1-2 refreshes fully *un*steered, harming high-burden proteins exactly when steering still matters. One deliberate config change, consistent with "one config per experiment."

---

### 6.2 Stage SC2 — Amplify-High (`g > 1`)

> **Not a current coder handoff.** Deferred until SC1 passes its readouts; kept here as the planned next lever, not work to dispatch now.

**Entry gate (SC1 → SC2):** SC1 passes its four readouts **and** true-high gain is preserved (not eroded). Then test the larger lever: push high-NoD-burden proteins harder.

Mechanism: the existing `smoothstep_pressure(B; B_low, B_high, g_min, g_max)` (`actionability.py:320-329`) is **already two-sided** — with `g_min < 1 < g_max` it maps low burden → `g_min` (suppress) and high burden → `g_max` (amplify) on one monotone curve. So SC2 is mostly a **calibration + bounds** change, not a new mapping function.

#### Task SC2.1: Allow `g_max > 1` (two-sided gain)

**Files:**

- Modify: `inverse_folding/reference_flow/controller_config.py` (global_pressure validation)
- Test: `tests/inverse_folding/test_reference_flow_controller_config.py`

- [ ] Add `global_pressure.amplify: bool = false`. If a `g_max <= 1` cap exists in validation, gate it on `amplify`: when `amplify=false` keep `g_max == 1.0` enforced; when `amplify=true` require `g_min < 1.0 < g_max <= 3.0` and the existing `0 <= unready_g <= g_max` (`controller_config.py:935`).
- [ ] `self_conditioned_gr.mode == "beta_pressure"` may use `g_max > 1` **only** when `global_pressure.amplify=true`, so SC1 (g_max=1) and SC2 (g_max>1) configs are distinct and a coder cannot silently amplify.

- [ ] Required tests:

```bash
pytest tests/inverse_folding/test_reference_flow_controller_config.py -q
```

Acceptance:

```text
amplify=false keeps g_max==1.0 enforced (SC1 unchanged)
amplify=true with g_min=0.5, g_max=1.5 loads
amplify=true with g_max=1.0 fails fast
```

#### Task SC2.2: Amplify calibration + run

**Files:**

- Modify: `scripts/analysis/sc_gr_monitor_probe.py` (emit amplify bands)
- Create: `inverse_folding/reference_flow/configs/d2_d3_full_stageC_scgr_amplify_aopen.yaml`

- [ ] `--emit-calibration --amplify` writes `B_low = q_low`, `B_high = q_high` so the smoothstep crossover (`g == 1`) falls at the NoD-burden median, with `g_min`/`g_max` from the config (suppress-low, amplify-high).
- [ ] Config copies SC1.4 and sets `global_pressure: {amplify: true, g_min: 0.5, g_max: 1.5}` (start conservative).
- [ ] Run on the pilot; readout:

```text
true-high / NoD-high reduction EXCEEDS SC1 (the amplify lever works)
true-low / NoD-low not harmed beyond SC1 (g_min < 1 still protects)
structure cost is bounded (scTM not materially worse than SC1)
```

- [ ] If structure degrades at `g_max=1.5`, sweep `g_max in {1.25, 1.5, 2.0}` and report the Pareto knee. Optional alternative mapping (defer unless the two-sided smoothstep is too stiff): `mapping: zscore_linear`, `g = clip(1 + a*z(B_sc), g_min, g_max)`, calibration carrying `B_mean`/`B_sd`.

---

### 6.3 Stage SC3 — Over-K Instability Trigger (optional, deferred)

Design **not settled**; do not implement until SC2 results are in. Hypothesis (`doc/Self-Cond_GR.md` §3 step 5): a protein whose unguided posterior *occasionally* spikes into a hotspot mode (high `G_*_max` / `G_*_std` over the `K` completions, low median) is a **different** case from "reliably high-burden" and may warrant a secondary amplify trigger. The monitor already logs `G_*_max` / `G_*_std` (§2), so SC3 is **analysis-first**: does instability predict NoD-burden tail variance that the median `B_sc` misses? Only if yes, add a secondary gain term — never fold it into the v1 `B_sc` gain.

---

### 6.4 Still out of scope (all stages)

- Feeding SC-GR maps into typed `b_env`, D3 `m_i`, or Phase C schedule editability (keep prospective probe signal out of persistence/targeting; `doc/Self-Cond_GR.md` §3).
- Full recycling-into-live-actuation (per-refresh `B_sc` driving `g_GR` without early-freeze) — closed-loop guard deferred to a characterized v2 (`doc/Self-Cond_GR.md` §4).

---

## 7. Self-Review Checklist

- [ ] Monitor-only: no logits, D2/D3, remask, pressure, or schedule changes.
- [ ] Probe firewall is pre-D2 and enforced by call signature.
- [ ] State update uses only structural argmax and structural confidence.
- [ ] Fresh and self-conditioned arms are both logged.
- [ ] Multi-refresh run is required before judging recycling.
- [ ] Sidecars are separate from `controller_events.parquet`.
- [ ] Analysis reports high-burden recall and true-high-as-low, not only Spearman.
- [ ] No new SLURM script.
- [ ] New analysis script is registered in `doc/SCRIPTS.md`.
