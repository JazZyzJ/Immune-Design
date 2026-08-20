# Uricase Enzyme Mode v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans`
> when implementing this plan task-by-task. The project also asks for
> `superpowers:writing-plans` when creating plans, but that skill is not exposed in
> the current Codex skill list; this file is written manually as an executable
> checklist.

**Goal:** Implement the first uricase enzyme mode: active-site-constrained global
Reference Flow (RF) redesign. The only runtime constraint is that hard active-site
anchors are fixed from step 0 and permanently protected from remask. All non-anchor
positions stay globally editable.

**Design source:** `doc/Uricases_Design.md`.

**Architecture:** Enzyme mode v0 is a sampler-owned hard legality layer. It does
not change D1, D2, D3, SC-GR, the global pressure control law, or the DPLM decoder.
The full WT sequence is not used as a generation prior. WT appears only in the
post-hoc decision layer as a comparator, no-op candidate, and rescue substrate.

**Tech stack:** Python 3.12, PyTorch, NumPy, pandas/pyarrow, existing
`inverse_folding/reference_flow/*`, existing `scripts/run_if_phase_c1.py`, existing
`scripts/submit_if_phase_c.slurm`, existing `scripts/evaluate_phase_c.py`.

---

## 0. Scope And Non-Goals

**This plan owns:**

- A constraint manifest format for hard anchors keyed by `protein_id`.
- Fail-fast per-case numbering and expected-AA validation.
- RF sampler initialization and permanent remask protection for hard anchors.
- CLI / SLURM wiring through the existing Phase C RF entrypoints.
- v0 telemetry for active-site overlap, risk deltas, and controller pressure
  misinterpretation diagnostics.
- `F_post` v0 measurement: `f_anchor` hard invariant plus monomer `f_fold`
  ranking-only reporting.

**Out of scope for v0:**

- No WT-conditioned generation.
- No automatic local polish.
- No WT-rescue implementation.
- No shell-specific beta / delta_struct / alpha policy.
- No reachability-aware `g_GR` or function-aware online gain.
- No multimer `f_assembly` or `f_site_scaffold` gate.
- No new SLURM script unless implementation discovers an unavoidable launcher gap.
  Prefer extending `scripts/run_if_phase_c1.py` and
  `scripts/submit_if_phase_c.slurm`; if a new script becomes necessary, register it
  in `doc/SCRIPTS.md`.

**TDD decision:** required. This touches the sampler state machine and controller
interaction surface. Write focused tests before or alongside each implementation
task.

---

## 1. Core Contract

For a constrained protein with hard anchor set `A`:

```text
x_T[i] = expected AA token, if i in A
x_T[i] = MASK,              otherwise
```

At every generation step:

```text
head / SC-GR may score every window
active-window selection may include windows overlapping anchors
D2 candidate positions must exclude anchors
D3 / reparam remask must exclude anchors
final generated sequence must preserve every anchor identity
```

This is the v0 interpretation of the key risk from `doc/Uricases_Design.md`:
anchors are visible to the immune head but noneditable. The implementation must
not hide anchor-overlap immune risk. It must instead preserve legality and record
telemetry that shows whether the controller is spending pressure on risk that is
not repairable under the hard-anchor constraint.

---

## 2. Constraint Manifest Contract

Create a manifest schema that is explicit about numbering and provenance. The
first implementation can be JSON or YAML; choose the parser style that matches the
existing config helpers best.

Example shape:

```yaml
schema_version: uricase_active_site_v0
description: Q00511 v0 hard-anchor constraint
entries:
  - protein_id: Q00511
    sequence_md5: optional_but_preferred
    source_sequence: resolved_parquet_sequence
    hard_anchors:
      - index_0b: 10
        expected_aa: K
        label: Lys11
        biological_role: catalytic_triad
        source: UniProt Q00511 active site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 57
        expected_aa: T
        label: Thr58
        biological_role: catalytic_triad_and_binding
        source: UniProt Q00511 active+binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 58
        expected_aa: D
        label: Asp59
        biological_role: binding
        source: UniProt Q00511 binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 159
        expected_aa: F
        label: Phe160
        biological_role: binding_purine_stacking
        source: UniProt Q00511 binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 176
        expected_aa: R
        label: Arg177
        biological_role: substrate_stabilizer
        source: UniProt Q00511 binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 228
        expected_aa: Q
        label: Gln229
        biological_role: substrate_stabilizer
        source: UniProt Q00511 binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 254
        expected_aa: N
        label: Asn255
        biological_role: binding_cross_protomer_W1
        source: UniProt Q00511 binding site / M-CSA:118 / doc/Uricases_Design.md
      - index_0b: 256
        expected_aa: H
        label: His257
        biological_role: catalytic_triad
        source: UniProt Q00511 active site / M-CSA:118 / doc/Uricases_Design.md
    monitored_shell:
      - index_0b: 227
        expected_aa: V
        label: Val228
        enforcement: monitored_posthoc
        rationale: UniProt binding annotation but family-variable; not hard-fixed in v0
```

Validation rules:

- `protein_id` must match the Phase C test-set row.
- Each `index_0b` must be within the resolved `sequence`.
- `resolved_sequence[index_0b] == expected_aa` must pass before generation.
- Duplicate hard-anchor indices are a hard error.
- Empty hard-anchor lists are allowed only for unconstrained controls and must be
  recorded as such.
- The manifest hash must be stamped into the run manifest.

---

## 3. Implementation Tasks

### Task U0 - Preflight And Contract Freeze

- [ ] Re-read `doc/Uricases_Design.md`, this plan, `AGENTS.md`,
      and `doc/SCRIPTS.md`.
- [ ] Confirm the Q00511 v0 hard set is
      `{10, 57, 58, 159, 176, 228, 254, 256}` with expected amino acids
      `K, T, D, F, R, Q, N, H`.
- [ ] Decide the on-disk manifest path for the first run. It must be passed by CLI
      or environment variable, not hardcoded in Python.
- [ ] Do not update `LOG.md` for pure doc/plan changes. Implementation changes
      later need normal project judgment.

### Task U1 - Constraint Loader

- [ ] Add a small library module, likely
      `inverse_folding/reference_flow/constraints.py`.
- [ ] Define typed records for `HardAnchor`, `MonitoredShellResidue`, and a
      per-protein `ActiveSiteConstraint`.
- [ ] Implement `load_constraint_manifest(path)` with schema validation and stable
      hash calculation.
- [ ] Implement `constraint_for_protein(protein_id)` with fail-fast behavior when
      the manifest is present but a requested constrained protein is missing.
- [ ] Implement `validate_against_sequence(sequence)` that asserts index bounds and
      expected AA identity.
- [ ] Convert hard anchors to model token ids using the existing DPLM alphabet
      mapping at the script/runtime boundary, not inside the manifest parser.

Tests:

- [ ] Valid Q00511-style manifest parses.
- [ ] Wrong expected AA fails.
- [ ] Out-of-range index fails.
- [ ] Duplicate anchor fails.
- [ ] Manifest hash is stable.

### Task U2 - Sampler Hard-Anchor State

- [ ] Extend `PositionDependentDFMSampler.sample()` with an optional fixed-token
      input, for example `fixed_tokens: Mapping[int, int] | None`.
- [ ] Extend `SamplerBatchLane` / `sample_batch()` with the same information.
- [ ] Initialize `x_t[index] = token_id` for every hard anchor before the denoising
      loop starts.
- [ ] Mark anchors as committed in a way that keeps them out of first-unmask.
      Prefer a separate `fixed_positions` set for telemetry over overloading
      `unmask_step_by_pos`.
- [ ] Set `unmask_step_by_pos[index] = 0` for hard anchors to mean
      "step-0 committed by constraint"; do not leave anchors at `-1`, because
      downstream telemetry may read `-1` as still masked / never committed.
      Also expose `fixed_positions` so constrained anchors remain distinguishable
      from ordinary positions first sampled at step 0.
- [ ] Audit every consumer of `unmask_step_by_pos` before implementation lands.
      If a consumer assumes `0` means stochastic first-unmask, use the explicit
      `fixed_positions` marker to branch.
- [ ] On every remask call, pass `protected_positions =
      permanent_fixed_positions union controller_post_protected`.
- [ ] Ensure this permanent protection runs even when `controller=None`.
- [ ] Keep the no-constraint path bit-equivalent to current behavior.

Tests:

- [ ] With no constraints, sampler output and telemetry are unchanged.
- [ ] With constraints and `remask.enabled=false`, anchors remain fixed.
- [ ] With constraints and `remask.enabled=true`, anchors remain fixed.
- [ ] With a stub controller that asks to remask broadly, anchors remain fixed.
- [ ] Batch and single-lane sampling enforce the same anchors.
- [ ] Anchor `unmask_step_by_pos` values are `0`, and `fixed_positions` identifies
      them as constraint-committed rather than sampled-at-step-0 residues.

### Task U3 - RF Driver And Manifest Wiring

- [ ] Add `--constraint-manifest PATH` to `scripts/run_if_phase_c1.py`.
- [ ] Validate each constrained protein before calling the sampler.
- [ ] Pass the per-protein fixed token map into the sampler.
- [ ] Emit a sidecar rather than changing the core `generated.parquet` schema:
      `constraints_applied.parquet`.
- [ ] Add run manifest fields:

```text
enzyme_mode_enabled
constraint_manifest_path
constraint_manifest_hash
constraint_schema_version
constraint_surface_version
num_constrained_proteins
num_hard_anchors_total
constraints_applied_path
```

- [ ] Extend `scripts/submit_if_phase_c.slurm` with
      `CONSTRAINT_MANIFEST`, forwarded only in `MODE=reference_flow`.
- [ ] Do not create a new launcher for v0.

Tests:

- [ ] CLI rejects a missing constraint manifest path.
- [ ] CLI accepts no manifest and preserves current behavior.
- [ ] Manifest fields are present only when constraints are enabled.
- [ ] Wrong-AA Q00511 spec fails before generation.

### Task U4 - Hard-Constraint Telemetry

Write `constraints_applied.parquet`, one row per generated design and anchor:

```text
protein_id
design_idx
anchor_index_0b
expected_aa
expected_token_id
generated_aa
generated_token_id
preserved
label
biological_role
constraint_manifest_hash
```

Also write a compact JSON summary:

```text
num_designs
num_anchor_rows
num_anchor_mismatches
all_anchors_preserved
mismatch_examples
```

Hard gate:

- [ ] Any `preserved=false` is a failed run, not a weak metric.

### Task U5 - Active-Site Immune Telemetry

Use external NetMHCIIpan outputs as the decision-layer immune source. The first
implementation can be an analysis utility inside the existing evaluation path or a
post-run helper, but it must be reproducible and keyed by the run manifest.

Inputs:

```text
generated.parquet
WT facade eval outputs (eval_immune/imm_nmp*.parquet) via --wt-facade-eval-dir
imm_nmp.parquet
imm_nmp_peptides.parquet from evaluate_phase_c.py --imm-full
constraint manifest
```

WT facade wiring (B1):

- The WT facade run dir is passed via a new optional analysis flag
  `--wt-facade-eval-dir PATH` (no hardcoded paths in Python). First-run instance:
  `Results/RF/HLA-DRB1_07_01/wt_uricase_v2_HLA-DRB1_07_01_imm_full__20260607T185243Z`.
- That dir currently carries `eval_immune/imm_nmp.parquet` +
  `imm_nmp_peptides.parquet` (same schema as a design eval) and the WT sequences
  under `generation/`, so immune deltas are computable now.
- When the flag is absent, every `*_vs_wt` immune column is `NA`, never a
  placeholder value.

Join key normalization:

- Normalize every input to `(protein_id, design_idx, design_id)` before joining.
  Current `evaluate_phase_c.py` reconstructs `design_id = design_{design_idx:04d}`,
  but sidecar builders must not assume every historical parquet carries both
  columns.
- If `design_idx` is missing but `design_id` has the canonical `design_0000`
  format, reconstruct `design_idx` and assert uniqueness.
- If `design_id` is missing, reconstruct it from `design_idx`.
- Any duplicate `(protein_id, design_idx)` after normalization is a hard error.

Required window-level sidecar: `enzyme_nmp_windows.parquet`

```text
protein_id
design_idx
source                      # design | WT
pep_length
pos
peptide
rank_EL
el_score
strong_binder               # rank_EL <= configured threshold
rank_threshold
overlaps_hard_anchor
num_hard_anchors_covered
min_distance_to_hard_anchor
covered_anchor_indices
```

Required design-level sidecar: `enzyme_immune_summary.parquet`

```text
protein_id
design_idx
nmp_total_strong
nmp_anchor_overlap_strong
nmp_non_anchor_strong
nmp_total_rank_margin_mass
nmp_anchor_overlap_rank_margin_mass
nmp_non_anchor_rank_margin_mass
delta_total_strong_vs_wt
delta_anchor_overlap_strong_vs_wt
delta_non_anchor_strong_vs_wt
delta_total_rank_margin_mass_vs_wt
delta_anchor_overlap_rank_margin_mass_vs_wt
delta_non_anchor_rank_margin_mass_vs_wt
hotspot_concentration_top1_frac
hotspot_concentration_top3_frac
hotspot_concentration_top5_frac
external_only_candidate_flag
```

Definitions:

- Compare design vs WT by matched `(protein_id, pep_length, pos)`.
- Negative delta means risk decreased relative to WT.
- `rank_margin_mass = sum(max(0, rank_threshold - rank_EL))` over the selected
  window subset.
- `hotspot_concentration_topK_frac` is the fraction of total risk mass carried by
  the top K windows; it routes concentrated residuals toward RF-polish later.
- `external_only_candidate_flag` marks final NMP hotspots with weak/no in-loop head
  evidence once controller joins are available.

### Task U6 - Controller Misinterpretation Telemetry

This is the key v0 diagnostic. It tests whether the controller treats visible
anchor-overlap risk as editable pressure.

Do not change controller decisions for v0. Prefer post-hoc joins over controller
logic changes. If runtime context must carry fixed positions, it is telemetry-only
and must not affect logits, active-window selection, D2, D3, or SC-GR.

Required sidecar: `enzyme_controller_overlap.parquet`

```text
protein_id
design_idx
seed
refresh_step
step
t
block_id
block_residue_start_0b
block_residue_end_0b
window_indices                      # IDs into refresh r_windows_dyn, not residue indices
window_spans_0b                     # expanded from r_windows_dyn[window_indices]
overlaps_hard_anchor
covered_anchor_indices
window_excess_sum
window_excess_max
candidate_feasibility
best_delta_R_B
safe_support_size_min
safe_support_size_mean
safe_support_size_values
selected_positions
selected_hard_anchor_count        # must be 0
selected_flank_in_anchor_window_count
realized_benefit_rate
final_persistence_rate
remasked_hard_anchor_count        # must be 0
visible_noneditable_pressure_flag
```

Field semantics:

- `window_indices` are `ActiveBlock.window_indices`: IDs of windows in the refresh
  record's `r_windows_dyn`, not residue indices.
- `block_residue_start_0b` / `block_residue_end_0b` come from the merged
  `ActiveBlock` span and are half-open.
- `overlaps_hard_anchor` must be computed as:

```text
bool(set(range(block_residue_start_0b, block_residue_end_0b)) & hard_anchor_set)
```

  The implementation may also expand `window_indices -> r_windows_dyn[id] ->
  (start_0b, end_0b)` to report `window_spans_0b`, but it must not treat
  `window_indices` themselves as residue positions.
- `window_excess_sum` / `window_excess_max` are reductions over the windows named
  by `window_indices`.
- `safe_support_size_values` is `list(exported_safe_support_sizes.values())` for
  the block. `safe_support_size_min` and `safe_support_size_mean` are reductions
  over those values only. In the current checkout the exported map is a
  block-local editable-position support-size map; older notes may describe this
  as a token/support count. Do not infer residue or token semantics from the dict
  keys; this sidecar only needs the value distribution.
- `realized_benefit_rate` and `final_persistence_rate` are not native per-window
  fields. The sidecar builder must compute them by joining block-level refresh
  diagnostics to `controller_events.parquet` and aggregating the D2 rows assigned
  to that `(protein_id, design_idx, refresh_step, block_id)`.
- Required block/event join:

```text
refresh_log.jsonl block diagnostics
  keyed by protein_id, design_idx, refresh_step, block_id
controller_events.parquet D2 rows
  keyed by protein_id, design_idx, refresh_step, block_id, position_i
```

  The builder owns this join; do not rely on the existing run aggregator to have
  precomputed per-block rates.
- `realized_benefit_rate` for a block is the mean of non-null
  `realized_benefit_flag` over joined D2 rows for that block.
- `final_persistence_rate` for a block is the fraction of joined selected D2
  positions that are not later remasked in the same design trajectory.

Suggested flag:

```text
visible_noneditable_pressure_flag =
  overlaps_hard_anchor
  and window_excess_max > 0
  and selected_hard_anchor_count == 0
  and (
        candidate_feasibility is false
        or best_delta_R_B >= 0
        or final NMP anchor-overlap delta does not improve
      )
```

Interpretation:

- Many anchor-overlap blocks with negative `best_delta_R_B` and improved final NMP
  risk means flanking/global edits can repair the visible risk.
- Many anchor-overlap blocks with no safe useful proposal means structural/proposal
  limitation, not a sampler bug.
- Many final NMP anchor-overlap hotspots with no in-loop head evidence means
  head-validator mismatch.
- Many high-pressure anchor-overlap blocks with no realized benefit is the trigger
  for E2 reachability-aware pressure, not a v0 hotfix.

### Task U7 - `F_post` v0 Measurement

Implement or wire a v0 measurement envelope:

```text
f_anchor: hard_invariant
f_fold: ranking_only
f_assembly: unavailable
f_site_scaffold: unavailable
```

Required sidecar: `f_post_v0.parquet`

```text
protein_id
design_idx
f_anchor_status              # pass | fail
num_anchor_mismatches
f_fold_status                # ranking_only | predictor_failed | unavailable
scTM
bb_RMSD
pLDDT
recovery
foldability
delta_scTM_vs_wt
delta_bb_RMSD_vs_wt
delta_pLDDT_vs_wt
delta_recovery_vs_wt
reference_type               # predicted_target for v0 uricase cases
predictor
predictor_version
calibration_status
```

Structure source:

- v0 `f_fold` reads `structural.parquet` produced by
  `scripts/evaluate_phase_c.py --mode struct` or `--mode all`.
- Required current columns are:

```text
protein_id, design_id, design_idx, sequence, scTM, pLDDT, bb_RMSD,
recovery, foldability, refold_backend
```

- Deltas require a same-protocol WT facade structural run (passed via
  `--wt-facade-struct-dir PATH`) normalized to the same
  `(protein_id, design_idx, design_id)` convention as U5.
- NOTE (B1): the packaged WT facade run
  `wt_uricase_v2_..._imm_full__20260607T185243Z` carries `eval_immune` only — no
  `structural.parquet`. Until a WT facade `--mode struct` run is produced, every
  `delta_*_vs_wt` structural column is `NA` (not placeholder); the absolute
  `scTM/bb_RMSD/pLDDT/recovery` columns are still emitted.
- Do not use `scRMSD` unless a future structural evaluator emits that exact column;
  the current evaluator column is `bb_RMSD`.

Rules:

- `f_anchor=fail` rejects the design.
- `f_fold` must not be called a calibrated gate in v0; it is ranking-only unless
  axis-matched negatives are later added.
- Missing multimer coordinates are `NA`, not pass.

### Task U8 - First-Run Protocol

Predeclare the first run before looking at outcomes.

Minimal first target:

```text
protein_id: Q00511
allele: HLA-DRB1*07:01
constraint: hard anchors {10, 57, 58, 159, 176, 228, 254, 256}
mode: reference_flow
generation: all-mask-except-hard-anchors global RF
immune eval: evaluate_phase_c.py --imm-full, external NMP primary
structure eval: monomer fold proxy only
WT baseline: WT facade scored under the same protocol
```

Optional expansion after the Q00511 smoke passes:

```text
25 characterized uricases
stratified subset of the 6740-case pool
```

Do not claim an allele comparison until 0401 has a same-protocol WT baseline.

### Task U9 - Verification And Review Gates

Local tests before any cluster run:

- [ ] Constraint parser tests pass.
- [ ] Sampler fixed-anchor tests pass.
- [ ] Batch sampler fixed-anchor tests pass.
- [ ] `run_if_phase_c1.py --help` shows `--constraint-manifest`.
- [ ] `bash -n scripts/submit_if_phase_c.slurm` passes.
- [ ] No-constraint RF path remains equivalent in focused tests.
- [ ] Wrong-AA constraint fails before generation.

Cluster / artifact gates:

- [ ] Q00511 constrained smoke produces `generated.parquet`.
- [ ] `constraints_applied.parquet` has zero mismatches.
- [ ] `enzyme_nmp_windows.parquet` and `enzyme_immune_summary.parquet` are written.
- [ ] `enzyme_controller_overlap.parquet` is written when controller telemetry is
      available; if controller is disabled, record `unavailable`.
- [ ] `f_post_v0.parquet` is written with `f_anchor=pass` and `f_fold=ranking_only`.
- [ ] WT baseline artifacts are real and same-protocol; no placeholder WT values.

Review questions after the first run:

1. Did total external NMP risk decrease relative to WT?
2. Did anchor-overlap risk decrease, stay flat, or increase?
3. Did active-site overlap windows consume controller budget without realized benefit?
4. Were residual hotspots concentrated enough to justify later RF-polish?
5. Did monomer `f_fold` collapse relative to WT?
6. Is the failure mode v0 mechanics, head-validator mismatch, structural proposal
   limitation, or genuinely blocked residual risk?

Only after answering these questions should E1 shell policy, E2 reachability-aware
gain, automatic polish, or WT-rescue be promoted from deferred design to
implementation.
