# Epitope Head Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
>
> **For Claude/Codex:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Improve the epitope head training protocol so the model learns a contiguous-window-supported residue/region immunogenicity landscape for redesign guidance, rather than an exact NetMHCIIpan-like peptide oracle.

**Architecture:** Keep the current epitope-head model and span scorer intact. Change the training protocol around it: reclassify near-positive windows, add configurable negative penalty schedules, derive residue hotspot supervision from scored contiguous windows, and train the residue layer with a ranking-based objective. Leave InfoNCE replacement, weak window-residue consistency, and global-risk redesign for later plans.

**Tech Stack:** Python, PyTorch, pandas/Parquet/JSON, existing `ProteinEntry` / chunked training pipeline, pytest.

---

## 1. Planning Rules

1. This file is the execution-facing plan for epitope-head improvement.
2. This file records frozen decisions, implementation scope, file touchpoints, verification gates, and explicit deferred items.
3. This file does not contain production code snippets.
4. Any change to the scientific target or loss semantics must be reflected here before implementation proceeds.
5. Every material planning update must be appended to `LOG.md`.

## 2. Source of Truth

1. Current improvement proposal: `doc/Epitope_Head_Improvement_Proposal.md`
2. New EL evaluation protocol: `doc/EL_new_evaluation.md`
3. Original epitope-head design: `doc/Epitope_Head_v1.md`
4. Current training and augmentation plan: `PLAN.md`
5. Meeting notes and compact result readout: `report/Experimentresults0430.md`
6. Script inventory and reuse policy: `doc/SCRIPTS.md`

## 3. Frozen Decisions

1. The head target is a residue/region immunogenicity landscape, not an exact peptide oracle.
2. Exact AP is a guardrail, not the primary optimization target.
3. Primary residue label is binary residue coverage from observed EL spans.
4. Raw density is not a primary target. Any density use must be deduplicated and capped/log-normalized, and is out of first implementation scope.
5. Near-positive windows are not reliable strong negatives.
6. Negative schedule family is configurable. v0 implements `ignore`, `linear_clamp`, `sigmoid`; the schedule registry is open for additional families (e.g. `exponential`, `thresholded_smooth`) but they are not implemented in v0.
7. Near-positive identification uses a fixed end-point distance threshold `near_gap_max` (default 10 residues). Continuous transition is expressed entirely through the chosen schedule function; there is **no** separate `softness_alpha` multiplier.
8. Keep the existing InfoNCE / mixed-margin span objective in this plan. Replacement loss design is deferred.
9. Residue supervision uses a ranking-based loss. v0 form is **pairwise margin hinge** with `m = residue.margin_m_residue` (default reuses span `margin_m=0.5`). `pairwise_bce` is reserved as a future option.
10. Residue-loss weight `lambda_residue` is a hyperparameter.
11. **One config file per experiment.** Backward-compatible defaults (near-positive and residue supervision disabled) are placed in shared schema files; new training behavior is opted in via a new config file authored alongside each experiment, never by mutating prior config defaults.
12. Weak window-residue consistency is deferred.
13. Global risk is a downstream output and is not changed in this training-plan pass.
14. Mutation augmentation remains available as a later counterfactual auxiliary / ablation, but is not part of the main EL-only objective in this plan.

## 4. File Structure Map

### Files to Modify

| File | Responsibility |
|---|---|
| `epitope_head/training/negatives.py` | Add near-positive classification and configurable negative penalty weights. |
| `epitope_head/training/losses.py` | Add weighted negative support and residue ranking loss. |
| `epitope_head/training/trainer.py` | Carry negative weights through span preparation, enumerate/score residue-supervision windows, and combine span + residue losses. |
| `epitope_head/configs/train.yaml` | Add default config keys for negative schedule and residue supervision. |
| `epitope_head/configs/model_ablation.yaml` | Add or document LC-style ablation profiles for the improved training protocol. |
| `epitope_head/configs/__init__.py` | Validate new train config keys and schedule names. |
| `doc/Epitope_Head_Improvement_Proposal.md` | Sync finalized implementation-facing decisions after plan freeze. |
| `report/Experimentresults0430.md` | Optional: keep meeting note wording aligned after final decision freeze. |
| `LOG.md` | Append one entry after planning and one after implementation completion. |

### Tests to Add or Modify

| File | Responsibility |
|---|---|
| `tests/epitope_head/training/test_negative_schedule.py` | Unit tests for near-positive classification, schedule weights, and disrupted-span handling. |
| `tests/epitope_head/training/test_residue_supervision.py` | Unit tests for residue labels, window-to-residue aggregation, and residue ranking loss. |
| `tests/epitope_head/training/test_improvement_train_integration.py` | Integration tests for trainer plumbing, config validation, and decomposed loss metrics. |

No new script or SLURM file is planned. The existing training and benchmark launchers remain the execution path.

## 5. Module HIMP: Epitope Head Training Improvement

**Objective**
- Correct the supervision mismatch in the epitope-head training protocol while preserving the existing model architecture and inference contract.

**Inputs**
- Existing strict-profile and allele-specific `ProteinSample` / `ProteinEntry` manifests.
- Existing train/val/test splits.
- Existing LC1-style model and training launcher.

**Outputs**
- Updated training code supporting configurable near-positive handling.
- Updated training code supporting span-derived residue ranking supervision.
- Updated config schema and default configs.
- Tests proving the new sampler/loss/trainer contracts.
- Training runs can be launched through existing scripts with config overrides.

**Scope Boundaries**
1. Do not replace InfoNCE in this plan.
2. Do not redesign global risk in this plan.
3. Do not add weak window-residue consistency in this plan.
4. Do not fold mutation augmentation into the main objective in this plan.
5. Do not add new top-level training scripts unless a later implementation blocker proves reuse impossible.

---

### Task HIMP0: Freeze Config Contract

**Goal**
- Make the new training behavior explicit and machine-validatable before changing sampler or loss code.

**Actions**
1. Add negative schedule config under `train.near_positive`:
   - `enabled` (bool, default `false`)
   - `near_gap_max` (int, default `10`): end-point distance threshold; gaps `> near_gap_max` are `far_decoy`
   - `schedule` (string): one of `ignore`, `linear_clamp`, `sigmoid`
   - `schedule_params` (dict): per-schedule keyword args (e.g. `linear_clamp: {w_low, w_high}`, `sigmoid: {center, slope}`)
   - `metric` (string, default `endpoint_gap`): only `endpoint_gap` is supported in v0; reserved for future IoU-based metrics
   - `apply_to_span_negatives` (bool, default `true` when `enabled`)
   - `apply_to_residue_labels` (bool, default `true` when `enabled`)
2. Supported schedule names (v0):
   - `ignore`: weight `w = 0` for all `adjacent_or_near` spans
   - `linear_clamp`: piecewise-linear ramp on end-point gap
   - `sigmoid`: logistic transition on end-point gap
   The schedule registry must be open for `exponential` and `thresholded_smooth`; v0 raises a clear `NotImplementedError` for those names.
3. Add residue supervision config under `train.residue`:
   - `enabled` (bool, default `false`)
   - `lambda_residue` (float, default `0.0`)
   - `label_mode` (string, default `binary_coverage`)
   - `aggregation` (string, default `log_mean_exp`)
   - `aggregation_params` (dict): per-aggregation kwargs (e.g. `log_mean_exp: {beta}`)
   - `loss_mode` (string, default `pairwise_margin`)
   - `margin_m_residue` (float, default `0.5`)
   - `window_mode` (string, default `sampled`)
   - `max_windows_per_chunk` (int, default `1024`): hard cap when `window_mode=all`
   - `min_far_bg_residues` (int, default `4`): chunks with fewer far-bg residues skip residue loss
4. Supported residue options (v0):
   - `label_mode`: `binary_coverage`
   - `aggregation`: `max`, `log_mean_exp` (default); `topk_mean` reserved, raises `NotImplementedError`
   - `loss_mode`: `pairwise_margin` (default); `pairwise_bce` reserved
   - `window_mode`: `sampled` (default), `all`
5. Default policy:
   - near-positive handling disabled by default for backward compatibility
   - residue supervision disabled by default for backward compatibility
   - new training profiles ship as **new config files** (one per experiment) that explicitly enable the desired combination; existing configs are not mutated

**TDD Gate**
1. RED: unknown schedule / aggregation / loss names pass validation.
2. GREEN: config validator rejects unknown names and accepts the frozen option set.

**Acceptance**
- Existing configs still load unchanged.
- New improved config keys are validated when present.

---

### Task HIMP1: Negative Semantics and Penalty Weights

**Goal**
- Stop treating near-positive non-exact windows as strong negatives and expose configurable penalty schedules for experiments.

**Actions**
1. Add span relation classification against observed positives, using IoU and end-point gap to the closest positive on the same protein:
   - `exact_positive`: identical span to a positive
   - `overlap_nonexact`: IoU > 0 but not exact
   - `adjacent_or_near`: IoU == 0 and end-point gap ≤ `near_gap_max`
   - `far_decoy`: IoU == 0 and end-point gap > `near_gap_max`
   - `counterfactual_disrupted`: pseudo-negative produced by mutation augmentation, regardless of overlap with WT positives
2. Each sampled negative carries a penalty weight `w ∈ [0, 1]` derived from its relation and the active schedule:
   - `exact_positive`: never sampled as negative
   - `overlap_nonexact`: schedule-defined (default mirrors `adjacent_or_near` at gap=0)
   - `adjacent_or_near`: schedule-defined as a function of end-point gap
   - `far_decoy`: `w = 1.0`
   - `counterfactual_disrupted`: `w = 1.0`, exempt from near-positive ignore logic
3. Preserve disrupted spans as high-confidence counterfactual negatives even if they overlap WT positives, because their pseudo-label evidence is mutation-derived.
4. Implement schedule families (v0): `ignore`, `linear_clamp`, `sigmoid`. Reserved: `exponential`, `thresholded_smooth` (raise `NotImplementedError`).
5. The schedule maps `(relation, end_point_gap)` to a scalar weight; no separate `softness_alpha` multiplier exists.
6. Ensure exact observed positives are never sampled as negatives.
7. Plumb negative weights through `sample_negatives()` return value alongside relation labels.

**TDD Gate**
1. RED: an `adjacent_or_near` span receives `w = 1.0` (full penalty) when `schedule=ignore`.
2. RED: an `exact_positive` span can be sampled as a negative.
3. RED: a `counterfactual_disrupted` span overlapping a WT positive is dropped by near-positive ignore logic.
4. RED: an unsupported schedule name (e.g. `exponential`) silently falls back to `ignore`.
5. GREEN: weights and relation labels match expected behavior on exact, overlap, near, far, and disrupted fixtures across all v0 schedules.

**Acceptance**
- `sample_negatives()` returns both spans and weights / relation metadata without breaking existing callers.
- Backward-compatible mode reproduces current negative sampling behavior.

---

### Task HIMP2: Span-Derived Residue Labels and Aggregation

**Goal**
- Build residue-level training targets and residue scores from contiguous windows, not from an unconstrained per-residue classifier.

**Actions**
1. Build binary residue coverage labels from observed positive spans (1 if covered by any positive span on this protein, else 0).
2. Support optional near-positive expansion controlled by `near_positive.apply_to_residue_labels`. When enabled, residues whose closest positive end-point falls within `near_gap_max` are marked as **ambiguous** (excluded from negative pairs but not promoted to positive).
3. Do not use raw density as a training target in this plan.
4. Enumerate residue-supervision windows according to config:
   - `sampled` (default): residue-supervision windows == span-loss positives ∪ sampled negatives (no extra forward cost)
   - `all`: all valid windows within the chunk's central region for `k ∈ [min_k, max_k]`, capped by `max_windows_per_chunk`
5. Aggregate window logits to residue scores:
   - `max`
   - `log_mean_exp` (default), parameterized by `aggregation_params.beta`
   - `topk_mean`: reserved, raises `NotImplementedError` in v0
6. Normalize aggregation for cover-count effects: `log_mean_exp` is mean-pooled (size-invariant); `max` requires no normalization.
7. **Per-chunk boundary discipline**: residue labels, ambiguous masks, and residue ranking pairs are restricted to the **chunk's central region** (chunk extent minus the boundary `margin`). Residues in chunk overlap zones are scored at most once across chunks of the same protein.
8. Keep aggregation configurable because it is a first-order design choice for gradient behavior.

**TDD Gate**
1. RED: residue label builder produces values > 1 under `binary_coverage` when a residue is covered by multiple positive spans.
2. RED: aggregation produces residue scores for residues outside the chunk's central region.
3. RED: `aggregation=topk_mean` silently falls back to another aggregation instead of raising.
4. GREEN: binary coverage labels are stable (∈ {0, 1}) under duplicate positive spans.
5. GREEN: aggregation functions produce expected outputs on small hand-computable fixtures.
6. GREEN: residue scores are defined only on the chunk's central region.

**Acceptance**
- Residue labels and residue scores are defined only on the chunk's central region.
- Contiguity is preserved because residue scores are derived from window logits.

---

### Task HIMP3: Ranking-Based Residue Loss

**Goal**
- Add a residue-level ranking objective that pushes covered residues above far-background residues without imposing absolute 0/1 probability calibration.

**Actions**
1. Implement pairwise residue ranking loss (default form **`pairwise_margin`**):
   - positives: binary-covered residues within the chunk's central region
   - negatives: far-background residues (closest positive end-point > `near_gap_max`) within the chunk's central region
   - loss: `mean over (p, n) pairs of max(0, m - (s_p - s_n))` with `m = residue.margin_m_residue` (default 0.5)
   - `pairwise_bce` form (`-log σ(s_p - s_n)`) is reserved as a future option, not implemented in v0
2. Exclude ambiguous residues (covered by `overlap_nonexact` or `adjacent_or_near` span context, per residue label builder) from negative residue pairs when `near_positive.apply_to_residue_labels` is set.
3. **High-density allele fallback**: if a chunk has fewer than `residue.min_far_bg_residues` far-background residues in its central region, skip residue loss for that chunk; increment a `residue_skipped_chunks` counter. Span loss is unaffected.
4. Combine losses:
   - existing span loss remains unchanged
   - total loss adds `lambda_residue * loss_residue` (averaged over chunks that contributed residue loss)
5. Add decomposed logging per training step:
   - `loss_span`, `loss_residue`, `loss_total`
   - `lambda_residue`
   - residue positive / negative pair counts
   - `residue_skipped_chunks` per epoch
6. Ensure `lambda_residue=0` (or `residue.enabled=false`) exactly recovers the previous training objective bit-for-bit.

**TDD Gate**
1. RED: residue loss treats ambiguous overlap residues as negative pairs when `near_positive.apply_to_residue_labels=true`.
2. RED: `lambda_residue=0` changes `loss_total` (vs. the `residue.enabled=false` baseline).
3. RED: a chunk with zero far-bg residues raises rather than skipping silently.
4. GREEN: covered residues ranking below far-bg residues produces positive loss; correct ordering lowers loss.
5. GREEN: chunks with `n_far_bg < min_far_bg_residues` are excluded from residue loss and the `residue_skipped_chunks` counter increments accordingly.

**Acceptance**
- Residue supervision is ranking-based and optional.
- Span loss behavior remains unchanged when residue supervision is disabled.

---

### Task HIMP4: Trainer Integration

**Goal**
- Wire the improved sampler and residue loss into training without changing model architecture or launcher scripts.

**Actions**
1. Extend `prepare_chunk_spans()` to carry:
   - negative weights and relation labels (one-to-one with sampled negatives)
   - disrupted-span counters
   - residue-supervision metadata: residue label vector, ambiguous mask, far-bg mask, central-region mask
2. **Union window forward (default path)**: build a unique window set per chunk as the union of:
   - span-loss positives
   - sampled negatives
   - residue-supervision windows (depends on `residue.window_mode`)
   keyed by `(start, end, allele_idx)`. Run a **single** model forward on this union, then gather logits via index maps for:
   - span-loss positive logits
   - span-loss negative logits (with weights)
   - residue-supervision window logits (for aggregation into residue scores)
3. `residue.window_mode = sampled` (default): residue-supervision windows == span-loss positives ∪ sampled negatives, so the union forward equals the span-loss forward — no additional cost.
4. `residue.window_mode = all`: enumerate all valid windows for `k ∈ [min_k, max_k]` within the chunk's central region. Hard cap by `max_windows_per_chunk`; if exceeded, **fail loudly** rather than fall back to a separate residue-only forward.
5. Extend `train_step()` and `val_step()` to:
   - run the union window forward
   - compute span loss from the gathered span-loss logits with negative weights
   - aggregate residue-supervision window logits into residue scores (per `residue.aggregation`)
   - compute residue ranking loss when `residue.enabled` and chunk passes the `min_far_bg_residues` gate
   - log decomposed loss terms (`loss_span`, `loss_residue`, `loss_total`, `residue_skipped_chunks`)
6. Keep current validation path compatible with existing configs.
7. Avoid changing inference payload schema.
8. Keep global risk computation unchanged.
9. Print all new hyperparameters and effective resolved config (including schedule kwargs, residue kwargs, `near_gap_max`, `lambda_residue`) to stdout at trainer start so experiments are confirmable from the SLURM log.

**TDD Gate**
1. RED: enabling residue supervision contributes zero gradient to scorer parameters.
2. RED: old configs break when new trainer code is loaded.
3. RED: `window_mode=all` exceeding `max_windows_per_chunk` silently falls back to a duplicated forward.
4. GREEN: old config path and improved config path both execute on a tiny synthetic batch.
5. GREEN: union window forward returns the same span-loss logits as the legacy code path when `residue.enabled=false`.

**Acceptance**
- Training can run with the old objective or the improved objective through config only.
- No new script is required.

---

### Task HIMP5: Config Profiles and Experiment Handles

**Goal**
- Provide clean experiment handles without adding new scripts.

**Actions**
1. Author a **new** LC-style training config file (e.g. `epitope_head/configs/cnn_himp_v1.yaml`) per experiment that enables:
   - near-positive schedule (`schedule=sigmoid` or `linear_clamp`, swept per experiment)
   - residue supervision (`residue.enabled=true`, `residue.window_mode=sampled`)
   - `lambda_residue` as a tunable hyperparameter (initial sweep around `0.05 / 0.1 / 0.2`)
   Existing configs (`train.yaml`, `cnn_full_aug.yaml`) remain unchanged so prior runs replay identically.
2. **One config per experiment** (project convention). Do not mutate prior config defaults; ablations live in their own config files (`cnn_himp_v1_aggmax.yaml`, `cnn_himp_v1_lin.yaml`, etc.).
3. Keep aggregation function and schedule family configurable so ablations are config-only.
4. Keep mutation augmentation off in the main improved profile.
5. Add a separate documented ablation handle for mutation augmentation if later enabled.
6. Make monitor metric explicit. Prefer existing held-out exact metric for backward compatibility, but require post-training benchmark comparison on the new protocol (`doc/EL_new_evaluation.md`) before judging success.

**TDD Gate**
1. RED: improved profile omits required new config keys.
2. RED: improved profile contains an unsupported schedule or aggregation name and validation passes.
3. GREEN: config loader accepts baseline and improved profiles; rejects unsupported schedule/aggregation names with a clear error citing the offending key.

**Acceptance**
- A single existing training launcher can run the improved objective with config/profile overrides.

---

### Task HIMP6: Evaluation and Acceptance

**Goal**
- Define how to judge the improvement without changing the benchmark protocol again.

**Actions**
1. Use the frozen new EL benchmark protocol from `doc/EL_new_evaluation.md`.
2. Primary readouts:
   - residue AP
   - residue Spearman
   - IoU>=0.7 AP
   - IoU>=0.5 AP
3. Guardrails:
   - exact AP should not collapse
   - far-FP fraction should not increase materially
   - coverage/stability should remain valid
4. Diagnostics:
   - residue Pearson
   - EMD similarity
   - near-miss / far-FP decomposition
5. Compare at least:
   - current LC1 baseline
   - near-positive schedule only
   - near-positive schedule + residue ranking loss
   - aggregation variants if compute budget allows

**TDD Gate**
1. RED: benchmark result parser cannot detect missing primary readouts.
2. GREEN: acceptance summary can load benchmark JSON and report primary/guardrail/diagnostic metrics.

**Acceptance**
- Improved head is accepted only if region/residue metrics improve while exact AP and far-FP guardrails remain acceptable.

---

## 6. Deferred Items

These are explicitly out of scope for this plan:

1. Replacing InfoNCE with a new primary span loss.
2. NMP distillation as a main training target.
3. Folding mutation augmentation into the main EL-only objective.
4. Weak window-residue consistency regularization.
5. Global risk formula redesign.
6. Architecture changes such as core-aware pooling or a separate residue head.
7. New SLURM scripts or new top-level training scripts.

## 7. Release Gates

1. Unit tests for negative schedule and residue supervision pass.
2. Trainer integration tests pass for baseline and improved configs.
3. Existing baseline config remains backward compatible.
4. Improved training run emits decomposed loss terms.
5. Post-training benchmark uses the same `benchmark_iedb_test.py` protocol as the current results.
6. `LOG.md` records implementation completion and benchmark outcome.
