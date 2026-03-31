# Epitope Head v2 Modular Deployment Plan

> **For Claude:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Build a decision-ready v2 ablation workflow that selects the best encoder backbone for epitope ranking, mutation sensitivity, and diffusion-time inference efficiency.

**Architecture:** Reuse A-G baseline data/training/inference pipeline as fixed substrate, and add a controlled encoder-ablation layer (E0/E1/E2) where only encoder-related variables move. Keep head/loss/split/evaluation protocol constant unless explicitly frozen as an ablation-specific exception.

**Tech Stack:** Python, PyTorch, existing Epitope Head A-G pipeline, Parquet/JSON artifacts, pytest.

---

## 1. Planning Rules (This File)

1. This file is a living execution plan for v2 modules, updated incrementally per user request.
2. Each module update must contain:
   - objective
   - exact file touchpoints
   - task-level execution steps
   - TDD verification gates
   - done criteria
3. This file records implementation strategy only, not production code.
4. If assumptions change, append codemap diff first, then sync this plan.
5. Every plan edit must produce a corresponding `LOG.md` entry.

## 2. Source of Truth

1. v1 baseline contract and behavior: `doc/Epitope_Head_v1.md`
2. v2 ablation spec and hypotheses: `doc/Epitope_Head_V2.md`
3. engineering map and diff ledger: `doc/Epitope_Head_codemap.md`
4. archived v1 implementation plan: `PLAN_epi_v1.md`

## 3. Global Milestones (v2)

### Milestone M5: V2 Encoder Ablation Green (Stage H)

- Outcome:
  - encoder matrix `E0/E1/E2` is trainable and evaluable under controlled constants
  - report includes ranking quality + mutation sensitivity + latency
  - one encoder recommendation is frozen for v2 downstream integration

### Milestone M6: CNN Enhancement Green (Stage I)

- Outcome:
  - validate whether CNN remains the most reasonable route after targeted optimization
  - complete controlled comparison on `B0/L1/C1/LC1` matrix (loss and encoder refinements)
  - freeze next-stage default config by AP-centric decision rule

## 4. Module Backlog (Ordered)

## Module H: Encoder Ablation (V2 Module 1)

**Objective**
- Execute controlled encoder ablation to answer:
  1. pretraining contribution (`E0` vs `E1/E2`)
  2. local vs shallow-global inductive bias (`E1` vs `E2`)
  3. mutation sensitivity and inference latency tradeoff for diffusion integration

**Inputs**
- `outputs/manifests/protein_samples_strict.parquet`
- `outputs/manifests/splits/strict/{train_ids.txt,val_ids.txt,test_ids.txt}`
- A-G stable training/inference stack (trainer, evaluator, predictor)

**Outputs**
- `outputs/ablation/encoder_v2/runs/<encoder>/<seed>/...`
- `outputs/ablation/encoder_v2/metrics/encoder_ablation_raw.jsonl`
- `outputs/ablation/encoder_v2/metrics/encoder_ablation_summary.csv`
- `outputs/ablation/encoder_v2/report/encoder_ablation_report.md`

**Ablation Matrix (Frozen)**

| ID | Encoder | Pretraining | Context | Trainability |
|---|---|---|---|---|
| E0 | Frozen ESM-2 (`esm2_t33_650M_UR50D`) | yes | deep global | frozen encoder, train head |
| E1 | Dilated 1D CNN | no | local (RF ~121 AA) | train from scratch |
| E2 | Shallow Transformer (4L) | no | global (shallow) | train from scratch |

**Frozen Comparison Constants**
1. Data profile: `strict` only for primary decision.
2. Split source: reuse existing strict split IDs from Module D (no re-splitting in H).
3. Span head: keep projection/span-feature/scorer logic identical to A-G baseline.
4. Loss and sampler: keep `neg_ratio`, hard negative policy, `min_k/max_k`, and loss terms unchanged.
5. Seeds for each encoder: `[42, 43, 44]`.
6. Early stop monitor: `pp_auc`.
7. Runtime chunking: keep v1.1 chunk config (`context_len=1022`, `stride=512`, `margin=32`).

**Encoder Parameter Freeze (H-specific)**
1. `E0`:
   - keep current model config and frozen-encoder path unchanged.
2. `E1` Dilated CNN:
   - token embedding dim: `256`
   - conv blocks: `8`
   - kernel size: `5`
   - dilation schedule: `[1, 2, 4, 8, 8, 4, 2, 1]` (effective RF = 121)
   - hidden channels: `256`
   - block dropout: `0.1`
   - output `d_enc = 256`
3. `E2` Shallow Transformer:
   - token embedding dim (`d_model`): `256`
   - encoder layers: `4`
   - attention heads: `8`
   - FFN dim: `1024`
   - dropout: `0.1`
   - positional embedding max length: `1022`
   - output `d_enc = 256`

**Planned File Touchpoints**
- Create/Modify: `epitope_head/training/model.py`
- Create: `epitope_head/training/encoders.py`
- Modify: `epitope_head/configs/model.yaml`
- Create: `epitope_head/configs/model_ablation.yaml`
- Modify: `epitope_head/configs/__init__.py`
- Create: `scripts/train_v2_ablation.py`
- Create: `scripts/eval_encoder_ablation.py`
- Create: `tests/epitope_head/training/test_module_h_encoder_contract.py`
- Create: `tests/epitope_head/training/test_module_h_eval_contract.py`

### Task H0: Freeze H-Module Experimental Contract

**Goal**
- Prevent confounded comparisons before code changes.

**Actions**
1. Freeze all constants listed above into config and plan.
2. Define explicit "allowed variable set" for H:
   - encoder architecture
   - encoder trainability
   - encoder hidden size (`d_enc`) per ablation profile
3. Forbid silent changes to sampler/loss/split during H runs.

**TDD Gate**
1. RED: config validator should fail when ablation profile omits required encoder keys.
2. GREEN: add minimal schema validation for ablation config profiles.

**Acceptance**
- H configuration is machine-validated and ablation-safe.

### Task H1: Introduce Encoder Factory + Unified Tokenizer Contract

**Goal**
- Make training path select `E0/E1/E2` via config without touching trainer logic.

**Actions**
1. Add encoder factory:
   - input: `encoder_type` (`esm2_frozen|dilated_cnn|shallow_transformer`)
   - output: module with interface `(token_ids, attention_mask) -> (H, lengths)`
2. Keep collate contract unchanged (`token_ids`, `attention_mask` tensors).
3. Add lightweight AA tokenizer path for non-ESM encoders, retaining BOS/EOS/PAD semantics.
4. Ensure `EpitopeScorer.encode_and_project` remains interface-stable.

**TDD Gate**
1. RED: factory request for each encoder type fails before implementation.
2. GREEN: each encoder type builds and returns expected output shape and length tensor.

**Acceptance**
- Trainer can switch encoders by config flag only.

### Task H2: Implement E1 (Dilated CNN Encoder)

**Goal**
- Add local-context, no-pretraining encoder baseline.

**Actions**
1. Implement residual dilated Conv1D stack with frozen H-parameters.
2. Respect attention mask so padding does not contaminate outputs.
3. Return residue embeddings aligned with chunk residue lengths.
4. Verify receptive field matches planned local-context target (~121 AA).

**TDD Gate**
1. RED: padded tokens affect outputs in masked region.
2. GREEN: masked output invariants pass and output shape matches expected `[B, L_max, 256]`.

**Acceptance**
- E1 is trainable end-to-end and contract-compatible with existing head.

### Task H3: Implement E2 (Shallow Transformer Encoder)

**Goal**
- Add shallow global-context, no-pretraining encoder baseline.

**Actions**
1. Implement 4-layer Transformer encoder with absolute positional embeddings.
2. Apply key-padding mask from `attention_mask`.
3. Return residue embeddings excluding BOS/EOS, aligned with lengths.
4. Keep output dimension fixed to `256`.

**TDD Gate**
1. RED: variable-length batch with padding produces incorrect lengths or shape mismatch.
2. GREEN: length/shape/mask tests pass on mixed sequence lengths.

**Acceptance**
- E2 is trainable end-to-end and drop-in compatible with head/scorer.

### Task H4: Build H-Runner for Matrix Training

**Goal**
- Execute the full `E0/E1/E2 x seed(42,43,44)` matrix reproducibly.

**Actions**
1. Add dedicated training entrypoint (`train_v2_ablation.py`) to avoid destabilizing v1 launcher.
2. Runtime args:
   - `--encoder-id` (`E0|E1|E2`)
   - `--seed`
   - `--profile strict`
   - `--device`
3. Standardize run directory schema:
   - `outputs/ablation/encoder_v2/runs/<encoder_id>/seed_<seed>/`
4. Write resolved config snapshot and training summary for each run.

**TDD Gate**
1. RED: missing `encoder-id` or unsupported id should fail CLI validation.
2. GREEN: smoke run creates expected run directory tree and summary artifacts.

**Acceptance**
- All matrix runs are reproducible, traceable, and registry-ready.

### Task H5: Implement Ablation Evaluation Extensions

**Goal**
- Extend evaluation beyond ranking metrics to match v2 spec, and add two diagnostic checks to validate whether CNN gains are real and interpretable.

**Actions**
1. Ranking metrics:
   - reuse existing `pp_auc`, `pp_ap`, `pp_recall_50`, `pp_recall_100`
   - aggregate mean/std across seeds per encoder
2. Mutation sensitivity:
   - val subset policy: proteins with `n_pos >= 1`, deterministic sample size `64`, seed `42`
   - per protein mutation budget: up to `1900` substitutions (all 19 AA replacements for up to 100 sampled positions)
   - statistic outputs: `delta_z_mean`, `delta_z_std`, `delta_z_abs_p90`, `delta_z_sign_balance`
3. Convergence/overfit:
   - `epoch_to_best_pp_auc`
   - `best_to_last_pp_auc_drop`
4. Latency:
   - batch size `1`, lengths `{128, 256, 512, 1022}`
   - warmup `20`, timed `100`
   - record `median_ms`, `p95_ms` (encoder+head forward)
5. Diagnostic A (per-protein AUC distribution):
   - compute per-protein `pp_auc_i` on full val proteins
   - export histogram-ready table and quantiles `{p10,p25,p50,p75,p90}`
   - report count/proportion of proteins with `pp_auc_i < 0.8`
   - attach protein-level context columns: `n_pos_spans`, `protein_len`
6. Diagnostic B (CNN motif plausibility check):
   - do not directly interpret conv1 weights as `21 x k` logo when input channels are embedding dim (`256`)
   - use activation-based motif extraction:
     - collect top-activating k-mers per first-layer filter from val proteins
     - build AA frequency logo/PSSM from these top windows
     - compare motif preference with known `HLA-DRB1*07:01` qualitative binding anchors
   - output per-filter motif summary and top example windows for manual inspection

**TDD Gate**
1. RED: mutation evaluator accepts nondeterministic sample order and fails reproducibility assertion.
2. GREEN: fixed-seed repeated eval gives identical mutation set and aggregate metrics.
3. RED: diagnostic artifact build fails when per-protein table is missing required columns.
4. GREEN: `pp_auc_i` distribution export and activation-based motif extraction outputs pass schema checks.

**Acceptance**
- H report contains complete metric panels plus diagnostic A/B artifacts for CNN rationality assessment.

### Task H6: Generate Decision Report + Freeze Recommendation

**Goal**
- Produce one explicit recommendation for downstream v2 integration.

**Actions**
1. Generate summary table:
   - performance (`pp_auc`, `pp_ap`, recall@K)
   - sensitivity (`delta_z_*`)
   - latency (`median_ms`, `p95_ms`)
   - diagnostics:
     - per-protein AUC distribution summary (quantiles + tail ratio)
     - activation-based motif plausibility summary
2. Map observed outcomes to V2 expected conclusion matrix in `doc/Epitope_Head_V2.md`.
3. Freeze one of:
   - keep ESM-2 route (E0)
   - move to local CNN route (E1)
   - move to shallow transformer route (E2)
4. Record freeze decision in both `PLAN.md` and `LOG.md`.

**TDD Gate**
1. RED: report builder with missing metrics should fail completeness check.
2. GREEN: complete metric artifacts produce valid report and recommendation block.

**Acceptance**
- Module H delivers a decision-ready and auditable ablation outcome.

**Module H Verification Gates (Release-Level)**
1. Contract gate: all 3 encoder backbones satisfy shared tensor interface.
2. Fairness gate: non-encoder constants are invariant across matrix runs.
3. Repro gate: same encoder+seed yields stable summary metrics within tolerance.
4. Coverage gate: ranking + mutation sensitivity + latency panels are all present.
5. Diagnostic gate: per-protein AUC distribution and CNN motif plausibility artifacts are generated and reviewable.
6. Decision gate: recommendation explicitly maps to V2 expected outcome matrix.

**Done Criteria**
- E0/E1/E2 matrix is completed with 3 seeds each.
- All planned metrics are available in machine-readable summary artifacts.
- One encoder route is frozen for the next v2 module with documented rationale.

## Module I: CNN Enhancement (V2 Module 2)

**Objective**
- Optimize CNN route from Module H by testing two targeted upgrades:
  1. loss-side hard example focus (margin objective)
  2. encoder-side multi-scale local context fusion
- Determine whether CNN is still the best practical choice under stricter AP-centric criteria.

**Inputs**
- Module H frozen conclusion and artifacts (E1 baseline runs, diagnostics A/B outputs)
- `outputs/manifests/protein_samples_strict.parquet`
- `outputs/manifests/splits/strict/{train_ids.txt,val_ids.txt,test_ids.txt}`

**Outputs**
- `outputs/ablation/cnn_enhance_v2/runs/<variant>/seed_<seed>/...`
- `outputs/ablation/cnn_enhance_v2/metrics/enhancement_raw.jsonl`
- `outputs/ablation/cnn_enhance_v2/metrics/enhancement_summary.csv`
- `outputs/ablation/cnn_enhance_v2/report/cnn_enhancement_report.md`

**Enhancement Matrix (Frozen)**

| ID | Variant | Change Type | Description |
|---|---|---|---|
| B0 | E1 baseline | control | current single-path dilated CNN + current InfoNCE |
| L1 | B0 + margin | loss | margin-based hard example learning (mixed objective first) |
| C1 | B0 + multi-scale | encoder | multi-branch CNN feature fusion (`k={3,5,9}`) |
| LC1 | L1 + C1 | combined | joint loss + encoder refinement |

**Frozen Comparison Constants (Stage I)**
1. Data profile and split: strict profile + Module D split (unchanged).
2. Training seeds per variant: `[42, 43, 44]`.
3. Span head, chunking, sampler and inference post-process remain unchanged unless explicitly listed in variant deltas.
4. Primary model-selection metric for Stage I: `pp_ap` (secondary: `pp_recall_50`, `pp_recall_100`; diagnostic: `pp_auc`).
5. Latency must remain deployment-friendly: no variant may exceed `2.0x` B0 latency at `L=1022`.

**Planned File Touchpoints**
- Modify: `epitope_head/training/losses.py`
- Modify: `epitope_head/training/trainer.py`
- Modify: `epitope_head/training/model.py`
- Modify/Create: `epitope_head/training/encoders.py`
- Modify: `epitope_head/configs/train.yaml`
- Modify/Create: `epitope_head/configs/model_ablation.yaml`
- Modify/Create: `scripts/train_v2_ablation.py`
- Modify/Create: `scripts/eval_encoder_ablation.py`
- Create: `tests/epitope_head/training/test_module_i_loss_contract.py`
- Create: `tests/epitope_head/training/test_module_i_encoder_contract.py`
- Create: `tests/epitope_head/training/test_module_i_matrix_contract.py`

### Task I0: Freeze Stage-I Contract and Baseline Anchor

**Goal**
- Ensure Stage-I improvements are measured against one stable CNN baseline (`B0`) without confounds.

**Actions**
1. Freeze `B0` as Module H E1 default config snapshot.
2. Freeze enhancement matrix (`B0/L1/C1/LC1`) and non-variable constants.
3. Freeze Stage-I decision metric order: `pp_ap` > recalls > stability > latency.

**TDD Gate**
1. RED: run metadata validator fails when variant id or baseline anchor is missing.
2. GREEN: stage-I config validator passes only when all frozen fields exist.

**Acceptance**
- Stage-I protocol is machine-checkable and reproducible.

### Task I1: Add Margin-Based Hard Example Loss (L1)

**Goal**
- Focus gradient on hard positive-negative pairs to improve AP-oriented ranking.

**Actions**
1. Extend loss config schema with:
   - `objective_mode`: `infonce | mixed_margin | margin_only`
   - `margin_m` (default `0.5`, grid `{0.3,0.5,0.8}`)
   - `hard_topk` (default `8`, grid `{4,8,16}`)
   - `lambda_margin` (default `0.5`, mixed mode only)
2. Implement pairwise margin term:
   - `g = z_pos - z_neg_hard`
   - `L_margin = mean(relu(m - g))`
3. Hard mining rule:
   - choose hardest negatives from current chunk logits per positive (`top-k` highest negative logits)
   - if `N_neg < k`, use all negatives (no failure)
4. Mixed objective default:
   - `L_total = L_infonce + lambda_margin * L_margin`
   - keep `margin_only` as ablation option, not default.

**TDD Gate**
1. RED: margin loss test fails for "all pair gaps already above margin" (expected zero).
2. GREEN: margin loss returns expected positive value when gap below margin.
3. RED: hard mining test fails when `hard_topk > N_neg`.
4. GREEN: fallback-to-all-negatives path passes with deterministic selection.

**Acceptance**
- Margin path is numerically stable and backward-compatible with existing loss/trainer API.

### Task I2: Trainer Integration for AP-Centric Selection

**Goal**
- Make training loop select checkpoints by `pp_ap` while preserving diagnostics.

**Actions**
1. Set Stage-I monitor metric to `pp_ap`.
2. Ensure monitor mode is `max` for AP metrics.
3. Log additional diagnostics:
   - `loss_margin`
   - `hard_neg_fraction_effective`
   - `best_to_last_pp_ap_drop`
4. Keep fallback behavior for missing monitor values deterministic.

**TDD Gate**
1. RED: checkpoint selector test fails when `pp_ap` is configured but not handled.
2. GREEN: best-checkpoint selection follows `pp_ap` monotonic improvements.

**Acceptance**
- AP-centric model selection works without breaking existing logging/checkpoint contracts.

### Task I3: Multi-Scale CNN Feature Fusion (C1)

**Goal**
- Improve CNN representation of mixed short/long local context while staying lightweight.

**Actions**
1. Add multi-branch conv block:
   - parallel kernels `k={3,5,9}`
   - branch outputs concatenated then projected back to `d_enc=256` via `1x1 conv/linear`
2. Keep residual path and mask alignment identical to baseline contract.
3. Constrain complexity:
   - parameter budget <= `1.5x` B0 encoder params
   - no change to encoder IO signature.

**TDD Gate**
1. RED: encoder contract test fails shape/length parity against baseline.
2. GREEN: multi-scale encoder returns contract-valid `(H, lengths)`.
3. RED: param-budget assertion fails when config exceeds limit.
4. GREEN: finalized config passes lightweight budget guard.

**Acceptance**
- Multi-scale CNN is contract-safe and remains in lightweight deployment budget.

### Task I4: Run Stage-I Matrix and Aggregate Results

**Goal**
- Produce fair evidence for B0/L1/C1/LC1 comparison.

**Actions**
1. Execute 4 variants x 3 seeds with identical data/split/runtime constraints.
2. Export per-run and aggregated metrics.
3. Preserve diagnostic A/B outputs from Module H for cross-checking whether gains are broad-based.

**TDD Gate**
1. RED: aggregation pipeline fails when one seed artifact is missing.
2. GREEN: summary generator handles complete matrix and marks missing runs explicitly.

**Acceptance**
- Stage-I matrix is complete and reproducible at artifact level.

### Task I5: Decision Rule and Promotion Criteria

**Goal**
- Promote only variants that improve AP materially without instability/latency regression.

**Actions**
1. Promotion criteria (must all pass vs B0):
   - mean `pp_ap` gain >= `0.01`
   - at least `2/3` seeds improve `pp_ap`
   - no unstable training signature (run divergence / pathological oscillation / abnormal early stop)
   - latency at `L=1022` <= `2.0x` B0
2. If none pass, keep B0 and close Stage-I with "no-promotion" decision.

**TDD Gate**
1. RED: promotion checker incorrectly accepts variant with only 1/3 seed improvement.
2. GREEN: checker enforces all criteria deterministically.

**Acceptance**
- Stage-I outputs an auditable promotion/no-promotion decision.

**Module I Verification Gates (Release-Level)**
1. Contract gate: L1/C1/LC1 preserve baseline training/inference interfaces.
2. Fairness gate: only intended loss/encoder deltas differ across variants.
3. Metric gate: AP/recall diagnostics are complete for all seeds.
4. Stability gate: no promoted variant shows invalid or unstable training pattern.
5. Deployment gate: latency and parameter budgets satisfy lightweight constraints.
6. Decision gate: promotion follows frozen rule set without manual override.

**Done Criteria**
- Stage-I matrix (`B0/L1/C1/LC1`) is complete with 3 seeds each.
- One variant (or B0 fallback) is frozen as next-stage CNN default.
- Decision rationale, evidence, and residual risks are recorded in `LOG.md`.

## 5. Current Open Decisions (Must Be Frozen Before/At Module Start)

1. Frozen:
   - H module uses `strict` profile only for primary ablation comparison.
2. Frozen:
   - seeds are `[42, 43, 44]` for each encoder.
3. Frozen:
   - E1 and E2 use `d_enc=256` to keep lightweight-encoder comparison controlled.
4. Frozen:
   - Stage-I primary metric is `pp_ap`; `pp_auc` becomes diagnostic metric.
5. Open:
   - Stage-I first run should start with `mixed_margin` only, or run `mixed_margin + margin_only` in parallel.

## 6. Risk Register (Execution-Level)

1. Confounding risk:
   - non-encoder settings drift across runs invalidates conclusions.
2. Capacity mismatch risk:
   - E1/E2 under-capacity could be misread as "pretraining required".
3. Evaluation cost risk:
   - mutation scans are expensive and may cause incomplete panels.
4. Interface regression risk:
   - encoder factory changes could break existing v1 E0 training/inference path.
5. Reproducibility risk:
   - nondeterministic mutation sampling undermines sensitivity claims.
6. Optimization drift risk:
   - introducing loss/encoder deltas can accidentally alter non-target training settings and invalidate comparisons.
7. Objective mismatch risk:
   - AP-centric optimization may improve `pp_ap` but hurt robustness/latency if constraints are not explicitly gated.

## 7. Codemap Annotation Policy (When Issues Appear)

When a blocking ambiguity or behavior change appears:
1. Add/append a diff entry in `doc/Epitope_Head_codemap.md` (do not silently edit assumptions).
2. Include:
   - change summary
   - rationale
   - impacted modules
   - config and artifact deltas
3. Update this `PLAN.md` module sections to reflect accepted diff only.

## 8. Log Governance (`LOG.md`)

### Log Location and Purpose

1. All planning and deployment progress logs are recorded in `LOG.md`.
2. `PLAN.md` defines strategy and module execution rules; `LOG.md` records time-ordered execution facts.
3. `LOG.md` is append-only. Existing entries are not rewritten; corrections are new entries linked to old IDs.

### Required Entry Schema (Scientific/Structured)

Each entry must include:
1. `log_id`: monotonic ID, format `L0001`, `L0002`, ...
2. `timestamp`: ISO-8601 with timezone (example: `2026-02-11T21:30:00-08:00`)
3. `type`: one of `PLAN_UPDATE`, `DECISION`, `RISK`, `VERIFICATION`, `CODEMAP_DIFF`
4. `module`: one of `GLOBAL`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`
5. `trigger`: why this entry was created (user request / gate failure / design change)
6. `change_summary`: one-line factual change
7. `rationale`: explicit reasoning or hypothesis behind the change
8. `artifacts`: affected files or outputs (exact paths)
9. `evidence`: command/test/check output summary (or `N/A` for plan-only changes)
10. `impact`: scope + risk level (`low`/`medium`/`high`) + confidence (`0.00-1.00`)
11. `status`: `open`, `in_progress`, `done`, `blocked`, `superseded`
12. `next_action`: exact next step
13. `refs`: optional links to codemap diff ID or plan section

### Logging Triggers

Create a new `LOG.md` entry whenever any of the following occurs:
1. A module plan section is added/expanded/changed.
2. Any open decision is frozen or changed.
3. A risk is discovered, severity changes, or mitigation is updated.
4. A TDD gate is executed for key module/function/pipeline checks.
5. A codemap diff is proposed/accepted/deprecated.

### Quality Rules

1. Factual first: no vague wording like "optimized" without evidence.
2. One entry = one primary event; avoid bundling unrelated updates.
3. Use absolute timestamps and exact paths.
4. If evidence is missing, status cannot be `done`.
5. For `CODEMAP_DIFF`, include diff id in `refs`.
6. In `evidence` blocks that report test results, must explicitly list which plan tasks/gates were **not covered by automated tests** (e.g. "NOT tested: A0 config load, A5 schema write — trivial logic covered by e2e run only"). Omitting untested items is not allowed.
