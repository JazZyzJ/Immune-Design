# Immune Design Inverse Folding Plan

> **For Claude:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Build an immunogenicity-aware inverse folding model whose core mechanism is position-dependent reference flow (emergent ordering). Infrastructure modules (K/L) provide the baseline and test set. The core contribution is implemented in Phase B (preparation) and Phase C (reference flow experiments).

**Architecture (restructured 2026-04-07):** Modules K and L are complete infrastructure. Module M is **superseded** (post-hoc resampling, not true generation-time steering). Module N is simplified to Level 1 post-hoc filter only. New Phase B (experiment preparation) and Phase C (core contribution: position-dependent DFM) are the active workstreams. See `doc/Reference_Flow_Derivation.md` for mathematical foundation.

**Tech Stack:** Python, PyTorch, byprot/DPLM, ESM-2 650M, GVP, Hydra, ESMFold, TM-align, NetMHCIIpan 4.3, pytest, JSON/CSV/FASTA/PDB artifacts.

---

## 1. Planning Rules (This File)

1. This file is the execution-facing plan for the inverse-folding stage, not a general project roadmap.
2. This file records module sequencing, frozen contracts, risk controls, and verification gates.
3. This file does not contain production code snippets.
4. Any change to inverse-folding assumptions must be reflected here before implementation proceeds.
5. All structural or immunogenicity claims must be traceable to the shared evaluation pipeline defined in this file.
6. Every material planning update must be appended to `LOG.md`.

## 2. Source of Truth

1. Inverse-folding v1 implementation proposal: `doc/Inverse_Folding_v1.md`
2. High-level scientific architecture and formulation: `doc/Immune_Design_Architecture_v2.md`
3. Shared epitope-head inference contract: `doc/Epitope_Head_v1.md`
4. Shared engineering ledger for epitope-head interface changes: `doc/Epitope_Head_codemap.md`
5. Active augmentation-stage plan that may later supply training/evaluation assets: `PLAN.md`

## 3. Global Milestones

### Milestone M0: Governance Green

- Outcome:
  - v1 scope is frozen
  - Level 2 guidance is explicitly separated from Phase 2/3 conditioning claims
  - artifacts, evaluation contracts, and release gates are defined

### Milestone M1: IF Baseline Green

- Outcome:
  - DPLM v1 + GVP adapter training path runs end-to-end on CATH
  - unconditional IF checkpoint and reproducibility metadata are materialized
  - baseline structural quality reaches at least smoke-level viability

### Milestone M2: Evaluation Green

- Outcome:
  - fixed test protein set is curated
  - shared evaluation pipeline runs end-to-end: PDB -> generate -> ESMFold/TM-align -> epitope head -> NetMHCIIpan
  - all later IF comparisons use the same test set and metric schema

### ~~Milestone M3: Guidance Validation Green~~ — SUPERSEDED (2026-04-07)

> Module M superseded. Resampling-based "guidance" is post-hoc selection, not generation-time steering. Replaced by Phase C core experiments.

### Milestone M4: Comparison Green — SIMPLIFIED (2026-04-07)

- Outcome:
  - Level 1 post-hoc baseline available (N1, done)
  - Core method (Phase C) evaluated on same test set
  - Comparison report: post-hoc filter vs reference flow (+ ablation tiers)

### Milestone M5: Experiment Preparation Green (NEW, 2026-04-07)

- Outcome:
  - PDB structures downloaded for test set proteins
  - h_i maps pre-computed for test set (both alleles) and CATH training set
  - End-to-end eval pipeline verified: generate → ESMFold → TM-align → head → NMP

### Milestone M6: Core Contribution Green (NEW, 2026-04-07)

- Outcome:
  - Position-dependent reference flow implemented and experimentally validated
  - Ablation structure: unguided baseline vs Tier 0 (sampling-only) vs full method
  - Paper-ready comparison data

## 4. Audit Conclusions (Frozen Into This Plan)

1. `doc/Inverse_Folding_v1.md` is coherent as a v1 engineering and validation stage if it is treated as infrastructure + Level 2 concept validation, not as the final Immune-Design claim.
2. DPLM v1 + GVP adapter is an acceptable enabling substrate because the architecture doc already positions richer conditioning and position-dependent reference flow as later phases.
3. The evaluation pipeline is not optional. Guidance results are only scientifically meaningful if structural fidelity and immunogenicity are measured through the same shared pipeline for all arms.
4. NetMHCIIpan must remain an independent external evaluator in v1. It cannot become the guidance signal or the optimization target, or the main validation claim becomes circular.
5. The `ProteinMPNN + DRAKES/DPO` wording is directionally useful but operationally under-specified because the method-model pairing is not yet frozen. This plan therefore treats Level 3 as a distinct module with an explicit compatibility freeze point before implementation.
6. The guidance algorithm in `doc/Inverse_Folding_v1.md` had one implementation ambiguity at the final selection step. This plan resolves v1 to risk-weighted candidate resampling, not weighted-logit averaging, because it is easier to audit and keeps the classifier-guidance semantics explicit.

## 5. Module Backlog (Ordered)

## Module K: DPLM v1 Infrastructure + Unconditional IF Baseline

**Objective**
- Stand up a reproducible unconditional inverse-folding baseline on top of DPLM v1 + GVP adapter, with enough artifact discipline that later guidance experiments are attributable to guidance rather than baseline instability.

**Inputs**
- CATH 4.3 training data
- DPLM source checkout / pinned commit
- pretrained checkpoint `airkingbd/dplm_650m`
- local or cluster GPU environment capable of adapter training

**Outputs**
- `outputs/if/manifests/cath_{train,val,test}.jsonl`
- `outputs/if/runs/dplm_v1_adapter/<run_id>/resolved_config.yaml`
- `outputs/if/runs/dplm_v1_adapter/<run_id>/checkpoints/*.ckpt`
- `outputs/if/runs/dplm_v1_adapter/<run_id>/baseline_validation.json`
- `outputs/if/runs/dplm_v1_adapter/<run_id>/failure_ledger.json`

**Scope Boundaries**
1. v1 baseline trains only the DPLM structure-injection adapter path; the pretrained sequence backbone remains frozen.
2. No hotspot conditioning, biased masking, FiLM conditioning, or position-dependent reference flow belongs in Module K.
3. Initial acceptance is based on unconditional IF behavior, not immunogenicity reduction.
4. Cluster submission scripts are optional in this stage plan; the required contract is a reproducible launcher and artifact schema.
5. Module K must not modify the existing epitope-head training or inference contracts.

**Planned File Touchpoints**
- Create: `inverse_folding/__init__.py`
- Create: `inverse_folding/configs/__init__.py`
- Create: `inverse_folding/configs/dplm_v1.yaml`
- Create: `inverse_folding/data/cath_manifest.py`
- Create: `inverse_folding/training/launcher.py`
- Create: `inverse_folding/training/checkpointing.py`
- Create: `scripts/prepare_if_cath.py`
- Create: `scripts/train_if_v1.py`
- Create: `tests/inverse_folding/test_module_k_contract.py`

### Task K0: Freeze Baseline Contract and Dependency Pins

**Goal**
- Remove ambiguity from the base IF substrate before any data or training work begins.

**Actions**
1. Pin DPLM/byprot source revision and pretrained checkpoint identifier.
2. Freeze trainable parameter set to the adapter path only.
3. Freeze initial smoke seed to `42`; extra seeds are a later decision, not a prerequisite for K-start.
4. Freeze artifact root and run directory schema under `outputs/if/runs/`.
5. Define a minimal config validator that refuses missing dataset root, checkpoint id, or adapter-trainability keys.

**TDD Gate**
1. RED: invalid baseline config passes schema validation.
2. GREEN: schema validator rejects incomplete configs and accepts the frozen v1 baseline profile only when required keys are present.

**Acceptance**
- Module K has a machine-checkable baseline contract.

### Task K1: Prepare CATH Manifest and Compatibility Checks

**Goal**
- Convert raw CATH inputs into a deterministic manifest that the training launcher can consume and audit.

**Actions**
1. Normalize train/val/test manifests with chain identifiers, sequence length, and structure-path provenance.
2. Validate coordinate completeness required by the chosen DPLM/GVP path.
3. Reject or quarantine malformed entries into a failure ledger instead of silently dropping them.
4. Emit manifest counts and rejection reasons as JSON for auditability.

**TDD Gate**
1. RED: malformed entries are silently accepted or silently dropped.
2. GREEN: manifest builder produces deterministic counts and explicit rejection reasons on fixture inputs.

**Acceptance**
- CATH ingestion is deterministic, auditable, and ready for training.

### Task K2: Build Training Launcher and Checkpoint Registry

**Goal**
- Run DPLM adapter training through a repo-local harness with reproducible configs and checkpoint bookkeeping.

**Actions**
1. Wrap the DPLM training invocation in a single launcher entrypoint.
2. Persist resolved config, seed, environment metadata, and checkpoint paths per run.
3. Support a local smoke mode on a tiny subset before full cluster execution.
4. Keep failure modes explicit: missing checkpoint, missing dataset, or invalid output path should fail fast.

**TDD Gate**
1. RED: launcher accepts invalid run arguments or writes incomplete run metadata.
2. GREEN: smoke launcher creates the expected run directory tree and metadata files on fixture inputs.

**Acceptance**
- Adapter training can be launched reproducibly and traced run-by-run.

### Task K3: Validate the Unconditional IF Baseline

**Goal**
- Establish that the baseline is viable enough to justify guidance work.

**Actions**
1. Measure held-out recovery, scTM, pLDDT, and foldability on a fixed validation subset.
2. Use tiered gates:
   - smoke viability gate: scTM `> 0.5` on a meaningful majority of pilot structures
   - target gate: scTM `> 0.8` on the intended report subset
3. Record failures explicitly, especially proteins that collapse structurally or show pathological recovery.
4. Publish a single validation summary artifact rather than scattered notebook outputs.

**TDD Gate**
1. RED: validation summary omits required structural fields or cannot distinguish smoke gate vs target gate.
2. GREEN: summary writer enforces the structural metric schema and gate reporting.

**Acceptance**
- There is a documented unconditional baseline with explicit viability evidence and known failure cases.

### Task K4: Freeze Baseline Release Package

**Goal**
- Make Module K outputs reusable by the downstream evaluation and guidance modules.

**Actions**
1. Freeze the checkpoint path and corresponding resolved config for downstream use.
2. Write a release note describing known limitations, metric values, and the exact artifact IDs to reuse.
3. Mark whether the baseline reached only smoke viability or full target acceptance.

**TDD Gate**
1. RED: downstream modules cannot locate a canonical checkpoint/config pair.
2. GREEN: baseline release metadata resolves to one canonical model artifact set.

**Acceptance**
- Module K exports one stable baseline package for Modules L/M/N.

## Module L: Shared Evaluation Pipeline

**Objective**
- Build the fixed test set and the reusable evaluation stack that measures structural fidelity and immunogenicity for every inverse-folding experiment.

**Inputs**
- Module K released checkpoint
- candidate PDB structures
- epitope-head checkpoint for `HLA-DRB1*07:01`
- NetMHCIIpan 4.3 runtime
- ESMFold and TM-align runtime

**Outputs**
- `outputs/if/test_set/test_proteins.csv`
- `outputs/if/test_set/test_proteins_summary.json`
- `outputs/if/test_set/wt_hotspot_maps.parquet`
- `outputs/if/eval/<run_id>/structural_metrics.csv`
- `outputs/if/eval/<run_id>/immunogenicity_head.csv`
- `outputs/if/eval/<run_id>/immunogenicity_nmp.csv`
- `outputs/if/eval/<run_id>/comparison.csv`
- `outputs/if/eval/<run_id>/summary.json`

**Scope Boundaries**
1. The same test set and metric schema must be reused by Modules M and N.
2. Test proteins must be screened for overlap against Module K training data before acceptance.
3. NetMHCIIpan remains an evaluation-only external scorer in this module.
4. Per-design metric caching is required so repeated analyses do not recompute ESMFold/TM-align unnecessarily.

**Planned File Touchpoints**
- Create: `inverse_folding/evaluation/test_set.py`
- Create: `inverse_folding/evaluation/esmfold_runner.py`
- Create: `inverse_folding/evaluation/tmalign.py`
- Create: `inverse_folding/evaluation/immunogenicity.py`
- Create: `inverse_folding/evaluation/aggregate.py`
- Create: `scripts/curate_if_test_set.py`
- Create: `scripts/evaluate_if.py`
- Create: `tests/inverse_folding/test_module_l_testset_contract.py`
- Create: `tests/inverse_folding/test_module_l_eval_contract.py`

### Task L0: Freeze Metric Contract and Test-Set Exclusion Rules

**Goal**
- Prevent metric drift or hidden dataset leakage before curation starts.

**Actions**
1. Freeze required structural metrics: `scTM`, `pLDDT`, `bb_RMSD`, `recovery`, `foldability`.
2. Freeze required sequence metrics: `diversity`, `mutation_count`.
3. Freeze required immunogenicity metrics:
   - epitope-head global risk and hotspot summaries
   - NetMHCIIpan `n_strong_binders`, `n_weak_binders`, `mean_best_rank`
4. Freeze exclusion policy against training overlap and low-quality structures.

**TDD Gate**
1. RED: evaluator can emit partial metric rows without failing.
2. GREEN: metric schema validator rejects incomplete outputs and enforces a fixed column set.

**Acceptance**
- Module L has one frozen evaluation contract shared by all later IF modules.

### Task L1: Curate Fixed Test Set and WT Hotspot Maps

**Goal**
- Create the fixed evaluation cohort once, with explicit inclusion/exclusion provenance.

**Actions**
1. Filter candidate PDB structures by length, chain quality, and practical evaluation constraints.
2. Run overlap checks against the CATH training substrate used in Module K.
3. Generate WT sequences and WT hotspot maps with the frozen epitope head.
4. Store per-protein provenance, rejection reason, and final acceptance flag.

**TDD Gate**
1. RED: train/test overlap slips through the curation filter on controlled fixtures.
2. GREEN: test-set builder deterministically flags overlaps and writes explicit rejection reasons.

**Acceptance**
- There is one fixed test set with precomputed WT hotspot maps and traceable curation provenance.

### Task L2: Structural Evaluation Wrappers

**Goal**
- Standardize structure prediction and structural comparison into reusable components.

**Actions**
1. Build an ESMFold wrapper that predicts structures from generated FASTA sequences.
2. Build a TM-align parser/wrapper that compares predicted structures to input backbones.
3. Cache structural outputs by `(protein_id, sequence_hash)` to avoid repeated work.
4. Emit parseable structural metric tables rather than free-form logs.

**TDD Gate**
1. RED: wrapper/parser pair accepts malformed outputs or emits schema-incompatible rows.
2. GREEN: fixture outputs are parsed into deterministic structural metric records.

**Acceptance**
- Structural evaluation is reusable, cached, and schema-stable.

### Task L3: Immunogenicity Evaluation Wrappers

**Goal**
- Standardize both internal and external immunogenicity scoring without circularity.

**Actions**
1. Reuse the existing epitope-head predictor as a frozen scorer for generated proteins.
2. Build a NetMHCIIpan aggregation wrapper across all `k in [12, 25]` windows.
3. Emit both raw per-window summaries and aggregated per-design metrics.
4. Keep the scorer interfaces deterministic and seed-independent.

**TDD Gate**
1. RED: NetMHCIIpan aggregation misclassifies boundary cases or drops windows.
2. GREEN: wrapper returns deterministic aggregated metrics on synthetic fixtures and preserves raw-window provenance.

**Acceptance**
- Both scorers are integrated under one stable evaluation interface.

### Task L4: Unified Evaluator and Report Assembly

**Goal**
- Turn the individual wrappers into the one evaluation entrypoint used by all downstream arms.

**Actions**
1. Implement a single evaluation CLI that takes PDBs, generated FASTAs, WT references, and scorer paths.
2. Produce the full artifact set: structural tables, immunogenicity tables, per-protein comparisons, and aggregate summary JSON.
3. Ensure reruns with the same inputs are cache-aware and deterministic.

**TDD Gate**
1. RED: end-to-end evaluator omits one required artifact or writes mismatched row counts across tables.
2. GREEN: smoke evaluation on fixtures produces the complete artifact bundle with aligned row counts.

**Acceptance**
- Module L provides the shared measurement substrate for the entire inverse-folding stage.

## Module M: Classifier Guidance Validation (Level 2) — SUPERSEDED (2026-04-07)

> **Status**: SUPERSEDED. The implemented "guidance" (M0-M3) is candidate resampling at the output level — post-hoc filtering with soft weights, not generation-time steering. Replaced by Phase C (position-dependent reference flow). Code preserved in `inverse_folding/guidance/`; scoring bridge may be reusable.

**Objective** *(original, preserved for LOG traceability)*
- Validate whether inference-time epitope-aware steering can reduce immunogenicity without retraining the inverse-folding model.

**Inputs**
- Module K released checkpoint
- Module L fixed test set and evaluator
- frozen epitope-head checkpoint for `HLA-DRB1*07:01`

**Outputs**
- `outputs/if/guidance/<run_id>/generated/<protein_id>.fasta`
- `outputs/if/guidance/<run_id>/guidance_config.yaml`
- `outputs/if/guidance/<run_id>/guidance_summary.csv`
- `outputs/if/guidance/<run_id>/pareto_summary.json`
- `outputs/if/guidance/<run_id>/failure_analysis.md`

**Scope Boundaries**
1. Module M is inference-time only. No DPLM retraining or hotspot-conditioned fine-tuning is allowed here.
2. Guidance is applied on predicted clean-sequence candidates, not on noisy intermediate tokens directly.
3. The v1 default guidance grid is `eta in {0, 0.5, 1, 2, 5, 10}` with candidate count `K = 8`, unless this plan is explicitly updated.
4. v1 selection rule is risk-weighted candidate resampling. Weighted-logit averaging is out of scope for this version.
5. The scientific claim is validated against NetMHCIIpan and structural fidelity, not only against the head used for guidance.

**Planned File Touchpoints**
- Create: `inverse_folding/guidance/classifier_guidance.py`
- Create: `inverse_folding/guidance/sampler_bridge.py`
- Create: `scripts/run_if_guidance_sweep.py`
- Create: `tests/inverse_folding/test_module_m_guidance_contract.py`
- Modify: `scripts/evaluate_if.py`

### Task M0: Freeze Guidance Contract

**Goal**
- Eliminate algorithmic ambiguity before sampler integration begins.

**Actions**
1. Freeze `eta` grid, candidate count, and selection rule.
2. Freeze whether guidance operates on global risk only in v1.
3. Freeze failure policy for degenerate candidate sets, identical candidates, and numerical underflow/overflow in weights.
4. Define the minimal required provenance per design: `eta`, seed, candidate count, selected risk, selected sequence hash.

**TDD Gate**
1. RED: guidance config accepts unsupported selection rules or missing provenance fields.
2. GREEN: guidance validator accepts only the frozen v1 contract and rejects incomplete configs.

**Acceptance**
- Module M has one auditable guidance algorithm contract.

### Task M1: Build Epitope-Head Scoring Bridge for Sampling

**Goal**
- Score candidate clean sequences inside the sampler without changing the epitope-head contract.

**Actions**
1. Add a bridge layer that converts sampled sequences into the predictor input contract.
2. Support micro-batching so K-candidate scoring remains tractable.
3. Ensure bridge outputs match direct predictor calls on the same sequence.

**TDD Gate**
1. RED: bridge scores differ from direct predictor scores on identical fixture sequences.
2. GREEN: bridge and direct scorer produce matching global-risk outputs within numerical tolerance.

**Acceptance**
- Candidate scoring is reusable, auditable, and interface-stable.

### Task M2: Implement Reweighting and Candidate Selection

**Goal**
- Realize the v1 guidance algorithm in a way that is numerically stable and experimentally attributable.

**Actions**
1. Compute stabilized weights from `exp(-eta * R)`.
2. Handle duplicate candidates, equal-risk candidates, and extreme `eta` values explicitly.
3. Sample the next clean candidate under the risk-weighted distribution and continue denoising from that selected path.
4. Persist candidate-level scoring provenance so post-hoc audits can reconstruct why a sequence was chosen.

**TDD Gate**
1. RED: weight computation becomes unstable or candidate provenance is lost.
2. GREEN: deterministic test fixtures demonstrate correct weight normalization, duplicate handling, and candidate selection provenance.

**Acceptance**
- Guidance selection is stable, reproducible, and fully attributable.

### Task M3: Run `eta` Sweep and Assemble Pareto Results

**Goal**
- Measure whether stronger guidance decreases external immunogenicity while preserving acceptable structure.

**Actions**
1. Run guided generation across the frozen `eta` grid on the fixed test set.
2. Evaluate every design through Module L without changing metrics or thresholds between arms.
3. Aggregate per-protein and global Pareto summaries:
   - `scTM` vs `delta risk`
   - `mutation_count` vs `delta risk`
4. Distinguish structural collapse from genuine de-immunization gains.

**TDD Gate**
1. RED: sweep runner writes incomplete per-eta artifacts or loses the link between generated sequences and evaluation rows.
2. GREEN: sweep summary contains aligned per-eta metrics and provenance for every evaluated protein.

**Acceptance**
- Module M produces a reproducible Pareto analysis for Level 2 guidance.

### Task M4: Failure Analysis and Decision Readout

**Goal**
- Translate sweep outputs into a concrete decision about whether classifier guidance is a credible route.

**Actions**
1. Analyze hotspot reduction at originally high-risk positions.
2. Quantify mutation burden and edit efficiency versus immunogenicity gain.
3. Partition failures into:
   - structural collapse
   - no-risk-change despite guidance
   - high internal gain but no external NetMHCIIpan gain
4. Produce one report-ready summary for the next design decision.

**TDD Gate**
1. RED: failure analysis cannot distinguish external vs internal scorer disagreement.
2. GREEN: report assembler groups outcomes into explicit failure modes with counts.

**Acceptance**
- Module M yields a decision-ready answer to "is inference-time classifier-guided epitope steering worth pursuing further?"

## Module N: Comparison Baselines (Level 1 Only)

> **Scope change (2026-04-04)**: Level 3 (DRAKES/DPO) dropped. Focus shifted to core reference flow contribution. Only Level 1 post-hoc filter retained as baseline. Level 3 can be added as reviewer response if requested.

**Objective**
- Provide the minimal comparison arm needed to interpret reference flow results.

**Inputs**
- Module L fixed test set and evaluator
- Module K released checkpoint (same as used by reference flow)

**Outputs**
- `outputs/if/baselines/level1_filter/<run_id>/...`
- `outputs/if/baselines/comparison_report.md`

**Scope Boundaries**
1. Level 1 post-hoc filter is the only mandatory baseline.
2. Every baseline must use the same test proteins, candidate budget, and evaluation schema.
3. Sampling-only position-dependent schedule (Tier 0 ablation) is part of the reference flow module, not Module N.

**Planned File Touchpoints**
- Done: `inverse_folding/baselines/level1_filter.py`
- Done: `scripts/run_if_level1_filter.py`
- Done: `scripts/submit_if_level1_filter.slurm`

### Task N1: Implement the Level 1 Post-Hoc Filter Baseline — DONE (2026-04-02)

**Status**: implemented, 11 tests passing.

**Actions**
1. Sample K candidates from the unconditional IF baseline.
2. Score candidates with head + NetMHCIIpan evaluation stack.
3. Select argmin global_risk candidate.

**Acceptance**
- There is a valid Level 1 comparator. ✓

### Task N2: Level 3 Learned Baseline — DROPPED

**Decision (2026-04-04)**: Level 3 (DRAKES/DPO) explicitly dropped from v1 scope.

**Rationale**:
1. Core contribution is position-dependent reference flow, not beating RL methods.
2. Implementation effort (~weeks) is better spent on making the core method work.
3. The ablation structure (post-hoc filter vs sampling-only vs full reference flow) is sufficient for paper.
4. Can be added as reviewer response if requested — DRAKES code is available in `DRAKES/`.

### Task N3: Assemble the Comparison Report

**Goal**
- Compare Level 1 post-hoc filter vs our method on a shared substrate.

**Actions**
1. Report structural quality (scTM), immunogenicity change (head + NetMHCIIpan), and mutation burden.
2. Include Tier 0 (sampling-only) ablation as intermediate point.
3. Highlight the Pareto front advantage of position-dependent training.

**Acceptance**
- Module N produces a scientifically interpretable comparison for the paper.

## Phase B: Experiment Preparation (NEW, 2026-04-07)

**Objective**: Bridge from infrastructure (K/L done) to core experiments (Phase C). All tasks are mechanical batch jobs.

**Inputs**: Module K checkpoint, Module L test set parquets, frozen epitope head checkpoints, CATH training data.

### Task B1: Download PDB Structures for Test Set

- Download PDB files for all test set proteins into `work/immune-design/if_test_set/pdbs/`
- Source: PDB IDs from test_proteins parquets
- Validation: count matches test set size, files parse without error

### Task B2: Pre-compute h_i Maps for Test Set

**Goal**
- Emit per-residue epitope-head signal `h_i` for every WT test protein (both alleles) so Phase C can feed `g(h_i)` into the sampling-time schedule (`doc/Reference_Flow_Derivation.md §4.3`) and into H1/H3/H5 analyses (`doc/Reference_Flow_Derivation.md §6`).

**Design principle (frozen here, 2026-04-22)**
1. **Storage–policy decoupling.** Persist the neutral physical quantity (log-mean-exp logits per residue) in two forms — uncentered and per-protein median-centered — with `clamp=none`. Every downstream policy (`g(h)` form, corpus-level normalization, binarization threshold for hotspot/non-hotspot, shuffle control) is deferred to Phase C read-time. Re-computing ~20k encoder forward passes is far more expensive than re-applying a policy to the stored arrays.
2. **One row per (protein, allele)**, wide layout (array columns) — sampling/training both consume full h maps per protein.

**Inputs**
- Test set parquets:
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet` (3,141 rows)
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet` (3,140 rows)
- Sequence source: in-row `sequence` column.
- Epitope head checkpoints (frozen):
  - 0701: `run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt`
  - 0401: `run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt`
- Inference config preset: `epitope_head/configs/inference.yaml` — locked at `center_method=median`, `clamp=none`, `min_k=12`, `max_k=25`, `chunking.enabled=true` (context_len=1022, stride=512, margin=32, stitch_mode=per_residue_stitch, enable_reliability=true).

**Outputs** (per allele: one parquet + one sidecar JSON)
- `work/immune-design/if_test_set/h_maps/h_maps_DRB1_07_01.parquet`
- `work/immune-design/if_test_set/h_maps/h_maps_DRB1_07_01.meta.json`
- `work/immune-design/if_test_set/h_maps/h_maps_DRB1_04_01.parquet`
- `work/immune-design/if_test_set/h_maps/h_maps_DRB1_04_01.meta.json`

**Parquet schema (wide, one row per protein)**

| Column | Type | Semantics |
|--------|------|-----------|
| `protein_id` | str | primary key; matches `protein_id` in source test_proteins parquet |
| `allele` | str | canonical header form, e.g. `"HLA-DRB1*07:01"` |
| `sequence_length` | int32 | `L = len(sequence)` |
| `h_raw` | list[float32] (length `L`) | log-mean-exp of covering-window logits, uncentered, unclamped (== `debug.h_raw` from `InferencePredictor.predict_protein`) |
| `h_processed` | list[float32] (length `L`) | per-protein median-centered `h_raw`, `clamp=none` (== `residue_hotspot` from `predict_protein` under the frozen inference config) |
| `global_risk` | float32 | log-mean-exp over all windows (== `global_risk` from `predict_protein`) |
| `n_windows` | int32 | number of (start, k) windows scored |

**Sidecar metadata JSON** (one per parquet)
```
run_id                      # h_maps_{allele_tag}_{YYYYMMDD_HHMMSS}
allele
head_checkpoint_path        # absolute cluster path
head_checkpoint_metadata    # {manifest_version, config_hash, diff_ids_applied}
inference_cfg               # resolved dict actually used
source_dataset              # absolute path to test_proteins parquet
source_dataset_rowcount     # int
git_commit                  # git rev-parse HEAD
timestamp                   # ISO-8601 with offset
n_proteins_total            # == source_dataset_rowcount
n_proteins_completed        # int
n_proteins_failed           # int
failures                    # list[{protein_id, reason}]
device                      # "cuda" | "cpu"
wall_clock_seconds          # float
```

**Planned File Touchpoints**
- Create: `scripts/precompute_h_maps.py` (shared CLI for B2 + B3)
- Create: `scripts/submit_precompute_h_test.slurm`
- Create: `inverse_folding/evaluation/h_maps.py` (load + schema validator used by Phase C and by TDD fixtures)
- Create: `tests/inverse_folding/test_h_maps_contract.py`
- Modify: `doc/SCRIPTS.md` (register new script + SLURM under Phase B)

**CLI contract**
- Required args: `--head-checkpoint`, `--inference-config`, `--source parquet|jsonl`, `--input <path>`, `--id-column`, `--sequence-column`, `--allele <header-form>`, `--output-parquet`, `--output-meta`.
- Optional: `--device cuda|cpu` (default cuda), `--window-batch-size` (default 4096, passed to `enumerate_and_score`), `--fail-pct-threshold` (default 0.05), `--checkpoint-every` (periodic parquet snapshot; default 500 proteins), `--resume-from` (path to a prior partial parquet).
- Must print every resolved hyperparameter to stdout at job start (per `feedback_print_hyperparams`).

**SLURM**
- `scripts/submit_precompute_h_test.slurm` follows `scripts/submit_cnn_enhance.slurm` template: QOS `gpu-short` (2 passes, one per allele), `--output`/`--error` route to `logs/`, `HF_HUB_OFFLINE=1`, openfold `PYTHONPATH` per `feedback_slurm_convention`.

**Failure handling**
1. Sequence contains non-canonical AA (outside the 20 + pad) → skip protein, append `{protein_id, reason: "non_canonical_aa"}` to `failures`.
2. Empty sequence → skip, log.
3. GPU OOM on one protein → retry that protein on CPU once; on second failure, log and continue.
4. Cumulative failure rate > `--fail-pct-threshold` → abort job with exit code 2 before overwriting the final parquet; partial parquet kept under `.partial/` for inspection.

**TDD Gate**
1. RED: output parquet is accepted when `len(h_raw) != sequence_length`, when `h_processed - median(h_raw) != h_raw` (within float tol), or when sidecar metadata is missing any required key.
2. GREEN: a 2-protein WT fixture round-trips through CLI → parquet → `inverse_folding/evaluation/h_maps.py` loader, where (a) each row satisfies `len(h_raw) == len(h_processed) == sequence_length`, (b) `global_risk` matches a direct `InferencePredictor.predict_protein(seq)` call within 1e-5, and (c) sidecar JSON contains every required key with non-empty `head_checkpoint_metadata`.

**Acceptance**
- Both allele parquets materialize under `h_maps/` with row count matching the source parquet minus logged failures.
- `n_proteins_completed / n_proteins_total >= 0.95` for each allele.
- Downstream loader `load_h_maps(path)` asserts schema + `len(h_raw) == sequence_length` for every row.

---

### Task B3: Pre-compute h_i Maps for CATH Training Set

**Goal**
- Extend B2 to the CATH training split so Phase C Tier 1 (C2, retrain with position-dependent forward process) has the same `h_i` signal available during training.

**Scope decision (frozen here, 2026-04-22)**
- **Alleles**: 0701 first, 0401 second. 1501 skipped (no test set exists; multi-allele H4 pairing requires matching test set coverage).
- **Split**: train split only. Val/test splits of CATH are internal DPLM validation; `h_i` for them is not required by Phase C (which evaluates on Module L test set, not CATH test).

**Inputs**
- CATH training data: `work/immune-design/cath_4.3/chain_set.jsonl`
- Split selection: `work/immune-design/cath_4.3/chain_set_splits.json["train"]` (16,631 chain ids)
- Sequence source: `seq` field in each JSONL record
- Chain id: `CATH` field (canonical primary key)
- Epitope head checkpoints: same as B2.
- Inference config: same as B2.

**Outputs**
- `work/immune-design/cath_4.3/h_maps/h_maps_cath_DRB1_07_01.parquet` (+ `.meta.json`)
- `work/immune-design/cath_4.3/h_maps/h_maps_cath_DRB1_04_01.parquet` (+ `.meta.json`)

**Parquet schema**: identical to B2 (`protein_id` column holds the CATH chain id).

**Sidecar metadata JSON**: identical keys, **plus** corpus-level stats (computed after the full run):
```
corpus_h_raw_mean           # float — mean of all per-residue h_raw, pooled across proteins
corpus_h_raw_std            # float — std, same pooling
corpus_h_raw_n_residues     # int  — total residue count used in stats
```
These are candidates for Phase C optional corpus-level normalization. Not applied to the stored arrays.

**Planned File Touchpoints**
- Reuse: `scripts/precompute_h_maps.py` with `--source jsonl --splits-json <path> --split train --id-field CATH --sequence-field seq`.
- Create: `scripts/submit_precompute_h_cath.slurm`
- Modify: `doc/SCRIPTS.md` (register SLURM under Phase B).

**SLURM**
- Walltime estimate: 16,631 × ~0.5–2 s/protein on A100 ≈ 2–9 h per allele (encoder forward dominates; varies with protein length).
- QOS: `gpu-medium` (3d) for safety margin.
- **Resumability required**: `--checkpoint-every 500` must write an intermediate parquet so preemption does not waste >500 proteins of work.

**Failure handling**
- Identical to B2, with `--fail-pct-threshold=0.02` (stricter; CATH is cleaner than Tier-2-sourced test set).
- Non-standard residues in `seq` field (CATH uses `-` / `X`): treat `-` as hard fail (record with no sequence is malformed); treat `X` as skip if count > 5% of chain length, else substitute with `X` token and proceed (chunker + tokenizer handle `X`).

**TDD Gate**
1. RED: corpus stats (`corpus_h_raw_mean`, `corpus_h_raw_std`) are written before the parquet is complete, or are computed over fewer residues than `sum(sequence_length)`.
2. GREEN: on a 3-protein CATH fixture, corpus stats match `np.concatenate([row.h_raw for row in parquet]).mean() / .std()` within 1e-6, and resume-from-checkpoint produces the same final parquet bit-for-bit as a single-shot run.

**Acceptance**
- Both allele parquets materialize, each with ≈16,631 rows.
- `n_proteins_completed / n_proteins_total >= 0.98` per allele.
- Corpus stats written in sidecar JSON.

---

### What Phase C inherits from B2/B3 — explicit mapping to hypotheses

See `doc/Reference_Flow_Derivation.md §6`.

| Hypothesis | Artifact consumed | How |
|---|---|---|
| H1 edit localization | Test set `h_processed` + analysis-time threshold | stratify edit counts at hotspot vs non-hotspot |
| H2 Pareto dominance | Test set `h_raw`/`h_processed` → `g(h_i)` | feeds sampling loop (§4.3) |
| H3 shuffle control | Test set `h_*` arrays | analysis-time permutation `π` |
| H4a/H4b retrain vs sample-only | CATH `h_*` for training + test set `h_*` for sampling | C2 training schedule uses CATH h; C1 sample-only uses test h only |
| H5 hotspot entropy | Test set `h_*` + analysis-time threshold | stratify per-position `H(p_θ(x_1^i \| x_t))` |

**Deliberately deferred (not frozen in Phase B)**:
- `g(h)` functional form (linear clamp / sigmoid / power — `doc/Reference_Flow_Derivation.md §7.1`): Phase C hyperparameter sweep.
- Corpus-level normalization `(μ_corpus, σ_corpus)`: may be applied at Phase C read-time using sidecar stats from B3; decision belongs to C1/C2 ablation.
- Binarization threshold for H1/H5 stratification: analysis-time, not at storage time.
- Multi-allele aggregation (§7.3 Q3): out of v1 scope.

### Task B4: End-to-End Eval Pipeline Integration

- Integrate existing evaluation code into one CLI: generate → ESMFold → TM-align → head → NMP
- Reuse relevant pieces from Module L evaluation wrappers
- Verify on 2-3 test proteins before full-scale runs

**Acceptance**: all B tasks complete → Phase C can begin experiments.

---

## Phase C: Core Contribution — Position-Dependent Reference Flow (expanded 2026-04-23)

**Objective**: Implement and validate position-dependent discrete flow matching as an immunogenicity-aware generative mechanism.

> **Scope of this plan section**: code contract + hyperparameter schema for the sampler / training machinery. **Experimental arm selection and hyperparameter values are not frozen here** — every tunable knob is exposed via YAML so the experimenter chooses arms, base schedules, amplification forms, `c` values, step counts, and shuffle controls without touching code.

**Planned experiment tiers** (all supported by the same code path once C1 lands; arm selection = YAML config):
- **C0**: DPLM native sampler (baseline) — Task C0
- **C1 (Tier 0)**: DFM sampler + position-dependent schedule (includes `g ≡ 1` uniform-schedule control via config) — Task C1
- **C2 (Tier 1)**: Retrain DPLM with position-dependent forward — STUB
- **C3 (Tier 2)**: FiLM + CFG conditioning — STUB

**Mathematical reference**: `doc/Reference_Flow_Derivation.md` §4 (sampling), §7 (design choices).

### Task C0: DPLM Native-Sampler Baseline

**Goal**: Drive Module K's trained adapter under DPLM's native sampler on the full test set. No algorithmic novelty; a reproducible driver + SLURM.

**Inputs**
- Module K checkpoint: `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt`
- Test set parquet (per allele) + B1 backbone PDBs
- Allele selection

**Outputs** (per run)
- `work/immune-design/if_phase_c/c0/<allele_tag>/<run_id>/generated.parquet` (columns: `protein_id`, `design_idx`, `sequence`, `seed`, `wall_seconds`)
- `generated.fasta`
- `run_config.yaml` (resolved)
- `manifest.json` (git SHA, checkpoint digest, timestamp)

**Planned File Touchpoints**
- Create: `scripts/run_if_phase_c0.py`
- Create: `scripts/submit_if_phase_c.slurm` (`MODE=native`)
- Modify: `doc/SCRIPTS.md` (register under Phase C)

**CLI contract**
- Required: `--checkpoint`, `--test-set-parquet`, `--pdb-root`, `--allele`, `--output-root`.
- Optional: `--n-designs-per-protein` (default 1), `--seed` (default 42), `--max-iter` (default 10, DPLM native), `--temperature` (default 1.0), `--device cuda|cpu` (default cuda).
- Echo all resolved hyperparameters to stdout at start (per `feedback_print_hyperparams`).

**TDD Gate**
1. RED: driver writes designs without linking to source `protein_id`, or silently swallows DPLM sampler errors.
2. GREEN: 3-protein fixture produces `generated.parquet` where (a) every row's `protein_id` matches a row in the input test set, (b) `len(sequence) == input sequence_length`, (c) manifest contains non-empty checkpoint digest + git SHA.

**Acceptance**
- `generated.parquet` contains `test_rows × n_designs_per_protein` rows minus logged failures.

---

### Task C1: Sampling-Only Position-Dependent DFM

**Goal**: Implement the §4.3 DFM sampling algorithm as a standalone module that consumes a frozen Module K checkpoint + B2 h_maps and generates sequences under a **fully config-driven** per-position unmasking schedule. No retraining. No model weight modification.

**Frozen design principles (2026-04-23)**
1. **Every experimental knob is a YAML field.** Same code path runs uniform-schedule control (`g ≡ 1`), linear-clamp / sigmoid / power amplification, three base schedule forms, variable step count, H3 h-shuffle — all by config.
2. **Sampler is denoiser-agnostic.** Takes a callable `denoiser(x_t, t, struct) → logits[L, V]` and applies §4.3. DPLM-specific wiring lives in the CLI driver, not in the sampler module (so C2/C3 reuse the module with a different denoiser).
3. **Resolved config is the arm identifier.** Every run serializes the fully-materialized config alongside its output; no arm label is hard-coded in the code.

**Hyperparameter schema** (YAML, validated by `inverse_folding/reference_flow/config.py`; all fields required unless marked optional)

```
sampler:
  n_steps: int                    # DFM step count N
  seed: int                       # sampling RNG seed
  temperature: float              # logits scaling before Categorical (optional, default 1.0)
  n_designs_per_protein: int      # independent samples per input

schedule:
  base_form: str                  # "linear" | "cosine" | "cubic"
                                  #   linear: κ_base(t) = t
                                  #   cosine: κ_base(t) = 1 − cos(π/2 · t)
                                  #   cubic : κ_base(t) = 3t² − 2t³

amplification:
  form: str                       # "constant_one" | "linear_clamp" | "sigmoid" | "power"
  c: float                        # max amplification strength
                                  #   required unless form == "constant_one"
                                  #   linear_clamp: g(h) = 1 + c · max(0, h − mu)
                                  #   sigmoid:      g(h) = 1 + c · σ(kappa · (h − mu))
                                  #   power:        g(h) = 1 + c · max(0, (h − mu))^p
  mu: float                       # threshold h₀ (optional, default 0.0 — assumes centered h)
  kappa: float                    # sigmoid sharpness (required iff form == "sigmoid")
  p: float                        # power exponent (required iff form == "power")
  h_source: str                   # "h_raw" | "h_processed" | "h_normalized_corpus"
                                  #   h_normalized_corpus uses (h_raw − μ_corpus) / σ_corpus
                                  #   from B3 sidecar JSON (must be passed via --h-corpus-stats)
  g_max_cap: float                # upper clamp on g(h) to guard tail outliers
                                  # (optional, default 20.0)

h_shuffle:                        # H3 control mode; permutes h per-protein before sampling
  enabled: bool                   # optional, default false
  seed: int                       # required iff enabled; acts as a base seed,
                                  # effective permutation is derived per
                                  # (protein_id, design_idx)
```

**Preset config scaffolds** (starting points only; the experimenter sweeps values, switches forms, and mixes modes — nothing in code depends on these file names):
- `inverse_folding/reference_flow/configs/c1_null.yaml`     — `form=constant_one` (DFM + uniform schedule)
- `inverse_folding/reference_flow/configs/c1_linclamp.yaml` — `form=linear_clamp, mu=0.0, h_source=h_processed`
- `inverse_folding/reference_flow/configs/c1_sigmoid.yaml`  — `form=sigmoid, mu=0.0`
- `inverse_folding/reference_flow/configs/c1_power.yaml`    — `form=power`
- `inverse_folding/reference_flow/configs/c1_shuffle.yaml`  — `form=linear_clamp` + `h_shuffle.enabled=true`

**Algorithm** (direct transcription of `doc/Reference_Flow_Derivation.md §4.3`)

```
Input:  denoiser p_θ(· | x_t, t, struct)
        h[1..L]                      (from B2)
        g(·)                         (from amplification config)
        κ_base(·), κ_base'(·)        (from schedule config)
        N, seed, temperature

Derive: g_i    = g(h[i])             (optionally clamped to g_max_cap)
        κ_i(t) = κ_base(t)^{g_i}
        κ_i'(t)= g_i · κ_base(t)^{g_i − 1} · κ_base'(t)

rng ← seed
x   ← [m, m, ..., m]                 (length L)
for k = 0, 1, ..., N − 1:
    t  ← k / N
    dt ← 1 / N
    logits ← p_θ(x, t, struct) / temperature    # [L, V]
    for i where x[i] == m:
        rate_i   ← κ_i'(t) / (1 − κ_i(t))
        p_unmask ← min(1, rate_i · dt)
        if rng.Bernoulli(p_unmask):
            x[i] ← rng.Categorical(softmax(logits[i]))
# final guard: force-unmask any residual mask tokens with the last logits
for i where x[i] == m:
    x[i] ← rng.Categorical(softmax(logits[i]))
return x
```

**Boundary / numerical notes** (implementation must honor)
- `t = 0`: `κ_i(0) = 0`, `1 − κ_i = 1`; `κ_i'(0) = g_i · 0^{g_i−1} · κ_base'(0)` — zero for `g_i > 1`, `κ_base'(0)` for `g_i = 1`. No division-by-zero.
- `t → 1`: rate diverges by construction (§2.2). `min(1, rate·dt)` clamps it.
- Compute `κ_i(t)` via `exp(g_i · log κ_base(t))` with a floor on `log κ_base(t)` (suggest `−50`) to prevent underflow.
- `g_max_cap` guards against extreme `h` outliers pushing `g_i` into numerically fragile ranges. Log a warning when the cap triggers.

**Inputs**
- Module K checkpoint (same path as C0)
- Test set parquet + backbone PDBs (B1)
- B2 h_maps parquet matching the allele
- B3 sidecar JSON (required iff `amplification.h_source == h_normalized_corpus`)
- YAML config matching the schema above

**Outputs** (per run)
- `work/immune-design/if_phase_c/c1/<allele_tag>/<run_id>/generated.parquet` (columns: `protein_id`, `design_idx`, `sequence`, `seed`, `wall_seconds`)
- `generated.fasta`
- `run_config.yaml` (resolved — every default materialized)
- `manifest.json` (git SHA, checkpoint digest, h_maps source path, h_source choice, schedule form, amplification params, timestamp)
- `trajectories/<protein_id>.parquet` (optional; written iff `--save-trajectories`; columns: `step`, `t`, `unmasked_mask` [bool, L], `token_argmax` [int, L])

**Planned File Touchpoints**
- Create: `inverse_folding/reference_flow/__init__.py`
- Create: `inverse_folding/reference_flow/schedule.py` (base-schedule functions + derivatives)
- Create: `inverse_folding/reference_flow/amplification.py` (g(h) functions)
- Create: `inverse_folding/reference_flow/sampler.py` (sampler class consuming a `denoiser` callable)
- Create: `inverse_folding/reference_flow/config.py` (dataclass + YAML validator)
- Create: `inverse_folding/reference_flow/configs/{c1_null,c1_linclamp,c1_sigmoid,c1_power,c1_shuffle}.yaml`
- Create: `scripts/run_if_phase_c1.py`
- Create: `scripts/submit_if_phase_c.slurm` (`MODE=reference_flow`)
- Create: `tests/inverse_folding/test_reference_flow_schedule.py`
- Create: `tests/inverse_folding/test_reference_flow_amplification.py`
- Create: `tests/inverse_folding/test_reference_flow_sampler.py`
- Create: `tests/inverse_folding/test_reference_flow_config.py`
- Modify: `doc/SCRIPTS.md` (register under Phase C)

**CLI contract**
- Required: `--checkpoint`, `--test-set-parquet`, `--pdb-root`, `--h-maps-parquet`, `--allele`, `--config <yaml>`, `--output-root`.
- Optional: `--h-corpus-stats <B3 sidecar json>` (required iff `h_source == h_normalized_corpus`), `--n-designs-per-protein` / `--seed` / `--n-steps` (override matching config fields if present), `--device cuda|cpu` (default cuda), `--save-trajectories` (default false), `--fail-pct-threshold` (default 0.05), `--resume-from <run_id>`.
- Echo every resolved hyperparameter to stdout at start (per `feedback_print_hyperparams`).

**SLURM**
- `scripts/submit_if_phase_c.slurm` follows `scripts/submit_cnn_enhance.slurm` template; default QOS `gpu-medium`; `--output`/`--error` route to `logs/`.
- `MODE=native` launches Task C0; `MODE=reference_flow` launches Task C1.
- Incremental parquet writes required (granularity is coder's call) so preemption does not lose a full test-set pass.

**Failure handling**
1. h_maps row missing for a test protein → skip protein, log `{protein_id, reason: "missing_h_map"}`.
2. h array length ≠ sequence length → hard abort with exit code 2 (data-consistency bug, not a skip).
3. NaN in logits at any step → abort that design, log `(protein_id, design_idx, step, t)`, continue with next design.
4. GPU OOM on one protein → retry single design on CPU once; on second failure, log and continue.
5. Cumulative skip rate > `--fail-pct-threshold` → abort before overwriting final parquet; partial parquet retained under `.partial/`.

**TDD Gate**
1. RED:
   - `schedule` returns `κ(0) ≠ 0` or `κ(1) ≠ 1` for any form.
   - `amplification` returns `g(h) < 1` for any valid `(form, params, h)` triple.
   - `sampler` emits sequences with mask tokens remaining, or with length ≠ input L.
   - `config` validator accepts a config with `form ≠ constant_one` but missing `c`, or `form == sigmoid` missing `kappa`, or `h_shuffle.enabled == true` missing `seed`.
2. GREEN:
   - `schedule`: `κ(0)=0, κ(1)=1, κ'(t)>0` on `(0,1)` over a dense grid for all three base forms; analytical `κ'` vs finite-difference approximation agree within 1e-4.
   - `amplification`: `g(h) ≥ 1` over `h ∈ [−10, 10]` for all four forms; `form == constant_one` returns exactly 1 for every h; sigmoid/power satisfy documented asymptotic sentinels.
   - `sampler`: on a stub denoiser returning uniform logits, a small fixture run (a) produces mask-free outputs of correct length, (b) with fixed seed + `form == constant_one` is deterministic (two calls → identical outputs), (c) with `form == linear_clamp, c > 0` and a stratified h input, top-`g_i`-quartile positions have a higher mean unmask-step index than bottom-`g_i`-quartile positions (directional sign required; exact threshold left to the implementer).
   - `config`: accepts all 5 preset YAMLs; rejects each missing-required-field case in RED.

**Acceptance**
- All TDD gates green.
- `scripts/run_if_phase_c1.py --config inverse_folding/reference_flow/configs/c1_null.yaml` completes one allele end-to-end under `--fail-pct-threshold=0.05` and emits the full artifact bundle.
- `inverse_folding/reference_flow/sampler.py` consumes `denoiser(x_t, t, struct)` as a plain callable — no import of DPLM-specific internals inside the sampler module.

---

### Task C2: Retrain with Position-Dependent Forward Process — STUB

Deferred. Spec to be added after C1 produces baseline + ablation runs and H4a/b signal is inspected.

### Task C3: Explicit h_i Conditioning (FiLM + CFG) — STUB

Deferred. Spec to be added after C2.

---

## 6. Current Open Decisions (Must Be Frozen Before/At Module Start)

1. Frozen:
   - v1 execution order: K → L → Phase B → Phase C. Module M superseded (2026-04-07).
2. Frozen:
   - NetMHCIIpan remains an external evaluator, not the guidance signal
3. Frozen (superseded 2026-04-07):
   - ~~Level 2 guidance uses risk-weighted candidate resampling~~ — Module M superseded; no longer relevant
4. Frozen:
   - Module K baseline is adapter-only training on top of frozen DPLM sequence priors
5. Frozen (2026-04-07):
   - Test set size: ~3,141 proteins (0701) / ~3,140 proteins (0401), assembled on cluster
6. Partially frozen (2026-04-23):
   - C1 (sampling-only) code contract + hyperparameter schema frozen in §Task C1. Every experimental knob — `g(h)` form, `c` / `mu` / `kappa` / `p`, `h_source`, base schedule, `n_steps`, h-shuffle — is YAML-driven; experimenter controls arm selection and sweep values without code edits.
   - C2 (retrain) and C3 (FiLM + CFG) specs still TBD.
7. Frozen (2026-04-04):
   - Level 3 dropped from v1 scope. DRAKES code available in `DRAKES/` for potential reviewer response.

## 7. Risk Register (Execution-Level)

1. Environment drift risk:
   - DPLM/byprot and ESMFold dependencies may drift enough that environment pinning becomes part of the critical path
2. Compute risk:
   - Module K adapter training is materially more expensive than epitope-head training and may force a smoke-first workflow
3. Leakage risk:
   - poor overlap screening could place CATH-like training structures into the evaluation cohort
4. Evaluation bottleneck risk:
   - ESMFold and TM-align throughput can dominate runtime unless caching is enforced early
5. Circularity risk:
   - if internal head scores are treated as the only success metric, guidance results become self-confirming rather than externally validated
6. Baseline comparability risk:
   - Level 3 comparisons dropped from v1 scope (2026-04-04). Level 1 post-hoc filter is the only comparison arm.

## 8. Codemap Annotation Policy (When Issues Appear)

When a blocking ambiguity or behavior change appears:
1. If the change touches shared epitope-head interfaces, add or append a diff entry in `doc/Epitope_Head_codemap.md`.
2. If the change is inverse-folding-only and no dedicated IF codemap exists yet, record the accepted decision in `PLAN_IF.md` and `LOG.md` first.
3. If inverse-folding implementation expands beyond one stable package tree, open a dedicated `doc/Inverse_Folding_codemap.md` before major refactors.
4. Do not silently change assumptions in implementation without reflecting them in the accepted plan.

## 9. Log Governance (`LOG.md`)

### Log Location and Purpose

1. All planning and deployment progress logs for both stage plans are recorded in `LOG.md`.
2. `PLAN.md` and `PLAN_IF.md` define strategy and module execution rules; `LOG.md` records time-ordered execution facts.
3. `LOG.md` is append-only. Existing entries are not rewritten; corrections are new entries linked to old IDs.

### Required Entry Schema (Scientific/Structured)

Each entry must include:
1. `log_id`: monotonic ID, format `L0001`, `L0002`, ...
2. `timestamp`: ISO-8601 with timezone
3. `type`: one of `PLAN_UPDATE`, `DECISION`, `RISK`, `VERIFICATION`, `CODEMAP_DIFF`
4. `module`: one of `GLOBAL`, `K`, `L`, `M` (superseded), `N`, `PHASE_B`, `PHASE_C` for IF-stage events, while older A-J modules remain valid for epitope-head stage events
5. `trigger`: why this entry was created
6. `change_summary`: one-line factual change
7. `rationale`: explicit reasoning or hypothesis
8. `artifacts`: affected files or outputs
9. `evidence`: command/test/check summary, or `N/A` for plan-only updates
10. `impact`: scope + risk level + confidence
11. `status`: `open`, `in_progress`, `done`, `blocked`, `superseded`
12. `next_action`: exact next step
13. `refs`: optional links to codemap diff ID or plan section

### Logging Triggers

Create a new `LOG.md` entry whenever any of the following occurs:
1. A module plan section is added, expanded, or changed.
2. Any open decision is frozen or changed.
3. A risk is discovered, severity changes, or mitigation is updated.
4. A TDD gate is executed for key inverse-folding module/function/pipeline checks.
5. A codemap diff is proposed, accepted, or deprecated.

### Quality Rules

1. Factual first: no vague wording without evidence.
2. One entry = one primary event.
3. Use exact paths and absolute timestamps.
4. If evidence is missing, status cannot be `done` for verification entries.
5. If automated tests do not cover a claimed verification event, the uncovered items must be stated explicitly in `evidence`.
