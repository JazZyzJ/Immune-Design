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

- Run frozen epitope head on all test set proteins (both alleles)
- Output: per-protein `residue_hotspot` arrays → `wt_hotspot_maps.parquet`
- These h_i maps are used as conditioning signal and for evaluation

### Task B3: Pre-compute h_i Maps for CATH Training Set

- Run frozen epitope head on ~16k CATH training proteins
- Output: `cath_hotspot_maps.parquet` (protein_id, residue_idx, h_i)
- Required for Phase C Tier 1 (position-dependent training)
- Can run in parallel with B1/B2

### Task B4: End-to-End Eval Pipeline Integration

- Integrate existing evaluation code into one CLI: generate → ESMFold → TM-align → head → NMP
- Reuse relevant pieces from Module L evaluation wrappers
- Verify on 2-3 test proteins before full-scale runs

**Acceptance**: all B tasks complete → Phase C can begin experiments.

---

## Phase C: Core Contribution — Position-Dependent Reference Flow (NEW, 2026-04-07)

> **This section is intentionally left as a stub.** Implementation details (g(h_i) form, training recipe, architecture modifications) are under active discussion in the Thinker track. Mathematical foundation is in `doc/Reference_Flow_Derivation.md`.

**Objective**: Implement and validate position-dependent discrete flow matching as an immunogenicity-aware generative mechanism.

**Planned experiment tiers** (details TBD):
- **C0**: Unguided DPLM baseline inference on test set (control arm)
- **C1 (Tier 0)**: Sampling-only position-dependent schedule — no retraining
- **C2 (Tier 1)**: Retrain DPLM with position-dependent forward process
- **C3 (Tier 2)**: Explicit h_i conditioning (architecture TBD)

**Comparison structure**: post-hoc filter (N1) vs C0 vs C1 vs C2/C3

**Key references**: `doc/Reference_Flow_Derivation.md`, `doc/Immune_Design_Architecture_v2.md`

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
6. Open:
   - Phase C implementation details (reference flow architecture, g(h_i) form, training recipe) — to be determined
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
