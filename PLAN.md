# Epitope Head Data Augmentation Plan

> **For Claude:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Build a train-only, split-safe data augmentation stage for Epitope Head that can incorporate external biological priors into new training samples, with the current v1.1 path focused on NetMHCIIpan-guided mutation disruption plus controlled runtime exposure for `HLA-DRB1*07:01`.

**Architecture:** This plan governs the full Data augmentation stage, not only one external predictor. In the current iteration, the active augmentation route is an offline NetMHCIIpan-based mutation pipeline layered on top of the strict-profile data/training stack: select strict train proteins -> verify WT positives -> enumerate candidate single mutants -> retain high-confidence disruptions -> write a mutation registry with label-hygienic affected-span metadata -> apply controlled train-only runtime replacement with `p_aug` over base train proteins. Optional materialized augmented `ProteinSample` artifacts remain available for audit/debug or fallback ablation, but they are no longer the default training path. Validation/test data remain untouched.

**Tech Stack:** Python, pandas/Parquet/JSON, existing `ProteinSample`/`ProteinEntry` pipeline, NetMHCIIpan 4.3 standalone, pytest.

---

## 1. Planning Rules (This File)

1. This file is the execution-facing plan for the Data augmentation stage, not a general project roadmap.
2. This file records module sequencing, frozen contracts, risk controls, and verification gates.
3. This file does not contain production code snippets.
4. Any change to augmentation assumptions must be reflected here before implementation proceeds.
5. Every material planning update must be appended to `LOG.md`.

## 2. Source of Truth

1. Current augmentation execution notes and v1.1 usage refinement: `doc/Data_augmentation.md`
2. Original augmentation proposal and scientific rationale: `doc/NetMHCIIpan_mutation_augmentation.md`
3. Current data/training contracts: `doc/Epitope_Head_v1.md`
4. Engineering map and diff ledger: `doc/Epitope_Head_codemap.md`
5. Archived baseline modular plan: `PLAN_epi_v1.md`
6. Archived encoder-ablation plan: `PLAN_enco_abl.md`

## 3. Global Milestones

### Milestone M0: Governance Green

- Outcome:
  - augmentation scope is frozen
  - train-only isolation is explicit
  - artifact contracts and release gates are defined

### Milestone M1: Pilot Green

- Outcome:
  - pilot subset confirms NetMHCIIpan WT-agreement rate, disruption yield, and throughput
  - threshold policy and batching strategy are validated before full offline run

### Milestone M2: Registry Green

- Outcome:
  - full mutation registry is generated for strict-train proteins
  - registry rows are deduplicated, schema-valid, and label-hygienic

### Milestone M3: Registry and Runtime-View Green

- Outcome:
  - mutation registry is generated with enough metadata to drive runtime replacement
  - optional materialized augmented train-only `ProteinSample` artifact can still be written for audit/debug
  - base train artifact is left unchanged
  - validation/test artifacts are unchanged

### Milestone M4: Training Integration Green

- Outcome:
  - training can apply train-only runtime replacement over base train proteins without expanding train-set cardinality
  - augmentation exposure (`p_aug`) is controllable and reproducible
  - model/loss architecture remains unchanged

### Milestone M5: Evaluation Green

- Outcome:
  - augmentation stage is auditable through counts, thresholds, and artifact lineage
  - downstream experiments can attribute gains to augmentation rather than split leakage or schema drift

## 4. Module Backlog (Ordered)

## Module J: NetMHCIIpan Mutation Augmentation (Data Augmentation Stage)

**Objective**
- Convert high-confidence NetMHCIIpan-predicted presentation disruptions into a train-only mutation registry and controlled runtime augmentation policy, so the model sees "single mutation flips positive -> negative" cases without letting near-duplicate augmented samples dominate training.

**Inputs**
- `outputs/manifests/protein_samples_strict.parquet`
- `outputs/manifests/splits/strict/train_ids.txt`
- `outputs/manifests/splits/strict/val_ids.txt`
- `outputs/manifests/splits/strict/test_ids.txt`
- NetMHCIIpan 4.3 standalone binary/runtime

**Outputs**
- `outputs/manifests/mutation_pilot_strict.json`
- `outputs/manifests/mutation_registry_strict.parquet`
- `outputs/manifests/mutation_registry_strict_summary.json`
- `outputs/manifests/protein_samples_strict_aug_train.parquet` (optional audit/debug artifact)
- `outputs/manifests/protein_samples_strict_aug_train_summary.json` (optional audit/debug artifact)

**Scope Boundaries**
1. This stage only augments the strict profile.
2. Augmentation is train-only. Validation/test are never mutated and never mixed with augmented entries.
3. This stage does not change encoder, scorer, or base loss function.
4. NetMHCIIpan is used only as an offline filter for disruption candidates, not as a source of positives.
5. First-stage mutation scan is limited to substitutions inside known positive spans, not arbitrary protein-wide mutation search.
6. Stage-J is chunking-agnostic: its registry and any optional materialized artifacts are defined at full-protein level before any optional downstream chunking policy is applied.
7. Stage-J default training policy is runtime `p_aug` replacement over base train proteins, not static union of all augmented entries.

**Major Draft Corrections (Frozen Into This Plan)**
1. The draft's `wt_EL_Rank < 2%` / `>5%` wording leaves a `2%-5%` ambiguity gap. This plan freezes `2%-5%` as `skip_uncertain`.
2. The draft suggests applying augmentation in `Dataset.__getitem__`, but the current training path stores `ChunkSample`, not mutable `ProteinEntry`. This plan keeps Stage-J chunking-agnostic and applies runtime mutation at the epoch-level train-entry view before chunk expansion, not inside chunk-level `__getitem__`.
3. The draft records only the seed disrupted span. This is insufficient when a single mutation affects multiple known positives on the same source protein. This plan requires `affected_spans` to be recomputed across all original known positives on that mutant protein.
4. Unique `AUG::...` sample IDs remain mandatory for optional materialized debug artifacts, but runtime replacement is not allowed to expand train-set cardinality by turning every retained mutation into a persistent additional train row.
5. Runtime-selected mutants that would leave `remaining_positive_count == 0` are excluded from the default `p_aug` training pool, because the current trainer skips empty-positive chunks and they would consume batch budget without contributing gradient.
6. Static union of all augmented train rows is deprecated as the default Stage-J training policy because it overexposes near-duplicate mutants and can drown out the base IEDB signal.

**Planned File Touchpoints**
- Create: `epitope_head/data/netmhciipan_mutation.py`
- Create: `epitope_head/data/materialize_augmented_samples.py`
- Modify: `epitope_head/training/datamodule.py`
- Modify: `epitope_head/configs/data.yaml`
- Create: `scripts/run_mutation_pilot.py`
- Create: `scripts/build_mutation_registry.py`
- Create: `scripts/materialize_augmented_train_set.py`
- Create: `tests/epitope_head/data/test_module_j_registry_contract.py`
- Create: `tests/epitope_head/data/test_module_j_materialization_contract.py`
- Create: `tests/epitope_head/training/test_module_j_train_integration_contract.py`

### Task J0: Freeze Scientific Contract and Tooling Assumptions

**Goal**
- Eliminate ambiguous semantics before any offline computation starts.

**Actions**
1. Freeze tool/runtime:
   - NetMHCIIpan version: `4.3`
   - mode: `EL`
   - allele: `HLA-DRB1*07:01`
2. Freeze threshold policy:
   - `wt_EL_Rank < 2%` -> eligible WT positive
   - `2% <= wt_EL_Rank <= 5%` -> `skip_uncertain`
   - `wt_EL_Rank > 5%` -> reject from augmentation
   - `mut_EL_Rank > 20%` and `delta_rank > 18` -> eligible disruption
3. Freeze split policy:
   - source proteins come only from strict-train split
   - no registry rows are produced for val/test proteins
4. Freeze artifact policy:
   - registry is one row per retained mutation candidate
   - augmented train artifact is materialized separately from base strict parquet
   - Stage-J does not assume any specific chunking mode at artifact-generation time

**TDD Gate**
1. RED: config/tooling validator fails on missing version/mode/threshold keys.
2. GREEN: validator passes only on complete Stage-J config.

**Acceptance**
- Stage-J assumptions are explicit, machine-validated, and auditable.

### Task J1: Pilot Feasibility Run

**Goal**
- Validate throughput and yield before full-scale NetMHCIIpan execution.

**Actions**
1. Select deterministic pilot subset:
   - `5-10` strict-train proteins
   - stratify by `positive_count` and `sequence_length`
2. Measure:
   - WT-agreement rate (`wt_EL_Rank < 2%`)
   - candidate disruption yield
   - runtime per protein / per mutant batch
3. Record pilot summary with enough detail to freeze full-run batching.

**TDD Gate**
1. RED: pilot summary writer fails when required counters are absent.
2. GREEN: pilot output contains reproducible counts, timing, and threshold buckets.

**Acceptance**
- Pilot gives enough evidence to proceed or revise thresholds before full run.

### Task J2: WT Verification on Strict-Train Positives

**Goal**
- Filter augmentation seeds to only those WT positives that NetMHCIIpan agrees are strong binders.

**Actions**
1. Load only strict-train proteins from `protein_samples_strict.parquet`.
2. For each known positive span, score the WT protein using that span's original `pep_len`.
3. Classify every positive into:
   - `keep_wt_confirmed`
   - `skip_uncertain`
   - `reject_wt_disagree`
4. Preserve deterministic provenance:
   - source protein id
   - span tuple `(start_0b, end_0b, pep_len)`
   - `support_n`
   - WT rank

**TDD Gate**
1. RED: WT classification test fails on boundary values (`1.99`, `2.00`, `5.00`, `5.01`).
2. GREEN: threshold bucket assignment is deterministic and exact.

**Acceptance**
- Only WT-confirmed seeds proceed to mutation enumeration.

### Task J3: Mutation Enumeration and Batched Scoring

**Goal**
- Enumerate candidate single mutants and score them efficiently enough for offline execution.

**Actions**
1. Enumerate only in-span substitutions:
   - for each eligible WT span
   - for each residue position in `[start_0b, end_0b)`
   - for each of `19` alternative amino acids
2. Deduplicate candidate mutants by:
   - `source_protein_id`
   - `mut_pos_0b`
   - `mut_aa`
3. Batch NetMHCIIpan calls by source protein and peptide length where possible, instead of one process launch per mutant.
4. Store raw mutant ranks and seed-span provenance before filtering.

**TDD Gate**
1. RED: dedupe test fails when two seed spans generate the same mutant candidate.
2. GREEN: candidate enumeration yields one canonical row per unique mutant.

**Acceptance**
- Candidate mutation generation is deterministic, deduplicated, and batching-aware.

### Task J4: Label Hygiene via Affected-Span Recalculation

**Goal**
- Ensure each retained mutant carries a clean label set, not just a single disrupted seed span.

**Actions**
1. For every retained mutant candidate, recompute all original known positive spans on that same source protein whose evaluation depends on the mutated sequence.
2. Construct `affected_spans` as the full set of known positives that transition out of the positive regime under this mutation.
3. Reject candidates with empty `affected_spans`.
4. Record both:
   - seed span that triggered enumeration
   - full affected-span set used for train artifact materialization

**TDD Gate**
1. RED: overlapping-positive fixture leaves an actually disrupted second span in the positive set.
2. GREEN: affected-span recomputation removes all disrupted known positives deterministically.

**Acceptance**
- Retained mutants have label-hygienic positive removal sets.

### Task J5: Mutation Registry Artifact

**Goal**
- Write a compact, queryable registry that downstream runtime replacement and optional materialization can consume directly.

**Actions**
1. Store registry as Parquet with one row per retained mutation:
   - `source_protein_id`
   - `mut_pos_0b`, `wt_aa`, `mut_aa`
   - `mutant_protein_id`
   - `seed_span_json`
   - `affected_spans_json`
   - `source_positive_count`
   - `remaining_positive_count`
   - `runtime_train_eligible`
   - `wt_el_rank_seed`
   - `mut_el_rank_seed`
   - `delta_rank_seed`
   - provenance/status fields
2. Write companion summary JSON with:
   - threshold values
   - total source proteins
   - WT-confirmed spans
   - enumerated mutants
   - retained mutants
   - runtime statistics
3. Enforce unique key:
   - `(source_protein_id, mut_pos_0b, mut_aa)`

**TDD Gate**
1. RED: registry validator accepts duplicate mutation keys.
2. GREEN: duplicate rows are rejected and summary counts remain consistent.

**Acceptance**
- Registry is schema-valid, deduplicated, and sufficient to drive runtime replacement plus any optional materialization path.

### Task J6: Freeze Registry-Primary Contract and Optional Materialization

**Goal**
- Keep the mutation registry as the primary Stage-J artifact, while retaining optional materialization for audit/debug or fallback ablation use.

**Actions**
1. Keep the mutation registry as the canonical Stage-J source of truth for training-time mutation application.
2. Ensure the registry contains enough metadata to support runtime replacement without needing a pre-expanded augmented parquet:
   - mutation identity
   - `affected_spans`
   - `remaining_positive_count`
   - runtime eligibility flag
3. Allow an optional materialization path that creates one augmented row per retained mutation with:
   - unique sample-key `protein_id` such as `AUG::<source_protein_id>::<pos><wt>><mut>`
   - mutated `protein_seq`
   - updated `positives_json = base positives - affected_spans`
   - source provenance in `metadata_json` including original `source_protein_id`
4. Preserve base schema columns if optional materialization is used:
   - `protein_id`, `allele`, `protein_seq`, `positives_json`, `sequence_length`
   - count fields and metadata fields consistent with current `ProteinSample` contract
5. Do not rewrite:
   - `protein_samples_strict.parquet`
   - split id files

**TDD Gate**
1. RED: runtime integration depends on a pre-expanded augmented parquet or optional materialized rows reuse original `protein_id`.
2. GREEN: registry alone is sufficient for runtime replacement; optional materialized rows remain schema-valid, provenance-tagged, and split-safe.

**Acceptance**
- Stage-J remains registry-primary, with optional materialized artifacts isolated from evaluation data.

### Task J7: Training Integration

**Goal**
- Apply augmentation at train time without expanding train-set cardinality or contaminating validation/test.

**Actions**
1. Build the train epoch view from base strict-train proteins only; validation/test remain base-only.
2. For each source protein with runtime-eligible registry rows:
   - with probability `1 - p_aug`, keep the original protein
   - with probability `p_aug`, sample one retained mutation weighted by `delta_rank_seed` (linear proportional: a mutation with Δrank=80% is sampled ~4× more often than one with Δrank=20%) and apply it to the source entry
3. Apply the mutation before chunk expansion so Stage-J stays chunking-agnostic and the downstream chunk builder sees one coherent protein view per epoch.
4. Freeze default runtime exposure policy:
   - initial `p_aug = 0.20`
   - optional follow-up sweep may test `p_aug = 0.30`
   - do not exceed `p_aug = 0.50` in the first pass
5. Seed runtime mutation selection deterministically from training seed + epoch + source protein identity (or an equivalent deterministic mapping).
6. Exclude runtime-selected mutations with `remaining_positive_count == 0` from the default training pool.
7. Keep static union of all materialized augmented rows only as a fallback ablation/debug path, not the default Stage-J training route.
8. Keep model/loss architecture unchanged in Stage-J.
9. If negative-sampler knobs are co-tuned downstream (`neg_ratio`, `hard_neg_offset_range`, etc.), log them as a separate experimental factor so runtime `p_aug` can still be attributed cleanly.

**TDD Gate**
1. RED: enabling augmentation changes base train protein cardinality, mutates validation/test, samples runtime-ineligible mutations, or uses uniform sampling when weighted is configured.
2. GREEN: train epoch view keeps base protein count fixed, val/test remain untouched, same seed+epoch reproduces the same mutation selections, runtime-selected mutants always retain at least one positive, and mutation sampling probabilities are proportional to `delta_rank_seed`.

**Acceptance**
- Augmentation is train-only, reproducible, and exposure-controlled without duplicating the training set.

### Task J8: Release Gates and Audit Summary

**Goal**
- Prove the augmentation stage is usable, safe, and attributable.

**Actions**
1. Produce final stage summary:
   - WT-confirmed rate
   - retained mutation count
   - affected-span statistics
   - runtime-eligible mutation count
   - configured `p_aug` and observed effective augmented fraction
   - optional augmented train artifact size, if materialization is used
2. Verify split safety and schema compatibility.
3. Log residual risks and open follow-ups.

**TDD Gate**
1. RED: final summary passes with missing key stage counters.
2. GREEN: summary completeness checker requires all frozen counters and artifact digests.

**Acceptance**
- Stage-J outputs are ready for downstream training experiments and audit.

**Module J Verification Gates (Release-Level)**
1. Split gate: no augmented rows appear in val/test data paths.
2. Threshold gate: WT and mutant thresholds are applied exactly, including gray-zone skip behavior.
3. Label gate: `affected_spans` are recomputed, not assumed from seed span only.
4. Positive-survival gate: runtime-selected mutants retain at least one positive span and do not generate no-gradient training views.
5. Exposure gate: train protein cardinality remains fixed and effective augmented exposure is logged against configured `p_aug`.
6. Schema gate: registry is sufficient for runtime training and any optional materialized artifact is loadable by the existing `ProteinEntry` path.
7. Provenance gate: every runtime or materialized augmented sample can be traced back to source protein, mutation, and seed span.
8. Repro gate: pilot/full registry generation and per-epoch runtime mutation selection are deterministic under fixed config.

**Done Criteria**
- Strict-train mutation registry exists and is schema-valid.
- Runtime `p_aug` training policy is frozen and reproducible.
- Any optional train-only augmented `ProteinSample` artifact remains split-safe.
- Existing validation/test artifacts remain unchanged.
- Stage-J summary and evidence are recorded in `LOG.md`.

## 5. Current Open Decisions (Must Be Frozen Before/At Module Start)

1. Frozen:
   - augmentation source is `strict` profile only
2. Frozen:
   - augmentation applies to strict-train proteins only; val/test stay pristine
3. Frozen:
   - ambiguous WT band `2%-5%` is skipped, not relabeled
4. Frozen:
   - first-stage scan is limited to in-span substitutions
5. Frozen:
   - Stage-J is chunking-agnostic and does not depend on whether chunking is later enabled or disabled
6. Frozen:
   - first full run uses `threshold + per-source topN` selection rather than keep-all
7. Frozen:
   - initial full-run selection uses `topN = 64` per source protein, ranked by `delta_rank_seed` descending; if fewer than 64 mutants pass thresholds, keep all passing mutants
8. Frozen:
   - Stage-J default training path uses runtime `p_aug` replacement over base train proteins, not static union of all augmented rows
9. Frozen:
   - initial runtime exposure uses `p_aug = 0.20`; first follow-up may test `0.30`, but not `> 0.50`
10. Frozen:
   - runtime mutation selection pool excludes retained mutants with `remaining_positive_count == 0`
11. Frozen:
   - static union of all materialized augmented rows is a fallback/debug ablation only, not the default Stage-J route
12. Frozen:
   - runtime mutation choice per protein is weighted by `delta_rank_seed` (linear proportional sampling), not uniform; rationale: mutations with higher disruption confidence are cleaner negatives and should be sampled more frequently, while borderline mutations (Δrank just above 18%) remain in the pool but with proportionally lower probability

## 6. Risk Register (Execution-Level)

1. Throughput risk:
   - full mutant enumeration may be expensive enough that batching strategy must be validated by pilot first
2. Label hygiene risk:
   - one mutation can affect multiple known positives; recording only the seed span would leave false positives in the augmented sample
3. Split contamination risk:
   - reusing base `protein_id` or modifying base parquet could leak augmented data into val/test
4. Distribution skew risk:
   - proteins with many positives may generate disproportionate numbers of mutants and dominate training
5. External-tool reproducibility risk:
   - NetMHCIIpan runtime/config differences can silently change retained mutation sets
6. Augmentation overexposure risk:
   - static union of all retained mutants can dominate training because augmented samples are near-duplicates of source proteins; runtime `p_aug` mitigates this but must be logged per epoch
7. Zero-positive runtime risk:
   - some retained mutants may remove all positives on a source protein and would consume train budget without contributing gradient unless excluded from the runtime pool
8. Attribution risk:
   - changing augmentation usage and negative-sampler settings in the same experiment would make gains or regressions hard to attribute

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
4. `module`: one of `GLOBAL`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`
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
6. In `evidence` blocks that report test results, must explicitly list which plan tasks/gates were **not covered by automated tests**. Omitting untested items is not allowed.
