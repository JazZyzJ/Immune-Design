# Epitope Head v1 Modular Deployment Plan

> **For Claude:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Build a reproducible Epitope Head v1 pipeline that outputs window logits, residue hotspot map, and global protein risk from full-length proteins.

**Architecture:** Follow `doc/Epitope_Head_codemap.md` as implementation contract and `doc/Epitope_Head_v1.md` as objective/assumption spec. Execute work by module (A->F + governance), with strict data contracts, artifact versioning, and split-safe evaluation. For critical logic, enforce RED->GREEN->REFACTOR checks via temporary tests before accepting behavior.

**Tech Stack:** Python, PyTorch, ESM-2 encoder, Parquet/JSON artifacts, pytest.

---

## 1. Planning Rules (This File)

1. This file is a living deployment plan, updated incrementally by module, not one-shot completion.
2. Default update unit is one module at a time, only after user explicitly requests that module.
3. This file focuses on:
   - global architecture awareness
   - implementation sequencing
   - contract checkpoints
   - risk control and verification gates
4. This file does not contain production code snippets.
5. If a requirement/assumption changes, record it in `doc/Epitope_Head_codemap.md` using diff-ledger style before downstream module updates.

## 2. Source of Truth

1. Primary contract: `doc/Epitope_Head_codemap.md`
2. Objective and modeling rationale: `doc/Epitope_Head_v1.md`
3. This `PLAN.md` is execution-facing and must stay consistent with both files.

## 3. Global Milestones

### Milestone M0: Governance Green

- Outcome:
  - module sequence and dependencies frozen for current iteration
  - unresolved decisions listed and explicitly owned
  - verification gates defined

### Milestone M1: Data Pipeline Green (Stage A-D)

- Outcome:
  - reproducible artifacts for span records, sequence-joined records, protein samples, and train/val/test splits
  - manifest includes counts, filtering policy, and long-protein loss accounting

### Milestone M2: Training Pipeline Green (Stage E)

- Outcome:
  - one end-to-end training run completes without schema/shape errors
  - primary sanity metrics recorded (`logit_gap`, pos-vs-neg separation)

### Milestone M3: Inference Pipeline Green (Stage F)

- Outcome:
  - per-protein prediction JSON exported with required fields:
    - `window_logits`
    - `residue_hotspot`
    - `global_risk`
    - `meta`

### Milestone M4: Evaluation & Registry Green

- Outcome:
  - run registry, resolved config snapshot, and checkpoint metadata are complete
  - comparability constraints across runs are enforceable

### Milestone M5: V2 Encoder Ablation Green (Stage H)

- Outcome:
  - encoder ablation matrix (`E0/E1/E2`) is executed under controlled constants
  - decision-ready comparison report links performance, mutation sensitivity, and latency

## 4. Module Backlog (Ordered)

## Module A: SpanRecord Build (Data Stage A)

**Objective**
- Build normalized positive span records from raw TSV under v0/v1 constraints, with dual SA tables for strict baseline and balanced experiment.

**Inputs**
- `data/mhc_if_v2.tsv`

**Outputs**
- `outputs/manifests/span_records_strict.parquet` (primary v1 baseline)
- `outputs/manifests/span_records_balanced.parquet` (auxiliary experiment set)

**Scope Boundaries**
1. Module A only produces normalized positive span records.
2. No FASTA join, no sequence match, no split logic in this module.
3. All coordinates in output must be residue-level 0-based half-open while preserving original 1-based fields.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/data/parse_mhc_if_v2.py`
- Create/Modify: `epitope_head/data/validators.py`
- Create/Modify: `epitope_head/data/build_dataset.py`
- Create/Modify: `epitope_head/configs/data.yaml`
- Test: `tests/epitope_head/data/test_module_a_contract.py` (consolidated: A2 explode edge cases, A3 coord/length boundary cases, A4 allele classification + deterministic sampling)

**Testing approach (actual)**
- A0/A1/A5 logic is trivial (YAML load, TSV read, schema select+coerce) — covered implicitly by pipeline e2e run, no dedicated unit tests.
- A2/A3/A4 have real edge cases — tested via 19 unit tests in `test_module_a_contract.py`.
- A6 reproducibility verified manually (two pipeline runs, digest comparison) rather than as a persistent test, to avoid slow TSV-reload overhead in CI.

**Execution Plan (Expanded)**

### Task A0: Freeze Runtime Decisions Before Parsing

**Goal**
- Lock current run assumptions so output is reproducible and comparable.

**Actions**
1. Freeze `target_allele = HLA-DRB1*07:01`.
2. Freeze SA policy profiles:
   - `strict_sa`: keep only `resolution == high_res_single` and `alleles == ["HLA-DRB1*07:01"]`.
   - `balanced_sa`: `strict_sa` union deterministic 0.5x sample from `multi` rows containing `HLA-DRB1*07:01`.
3. Freeze balanced sampling method: deterministic row-level sampling with fixed seed.
4. Freeze accepted peptide length range (`12-25`).
5. Freeze coordinate conversion rule (`1b inclusive -> 0b half-open`).
6. Register these values in config and manifest defaults.

**TDD Gate**
1. RED: run policy-validation test with missing `target_allele`; expected FAIL.
2. GREEN: add minimal config wiring; rerun and expect PASS.

**Acceptance**
- Policies are explicit, versioned, and non-implicit in parser code path.

### Task A1: Build Raw Row Reader + Structural Parsing

**Goal**
- Load TSV rows and safely parse JSON-like columns used by Stage A.

**Actions**
1. Parse `alleles` and `peptide_position_info` with strict error handling.
2. Preserve source columns needed downstream (`source`, `dataset_source`).
3. Emit structured parse errors to rejection ledger (not silent drop).

**TDD Gate**
1. RED: malformed JSON row should fail structural validation test.
2. GREEN: minimal parser branch should classify row as rejected with reason.

**Acceptance**
- All malformed input rows are categorized with deterministic rejection reasons.

### Task A2: Explode Position Info to Record-Per-Protein

**Goal**
- Convert one TSV row with multi-position entries into normalized per-protein records.

**Actions**
1. Explode each `peptide_position_info` entry to one candidate record.
2. Resolve `protein_id`, `start`, `end` from each exploded item.
3. Keep parent row linkage for traceability (row index or record key).

**TDD Gate**
1. RED: multi-position row expected to create multiple records but test fails initially.
2. GREEN: implement minimal explode logic; verify exact record count and mapping.

**Acceptance**
- Explosion count equals valid position entry count per source row.

### Task A3: Coordinate Normalization + Length Filter

**Goal**
- Standardize coordinates and enforce peptide length contract.

**Actions**
1. Compute:
   - `start_0b = start_1b - 1`
   - `end_0b = end_1b`
   - `pep_len = end_0b - start_0b`
2. Validate `pep_len == end_1b - start_1b + 1`.
3. Filter to `pep_len in [12, 25]` and track filtered-out counts.

**TDD Gate**
1. RED: conversion invariant test fails for boundary cases (`start=1`, short/long peptides).
2. GREEN: implement minimal conversion + filter until invariants pass.

**Acceptance**
- No output row violates coordinate or peptide length invariants.

### Task A4: Allele Filter Application (SA v1 Policy)

**Goal**
- Keep only rows compliant with frozen single-allele policy.

**Actions**
1. Build `strict_sa` using exact-match single-allele rules.
2. Build `balanced_sa` by adding deterministic 0.5x sample of qualifying `multi` rows.
3. Reject non-matching allele rows with explicit reason codes per profile.
4. Produce per-profile counts (`strict_rows`, `balanced_rows`, `multi_candidates`, `multi_sampled`) for manifest.

**TDD Gate**
1. RED: test cases for SA/MA/ambiguous alleles fail classification expectations.
2. GREEN: implement minimal rule evaluator until class mapping passes.

**Acceptance**
- Both profiles are policy-compliant, deterministic, and auditable by counts.

### Task A5: SpanRecord Schema Validation + Artifact Write

**Goal**
- Emit canonical `SpanRecord` artifact with strict schema checks.

**Actions**
1. Validate required fields:
   - `protein_id`, `allele`
   - `start_1b`, `end_1b`, `start_0b`, `end_0b`, `pep_len`
   - `peptide_seq`, `source`, `dataset_source`
2. Enforce type and non-null checks.
3. Write `outputs/manifests/span_records_strict.parquet`.
4. Write `outputs/manifests/span_records_balanced.parquet`.
5. Write Stage A profile-aware count summary for downstream manifest aggregation.

**TDD Gate**
1. RED: schema test with missing required fields must fail.
2. GREEN: add minimal validator and writer path until schema tests pass.

**Acceptance**
- Both output parquets are schema-correct and loadable by next stage without transformation.

### Task A6: Determinism + Manifest Registration

**Goal**
- Prove Stage A output is reproducible under fixed config/seed.

**Actions**
1. Run Stage A twice with same inputs/config.
2. Compare:
   - retained row count
   - rejection counts by reason
   - key-level digest (stable subset)
3. Register artifact metadata and counts into manifest section.

**TDD Gate**
1. RED: reproducibility test should fail before deterministic ordering/serialization handling is added.
2. GREEN: add minimal deterministic handling and re-run to PASS.

**Acceptance**
- Rebuild under same conditions produces equivalent Stage A statistics and digest.

**Module A Verification Gates (Release-Level)**
1. Contract gate: coordinate, length, and required field invariants all pass.
2. Failure-mode gate: malformed JSON and malformed positions are rejected with explicit reasons.
3. Policy gate: allele filtering behavior matches frozen SA policy test matrix.
4. Repro gate: repeated run under same config is deterministic.

**Done Criteria**
- `outputs/manifests/span_records_strict.parquet` meets codemap `SpanRecord` contract and is the default v1 training input.
- `outputs/manifests/span_records_balanced.parquet` meets the same schema with deterministic 0.5x multi augmentation.
- Stage A profile-aware rejection taxonomy and counts are recorded and explainable.
- Module A gates are all PASS and logged in `LOG.md`.

## Module B: FASTA Join + Sequence Validation (Data Stage B)

**Objective**
- Attach full-length protein sequences and remove invalid records for both Stage A profiles (`strict_sa`, `balanced_sa`).

**Inputs**
- `outputs/manifests/span_records_strict.parquet`
- `outputs/manifests/span_records_balanced.parquet`
- `data/all_sequences.fasta`

**Outputs**
- `outputs/manifests/span_records_with_seq_strict.parquet` (default downstream input)
- `outputs/manifests/span_records_with_seq_balanced.parquet` (auxiliary experiment track)

**Scope Boundaries**
1. Module B only performs sequence attachment and span-level sequence integrity validation.
2. No deduplication/grouping/split logic in this module.
3. Input row identity must be preserved except for explicitly rejected records.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/data/join_fasta.py`
- Create/Modify: `epitope_head/data/validators.py`
- Create/Modify: `epitope_head/data/build_dataset.py`
- Create/Modify: `epitope_head/configs/data.yaml`
- Test: `tests/epitope_head/data/test_module_b_contract.py` (consolidated: lookup fallback, boundary checks, sequence match checks)

**Testing approach (planned)**
- B1/B2/B3/B4 edge cases validated with focused unit tests in `test_module_b_contract.py`.
- B5/B6 artifact integrity and reproducibility verified via e2e pipeline run + digest/statistics comparison.

**Execution Plan (Expanded)**

### Task B0: Freeze Join Policy and Error Taxonomy

**Goal**
- Lock exact lookup/match behavior so results are deterministic and auditable.

**Actions**
1. Freeze FASTA lookup order:
   - exact `protein_id` match first
   - if miss, retry with version suffix stripped (e.g., `P02751.4 -> P02751`)
2. Freeze mismatch handling policy:
   - unresolved protein id -> reject
   - out-of-range coordinates -> reject
   - extracted peptide mismatch -> reject
3. Freeze rejection reason codes for manifest accounting.

**TDD Gate**
1. RED: missing lookup policy test should fail.
2. GREEN: implement minimal policy resolver until test passes.

**Acceptance**
- Join policy is explicit and encoded in config/validator path.

### Task B1: Build FASTA Index and Lookup Resolver

**Goal**
- Provide deterministic sequence retrieval for all candidate protein IDs.

**Actions**
1. Parse `all_sequences.fasta` into in-memory index keyed by accession.
2. Implement version-strip fallback resolver.
3. Emit resolver diagnostics (`exact_hit`, `stripped_hit`, `miss`).

**TDD Gate**
1. RED: fallback case test fails on versioned accession.
2. GREEN: implement minimal resolver fallback and verify expected hit type.

**Acceptance**
- Resolver behavior is deterministic and classification counts are available.

### Task B2: Profile-Aware Join Pass (`strict` and `balanced`)

**Goal**
- Run identical join/validation logic on both profile inputs without branch drift.

**Actions**
1. Execute same join kernel for `span_records_strict` and `span_records_balanced`.
2. Preserve profile tag in intermediate accounting.
3. Ensure schema parity between the two outputs.

**TDD Gate**
1. RED: schema parity test between two profile outputs fails initially.
2. GREEN: apply shared join path until parity check passes.

**Acceptance**
- Both profiles are processed with one canonical logic path.

### Task B3: Coordinate Range and Boundary Validation

**Goal**
- Ensure coordinates are valid against resolved protein sequence length.

**Actions**
1. Validate `0 <= start_0b < end_0b <= len(protein_seq)`.
2. Validate `pep_len == end_0b - start_0b`.
3. Reject and count boundary violations by reason/profile.

**TDD Gate**
1. RED: boundary tests for negative/start-overflow/end-overflow fail.
2. GREEN: implement minimal boundary validator until all pass.

**Acceptance**
- No retained row has invalid coordinates.

### Task B4: Peptide Exact-Match Validation

**Goal**
- Verify span extraction matches observed peptide sequence exactly.

**Actions**
1. Extract `protein_seq[start_0b:end_0b]`.
2. Compare with `peptide_seq` using strict exact-match policy.
3. Reject mismatches with reason code; retain only exact matches.

**TDD Gate**
1. RED: mismatch case should fail validation test.
2. GREEN: implement strict matcher until mismatch test passes.

**Acceptance**
- Every retained row is sequence-consistent with its span.

### Task B5: Artifact Write + Profile-Aware Summary

**Goal**
- Emit canonical Stage B artifacts for both profiles.

**Actions**
1. Write `span_records_with_seq_strict.parquet`.
2. Write `span_records_with_seq_balanced.parquet`.
3. Ensure required fields for Stage C are present, including `protein_seq`.
4. If all input rows are rejected, still emit empty parquet with canonical Stage B columns (no zero-column artifact).
5. Persist per-profile retention/drop summaries and reason counts.

**TDD Gate**
1. RED: required-field presence test fails before writer update.
2. GREEN: implement minimal writer schema enforcement until pass.

**Acceptance**
- Both outputs are schema-complete and consumable by Module C.

### Task B6: Determinism and Manifest Registration

**Goal**
- Verify Stage B outputs are reproducible under fixed inputs/config.

**Actions**
1. Re-run Stage B with same inputs/seed.
2. Compare per-profile:
   - retained rows
   - drop reason counts
   - stable digest subset
3. Register Stage B stats and artifact metadata into manifest.

**TDD Gate**
1. RED: reproducibility assertion fails before stable ordering/serialization is enforced.
2. GREEN: enforce deterministic output path and re-run to PASS.

**Acceptance**
- Two identical runs produce matching profile-wise counts and digests.

**Module B Verification Gates (Release-Level)**
1. Lookup gate: exact/fallback/miss classification is correct and counted.
2. Boundary gate: no retained invalid coordinates.
3. Match gate: retained rows satisfy strict peptide extraction equality.
4. Repro gate: repeated run under same config is deterministic.

**Done Criteria**
- `outputs/manifests/span_records_with_seq_strict.parquet` is complete and sequence-valid for default downstream training.
- `outputs/manifests/span_records_with_seq_balanced.parquet` is complete and sequence-valid for auxiliary experiments.
- Stage B rejection taxonomy and per-profile join statistics are recorded and explainable.

## Module C: ProteinSample Aggregation (Data Stage C)

**Objective**
- Convert row-level span records into protein-centric training units for both strict and balanced tracks.

**Inputs**
- `outputs/manifests/span_records_with_seq_strict.parquet` (default)
- `outputs/manifests/span_records_with_seq_balanced.parquet` (optional auxiliary track)

**Outputs**
- `outputs/manifests/protein_samples_strict.parquet` (default Stage E input)
- `outputs/manifests/protein_samples_balanced.parquet` (auxiliary experiment input)
- `outputs/manifests/protein_samples.parquet` (strict alias for downstream compatibility)

**Scope Boundaries**
1. Module C only performs aggregation and schema validation of protein-level samples.
2. No split logic, negative sampling, or model-related transforms in this module.
3. Span coordinates remain residue-level 0-based half-open in serialized positives.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/data/make_protein_samples.py`
- Create/Modify: `epitope_head/data/validators.py`
- Create/Modify: `epitope_head/data/build_dataset.py`
- Create/Modify: `epitope_head/configs/data.yaml` (if C-specific config keys are needed)
- Test: `tests/epitope_head/data/test_module_c_contract.py` (consolidated: dedup, grouping isolation, metadata and reproducibility)

**Testing approach (planned)**
- C1/C2/C3 core behavior covered by focused unit tests in `test_module_c_contract.py`.
- Stage C end-to-end outputs and deterministic digests verified by repeated run checks.

**Execution Plan (Expanded)**

### Task C0: Freeze ProteinSample Contract

**Goal**
- Lock the exact Stage C output schema and invariants before implementation.

**Actions**
1. Freeze required fields:
   - `protein_id`, `allele`, `protein_seq`
   - `positives_json` (each span includes `start_0b`, `end_0b`, `pep_len`, `support_n`; see codemap d002)
   - `sequence_length`, `input_span_count`, `positive_count`, `duplicate_span_count`
   - `metadata_json`
2. Freeze grouping key as `(protein_id, allele)`.
3. `support_n` records per-span observation frequency (invariant: `sum(support_n) == input_span_count`). v0 training ignores it; reserved for future loss weighting and eval stratification.
3. Freeze strict alias rule (`protein_samples.parquet` mirrors strict output).

**TDD Gate**
1. RED: schema-required-field test fails before contract enforcement exists.
2. GREEN: implement minimal schema validator until required-field tests pass.

**Acceptance**
- Stage C contract is explicit and version-stable.

### Task C1: Grouping and Positive Span Deduplication

**Goal**
- Aggregate row-level records into one protein-level sample per `(protein_id, allele)`.

**Actions**
1. Group rows by `(protein_id, allele)`.
2. Deduplicate positives by `(start_0b, end_0b, pep_len)`.
3. Serialize positives to deterministic JSON ordering.
4. Track `input_span_count`, `positive_count`, `duplicate_span_count`.

**TDD Gate**
1. RED: duplicate-collapse test fails initially.
2. GREEN: implement minimal grouping + dedup path until expected counts pass.

**Acceptance**
- Duplicates are removed deterministically and accounting fields are correct.

### Task C2: Group Isolation and Sequence Consistency

**Goal**
- Prevent leakage across proteins/alleles and enforce sequence consistency inside each group.

**Actions**
1. Ensure no cross-protein/cross-allele merging.
2. Validate single consistent `protein_seq` per group.
3. Reject or fail-fast on inconsistent sequence values within one group.

**TDD Gate**
1. RED: isolation test with mixed protein/allele keys fails before fix.
2. GREEN: implement key-isolated grouping and consistency checks until pass.

**Acceptance**
- Each output row corresponds to exactly one `(protein_id, allele)` group with one consistent sequence.

### Task C3: Metadata Construction and Invariants

**Goal**
- Build useful per-protein metadata and enforce internal consistency.

**Actions**
1. Compute `sequence_length = len(protein_seq)`.
2. Build `metadata_json` with source composition (e.g., source counts, dataset_source counts).
3. Validate:
   - `positive_count == len(positives_json)`
   - `duplicate_span_count == input_span_count - positive_count`
   - each positive span satisfies boundary and length invariants.

**TDD Gate**
1. RED: metadata consistency tests fail before invariant validation.
2. GREEN: add minimal metadata builder + invariant checks until tests pass.

**Acceptance**
- Metadata and counters are internally consistent and auditable.

### Task C4: Profile-Aware Artifact Writing

**Goal**
- Produce strict/balanced outputs with schema parity and strict alias.

**Actions**
1. Write `protein_samples_strict.parquet`.
2. Write `protein_samples_balanced.parquet`.
3. Write strict alias `protein_samples.parquet`.
4. Ensure all outputs share canonical Stage C schema.

**TDD Gate**
1. RED: schema parity test across strict/balanced outputs fails initially.
2. GREEN: unify writer path until parity tests pass.

**Acceptance**
- Stage C artifacts are profile-complete and downstream-compatible.

### Task C5: Empty-Input and Edge-Case Handling

**Goal**
- Keep Stage C robust when an input profile has zero valid rows.

**Actions**
1. Emit empty parquet with canonical columns (not zero-column artifact).
2. Preserve deterministic behavior for empty and single-row edge cases.

**TDD Gate**
1. RED: empty-input schema test fails before explicit empty-schema handling.
2. GREEN: implement empty-output guard until tests pass.

**Acceptance**
- Empty-profile outputs remain schema-valid and loadable.

### Task C6: Determinism and Summary Registration

**Goal**
- Ensure Stage C is reproducible and emits stable summaries.

**Actions**
1. Re-run Stage C with same inputs/config.
2. Compare per-profile:
   - output rows
   - unique proteins
   - stable output digest subset
3. Register profile-wise aggregation summary for manifest/log consumption.

**TDD Gate**
1. RED: repeated-run equality test fails before deterministic sorting/serialization.
2. GREEN: enforce deterministic ordering and retry to PASS.

**Acceptance**
- Repeated runs produce identical summaries and digests.

**Module C Verification Gates (Release-Level)**
1. Dedup gate: duplicate spans collapse correctly with accurate counts.
2. Isolation gate: no cross-protein/cross-allele leakage.
3. Metadata gate: sequence/counter/json invariants hold.
4. Artifact gate: strict/balanced outputs and strict alias are schema-correct.
5. Repro gate: repeated runs are deterministic.

**Done Criteria**
- `protein_samples_strict.parquet` and `protein_samples_balanced.parquet` are contract-valid and loadable by Module D/E.
- `protein_samples.parquet` alias correctly mirrors strict output.
- Stage C summary metrics and invariants are recorded and explainable.

## Module D: Split + Homology Diagnostic (Data Stage D)

**Objective**
- Implement v1.1 leakage-aware split: exact-sequence dedup isolation (`seq_hash`) + mmseqs cluster-level split isolation for high-homology proteins.

**Inputs**
- `outputs/manifests/protein_samples.parquet` (strict alias; default split source)
- `outputs/manifests/protein_samples_balanced.parquet` (optional auxiliary split track)

**Outputs**
- `outputs/manifests/splits/strict/train_ids.txt`
- `outputs/manifests/splits/strict/val_ids.txt`
- `outputs/manifests/splits/strict/test_ids.txt`
- `outputs/manifests/splits/strict/seq_hash_groups.parquet`
- `outputs/manifests/splits/strict/mmseq90_clusters.tsv`
- `outputs/manifests/splits/strict/cluster_assignment.parquet`
- `outputs/manifests/splits/train_ids.txt` (strict alias for compatibility)
- `outputs/manifests/splits/val_ids.txt` (strict alias for compatibility)
- `outputs/manifests/splits/test_ids.txt` (strict alias for compatibility)
- `outputs/manifests/splits/strict/split_diagnostics.json`
- (optional) `outputs/manifests/splits/balanced/{train,val,test}_ids.txt`

**Scope Boundaries**
1. Module D only generates split IDs and diagnostics; no model training/evaluation here.
2. Primary split authority is strict track; balanced track follows explicit policy (optional).
3. v1.1 split minimum unit is `seq_hash`/cluster, but outputs remain `protein_id` ID lists.
4. Training unit remains `protein_id` + span coordinates; split logic only controls partition.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/data/create_splits.py`
- Create/Modify: `epitope_head/data/build_dataset.py`
- Create/Modify: `epitope_head/configs/data.yaml` (split policy keys)
- Test: `tests/epitope_head/data/test_module_d_contract.py` (seq_hash isolation, cluster isolation, weighted balancing, reproducibility)

**Testing approach (planned)**
- D1 exact-duplicate isolation covered via seq_hash fixtures.
- D2/D3 cluster-level isolation and weighted assignment validated with synthetic skewed clusters.
- D4 mmseqs parameter/coverage-mode contract validated by command-generation tests + fixture integration.
- D5 alias and diagnostics schema validated by integration tests.

**Execution Plan (Expanded)**

### Task D0: Freeze Split Policy and Profile Strategy

**Goal**
- Lock split behavior before generating IDs.

**Actions**
1. Freeze strict split as default authority.
2. Freeze ratios and seed:
   - `split_ratios = [0.8, 0.1, 0.1]`
   - `split_seed = 42`
3. Freeze split algorithm hierarchy:
   - step-1: enforce exact sequence grouping by `seq_hash`
   - step-2: enforce mmseq cluster grouping (cluster-level non-crossing split)
4. Freeze mmseq clustering parameters for v1.1:
   - sequence identity: `0.90`
   - coverage: `>= 0.80`
   - coverage mode: `--cov-mode 2` (query/shorter-sequence alignment coverage)
   - determinism: `--threads 1`
   - mmseqs binary path: configurable via `data.mmseqs_bin` (default: `mmseqs`)
5. Freeze cluster load definition for approximate stratified split:
   - `cluster_weight = #unique_positive_spans` (sum across cluster proteins)
6. Freeze balanced handling policy:
   - default: optional auxiliary split generated independently on balanced protein universe
   - strict aliases always map to strict outputs
7. Freeze warning threshold for cross-split similarity (`max_cross_split_seqid > 0.80` => warning flag).

**TDD Gate**
1. RED: missing/invalid ratio config test fails.
2. GREEN: minimal policy loader and validator until tests pass.

**Acceptance**
- Split policy is explicit, versioned, and deterministic.

### Task D1: Build seq_hash Groups (Exact Identity Guard)

**Goal**
- Guarantee completely identical sequences never cross splits.

**Actions**
1. Load strict input (`protein_samples.parquet` alias).
2. Compute deterministic `seq_hash` from full `protein_seq`.
3. Build `seq_hash -> [protein_id]` groups and persist `seq_hash_groups.parquet`.
4. Treat each `seq_hash` group as indivisible split atom.

**TDD Gate**
1. RED: fixture with duplicate sequence in two protein_ids should incorrectly split before fix.
2. GREEN: implement seq_hash grouping until duplicates co-locate in one split atom.

**Acceptance**
- Same-sequence proteins cannot be assigned to different splits.

### Task D2: mmseq90 Clustering and Cluster Assignment

**Goal**
- Cluster high-homology sequences and create cluster-level split units.

**Actions**
1. Run mmseqs clustering on representative sequences (by seq_hash group).
2. Use frozen params:
   - identity `0.90`
   - coverage `>= 0.80`
   - short-sequence coverage mode
3. Persist cluster outputs (`mmseq90_clusters.tsv`, `cluster_assignment.parquet`).
4. Expand cluster assignment back to all protein_ids via seq_hash mapping.

**TDD Gate**
1. RED: command/param contract test fails if identity/coverage/mode mismatch.
2. GREEN: implement mmseq invocation wrapper + parser until contract test passes.

**Acceptance**
- All proteins receive one cluster id; cluster mapping is deterministic.

### Task D3: Weighted Cluster-Level Split Assignment

**Goal**
- Assign clusters to train/val/test by target ratios while controlling load skew.

**Actions**
1. Compute per-cluster weight:
   - `#unique_positive_spans` summed across proteins in cluster.
2. Perform approximate stratified assignment (greedy/bin-packing style) toward ratio targets.
3. Enforce hard constraint: one cluster -> one split only.
4. Emit split membership at protein_id level.

**TDD Gate**
1. RED: skewed synthetic clusters produce severe imbalance before weighted assignment.
2. GREEN: implement weighted cluster allocator until load deviation is within tolerance.

**Acceptance**
- Cluster non-crossing holds and split load is close to target ratios.

### Task D4: Split Integrity, Coverage, and Diagnostics

**Goal**
- Validate split integrity and report post-split homology diagnostics.

**Actions**
1. Validate pairwise disjointness of protein_id sets.
2. Validate union coverage against strict protein universe.
3. Validate seq_hash and cluster non-crossing constraints.
4. Run train-vs-val diagnostic and report:
   - `max_cross_split_seqid`
   - count of high-similarity pairs above threshold
5. Emit warning flag when threshold exceeded.
6. Persist diagnostics JSON.

**TDD Gate**
1. RED: any intentional cross-cluster split fixture should fail integrity validation.
2. GREEN: enforce integrity checks + diagnostic reporter until pass.

**Acceptance**
- Leakage-risk signal is measurable and logged.

### Task D5: Optional Balanced Track and Compatibility Aliases

**Goal**
- Preserve downstream compatibility and keep strict/balanced outputs isolated.

**Actions**
1. (Optional) generate balanced split track with same seq_hash/cluster logic.
2. Write strict aliases:
   - `splits/train_ids.txt`
   - `splits/val_ids.txt`
   - `splits/test_ids.txt`
3. Register split stats and diagnostics metadata for manifest/log ingestion.
4. Ensure file format and newline conventions are deterministic.

**TDD Gate**
1. RED: fixture where balanced write contaminates strict alias should fail.
2. GREEN: enforce profile-isolated writers + strict alias copy semantics until pass.

**Acceptance**
- Downstream modules can consume legacy alias paths unchanged.

### Task D6: Determinism Validation

**Goal**
- Prove split and diagnostics are reproducible with fixed input/config.

**Actions**
1. Re-run Stage D twice with same inputs/config.
2. Compare:
   - train/val/test ID file digests
   - seq_hash group digest
   - cluster assignment digest
   - diagnostics JSON digest
   - split-size summary
3. Record deterministic run evidence in log.

**TDD Gate**
1. RED: repeated-run digest mismatch test fails before sorting/serialization is fixed.
2. GREEN: enforce stable ordering and rerun to PASS.

**Acceptance**
- Split artifacts and diagnostics are stable across repeated runs.

**Module D Verification Gates (Release-Level)**
1. Disjointness gate: no protein ID overlap across splits.
2. Coverage gate: strict protein universe fully partitioned.
3. Exact-identity gate: same `seq_hash` never appears in multiple splits.
4. Homology gate: same mmseq cluster never appears in multiple splits.
5. Balance gate: weighted load by `#unique_positive_spans` remains near targets.
6. Repro gate: IDs, groups, clusters, and diagnostics deterministic under fixed seed.
7. Compatibility gate: strict alias files exist and match strict split content.

**Done Criteria**
- Strict split manifests satisfy seq_hash + cluster-level isolation and are reproducible.
- Balanced split manifests (if enabled) follow same isolation constraints without contaminating strict aliases.
- mmseq parameter contract (`90% id`, `coverage>=0.8`, short-seq coverage mode) is fixed and logged.
- Homology diagnostics are generated and threshold warnings are explicit.

## Module E: Training Runtime (Stage E)

**Objective**
- Train implicit-window scorer with frozen encoder and contrastive objective.

**Inputs**
- `outputs/manifests/splits/strict/{train,val,test}_ids.txt` (default authority)
- `outputs/manifests/splits/balanced/{train,val,test}_ids.txt` (optional auxiliary track)
- `outputs/manifests/protein_samples_strict.parquet` (default)
- `outputs/manifests/protein_samples_balanced.parquet` (optional auxiliary track)
- `epitope_head/configs/{data,model,train}.yaml` (resolved per run)

**Outputs**
- `outputs/checkpoints/epoch_*.pt`
- `outputs/checkpoints/best.pt`
- `outputs/metrics/train_log.jsonl`
- `outputs/metrics/val_log.jsonl`
- `outputs/metrics/run_<run_id>/resolved_config.yaml`

**Scope Boundaries**
1. Module E covers train/val runtime only; inference export remains Module F.
2. Default training profile is strict split; balanced profile is optional experiment track.
3. Encoder remains frozen in `v0`; no encoder fine-tuning in this module.
4. Runtime must honor Module D split isolation; no runtime re-shuffling across split files.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/configs/model.yaml`
- Create/Modify: `epitope_head/configs/train.yaml`
- Create/Modify: `epitope_head/model/encoder.py`
- Create/Modify: `epitope_head/model/projection.py`
- Create/Modify: `epitope_head/model/span_pooling.py`
- Create/Modify: `epitope_head/model/scorer.py`
- Create/Modify: `epitope_head/model/aggregators.py`
- Create/Modify: `epitope_head/training/negatives.py`
- Create/Modify: `epitope_head/training/losses.py`
- Create/Modify: `epitope_head/training/datamodule.py`
- Create/Modify: `epitope_head/training/metrics.py`
- Create/Modify: `epitope_head/training/trainer.py`
- Create/Modify: `scripts/train_v1.sh`
- Test: `tests/epitope_head/training/test_module_e_contract.py` (consolidated: batch/sampler/model/loss/runtime artifact checks)

**Testing approach (planned)**
- E1/E2/E3/E4 contract behavior is covered by focused unit tests in `test_module_e_contract.py`.
- E5/E6 runtime integrity is verified by small-scale smoke training (single short epoch on strict profile subset).
- Determinism/reproducibility checks use repeated-run digest comparison on tiny fixed fixtures.

**Long-Protein Chunking Policy (v1.1 Frozen Baseline)**

1. Fixed constants:
   - encoder context length: `C = 1022`
   - chunk stride: `S = 512`
   - max peptide length: `kmax = 25`
   - required flank for span features: left/right `1` residue
   - safe margin: `M = 32` (`M >= kmax + 2`)
2. Chunk start generation for sequence length `L`:
   - `t_last = max(0, L - C)`
   - starts: `0, S, 2S, ...` with `t <= t_last`
   - if `t_last` not already present, append `t_last`
   - for `L <= C`, no chunking (single encoder pass)
3. Per-chunk trusted interior interval:
   - chunk range: `[t_j, u_j)` where `u_j = t_j + C`
   - trusted interior: `I_j = [t_j + M_L(j), u_j - M_R(j))`
   - `M_L(j)=0` for first chunk (`t_j=0`), else `M`
   - `M_R(j)=0` for last chunk (`t_j=t_last` or `u_j>=L`), else `M`
4. Stitch mode (default): `per_residue_stitch` with deterministic center-crop:
   - for residue `i`, collect chunks where `i in I_j`
   - choose chunk with minimal `|i - c_j|`, `c_j=t_j + C/2`
   - tie-breaker fixed to smallest chunk index
5. Optional stitch mode: `center_weighted` (non-default fallback):
   - only chunks with `i in I_j` may contribute non-zero weight
   - weighted average in overlap regions; no edge-region mixing
6. Optional training mode: `per_window_single_chunk` (feature flag only):
   - for span `[s, s+k)`, require expanded interval `E_{s,k}=[s-1,s+k]` fully contained in one `I_j`
   - choose valid chunk by nearest window center to chunk center
   - if no valid chunk exists: fallback allowed for diagnostics only, set reliability `r_{s,k}=0`
7. Reliability contract:
   - residue reliability `r_i` exported by stitcher
   - true sequence termini (`i<M` or `L-1-i<M`) are not auto-penalized solely due to natural N/C boundary
   - optional span reliability `r_{s,k}` used for filtering or loss weighting of sampled negatives
8. Training compute policy for long proteins:
   - mini-batch unit can be `(protein_id, chunk_id)` to avoid full-protein all-chunk forward each step
   - positives assigned to owning chunk by trusted-interval rule
   - sampled negatives drawn from the same chunk coverage region by default
9. Required diagnostics:
   - window bias vs distance-to-chunk-boundary bucket (`d_{s,k}`) for positives/negatives
   - long-vs-short subgroup validation comparison (`L > 1022` vs `L <= 1022`)

**Execution Plan (Expanded)**

### Task E0: Freeze Runtime Protocol and Config Contract

**Goal**
- Lock train-time behavior before writing runtime code.

**Actions**
1. Freeze `strict` profile as default training authority.
2. Freeze long-protein policy to chunking v1.1 baseline:
   - no skip-by-length as default path
   - use `C=1022`, `S=512`, `M=32`, trusted-interval stitching.
3. Freeze sampling/loss defaults:
   - `train.neg_ratio = 7`
   - `train.hard_negative_fraction = 0.3`
   - `train.hard_neg_max_overlap_ratio = 0.8`
   - `train.hard_neg_offset_range = 5`
   - `train.neg_length_sampling = match_positive`
   - `train.loss.tau = 0.1`
   - `train.loss.lambda_mp = 0.0`
   - `train.loss.lambda_smooth = 0.0`
4. Freeze optimization/checkpoint defaults:
   - optimizer `adamw`, scheduler `cosine`, monitor metric `logit_gap`
   - checkpoint cadence and early-stopping policy from `train.yaml`.
5. Freeze reproducibility knobs (`train.seed`, deterministic flag behavior).
6. Freeze chunking config surface:
   - `train.chunking.enabled = true`
   - `train.chunking.context_len = 1022`
   - `train.chunking.stride = 512`
   - `train.chunking.margin = 32`
   - `train.chunking.stitch_mode = per_residue_stitch`
   - `train.chunking.window_mode = per_residue_stitch` (reserve `per_window_single_chunk` as switch)
   - `train.chunking.enable_reliability = true`
7. Register codemap diff requirement for this policy freeze before implementation merge (long-protein policy changed from skip to chunking).

**TDD Gate**
1. RED: missing key runtime config should fail config-validation tests.
2. GREEN: add minimal loader/validator until runtime config contract tests pass.

**Acceptance**
- Runtime protocol is explicit, versioned, and testable.

### Task E1: Build DataModule and Token-Budget Dynamic Batching

**Goal**
- Load split-scoped proteins and batch by token budget safely.

**Actions**
1. Load per-split `protein_id` manifests and join with `ProteinSample`.
2. Enforce strict split scope (`train`/`val` IDs only) with fail-fast on missing IDs.
3. Build chunk plans for proteins with `L > 1022` using frozen `C/S/M` policy.
4. Emit chunk-aware sample units while preserving protein-level labels/metadata linkage.
   - Span ownership: each positive is pre-assigned to exactly one chunk via `assign_span_to_chunk` (expanded interval `E_{s,k}=[s-1,s+k] ⊆ I_j` + nearest-center deterministic tie-break; see `chunking.md` §5).
   - Orphan positives (fit in no chunk's trusted interior) trigger `ValueError`, not silent drop.
5. Build token-based dynamic batches constrained by `train.max_tokens`.
6. Emit batch tensors and masks compatible with frozen ESM-2 encoder.

**TDD Gate**
1. RED: synthetic long-protein fixture should fail chunk coverage/interior ownership assertions before fix.
2. GREEN: implement chunk planner + collate until chunk-coverage, budget, and shape tests pass.

**Acceptance**
- Data loader produces valid, bounded chunk-aware batches with deterministic split membership.

### Task E2: Implement Negative Sampler Contract (Intra-Protein)

**Goal**
- Generate leakage-safe sampled negatives for each protein.

**Actions**
1. Build per-protein `positive_span_set={(start_0b,end_0b)}` from `positives_json`.
2. Sample negatives with configured hard/easy ratio.
3. Hard negatives follow overlap/offset rules without exceeding `hard_neg_max_overlap_ratio`.
4. Length sampling follows configured policy (`match_positive` / `uniform`).
5. Enforce global exclusion: no known positive span can be sampled as negative.
6. Make sampler deterministic under fixed seed + worker seed derivation.

**TDD Gate**
1. RED: fixture where positive span appears in negative pool must fail.
2. GREEN: enforce exclusion + ratio/length constraints until sampler tests pass.

**Acceptance**
- Negative sampler satisfies exclusion, ratio, and length contracts.

### Task E3: Implement Model Path (Encoder -> Projection -> Span Features -> Scorer)

**Goal**
- Build contract-compliant span scoring forward path.

**Actions**
1. Frozen encoder wrapper:
   - run under `torch.no_grad()`
   - strip BOS/EOS and return residue-level embeddings.
   - for long proteins, run per chunk and stitch by frozen policy.
2. Projection head maps `D_enc -> D_proj`.
3. Span feature builder composes:
   - interior mean pooling (prefix-sum)
   - in-span endpoints
   - boundary flanks with pad vectors at boundaries
   - length and allele embeddings.
4. Scorer MLP outputs one logit per span.

**TDD Gate**
1. RED: shape/boundary tests and seam-consistency tests fail for long-protein chunk fixtures before implementation.
2. GREEN: implement minimal forward + stitch path until tensor shape, boundary, and seam tests pass.

**Acceptance**
- Forward path matches codemap tensor contracts and handles both boundary and chunk-seam conditions.

### Task E4: Implement Training Losses (InfoNCE Primary, Optional Multi-Positive)

**Goal**
- Train on relative ranking constraints with stable numerics.

**Actions**
1. Implement intra-protein InfoNCE with temperature `tau`.
2. Keep optional multi-positive softmax branch (`lambda_mp`) with default disabled.
3. Add optional smoothness regularizer hook (`lambda_smooth`) with default disabled.
4. Enforce numerically stable log-sum-exp computations.
5. Return decomposed loss terms for logging (`loss_total`, `loss_intra`, optional terms).

**TDD Gate**
1. RED: controlled logits fixture should fail expected loss-order assertions before fix.
2. GREEN: implement stable loss kernels until ranking/loss-direction tests pass.

**Acceptance**
- Loss implementation is stable, configurable, and aligned with v0 objective.

### Task E5: Implement Trainer Loop and Validation Metrics

**Goal**
- Execute train/val steps with sanity metrics and optimization controls.

**Actions**
1. Build train/val step plumbing:
   - forward pass
   - loss backward/optimizer step
   - scheduler and gradient clipping.
2. Compute and log per-step/per-epoch sanity metrics:
   - `mean_pos_logit`
   - `mean_neg_logit`
   - `logit_gap`
   - `per_protein_auc` (when both classes present).
3. Ensure frozen encoder parameters never receive optimizer updates.
4. Add NaN/Inf guard with fail-fast error reporting.

**TDD Gate**
1. RED: one-step fixture should show encoder-update or metric-shape failures before fix.
2. GREEN: implement trainer guards until parameter-freeze and metric contract tests pass.

**Acceptance**
- Training loop runs safely and emits contract-valid sanity metrics.

### Task E6: Checkpointing, Metrics Logs, and Run Metadata

**Goal**
- Persist reproducible training artifacts and metadata.

**Actions**
1. Write epoch checkpoints and best checkpoint according to monitor metric.
2. Persist checkpoint metadata:
   - `manifest_version`
   - `diff_ids_applied`
   - `config_hash`
   - monitoring protocol fields.
3. Write train/val JSONL logs with stable key schema.
4. Save `resolved_config.yaml` under `outputs/metrics/run_<run_id>/`.
5. Emit concise run summary for Module G registry ingestion.

**TDD Gate**
1. RED: checkpoint/log schema tests fail when required metadata keys are absent.
2. GREEN: implement artifact writers until schema and metadata tests pass.

**Acceptance**
- Runtime artifacts are complete and directly consumable by Module F/G.

### Task E7: End-to-End Smoke Run and Reproducibility Gate

**Goal**
- Prove Stage E can run end-to-end on current data contracts.

**Actions**
1. Run a short smoke training on strict profile (`1` short epoch or fixed step budget).
2. Verify artifact existence and loadability:
   - checkpoints
   - train/val logs
   - resolved config snapshot.
3. Verify sanity trend:
   - no schema/shape crash
   - `logit_gap` finite and trackable.
4. Run long-protein chunking sanity diagnostics:
   - `d_{s,k}` bucket bias check
   - long-vs-short subgroup metrics comparison.
5. Re-run tiny deterministic fixture and compare stable digests/statistics.

**TDD Gate**
1. RED: smoke command expected to fail before trainer/artifact wiring completes.
2. GREEN: complete minimal runtime path until smoke run passes.

**Acceptance**
- Stage E is executable, auditable, and ready for Module F handoff.

**Module E Verification Gates (Release-Level)**
1. Data gate: split-scoped loading and chunking policy (`C/S/M/interior`) are enforced and counted.
2. Sampler gate: negatives never include known positives; ratio/length constraints hold.
3. Model gate: span feature/scorer tensor contracts pass edge-boundary + seam-consistency tests.
4. Loss gate: InfoNCE and optional branches are numerically stable and configurable.
5. Training gate: frozen encoder remains frozen; optimizer/scheduler/grad-clip behavior is valid.
6. Artifact gate: checkpoints/logs/resolved-config carry required metadata and schemas.
7. Smoke gate: one end-to-end strict run completes with valid sanity metrics.
8. Chunk bias gate: no severe score drift across boundary-distance buckets; long-vs-short subgroup gap is explainable.

**Done Criteria**
- Stage E completes one strict-profile run end-to-end without schema/shape/runtime failures.
- `logit_gap`/pos-vs-neg metrics are recorded and interpretable for sanity check.
- Long-protein chunking diagnostics are reported and pass release thresholds.
- Checkpoint and metric artifacts are reproducible enough for Module F inference and Module G registry.

## Module F: Inference + Export (Stage F)

**Objective**
- Produce canonical prediction payload per protein.

**Inputs**
- `outputs/checkpoints/best.pt` (default inference checkpoint)
- protein sequence input (single protein or batch list)
- `epitope_head/configs/{model,inference}.yaml` (resolved per run)
- optional allele token/interface (reserved for v2; v0 uses single-allele placeholder)

**Outputs**
- `outputs/predictions/<protein_id>.json`
- optional run summary: `outputs/predictions/prediction_summary.json` (counts + digest + config hash)

**Scope Boundaries**
1. Module F performs inference and payload export only; no training updates and no registry writes here.
2. Canonical inference computes `h` and `R` over all enumerated windows (`k=12..25`) per codemap contract.
3. Output payload schema is fixed and backward-compatible for downstream consumers.
4. Long proteins use frozen chunking/stitch policy from Module E runtime decisions.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/configs/inference.yaml`
- Create/Modify: `epitope_head/inference/predictor.py`
- Create/Modify: `epitope_head/inference/export.py`
- Create/Modify: `epitope_head/training/model.py` (reuse-safe inference hooks only if required)
- Create/Modify: `scripts/infer_v1.sh`
- Test: `tests/epitope_head/inference/test_module_f_contract.py` (consolidated: enumeration/aggregation/export/repro checks)

**Testing approach (planned)**
- F1/F2/F3/F4 contract behavior covered by focused unit tests in `test_module_f_contract.py`.
- F5/F6/F7 validated by small e2e inference smoke on representative short + long proteins.
- Reproducibility verified by repeated-run payload digest comparison under fixed config.

**Execution Plan (Expanded)**

### Task F0: Freeze Inference Protocol and Config Contract

**Goal**
- Lock post-training inference behavior before implementing predictor/export logic.

**Actions**
1. Freeze inference input checkpoint source (`best.pt` by default).
2. Freeze candidate window range to `k in [12, 25]`.
3. Freeze aggregation formulas:
   - residue hotspot raw: `h_i = logsumexp(z_cover_i) - log(count_cover_i)`
   - global risk: `R = logsumexp(z_all) - log(num_windows)`
4. Freeze post-processing keys and defaults:
   - `inference.hotspot_center_method in {median, mean, none}`
   - `inference.hotspot_clamp in {softplus, relu, none}`
5. Freeze output payload key schema (`window_logits`, `residue_hotspot`, `global_risk`, `meta`).
6. Freeze long-protein inference behavior to chunking v1.1 stitched representation (same core policy family as Module E).

**TDD Gate**
1. RED: config validation tests fail with missing/invalid inference keys.
2. GREEN: add minimal inference config loader/validator until schema tests pass.

**Acceptance**
- Inference protocol is explicit, deterministic, and contract-valid.

### Task F1: Checkpoint Load and Predictor Initialization

**Goal**
- Initialize inference runtime from trained checkpoint with strict metadata checks.

**Actions**
1. Load checkpoint and validate required metadata (`manifest_version`, `config_hash`, `diff_ids_applied`).
2. Reconstruct model components from resolved model config and checkpoint weights.
3. Enforce frozen encoder inference mode and device placement policy.
4. Fail-fast on checkpoint/config incompatibility (shape mismatch, missing keys).

**TDD Gate**
1. RED: malformed checkpoint fixture should fail metadata/shape checks.
2. GREEN: implement loader and compatibility guards until tests pass.

**Acceptance**
- Predictor starts from checkpoint deterministically with validated model state.

### Task F2: Sequence Handling and Long-Protein Inference Path

**Goal**
- Produce residue embeddings for both short and long proteins under frozen policy.

**Actions**
1. Tokenize sequence input with BOS/EOS handling consistent with training encoder path.
2. For `L <= 1022`, run single-pass encoder inference.
3. For `L > 1022`, run chunked encoder inference and stitch full-length residue embeddings using frozen policy.
4. Preserve deterministic residue ownership/tie-break behavior in overlap regions.
5. Expose per-residue reliability diagnostics for optional debug/meta fields.

**TDD Gate**
1. RED: long-protein seam fixture should fail continuity/determinism assertions before stitch implementation.
2. GREEN: implement short+long path until seam consistency tests pass.

**Acceptance**
- Predictor returns stable full-length residue embeddings for all supported lengths.

### Task F3: Full Window Enumeration and Logit Scoring

**Goal**
- Score all candidate windows needed for canonical inference outputs.

**Actions**
1. Enumerate all windows over `k=12..25` with 0-based half-open coordinates.
2. Build span features and run scorer logits for all windows.
3. Preserve deterministic output ordering (`start_0b`, then `k`).
4. Validate boundary handling at N/C termini (flank pads and index safety).

**TDD Gate**
1. RED: enumeration-count and boundary fixtures fail before implementation.
2. GREEN: implement enumerator+scorer loop until count/order/boundary tests pass.

**Acceptance**
- `window_logits` are complete, ordered, and coordinate-correct.

### Task F4: Hotspot and Global Risk Aggregation + Post-Processing

**Goal**
- Convert window logits into canonical residue/global outputs.

**Actions**
1. Compute `h_raw` per residue using normalized log-mean-exp over covering windows.
2. Compute `global_risk` over all windows using normalized log-mean-exp.
3. Apply centered post-processing to get `h_processed` per config.
4. Apply optional clamp (`softplus`/`relu`/none) and validate numeric stability.

**TDD Gate**
1. RED: analytic toy fixtures fail LSE/log-mean-exp expected values before implementation.
2. GREEN: implement aggregation/post-process path until formula tests pass.

**Acceptance**
- `h_raw`, `h_processed`, and `global_risk` match formula contracts and remain finite.

### Task F5: Canonical JSON Export

**Goal**
- Export one contract-complete prediction JSON per protein.

**Actions**
1. Emit required payload keys and field types.
2. Export `window_logits` entries as `{start_0b, end_0b, k, z}`.
3. Export `residue_hotspot` entries as `{index_0b, h_raw, h_processed}`.
4. Export `meta` with model/config/protocol/checkpoint identifiers and timestamp.
5. Write to `outputs/predictions/<protein_id>.json` with deterministic serialization settings.

**TDD Gate**
1. RED: schema tests fail for missing keys/type mismatches before exporter implementation.
2. GREEN: implement exporter until schema and deterministic-key-order tests pass.

**Acceptance**
- Output JSON is schema-complete and stable for downstream consumers.

### Task F6: Reproducibility and Payload Validation

**Goal**
- Ensure inference artifacts are reproducible and auditable.

**Actions**
1. Re-run inference on fixed fixtures with identical config/checkpoint.
2. Compare payload digest and key numeric summaries.
3. Validate no NaN/Inf in logits/hotspots/global risk.
4. Emit concise summary artifact for batch runs (counts + digest + config hash).

**TDD Gate**
1. RED: repeated-run digest mismatch test fails before deterministic ordering/serialization safeguards.
2. GREEN: enforce deterministic export path until reproducibility tests pass.

**Acceptance**
- Inference outputs are reproducible under fixed inputs/config/checkpoint.

### Task F7: End-to-End Inference Smoke Gate

**Goal**
- Prove Stage F is executable on current E outputs and ready for Module G.

**Actions**
1. Run inference smoke on representative proteins (at least one short and one long sequence).
2. Verify output files exist and are loadable JSON.
3. Verify shape sanity:
   - window count matches enumeration formula
   - hotspot length equals protein length
   - `global_risk` is finite.
4. Record smoke evidence and uncovered edges in log.

**TDD Gate**
1. RED: smoke expected to fail before predictor/export wiring is complete.
2. GREEN: complete minimal pipeline until smoke run passes.

**Acceptance**
- Stage F exports canonical prediction payloads and is ready for registry/comparability integration.

**Module F Verification Gates (Release-Level)**
1. Input gate: checkpoint/config compatibility checks are strict and explicit.
2. Encoding gate: short/long sequence inference paths are deterministic and seam-safe.
3. Enumeration gate: all candidate windows are covered with correct boundary semantics.
4. Aggregation gate: `h_raw` and `global_risk` formulas are numerically correct and finite.
5. Post-process gate: centering/clamp behavior matches inference config contract.
6. Export gate: per-protein JSON payload passes schema and deterministic serialization checks.
7. Repro gate: repeated inference under fixed conditions yields matching digests.
8. Smoke gate: representative short+long proteins successfully export canonical payloads.

**Done Criteria**
- Stage F generates contract-complete prediction JSON for representative proteins.
- Long-protein inference path is stable under chunking/stitch policy and seam diagnostics.
- Payloads are deterministic enough for Module G registry/comparability ingestion.

## Module G: Metrics + Registry + Comparability

**Objective**
- Ensure run traceability and cross-run comparability.

**Inputs**
- Stage E artifacts under `outputs/metrics/run_<run_id>/`
- Stage F prediction artifacts under `outputs/predictions/` (optional for training-only runs)
- checkpoint metadata (`manifest_version`, `diff_ids_applied`, `config_hash`)
- resolved config snapshot(s) and frozen protocol parameters

**Outputs**
- `outputs/metrics/run_registry.jsonl`
- `outputs/metrics/run_<run_id>/resolved_config.yaml`
- `outputs/metrics/run_<run_id>/run_summary.json` (already generated in E; validated in G)
- optional comparability report: `outputs/metrics/comparability/<run_id_a>__<run_id_b>.json`

**Scope Boundaries**
1. Module G only governs metrics/registry contracts and comparability checks; it does not retrain models or re-export per-protein payload internals.
2. Registry row schema is frozen by codemap Section 12.4; new fields must be additive and backward-compatible.
3. Comparability is a strict gate: runs are comparable only when `manifest_version` and evaluation protocol signatures match.
4. Any protocol drift discovered in G must be reflected in codemap diff entry before being accepted into registry semantics.

**Planned File Touchpoints**
- Create/Modify: `epitope_head/training/registry.py` (registry row build/validate/append + comparability check)
- Create/Modify: `epitope_head/training/trainer.py` (run_id propagation + registry write hook integration)
- Create/Modify: `epitope_head/inference/export.py` (optional run-level summary hash handoff for registry)
- Create/Modify: `scripts/register_run_v1.sh` (deterministic registry append/check command wrapper)
- Test: `tests/epitope_head/training/test_module_g_contract.py` (schema/comparability/fail-fast contracts)

**Execution Plan (Expanded)**

### Task G0: Freeze Registry Schema and Comparability Contract

**Goal**
- Lock Module G data contract before implementation to prevent post-hoc registry drift.

**Actions**
1. Freeze required registry fields:
   - `run_id`, `timestamp`, `manifest_version`, `diff_ids_applied`
   - `resolved_config_path`, `best_checkpoint_path`
   - `primary_metrics` (at least `logit_gap`, `per_protein_auc`; extensible map)
   - `config_hash`, `protocol_signature`
2. Freeze comparability predicate:
   - strict equality on `manifest_version`
   - strict equality on `protocol_signature`
3. Define additive-only extension policy for registry rows.

**TDD Gate**
1. RED: schema fixture missing required keys should fail validation.
2. GREEN: implement schema validator until required/minimal rows pass.

**Acceptance**
- Registry contract is explicit, versionable, and machine-validated.

### Task G1: Deterministic Run Identity and Protocol Signature

**Goal**
- Produce stable run identity and protocol fingerprints for downstream comparability checks.

**Actions**
1. Freeze `run_id` format (stable, sortable, collision-safe under local single-process assumptions).
2. Build `protocol_signature` from comparability-critical fields only:
   - window range (`min_k/max_k`)
   - hotspot post-process config
   - chunking/stitch policy keys
   - core evaluation policy marker
3. Exclude non-semantic runtime noise fields (timestamps, device, output paths) from signature.

**TDD Gate**
1. RED: two semantically identical configs with reordered keys should produce different signatures pre-fix.
2. GREEN: canonical serialization makes signatures identical.

**Acceptance**
- `run_id` and `protocol_signature` are deterministic and reproducible.

### Task G2: Registry Writer with Strict Fail-Fast Validation

**Goal**
- Append only valid rows to `run_registry.jsonl`.

**Actions**
1. Implement row validator with required-key/type checks.
2. Enforce path existence checks for:
   - `resolved_config_path`
   - `best_checkpoint_path`
3. Enforce checkpoint metadata consistency:
   - checkpoint `config_hash` equals row `config_hash`
   - checkpoint `manifest_version` equals row `manifest_version`
4. Append JSONL entry atomically (single-row append semantics; no partial writes).

**TDD Gate**
1. RED: malformed rows and broken paths must fail append.
2. GREEN: valid row appends exactly once and reloads cleanly.

**Acceptance**
- Registry file contains only schema-valid, artifact-consistent rows.

### Task G3: Trainer/Inference Integration for Registry Population

**Goal**
- Ensure Stage E/F outputs can feed Module G without manual patching.

**Actions**
1. Add lightweight integration hook from trainer summary to registry builder.
2. Propagate `run_id`, `config_hash`, and checkpoint pointer consistently.
3. Optionally attach Stage F prediction summary digest when available; allow training-only rows without prediction fields.
4. Keep backward compatibility with existing `run_summary.json` schema.

**TDD Gate**
1. RED: integration fixture writes summary but registry row is incomplete.
2. GREEN: integrated flow emits complete row and passes validator.

**Acceptance**
- Standard train (+optional inference) workflow can register runs automatically.

### Task G4: Comparability Guard and Pairwise Check API

**Goal**
- Provide a deterministic guard that decides whether two runs are comparable.

**Actions**
1. Implement `is_comparable(run_a, run_b)` with explicit reason codes:
   - `manifest_mismatch`
   - `protocol_mismatch`
   - `comparable`
2. Expose a small report artifact for pairwise checks.
3. Fail-fast on missing required fields instead of returning ambiguous results.

**TDD Gate**
1. RED: mismatched manifests/protocols are incorrectly marked comparable.
2. GREEN: guard blocks mismatches with correct reason codes.

**Acceptance**
- Comparability decisions are deterministic, explainable, and auditable.

### Task G5: Backfill and Legacy Registry Hygiene

**Goal**
- Bring existing run artifacts into the new registry without silent corruption.

**Actions**
1. Define a backfill command to scan existing `outputs/metrics/run_*` directories.
2. Validate each candidate row before append.
3. Mark unverifiable historical runs with explicit status/reason instead of forced append.

**TDD Gate**
1. RED: mixed valid/invalid backfill fixtures silently pass.
2. GREEN: only valid rows are appended; invalid rows are reported.

**Acceptance**
- Historical runs are ingested with transparent validation outcomes.

### Task G6: Release Smoke and Documentation Sync

**Goal**
- Prove Module G is executable and aligned with codemap/plan governance.

**Actions**
1. Run smoke flow:
   - generate/collect one training run record
   - append to registry
   - run one comparability check against another run
2. Verify registry and comparability artifacts are loadable JSON/JSONL.
3. Log evidence and residual risks in `LOG.md`.

**TDD Gate**
1. RED: smoke expected to fail before full wiring.
2. GREEN: end-to-end registry/comparability smoke passes.

**Acceptance**
- Module G contracts are executable and ready for release-level governance.

**Module G Verification Gates (Release-Level)**
1. Schema gate: registry row schema is enforced with strict required fields/types.
2. Artifact gate: referenced config/checkpoint artifacts exist and metadata-consistent.
3. Identity gate: `run_id` and `protocol_signature` are deterministic.
4. Comparability gate: non-comparable runs are blocked with explicit reason codes.
5. Integration gate: trainer/inference outputs register without manual patching.
6. Backfill gate: legacy run ingestion is validated and auditable.
7. Smoke gate: end-to-end registry append + pairwise comparability check passes.

**Done Criteria**
- Every accepted run has a complete, validated registry row linked to reproducible artifacts.
- Cross-run comparability can be answered deterministically from registry fields alone.
- Module G outputs are auditable and ready for downstream experiment governance.

## 5. Current Open Decisions (Must Be Frozen Before/At Module Start)

1. Frozen:
   - `target_allele = HLA-DRB1*07:01`
2. Frozen:
   - SA profiles for Module A:
     - `strict_sa`: high_res_single exact single allele match
     - `balanced_sa`: strict + deterministic `0.5x` multi augmentation containing target allele
3. Frozen:
   - Long protein policy: chunking v1.1 baseline (`C=1022`, `S=512`, `M=32`, trusted interior + deterministic stitch), replacing `skip` as default.
4. Open:
   - Homology split upgrade threshold and rollout trigger.

## 6. Risk Register (Execution-Level)

1. Data leakage risk:
   - split overlap or high homology across splits inflates validation metrics.
2. Label contamination risk:
   - positive windows sampled as negatives if exclusion set is incomplete.
3. Length shortcut risk:
   - mismatched negative length distribution biases scorer.
4. Coordinate drift risk:
   - mixed 1-based and 0-based semantics cause silent training corruption.
5. Reproducibility risk:
   - missing manifest/config linkage breaks run comparability.
6. Chunk seam bias risk:
   - long-protein stitching could introduce boundary-distance-dependent score drift without diagnostics/guards.

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

### 8.1 Log Location and Purpose

1. All planning and deployment progress logs are recorded in `LOG.md`.
2. `PLAN.md` defines strategy and module execution rules; `LOG.md` records time-ordered execution facts.
3. `LOG.md` is append-only. Existing entries are not rewritten; corrections are new entries linked to old IDs.

### 8.2 Required Entry Schema (Scientific/Structured)

Each entry must include:
1. `log_id`: monotonic ID, format `L0001`, `L0002`, ...
2. `timestamp`: ISO-8601 with timezone (example: `2026-02-11T21:30:00-08:00`)
3. `type`: one of `PLAN_UPDATE`, `DECISION`, `RISK`, `VERIFICATION`, `CODEMAP_DIFF`
4. `module`: one of `GLOBAL`, `A`, `B`, `C`, `D`, `E`, `F`, `G`
5. `trigger`: why this entry was created (user request / gate failure / design change)
6. `change_summary`: one-line factual change
7. `rationale`: explicit reasoning or hypothesis behind the change
8. `artifacts`: affected files or outputs (exact paths)
9. `evidence`: command/test/check output summary (or `N/A` for plan-only changes)
10. `impact`: scope + risk level (`low`/`medium`/`high`) + confidence (`0.00-1.00`)
11. `status`: `open`, `in_progress`, `done`, `blocked`, `superseded`
12. `next_action`: exact next step
13. `refs`: optional links to codemap diff ID or plan section

### 8.3 Logging Triggers

Create a new `LOG.md` entry whenever any of the following occurs:
1. A module plan section is added/expanded/changed.
2. Any open decision is frozen or changed.
3. A risk is discovered, severity changes, or mitigation is updated.
4. A TDD gate is executed for key module/function/pipeline checks.
5. A codemap diff is proposed/accepted/deprecated.

### 8.4 Quality Rules

1. Factual first: no vague wording like "optimized" without evidence.
2. One entry = one primary event; avoid bundling unrelated updates.
3. Use absolute timestamps and exact paths.
4. If evidence is missing, status cannot be `done`.
5. For `CODEMAP_DIFF`, include diff id in `refs`.
6. In `evidence` blocks that report test results, must explicitly list which plan tasks/gates were **not covered by automated tests** (e.g. "NOT tested: A0 config load, A5 schema write — trivial logic covered by e2e run only"). Omitting untested items is not allowed.
