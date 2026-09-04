# All-6388 Uricase Evolution, Tetramer Evidence, and DRB1*15:01 Core Release

## 1. Objective and claim boundary

Process every exact-sequence-unique IF-ready uricase in the canonical 6,388-row
case-study cohort through the same parent-specific evolutionary and tetramer-evidence
workflow used for Active-15, then emit near-WT DRB1*15:01 RF constraint manifests only
for cells with at least one WT strong core and at least one admissible core anchor.

The Q00511 alignment projection is retained as an independent functional prior and QC
annotation. It is not a cohort-membership gate. Every parent receives its own
query-centered MSA and PLMC model. A parent with no assay-confirmed WT activity remains a
computational redesign target, not an experimentally established activity-rescue parent.

This plan freezes one safety policy; it does not create a structure/evolution tier ladder.
It preserves the Active-15 release semantics: all safe P1/P4/P6/P9 anchors in every
actionable WT core remain editable, and every other parent-WT position is hard-fixed. It
does not replace that experimentally supported search-space definition with a minimum
hitting set.

## 2. Frozen inputs and frames

- Canonical cohort:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/uricase_caseset_if_ready_unified.parquet`
  (measured 2026-08-29: 6,388 rows, 6,388 unique `protein_id`, 6,388 unique
  `sequence`; source counts AFDB 4,958 / ESMFold2 1,429 / AF3 1).
- Historical Q00511-gated cohort labels:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/Uricases_RF/uricase_caseset_if_ready_dedup_labeled.parquet`.
  Preserve `n_anchors_matched`, `active_site_complete`, `viability`, and
  `design_viable` as independent retrospective annotations; none is a membership gate for this
  campaign. The current table contains 5,492 legacy design-viable and 896 legacy non-viable
  parents (measured 2026-08-29).
- Q00511 projection audit:
  `inverse_folding/reference_flow/configs/uricase_q00511_active_site_perprotein_v0.projection_audit.parquet`
  (6,388 parents x 9 Q00511 roles; target indices retained even when identity is not).
- WT DRB1*15:01 NMP peptide table:
  `/scratch/gpfs/KAIYIJIANG/zijie/run/benchmark/wt_v2/HLA-DRB1_15_01/wt_uricase_a1res03_HLA-DRB1_15_01_imm_full/imm_nmp_peptides.parquet`
  (25,410,420 rows; NetMHCIIpan rank values are percentage units).
- Strong-core rule: `rank_EL < 2.0`; `core_start_0b = pos + peptide.find(core)`;
  canonical anchor offsets P1/P4/P6/P9 = `0/3/5/8`.
- RF/backbone sequence authority is the canonical parquet `sequence`, not
  `original_sequence`. Every sequence/backbone coverage exception remains explicit QC.
- Allele scope: DRB1*15:01 only.
- Q00511 crystal remains a control, never a coordinate mask transferred to another parent.

Every materialized input records path, SHA-256, row count, unique IDs/sequences, sequence
MD5/SHA-256, length, structure source, IF-ready mapping fields, characterized/activity
annotations when available, Q00511 projection status, and NMP-core status.

## 3. Evidence and release policy

### 3.1 Parent-specific evolution

For every parent:

1. run the registered local ColabFold search against the same frozen databases;
2. normalize one strict query-centered focus MSA shared by conservation and PLMC;
3. fit a 500-iteration PLMC model with `theta=0.8`, `lambda_h=0.01`, and
   `lambda_J=0.01 * 20 * (L-1)`;
4. calculate stable `C80` using both gap-excluded conservation and WT identity frequency;
5. calculate `sigma80_robust` from the minimum percentile across top-0.5L/L/2L EC sets;
6. retain within-parent full-sequence Potts calibration and never compare raw Hamiltonians
   between parents.

Every model is fitted even when covariance is shallow. Sigma becomes a hard lock only when
`N_eff/L >= 10` and the later top-L tetramer-contact precision is at least 0.60.
Exploratory/unqualified sigma remains descriptive.

### 3.2 Q00511 prior and homolog functional mapping

The existing Q00511 projection audit is retained for all 6,388 parents, including mismatched
residue identities. A nonidentity mapped position is a candidate homolog analog, not an
automatic exclusion and not an automatic functional claim. The five-sample tetramer relay
geometry must independently resolve the catalytic D2 matching. Missing/ambiguous projection
or failed relay geometry is persisted as unresolved.

### 3.3 Parent-specific tetramer evidence

For every parent, use the same local-MSA Protenix route and five-sample contract as Active-15:
four identical protein copies, the same uricase ligand convention, and the frozen predictor
sampling identity. Derive parent-specific biological-interface matchings, 5 A residue-pair
contacts, two-symmetry-copy 5-of-5/4-of-5/1-of-5 masks, and the representative-sample Rosetta
alanine scan for the catalytic and assembly interfaces.

Complete four-chain sequence mapping, three resolvable D2 matchings, and one unique
bidirectional catalytic relay are required for `structure_qualified`. Every parent is run and
reported, but an unqualified ensemble cannot fabricate a `full_contact_5of5` safety claim.

### 3.4 Frozen strict policy and final manifests

For each non-clean parent, a candidate anchor is blocked by the production Active-15 union:

`C80 | eligible_sigma80_robust | full_contact_5of5 | functional_analog | ddG>=1_REU | scanned_noninterpretable_AGP`

- WT-clean parent: zero DRB1*15:01 strong cores; retain as a clean-control subset and emit no
  redesign manifest.
- Actionable core: at least one safe P1/P4/P6/P9 anchor.
- Blocked core: no safe canonical anchor.
- Emitted cell: at least one actionable core, complete sequence/backbone identity, and the
  evidence needed by the declared strict policy.
- Free set: union of all safe anchors from all actionable cores.
- Fixed set: exact parent-WT complement.
- No-actionable or unresolved-safety cell: retain a decision row and emit no manifest; never
  loosen the gate silently.

Every final sequence must later be rescored over the complete DRB1*15:01 landscape for retained,
shifted-register, and newly created strong cores.

## 4. Tasks and validation

### T0 - Freeze cohort and WT-core universe

Produce all-parent and DRB1*15:01 clean/non-clean FASTAs, ID lists, core tables, source hashes,
and a cohort manifest under a caller-supplied `work/` root.

Acceptance:

- exactly 6,388 unique IDs and sequences;
- exact parity with the canonical parquet;
- NMP summary/peptide failures empty;
- clean/non-clean sets disjoint and exhaustive;
- every long-table core sequence equals the parent sequence slice;
- every core has exactly P1/P4/P6/P9 positions in bounds.

The measured preflight value is 89 WT-clean parents and 6,299 non-clean parents. It is an
acceptance assertion for these frozen inputs, not a generic constant.

Retrospective comparison must retain the old labels: among the 896 parents excluded by the old
Q00511 identity gate, 18 are DRB1*15:01 WT-clean, the distinct strong-core median is 4 (versus 3
inside the 5,492), and 508 still map all eight hard relay roles. These measured facts prohibit
using `design_viable=false` as a new exclusion rule.

### T1 - Materialize and audit the Q00511 prior

Build one long parent x role table from the existing projection audit without identity-based
cohort exclusion. Preserve mapped/unmapped and identity/nonidentity states separately.

Acceptance: 6,388 x 9 rows; all mapped indices/AA identities reproduce the canonical sequence;
unmapped values remain explicit. The measured preflight is 5,958 parents with all eight relay
roles mapped and 430 with at least one missing relay role.

### T2 - Query-centered MSA and PLMC

Shard the 6,388-query local search and normalization deterministically by frozen parent order.
Fit one PLMC model per parent through the registered CPU array launcher. Retry incomplete shards
without overwriting completed content-addressed artifacts.

Acceptance: the 6,388-parent decision ledger is exhaustive. Every parent with a successful
exact-query search has normalized MSA QC records plus PLMC summaries/models/complete EC tables
and no identity/index/site failures. A parent may leave the compute cohort only through a signed,
reproducible terminal-search exclusion after deterministic retries; the current frozen exception
is `A0AAV7HJZ4`, leaving 6,387 computed parents. Optimizer termination remains diagnostic under
the existing validation contract.

### T3 - Evolution analysis

Run the registered evolution analysis over deterministic manifest shards and merge only after
every parent passes its ten-file schema and identity checks.

Acceptance: one complete per-parent output for every T2-computable parent (currently 6,387), plus
one consolidated QC table with qualified / exploratory / unqualified counts and persisted mask
memberships. The consolidated decision view retains the signed T2 exclusion rather than silently
converting it to missing evidence.

### T4 - WT tetramer prediction, relay, contact, and energy

Build parent-specific Protenix inputs from the T2 local A3Ms, predict five samples per WT, resolve
relay/interface classes, compute contact consensus, and run representative-sample Rosetta scans.

Acceptance: every parent has a terminal structure status; qualified parents have exactly five
distinct source hashes, complete four-chain mappings, stable D2 classification, contact masks,
and energy provenance bound to one of those five sources. Failure categories are explicit and
never converted into empty contact masks.

### T5 - Join DRB1*15:01 cores and emit strict manifests

Generalize the Active-15 join/builder only where cohort cardinality, zero-core handling, or
nullable structure status requires it. Preserve its gate semantics and fail-fast provenance.
The 5,255-parent evidence join may run as deterministic round-robin parent shards; each shard
must retain the full frozen-input identity, and the canonical join is published only after a
hash-validated disjoint/exhaustive merge of all five output tables.

Acceptance: exact parent/core/position joins; all free positions are safe canonical core anchors;
fixed/free is a disjoint exhaustive partition; every YAML round-trips through
`load_constraint_manifest` and validates against its parent; clean/blocked/unresolved cells emit
no manifest but remain in the decision ledger.

### T6 - Reproducible handoff

Persist commands, code/input hashes, job IDs, shard completeness, software identities, and
machine-readable QC. Update project status only after the corresponding data/config/pipeline
artifacts exist and pass their validation.

## 5. Cost and feasibility model

Measured Active-15 source costs:

- 15-query local ColabFold search: 3m54s on one GPU allocation.
- 15 PLMC fits: 39m06s sequential in one 32-CPU allocation.
- 15 evolution analyses: 5m57s sequential.
- 15 contact/energy analyses: 35m39s sequential after predictions existed.
- A validated Protenix five-sample model forward has been observed at 18.72 seconds for one
  target; this is a lower-bound model-forward unit, not a full campaign wall-time estimate.

Linear unsharded extrapolation gives approximately 277 hours for PLMC, 42 hours for evolution
analysis, and 253 hours for contact/energy CPU work. Deterministic arrays reduce wall time but not
total work. PLMC models alone are expected to occupy approximately 1.0-1.5 TB using the measured
160-232 MB per-parent range. Before full submission, the launch gate must measure available quota
and complete a representative 32-parent end-to-end structure pilot to bind actual Protenix
throughput/output size; the pilot does not remove any parent from the promised all-parent run.

No budget ceiling is imposed. Concurrency remains bounded by scheduler policy, storage capacity,
and artifact-integrity constraints; resource pressure never authorizes skipping a parent or an
evidence stage.

## 6. Stop conditions

Return to planning rather than weakening the scientific contract if:

- canonical sequence, NMP, backbone, or Q00511-prior identity cannot be reconciled;
- storage cannot hold the measured full artifact projection;
- local MSA or PLMC failures repeat after deterministic retry;
- the five-sample structure route cannot reproduce the Active-15 positive controls;
- generalizing the relay map would require assigning an unresolved catalytic mechanism by guess;
- emitted manifests change the Active-15 safe-anchor semantics.
