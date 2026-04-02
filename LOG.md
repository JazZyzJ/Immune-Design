# MHC-IF Deployment Log

This file is append-only and follows rules defined in the active stage plans (`PLAN.md`, `PLAN_IF.md`).

---

## Entry Template

```markdown
### LXXXX
- timestamp: YYYY-MM-DDTHH:MM:SS±HH:MM
- type: PLAN_UPDATE | DECISION | RISK | VERIFICATION | CODEMAP_DIFF
- module: GLOBAL | A | B | C | D | E | F | G | H | I | J | K | L | M | N
- trigger: <why this log entry exists>
- change_summary: <one-line factual statement>
- rationale: <reasoning or hypothesis>
- artifacts:
  - <absolute-or-workspace path>
- evidence: <tests/commands/check summaries, or N/A>
- impact:
  - scope: <what this change touches>
  - risk: low | medium | high
  - confidence: <0.00-1.00>
- status: open | in_progress | done | blocked | superseded
- next_action: <single concrete next step>
- refs:
  - <optional reference: codemap diff id / plan section>
```

---

### L0001
- timestamp: 2026-02-11T21:45:00-08:00
- type: PLAN_UPDATE
- module: GLOBAL
- trigger: User requested moving update log out of `PLAN.md` and enforcing scientific logging.
- change_summary: Migrated plan progress logging from `PLAN.md` to standalone `LOG.md` and standardized log schema.
- rationale: Separate strategy (`PLAN.md`) from execution history (`LOG.md`) improves traceability, auditability, and incremental control.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Plan/log document structure updated; no runtime tests required for documentation-only change.
- impact:
  - scope: Planning governance and project documentation workflow.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Continue module-by-module planning updates and append one log entry per requested module update.
- refs:
  - `PLAN.md:294`

### L0002
- timestamp: 2026-02-11T22:05:00-08:00
- type: PLAN_UPDATE
- module: A
- trigger: User requested first collaboration step to expand Module A execution plan.
- change_summary: Expanded `Module A` in `PLAN.md` from high-level bullets to task-level executable plan with TDD-aligned gates.
- rationale: Module execution needs finer granularity to reduce ambiguity and enable controlled handoff/verification.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only planning update; no runtime code/tests executed in this step.
- impact:
  - scope: Stage A planning precision, verification readiness, and execution governance.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Freeze Module A open decisions (`target_allele` and SA parser policy) before implementation kickoff.
- refs:
  - `PLAN.md:69`

### L0003
- timestamp: 2026-02-11T22:20:00-08:00
- type: DECISION
- module: A
- trigger: User selected `HLA-DRB1*07:01` and requested SA parsing policy decision based on `doc/analysis.md`.
- change_summary: Frozen Module A allele policy to dual profiles: `strict_sa` and `balanced_sa` with deterministic `0.5x` multi augmentation.
- rationale: `strict_sa` provides clean SA supervision; `balanced_sa` increases coverage while controlling noise for comparative experiments.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/doc/Epitope_Head_codemap.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: data summary used for decision: target allele appears in 74,428 rows; split includes 22,481 `high_res_single` and 51,947 `multi`.
- impact:
  - scope: Stage A data construction and downstream training-track definitions.
  - risk: medium
  - confidence: 0.97
- status: done
- next_action: Implement Module A parsers/filters with per-profile counters and deterministic sampling tests.
- refs:
  - `PLAN.md:96`
  - `doc/Epitope_Head_codemap.md:508`

### L0004
- timestamp: 2026-02-12T12:00:00-08:00
- type: VERIFICATION
- module: A
- trigger: User requested Module A execution (Tasks A0-A6).
- change_summary: Implemented and verified full Stage A pipeline producing span_records_strict.parquet (23,988 rows, 1,308 proteins) and span_records_balanced.parquet (54,945 rows, 4,069 proteins).
- rationale: Stage A converts raw TSV into normalized SpanRecord artifacts per codemap v0+d001 dual SA profile contract.
- artifacts:
  - `epitope_head/configs/data.yaml`
  - `epitope_head/configs/__init__.py`
  - `epitope_head/data/parse_mhc_if_v2.py`
  - `epitope_head/data/validators.py`
  - `epitope_head/data/build_dataset.py`
  - `tests/epitope_head/data/test_module_a_contract.py`
  - `outputs/manifests/span_records_strict.parquet`
  - `outputs/manifests/span_records_balanced.parquet`
- evidence: |
    17/17 unit tests pass (A2 explode: 5, A3 coords: 5, A4 allele filter: 7).
    Pipeline e2e: strict=23,988 rows, balanced=54,945 rows.
    Rejection ledger: allele_strict_kept=23,988; allele_multi_candidates=61,915; allele_multi_sampled=30,957; allele_rejected=1,156,398; pep_too_short=126,622; pep_too_long=1,142.
    Determinism: two identical runs produce matching counts and parquet digest (strict=75e38071d7a92fe5, balanced=231eb1742fa0acba).
- impact:
  - scope: Stage A data pipeline complete; unblocks Module B (FASTA join).
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Execute Module B (FASTA join + sequence validation) when user requests.
- refs:
  - `PLAN.md:69`
  - `doc/Epitope_Head_codemap.md:155`
  - codemap d001

### L0005
- timestamp: 2026-02-12T12:30:00-08:00
- type: PLAN_UPDATE
- module: A
- trigger: User noted PLAN.md listed test files (test_module_a_failures.py, test_module_a_reproducibility.py) that were never created, causing cross-session confusion.
- change_summary: Updated PLAN.md Module A file touchpoints to reflect actual test strategy — single consolidated test file + manual reproducibility check.
- rationale: Avoid phantom file references that mislead other sessions; document the pragmatic decision to skip trivial-logic unit tests and avoid slow reproducibility tests in CI.
- artifacts:
  - `PLAN.md`
- evidence: N/A (documentation-only change)
- impact:
  - scope: Module A planning accuracy and cross-session clarity.
  - risk: low
  - confidence: 1.00
- status: done
- next_action: Continue to Module B when user requests.
- refs:
  - `PLAN.md:86`

### L0006
- timestamp: 2026-02-12T17:43:13+08:00
- type: VERIFICATION
- module: A
- trigger: User requested audit of current `LOG.md` and Module A result validity before starting Module B.
- change_summary: Re-verified Module A artifacts/tests and confirmed Stage A is ready; noted minor documentation drift in prior test-count statement.
- rationale: Gate Module B start on objective evidence, not only historical log claims.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/span_records_strict.parquet`
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/span_records_balanced.parquet`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/data/test_module_a_contract.py`
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: `pytest -q tests/epitope_head/data/test_module_a_contract.py` => 19 passed; parquet sanity check => strict rows=23,988 proteins=1,308, balanced rows=54,945 proteins=4,069 with required SpanRecord columns present.
- impact:
  - scope: Module A readiness confirmation and log consistency maintenance.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Begin Module B plan expansion and execution gating.
- refs:
  - `LOG.md:L0004`
  - `PLAN.md:69`

### L0007
- timestamp: 2026-02-12T17:43:13+08:00
- type: PLAN_UPDATE
- module: B
- trigger: User approved moving forward to Module B after A validation.
- change_summary: Expanded Module B from high-level bullets to executable task-level plan (B0-B6) with dual-profile IO contracts and TDD-aligned release gates.
- rationale: Stage B has high failure risk at ID lookup and sequence consistency boundaries; explicit task decomposition reduces implementation ambiguity.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation update only; implementation commands/tests for Module B not executed in this entry.
- impact:
  - scope: Stage B planning precision and implementation readiness.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Execute Module B Task B0 with frozen join policy and rejection taxonomy.
- refs:
  - `PLAN.md:248`

### L0008
- timestamp: 2026-02-12T13:30:00-08:00
- type: VERIFICATION
- module: B
- trigger: User requested Module B execution.
- change_summary: Implemented and verified Stage B FASTA join pipeline for both profiles; 100% retention (zero drops across all validation gates).
- rationale: Stage B attaches protein sequences and validates span integrity per codemap v0 Stage B contract.
- artifacts:
  - `epitope_head/data/join_fasta.py`
  - `epitope_head/data/build_dataset.py` (updated with build_stage_b)
  - `tests/epitope_head/data/test_module_b_contract.py`
  - `outputs/manifests/span_records_with_seq_strict.parquet`
  - `outputs/manifests/span_records_with_seq_balanced.parquet`
- evidence: |
    11/11 unit tests pass (B1 lookup: 4, B3 boundary: 4, B4 match: 3).
    Pipeline e2e: strict 23,988→23,988 (100%), balanced 54,945→54,945 (100%).
    All lookups were exact_hit (FASTA contains both versioned and unversioned accessions).
    Zero drops: no unresolved, no out-of-range, no sequence mismatch.
    Protein seq length range: 59-6,977 AA. Unique proteins: strict=1,308, balanced=4,069.
    Determinism: two runs produce matching digests (strict=62f93dce1a056c6a, balanced=f97a18a127dffecf).
- impact:
  - scope: Stage B data pipeline complete; unblocks Module C (ProteinSample aggregation).
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Execute Module C (ProteinSample aggregation) when user requests.
- refs:
  - `PLAN.md:248`
  - `doc/Epitope_Head_codemap.md:169`

### L0009
- timestamp: 2026-02-12T18:15:56+08:00
- type: VERIFICATION
- module: B
- trigger: User requested follow-up review closure and explicit logging after fixing Module B findings.
- change_summary: Re-verified B1/B3/B4 fixes, added automated B2/B5/B6 gate tests, and updated Module B plan note for empty-output schema guarantee.
- rationale: Close the gap between release-level gates in `PLAN.md` and executable test coverage, while confirming recent bugfix behavior with reproducible evidence.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/epitope_head/data/join_fasta.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/data/test_module_b_contract.py`
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: |
    `pytest -q tests/epitope_head/data/test_module_b_contract.py` => 16 passed.
    New B2/B5/B6 coverage added and passing:
      - profile schema parity
      - required Stage C fields (including `protein_seq`)
      - repeated `build_stage_b` summary+artifact digest consistency
    Targeted behavior checks confirm:
      - `pep_len` invariant violations are dropped via `drop_len_invariant`
      - all-rejected input still yields canonical columns (no zero-column artifact).
- impact:
  - scope: Module B test-gate completeness(now has 16/16 tests) and contract reliability for Stage C handoff.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Proceed to Module C review/implementation when requested.
- refs:
  - `PLAN.md:370`
  - `PLAN.md:408`

### L0010
- timestamp: 2026-02-12T18:19:09+08:00
- type: VERIFICATION
- module: B
- trigger: User requested readiness check of completed Module B and decision on entering Module C.
- change_summary: Fresh verification passed for Module B tests, Stage B artifacts, and core invariants; no blocking issues found for C handoff.
- rationale: Entering Module C requires current evidence that Stage B outputs are contract-valid and reproducible enough for protein-level aggregation.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/data/test_module_b_contract.py`
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/span_records_with_seq_strict.parquet`
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/span_records_with_seq_balanced.parquet`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: |
    `pytest -q tests/epitope_head/data/test_module_b_contract.py` => 16 passed.
    Artifact checks:
      - strict rows=23,988, proteins=1,308, required Stage B columns complete.
      - balanced rows=54,945, proteins=4,069, required Stage B columns complete.
    Invariants on both outputs:
      - `pep_len == end_0b - start_0b` => True
      - `0 <= start_0b < end_0b <= len(protein_seq)` => True
      - `protein_seq[start_0b:end_0b] == peptide_seq` => True
    Cross-stage retention:
      - strict 23,988 -> 23,988 (100%)
      - balanced 54,945 -> 54,945 (100%)
- impact:
  - scope: Module B completion confidence and Stage C entry decision.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Start Module C planning/execution checks on `span_records_with_seq_*` inputs.
- refs:
  - `PLAN.md:248`
  - `PLAN.md:420`

### L0011
- timestamp: 2026-02-12T18:27:44+08:00
- type: PLAN_UPDATE
- module: C
- trigger: User explicitly requested writing Module C plan first before any further C checks.
- change_summary: Expanded Module C into executable plan (C0-C6) with dual-profile outputs, strict alias rule, and TDD-aligned verification gates.
- rationale: Restore planning-first workflow and remove ambiguity before C execution/validation.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation update only (no new execution claim in this entry).
- impact:
  - scope: Stage C planning completeness and workflow alignment.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Execute C0->C6 in order only after user confirms to proceed with C implementation checks.
- refs:
  - `PLAN.md:420`

### L0012
- timestamp: 2026-02-12T14:00:00-08:00
- type: VERIFICATION
- module: C
- trigger: User requested Module C execution.
- change_summary: Implemented and verified Stage C ProteinSample aggregation for both profiles with dedup, isolation, metadata, and strict alias.
- rationale: Stage C converts span-level records to protein-centric training units per codemap v0 ProteinSample contract.
- artifacts:
  - `epitope_head/data/make_protein_samples.py`
  - `epitope_head/data/build_dataset.py` (updated with build_stage_c)
  - `tests/epitope_head/data/test_module_c_contract.py`
  - `outputs/manifests/protein_samples_strict.parquet`
  - `outputs/manifests/protein_samples_balanced.parquet`
  - `outputs/manifests/protein_samples.parquet` (strict alias)
- evidence: |
    8/8 unit tests pass (C1 dedup: 3, C2 isolation: 2, C3 metadata: 3).
    Pipeline e2e:
      strict: 23,988 spans -> 1,308 proteins (4,913 unique positives, 19,075 duplicates removed)
      balanced: 54,945 spans -> 4,069 proteins (34,009 unique positives, 20,936 duplicates removed)
    All invariants validated (positive_count, duplicate_span_count, sequence_length, span boundaries).
    Strict alias matches strict output.
    Determinism: two runs produce matching digests (strict=17f095386e86ba71, balanced=5844a3ba5e0a784d).
- impact:
  - scope: Stage C data pipeline complete; unblocks Module D (split).
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Execute Module D (split + homology diagnostic) when user requests.
- refs:
  - `PLAN.md:420`
  - `doc/Epitope_Head_codemap.md:186`

### L0013
- timestamp: 2026-02-12T14:30:00-08:00
- type: CODEMAP_DIFF
- module: C
- trigger: User proposed adding per-span `support_n` to preserve observation frequency after dedup.
- change_summary: Added `support_n` field to each span in `positives_json`; codemap d002 accepted; v0 training ignores it.
- rationale: Dedup collapses repeated observations (e.g. P04114 span seen 58x → support_n=58). Retaining frequency enables future loss weighting and eval stratification without changing v0 behavior.
- artifacts:
  - `doc/Epitope_Head_codemap.md` (d002 added, Span definition updated)
  - `PLAN.md` (C0 actions updated)
  - `epitope_head/data/make_protein_samples.py` (support_n in span dict + validation)
  - `tests/epitope_head/data/test_module_c_contract.py` (updated: 9 tests)
  - `outputs/manifests/protein_samples_strict.parquet` (regenerated)
  - `outputs/manifests/protein_samples_balanced.parquet` (regenerated)
  - `outputs/manifests/protein_samples.parquet` (regenerated alias)
- evidence: |
    9/9 unit tests pass.
    Invariant verified: sum(support_n) == input_span_count for all proteins.
    Example: P04114 has 113 unique spans from 1,068 input records; top span support_n=58.
    Artifacts regenerated and pipeline deterministic.
- impact:
  - scope: ProteinSample schema (additive, non-breaking for v0 consumers).
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Execute Module D when user requests.
- refs:
  - `doc/Epitope_Head_codemap.md:d002`
  - `PLAN.md:456`

### L0014
- timestamp: 2026-02-12T23:21:26+08:00
- type: VERIFICATION
- module: C
- trigger: User requested review of Module C implementation quality before deciding to move to Module D planning.
- change_summary: Re-verified C tests and Stage C pipeline execution; no blocking contract violations found.
- rationale: Gate transition to D on fresh evidence rather than historical logs.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/epitope_head/data/make_protein_samples.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/data/build_dataset.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/data/test_module_c_contract.py`
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/protein_samples_strict.parquet`
  - `/Users/jerry/Project/MHC-IF/outputs/manifests/protein_samples_balanced.parquet`
- evidence: |
    `pytest -q tests/epitope_head/data/test_module_c_contract.py` => 9 passed.
    `python -m epitope_head.data.build_dataset --stage c` completed:
      - strict: 23,988 spans -> 1,308 proteins
      - balanced: 54,945 spans -> 4,069 proteins
    Spot-check invariants on outputs passed:
      - positive_count consistency
      - duplicate/input count relation
      - sum(support_n) == input_span_count
- impact:
  - scope: Module C completion confidence for handoff to split stage.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Expand Module D plan and prepare split-policy implementation checklist.
- refs:
  - `PLAN.md:420`

### L0015
- timestamp: 2026-02-12T23:21:26+08:00
- type: PLAN_UPDATE
- module: D
- trigger: User allowed moving to D planning after C review.
- change_summary: Expanded Module D from high-level bullets to executable task-level plan (D0-D6), including strict aliases, optional balanced track, homology diagnostics, and deterministic gating.
- rationale: Split stage is high risk for leakage; detailed planning reduces implementation ambiguity and evaluation bias.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation update only.
- impact:
  - scope: Stage D planning completeness and execution readiness.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Start D0 policy freeze and D1 strict split generation when requested.
- refs:
  - `PLAN.md:602`

### L0016
- timestamp: 2026-02-13T12:00:00-08:00
- type: VERIFICATION
- module: D
- trigger: User requested Module D execution (Tasks D0-D6).
- change_summary: Implemented and verified Stage D split pipeline for both profiles with disjointness validation, homology diagnostic, strict aliases, and determinism check.
- rationale: Stage D creates leakage-safe protein-ID splits per codemap v0 Stage D contract and flags homology risk for future upgrade.
- artifacts:
  - `epitope_head/data/create_splits.py`
  - `epitope_head/data/build_dataset.py` (updated with build_stage_d)
  - `tests/epitope_head/data/test_module_d_contract.py`
  - `outputs/manifests/splits/strict/train_ids.txt`
  - `outputs/manifests/splits/strict/val_ids.txt`
  - `outputs/manifests/splits/strict/test_ids.txt`
  - `outputs/manifests/splits/strict/split_diagnostics.json`
  - `outputs/manifests/splits/balanced/train_ids.txt`
  - `outputs/manifests/splits/balanced/val_ids.txt`
  - `outputs/manifests/splits/balanced/test_ids.txt`
  - `outputs/manifests/splits/balanced/split_diagnostics.json`
  - `outputs/manifests/splits/train_ids.txt` (strict alias)
  - `outputs/manifests/splits/val_ids.txt` (strict alias)
  - `outputs/manifests/splits/test_ids.txt` (strict alias)
- evidence: |
    13/13 unit tests pass (D1 split: 5, D2 disjointness: 5, D4 homology: 3).
    Pipeline e2e:
      strict: 1,308 proteins -> train=1,046 val=131 test=131
      balanced: 4,069 proteins -> train=3,255 val=407 test=407
    Disjointness: pairwise overlap=0 on both profiles. Union=full universe.
    Strict aliases match strict/ content: OK.
    Determinism: re-split with same seed produces identical ID sets.
    Homology diagnostic:
      strict: max_kmer_jaccard=0.9838, warning=True
      balanced: max_kmer_jaccard=1.0000, warning=True
    Note: high similarity is expected — proteins from same allele dataset include homologous families. Warning flag recorded for future homology-clustered split upgrade.
    NOT tested: D0 config freeze (trivial, values already in data.yaml), D3 balanced split isolation from strict (covered by e2e run + alias equality check, not as standalone unit test), D5 alias write (verified by e2e artifact check), D6 determinism (verified by re-split comparison, not as persistent unit test).
    Full test suite: 57/57 passed across all data modules (A+B+C+D).
- impact:
  - scope: Stage D data pipeline complete; unblocks Module E (training runtime). Homology warning flags dataset leakage risk for future mitigation.
  - risk: medium (homology risk flagged but accepted for v0 baseline)
  - confidence: 0.97
- status: done
- next_action: Execute Module E (training runtime) when user requests.
- refs:
  - `PLAN.md:595`
  - `doc/Epitope_Head_codemap.md:197`

### L0017
- timestamp: 2026-02-15T12:35:52+08:00
- type: PLAN_UPDATE
- module: D
- trigger: User requested v1.1 split policy update with seq_hash minimum unit and mmseq cluster-level isolation.
- change_summary: Upgraded Module D plan to `seq_hash -> mmseq90 cluster -> weighted cluster split` strategy and added a concise v1.1 proposal note.
- rationale: Prevent exact-sequence leakage and high-homology leakage across splits while keeping split loads balanced by training-relevant positive-span volume.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/doc/Epitope_Head_v1.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update (no implementation code changes in this step).
- impact:
  - scope: Stage D planning policy, split leakage control strategy, and v1.1 design notes.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Execute D0 policy freeze and D1-D6 implementation under the new seq_hash+cluster constraints when requested.
- refs:
  - `PLAN.md:595`
  - `doc/Epitope_Head_v1.md:338`

### L0018
- timestamp: 2026-02-15T13:00:00+08:00
- type: VERIFICATION
- module: D
- trigger: User approved v1.1 split plan with cov-mode/threads/config corrections and requested execution.
- change_summary: Implemented and verified Stage D v1.1 leakage-aware split with seq_hash grouping + mmseqs90 cluster isolation + weighted bin-packing for both profiles.
- rationale: v1.1 split eliminates exact-sequence leakage (4 seq_hash groups with >1 protein) and homology leakage (1,304→1,261 clusters at 90% identity) that v0 random split allowed.
- artifacts:
  - `epitope_head/data/create_splits.py` (rewritten: seq_hash, mmseqs wrapper, weighted split, cross-split validation)
  - `epitope_head/data/build_dataset.py` (build_stage_d rewritten for v1.1)
  - `epitope_head/configs/data.yaml` (added mmseqs_bin, cluster_identity, cluster_coverage, cluster_cov_mode)
  - `tests/epitope_head/data/test_module_d_contract.py` (rewritten: 15 tests)
  - `outputs/manifests/splits/strict/{train,val,test}_ids.txt`
  - `outputs/manifests/splits/strict/seq_hash_groups.parquet`
  - `outputs/manifests/splits/strict/mmseq90_clusters.tsv`
  - `outputs/manifests/splits/strict/cluster_assignment.parquet`
  - `outputs/manifests/splits/strict/split_diagnostics.json`
  - `outputs/manifests/splits/balanced/` (same set)
  - `outputs/manifests/splits/{train,val,test}_ids.txt` (strict aliases)
  - `PLAN.md` (D0 corrected: --cov-mode 2, --threads 1, mmseqs_bin config)
- evidence: |
    15/15 unit tests pass (D1 seq_hash: 4, D2 disjointness: 3, D3 weighted split: 5, D4 cross-split: 3).
    Pipeline e2e:
      strict: 1,308 proteins -> 1,304 unique sequences -> 1,261 clusters -> train=1,056 val=126 test=126
      balanced: 4,069 proteins -> 4,042 unique sequences -> 3,798 clusters -> train=3,259 val=404 test=406
    Weight fractions: strict {train: 0.7997, val: 0.1001, test: 0.1001}, balanced {train: 0.8, val: 0.1, test: 0.1}
    Invariants verified:
      - Pairwise disjointness: OK
      - Full coverage: OK
      - seq_hash non-crossing: OK (4 groups with >1 protein, all co-located)
      - cluster non-crossing: OK
      - Strict aliases match strict/ content: OK
    mmseqs params: --min-seq-id 0.90, -c 0.80, --cov-mode 2, --threads 1
    NOT tested: D0 config freeze (trivial), D2 mmseqs invocation (requires binary; tested via e2e pipeline run, not unit test), D5 alias write (verified by e2e artifact check), D6 determinism (verified by greedy algorithm determinism in unit test + artifact consistency, not as separate re-run test).
    Full test suite: 59/59 passed across all data modules (A+B+C+D).
- impact:
  - scope: Stage D v1.1 complete; eliminates exact-sequence and high-homology leakage. Unblocks Module E.
  - risk: low (leakage risk mitigated by cluster isolation)
  - confidence: 0.98
- status: done
- next_action: Execute Module E (training runtime) when user requests.
- refs:
  - `PLAN.md:595`
  - `LOG.md:L0016` (superseded by this v1.1 implementation)
  - `LOG.md:L0017` (plan update for v1.1)

### L0019
- timestamp: 2026-02-15T13:45:00+08:00
- type: PLAN_UPDATE
- module: E
- trigger: User requested review of completed Module D and expansion of Module E plan if no blocking logic issues remained.
- change_summary: Reviewed D plan/log consistency and expanded Module E from summary bullets to executable task-level plan (E0-E7) with TDD-aligned runtime gates.
- rationale: Stage E is the first model-training stage and carries the highest contract-coupling risk (splits, sampler leakage, tensor interfaces, artifact metadata), so implementation requires explicit per-task guardrails.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no production code or tests were modified/executed in this entry.
- impact:
  - scope: Stage E planning precision, runtime contract freezing checklist, and execution readiness.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Start Module E Task E0 (runtime protocol/config freeze) before any training runtime implementation.
- refs:
  - `PLAN.md:808`
  - `LOG.md:L0018`

### L0020
- timestamp: 2026-02-15T14:20:00+08:00
- type: PLAN_UPDATE
- module: E
- trigger: User requested audit of `doc/chunking.md` strategy and, if valid, full integration into Module E plan as first implementation baseline.
- change_summary: Validated chunking strategy as non-blocking and integrated complete long-protein chunking v1.1 policy into Module E plan, replacing default long-protein skip policy.
- rationale: Existing `skip if L>1022` policy discards supervision on long proteins; frozen chunk+stitch policy preserves training signal while keeping deterministic context constraints and seam diagnostics.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/chunking.md`
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no production code/tests executed in this entry.
- impact:
  - scope: Module E runtime planning, long-protein handling contract, release gates, and open-decision freeze state.
  - risk: medium
  - confidence: 0.97
- status: done
- next_action: Execute Module E Task E0 with chunking config freeze + codemap diff registration before runtime implementation.
- refs:
  - `PLAN.md:808`
  - `PLAN.md:1112`
  - `LOG.md:L0019`

### L0021
- timestamp: 2026-02-17T16:00:00+08:00
- type: VERIFICATION
- module: E
- trigger: User completed iterative review-implement cycle for Module E (Tasks E0-E7) and requested LOG entry.
- change_summary: Implemented and verified full Stage E training runtime: config validation, chunking datamodule, negative sampler, ESM-2 frozen encoder + scorer model, InfoNCE loss, trainer loop with checkpointing/logging, and smoke run diagnostics.
- rationale: Stage E is the first model-training stage; iterative review at each batch (E0-E2, E3, E4, E5-E7) caught and resolved real design issues including eval-mode lock, config key mismatches, span length guards, seam consistency, loss-config normalization, and checkpoint metadata completeness.
- artifacts:
  - `epitope_head/configs/__init__.py` (E0: deep validation of 21 top-level + 4 loss + 7 chunking sub-keys)
  - `epitope_head/configs/train.yaml` (E0: frozen training config)
  - `epitope_head/training/chunking.py` (E1: chunk plan, residue assignment, reliability)
  - `epitope_head/training/datamodule.py` (E1: make_collate_fn factory, token-budget batching)
  - `epitope_head/training/negatives.py` (E2: negative sampler with strict/non-strict modes)
  - `epitope_head/training/model.py` (E3: ESMTokenizer, FrozenESMEncoder, ProjectionHead, SpanFeatureBuilder, ScorerMLP, EpitopeScorer)
  - `epitope_head/training/losses.py` (E4: info_nce_loss, multi_positive_loss, smoothness_loss, compute_loss)
  - `epitope_head/training/trainer.py` (E5-E7: trainer loop, checkpointing, logging, smoke diagnostics)
  - `tests/epitope_head/training/test_module_e_contract.py` (101 tests)
- evidence: |
    `pytest -v tests/epitope_head/training/test_module_e_contract.py` => 101 passed in 2.62s.
    Test coverage by task:
      E0 config: 4 tests (valid load, missing top key, missing loss subkey, missing chunking subkey)
      E1 chunking/datamodule: 19 tests (chunk plan: 8, residue assignment: 3, reliability: 2, token budget: 3, collate: 3)
      E2 negatives: 9 tests (no-overlap, count, empty, bounds, hard-neg overlap, determinism, length-match, strict/non-strict shortfall)
      E3 model: 19 tests (projection: 1, span features: 6, scorer: 2, full model: 5, seam consistency: 2, span length guard: 2, tokenizer: 3)
      E4 losses: 19 tests (InfoNCE: 8, multi-positive: 3, smoothness: 3, compute_loss: 5 incl config key mapping + normalize alias)
      E5 trainer loop: 9 tests (NaN guard: 3, optimizer: 1, sanity metrics: 2, chunk spans: 2, train/val step: 4 + e6 artifacts: 1)
      E6 checkpointing/logging: 6 tests (checkpoint roundtrip + missing metadata, log schema + missing keys, config hash: 2)
      E7 smoke/diagnostics: 9 tests (smoke artifacts: 1, logit_gap finite: 1, reproducibility: 1, boundary buckets: 3, long-vs-short: 3)
    Key design decisions verified:
      - FrozenESMEncoder forces eval mode and overrides train() to prevent dropout
      - SpanFeatureBuilder raises ValueError for illegal span lengths (no silent clamp)
      - Seam consistency: identical scores for overlapping spans across adjacent chunks
      - normalize_loss_cfg maps config keys to compute_loss signature (tau_mp → T_mp alias)
      - Checkpoint metadata includes diff_ids_applied
      - per_protein_auc (pairwise AUC) in StepMetrics and log schema
      - Split train/val logs (train_log.jsonl, val_log.jsonl) with best.pt / epoch_N.pt naming
      - Smoke reproducibility: two runs with same seed produce identical metrics
      - Boundary-distance buckets: no severe score drift across d_boundary quantiles
- impact:
  - scope: Stage E training pipeline complete; unblocks Module F (inference/export).
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Execute Module F (inference + export) when user requests.
- refs:
  - `PLAN.md:808`
  - `PLAN.md:1069`
  - `LOG.md:L0019` (E plan update)
  - `LOG.md:L0020` (chunking integration)

### L0022
- timestamp: 2026-02-18T16:30:00+08:00
- type: VERIFICATION
- module: E
- trigger: User identified that `prepare_chunk_spans` only filters by `s>=chunk_start && e<=chunk_end`, causing duplicate supervision for positives in chunk overlap regions.
- change_summary: Implemented deterministic exactly-once span ownership via `assign_span_to_chunk` using expanded-interval trusted-interior rule `E_{s,k}=[s-1,s+k] ⊆ I_j` + nearest-center tie-break. Orphan positives raise ValueError instead of silent drop.
- rationale: Overlapping chunks (stride=512, context=1022 → overlap=510) would include the same positive in both chunks, causing biased gradients, inflated n_pos counts, and AUC double-counting. The fix matches `chunking.md` §5 spec and PLAN.md §8 (trusted-interior ownership rule). Silent drop was rejected to prevent the ">1022 丢正例" regression from reappearing under different parameters.
- artifacts:
  - `epitope_head/training/chunking.py` (new: `assign_span_to_chunk`)
  - `epitope_head/training/datamodule.py` (modified: `build_chunk_samples` pre-assigns positives with orphan guard)
  - `tests/epitope_head/training/test_module_e_contract.py` (new: `TestSpanOwnership` — 5 tests)
  - `PLAN.md` (E1 actions updated with span ownership details)
  - `doc/Epitope_Head_v1.md` (§2.1.1 updated: v1.1 now reflects implemented span ownership + seam verification)
- evidence: |
    `pytest -v tests/epitope_head/training/test_module_e_contract.py` => 106 passed in 3.57s.
    New tests:
      - test_short_protein_all_to_chunk_zero: single-chunk → all spans to chunk 0
      - test_overlap_span_assigned_to_nearest_center: overlap span goes to nearer chunk
      - test_exactly_once_long_protein: 3000 AA protein, 60 positives → sum across chunks == 60 (no dup, no loss)
      - test_deterministic: same input → same assignment
      - test_orphan_raises: context_len=10 forces orphan → ValueError
    Safety argument: current params (overlap=510, max_k=25, margin=32) guarantee every 12-25mer span fits in at least 2 chunks' trusted interiors, so owner==-1 is unreachable. ValueError guard is for future parameter changes.
- impact:
  - scope: Training gradient correctness, metric accuracy, and span-level loss attribution for chunked proteins.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Execute Module F (inference + export) when user requests.
- refs:
  - `doc/chunking.md:176` (§5 span ownership spec)
  - `PLAN.md:945` (E1 actions, span ownership)
  - `LOG.md:L0021` (Module E completion)

### L0023
- timestamp: 2026-02-18T17:10:00+08:00
- type: PLAN_UPDATE
- module: F
- trigger: User confirmed E-stage fixes and requested Module F plan authoring.
- change_summary: Expanded Module F from high-level bullets to executable task-level plan (F0-F7) with codemap-aligned inference/export contracts, TDD gates, and release-level verification gates.
- rationale: Stage F requires strict definition of checkpoint loading, full-window enumeration, aggregation formulas, long-protein inference behavior, and canonical JSON export to avoid downstream comparability drift.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no production code/tests executed in this entry.
- impact:
  - scope: Stage F planning precision, inference contract freezing checklist, and implementation readiness.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Start Module F Task F0 (inference protocol/config freeze) before implementation.
- refs:
  - `PLAN.md:1113`
  - `LOG.md:L0022`

### L0024
- timestamp: 2026-02-18T22:06:31+08:00
- type: VERIFICATION
- module: F
- trigger: User requested to start Module F execution and confirmed reliability should be emitted in debug.
- change_summary: Completed Module F batch-1 (F0-F2) with TDD: inference config contract, checkpoint-based predictor init, and short/long sequence encoding with deterministic chunk stitching + reliability debug output.
- rationale: F0-F2 defines the runtime contract and input correctness boundary for all downstream F3-F7 logic; implementing this first reduces risk of refactor churn in window scoring and export.
- artifacts:
  - `epitope_head/configs/__init__.py` (added `load_inference_config` validator)
  - `epitope_head/configs/inference.yaml` (frozen inference defaults)
  - `epitope_head/inference/__init__.py`
  - `epitope_head/inference/predictor.py` (checkpoint guards + encode short/long + stitch)
  - `tests/epitope_head/inference/test_module_f_contract.py` (F0-F2 contract tests)
- evidence: |
    RED:
      `python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py`
      -> ImportError: cannot import name `load_inference_config` (expected pre-implementation failure).
    GREEN:
      `python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py`
      -> 11 passed in 2.12s.
    Regression check:
      `python -m pytest -q tests/epitope_head/training/test_module_*_contract.py tests/epitope_head/inference/test_module_f_contract.py`
      -> 117 passed in 2.57s.
    NOT tested:
      - F3 full window enumeration/scoring loop
      - F4 hotspot/global-risk aggregation formulas + post-process
      - F5/F6/F7 JSON export, digest reproducibility, and smoke run artifacts
      - real ESM checkpoint compatibility on production weights (current tests use deterministic mock encoder)
- impact:
  - scope: Stage F runtime foundation established; unblocks F3-F4 implementation and review.
  - risk: medium
  - confidence: 0.96
- status: done
- next_action: Execute Module F batch-2 (F3-F4) with RED->GREEN tests for enumeration and aggregation formula contracts.
- refs:
  - `PLAN.md:1149` (F0)
  - `PLAN.md:1173` (F1)
  - `PLAN.md:1191` (F2)

### L0025
- timestamp: 2026-02-19T01:10:22+08:00
- type: VERIFICATION
- module: F
- trigger: User review reported two contract gaps in F0/F2 (stitch_mode handling and config freeze strictness).
- change_summary: Tightened F0 inference config freeze checks (`k=12..25`, chunking v1.1 fixed params, required per_residue_stitch) and added predictor fail-fast for unimplemented stitch modes.
- rationale: Prevent silent contract drift and ambiguous runtime behavior; ensure unsupported stitch mode cannot run with per_residue semantics by accident.
- artifacts:
  - `epitope_head/configs/__init__.py` (strengthened `load_inference_config` hard constraints)
  - `epitope_head/inference/predictor.py` (explicit `stitch_mode` fail-fast in `InferencePredictor.__init__`)
  - `tests/epitope_head/inference/test_module_f_contract.py` (new RED->GREEN tests for frozen k range, chunking enabled, stitch mode freeze, and predictor fail-fast)
- evidence: |
    RED:
      `python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py`
      -> 5 failed, 11 passed.
      Failures matched missing constraints:
        - min_k not frozen to 12 accepted
        - max_k not frozen to 25 accepted
        - chunking.enabled=false accepted
        - stitch_mode=center_weighted accepted by config loader
        - predictor did not reject unsupported stitch_mode
    GREEN:
      `python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py`
      -> 16 passed in 1.72s.
    Regression:
      `python -m pytest -q tests/epitope_head/training/test_module_*_contract.py tests/epitope_head/inference/test_module_f_contract.py`
      -> 122 passed in 2.14s.
    NOT tested:
      - center_weighted actual stitching implementation (still intentionally unsupported)
      - F3/F4 window scoring and aggregation formulas
      - F5/F6/F7 JSON export, digest reproducibility, and smoke CLI path
- impact:
  - scope: F0/F2 contract correctness and runtime safety for long-protein inference path selection.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Proceed to Module F batch-2 (F3-F4) with RED->GREEN tests for enumeration + aggregation.
- refs:
  - `PLAN.md:1149`
  - `PLAN.md:1191`
  - `LOG.md:L0024`

### L0026
- timestamp: 2026-02-19T22:53:19+08:00
- type: VERIFICATION
- module: GLOBAL
- trigger: User requested repository整理 for V1 and GitHub push readiness before cluster full run.
- change_summary: Prepared codebase for publish/run by adding Git ignore strategy, root runbook, and inference launcher scripts while preserving existing training/data/test contracts.
- rationale: Current workspace contained large local datasets and generated artifacts that should not be pushed; cluster-side clone requires clear executable entrypoints and reproducible run instructions.
- artifacts:
  - `.gitignore` (exclude large data/artifacts/caches from Git tracking)
  - `README.md` (root runbook for A-D build, E training, F inference, and tests)
  - `scripts/infer_v1.py` (inference CLI for single-seq and FASTA batch export)
  - `scripts/infer_v1.sh` (reproducible shell launcher for inference)
  - `epitope_head/configs/data.yaml` (kept user-owned mmseq config unchanged after temporary edit)
- evidence: |
    `python -m pytest -q tests/epitope_head/data/test_module_*_contract.py tests/epitope_head/training/test_module_*_contract.py tests/epitope_head/inference/test_module_f_contract.py`
    -> 237 passed in 4.83s.
    Sanity checks:
      - New inference launcher imports existing predictor/export stack and uses checkpoint metadata.
      - Existing training and inference contracts remain green after repository hygiene updates.
    NOT tested:
      - End-to-end cluster job execution of `scripts/infer_v1.sh` against a real ESM checkpoint on GPU
      - Real-world multi-protein FASTA throughput and wall-clock profiling
      - GitHub remote push/network operation itself (pending explicit push step)
- impact:
  - scope: Release packaging/readiness for V1 code delivery and cluster clone/run workflow.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Stage and commit V1 code/doc/script set, then push selected branch to origin for cluster clone.
- refs:
  - `README.md`
  - `scripts/infer_v1.py`
  - `scripts/infer_v1.sh`

### L0026
- timestamp: 2026-02-19T17:34:28+08:00
- type: VERIFICATION
- module: F
- trigger: User requested full-flow review of Module F implementation and asked whether Stage G planning can start.
- change_summary: Reviewed F0-F7 implementation and tests; functional path is complete and unit-contract suite passes, with two non-blocking completeness/robustness gaps recorded for follow-up.
- rationale: Stage handoff to G requires confirming F contracts are operational, while still surfacing issues that could weaken export/automation robustness.
- artifacts:
  - `epitope_head/inference/predictor.py`
  - `epitope_head/inference/export.py`
  - `tests/epitope_head/inference/test_module_f_contract.py`
  - `epitope_head/configs/inference.yaml`
  - `epitope_head/configs/__init__.py`
  - `scripts/` (checked for `infer_v1.sh`)
- evidence: |
    F-contract suite:
      `python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py`
      -> 41 passed in 4.90s.
    Cross-module regression attempt:
      `python -m pytest -q tests/epitope_head/training/test_module_*_contract.py tests/epitope_head/inference/test_module_f_contract.py`
      -> 3 failed, 144 passed.
      Failure cause: `ModuleNotFoundError: No module named 'esm'` in training tokenizer tests (environment dependency), not a newly introduced F logic regression.
    Static review findings (non-blocking for G planning):
      - `epitope_head/inference/export.py`: payload schema validator only checks first item of `window_logits` and `residue_hotspot`, so malformed later items can pass.
      - Planned touchpoint `scripts/infer_v1.sh` is not present yet.
- impact:
  - scope: Module F release-readiness assessment and Stage G handoff decision.
  - risk: medium
  - confidence: 0.96
- status: done
- next_action: Start Module G plan expansion; keep F export-validator hardening and inference CLI/script completion as follow-up items.
- refs:
  - `PLAN.md:1113`
  - `PLAN.md:1319`
  - `LOG.md:L0025`

### L0027
- timestamp: 2026-02-19T17:34:28+08:00
- type: PLAN_UPDATE
- module: G
- trigger: After F review, user requested to proceed with Module G planning.
- change_summary: Expanded Module G from high-level bullets into executable task-level plan (G0-G6) covering registry schema freeze, run identity/protocol signatures, strict registry writer, comparability guards, legacy backfill, and release smoke gates.
- rationale: Codemap Section 12 requires machine-checkable run traceability/comparability; expanding G into task-level gates prevents ad-hoc registry drift and makes release criteria testable.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update for planning; no production code edited in this entry.
- impact:
  - scope: Stage G execution readiness, verification-gate precision, and reproducibility governance.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Execute G0 (registry/comparability contract freeze) before any Module G code implementation.
- refs:
  - `PLAN.md:1319`
  - `LOG.md:L0026`

### L0028
- timestamp: 2026-02-19T18:36:40+08:00
- type: VERIFICATION
- module: GLOBAL
- trigger: User requested pre-H audit comparing `doc/Epitope_Head_v1.md` against A-G baseline implementation, with focus on optional/frozen features and extra delivered capabilities.
- change_summary: Completed spec-vs-implementation audit and updated `Epitope_Head_v1.md` with an explicit A-G status matrix (`implemented` / `implemented but frozen` / `not implemented`) plus a section listing scope additions beyond original v1 text.
- rationale: Entering H (cluster training) requires a frozen scope boundary to avoid accidental feature creep and to make first-run outcomes attributable.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Epitope_Head_v1.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation audit update; no production code changed. Comparison performed against current A-G modules (`data`, `training`, `inference`, `registry`) and corresponding config contracts.
- impact:
  - scope: H-stage scope freeze readiness and expectation alignment.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Start H planning with frozen baseline boundaries from `doc/Epitope_Head_v1.md` §6.
- refs:
  - `doc/Epitope_Head_v1.md:383`
  - `LOG.md:L0027`

### L0028
- timestamp: 2026-02-19T21:45:00+08:00
- type: VERIFICATION
- module: G
- trigger: User approved G plan (L0027) and requested G0-G6 implementation.
- change_summary: Implemented Module G (registry, protocol signature, comparability, backfill) with 31 tests; user applied post-implementation hardening to registry dedup guard, trainer registry integration, and F-module robustness fixes.
- rationale: Module G closes the run traceability and cross-run comparability requirements from PLAN.md/codemap §12; user hardening fixes prevent duplicate registry entries, ensure trainer auto-registers runs, and improve F-stage numeric robustness.
- artifacts:
  - `epitope_head/training/registry.py` (new: G0-G5 core logic + user-added dedup guard in `append_registry_row`)
  - `epitope_head/training/trainer.py` (modified: G3 integration — `registry_path` param, `run_id`/`manifest_version`/`protocol_signature` in `Trainer.__init__`, auto-append in `fit()`)
  - `epitope_head/inference/predictor.py` (user-hardened: min_k/max_k config resolution with model range check, centering on finite-only residues)
  - `epitope_head/inference/export.py` (user-hardened: protein_ids/payload_digests length mismatch guard in `write_prediction_summary`)
  - `tests/epitope_head/training/test_module_g_contract.py` (new: 31 tests)
- evidence: |
    G-contract suite:
      `python -m pytest -q tests/epitope_head/training/test_module_g_contract.py`
      -> 31 passed in 2.57s.
    Cross-module regression:
      `python -m pytest -q tests/epitope_head/training/test_module_e_contract.py tests/epitope_head/inference/test_module_f_contract.py tests/epitope_head/training/test_module_g_contract.py`
      -> 178 passed in 3.63s (E=106, F=41, G=31).
    Test coverage by task:
      - G0 schema: 5 tests (valid row, missing key, empty run_id, missing logit_gap, wrong timestamp type)
      - G1 identity: 5 tests (run_id format, deterministic seed, signature deterministic, key-order invariant, value-change diverges)
      - G2 writer: 6 tests (valid append, schema reject, missing path reject, checkpoint consistency reject, sequential multi-append, empty load)
      - G3 integration: 3 tests (build_registry_row schema-valid, build_from_summary, optional prediction_digest)
      - G4 comparability: 5 tests (identical comparable, manifest mismatch, protocol mismatch, missing key raises, write report)
      - G5 backfill: 4 tests (valid runs, skip invalid, skip duplicates, empty dir)
      - G6 smoke: 3 tests (full flow, incompatible detection, backfill+compare)
    NOT tested:
      - `scripts/register_run_v1.sh` not yet created (planned touchpoint deferred)
      - Live trainer→registry integration with real ESM encoder (covered by mock in E-contract suite)
    User-applied hardening (not from automated TDD cycle):
      - `registry.py:226-232`: dedup guard rejects duplicate run_id in `append_registry_row`
      - `trainer.py:29-34,511,567-570,706-730`: G3 integration wiring (imports, Trainer params, auto-register in fit)
      - `predictor.py:91-112`: min_k/max_k inference config override with model-range validation
      - `predictor.py:293-303`: centering skips -inf residues to avoid NaN
      - `export.py:221-225`: length mismatch guard in `write_prediction_summary`
- impact:
  - scope: Module G complete — run traceability, comparability, and backfill contracts are operational. F-stage numeric robustness improved.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: All modules A-G complete. Next steps: real training run on strict profile, or user-directed follow-up (scripts, documentation sync, etc.).
- refs:
  - `PLAN.md:1319`
  - `LOG.md:L0027`

### L0029
- timestamp: 2026-02-20T00:00:00+08:00
- type: DECISION
- module: E
- trigger: First training run on strict profile showed severe overfitting from epoch 1 — val/loss_total monotonically increasing (4→13) while train/loss_total decreased (3.5→0.5); train AUC reached 0.98 but val AUC stagnated at 0.79-0.83.
- change_summary: Hyperparameter and architecture tuning to address overfitting on small dataset (23,988 rows, 1,308 proteins, 126 val proteins).
- rationale: |
    Root cause analysis from training curves:
    - train/logit_gap grew to 4.5 while val/logit_gap only reached 1.7 → model memorizing train set
    - val/mean_pos_logit and val/mean_neg_logit both declining → model producing increasingly extreme logits that diverge from train distribution
    - val/loss_total monotonically increasing despite val/logit_gap increasing → tau=0.1 amplifies absolute logit scale differences exponentially

    Changes applied (10 items):

    **P0 — Config (train.yaml):**
    1. `loss.tau: 0.1 → 0.3` — tau=0.1 multiplies logits by 10x before softmax, causing extreme sensitivity to logit scale mismatch between train/val. With only 7 negatives per positive, tau=0.3 is more appropriate for small-data regime.
    2. `monitor_metric: logit_gap → loss_total` — logit_gap was monotonically increasing on val, so early stopping never triggered. loss_total directly reflects generalization.
    3. `early_stopping_patience: 10 → 5` — with 126 val proteins, 10 epochs of patience wastes compute on clearly diverging runs.

    **P0 — Architecture (model.py, model.yaml):**
    4. `ScorerMLP`: added `Dropout(0.3)` after GELU activation — zero dropout was the primary regularization gap; 0.3 is standard for small MLP heads.
    5. `ProjectionHead`: added `LayerNorm` after linear projection — normalizes scale of projected embeddings before span feature concatenation, preventing scale disparity between the 5 span components (mean_pool, endpoints, flanks).

    **P1 — Optimization (train.yaml):**
    6. `lr: 1e-3 → 3e-4` — 1e-3 too aggressive for ~340K trainable params on frozen 652M encoder; 3e-4 is standard for projection heads.
    7. `weight_decay: 1e-4 → 1e-2` — with dropout now present, stronger L2 provides complementary regularization; 1e-2 is standard AdamW decoupled weight decay.

    **P1 — Trainer (trainer.py):**
    8. Early stopping direction: added `monitor_mode` ("min" for loss metrics, "max" for gap/auc) so `loss_total` correctly triggers early stop when val loss stops decreasing.
    9. Warmup: changed from flat plateau at full LR to true linear ramp from 0 → base_lr over warmup_steps. Previous implementation exposed model to full lr=1e-3 from step 0.

    **P2 — Trainer stability (trainer.py):**
    10. Val RNG reset: `val_rng` now resets to fixed seed each epoch for deterministic val metrics, eliminating sampling noise in early stopping signal.
    11. Empty-chunk filtering: `aggregate_epoch_metrics` now excludes chunks with n_pos=0, preventing zero-metrics from diluting epoch averages.
- artifacts:
  - `epitope_head/configs/train.yaml`
  - `epitope_head/configs/model.yaml`
  - `epitope_head/training/model.py`
  - `epitope_head/training/trainer.py`
- evidence: N/A — changes are pre-run tuning based on diagnostic curves from cerulean-glade-1 baseline run. Next run will validate.
- impact:
  - scope: Training hyperparameters, model architecture regularization, trainer loop correctness.
  - risk: medium
  - confidence: 0.85
- status: done
- next_action: Re-run training on strict profile and compare val/loss_total curve against cerulean-glade-1 baseline.
- refs:
  - `LOG.md:L0021` (Module E baseline)
  - W&B run: cerulean-glade-1

### L0030
- timestamp: 2026-02-20T19:15:17+08:00
- type: VERIFICATION
- module: E
- trigger: User requested logic review for val-metric update in commit range `f509332..7a41491`, and asked to sync this improvement into LOG.
- change_summary: Integrated per-protein full-window validation metrics into training loop (`pp_auc`, `pp_ap`, `pp_recall_50`, `pp_recall_100`), switched monitor target to `pp_auc`, and passed real `val_entries + tokenizer` from training script so val metrics are computed on full candidate windows rather than chunk-level proxy only.
- rationale: Previous val signal was dominated by chunk-sampled proxy metrics; full-scan per-protein ranking metrics better match release objective and make checkpoint/early-stop selection align with inference-time behavior.
- artifacts:
  - `epitope_head/training/eval_metrics.py` (new: full-window val evaluator + AUC/AP/Recall@K)
  - `epitope_head/training/trainer.py` (val_epoch merge per-protein metrics; monitor fallback when metric missing)
  - `epitope_head/configs/train.yaml` (monitor metric changed to `pp_auc`; patience/lr/wd/tau retuned)
  - `scripts/train_v1.py` (wired `val_entries`, `tokenizer`, `min_k`, `max_k` into `Trainer`)
  - `epitope_head/training/model.py` (scorer dropout adjusted)
- evidence: |
    Commit diff:
      `git diff --stat f509332..7a41491`
      -> 5 files changed, 289 insertions(+), 9 deletions(-).
    Logic check:
      - `Trainer.val_epoch()` now conditionally calls `full_val_eval(...)` and merges per-protein metrics into val epoch logs.
      - `Trainer.fit()` now supports missing monitor metric values with direction-aware fallback (`-inf`/`inf`), preventing crashes when metric unavailable.
      - `scripts/train_v1.py` now provides required eval resources (`val_entries`, `tokenizer`) so `pp_auc` can be computed in normal runs.
    Residual risk:
      - Registry auto-write path still mirrors non-`logit_gap` monitor values into `primary_metrics.logit_gap`; this is semantically imprecise when monitoring `pp_auc` and should be corrected in a follow-up patch.
- impact:
  - scope: Validation metric semantics, checkpoint selection criterion, and training-runtime evaluation fidelity.
  - risk: medium
  - confidence: 0.93
- status: done
- next_action: Add a small patch to registry metric mapping so `logit_gap` comes from actual val logs (or set `None`) when monitor metric is `pp_auc`, then add a contract assertion for this case.
- refs:
  - `7a41491`
  - `f509332`
  - `LOG.md:L0029`

### L0031
- timestamp: 2026-02-26T23:23:14+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User requested starting Epitope Head v2 planning and asked to expand Module 1 (encoder ablation) into a complete hierarchical execution plan in `PLAN.md`.
- change_summary: Rebuilt `PLAN.md` as v2-oriented modular plan and added full Module H execution design for encoder ablation (`E0/E1/E2`) with frozen constants, encoder parameter specs, task-level gates, evaluation protocol (ranking + mutation sensitivity + latency), and decision freeze criteria.
- rationale: Module H is the v2 entry gate; without a strict, testable plan, encoder conclusions would be confounded by hidden config drift and incomplete evaluation panels.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: N/A (planning/documentation update only; no runtime code execution in this entry)
- impact:
  - scope: v2 module kickoff readiness, ablation reproducibility, and governance consistency.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Start Module H Task H0 by wiring ablation config schema validation and encoder-type guardrails before implementation.
- refs:
  - `PLAN.md:44`
  - `PLAN.md:111`
  - `PLAN.md:274`

### L0032
- timestamp: 2026-03-04T14:41:56+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User reported accidental deletion of encoder-ablation section in `doc/Epitope_Head_V2.md` and requested restoration.
- change_summary: Reconstructed and restored V2 encoder-ablation proposal, including motivation, E0/E1/E2 matrix, expected-outcome interpretation, LoRA stance, evaluation focus, and an implementation-aligned proposal section consistent with current `PLAN.md` Module H freezes.
- rationale: V2 design document must remain the high-level hypothesis/decision reference; missing ablation section breaks experiment traceability and weakens downstream decision auditability.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Epitope_Head_V2.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation restore only (`doc/Epitope_Head_V2.md` now contains Section 1.1-1.6 and Section 2 heading); no runtime tests executed.
- impact:
  - scope: V2 design/spec integrity for encoder ablation phase.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: If needed, continue filling Section 2 (`Split Evaluation`) with the same frozen-policy level as Module H before implementation kickoff.
- refs:
  - `doc/Epitope_Head_V2.md:7`
  - `PLAN.md:44`

### L0033
- timestamp: 2026-03-04T17:27:05+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User requested review of newly added two diagnostics in V2 encoder-ablation proposal and asked to write them into PLAN if valid.
- change_summary: Integrated the two diagnostics into Module H Task H5/H6 and release gates, while correcting diagnostic-B implementation to activation-based motif extraction (instead of directly reading conv1 kernel as 21-channel AA logo under embedding-channel input).
- rationale: The diagnostics are valid for judging whether CNN is a reasonable method, but direct conv-weight logo interpretation would be methodologically inconsistent with current E1 encoder design (`token embedding dim=256`); activation-based extraction preserves interpretability under the actual architecture.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation update only; no runtime tests executed in this entry.
- impact:
  - scope: Module H evaluation protocol and CNN-rationality evidence quality.
  - risk: low
  - confidence: 0.96
- status: done
- next_action: Implement H5 diagnostic exports (per-protein AUC distribution table + activation-based motif artifacts) before decision freeze in H6.
- refs:
  - `doc/Epitope_Head_V2.md:91`
  - `doc/Epitope_Head_V2.md:104`
  - `PLAN.md:210`

### L0035
- timestamp: 2026-03-05T00:57:43+08:00
- type: PLAN_UPDATE
- module: I
- trigger: User requested review of newly added proposal section (3 parts) for encoder optimization focused on loss and CNN refinement, and asked to sync into `PLAN.md` if sound.
- change_summary: Added Stage-I (`Module I`) CNN enhancement plan covering margin-based hard-example loss (`L1`), multi-scale CNN fusion (`C1`), combined variant (`LC1`), AP-centric decision rules, and lightweight deployment constraints; also added milestone `M6` and updated log schema module enum to include `I`.
- rationale: Proposal direction is valid, but requires explicit implementation constraints to avoid confounded conclusions: compatibility with current loss API, deterministic hard-negative fallback, and parameter/latency budget guards for "lightweight CNN" claims.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation/planning update only; no runtime code/test execution in this entry.
- impact:
  - scope: V2 post-ablation optimization roadmap and CNN feasibility decision quality.
  - risk: low
  - confidence: 0.96
- status: done
- next_action: Freeze open decision on Stage-I run mode (`mixed_margin` only vs `mixed+margin_only`) and then start Module I Task I0.
- refs:
  - `doc/Epitope_Head_V2.md:123`
  - `PLAN.md:44`
  - `PLAN.md:299`

### L0034
- timestamp: 2026-03-04T20:08:00+08:00
- type: VERIFICATION
- module: E
- trigger: User requested checking latest pulled code for `model.py` module additions and logit constraints, and asked to sync LOG if missing.
- change_summary: Confirmed commit `9a96e60` introduced a new bounded-logit scorer path in `model.py` (cosine similarity + learnable `logit_scale` with upper clamp), and propagated related config/runtime wiring for cluster training.
- rationale: Previous MLP direct logit head allowed unconstrained logit magnitude growth; CLIP-style normalized scoring with bounded scale is intended to stabilize InfoNCE training and reduce validation loss blow-up from extreme logits.
- artifacts:
  - `epitope_head/training/model.py` (replaced final linear-logit head with normalized similarity head + `logit_scale` constraint)
  - `epitope_head/configs/model.yaml` (added `logit_scale_init`, `logit_scale_max`)
  - `epitope_head/configs/train.yaml` (set `loss.tau` to `1.0` with scale moved into scorer)
  - `epitope_head/training/trainer.py` (W&B logging adds runtime `logit_scale`)
  - `scripts/train_v1.py` (wires new scorer/logit-scale config into model init)
- evidence: |
    Commit scan:
      `git show --name-only --oneline 9a96e60`
      -> files include `model.py`, `trainer.py`, `train.yaml`, `model.yaml`, `train_v1.py`.
    Logic scan:
      `git show 9a96e60 -- epitope_head/training/model.py`
      -> `ScorerMLP` changed to hidden projection + normalized prototype similarity;
         `logit_scale = exp(log_logit_scale).clamp(max=logit_scale_max)`;
         output is bounded by learned scale.
    LOG check:
      `rg -n \"9a96e60|logit_scale|ScorerMLP\" LOG.md`
      -> no prior entry for this commit before this update.
- impact:
  - scope: Training scorer parameterization and logit-scale control on cluster run path.
  - risk: medium
  - confidence: 0.95
- status: done
- next_action: Run at least one strict-profile training smoke and track `logit_scale` trajectory with `val/pp_auc` to confirm stability gain over previous head.
- refs:
  - `9a96e60`
  - `7a41491`

### L0036
- timestamp: 2026-03-05T11:40:00+08:00
- type: CODEMAP_DIFF
- module: I
- trigger: Module H ablation complete (E1 winner); Module I implements two controlled improvements — margin-based hard-example loss (L1) and multi-scale CNN feature fusion (C1) — to improve AP-centric ranking quality.
- change_summary: Implemented Module I full variant matrix (B0/L1/C1/LC1) including margin hard-example loss, MultiScaleDilatedCNNEncoder, config profiles, training script variant routing, and 22 new contract tests; 188 total tests passing.
- rationale: E1 (Dilated CNN) won Module H with pp_AUC=0.972, pp_AP=0.272, Recall@50=0.705. Module I applies two independent improvements: (1) L1 adds `mixed_margin` loss with top-k hard negative mining to improve precision in the AP curve; (2) C1 replaces single-branch dilated conv with parallel {3,5,9}-kernel branches plus 1×1 fusion, widening receptive-field diversity without full-model redesign. All improvements are hyperparameter-controlled; defaults reproduce B0 (E1 baseline) exactly.
- artifacts:
  - `epitope_head/training/losses.py` (added `margin_hard_loss()`, extended `compute_loss()` with `objective_mode`/`margin_m`/`hard_topk`/`lambda_margin`)
  - `epitope_head/training/trainer.py` (normalize_loss_cfg optional keys; StepMetrics.loss_margin; LOG_ENTRY_KEYS; aggregate_epoch_metrics; compute_sanity_metrics)
  - `epitope_head/training/encoders.py` (added `_MultiScaleResidualBlock`, `MultiScaleDilatedCNNEncoder`; updated `build_encoder()` for `multiscale_cnn`)
  - `epitope_head/configs/__init__.py` (registered `multiscale_cnn` in validator with required cfg keys)
  - `epitope_head/configs/train.yaml` (added margin loss defaults: objective_mode=infonce, margin_m=0.5, hard_topk=8, lambda_margin=0.5)
  - `epitope_head/configs/model_ablation.yaml` (added B0/L1/C1/LC1 profiles and stage_i_frozen_constants block)
  - `scripts/train_v2_ablation.py` (added --variant-id; loss_overrides merge; Stage-I routing to cnn_enhance_v2/ with pp_ap monitor)
  - `tests/epitope_head/training/test_module_i_loss_contract.py` (14 tests: I1 margin_hard_loss, I2 objective_mode variants, I3 trainer schema)
  - `tests/epitope_head/training/test_module_i_encoder_contract.py` (8 tests: I4 shape/padding, I5 param budget/factory, I6 config validator, I7 integration)
  - `tests/epitope_head/training/test_module_h_encoder_contract.py` (relaxed profile-set assertion to superset check)
  - `tests/epitope_head/training/test_module_e_contract.py` (updated for loss_margin key and tau=1.0)
- evidence: |
    python -m pytest tests/epitope_head/training/ -v --tb=short
    -> 188 passed, 0 failed, 5 warnings
    Module I tests: 22 new tests (14 loss + 8 encoder) all pass.
    Module H regression: 24 tests all pass.
    Module E regression: 142 tests all pass (3 updated for schema changes).
    MultiScale param budget: ~2.8M params = 1.06× E1 baseline (well within 1.5× budget).
- impact:
  - scope: Loss function, encoder architecture, config system, training script, test suite.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Run smoke tests for B0/L1/C1 variants (`python scripts/train_v2_ablation.py --variant-id {B0,L1,C1} --seed 42 --device cpu --smoke`) then launch full matrix on cluster.
- refs:
  - `.claude/plans/sprightly-growing-riddle.md`
  - `PLAN.md §Module I`

### L0037
- timestamp: 2026-03-05T16:58:14+08:00
- type: CODEMAP_DIFF
- module: F
- trigger: User required flank ablation inference to run only on CNN encoder variants and reject non-CNN paths.
- change_summary: Converted flank ablation inference to CNN-only variant flow using `model_ablation.yaml` profiles (`E1/B0/L1/C1/LC1`), removed ESM/mock branches, added explicit non-CNN fail-fast validation, and updated SLURM launcher to pass `--variant-id B0`.
- rationale: Current experiments are CNN-only; keeping ESM/default paths in this script can silently run the wrong encoder family and invalidate the ablation conclusion.
- artifacts:
  - `epitope_head/inference/flank_ablation.py`
  - `scripts/flank_ablation_test.py`
  - `scripts/submit_flank_ablation.slurm`
  - `tests/epitope_head/inference/test_flank_ablation.py`
- evidence: |
    python -m pytest -q tests/epitope_head/inference/test_flank_ablation.py
    -> 4 passed
    python -m pytest -q tests/epitope_head/inference/test_module_f_contract.py
    -> 41 passed
    python scripts/flank_ablation_test.py --help
    -> CLI includes --variant-id and no mock/ESM switches
- impact:
  - scope: Flank ablation inference entrypoint and validation guardrails.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Run one checkpoint with `--variant-id` matching training run directory (e.g., `B0` or `C1`) and compare `pp_auc_ablated` vs `pp_auc_original`.
- refs:
  - `PLAN.md`

### L0038
- timestamp: 2026-03-05T17:10:00+08:00
- type: CODEMAP_DIFF
- module: I
- trigger: CNN encoders have no context-length limit but were still using ESM-2's 1022-residue chunking pipeline, splitting long proteins during training and skipping 34/126 val proteins (27%) during evaluation.
- change_summary: Bypassed chunking for CNN encoders in `train_v2_ablation.py` — CNN variants now train on full-length proteins and evaluate all val proteins including those >1022 residues.
- rationale: DilatedCNN (RF=121 AA) and MultiScaleCNN can process arbitrary-length sequences in a single forward pass. Chunking was only needed for ESM-2's 1024-token hard limit. Removing it eliminates chunk-boundary artifacts, lets the model see full protein context, and includes previously-skipped long proteins in pp_auc/pp_ap evaluation.
- artifacts:
  - `scripts/train_v2_ablation.py` (conditional `ck_params` based on `encoder_type`; pass `ck_params["context_len"]` to Trainer for eval)
- evidence: |
    Code change: 2 edits in `train_v2_ablation.py`.
    For CNN: `context_len = max(protein_lengths)`, `stride = same`, `margin = 0` → 1 chunk per protein.
    TokenBudgetSampler handles oversized proteins safely (always yields ≥1 sample per batch).
    `full_val_eval` skip condition `if L > context_len` is now never triggered for CNN.
- impact:
  - scope: Training data pipeline and per-protein evaluation coverage for CNN encoder variants.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Re-run LC1 on cluster with full-protein mode and compare pp_ap/pp_auc with previous chunked run; verify pp_n_skipped=0 in val logs.

### L0039
- timestamp: 2026-03-17T14:08:05+08:00
- type: PLAN_UPDATE
- module: J
- trigger: User requested a rigorous review of `doc/NetMHCIIpan_mutation_augmentation.md` and asked to write a new augmentation-stage `PLAN.md` if the proposal was sound.
- change_summary: Rebuilt `PLAN.md` as a dedicated Data augmentation stage plan centered on NetMHCIIpan mutation augmentation, and folded draft corrections into the frozen contract: train-only isolation, explicit WT gray-zone skip policy, full `affected_spans` recomputation, and offline materialized augmented train artifacts instead of late lazy mutation inside `ChunkSample.__getitem__`.
- rationale: The draft direction is valid, but several implementation assumptions would break against the current codebase if left uncorrected: the current dataloader wraps `ChunkSample`, not mutable `ProteinEntry`; reusing base `protein_id` would leak augmented rows through split-id filtering; and recording only the seed disrupted span is not label-hygienic when one mutation affects multiple known positives.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/NetMHCIIpan_mutation_augmentation.md`
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: |
    Documentation/planning update only; no runtime code or tests executed in this entry.
    Planning decisions grounded against current code contracts:
      - `epitope_head/training/datamodule.py` loads immutable `ProteinEntry` -> `ChunkSample` before Dataset access
      - `epitope_head/data/build_dataset.py` and split ids are keyed by `protein_id`
      - `protein_samples_strict.parquet` is the strict-train source artifact; `protein_samples.parquet` is only a strict alias
    NOT tested:
      - NetMHCIIpan standalone invocation
      - pilot throughput
      - augmented train artifact materialization
- impact:
  - scope: Data augmentation stage planning, artifact design, train/eval isolation policy.
  - risk: medium
  - confidence: 0.97
- status: done
- next_action: Freeze the two remaining Stage-J open decisions (`per-source cap` and `top-delta selection policy`) before implementation starts with Task J0/J1.
- refs:
  - `PLAN.md:1`
  - `PLAN.md:73`
  - `doc/NetMHCIIpan_mutation_augmentation.md:142`

### L0040
- timestamp: 2026-03-17T14:17:06+08:00
- type: DECISION
- module: J
- trigger: User clarified Stage-J design intent for chunking, multi-positive handling, augmented sample identity, and first-run quantity control.
- change_summary: Frozen Stage-J as chunking-agnostic, required full recomputation of `affected_spans` for multi-positive proteins, defined augmented `protein_id` as sample-level identity with source provenance preserved, and selected `threshold + per-source topN` as the first-run retention policy with `topN=64`.
- rationale: The user's intent is to add mutation-derived negatives at the source-protein level without tying augmentation to any specific chunking strategy. In the current codebase, the safest realization is to keep augmented samples as separate train rows with unique sample keys while preserving original source linkage in metadata; this keeps the biological intent intact and avoids split leakage or loader ambiguity.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation/decision update only; no runtime code or tests executed in this entry.
- impact:
  - scope: Stage-J artifact semantics, training integration policy, and retained-mutant volume control.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Start Stage-J implementation with Task J0/J1 using the frozen `topN=64` and chunking-agnostic contract.
- refs:
  - `PLAN.md:90`
  - `PLAN.md:297`
  - `PLAN.md:358`

### L0041
- timestamp: 2026-03-17T14:21:07+08:00
- type: PLAN_UPDATE
- module: GLOBAL
- trigger: User requested broadening the plan theme so the document represents the whole Data augmentation stage rather than only the current NetMHCIIpan route.
- change_summary: Updated `PLAN.md` top-level framing from a NetMHCIIpan-specific title/goal to a broader Data augmentation stage plan, while keeping `Module J` as the concrete current implementation track for NetMHCIIpan-guided mutation augmentation.
- rationale: The execution document should describe the stage-level objective and leave room for future augmentation routes, without obscuring that the active implementation path is still the NetMHCIIpan mutation pipeline.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only wording update; no runtime code or tests executed in this entry.
- impact:
  - scope: Top-level planning semantics and future extensibility of the augmentation roadmap.
  - risk: low
  - confidence: 0.99
- status: done
- next_action: Continue Stage-J implementation under the broadened Data augmentation framing, starting from Task J0/J1.
- refs:
  - `PLAN.md:1`
  - `PLAN.md:5`
  - `PLAN.md:7`

### L0042
- timestamp: 2026-03-17T15:30:00+08:00
- type: VERIFICATION
- module: J
- trigger: User approved plan review and confirmed design decisions (Q1-Q3), requested implementation start.
- change_summary: Implemented Stage-J Tasks J0-J7 — augmentation config, NetMHCIIpan runner abstraction, WT verification, mutation enumeration/dedup, scoring + affected-span recalculation, top-N selection, registry artifact builder, materialization, and training integration.
- rationale: Module J provides train-only data augmentation via NetMHCIIpan-guided single-point mutation disruption. Offline pipeline produces augmented ProteinSample artifacts that plug into existing training with zero model/loss changes.
- artifacts:
  - `epitope_head/configs/augmentation.yaml` (new — Stage-J config, isolated from data.yaml)
  - `epitope_head/data/netmhciipan_mutation.py` (new — config validation, WT classification, mutation enumeration, scoring, top-N)
  - `epitope_head/data/netmhciipan_runner.py` (new — runner abstraction: StandaloneRunner + MockRunner)
  - `epitope_head/data/materialize_augmented_samples.py` (new — registry→ProteinSample materialization + validation)
  - `epitope_head/training/datamodule.py` (modified — added load_augmented_proteins())
  - `scripts/run_mutation_pilot.py` (new — J1 pilot script)
  - `scripts/build_mutation_registry.py` (new — J2-J5 full registry builder)
  - `scripts/materialize_augmented_train_set.py` (new — J6 materialization script)
  - `scripts/train_v2_ablation.py` (modified — added --aug-train-parquet arg)
  - `tests/epitope_head/data/test_module_j_registry_contract.py` (new — 23 tests: J0 config, J2 WT classification, disruption checks)
  - `tests/epitope_head/data/test_module_j_materialization_contract.py` (new — 15 tests: J3 dedup, J5 top-N, J6 materialization + validation)
  - `tests/epitope_head/training/test_module_j_train_integration_contract.py` (new — 6 tests: J7 train/val isolation)
  - `outputs/augmentation/` (new output directory for Stage-J artifacts)
- evidence: |
    44/44 Module J tests passed (pytest).
    Pilot mock run: 4 proteins, 55.6% WT confirmed rate, 40.7% disruption yield.
    Existing test suite: 197 passed, 1 pre-existing esm import failure (unrelated).
    **Tasks NOT covered by automated tests:**
    - J1 pilot protein selection diversity (only tested via mock run, no unit test for stratification balance)
    - J4 affected-span recalculation with real NetMHCIIpan scores (tested with mock only)
    - J8 release gate summary completeness (not yet implemented)
    - StandaloneRunner output parsing (no real NetMHCIIpan binary available for integration test)
- impact:
  - scope: New offline augmentation pipeline + 1 modified datamodule function + 1 modified training script arg. No changes to model, loss, or evaluation.
  - risk: low
  - confidence: 0.92
- status: done
- next_action: Install NetMHCIIpan 4.3 (locally or on cluster), run real pilot with --backend standalone, then full registry build + materialization + augmented training experiment.
- refs:
  - `PLAN.md` Module J (Tasks J0-J7)
  - `doc/NetMHCIIpan_mutation_augmentation.md`

### L0043
- timestamp: 2026-03-18T16:00:06+08:00
- type: PLAN_UPDATE
- module: GLOBAL
- trigger: User requested opening the inverse-folding stage, auditing `doc/Inverse_Folding_v1.md`, and creating a dedicated `PLAN_IF.md`.
- change_summary: Created `PLAN_IF.md` for the inverse-folding stage, froze the v1 module structure (`K/L/M/N`), clarified Level 2 guidance guardrails, and broadened `LOG.md` module governance to include IF-stage entries.
- rationale: The inverse-folding work now has its own execution stage and should not remain implicit inside the epitope-head planning documents. The v1 proposal is coherent, but only if baseline training, shared evaluation, and Level 2 guidance are kept separated from later property-aware claims and from the still-under-specified Level 3 comparison route.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md`
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only planning/audit update; no runtime code or tests executed in this entry.
- impact:
  - scope: Inverse-folding stage planning, shared logging governance, and v1 scientific boundary definition.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Start Module K planning/execution by freezing the exact DPLM environment pin and the initial CATH smoke-run contract.
- refs:
  - `PLAN_IF.md:1`
  - `doc/Inverse_Folding_v1.md`
  - `doc/Immune_Design_Architecture_v2.md`

### L0044
- timestamp: 2026-03-23T16:15:00+08:00
- type: PLAN_UPDATE
- module: J
- trigger: User proposed a v1.1 refinement in `doc/Data_augmentation.md` to change how NetMHCIIpan-derived mutants are used during training.
- change_summary: Updated Stage-J planning from static union of all augmented rows to registry-primary runtime `p_aug` replacement, kept materialized augmented parquet as optional debug/audit output, and added zero-positive filtering plus exposure/determinism gates.
- rationale: The scientific idea is sound: near-duplicate augmented proteins should not dominate the training distribution. The only material correction is implementation placement: because the current pipeline builds `ChunkSample` objects from `ProteinEntry`, runtime mutation must happen at the epoch-level train-entry view before chunk expansion, not inside chunk-level `Dataset.__getitem__`. Filtering out mutants with `remaining_positive_count == 0` avoids spending batch budget on no-gradient samples.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
  - `/Users/jerry/Project/MHC-IF/doc/Data_augmentation.md`
- evidence: Documentation-only planning review/update; local code-path audit checked `epitope_head/training/datamodule.py`, `epitope_head/training/trainer.py`, `epitope_head/data/materialize_augmented_samples.py`, and `scripts/train_v2_ablation.py`. No tests executed in this entry.
- impact:
  - scope: Stage-J architecture, training integration policy, release gates, and risk controls.
  - risk: medium
  - confidence: 0.95
- status: done
- next_action: If Stage-J is revised in code, implement epoch-level runtime replacement over base train entries, add deterministic epoch-view tests, and log `effective_p_aug` separately from any sampler-tuning experiments.
- refs:
  - `PLAN.md:5`
  - `PLAN.md:299`
  - `doc/Data_augmentation.md`

### L0045
- timestamp: 2026-03-26T17:45:00+08:00
- type: CODEMAP_DIFF
- module: L
- trigger: Execution of PLAN_DATA_SEL.md tasks L0–L5 (data selection and evaluation pipeline).
- change_summary: Implemented three-tier test protein schema, MMseqs2 overlap detection, tier validation scripts, Tier 2 prescreen pipeline (dual-scorer + CATH topology sampling), test set assembly with post-hoc validation, and all supporting CLI/SLURM scripts. 131 tests pass.
- rationale: Module L provides the curated test set infrastructure that Module M (guidance sweep) and Module N (comparison baselines) depend on. All code-only tasks are complete; remaining L6/L7 require manual curation steps and cluster execution with real data.
- artifacts:
  - `inverse_folding/evaluation/schema.py` (added TEST_PROTEIN_COLUMNS, TIER_REQUIRED_FIELDS, validate_test_protein_entry)
  - `inverse_folding/evaluation/overlap.py` (new: MMseqs2 overlap wrapper)
  - `inverse_folding/evaluation/tier_validators.py` (new: Tier 1/3 candidate validators)
  - `inverse_folding/evaluation/prescreen.py` (new: Tier 2 dual-scorer filter)
  - `inverse_folding/evaluation/cath_topology.py` (new: CATH topology assignment + diversity sampling)
  - `inverse_folding/evaluation/assembly.py` (new: test set assembly + validation)
  - `scripts/run_overlap_filter.py` (new CLI)
  - `scripts/validate_tier1.py` (new CLI)
  - `scripts/validate_tier3.py` (new CLI)
  - `scripts/prescreen_tier2.py` (new CLI)
  - `scripts/assemble_if_test_set.py` (new CLI)
  - `scripts/submit_prescreen_tier2.slurm` (new SLURM)
  - `scripts/submit_assemble_test_set.slurm` (new SLURM)
  - `tests/inverse_folding/test_module_l_tier_schema.py` (30 tests)
  - `tests/inverse_folding/test_module_l_overlap.py` (10 tests)
  - `tests/inverse_folding/test_module_l_validate_tier1.py` (12 tests)
  - `tests/inverse_folding/test_module_l_prescreen.py` (11 tests)
  - `tests/inverse_folding/test_module_l_validate_tier3.py` (10 tests)
  - `tests/inverse_folding/test_module_l_assembly.py` (8 tests)
- evidence: `python -m pytest tests/inverse_folding/test_module_l_*.py -v` → 131 passed, 0 failed.
- impact:
  - scope: Module L data selection pipeline, evaluation infrastructure, downstream M/N contracts.
  - risk: low
  - confidence: 0.95
- status: in_progress
- next_action: Execute manual curation steps (IEDB query for Tier 1, PDB candidate pool for Tier 2, literature search for Tier 3) then run L6 E2E verification and L7 release gates on cluster.
- refs:
  - `PLAN_DATA_SEL.md:§L0–L5`

### L0046
- timestamp: 2026-03-31T20:35:00-04:00
- type: DECISION
- module: J
- trigger: User request to extend epitope head training to DRB1*04:01 and DRB1*15:01.
- change_summary: Added multi-allele support via output_subdir config field; generated data manifests, augmentation configs, and SLURM scripts for DRB0401 and DRB1501.
- rationale: Multi-allele coverage is needed for paper Figure F2 (multi-allele analysis). Reuse existing LC1_lite_aug pipeline (CNN encoder + p_aug=0.20) with per-allele data isolation via output_subdir in config.
- artifacts:
  - `epitope_head/data/build_dataset.py` (added _resolve_manifest_dir helper)
  - `epitope_head/configs/data_drb0401.yaml` (new)
  - `epitope_head/configs/data_drb1501.yaml` (new)
  - `epitope_head/configs/augmentation_drb0401.yaml` (new)
  - `epitope_head/configs/augmentation_drb1501.yaml` (new)
  - `scripts/submit_aug_drb0401.slurm` (new)
  - `scripts/submit_aug_drb1501.slurm` (new)
  - `scripts/submit_train_drb0401.slurm` (new)
  - `scripts/submit_train_drb1501.slurm` (new)
  - `outputs/manifests/drb0401/` (1865 proteins: train=1490, val=186, test=189)
  - `outputs/manifests/drb1501/` (1536 proteins: train=1236, val=153, test=147)
- evidence: Pipeline stages A-D completed locally for both alleles. Splits verified. SCP to cluster pending (VPN required).
- impact:
  - scope: Epitope head multi-allele extension; no changes to existing DRB1*07:01 artifacts.
  - risk: low
  - confidence: 0.95
- status: in_progress
- next_action: SCP manifests to cluster, run augmentation (sbatch submit_aug_*.slurm), then training (sbatch submit_train_*.slurm).
- refs:
  - `PLAN.md:§Module J`
  - `PROGRESS.md:§Epitope Head`

### L0047
- timestamp: 2026-04-01T02:00:33-04:00
- type: BUGFIX
- module: L
- trigger: Tier 2 prescreen cluster run failed because `submit_prescreen_tier2.slurm` pointed MMseqs2 to a nonexistent `cath_train_seqs.fasta` while the actual CATH 4.3 dataset on cluster is stored as `chain_set.jsonl` plus `chain_set_splits.json`.
- change_summary: Updated Tier 2 prescreen CATH input handling to accept `chain_set.jsonl` directly for both MMseqs2 overlap filtering and topology lookup, materialize a train-only temporary FASTA for MMseqs2, and switched the SLURM launcher to pass the JSONL source path instead of nonexistent FASTA/domain-list artifacts.
- rationale: The Tier 2 pipeline should stay split-safe and consume the canonical CATH 4.3 storage format already used elsewhere in the repo, rather than depending on separately prepared FASTA or CathDomainList artifacts that may drift or be missing on cluster.
- artifacts:
  - `inverse_folding/evaluation/overlap.py` (added `chain_set.jsonl` support with train-split FASTA materialization)
  - `tests/inverse_folding/test_module_l_overlap.py` (added coverage for JSONL train-only materialization and missing-splits guard)
  - `inverse_folding/evaluation/cath_topology.py` (added `chain_set.jsonl` topology lookup via entry `CATH`)
  - `tests/inverse_folding/test_module_l_prescreen.py` (added topology assignment coverage for `chain_set.jsonl`)
  - `scripts/prescreen_tier2.py` (updated CLI help text to document FASTA or JSONL source)
  - `scripts/submit_prescreen_tier2.slurm` (now passes `${CATH_DIR}/chain_set.jsonl`)
- evidence: Manual Python verification passed for (1) JSONL source materializes only the `train` split and flows through `run_mmseqs_overlap`, (2) missing sibling `chain_set_splits.json` fails fast with `FileNotFoundError`, and (3) topology lookup returns the first `CATH` code directly from `chain_set.jsonl`. Local `pytest` execution was not available in the current shell because `pytest` is not installed in that environment.
- impact:
  - scope: Module L Tier 2 prescreen overlap/topology stages and SLURM launcher compatibility with cluster CATH 4.3 data layout.
  - risk: low
  - confidence: 0.91
- status: done
- next_action: Re-run `scripts/submit_prescreen_tier2.slurm` on cluster and, once in the correct env, execute `python -m pytest tests/inverse_folding/test_module_l_overlap.py -q` for full automated confirmation.
- refs:
  - `PLAN_DATA_SEL.md:§L0–L5`

### L0048
- timestamp: 2026-04-02T00:30:00+08:00
- type: CODEMAP_DIFF
- module: N
- trigger: N1 Level 1 post-hoc filter baseline implementation.
- change_summary: Implemented N1 (argmin-risk selection from M3 candidates) with TDD, CLI script, and SLURM submission script.
- rationale: Level 1 filter reuses M3 eta=0 candidate pool (K=8) with argmin selection — same compute budget, different selection rule. Provides the "generate then filter" comparison arm for paper. No new model training required.
- artifacts:
  - `inverse_folding/baselines/__init__.py` (new)
  - `inverse_folding/baselines/level1_filter.py` (new: parse_candidates_fasta, select_argmin_candidate, filter_protein_set)
  - `scripts/run_if_level1_filter.py` (new CLI)
  - `scripts/submit_if_level1_filter.slurm` (new SLURM, follows submit_cnn_enhance.slurm template)
  - `tests/inverse_folding/test_module_n_level1_filter.py` (11 tests)
- evidence: `python -m pytest tests/inverse_folding/ -v` → 236 passed, 0 failed.
- impact:
  - scope: Module N comparison baseline; reads M3 outputs, writes same format for evaluate_if.py.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Execute after M3 sweep completes on cluster (sbatch submit_if_level1_filter.slurm).
- refs:
  - `PLAN_IF.md:§Task N1`
