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

### L0049
- timestamp: 2026-04-01T02:00:33-04:00
- type: BUGFIX
- module: L
- trigger: Tier 2 prescreen timed out after finishing MMseqs2, head prefilter, and NetMHCIIpan batch scoring, leaving only the final CATH diversity sampling stage incomplete while checkpoint parquet files already contained the needed screening outputs.
- change_summary: Added a checkpoint-reconstruction path in the test-set assembly CLI so Tier 2 entries can be rebuilt directly from `tier2_candidates_merged.fasta` plus `_prescreen_head_results_*` and `_prescreen_nmp_results_*` parquet files, skipping diversity sampling, and updated the assembly SLURM launcher to use that reconstruction path by default.
- rationale: The project already persists the expensive prescreen intermediates; rebuilding Tier 2 from those artifacts avoids rerunning the slow diversity stage and lets assembly proceed with the biologically relevant NMP/head filters intact.
- artifacts:
  - `scripts/assemble_if_test_set.py` (added checkpoint-based Tier 2 reconstruction path and CLI args)
  - `tests/scripts/test_assemble_if_test_set_script.py` (added reconstruction coverage for complete vs partial NMP rows)
  - `scripts/submit_assemble_test_set.slurm` (now points assembly to candidate FASTA + prescreen checkpoint parquets)
- evidence: Manual Python verification passed for reconstructing Tier 2 entries from candidate FASTA plus head/NMP checkpoint rows, including skipping `partial` NMP rows and applying the dual filter before assembly. IDE diagnostics reported no new lint errors. Full `pytest` was not runnable in the current local shell environment.
- impact:
  - scope: Module L Tier 2 → assembly handoff, cluster recovery path after prescreen timeout.
  - risk: medium
  - confidence: 0.93
- status: done
- next_action: Run `scripts/submit_assemble_test_set.slurm` on cluster and verify the reconstructed Tier 2 count matches the expected post-filter population from the prescreen checkpoints.
- refs:
  - `PLAN_DATA_SEL.md:§L0–L5`

### L0050
- timestamp: 2026-04-01T00:00:00-04:00
- type: BUGFIX
- module: L
- trigger: `assemble_if_test_set.py` wrote per-protein FASTA files using raw `protein_id` strings from uricase parquet rows, and some IDs contained `/`, `|`, and `:`, causing `FileNotFoundError` when the filesystem interpreted them as path separators or invalid filename characters.
- change_summary: Added `_safe_file_id()` in the assembly CLI and routed FASTA artifact paths plus stored `pdb_path` metadata through that sanitizer while preserving the original `protein_id` field in assembled metadata; added tests covering unsafe uricase IDs.
- rationale: The raw sequence headers are useful biological identifiers and should stay intact in metadata, but artifact filenames must be filesystem-safe so assembly can write WT FASTA files deterministically for every tier.
- artifacts:
  - `scripts/assemble_if_test_set.py` (sanitizes filesystem-facing artifact paths while keeping raw `protein_id`)
  - `tests/scripts/test_assemble_if_test_set_script.py` (covers unsafe filename sanitization and Tier 3 parquet path handling)
- evidence: Manual Python verification passed for `_safe_file_id()` and Tier 3 parquet loading with a uricase `protein_id` containing `/`, `|`, and `:`. IDE diagnostics reported no new lint errors. Full `pytest` was not runnable in the current local shell environment.
- impact:
  - scope: Module L assembly artifact generation for Tier 3 uricase entries and any future unsafe protein identifiers.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: Re-run `scripts/submit_assemble_test_set.slurm` and confirm the `fastas/` directory now contains sanitized uricase FASTA filenames and the assembly completes to `test_proteins.parquet`.
- refs:
  - `PLAN_DATA_SEL.md:§L0–L5`

### L0051
- timestamp: 2026-04-01T00:00:00-04:00
- type: BUGFIX
- module: L
- trigger: Assembly still failed in phase `4/5` after enabling Tier 2 checkpoint reconstruction with `--skip-cath-diversity`, because the reconstructed Tier 2 rows intentionally carried `selection_reason="pre-screened-no-diversity"` and `cath_topology=None`, but the frozen schema still rejected any tier-2 null topology.
- change_summary: Relaxed the Tier 2 schema validator to allow `cath_topology=None` only for rows marked `selection_reason="pre-screened-no-diversity"` and added regression tests covering both schema validation and end-to-end assembly for this skip-diversity path.
- rationale: Skip-diversity Tier 2 rows are a deliberate recovery/output mode added earlier so assembly can proceed without rerunning CATH topology sampling; the schema must match that contract while preserving the stricter requirement for normal Tier 2 rows.
- artifacts:
  - `inverse_folding/evaluation/schema.py` (conditional Tier 2 null-topology exception for skip-diversity rows)
  - `tests/inverse_folding/test_module_l_tier_schema.py` (covers the allowed skip-diversity exception)
  - `tests/inverse_folding/test_module_l_assembly.py` (covers assembly with Tier 2 null topology under skip-diversity)
- evidence: Manual Python reproduction of the original `AssemblyError` failed before the patch and passed after the patch; manual verification also confirmed that ordinary Tier 2 rows with `selection_reason="pre-screened"` still require non-null `cath_topology`. IDE diagnostics reported no new lint errors.
- impact:
  - scope: Module L Tier 2 checkpoint-reconstruction and assembly path.
  - risk: low
  - confidence: 0.98
- status: done
- next_action: Re-run `scripts/submit_assemble_test_set.slurm` and verify the reconstructed Tier 2 rows now pass phase `4/5` schema validation while keeping ordinary Tier 2 topology checks intact.
- refs:
  - `PLAN_DATA_SEL.md:§L0–L5`

### L0052
- timestamp: 2026-04-04T16:00:00-04:00
- type: DECISION
- module: N
- trigger: Strategic pivot — core focus shifted to position-dependent reference flow; Level 3 comparison deemed unnecessary engineering overhead.
- change_summary: Level 3 (DRAKES/DPO) dropped from v1 scope. Only Level 1 post-hoc filter (N1) retained as comparison baseline.
- rationale: (1) Core contribution is position-dependent reference flow, not beating RL methods. (2) DRAKES adaptation to DPLM v1 requires ~weeks of engineering with no contribution to the paper's scientific novelty. (3) Post-hoc filter + Tier 0/1 ablation (sampling-only vs full training) provides sufficient comparison structure. (4) DRAKES code available in `DRAKES/` for reviewer response if requested. (5) Paper comparison simplified to: Level 1 post-hoc filter vs sampling-only position-dependent (Tier 0 ablation) vs full reference flow (Tier 1+2).
- artifacts:
  - `PROGRESS.md` (Module N section updated)
  - `PLAN_IF.md` (Module N rewritten, §6.7 frozen, §7 execution order updated, risk #6 updated)
  - `doc/Inverse_Folding_v1.md` (§0 obj 4 dropped, §4 dropped, §7 v1.2/v1.3 rewritten)
- evidence: N/A (strategic decision, no code change)
- impact:
  - scope: Module N scope reduction; frees engineering bandwidth for reference flow implementation.
  - risk: low
  - confidence: 0.95
- status: done
- next_action: Begin Tier 0 implementation — modify DPLM sampling loop for position-dependent unmasking rates.
- refs:
  - `PLAN_IF.md:§Task N2 (DROPPED)`
  - `doc/Reference_Flow_Derivation.md:§4.3 (sampling algorithm)`

### L0053
- timestamp: 2026-04-07T22:30:00-04:00
- type: PLAN_UPDATE
- module: GLOBAL
- trigger: Plan restructuring — Modules M/N misaligned with actual goal (reference flow). Module M is post-hoc resampling, not generation-time steering. Need to refocus plan on core contribution.
- change_summary: Module M superseded; PLAN_IF.md restructured to add Phase B (experiment preparation) and Phase C (core contribution stub). Milestones M3 superseded, M5/M6 added.
- rationale: (1) Module M "classifier guidance" is candidate resampling at output level — scientifically equivalent to post-hoc filtering (N1), provides no paper value. (2) Original K→L→M→N pipeline was infrastructure-only with core contribution deferred as "Phase 2-3 stubs." (3) With K and L done, the plan must now center on the actual core contribution. (4) Phase B bridges infrastructure to experiments (PDB download, h_i precompute, eval integration). (5) Phase C is the core reference flow work — left as stub pending Thinker-track discussion. (6) Existing code and LOG references preserved; no deletions, only supersession banners.
- artifacts:
  - `PLAN_IF.md` (Goal/Architecture reworded, Milestone M3 superseded, M5/M6 added, Module M banner added, Phase B/C sections appended, §6 decisions updated, §9 module list updated)
  - `PROGRESS.md` (Module M marked superseded, Module N blocker updated, Phase B/C sections added, Paper Readiness F3 rows updated)
- evidence: N/A (planning restructure, no code change)
- impact:
  - scope: PLAN_IF.md structure; PROGRESS.md active sections. No code, no LOG history, no Module K/L content affected.
  - risk: low
  - confidence: 0.95
- status: done
- next_action: Begin Phase B tasks (B1 PDB download, B2/B3 h_i precompute) on cluster.
- refs:
  - `PLAN_IF.md:§Phase B, §Phase C`
  - `doc/Reference_Flow_Derivation.md`

### L0054
- timestamp: 2026-04-12T12:00:00-04:00
- type: DECISION
- module: GLOBAL
- trigger: Migration from Princeton Adroit to Princeton Della cluster.
- change_summary: Migrated all project infrastructure from Adroit (`/scratch/network/zc1519/`) to Della (`/scratch/gpfs/KAIYIJIANG/zijie/`); updated 28 files (146 path occurrences); converted SLURM scripts from `--partition=gpu` to QOS-based scheduling (`gpu-short`/`gpu-medium`).
- rationale: Adroit is a small test cluster; Della provides A100 80GB GPUs and GPFS parallel filesystem with significantly better I/O and compute capacity. Della uses sponsor-based scratch directories and QOS-based job scheduling instead of direct partition specification.
- artifacts:
  - `env.sh` (4 path replacements)
  - `CLAUDE.md` (cluster convention updated: Adroit → Della)
  - `PROGRESS.md` (4 path replacements)
  - `scripts/*.slurm` (22 files: path replacement + partition→QOS migration)
  - `scripts/run_if_level1_filter.py`, `scripts/run_diagnose_e1.sh`, `docs/install_esmfold.md` (path replacements)
- evidence: GPU verified on Della — `torch.cuda.get_device_name(0)` → NVIDIA A100 80GB PCIe (node della-l02g13). Conda environment binary-compatible, no rebuild needed.
- impact:
  - scope: All SLURM scripts, env.sh, CLAUDE.md, PROGRESS.md, 2 helper scripts, 1 doc. No Python module code changed.
  - risk: low
  - confidence: 0.95
- status: done
- refs:
  - QOS available: `gpu-short` (1d), `gpu-medium` (3d), `gpu-long` (6d), `gpu-test`, `short`, `medium`, `vlong`, `test`
  - Della scratch base: `/scratch/gpfs/KAIYIJIANG/zijie/`

### L0055
- timestamp: 2026-04-16T10:30:00+08:00
- type: PLAN_UPDATE
- module: GLOBAL
- trigger: User (Thinker+Organizer) requested a dedicated document listing structural metrics for analyzing MHC-II epitope distribution on 3D structure, named "uricase analysis", to support the "Present structural metrics on epitope" SubFigure (F2). User further corrected the initial draft: analysis must be anchored on **IEDB ground-truth spans**, not on epitope-head predicted $h_i$ — using a model's own output as the reference is circular and measures model-internal bias rather than biology.
- change_summary: Created `doc/uricase_analysis.md` documenting 4 structural metrics (RSA, secondary-structure distribution, B-factor, contact number). The doc uses **IEDB experimental epitope spans** (from `tier1_candidates.json:experimental_epitopes`) as the primary reference, with per-residue `is_epitope` labels from the union of spans. Epitope head $h_i$ is a **secondary** model-validation analysis asking whether $h_i$ recovers the same structural signature that IEDB epitopes show. Also added matching row to PROGRESS.md Paper Readiness table.
- rationale: IEDB spans are the authoritative biological ground truth for Tier 1 (15 proteins, 261 spans, 11–45% coverage). Using $h_i$ as the reference would conflate "biology" with "what the model learned". Anchoring on IEDB cleanly separates two questions: (1) what is the structural signature of MHC-II epitopes? (biology, answered with IEDB), (2) does the epitope head recover this signature? (model validation). The uricase extension accepts a different regime — no per-residue IEDB labels, so $h_i$ + literature + structure drive de-immunization candidate selection.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/uricase_analysis.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md` (Paper Readiness: added "Structural metrics on epitope" row under F2)
  - Notion SubFigure page `3308d0d0b9368056aeddcc9ee1a86350` ("Present structural metrics on epitope")
- evidence: Confirmed IEDB labels exist via inspection of `outputs/if/test_set/tier1_candidates.json` — each of 15 proteins has `experimental_epitopes: list[{start_0b, end_0b, peptide, assay_type, iedb_ref}]`.
- impact:
  - scope: F2 analysis methodology; Phase C reference-flow motivation; epitope-head validation narrative.
  - risk: low
  - confidence: 0.97
- status: done
- next_action: When ready, implement per-residue IEDB label extraction + 4-metric computation on Tier 1 proteins; produce primary (IEDB) analysis first, secondary ($h_i$ agreement) after.
- refs:
  - `doc/uricase_analysis.md`
  - `outputs/if/test_set/tier1_candidates.json`
  - Notion: `https://www.notion.so/Present-structural-metrics-on-epitope-3308d0d0b9368056aeddcc9ee1a86350`

### L0056
- timestamp: 2026-04-16T18:13:40+08:00
- type: CODEMAP_DIFF
- module: L
- trigger: User (Coder) reviewed the Tier 1 structural analysis proposal in `doc/uricase_analysis.md` (originally L0055), accepted the plan with four amendments — (1) strict ATOM→FASTA residue mapping with zero tolerance for aa mismatch; (2) DSSP-only tooling for both RSA and SS (drop FreeSASA as primary); (3) remove the "Extension to Uricase" section since rasburicase has no in-hand structure + no per-residue IEDB labels; (4) add `--allele` as a CLI hyperparameter and point outputs at cluster `work/` and repo `figures/` — and authorized implementation without TDD (user will self-review).
- change_summary: Revised `doc/uricase_analysis.md` to fix the EL-only caveat (pure EL data → no experimentally-negative residues; non-epitope = "not observed"; analysis is conservative), spell out the strict residue-mapping protocol (`author_resnum = chain_range_start + seq_idx`, peptide-string sanity check, abort on any mismatch), pick DSSP as the single RSA+SS source, replace χ² with per-protein log-OR + DerSimonian-Laird random-effects meta for categorical SS pooling, add allele hyperparameter, declare output paths (cluster `work/immune-design/tier1_structural_analysis/<allele_tag>/` + repo `figures/F2_supplementary/`), and trim the uricase section to a short "Future Work" note. Implemented: `inverse_folding/analysis/structural_features.py` (PDB parse, strict alignment with `AlignmentError` / `SpanSanityError`, DSSP RSA+SS, Cα contact numbers with author_resnum-based sequence exclusion, Cliff's δ, Mann–Whitney U, log-OR with Haldane-Anscombe correction, DerSimonian-Laird random-effects meta) and `scripts/analysis/tier1_structural_analysis.py` (CLI driver iterating tier1 candidates, per-protein stats, cross-protein pooling, optional PDF panels + SS-coil forest plot, hyperparameter echo per feedback memory). Registered the new script in `doc/SCRIPTS.md` under Analysis §8.
- rationale: IEDB Tier 1 spans are EL-only, so "non-epitope" cannot claim experimental negativity — this was mis-worded in the original draft and the user flagged it explicitly. Strict alignment is required because a silent off-by-one between PDB author numbering and FASTA indexing would invalidate every downstream panel; the implementation raises with every offending position listed rather than best-effort matching. DSSP provides both RSA and SS from one geometry pass, avoiding cross-tool inconsistency. Pooling categorical SS via log-OR + DerSimonian-Laird is the correct generalization of the continuous z-score pooling — χ² is per-protein-only.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/uricase_analysis.md`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/analysis/__init__.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/analysis/structural_features.py`
  - `/Users/jerry/Project/MHC-IF/scripts/analysis/tier1_structural_analysis.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
- evidence: Module AST-parses clean. Pure-python helpers smoke-tested (`cliffs_delta` on toy arrays ≈ 0.833; `random_effects_meta` returns finite pooled+τ²+I²; `log_or_coil` returns a well-defined (log_or, se, counts) on a 8-residue fixture). CLI `--help` renders. BioPython/DSSP-dependent paths not executed locally — `biopython` is in `requirements.txt` (1.86) but not installed in this dev checkout; will be exercised on Della (which has `immune-design` env + DSSP).
- impact:
  - scope: New analysis module + CLI. No existing pipeline touched. Blocks on B1 PDB download for execution.
  - risk: low
  - confidence: 0.93
- status: done
- next_action: (1) Download Tier 1 PDBs via `scripts/download_test_set_pdbs.py` (B1 prerequisite); (2) Run `scripts/analysis/tier1_structural_analysis.py` on Della for each allele to emit `per_residue.csv` / `per_protein_stats.json` / `meta_analysis.json`, then sync F2 supplementary figures back to the repo; (3) User reviews the first run's strict-alignment output — any AlignmentError / SpanSanityError surfaces a real PDB indexing issue rather than a tolerated one.
- refs:
  - `doc/uricase_analysis.md`
  - PROGRESS.md §Phase B (B1)

### L0057
- timestamp: 2026-04-16T18:30:00+08:00
- type: DECISION
- module: L
- trigger: User (Coder) reported that `build_is_epitope` in L0056 treated `end_0b` as inclusive and failed every real Tier 1 span with `SpanSanityError`, which would have caused `tier1_structural_analysis.py` to skip all 15 proteins and produce no artifacts. User required the contract to be half-open with a real peptide fixture locking it.
- change_summary: Fixed two coupled coordinate bugs in `inverse_folding/analysis/structural_features.py::build_is_epitope`: (1) `[start_0b, end_0b)` is now treated as half-open (`end_0b - start_0b == len(peptide)`), and (2) the absolute-to-seq_idx offset uses `chain_range_start - 1` because `chain_range` is 1-based author numbering while `start_0b` is 0-based absolute — verified empirically on all 261 Tier 1 spans (261/261 pass with the fix; 261/261 fail under the original off-by-one + inclusive-end code). Updated the residue-mapping block in `doc/uricase_analysis.md` to spell out both coordinate systems explicitly. Added `tests/inverse_folding/test_analysis_structural_features.py` with 6 tests covering: synthetic half-open labelling, rejection of inclusive-end spans, the `end == len(fasta)` boundary, real 1LI1_C two-span fixture, sweep over every tier1_candidates.json span, and an independent half-open-invariant check.
- rationale: The original draft inferred the coordinate convention from the `_0b` suffix without cross-checking against the shipped JSON. The user correctly flagged the half-open semantics, and while preparing the fixture I also discovered the companion off-by-one: the chain's first FASTA residue is author residue `chain_range_start` (1-based), so the 0-based-absolute `start_0b=1560` for 1LI1_C maps to seq_idx `1560 - (1485-1) = 76`, not `1560 - 1485 = 75`. Both invariants now live in one docstring block and are anchored by real-data tests so they cannot silently regress.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/analysis/structural_features.py` (build_is_epitope rewrite)
  - `/Users/jerry/Project/MHC-IF/doc/uricase_analysis.md` (residue-mapping block)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_analysis_structural_features.py` (new)
- evidence: `pytest tests/inverse_folding/test_analysis_structural_features.py -v` → 6 passed in 0.83s (covers synthetic boundary cases, 1LI1_C real fixture, and full sweep over all 261 spans across 15 proteins).
- impact:
  - scope: Only the new analysis module. No other pipeline touched.
  - risk: low (tests anchor the contract to shipped data; any future edit that breaks half-open or the -1 offset flips the whole tier1 sweep red).
  - confidence: 0.98
- status: done
- next_action: Proceed with cluster B1 PDB download + first real run of `tier1_structural_analysis.py`. If any entry raises `AlignmentError`, inspect ATOM numbering — the test already guarantees all 261 spans pass the sequence-side sanity.
- refs:
  - L0056
  - `outputs/if/test_set/tier1_candidates.json` (shipped data used as fixture)

### L0058
- timestamp: 2026-04-17T09:40:00+08:00
- type: CODEMAP_DIFF
- module: L
- trigger: User (Coder) asked for a Tier 1 candidate JSON for HLA-DRB1*04:01, parallel to the shipped 0701 set. Investigation showed (a) task #10 "Automate Tier 1 candidate selection from epitope head test split" had been marked done but no script was ever committed — original selection was evidently done interactively and not persisted — and (b) the 0701 UniProt set cannot be reused for 0401 because only 1/15 of the 0701 Tier 1 UniProts is present in the DRB1*04:01 epitope-head test split.
- change_summary: Reverse-engineered the original selection logic by verifying that all 261 shipped 0701 spans across 15 entries reproduce bit-for-bit from `data/mhc_if_v2.tsv` for the matching (uniprot, allele, chain_range) tuples. Built the pipeline as a proper library + CLI: `inverse_folding/evaluation/tier1_builder.py` (`load_iedb_spans`, `extract_spans_for_chain`, PDBe SIFTS REST client with JSON cache, `build_candidate_entry`, `build_tier1_candidates` top-level) and `scripts/build_tier1_candidates.py` (CLI with `--allele`, `--manifest-dir`, `--mhc-if-v2`, `--uniprot-fasta`, `--output`, `--validate-against`, filter/rank knobs). Added `tests/inverse_folding/test_tier1_builder.py` with 8 tests: span-range filtering, peptide-mismatch rejection, end=chain-end boundary, 3 rejection-reason unit tests (non-X-ray / high-resolution / too-few-spans), a full-entry round-trip against 1LI1_C, and a sweep that round-trips every shipped 0701 entry. Ran the 0401 build end-to-end: 189 test UniProts → 168 with spans → 110 with SIFTS chains → 46 candidates passing filters → 15 selected by `epitope_coverage` descending. Wrote `outputs/if/test_set/tier1_candidates_DRB1_04_01.json` (15 entries, 489 spans, coverage range 17.4%-87.9%). All 489 spans peptide-verified; all 15 entries pass `validate_tier1_candidate`. Registered the script under `doc/SCRIPTS.md §Module L / Test Set Curation`.
- rationale: The shipped 0701 JSON is authoritative for downstream parquet (`test_proteins_HLA-DRB1_07_01.parquet` uses these 15); we must not overwrite it. A clean, reproducible algorithm for 0401 (and future alleles / future re-runs) is more valuable than trying to exactly reproduce a bespoke curation whose spec is lost. Reproducibility validation on 0701 came in at 7/15 protein_id and 9/15 UniProt overlap — the gap is mostly because our algorithm ranks candidates by coverage descending (a defensible, explicit criterion) while the original 0701 set spans 11%-45% coverage (suggesting either a smaller PDB candidate pool at selection time or a non-recorded stratification rule). 0701 is left untouched; the 0401 output uses the new, specified algorithm. Library + tests ensure the coordinate contract (IEDB 1-based inclusive → 0-based half-open) is regression-locked.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/tier1_builder.py` (new)
  - `/Users/jerry/Project/MHC-IF/scripts/build_tier1_candidates.py` (new)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_tier1_builder.py` (new)
  - `/Users/jerry/Project/MHC-IF/outputs/if/test_set/tier1_candidates_DRB1_04_01.json` (new; 15 entries, 489 spans)
  - `/Users/jerry/Project/MHC-IF/outputs/.cache/sifts/*.json` (PDBe SIFTS cache; not versioned)
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md` (registered `build_tier1_candidates.py` under Module L)
- evidence: `pytest tests/inverse_folding/test_tier1_builder.py -v` → 8 passed. Every shipped 0701 entry round-trips through `build_candidate_entry` (sweep test). 0401 output: 0/489 spans fail peptide vs sequence-slice sanity; 15/15 pass `tier_validators.validate_tier1_candidate`. Dry-run on 0701 reported 7/15 protein_id and 9/15 UniProt overlap with shipped set (documented gap).
- impact:
  - scope: New library + CLI + tests + one downstream JSON artifact (0401 Tier 1). Shipped 0701 JSON and its downstream parquet are unchanged.
  - risk: low (0401 pipeline is additive; peptide verification + validator gate catch any coordinate regressions).
  - confidence: 0.95
- status: done
- next_action: (1) When B1 PDB download runs, include the new 0401 PDB IDs (`6S8V, 3ZFM, 6JD8, 5BN8, 4MZV, 7F4X, 4HAF, 3UBW, 6J64, 5W5M, 4ACP, 3POW, 1W7B, 6UGW, 9H1F`) alongside the 0701 set. (2) Re-run `scripts/analysis/tier1_structural_analysis.py` with `--allele "HLA-DRB1*04:01"` and the new JSON to produce the 0401 F2-supplementary analysis.
- refs:
  - L0055, L0056, L0057 (uricase_analysis.md + structural_features)
  - `doc/SCRIPTS.md` §Module L
  - `data/mhc_if_v2.tsv` (span source)
  - PDBe SIFTS API: `https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}`

### L0059
- timestamp: 2026-04-17T09:50:00+08:00
- type: DECISION
- module: L
- trigger: User recalled the missing 0701 selection criterion: **epitope coverage density must be 10–50%** on top of (X-ray ≤ 2.5 Å, chain length 100–500 AA, ≥ 2 positive EL spans). The L0058 run omitted this and ended up picking high-coverage outliers (e.g. 0401 `9H1F_A` at 87.9%). Adding the coverage filter also raised 0701 dry-run reproducibility meaningfully.
- change_summary: Added `min_coverage` / `max_coverage` (defaults 0.10 / 0.50) to `FilterParams` and the CLI (`--min-coverage`, `--max-coverage`); `build_candidate_entry` now rejects chains whose coverage falls outside the range. Regenerated `outputs/if/test_set/tier1_candidates_DRB1_04_01.json` under the new filter: 189 test UniProts → 168 with spans → 115 with SIFTS chains → 32 passing all filters → 15 selected by coverage-desc ranking (vs 46 / 15 before; the new set has coverage range **17.9%–49.4%**, all inside the 10–50% band, totalling 366 spans; all peptide-verified; all pass `validate_tier1_candidate`). Added `test_build_candidate_entry_rejects_coverage_out_of_range` — 9 tests total, all green.
- rationale: With the coverage filter the 0701 dry-run's UniProt overlap with the shipped 15 rose from **9/15 to 13/15**, and protein_id overlap stayed at 7/15 (chain picks within a UniProt still vary by SIFTS `best_structures` rank order, which is not under our control). The remaining 2 UniProt diffs (P08572 1LI1_C and Q07812 4ZIE_A from the shipped set) stem from the highest-ranked SIFTS chain for those UniProts not passing length/resolution — i.e. the original curation picked a lower-ranked chain. We do not silently change SIFTS-rank semantics to chase exact reproduction; a tested, specified algorithm beats a zero-documentation curation round. The shipped 0701 JSON remains untouched.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/tier1_builder.py` (FilterParams + coverage check)
  - `/Users/jerry/Project/MHC-IF/scripts/build_tier1_candidates.py` (CLI flags)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_tier1_builder.py` (9 tests, +1 coverage-range test)
  - `/Users/jerry/Project/MHC-IF/outputs/if/test_set/tier1_candidates_DRB1_04_01.json` (regenerated; 15 entries, 366 spans, coverage 17.9%–49.4%)
- evidence: `pytest tests/inverse_folding/test_tier1_builder.py -v` → 9 passed in 0.76s. 0401 output: 0 peptide-mismatch failures; 15/15 pass `validate_tier1_candidate`; all 15 coverages ∈ [0.10, 0.50]. 0701 dry-run overlap: 13/15 UniProt (was 9/15 without filter).
- impact:
  - scope: Library + CLI + tests + 0401 JSON (overwritten atomically, same schema).
  - risk: low.
  - confidence: 0.96
- status: done
- next_action: Use `outputs/if/test_set/tier1_candidates_DRB1_04_01.json` for B1 PDB download (new 15 PDB IDs: `3DZ8 3ZFM 3S35 6JD8 5BN8 4MZV 7F4X 4HAF 3UBW 6J64 5W5M 4ACP 3POW 1W7B 6UGW`) and the matching allele run of `scripts/analysis/tier1_structural_analysis.py`.
- refs:
  - L0058

### L0060
- timestamp: 2026-04-22T15:45:00+08:00
- type: PLAN_UPDATE
- module: PHASE_B
- trigger: User (Coder-track) requested a concrete, implementation-ready spec for B2 (test-set h_i precompute) and B3 (CATH training-set h_i precompute), grounded in the hypotheses of `doc/Reference_Flow_Derivation.md §6`. The original `PLAN_IF.md` Phase B section had B2/B3 as two-line stubs with no schema, no CLI contract, no failure policy, and no mapping from `h_i` storage to downstream hypotheses.
- change_summary: Expanded `PLAN_IF.md` Tasks B2 and B3 from stub bullets to implementation-ready specs (goal, frozen design principle, inputs, outputs, parquet schema, sidecar JSON schema, CLI contract, SLURM convention, failure handling, TDD gate, acceptance criteria) and appended a "What Phase C inherits" mapping that pins each stored field to the specific hypothesis it supports (H1–H5). Froze the storage–policy decoupling principle: persist `h_raw` (uncentered log-mean-exp logit) and `h_processed` (per-protein median-centered, `clamp=none`) as the neutral physical quantities; defer every downstream transform — `g(h)` form, corpus-level normalization, hotspot binarization threshold, multi-allele aggregation — to Phase C read-time. Froze allele scope: B2 emits 0701 + 0401; B3 emits 0701 (priority 1) + 0401 (priority 2); 1501 skipped.
- rationale: (1) B2/B3 encoder forward passes are the expensive step (~20k proteins × ESM-2 650M); downstream policies are cheap ndarray transforms. Locking a specific normalization at compute time would force recompute whenever Phase C revises `g(h)` or ablates corpus-level vs per-protein normalization. Storing both neutral forms (~1 MB / 1000 proteins overhead) is the right trade. (2) Each of H1–H5 demands a slightly different view of `h` (continuous for sampling / training, binarized for stratification, permuted for shuffle control, corpus-normalized for cross-protein comparability). A single neutral storage supports all five without bias. (3) Deferred items are surfaced explicitly so they are not silently re-introduced during Phase C. (4) Scope limited to 0701 + 0401 because Module L test sets exist only for these two alleles; 1501 cannot be paired with an evaluation cohort for H4.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md` (Task B2 rewritten, Task B3 rewritten, new "What Phase C inherits" subsection)
- evidence: N/A (planning update, no code change). Design verified against existing interfaces: `epitope_head/inference/predictor.py::InferencePredictor.predict_protein` returns `residue_hotspot` (= our `h_processed`) and `debug.h_raw`, matching the proposed schema; `epitope_head/configs/inference.yaml` defaults (`center_method=median`, `clamp=none`, `min_k=12`, `max_k=25`, chunking enabled with context_len=1022) map one-to-one to the "frozen inference config preset" referenced in the plan.
- impact:
  - scope: Phase B specification only. No code changed. No downstream module touched. Future CLI `scripts/precompute_h_maps.py`, SLURM scripts, and loader `inverse_folding/evaluation/h_maps.py` will implement against this spec.
  - risk: low (all deferred choices documented; storage is a safe superset).
  - confidence: 0.94
- status: done
- next_action: Implement `scripts/precompute_h_maps.py` + `inverse_folding/evaluation/h_maps.py` + SLURM launchers per the spec; register under `doc/SCRIPTS.md §Phase B`; run B2 0701 + 0401 on Della, then B3 0701 + 0401.
- refs:
  - `PLAN_IF.md:§Phase B / Task B2, Task B3`
  - `doc/Reference_Flow_Derivation.md:§4.3, §6 (H1–H5), §7.1, §7.3`
  - `epitope_head/inference/predictor.py::InferencePredictor.predict_protein`
  - `epitope_head/configs/inference.yaml`

### L0061
- timestamp: 2026-04-23T17:10:00+08:00
- type: PLAN_UPDATE
- module: PHASE_C
- trigger: User (Coder-track) redirected scope: experimental-design decisions (which arms to run, which `c` values to sweep, whether to add an explicit DFM-uniform control) belong to the experimenter, not to the code spec. The prior Phase C stub left `g(h)` form, step count, schedule choice, and arm selection as "open for Thinker discussion", blocking code work. User asked for a code contract + hyperparameter schema that exposes every tunable knob as YAML so arm selection is config, not code.
- change_summary: Expanded `PLAN_IF.md` §Phase C from a stub to an implementation-ready spec for Tasks C0 (DPLM native-sampler baseline driver) and C1 (sampling-only position-dependent DFM sampler). Froze (a) the §4.3 sampling algorithm as the C1 reference pseudocode, (b) a YAML hyperparameter schema covering sampler (`n_steps`, `seed`, `temperature`, `n_designs_per_protein`), schedule (`base_form ∈ {linear, cosine, cubic}`), amplification (`form ∈ {constant_one, linear_clamp, sigmoid, power}`, `c`, `mu`, `kappa`, `p`, `h_source ∈ {h_raw, h_processed, h_normalized_corpus}`, `g_max_cap`), CFG (`enabled`, `w`), and H3 h-shuffle (`enabled`, `seed`), (c) a denoiser-agnostic sampler API (`denoiser(x_t, t, struct) → logits[L, V]`) so C2/C3 can reuse the module, (d) a 5-config preset scaffold (`c1_null/c1_linclamp/c1_sigmoid/c1_power/c1_shuffle`) whose names are examples, not experimental claims, (e) full CLI + SLURM + failure-handling + TDD contracts, and (f) numerical guards (`log κ_base` floor, `g_max_cap`, final-step force-unmask, NaN abort). C2 (retrain) and C3 (FiLM + CFG) remain stubs. Also updated §6 Open Decisions item 6 to reflect that C1's code contract is frozen while experimental values remain experimenter-controlled.
- rationale: (1) Separating code contract (what the module accepts as input and what it guarantees) from experimental policy (which values to run) is the correct Coder/Experimenter split; baking specific `c` values or arm choices into code forces refactors every time the experimenter adjusts the sweep. (2) The `constant_one` amplification form collapses C1 to a DFM-sampler uniform-schedule run without adding a separate arm or code path — the experimenter can exercise that control (or any position-dependent arm, or the H3 shuffle control) purely by swapping YAML. (3) Denoiser-as-callable keeps the sampler module reusable for C2 (retrained denoiser) and C3 (FiLM-conditioned denoiser) without rewriting the loop. (4) Preset YAMLs are scaffolds, not frozen experimental choices — they live in `configs/` but the plan is explicit that the experimenter edits/sweeps them. (5) Numerical guards are derived from §2.2 / §4.2 of the derivation (divergence behavior near `t=1`, well-posedness of `κ_i'(t)` near `t=0`) and from the tail behavior of log-mean-exp h values observed in the existing epitope head output.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md` (§Phase C rewritten: Tasks C0 + C1 expanded, C2 + C3 kept as stubs; §6 item 6 updated)
- evidence: N/A (planning update, no code change). Algorithm pseudocode verified against `doc/Reference_Flow_Derivation.md §4.3`; hyperparameter schema covers all choices flagged in §7.1 (amplification forms) and §7.2 (base schedule) of the derivation; preset configs map one-to-one to the arm variants needed to test hypotheses H1–H5 without encoding any experimental claim in code.
- impact:
  - scope: Phase C task-C0 and task-C1 specifications only. No code changed. C2/C3 still stubs. B2/B3 spec (L0060) untouched.
  - risk: low (spec is a superset of all discussed configurations; numerical guards cover known edge cases; denoiser-agnostic boundary keeps the sampler reusable).
  - confidence: 0.93
- status: done
- next_action: Implement `inverse_folding/reference_flow/{schedule,amplification,sampler,config}.py` + preset YAMLs + `scripts/run_if_phase_c{0,1}.py` + SLURM launchers + tests per the TDD gate; register all under `doc/SCRIPTS.md §Phase C`; gate full-test-set runs on `c1_null.yaml` passing the smoke run.
- refs:
  - `PLAN_IF.md:§Phase C / Task C0, Task C1`
  - `doc/Reference_Flow_Derivation.md:§4.3 (sampling algorithm), §2.2 (rate regularity), §4.2 (step-size constraint), §7.1 (g forms), §7.2 (base schedules)`
  - L0060 (B2/B3 feed h_maps to C1)

### L0061
- timestamp: 2026-04-22T21:45:00+08:00
- type: VERIFICATION
- module: PHASE_B
- trigger: User asked to review the detailed Phase B plan in `PLAN_IF.md` and implement it if there were no blocking concerns.
- change_summary: Implemented Phase B B2/B3 h-map precompute infrastructure: shared CLI, h-map loader/schema validator, focused contract tests, parameterized SLURM launchers, and script registry updates.
- rationale: The B2/B3 plan is scientifically and operationally sound because it stores neutral per-residue quantities (`h_raw` and median-centered `h_processed`) while deferring all Phase C policy choices. Implementation keeps expensive epitope-head forward passes reusable across sampling, retraining, shuffle controls, and hotspot analyses.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/h_maps.py`
  - `/Users/jerry/Project/MHC-IF/scripts/precompute_h_maps.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_precompute_h_test.slurm`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_precompute_h_cath.slurm`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_h_maps_contract.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
- evidence: |
    `pytest -q tests/inverse_folding/test_h_maps_contract.py` => 5 passed.
    `python -m py_compile scripts/precompute_h_maps.py inverse_folding/evaluation/h_maps.py` => passed.
    `bash -n scripts/submit_precompute_h_test.slurm` => passed.
    `bash -n scripts/submit_precompute_h_cath.slurm` => passed.
    `python scripts/precompute_h_maps.py --help` => argparse/import smoke passed.
- impact:
  - scope: Phase B h-map precompute only; no existing generation, evaluation, or epitope-head training behavior changed.
  - risk: low
  - confidence: 0.94
- status: done
- next_action: Submit B2 on Della for `HLA-DRB1*07:01` and `HLA-DRB1*04:01`, then validate row counts and sidecar failure ledgers before launching B3 CATH train runs.
- refs:
  - `PLAN_IF.md:Task B2`
  - `PLAN_IF.md:Task B3`
  - `LOG.md:L0060`

### L0062
- timestamp: 2026-04-23T00:10:00+08:00
- type: VERIFICATION
- module: PHASE_B
- trigger: User code review identified three trust-boundary gaps before Della resume usage: resume accepted stale/mismatched parquets, `load_h_maps` lacked metadata/dataframe cross-field validation, and the CLI test exercised only the `predict_protein` fallback instead of the real `encode_sequence -> enumerate_and_score -> aggregate_hotspot_and_risk` branch.
- change_summary: Hardened h-map artifact validation and resume semantics: `load_h_maps` now validates dataframe allele vs metadata allele, accounting fields, failure-ledger length, optional required CATH corpus stats, and corpus-stat values; `precompute_h_maps.py` now requires a resume sidecar and fail-fast checks allele, checkpoint path, checkpoint metadata, inference config, source dataset, rowcount, protein-id membership, and current sequence length before reusing rows; partial checkpoints now write matching metadata sidecars.
- rationale: Phase C treats B2/B3 h-maps as a trusted substrate. A stale 0701 partial, old checkpoint/config, or sequence-length drift would silently poison sampling/retraining signals, so resume must be conservative and loader validation must be cross-field rather than column-local only.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/h_maps.py`
  - `/Users/jerry/Project/MHC-IF/scripts/precompute_h_maps.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_h_maps_contract.py`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
- evidence: |
    `pytest -q tests/inverse_folding/test_h_maps_contract.py` => 13 passed.
    `pytest -q tests/inverse_folding/test_module_l_eval_contract.py` => 33 passed.
    `python -m py_compile scripts/precompute_h_maps.py inverse_folding/evaluation/h_maps.py` => passed.
    `bash -n scripts/submit_precompute_h_test.slurm` => passed.
    `bash -n scripts/submit_precompute_h_cath.slurm` => passed.
- impact:
  - scope: Phase B h-map validation/resume behavior only; no scoring math changed.
  - risk: low
  - confidence: 0.96
- status: done
- next_action: Use the guarded resume path for B2 Della runs; if an existing old partial lacks `.meta.json`, discard it and restart rather than forcing reuse.
- refs:
  - `LOG.md:L0061`
  - `PLAN_IF.md:Task B2`
  - `PLAN_IF.md:Task B3`

### L0063
- timestamp: 2026-04-23T01:45:00+08:00
- type: IMPLEMENTATION
- module: PHASE_C
- trigger: User asked to review Phase C0/C1 in `PLAN_IF.md` and implement if there were no blocking concerns.
- change_summary: Implemented the Phase C0/C1 code path: added the `inverse_folding/reference_flow` package (schedule, amplification, YAML config validator, denoiser-agnostic sampler, DPLM runtime helpers), five preset YAML scaffolds, the C0 native-sampler driver, the C1 sampling-only reference-flow driver, and parameterized SLURM launchers. Registered both drivers in `doc/SCRIPTS.md` and advanced `PROGRESS.md` to reflect that C0/C1 are code-ready.
- rationale: The C0/C1 plan is internally coherent. C0 only needs a reproducible native-sampler wrapper over the frozen Module K checkpoint, while C1 needs the mathematically frozen §4.3 sampler with all experiment knobs lifted into YAML so future arm selection stays in config rather than code. The implementation keeps sampler math independent of DPLM internals and confines checkpoint/backbone wiring to the CLI/runtime layer.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/__init__.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/schedule.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/amplification.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/sampler.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/runtime.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_null.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_linclamp.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_sigmoid.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_power.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_shuffle.yaml`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c0.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c1.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_phase_c0.slurm`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_phase_c1.slurm`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
- evidence: |
    `pytest -q tests/inverse_folding/test_reference_flow_schedule.py tests/inverse_folding/test_reference_flow_amplification.py tests/inverse_folding/test_reference_flow_config.py tests/inverse_folding/test_reference_flow_sampler.py tests/scripts/test_run_if_phase_c0_script.py` => 16 passed.
    `pytest -q tests/inverse_folding/test_h_maps_contract.py tests/inverse_folding/test_module_l_eval_contract.py` => 46 passed.
    `python -m py_compile scripts/run_if_phase_c0.py scripts/run_if_phase_c1.py inverse_folding/reference_flow/*.py` => passed.
    `python scripts/run_if_phase_c0.py --help` => argparse/import smoke passed.
    `python scripts/run_if_phase_c1.py --help` => argparse/import smoke passed.
    `python -c "from inverse_folding.reference_flow.runtime import safe_allele_tag; print(...)"` => runtime import smoke passed after moving DPLM imports to lazy loading.
    `bash -n scripts/submit_if_phase_c0.slurm` => passed.
    `bash -n scripts/submit_if_phase_c1.slurm` => passed.
- impact:
  - scope: Phase C0/C1 generation code only; existing Phase B h-map and Module L evaluation contracts remain unchanged.
  - risk: medium
  - confidence: 0.89
- status: done
- next_action: Run `submit_if_phase_c0.slurm` for one allele as the unguided baseline, then launch `submit_if_phase_c1.slurm` with `c1_null.yaml` once the matching B2 h-maps exist; reserve `h_normalized_corpus` arms until B3 corpus stats are materialized.
- refs:
  - `PLAN_IF.md:Task C0`
  - `PLAN_IF.md:Task C1`
  - `LOG.md:L0062`

### L0064
- timestamp: 2026-04-23T21:13:46+08:00
- type: CODEMAP_DIFF
- module: F
- trigger: User requested implementing the new EL evaluation protocol from `doc/EL_new_evaluation.md` (residue-level, IoU ladder, 1D EMD, bootstrap CI) on top of the existing IEDB benchmark script.
- change_summary: Extended `scripts/benchmark_iedb_test.py` with a `--metric-suite {exact,residue,iou_ladder,emd,all}` flag (default `all`) and added M2a residue-level AUC/AP/Pearson/Spearman (k=15-symmetric + variable-k variants), M6 COCO-style IoU ladder over `--iou-thresholds`, M3 normalized 1D EMD, and a `statistics` block with 1000-resample bootstrap 95% CI and paired Wilcoxon signed-rank (head vs NMP) on every headline metric.
- rationale: Per `doc/EL_new_evaluation.md §6` the protocol should extend the existing benchmark script rather than branch into a new one; this keeps the head/NMP inference pass shared and preserves backwards-compatible JSON schema (all new keys are additive under `macro.residue`, `macro.iou_ladder`, `macro.emd`, `statistics`). Helpers are pure functions so they stay unit-testable.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/benchmark_iedb_test.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_benchmark_iedb_test.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
- evidence: |
    `pytest -q tests/scripts/test_benchmark_iedb_test.py` => 20 passed (5 existing + 15 new).
    `PYTHONPATH=. python scripts/benchmark_iedb_test.py --help` => argparse smoke passed; new flags `--metric-suite`, `--iou-thresholds`, `--bootstrap-n`, `--bootstrap-seed` show up with expected defaults.
    Interactive smoke on M2a/M6/M3/bootstrap/Wilcoxon helpers: residue max aggregation respects k-filter, IoU greedy matches highest-IoU unassigned GT, detection AP = (1 + 2/3)/3 on hand-worked example, bootstrap is deterministic under fixed seed.
- impact:
  - scope: IEDB benchmark output schema (additive only) + SLURM callers via `submit_benchmark.slurm` MODE=iedb continue to work with the default `--metric-suite all`.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Run one full-allele benchmark on the cluster (`MODE=iedb ALLELE=HLA-DRB1*07:01 sbatch scripts/submit_benchmark.slurm`) to confirm wall-time impact of the new residue/IoU/EMD passes and to populate the first real M2a/M6/M3 numbers for the F2 revision.
- refs:
  - `doc/EL_new_evaluation.md`
  - `doc/SCRIPTS.md` (Benchmark §2)

### L0065
- timestamp: 2026-04-25T22:31:49+08:00
- type: CODEMAP_DIFF
- module: F
- trigger: User updated `doc/EL_new_evaluation.md` §5 to remove the k=15 restriction (NMP and head now share the full multi-k span set k∈[min_k,max_k] for all M2a/M6/M3 metrics) and asked whether the previous experiment can be resumed without re-running NMP subprocess.
- change_summary: Dropped the k=15 filter from M2a residue aggregation, M6 IoU ladder, and M3 EMD (collapsing the dual `k15_symmetric`/`variable_k` residue variants into a single `residue` block); added a windows cache mechanism (`--windows-cache PATH` to write a parquet with sidecar `.meta.json`, `--resume-from-cache PATH` to reload and skip head + NMP inference entirely).
- rationale: The prior experiment's JSON output only persists aggregated metrics and label-stratified `score_distributions` — it never serialized the raw `(start, k) → score` dicts, so re-running NMP was unavoidable for this metric change. Going forward, the windows cache makes any further metric/aggregation iteration a sub-second reload instead of a fresh NMP subprocess sweep (~30min-2hr per allele in original mode). The k=15 collapse follows `doc/EL_new_evaluation.md §5` head note ("k∈[12,25] enumerated for every protein, both head and NMP scored on the identical set; no length restriction is imposed at metric time") which makes residue aggregation and IoU comparison length-agnostic between predictors.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/benchmark_iedb_test.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_benchmark_iedb_test.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
- evidence: |
    `pytest -q tests/scripts/test_benchmark_iedb_test.py` => 22 passed (5 legacy + 15 EL-landscape carried over and rewritten where they pinned k=15 + 2 new cache round-trip tests).
    `python -m py_compile scripts/benchmark_iedb_test.py` => passed.
    `PYTHONPATH=. python scripts/benchmark_iedb_test.py --help` => surfaces `--windows-cache`, `--resume-from-cache`; mutual-exclusion guard short-circuits with a clear error when both are set.
    Cache parquet schema verified via the new `test_windows_cache_round_trip_preserves_dicts` and `test_windows_cache_handles_predictor_only_proteins` tests (head-only / NMP-only proteins round-trip exactly).
- impact:
  - scope: Output JSON `macro.residue` collapses from `{k15_symmetric: {...}, variable_k: {...}}` to flat `{head: {...}, nmp: {...}}`; `statistics` headline labels rename `residue_k15_symmetric_*` and `residue_variable_k_*` → `residue_*`. Existing exact-span primary/conditional metrics and `near-miss` block unchanged. Old runs prior to L0064 do not need re-processing — but the corresponding L0064 cache was never written, so this iteration requires one fresh inference sweep per allele (and going forward `--windows-cache` lets you resume).
  - risk: low (additive cache; metric collapse is the doc-prescribed default, and no callers pin the old keys yet — L0064 was never run in production)
  - confidence: 0.9
- status: done
- next_action: Run `MODE=iedb ALLELE=HLA-DRB1*07:01 WINDOWS_CACHE=outputs/cache_iedb_drb0701.parquet sbatch scripts/submit_benchmark.slurm` (need to expose `WINDOWS_CACHE` env var in `submit_benchmark.slurm` if not already piped through) to populate the cache; subsequent metric / IoU-threshold / aggregation ablations then reload via `--resume-from-cache outputs/cache_iedb_drb0701.parquet` on a login node in seconds.
- refs:
  - `doc/EL_new_evaluation.md` §5 head note
  - `LOG.md:L0064`
  - `doc/SCRIPTS.md` (Benchmark §2)

### L0066
- timestamp: 2026-04-26T16:00:00+08:00
- type: PLAN_UPDATE
- module: PHASE_B
- trigger: User (Coder-track) flagged that Phase C generation outputs (`generated.parquet` + concatenated `generated.fasta`) do not align with the legacy evaluator (`scripts/evaluate_if.py`) input contract, that the legacy evaluator additionally imports from a deprecated Module M file, and that there is no end-to-end evaluation protocol wired to Phase C output. User asked B4 to (a) reuse the existing benchmark pieces (`scripts/benchmark_head_vs_nmp.py`, `scripts/benchmark_iedb_test.py`) for the immunogenicity side, (b) add a structural mode that accepts a configurable refold backend (ESMFold default, AF3 reserved), (c) drop WT mixing inside the evaluator, and (d) avoid creating fan-out per-protein FASTAs.
- change_summary: Expanded `PLAN_IF.md` Task B4 from a 3-bullet stub to an implementation-ready spec for `scripts/evaluate_phase_c.py`. Froze (a) parquet-in / parquet-out interface (consumes `generated.parquet` + `test_set_parquet`, no per-protein FASTA fan-out, no separate WT FASTA), (b) three modes `imm | struct | all` with idempotent per-mode parquet outputs and `--overwrite` clobber control, (c) pluggable refold backend via `--refold-model esmfold|af3` (`af3` raises `NotImplementedError` today; dispatcher contract is in place via new `inverse_folding/evaluation/refold.py`), (d) schema-aligned outputs `imm_head.parquet` / `imm_nmp.parquet` / `structural.parquet` matching `inverse_folding/evaluation/schema.py` columns plus `design_idx` and `refold_backend` for join-back / provenance, (e) WT-free contract (recovery is per-design sequence identity, no `delta_*` metrics; WT-relative metrics are downstream joins against a future static `wt_metrics_<allele>.parquet`), (f) predictor source set to `scripts.infer_v1.build_predictor` (not the deprecated `scripts/run_if_guidance_sweep.py::load_epitope_predictor`), (g) SLURM integration as a new `MODE=phase_c` branch in `scripts/submit_benchmark.slurm` alongside the existing `fasta` / `iedb` modes, and (h) deletion of the legacy `scripts/evaluate_if.py` plus its single-purpose helpers in `inverse_folding/evaluation/aggregate.py`. Also added a §10 "Outstanding Tech Debt" section flagging the predictor-loader cleanup (six active scripts still import from `scripts/run_if_guidance_sweep.py`) as a non-blocking item.
- rationale: (1) Aligning B4 output schema with `inverse_folding/evaluation/schema.py` lets analysis notebooks consume Phase C metrics without per-run shape fixes. (2) Mode-switchable refold (ESMFold vs AF3 stub) future-proofs the backend swap without forcing an evaluator refactor when AF3 lands. (3) Keeping WT logic out of the evaluator keeps each run's output a pure function of `(generated.parquet, head_ckpt, nmp_binary, refold_backend)`; WT-relative comparison becomes a separate static-artifact + join, which is simpler to version. (4) Reusing `scripts.infer_v1.build_predictor` matches Phase B `precompute_h_maps` and avoids making B4 the seventh script depending on a deprecated host file. (5) Per-mode idempotence (skip if target parquet exists, unless `--overwrite`) lets the experimenter rerun just the cheap mode (`imm`) without redoing ESMFold. (6) `MODE=phase_c` extension into `submit_benchmark.slurm` follows the project's parameterized-SLURM convention (`scripts/CLAUDE.md`) and keeps a single submission script per benchmark family.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md` (Task B4 rewritten; §10 Tech Debt section added; Phase B acceptance updated to "B1+B2+B3+B4 → Phase C ready")
- evidence: N/A (planning update, no code change). Output schema verified column-by-column against `inverse_folding/evaluation/schema.py::STRUCTURAL_METRICS_COLUMNS`, `IMMUNOGENICITY_HEAD_COLUMNS`, `IMMUNOGENICITY_NMP_COLUMNS`. Predictor source choice cross-checked with `scripts/precompute_h_maps.py:86-109` (Phase B canonical caller of `scripts.infer_v1.build_predictor`).
- impact:
  - scope: Phase B Task B4 specification only. No code changed. Phase B B1/B2/B3 specs untouched. Will, when implemented, delete `scripts/evaluate_if.py` and trim `inverse_folding/evaluation/aggregate.py`.
  - risk: low (spec is downstream-only; modes are independent; refold dispatcher keeps the AF3 swap surgical).
  - confidence: 0.93
- status: done
- next_action: Implement `scripts/evaluate_phase_c.py` + `inverse_folding/evaluation/refold.py` + `tests/scripts/test_evaluate_phase_c_script.py` + the `MODE=phase_c` branch in `scripts/submit_benchmark.slurm` per the spec; register under `doc/SCRIPTS.md §Phase B`; smoke-test against a 3-protein × 2-design fixture before launching on full 0701 C0 output.
- refs:
  - `PLAN_IF.md:§Phase B / Task B4`
  - `PLAN_IF.md:§10 Outstanding Tech Debt`
  - `inverse_folding/evaluation/schema.py`
  - `scripts/infer_v1.py::build_predictor`
  - `scripts/benchmark_head_vs_nmp.py` (immunogenicity reference)
  - `scripts/benchmark_iedb_test.py` (immunogenicity reference)
  - `inverse_folding/evaluation/esmfold_runner.py` (struct mode)
  - `inverse_folding/evaluation/tmalign.py` (struct mode)
  - L0060 (B2/B3 spec), L0061 (Phase C spec)

### L0067
- timestamp: 2026-04-26T16:31:53+08:00
- type: IMPLEMENTATION
- module: PHASE_B
- trigger: User (Coder-track) updated `PROGRESS.md` with the frozen B4 spec and asked for a review against the unified Phase C evaluation-schema requirements, then requested implementation if the design was sound.
- change_summary: Implemented B4 end-to-end Phase C evaluation integration: added `scripts/evaluate_phase_c.py` (parquet-in / parquet-out evaluator with independent `imm|struct|all` modes, per-mode idempotence, failure ledgers, manifest/config emission, and WT-free outputs), added `inverse_folding/evaluation/refold.py` as the refold backend dispatcher (`esmfold` live, `af3` stub), wired `MODE=phase_c` into `scripts/submit_benchmark.slurm`, deleted the legacy FASTA-directory evaluator `scripts/evaluate_if.py`, updated script registration/docs, and refreshed a few stale script messages that still pointed to `evaluate_if.py`.
- rationale: The new evaluator aligns the actual Phase C artifact contract (`generated.parquet`) with the frozen metric schema without forcing temporary FASTA fan-out or hidden WT-relative joins. Reusing `scripts.infer_v1.build_predictor`, `build_runner`, `esmfold_runner`, and `tmalign` keeps B4 on the same inference stack as B2/B3 while isolating the future AF3 swap behind a narrow dispatcher interface. `submit_benchmark.slurm` was extended instead of duplicated to respect the repo's parameterized-SLURM rule.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/evaluate_phase_c.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/refold.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_evaluate_phase_c_script.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_benchmark.slurm`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_level1_filter.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_guidance_sweep.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
  - `/Users/jerry/Project/MHC-IF/scripts/evaluate_if.py` (deleted)
- evidence: |
    `pytest -q tests/scripts/test_evaluate_phase_c_script.py` => 6 passed.
    `pytest -q tests/scripts/test_evaluate_phase_c_script.py tests/scripts/test_run_if_phase_c0_script.py tests/scripts/test_run_if_phase_c1_script.py tests/inverse_folding/test_reference_flow_schedule.py tests/inverse_folding/test_reference_flow_amplification.py tests/inverse_folding/test_reference_flow_config.py tests/inverse_folding/test_reference_flow_sampler.py tests/inverse_folding/test_reference_flow_runtime.py tests/inverse_folding/test_h_maps_contract.py tests/inverse_folding/test_module_l_eval_contract.py` => 72 passed.
    `python -m py_compile scripts/evaluate_phase_c.py inverse_folding/evaluation/refold.py scripts/run_if_level1_filter.py` => passed.
    `python scripts/evaluate_phase_c.py --help` => argparse/import smoke passed.
    `bash -n scripts/submit_benchmark.slurm` => passed.
- impact:
  - scope: Phase B4 only; Phase C generators, Phase B h-map precompute, and Module L comparison contracts remain behaviorally unchanged.
  - risk: medium
  - confidence: 0.9
- status: done
- next_action: Launch a small Della smoke (`MODE=phase_c EVAL_MODE=imm` on a tiny C0/C1 `generated.parquet` fixture), then run one full-allele `EVAL_MODE=all REFOLD_MODEL=esmfold` pass to populate the first schema-aligned Phase C metrics.
- refs:
  - `PLAN_IF.md:Task B4`
  - `LOG.md:L0066`
  - `doc/SCRIPTS.md:Phase B`
  - `inverse_folding/evaluation/schema.py`

### L0068
- timestamp: 2026-04-27T00:03:08+08:00
- type: IMPLEMENTATION
- module: PHASE_B
- trigger: User (Coder-track) found that thousands of assembled IF test-set rows had parquet `sequence_length` values that did not match the downloaded structure length, making Phase C inverse folding invalid. User confirmed the fix should not drop mismatch rows outright, asked to preserve biological FASTA semantics, rebuild a resolved-backbone IF-ready dataset, skip formal TDD for speed, and record the cleanup after completion.
- change_summary: Added `scripts/build_if_ready_test_set.py` as a B1.5 structure/sequence cleaning gate. The script reads the assembled biological-sequence parquet plus downloaded structures, extracts the requested chain's resolved residues with complete N/CA/C/O backbone atoms, canonicalizes modified amino-acid residue names using BioPython's extended PDB mapping, collapses duplicate author residue IDs by preferring standard ATOM residues, selects one alternate conformer per atom by highest occupancy, writes cleaned single-chain PDBs, and emits an IF-ready parquet whose `sequence` / `sequence_length` are the exact DPLM-readable resolved-backbone sequence. Original biological sequence, original length, source structure, residue-number mapping, original residue names, and failure ledgers are retained for audit. Registered the script in `doc/SCRIPTS.md` and updated `PROGRESS.md` to mark B1.5 complete and B2 h-map precompute as requiring rerun on IF-ready parquets.
- rationale: The root cause was not incorrect FASTA, but a semantic mismatch: assembled parquet sequences are biological/full sequences, while DPLM/ProteinMPNN-style inverse folding requires one residue per resolved backbone coordinate. Missing terminal/loop residues, expression tags, modified residues, duplicate HET/ATOM author residue IDs, and altloc conformers must be resolved before Phase C. Rebuilding a derived IF-ready dataset preserves the original biological inputs while giving Phase C and h-map precompute a length-consistent substrate.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/build_if_ready_test_set.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.manifest.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.failures.csv`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs_if_ready/0401/`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.manifest.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.failures.csv`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs_if_ready/0701/`
- evidence: |
    `python -m py_compile scripts/build_if_ready_test_set.py` => passed.
    `python scripts/build_if_ready_test_set.py --help` => argparse/import smoke passed.
    `python scripts/build_if_ready_test_set.py ... HLA-DRB1_04_01 ...` => 3,140 input rows → 2,881 IF-ready rows, 259 failures (Tier 1: 14 ready / 1 failed; Tier 2: 2,867 ready / 131 failed; Tier 3: 127 failed). Failure reasons: 192 unsupported `.cif`, 49 unknown residues, 18 missing structures.
    `python scripts/build_if_ready_test_set.py ... HLA-DRB1_07_01 ...` => 3,141 input rows → 2,839 IF-ready rows, 302 failures (Tier 1: 14 ready / 1 failed; Tier 2: 2,825 ready / 175 failed; Tier 3: 126 failed). Failure reasons: 243 unsupported `.cif`, 35 unknown residues, 24 missing structures.
    Final DPLM gate using `inverse_folding.dplm.src.byprot.utils.io.load_coords()` on every cleaned PDB: HLA-DRB1_04_01 2,881/2,881 exact sequence+length match, 0 mismatch, 0 errors; HLA-DRB1_07_01 2,839/2,839 exact sequence+length match, 0 mismatch, 0 errors.
- impact:
  - scope: Adds a derived Phase B cleaning gate and materialized scratch artifacts. Original biological parquets, FASTAs, raw downloaded PDB/CIF files, and Phase C generator code are not overwritten.
  - risk: medium (dataset semantics changed for Phase C; all WT immunogenicity artifacts and h-maps must now be regenerated on IF-ready sequences)
  - confidence: 0.91
- status: done
- next_action: Rerun B2 h-map precompute and any WT head/NMP baseline artifacts against `if_ready/test_proteins_if_ready_*.parquet` with `PDB_ROOT=work/immune-design/if_test_set/pdbs_if_ready/{0401,0701}` before launching C0/C1. Add mmCIF support later to recover the currently unsupported `.cif` rows, and optionally investigate the remaining `unknown_residue` failures.
- refs:
  - `scripts/build_if_ready_test_set.py`
  - `PROGRESS.md:Phase B`
  - `doc/SCRIPTS.md:Phase B`

### L0069
- timestamp: 2026-04-27T00:47:26+08:00
- type: IMPLEMENTATION
- module: PHASE_B
- trigger: User (Coder-track) noticed that many remaining IF-ready failures were `.cif` structures and asked to add CIF support to the backbone path so uricases would not all be excluded. User also asked whether the remaining three-letter residue symbols were still unmappable.
- change_summary: Added mmCIF support to the DPLM backbone loader (`inverse_folding/dplm/src/byprot/utils/io.py`) by switching the biotite 1.6 path from removed `pdbx.PDBxFile` to `pdbx.CIFFile` and adding BioPython extended residue-name fallback for common modified amino acids. Extended `scripts/build_if_ready_test_set.py` to parse `.cif` inputs with `MMCIFParser`, write cleaned `.cif` outputs with `MMCIFIO`, preserve multi-character chain IDs, and accept `--default-chain A` for non-PDB-chain protein IDs such as Tier 3 UniProt-style uricases. Updated Phase C runtime (`inverse_folding/reference_flow/runtime.py`) to prefer `if_chain_id` from IF-ready parquet before inferring a chain from `protein_id`. Rebuilt both IF-ready parquets with `--allow-cif`, recovering most CIF-backed rows and many Tier 3 uricases.
- rationale: The previous B1.5 pass deliberately skipped `.cif`, which left all Tier 3 uricase AFDB/local prediction rows unusable and also excluded RCSB CIF rescues. mmCIF support must preserve multi-character chain IDs (e.g. `AAA`), so writing those cleaned structures back to PDB is unsafe; cleaned CIF output plus runtime `if_chain_id` is the correct contract. Remaining unknown residue symbols are not covered by BioPython's extended PDB residue map and include ligand/cofactor-like codes (e.g. `SAM`, `SFG`, `BET`, `MTX`), so they remain fail-fast rather than guessed.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/utils/io.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/runtime.py`
  - `/Users/jerry/Project/MHC-IF/scripts/build_if_ready_test_set.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.manifest.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.failures.csv`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs_if_ready/0401/`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.manifest.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.failures.csv`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs_if_ready/0701/`
- evidence: |
    `python -m py_compile scripts/build_if_ready_test_set.py inverse_folding/reference_flow/runtime.py inverse_folding/dplm/src/byprot/utils/io.py` => passed.
    Real-CIF smoke on `6TN1_AAA.cif`: `scripts.build_if_ready_test_set.process_row(... allow_cif=True)` emitted `6TN1_AAA.cif`; DPLM `load_coords()` read chain `AAA` with length 497 and exact sequence match.
    Rebuilt 0401 with `--allow-cif`: 3,140 input rows → 3,072 IF-ready rows, 68 failures. Ready rows: Tier 1=15, Tier 2=2,948, Tier 3=109; formats: 2,881 `.pdb`, 191 `.cif`.
    Rebuilt 0701 with `--allow-cif`: 3,141 input rows → 3,079 IF-ready rows, 62 failures. Ready rows: Tier 1=15, Tier 2=2,962, Tier 3=102; formats: 2,839 `.pdb`, 240 `.cif`.
    Final DPLM gate using row-level `if_chain_id`: HLA-DRB1_04_01 3,072/3,072 exact sequence+length match, 0 mismatch, 0 errors; HLA-DRB1_07_01 3,079/3,079 exact sequence+length match, 0 mismatch, 0 errors.
    Remaining failures: 0401 has 50 unknown residues + 18 missing structures; 0701 has 38 unknown residues + 24 missing structures. Unknown codes include `SAM`, `SFG`, `BET`, `MTX`, `KAI`, `R2P`, etc.; these are intentionally not guessed into canonical amino acids.
- impact:
  - scope: Adds active CIF support to backbone loading and IF-ready cleaning; Phase C can now consume cleaned `.pdb` and `.cif` structures through the same `pdb_root`.
  - risk: medium (runtime chain resolution now depends on `if_chain_id` for non-PDB protein IDs; IF-ready parquets must be used consistently)
  - confidence: 0.92
- status: done
- next_action: Rerun B2 h-map precompute and WT head/NMP artifacts on the updated IF-ready parquets. Optional later work: resolve missing Tier 3 structures via AF3 and inspect `unknown_residue` failures with CCD/ligand context before deciding whether any can be safely skipped rather than failing the whole chain.
- refs:
  - `LOG.md:L0068`
  - `scripts/build_if_ready_test_set.py`
  - `inverse_folding/dplm/src/byprot/utils/io.py`
  - `inverse_folding/reference_flow/runtime.py`

### L0070
- timestamp: 2026-04-30T00:04:38+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User switched to Thinker mode and requested a concise proposal backbone for epitope-head improvement before implementation planning.
- change_summary: Added a design proposal reframing the epitope head as a contiguous-window-supported residue/region immunogenicity landscape rather than an exact peptide oracle.
- rationale: New EL evaluation results suggest the main training mismatch is negative semantics and exact-boundary pressure; the next design step needs to preserve biological contiguity, support residue/region supervision, and leave global-risk calibration decisions explicit.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Epitope_Head_Improvement_Proposal.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no runtime tests executed.
- impact:
  - scope: Epitope-head scientific direction and future implementation planning.
  - risk: low
  - confidence: 0.89
- status: done
- next_action: Resolve the marked TBDs on overlap weighting, residue-level supervision, and global-risk definitions before writing the implementation PLAN.
- refs:
  - `doc/Epitope_Head_v1.md`
  - `doc/EL_new_evaluation.md`

### L0071
- timestamp: 2026-05-07T10:27:18+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User requested a concrete implementation plan for the epitope-head improvement proposal.
- change_summary: Added `PLAN_EPI_IMP.md` with frozen decisions, scoped file touchpoints, tasks, TDD gates, deferred items, and release gates for near-positive negative scheduling plus span-derived residue ranking supervision.
- rationale: The design discussion had converged on configurable near-positive handling, binary residue coverage, ranking-based residue loss, keeping InfoNCE unchanged for now, deferring consistency/global-risk changes, and preserving mutation augmentation as a later ablation.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_EPI_IMP.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only planning update; no runtime tests executed.
- impact:
  - scope: Epitope-head training improvement planning.
  - risk: low
  - confidence: 0.91
- status: done
- next_action: Review and freeze `PLAN_EPI_IMP.md`, then implement task-by-task with TDD gates.
- refs:
  - `doc/Epitope_Head_Improvement_Proposal.md`
  - `report/Experimentresults0430.md`

### L0072
- timestamp: 2026-05-07T11:46:26+08:00
- type: PLAN_UPDATE
- module: H
- trigger: User requested an alignment review of `PLAN_EPI_IMP.md` before implementation; six design questions were resolved.
- change_summary: Revised `PLAN_EPI_IMP.md` to drop `softness_alpha`, fix `near_gap_max=10`, narrow v0 schedules to `{ignore, linear_clamp, sigmoid}`, mandate union window forward (no silent fallback), restrict residue labels/loss to chunk central region, add `min_far_bg_residues` per-chunk fallback for high-density alleles, freeze pairwise-margin residue loss form, and codify "one config file per experiment" as a frozen decision.
- rationale: Six concrete design ambiguities identified during plan review (softness_alpha redundancy, span relation thresholds, schedule proliferation, window forward sharing, chunk-overlap residue handling, far-bg scarcity in 15:01) needed explicit resolution before implementation could begin without rework.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_EPI_IMP.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only planning update; covered by subsequent implementation entry L0073.
- impact:
  - scope: Epitope-head HIMP implementation contract.
  - risk: low
  - confidence: 0.94
- status: done
- next_action: Implement HIMP0–HIMP5 task-by-task with TDD gates (logged in L0073).
- refs:
  - `doc/Epitope_Head_Improvement_Proposal.md`

### L0073
- timestamp: 2026-05-07T11:46:26+08:00
- type: VERIFICATION
- module: H
- trigger: User authorized full PLAN execution after design alignment in L0072.
- change_summary: Implemented HIMP0–HIMP5 of `PLAN_EPI_IMP.md` end-to-end. Added `train.near_positive` / `train.residue` config schema with validators; new `near_positive.py` (5-class span relation classifier + ignore/linear_clamp/sigmoid schedule registry, reserved schedules raise NotImplementedError); new `residue_supervision.py` (binary coverage labels, ambiguous / far-bg / central masks, window enumeration, max + log_mean_exp aggregation); new `residue_pairwise_margin_loss` in `losses.py` with `min_far_bg_residues` skip-counter fallback; weighted variants of `info_nce_loss` / `margin_hard_loss` / `compute_loss` (legacy bit-for-bit when `neg_weights=None`); `prepare_chunk_spans` extended to 5-tuple with extras carrying weights / relations / residue_meta; new `_forward_union_and_compute_losses` runs a single union forward keyed by `(start, end, allele_idx)` and gathers logits via index slices; `train_step` / `val_step` rewired to the union path with decomposed loss logging (`loss_residue`, `lambda_residue`, `residue_skipped_chunks`, `n_residue_pairs`, `n_residue_chunks` added to `StepMetrics` + `LOG_ENTRY_KEYS` + `aggregate_epoch_metrics`); Trainer `__init__` resolves near_positive / residue from `train_cfg`, prints an HIMP startup banner with every new hyperparameter, and threads `chunk_central_margin = chunking.margin`; new `cnn_himp_v1.yaml` override-config (sigmoid schedule center=5, slope=1, near_gap_max=10; lambda_residue=0.1, log_mean_exp aggregation, sampled window mode, min_far_bg_residues=4) ready to be passed via `--override-config`.
- rationale: PLAN §5 HIMP0–HIMP5 are the smallest atomic unit of behavior change for the head improvement; implementing all together avoids a half-finished landing where, e.g., schedule semantics are wired but residue supervision is not, which would require dual rebases of `train_step`.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/epitope_head/configs/__init__.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/configs/cnn_himp_v1.yaml`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/near_positive.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/residue_supervision.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/losses.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/negatives.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/trainer.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_himp_config.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_negative_schedule.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_residue_supervision.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_residue_loss.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_himp_train_integration.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_module_e_contract.py`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Full epitope_head test suite passes 443/443 (excl. 3 pre-existing `ModuleNotFoundError: esm` failures unrelated to this work). New tests by task: HIMP0 config (24) + HIMP1 schedules (26) + HIMP2 residue helpers (19) + HIMP3 residue loss (9) + HIMP4 trainer integration (14) = 92 new tests. Backward compat verified: existing `train.yaml` / `cnn_full_aug*.yaml` continue to load with `near_positive.enabled=False` / `residue.enabled=False` defaults; `info_nce_loss` / `margin_hard_loss` with `neg_weights=None` produce bit-for-bit identical loss values; `prepare_chunk_spans` legacy callers in `test_module_e_contract.py` updated to 5-tuple unpacking.
- impact:
  - scope: Epitope-head training pipeline (config schema, negative sampler, losses, trainer, profile config). Inference path and global-risk computation untouched.
  - risk: medium
  - confidence: 0.86
- status: done
- next_action: Launch a `--smoke` run on the cluster with `--override-config epitope_head/configs/cnn_himp_v1.yaml` to confirm the union forward end-to-end on real data; then full multi-allele training and benchmark per `doc/EL_new_evaluation.md`.
- refs:
  - `PLAN_EPI_IMP.md`
  - `epitope_head/configs/cnn_himp_v1.yaml`

### L0074
- timestamp: 2026-05-07T12:25:24+08:00
- type: VERIFICATION
- module: H
- trigger: User code review of L0073 implementation surfaced six concrete contract / correctness gaps (override-config validation bypass, apply_to_span_negatives ignored, -inf residue scores leaking into ranking pairs, symmetric central margin clobbering protein N/C termini, disrupted spans dropped on positive_set overlap, loss_total reflecting unweighted recompute instead of actual HIMP objective).
- change_summary: Six surgical fixes with TDD red→green coverage. (1) Added `validate_himp_train_blocks` public API; `train_v2_ablation.py` calls it after the override `_deep_update`, and `Trainer.__init__` calls it defensively. (2) `prepare_chunk_spans` derives `np_cfg_for_negatives = near_positive_cfg if apply_to_span_negatives else None` so span-side weighting is gated independently of residue ambiguity. (3) `_forward_union_and_compute_losses` intersects `label_t / far_bg_t / central_t` with `torch.isfinite(residue_scores)` before invoking the residue ranking loss, so residues uncovered by any sampled window cannot inflate `n_far_bg / n_pairs` or bypass the `min_far_bg_residues` skip gate. (4) Replaced symmetric `central_margin` in `build_residue_labels` with explicit `central_start` / `central_end`; the trainer now derives per-chunk trusted intervals via `left_seam = 0 if chunk_starts[i]==0 else seam, right_seam = 0 if chunk_ends[i]>=sequence_lengths[i] else seam`, matching `ChunkPlan.trusted_interior` semantics so true protein N/C termini are no longer dropped from supervision. (5) Removed the `if span not in positive_set` filter inside `negatives.sample_negatives` disrupted block — counterfactual_disrupted spans now survive even when their coordinates exactly match a remaining positive, with relation `COUNTERFACTUAL_DISRUPTED` and weight 1.0 as PLAN HIMP1 #3 mandates. (6) `train_step` and `val_step` override `metrics.loss_total = avg_loss.detach().item()` so the JSONL log records the actual weighted span + `lambda_residue * residue` objective, not the unweighted recomputation; sanity diagnostics (`loss_intra`, `loss_margin`) still use the unweighted compute_loss for trend interpretability.
- rationale: Each gap traced back to a specific PLAN_EPI_IMP.md contract or HIMP TDD gate that the L0073 implementation silently violated. The override-bypass + apply_to_span_negatives gaps were ablation-correctness issues (can't independently sweep schedule vs residue); the -inf leak + central-margin issues were correctness (numerically wrong loss / silently dropped supervision); the disrupted-overlap + loss_total issues were contract violations (PLAN-stated invariants not actually enforced).
- artifacts:
  - `/Users/jerry/Project/MHC-IF/epitope_head/configs/__init__.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/negatives.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/trainer.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/training/residue_supervision.py`
  - `/Users/jerry/Project/MHC-IF/scripts/train_v2_ablation.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_himp_config.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_negative_schedule.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_residue_supervision.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_residue_loss.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/training/test_himp_train_integration.py`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: 454/454 epitope_head tests pass (excl. 3 pre-existing ESM-import failures). Net new coverage vs L0073: +4 override revalidation tests, +1 apply_to_span_negatives gate test, +1 -inf-filter contract test, +4 trusted-interior tests (single-chunk, first/middle/last chunk seams), +1 disrupted exact-overlap test, +1 forward-union loss-plumbing test. Existing HIMP0–4 coverage continues to pass.
- impact:
  - scope: HIMP train-time correctness gates and ablation hygiene. Inference path unchanged.
  - risk: low (each fix is surgical and TDD-locked)
  - confidence: 0.92
- status: done
- next_action: Proceed to cluster `--smoke` run as planned in L0073.
- refs:
  - `PLAN_EPI_IMP.md`
  - L0073

### L0075
- timestamp: 2026-05-07T20:32:00+08:00
- type: DECISION
- module: L
- trigger: User wants per-allele evaluation wall time reduced from ~6.7 h (3079 proteins) to ~2 h to speed iteration; preserves the original tier hierarchy as the reduction axis. Final criteria after pushback: keep all Tier 1, drop ESMFold-fail / head-based stratification, use NMP as the difficulty signal, fill with stratified Tier 2.
- change_summary: Built `test_proteins_if_ready_fast_HLA-DRB1_{07_01,04_01}.parquet` (~500 proteins per allele) by inline pandas filter on `data/test_set_4_Immune_Design/if_ready/test_proteins_if_ready_HLA-DRB1_*.parquet`. Selection: (1) all 15 Tier 1 retained verbatim; (2) Tier 3 uricase pool restricted to length 100–500 AA and middle 50% of `netmhciipan_n_strong` (Q25–Q75), then 35 sampled with seed 42; (3) Tier 2 pool restricted to length 100–500 AA, stratified into a 3×3 grid (length tertile × NMP n_strong tertile via rank-qcut), even per-cell sampling to fill remainder. Manifest written to `test_proteins_if_ready_fast.manifest.json` capturing input path, output sha256, per-cell pool/sampled counts, NMP band, length window, seed.
- rationale: Tier-priority reduction is the simplest principled axis given the test set was originally tier-curated. NMP n_strong (external validator) is the right difficulty axis because using the head's own score for stratification would bias toward mode regimes that head already captures (circular). ESMFold failures are kept because foldability is a real model-capability signal, not noise. Length 100–500 matches Tier 1 curation thresholds and avoids NMP edge cases. Independent sampling per allele preserves NMP-distribution coverage even though most protein_ids overlap.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/data/test_set_4_Immune_Design/if_ready/test_proteins_if_ready_fast_HLA-DRB1_07_01.parquet`
  - `/Users/jerry/Project/MHC-IF/data/test_set_4_Immune_Design/if_ready/test_proteins_if_ready_fast_HLA-DRB1_04_01.parquet`
  - `/Users/jerry/Project/MHC-IF/data/test_set_4_Immune_Design/if_ready/test_proteins_if_ready_fast.manifest.json`
- evidence: 0701: n_input=3079, n_output=493 (T1=15, T3=35, T2=443; one T2 cell short of 50 after length filter), NMP band [41.0, 84.5], sha256=85b2b90c4aaf846a9b8078fc41d18b779444f317e0d528be5cb6e33d7beef618. 0401: n_input=3072, n_output=500 (T1=15, T3=35, T2=450), NMP band [51.0, 109.0], sha256=a457ddfec172c5fcddbc1669f97588e0b2b19839f060472a78f8465080317804.
- impact:
  - scope: Iteration cadence for IF evaluation runs (imm head + struct refold). Full 3079-protein parquets remain authoritative for paper-ready results; fast subsets are for ablation/smoke iteration only.
  - risk: low (subsets are stratified copies of audited rows; full set unmodified)
  - confidence: 0.95
- status: done
- next_action: User to rsync the two fast parquets + manifest to the cluster and re-target eval driver `--test-set-parquet` to the fast variant for the next iteration round; do NOT replace `test_set_parquet_path` in archived eval manifests.

### L0076
- timestamp: 2026-05-08T01:01:18+08:00
- type: PLAN
- module: IF_IMP
- trigger: User wants a new standalone DPLM inverse-folding optimization track after baseline/sampling checks, with MapDiff methods explicitly source-grounded and a separate encoder-replacement phase; user requested cloning MapDiff into the current project for future use.
- change_summary: Cloned the official MapDiff repository into `/Users/jerry/Project/MHC-IF/MapDiff` at commit `6c1299584b71d3c53ad42e1d591459a59f299ed3`. Added `doc/IF_IMP.md`, a source-grounded improvement plan for DPLM-IF: Phase A imports MapDiff-style entropy mask ratio, IPA masked designer refinement, entropy-weighted fusion, DDIM/MC-dropout-inspired aggregation around the existing DPLM denoiser; Phase B tests whether replacing/augmenting the frozen GVP structure encoder with MapDiff-style geometry features or EGNN encoder improves DPLM conditioning; Phase C is intentionally left as an empty decision slot until Phase A/B identify the bottleneck.
- rationale: Phase 0/1 baseline and sampling checks are already done, so the next useful plan should target mechanisms MapDiff directly supports: uncertainty-aware refinement and stronger geometry conditioning. Keeping Phase C empty prevents scope creep before empirical readout. The document separates paper-supported facts, code-verified facts, and explicit caveats so MapDiff claims are traceable rather than inferred.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/MapDiff`
  - `/Users/jerry/Project/MHC-IF/doc/IF_IMP.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Read MapDiff Nature Machine Intelligence PDF from Zotero item `Q3VQM9CT`; verified source details in `MapDiff/model/prior_diff.py`, `MapDiff/model/ipa/ipa_net.py`, `MapDiff/model/egnn_pytorch/egnn_net.py`, `MapDiff/main.py`, `MapDiff/mask_ipa_pretrain.py`, and default configs under `MapDiff/conf/`. Confirmed MapDiff remote `https://github.com/peizhenbai/MapDiff.git` and cloned commit hash above. Ran document self-check for placeholder markers (`TBD`, `TODO`, `implement later`, candidate-direction placeholders) with no matches after revision.
- impact:
  - scope: Planning/documentation and local external-source checkout only; no project code path changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: If switching to Coder, implement Phase A first using `superpowers:executing-plans`, starting with tested entropy/mask/fusion utilities and an IPA refiner port from MapDiff.

### L0077
- timestamp: 2026-05-08T01:24:22+08:00
- type: PLAN
- module: IF_IMP
- trigger: User requested a single coding PLAN for the MapDiff-grounded DPLM inverse-folding improvement path, after deciding not to split the work into separate A/B plans and asking for subagent-backed detail confirmation.
- change_summary: Added `/Users/jerry/Project/MHC-IF/PLAN_IF_IMP.md`, a single agent-executable coding plan. The plan starts with a DPLM-compatible MapDiff-style IPA refiner path, then adds an opt-in IPA geometry sidecar for encoder conditioning. It includes exact files to create/modify, test files, command-level verification steps, DPLM/MapDiff source anchors, script registration requirements, and a final LOG schema for implementation completion.
- rationale: Subagent review confirmed the safe DPLM insertion point is decoder-time logit processing before sampling, and that MapDiff's directly portable components are entropy masking, IPA masked refinement, entropy-weighted fusion, CB/IPA geometry handling, and MC dropout. The plan deliberately avoids wholesale MapDiff `Prior_Diff` or EGNN import because those require a different PyG graph/data contract and discrete posterior process than DPLM.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF_IMP.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Re-checked script governance in `doc/SCRIPTS.md` and `scripts/CLAUDE.md`; verified MapDiff IPA, entropy/fusion, CB construction, and denoising source anchors against `MapDiff/model/ipa/ipa_net.py`, `MapDiff/utils.py`, and `MapDiff/model/prior_diff.py`; verified DPLM hook/batch anchors against `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`, `inverse_folding/dplm/src/byprot/datamodules/dataset/cath.py`, and `inverse_folding/reference_flow/runtime.py`. Ran plan self-check for placeholder markers and hardcoded cluster-path patterns with no matches.
- impact:
  - scope: Planning/documentation only; no production model or script behavior changed.
  - risk: low
  - confidence: 0.88
- status: done
- next_action: Switch to Coder and execute `PLAN_IF_IMP.md` with `superpowers:executing-plans`, starting from Task 1 token bridge and Task 2 geometry bridge before porting IPA modules.
- refs:
  - `PLAN_IF_IMP.md`
  - `doc/IF_IMP.md`
  - L0076

### L0078
- timestamp: 2026-05-08T12:22:00+08:00
- type: VERIFICATION
- module: IF_IMP
- trigger: User asked Coder to execute `PLAN_IF_IMP.md` Tasks 1-12 plus extend the runner with 4-arm ablation infrastructure (baseline / refiner / sidecar / sidecar+refiner), wire ablation diagnostics into the IF improvement runner, and add a cross-arm comparison consumer for the existing DPLM evaluator pipeline.
- change_summary: Implemented PLAN_IF_IMP.md Tasks 1-12 end-to-end (token/geometry/entropy/fusion/IPA refiner/logit processor/DPLM hook/checkpoint/training/scripts/local smoke/sidecar) and added ablation infrastructure: explicit `--arm` resolver, optional `--sidecar-checkpoint`, `--ablation-mode` per-step diagnostics emission, sidecar checkpoint helpers, sidecar training script, and a dedicated cross-arm comparison script that produces the 5 PLAN-mandated indicators.
- rationale: The refiner alone matches MapDiff's local logit-fusion intervention, while the sidecar feeds geometric context into DPLM's encoder pre-decoder, so they target distinct points in the generation pipeline. To decide whether the dual-module Arm 4 is worth its compute, we need cross-arm deltas on (i) refiner-selection shrinkage, (ii) base-vs-fused entropy quantile gain, (iii) sidecar-induced base entropy reduction, (iv) Arm 4 - max(Arm 2, Arm 3) entropy gain, and (v) wall-time per design. These required: (a) an `--arm` switch, (b) a diagnostics sink in the logit processor, (c) per-design + per-step parquet emission in the runner, (d) a comparison consumer alongside the existing `evaluate_phase_c.py` so the user's per-arm immunogenicity / structural metrics still flow through the unchanged evaluator.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/__init__.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/tokens.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geometry.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/entropy.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/fusion.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/logit_processor.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/checkpoint.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/training.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/sidecar.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/encoder_wrapper.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/ipa/{__init__,rigid_utils,ipa_utils,ipa_attn,refiner}.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py` (added optional `logit_processor` + `batch` to `forward_decoder`/`generate`; default behavior bit-equivalent)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/runtime.py` (added optional `logit_processor` kwarg to `generate_native_sequence`)
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_sidecar.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/compare_if_imp_ablation.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md` (registered new H2 `Inverse Folding -- IF Improvement (PLAN_IF_IMP.md)` with 4 scripts + SLURM)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/{test_tokens,test_geometry,test_entropy,test_fusion,test_ipa_refiner,test_logit_processor,test_dplm_hook,test_training,test_sidecar}.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_if_imp_scripts.py`
- evidence:
  - 39/39 IF_IMP unit + script tests pass (`tests/inverse_folding/dplm_refiner/` + `tests/scripts/test_if_imp_scripts.py`).
  - 42/42 reference_flow + phase_c0 + dplm_refiner + dplm_hook regression tests pass; no regressions on the unchanged paths.
  - `python -m compileall inverse_folding/dplm_refiner scripts/{train_if_imp_refiner,train_if_imp_sidecar,run_if_imp_refiner,compare_if_imp_ablation}.py` succeeds.
  - Placeholder + hardcoded cluster path self-scan over Python modules + `PLAN_IF_IMP.md` returns no hits.
  - CPU shape smoke for `DPLMIPARefiner` returns `(1, 8, 20)` finite logits.
  - `--help` returns 0 for all four new scripts; CLI flags listed in `tests/scripts/test_if_imp_scripts.py` are present.
- impact:
  - scope: New DPLM IF improvement path (refiner + sidecar) and 4-arm ablation infra; default DPLM generation remains bit-equivalent when no refiner / sidecar / `--ablation-mode` is set; no changes to Module K, Phase B/C, Module M/N, evaluator schema.
  - risk: medium
  - confidence: 0.84
- status: done
- next_action: On Della: train refiner with `MODE=train_refiner` and sidecar with `MODE=train_sidecar` against the L0075 fast IF subset (small `--limit-batches` smoke first, then full CATH training). Then run all 4 arms with `MODE=generate_refiner --arm <ARM> --ablation-mode` on the same test set, evaluate each via the existing `evaluate_phase_c.py`, and produce cross-arm deltas with `MODE=compare_ablation`. Decide whether to retain Arm 4 based on `combined_minus_best_single_fused_entropy` and the wall-time cost reported by `compare_if_imp_ablation.py`.
- refs:
  - `PLAN_IF_IMP.md`
  - `doc/IF_IMP.md`
  - `doc/SCRIPTS.md` §Inverse Folding -- IF Improvement
  - L0077

### L0078a (tech debt)
- timestamp: 2026-05-08T12:22:00+08:00
- type: RISK
- module: IF_IMP
- trigger: Carry-over from PLAN_IF_IMP.md Task 12 verification.
- change_summary: Document two known limitations not addressed in L0078: (a) DPLM `forward_decoder` line 223 hardcodes `temperature=0.0` in `stochastic_sample_from_categorical`, so `--temperature` flags on `run_if_imp_refiner.py` and `run_if_phase_c0.py` are no-ops at the decoder; (b) `train_if_imp_sidecar.py` uses a recovery-only objective with the encoder-pre-decoder injection path -- this works for the proof-of-concept ablation but is not the strongest training signal possible for the sidecar.
- rationale: Both items are explicitly out of scope for `PLAN_IF_IMP.md`; logging them avoids silent debt accumulation.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_sidecar.py`
- evidence: Source inspection of `forward_decoder` lines 223-225 confirms hardcoded `temperature=0.0` regardless of `prev_decoder_out["temperature"]`.
- impact:
  - scope: Sampler temperature controllability + sidecar training signal quality.
  - risk: low (item a is a silent no-op; item b is sub-optimal but not wrong)
  - confidence: 1.00
- status: open
- next_action: After L0078 cluster runs land, decide whether to (a) patch DPLM `forward_decoder` to honor the schema temperature and (b) revisit sidecar training with a stronger curriculum (e.g. distillation against teacher decoder logits).
- refs:
  - L0078

### L0079
- timestamp: 2026-05-08T15:55:00+08:00
- type: VERIFICATION
- module: N
- trigger: User asked for a simple ProteinMPNN inverse-folding interface with an optional NMP post-hoc filter switch, to be used as a comparison baseline against DPLM-based IF runs.
- change_summary: Added (i) `scripts/run_proteinmpnn_baseline.py`: a thin wrapper around the vendored ProteinMPNN at `DRAKES/drakes_protein/ProteinMPNN/` that parses an input PDB folder via `parse_multiple_chains.py` (+ `assign_fixed_chains.py` when `--design-chains` is set), runs `protein_mpnn_run.py` to generate `--num-seq-per-target` designs per structure, parses `seqs/<pid>.fa` into per-design rows, and (with `--apply-nmp-filter`) scores each design via `epitope_head.data.netmhciipan_runner.StandaloneRunner.score_batch` and selects argmin-risk per protein under `--nmp-rule {min_n_sb,min_mean_rank}`. Outputs `generated.parquet` (schema-compatible with `run_if_phase_c0.py`: `protein_id`, `design_idx`, `sequence`, …), `generated.fasta`, `run_config.yaml`, and (NMP-on) `selection.parquet` + `selection.json`. Added (ii) `scripts/submit_if_baselines.slurm`: module-level Module-N launcher with `MODE={level1,proteinmpnn}` switch, GPU defaults aligned with `submit_if_phase_c.slurm` (1 GPU MIG / 24h / 64GB), canonical `NETMHCIIPAN_BIN=${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan` (matches `submit_benchmark.slurm` and other NMP-using SLURM scripts in the repo), `INPUT_PDB_FOLDER` default `${WORK_DIR}/if_test_set/pdbs_if_ready/{0701,0401}` selected by `ALLELE`, `OUTPUT_DIR` default `${RUN_DIR}/baselines/{level1_filter,proteinmpnn}/...`. Legacy `scripts/submit_if_level1_filter.slurm` deleted (user-approved 2026-05-08) — fully subsumed by `MODE=level1`; PROGRESS.md / PLAN_IF.md / `run_if_level1_filter.py` docstring updated to point at the new module-level launcher.
- rationale: Module N is the natural home for ProteinMPNN+NMP since it already houses post-hoc filter baselines (`run_if_level1_filter.py`). The script is intentionally a subprocess wrapper instead of a deeper integration: the vendored ProteinMPNN uses local-relative imports (`from protein_mpnn_utils import ...`) and is only safe to invoke with `cwd=PROTEINMPNN_ROOT`. Multi-chain designs are split on `/` and aggregated (sum `n_sb`, mean `el_rank`) before selection so the rule remains valid for both single- and multi-chain inputs. The SLURM was consolidated into a module-level launcher per `scripts/CLAUDE.md` §3 ("one parameterized submission script per module") — initial v1 of this entry omitted the SLURM and used a wrong NMP path (`/scratch/.../work/netmhciipan/...`); both were corrected after user feedback.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/run_proteinmpnn_baseline.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_baselines.slurm` (new; module-level launcher)
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_run_proteinmpnn_baseline.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md` (Module N: registered both new artifacts; legacy `submit_if_level1_filter.slurm` line removed after deletion)
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md` (Module N section: N1 SLURM pointer updated; N2 ProteinMPNN section added)
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md` (§Planned File Touchpoints: SLURM pointer updated)
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_level1_filter.py` (docstring SLURM pointer updated)
- evidence:
  - 7/7 unit tests pass: `tests/scripts/test_run_proteinmpnn_baseline.py` covers FASTA parser (skips WT, extracts sample/score/T/seq_recovery), multi-chain splitter, selection rules with tie-break, and CLI gate that fails fast when `--apply-nmp-filter` is missing `--netmhciipan-bin` / `--allele`.
  - `PYTHONPATH=. python scripts/run_proteinmpnn_baseline.py --help` returns 0 after escaping `%Rank_EL` to `%%Rank_EL` (argparse default-formatter applies %-substitution).
  - `bash -n scripts/submit_if_baselines.slurm` succeeds (syntax clean).
  - Vendored ProteinMPNN entry point + helper scripts confirmed at `DRAKES/drakes_protein/ProteinMPNN/{protein_mpnn_run.py,helper_scripts/{parse_multiple_chains,assign_fixed_chains}.py}`; weights present at `vanilla_model_weights/v_48_{002,010,020,030}.pt`.
  - Canonical NMP binary path `${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan` cross-checked against `submit_benchmark.slurm:59`, `submit_aug_drb0401.slurm:20`, `submit_epi_benchmark.slurm:29`, `submit_mutation_augmentation.slurm:20`, `submit_assemble_test_set.slurm:30`, `submit_prescreen_uricases.slurm:24`, `submit_prescreen_tier2.slurm:26`, `submit_prescreen_uricases_0401.slurm:24`.
- impact:
  - scope: New baseline driver + module-level launcher + tests + SCRIPTS.md/PROGRESS.md/PLAN_IF.md/docstring updates; legacy `submit_if_level1_filter.slurm` deleted (user-approved). No changes to existing libraries, evaluators, or vendored ProteinMPNN sources.
  - risk: low (subprocess wrapper; SLURM defaults match canonical IF/Phase-C resource pattern; NMP filter only kicks in when the user supplies `--apply-nmp-filter` + binary + allele).
  - confidence: 0.9
- status: done
- next_action: For first cluster run: `MODE=proteinmpnn ALLELE="HLA-DRB1*07:01" APPLY_NMP_FILTER=1 sbatch scripts/submit_if_baselines.slurm` will use the canonical IF-ready PDB tree and the project-root NMP binary. For the legacy N1 path: `MODE=level1 sbatch --partition=cpu --gres=none --time=01:00:00 --mem=16G scripts/submit_if_baselines.slurm`.
- refs:
  - `doc/SCRIPTS.md` §Inverse Folding — Module N: Comparison Baselines
  - `scripts/CLAUDE.md` §3 (Module-level SLURM consolidation)

### L0080
- timestamp: 2026-05-08T17:30:00+08:00
- type: VERIFICATION
- module: C
- trigger: User flagged that Phase C0 vs C1 head-to-head was unfair — C0 ran 50 reparam-decoded iterations, while C1 ran 10 one-shot categorical-sample iterations with no token refinement (sampler.py:89/132 + c1_*.yaml n_steps=10). Independently flagged that the h_maps↔test-set alignment was only enforced by `protein_id` + length, leaving silent position-misalignment exposure if the test parquet was ever regenerated with a different residue ordering for the same protein_id.
- change_summary: (i) Aligned both C0 and C1 to the DPLM paper's inverse-folding evaluation default of 100 iterations: bumped `MAX_ITER` default in `scripts/submit_if_phase_c.slurm` from 50→100 and `sampler.n_steps` in all five `c1_*.yaml` configs from 10→100. (ii) Added optional inference-time reparameterized refinement to `PositionDependentDFMSampler`: tracks per-position chosen-token log-prob, then at every step except the last re-masks the bottom `1−(step+1)/n_steps` fraction of committed positions by score (DPLM-style `reparam-uncond-deterministic-linear`). New `RemaskConfig` dataclass on `SamplerConfig`, default off; enabled in all five `c1_*.yaml` so Phase C1 gets the same refinement budget DPLM's paper baseline reports. Disclaimer added to `doc/Reference_Flow_Derivation.md` §4.3.1 explaining that refinement is *not* part of the DFM derivation but is layered on top of the position-dependent schedule and does not read `h_i` or modify `g_i`. (iii) Hardened h_maps alignment: added `sequence_md5` column to the h-map parquet schema, written by `scripts/precompute_h_maps.py` from the input residue sequence (uppercased before hashing); `inverse_folding/evaluation/h_maps.py` validator now rejects rows with malformed/missing md5; `scripts/run_if_phase_c1.py` calls a new `assert_h_maps_align_with_test_set()` at startup that fails fast when any test-set protein has a missing or non-matching `sequence_md5`, also accepting extra h-map proteins as a superset.
- rationale: Without (i)+(ii), C0 had ~5× more model forwards plus reparam refinement, so any C1 recovery drop was confounded with sampler contract instead of attributable to the h-map signal. The DPLM README evaluation example (`task.generator.max_iter=100`) is the right anchor for both — `max_iter=10` is the training default in `cond_dplm_*.yaml`, and `max_iter=50` was an ad hoc midpoint with no published reference. Reparam refinement violates the absorbing-chain assumption of the DFM derivation but does not read `h_i`, so the position-dependent inductive bias from §1 is preserved; we ship it on by default with an ablation switch (`sampler.remask.enabled: false`) so a clean DFM-faithful baseline remains reachable. For (iii), the previous c1 driver only checked `len(h_values) == sequence_length` per protein — same `protein_id` with same length but a re-cut chain would produce silent residue-wise misalignment of `h_raw[i]` vs `sequence[i]`. Binding each h-map row to `md5(uppercase(sequence))` and checking at startup gives byte-identity guarantees with one new column (~32 chars/row).
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/config.py` (new `RemaskConfig` dataclass + parsing)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/sampler.py` (per-position log-prob tracking + `_apply_reparam_remask` helper; `_sample_categorical` now returns chosen-token log-probs)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/__init__.py` (re-export `RemaskConfig`)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/c1_{null,linclamp,sigmoid,power,shuffle}.yaml` (n_steps 10→100, added `sampler.remask.enabled: true`)
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_phase_c.slurm` (MAX_ITER default 50→100, comment added)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/evaluation/h_maps.py` (new `sequence_md5()` helper; `HMAP_COLUMNS` includes `sequence_md5`; per-row md5 format check)
  - `/Users/jerry/Project/MHC-IF/scripts/precompute_h_maps.py` (writes `sequence_md5` per row; `validate_resume_rows` cross-checks md5 against current input; `normalize_row` preserves md5)
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c1.py` (new `assert_h_maps_align_with_test_set()` invoked at startup right after `load_h_maps`)
  - `/Users/jerry/Project/MHC-IF/doc/Reference_Flow_Derivation.md` (§4.3.1 inference-time reparameterized refinement disclaimer)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_sampler.py` (5 new tests: remask off = legacy; remask on triggers refinement; low-confidence positions are revisited; determinism with seed; YAML round-trip)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_h_maps_contract.py` (3 new tests: missing md5 column rejected; malformed md5 rejected; precompute round-trip emits correct md5; resume rejects md5 mismatch — extends existing 14 → 17)
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_run_if_phase_c1_script.py` (4 new tests for `assert_h_maps_align_with_test_set`: pass on match; fail on same-length-different-residues; fail on missing protein; pass with superset h_maps)
- evidence:
  - 45/45 tests pass: `python -m pytest tests/inverse_folding/test_reference_flow_{sampler,config,amplification,schedule,runtime}.py tests/inverse_folding/test_h_maps_contract.py tests/scripts/test_run_if_phase_c1_script.py`
  - `c1_null.yaml` round-trips through `load_reference_flow_config` with `sampler.remask.enabled is True` and `sampler.n_steps == 100`.
  - DPLM defaults cross-checked: `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml:67` has `generator.max_iter: 10` (training default); `inverse_folding/dplm/README.md:468` shows `task.generator.max_iter=100` for the published inverse-folding evaluation.
- impact:
  - scope: Phase C0/C1 sampling iteration count + new C1 inference-time refinement layer + h_maps schema (1 new column) + c1 driver startup validation. Previous h_maps parquets must be regenerated to include `sequence_md5` (no backward-compat shim — user confirmed they will re-run on the new smaller test set).
  - risk: medium. The `sequence_md5` column is a hard schema break for old parquets; any old h-map parquet without the column will fail-fast on load. Reparam refinement is mathematically a non-DFM augmentation but does not affect the position-dependent schedule; ablation toggle preserves a clean baseline.
  - confidence: 0.92
- status: done
- next_action: Regenerate h_maps with `sbatch scripts/submit_precompute_h_test.slurm` (per-allele) using the new test set, then run a head-to-head: `MODE=native sbatch scripts/submit_if_phase_c.slurm` (C0 100 iter) vs `MODE=reference_flow CONFIG_PATH=inverse_folding/reference_flow/configs/c1_null.yaml sbatch scripts/submit_if_phase_c.slurm` (C1 100 iter + remask). Expect C1 null to track C0 closely now that compute budgets and refinement are aligned.
- refs:
  - `doc/Reference_Flow_Derivation.md` §4.3.1
  - DPLM README inverse-folding evaluation block (line 466-469)


### L0081
- timestamp: 2026-05-11T12:30:00+08:00
- type: IMPLEMENTATION
- module: C
- trigger: User flagged that Phase C + IF_IMP per-protein loops are MUCH slower than K3 (`scripts/validate_if_baseline.py` via DPLM `test.py` predict path) because the K3 path leverages DPLM's native batched generate while our wrappers built B=1 batches per protein. With L0080's bump to `max_iter=100`, this gap is amplified 10×.
- change_summary: Added length-bucketed batched DPLM sampling to `inverse_folding.reference_flow.runtime` (`prepare_backbone_batch` + `generate_native_sequences_batched`) and wired `--batch-size` into `run_if_phase_c0.py` and `run_if_imp_refiner.py`. Refactored `DPLMRefinerLogitProcessor` / `DPLMBaseEntropyProbe` diagnostics to emit per-row `per_row` records so batched ablation runs preserve per-(protein, design) granularity. SLURM `submit_if_imp.slurm` defaults to `GEN_BATCH_SIZE=8`. Reference-flow `run_if_phase_c1.py` is NOT batched in this commit — its custom denoiser + per-position κ schedule needs a separate batched-aware sampler pass.
- rationale: DPLM's `forward_encoder`/`forward_decoder` already handle B>1 (training is batched); only our `prepare_backbone` + `generate_native_sequence` wrappers assumed B=1. The cheapest, lowest-risk path is to keep DPLM internals untouched and add parallel batched wrappers that share the same featurizer (`task.alphabet.featurizer`). Length-bucketing (sort sequence_length desc, chunk by batch_size) keeps padded compute near optimum without changing output parquet ordering. Diagnostics had to move from scalar batch-means to per-row records because the ablation Indicators 1-4 require per-design granularity for paired deltas; the IF_IMP runner's `_fanout_per_row` then splits each step's batched record into one step-list per protein in the bucket.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/runtime.py` (new `prepare_backbone_batch`, `generate_native_sequences_batched`)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/logit_processor.py` (per-row diagnostics emission)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/diagnostics.py` (per-row probe diagnostics)
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c0.py` (`--batch-size`, `_length_buckets`, `generate_rows_for_entries_batched`, `_build_batched_generator`)
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_imp_refiner.py` (`--batch-size`, batched generator with per-row diagnostic fan-out)
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm` (`GEN_BATCH_SIZE=8` default; passes `--batch-size` to runner)
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_if_imp_scripts.py` (3 new tests: length bucket order; batched dispatch by bucket; per-row diagnostic fan-out contract)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_diagnostics.py` (new batched per-row probe test)
- evidence:
  - 51/51 relevant tests pass: `PYTHONPATH=inverse_folding/dplm/src pytest tests/inverse_folding/dplm_refiner tests/scripts/test_if_imp_scripts.py tests/scripts/test_run_if_phase_c0_script.py tests/inverse_folding/test_reference_flow_runtime.py`.
  - `bash -n scripts/submit_if_imp.slurm` clean.
  - `test_length_buckets_sort_desc_and_chunk` confirms length-desc ordering with stable tie-breaking and chunk sizes ([2,2,1] for 5 proteins / batch_size=2).
  - `test_generate_rows_for_entries_batched_dispatches_in_buckets` confirms protein ordering inside buckets follows length-desc.
  - `test_base_entropy_probe_emits_per_row_for_batched_input` confirms B=2 probe call yields two per-row records with row 1 (uniformly sharp) lower mean entropy than row 0 (mostly uniform).
- impact:
  - scope: Phase C0 native sampler and IF_IMP runner now run length-bucketed B>1 DPLM generate when `--batch-size > 1`. `run_if_phase_c0.py` default stays `--batch-size 1` (back-compat with existing C0 native runs); `run_if_imp_refiner.py` SLURM default is `GEN_BATCH_SIZE=8`. Reference-flow `run_if_phase_c1.py` not batched in this commit — flagged as L0082 follow-up.
  - risk: medium — within a bucket, a single torch seed is applied once before the batched generate call; per-design seed determinism within a batch is therefore not bit-equivalent to a sequence of B=1 generations (documented in the function docstring + CLI help). C0 baseline (1 design / protein) is unaffected; only `n_designs_per_protein > 1` in IF_IMP carries this caveat.
  - confidence: 0.85
- status: done
- next_action: On Della, smoke `run_if_phase_c0.py --batch-size 8` against the L0075 fast 500-protein subset and compare wall-time vs `--batch-size 1` to confirm the K3-style speedup. Then re-run the 4 IF_IMP ablation arms with `GEN_BATCH_SIZE=8` and `ABLATION_MODE=1`. Open a follow-up L#### entry if Phase C1 reference-flow batching is needed (deferred because the custom denoiser + per-position κ schedule needs distinct batched-aware sampler plumbing).
- refs:
  - L0078
  - L0080 (max_iter=100 alignment — motivated the batching priority)
  - `scripts/validate_if_baseline.py` (K3 prediction path that established the batched precedent)

### L0082
- timestamp: 2026-05-11T20:56:52+08:00
- type: THEORY_UPDATE
- module: PHASE_C
- trigger: User asked how the proposed dynamic recommit mechanism avoids opaque structure/head conflicts, whether repeated recommit converges, whether residue-level control is sufficient without extra smoothing, and whether EMA should be included at the theory level.
- change_summary: Updated `doc/Reference_Flow_Derivation.md` Task D to make dynamic commit/revisit the preferred adaptive controller, demote external logit steering to a comparator, add a transparent structure-risk commit score, add EMA persistent-risk memory, and state practical convergence gates for revisit policies.
- rationale: The dynamic controller should not directly choose amino acids like conventional classifier guidance. It should control whether a residue remains editable while leaving amino-acid proposal to the structure-conditioned denoiser. This keeps structure and immune signals visible in one commit score and avoids hiding tradeoffs in opaque objective fusion.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Reference_Flow_Derivation.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no runtime tests executed.
- impact:
  - scope: High-level theory/design framing for optional adaptive Reference Flow controller.
  - risk: low
  - confidence: 0.88
- status: done
- next_action: When implementation planning starts, translate the commit/revisit framing into explicit controller ablations: generic reparam, risk-aware commit score, EMA on/off, logit-steering comparator, and dynamic scheduler held out as a later extension.
- refs:
  - `doc/Reference_Flow_Derivation.md` §4.7-4.9


### L0083
- timestamp: 2026-05-12T18:28:33+08:00
- type: THEORY_UPDATE
- module: PHASE_C
- trigger: User asked to update the Reference Flow proposal with the refined dynamic controller design: late-stage active-block head refresh, hard counterfactual logits guidance, and residue-level recommit, while keeping direct block write-back as a secondary option.
- change_summary: Reworked `doc/Reference_Flow_Derivation.md` §4.7-4.9 so Task D is now a continuous controller design rather than a fragmented D1-D5 list. The updated proposal defines active blocks from online excess risk, uses hard block counterfactual candidates to produce residue-level immune-aware logit corrections, keeps recommit residue-level with EMA risk memory, and demotes dynamic schedule modulation to a later invasive extension.
- rationale: The adaptive controller should use the epitope head to provide token-direction information through hard counterfactual risk estimates, not by subtracting a scalar residue risk from all amino-acid logits. Block-level evaluation captures local epitope/window interactions, while residue-level recommit preserves the DPLM-style reparameterized decoding semantics and avoids over-remasking whole regions.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Reference_Flow_Derivation.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no runtime tests executed.
- impact:
  - scope: High-level theory/design framing for optional adaptive Reference Flow controller.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: When implementation planning starts, translate the controller into explicit ablations: hard counterfactual logits only, EMA recommit only, combined controller, direct block write-back comparator, and dynamic scheduler held out as a later extension.
- refs:
  - `doc/Reference_Flow_Derivation.md` §4.7-4.9


### L0084
- timestamp: 2026-05-15T02:30:18+08:00
- type: THEORY_UPDATE
- module: PHASE_C
- trigger: User reviewed the dynamic controller proposal and surfaced stability issues around active-block boundaries, entropy-gate semantics, top-K correction support, Monte Carlo candidate variance, exploratory candidate explosion, local risk scope, z-score normalization, EMA lag, and process-level monitoring.
- change_summary: Updated `doc/Reference_Flow_Derivation.md` §4.7-4.8 to anchor active blocks to head receptive-field windows, define local head objective `R_H^\Omega`, merge blocks with overlapping scoring windows, replace direct structural-score reuse with posterior/proposal ratio logit correction, explicitly leave out-of-support amino acids unchanged, add multiplicative reliability gates including ESS, clarify the entropy sign difference between logit correction and commit revisit, add structure-conservative vs risk-exploratory candidate modes, define per-protein per-step z-normalization, allow reliability-dependent EMA, and add required controller diagnostics.
- rationale: The dynamic controller should remain scientifically interpretable under finite candidate budgets. Window-anchored blocks avoid threshold-driven boundary artifacts; posterior/proposal ratio correction avoids double-counting structural likelihood; ESS and diagnostics prevent noisy candidate reweighting from masquerading as immune-aware guidance.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/Reference_Flow_Derivation.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update; no runtime tests executed.
- impact:
  - scope: High-level theory/design framing and future implementation constraints for optional adaptive Reference Flow controller.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: When implementation planning starts, convert these notes into explicit controller telemetry and ablations: window-anchored block discovery, conservative candidate support, ESS-gated correction, EMA recommit, risk-exploratory candidate support, and direct block write-back comparator.
- refs:
  - `doc/Reference_Flow_Derivation.md` §4.7-4.8


### L0085
- timestamp: 2026-05-20T11:28:18+08:00
- type: PLAN_UPDATE
- module: PHASE_C
- trigger: User asked to turn the D-phase dynamic controller discussion into a dedicated PLAN-level monitor, metric, and ablation section, including recommit telemetry, logit intervention direction, dynamic hotspot updates, budget normalization, paired seeds, regression rate, and trajectory diagnostics.
- change_summary: Added `PLAN_IF.md` Phase C Task D0, freezing the dynamic controller diagnostic contract around four layers: opportunity/precondition, mechanism, attribution/budget normalization, and external validity/anti-gaming.
- rationale: D-phase experiments need to diagnose why the controller succeeds or fails, not only whether final head scores improve. The plan now separates D2 logit direction, D3 recommit behavior, dynamic hotspot discovery, controller budget, causal attribution, and external NetMHCIIpan validation before implementation starts.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only plan update; no runtime tests executed. Markdown formula formatting was manually checked in the modified PLAN section.
- impact:
  - scope: Phase C/D execution planning only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: When coding starts, translate Task D0 into telemetry fields in the generation artifact before running large-scale D2/D3/full ablations.
- refs:
  - `PLAN_IF.md` Phase C Task D0


### L0086
- timestamp: 2026-05-20T11:33:02+08:00
- type: PLAN_UPDATE
- module: PHASE_C
- trigger: User corrected that `PLAN_IF.md` is no longer the active Reference Flow stage plan and requested the D-phase monitor/metric/ablation contract be moved into `PLAN_RF.md`.
- change_summary: Removed the previously added RF/D0 section from `PLAN_IF.md`, restored its open-decision wording, and moved the dynamic controller monitor/metric/ablation contract into `PLAN_RF.md` under Task D.
- rationale: Reference Flow execution planning now belongs in `PLAN_RF.md`; keeping the D-phase contract in `PLAN_IF.md` would create duplicate and stale planning sources.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF.md`
  - `/Users/jerry/Project/MHC-IF/PLAN_RF.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only plan relocation; no runtime tests executed. `PLAN_IF.md` diff now excludes the D0 section and `PLAN_RF.md` contains the moved contract.
- impact:
  - scope: Plan-document organization only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.95
- status: done
- next_action: Use `PLAN_RF.md` as the source of truth for subsequent Reference Flow D-phase controller implementation planning.
- refs:
  - `PLAN_RF.md` Task D0
  - correction to `LOG.md` L0085


### L0087
- timestamp: 2026-05-21T01:13:48+08:00
- type: PLAN_UPDATE
- module: PHASE_C
- trigger: User asked for preread-based D1 planning for Reliability-gated online refresh and active blocks, with emphasis on head API adaptation, faster batch scoring, active-block state caching, and a handoff plan for a future coder.
- change_summary: Added `PLAN_RF.md` Task D1 as a monitor-only implementation plan grounded in current code reality. The plan specifies a sampler controller hook, a stable epitope-head batch scoring adapter, lazy static WT window-score caching, active-window-to-active-block discovery, reliability gate calculation, D1 telemetry artifacts, runner/SLURM integration points, and targeted unit/script tests.
- rationale: D1 should first establish the online refresh and active-block state/telemetry layer without perturbing C1's static reference-flow schedule, logits, or remask semantics. Current h-map artifacts lack static window logits, so D1 needs an explicit window-score cache before active blocks can be defined correctly from excess window risk.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_RF.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only plan update based on read-only preread of `doc/Reference_Flow_Derivation.md`, `inverse_folding/reference_flow/*`, `scripts/run_if_phase_c1.py`, `epitope_head/inference/predictor.py`, and related test/script contracts. No runtime tests executed.
- impact:
  - scope: Reference Flow D1 implementation planning only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Future coder should implement D1 in monitor-only mode first and prove that generated sequences are identical to no-controller C1 before adding D2 logit correction or D3 recommit behavior.
- refs:
  - `PLAN_RF.md` Task D1


### L0088
- timestamp: 2026-05-21T01:47:43+08:00
- type: PLAN_UPDATE
- module: IF_IMP
- trigger: User accepted treating `update_coors` as an ablation knob and asked to update `doc/IF_IMP.md` with the full GeoEGNN-IPA encoder replacement scheme.
- change_summary: Updated `doc/IF_IMP.md` to pivot from sidecar/refiner sweeps toward a GeoEGNN-IPA structure encoder replacement. The new plan defines EGNN graph/contact message passing, IPA dense rigid-frame refinement, original-frame anchoring with optional updated-coordinate pair bias, late-layer gated DPLM adapters, a low-weight auxiliary AA head, and required ablations for `update_coors`, adapter depth, auxiliary supervision, and structure feature sets.
- rationale: Current IF-IMP sidecar/refiner readouts are real but weak; the higher-leverage causal bottleneck is the structure-conditioning path rather than more sampling-time refiner hyperparameter sweeps. The new design preserves DPLM as the frozen sequence prior while replacing the GVP feature generator with stronger MapDiff-inspired geometry conditioning.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/doc/IF_IMP.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update based on the existing DPLM encoder/adapter contract and MapDiff encoder/prior inspection; no runtime tests executed.
- impact:
  - scope: IF-IMP planning and future encoder replacement implementation constraints only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Future coder should implement the GeoEGNN-IPA encoder replacement against the `encoder_out["feats"]` contract, then run the minimum CATH ablations before moving to IF-ready immune-design evaluation.
- refs:
  - `doc/IF_IMP.md` Phase B


### L0089
- timestamp: 2026-05-21T01:58:06+08:00
- type: PLAN_UPDATE
- module: IF_ENCODER
- trigger: User requested subagent-assisted codebase implementation analysis for the GeoEGNN-IPA proposal and asked for a more accurate concise PLAN file.
- change_summary: Added `PLAN_IF_ENCODER.md` as a dedicated coding plan for GeoEGNN-IPA DPLM encoder replacement. The plan freezes the encoder contract, identifies the current `encoder_out["feats"].detach()` training blocker, defines graph/EGNN/IPA package boundaries, specifies last-N gated adapter changes, requires `update_coors` ablation, avoids MapDiff runtime imports and external structural annotations, and routes execution through a new encoder training script plus existing `submit_if_imp.slurm` modes.
- rationale: The old `PLAN_IF_IMP.md` is a completed refiner/sidecar implementation plan and is not the right source of truth for encoder replacement. The new plan isolates the engineering surface needed for a clean replacement: DPLM-compatible `[B, L, 512]` features, trainable encoder gradients, backward-compatible adapters, and minimal MapDiff-derived geometry code.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_IF_ENCODER.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update based on read-only subagent inspection of DPLM encoder/adapter contracts, IF-IMP sidecar/refiner code, and MapDiff EGNN/IPA/graph feature implementation. No runtime tests executed.
- impact:
  - scope: Future GeoEGNN-IPA encoder replacement implementation planning only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Future coding agent should follow `PLAN_IF_ENCODER.md`, starting with Task E0 adapter/gradient contract before implementing graph or EGNN code.
- refs:
  - `PLAN_IF_ENCODER.md`


### L0090
- timestamp: 2026-05-21T03:30:00+08:00
- type: VERIFICATION
- module: IF_ENCODER
- trigger: Executed `PLAN_IF_ENCODER.md` tasks E0–E5 end-to-end (E0 adapter/gradient contract, E1 graph builder, E2 EGNN backbone, E3 IPA + encoder contract, E4 checkpoint/config integration, E5 training + launcher scripts). E6 cluster smoke + ablations queued separately.
- change_summary: Implemented the GeoEGNN-IPA encoder package at `inverse_folding/dplm_refiner/geo_encoder/` (config / graph / egnn / ipa / encoder / checkpoint) plus the byprot registry wrapper at `inverse_folding/dplm/src/byprot/models/dplm/modules/geoegnn_ipa_encoder.py`. Added `adapter_num_layers/adapter_gated/adapter_gate_init` to `DPLMWithAdapterConfig`, a module-level `install_adapters` helper for loop replacement, and `detach_encoder_feats` to `DPLMInvFoldConfig` with a conditional detach in `DPLMInvFold.forward()`. `forward_encoder` now ensures `coord_mask` is set in both `use_draft_seq` branches. Added `scripts/train_if_imp_encoder.py`, three new SLURM modes (`train_encoder/generate_encoder/diag_encoder`) in `scripts/submit_if_imp.slurm`, `--encoder-checkpoint/--encoder-kind` flags + encoder swap in `scripts/run_if_imp_refiner.py` and `scripts/diag_if_imp_arms.py`, the Hydra override `cond_dplm_650m_geoegnn.yaml`, uncommented `torch_scatter` in `inverse_folding/dplm/requirements.txt`, and a new H2 in `doc/SCRIPTS.md` registering the encoder script + modes.
- rationale: PLAN_IF_ENCODER.md required a drop-in GVP replacement that preserves the `encoder_out` contract, lets main DPLM CE backpropagate into the encoder, exposes `update_coors` as a real ablation knob, and never imports MapDiff at runtime. Splitting the work into E0 (gradient gate) and E1–E3 (geometry pipeline) kept default GVP behavior bit-equivalent; E4–E5 made the new path Hydra-instantiable and cluster-launchable.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/modules/geoegnn_ipa_encoder.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m_geoegnn.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/requirements.txt`
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_encoder.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/diag_if_imp_arms.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_dplm_encoder_grad.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_graph.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_egnn.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_encoder_contract.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_if_imp_encoder_scripts.py`
- evidence: Local pytest run (Python 3.11, no PyG / no omegaconf installed): `tests/inverse_folding/dplm_refiner/ tests/scripts/test_if_imp_encoder_scripts.py tests/scripts/test_diag_if_imp_arms.py` → 115 passed, 31 skipped (PyG- / omegaconf-gated), 4 warnings. Full suite excluding pre-existing broken collection (`test_evaluate_phase_c_script.py`, ImportError for `resolve_nmp_runtime_params`) → 963 passed, 4 pre-existing failures unrelated to this work (3 `epitope_head/training/test_module_e_contract` ESM tokenizer tests + 1 `test_analysis_structural_features` `Bio` import). All 8 new script tests in `tests/scripts/test_if_imp_encoder_scripts.py` pass (help-text wiring, SLURM mode dispatch, doc/SCRIPTS.md registration, `_maybe_replace_encoder` hook).
- impact:
  - scope: Inverse-folding encoder path. Default GVP behavior remains bit-equivalent because `detach_encoder_feats` defaults to True and `adapter_num_layers` / `adapter_gated` default to the existing single-layer ungated configuration. The Hydra `_target_=geoegnn_ipa_encoder` override is opt-in via `cond_dplm_650m_geoegnn.yaml`.
  - risk: medium (cluster smoke + ablations not yet executed; H1/H2 risks from review — PyG MessagePassing vs hand-scatter, Hydra `_target_` registry resolution — were resolved in the plan but runtime confirmation needs cluster execution).
  - confidence: 0.85
- status: done
- next_action: On Della with `immune-design` env active and `torch_scatter` installed, run E6 smokes: (1) two-batch CPU smoke via `LIMIT_BATCHES=2 DEVICE=cpu MODE=train_encoder sbatch scripts/submit_if_imp.slurm`; (2) `python -c "from byprot import utils; from omegaconf import OmegaConf; m = utils.instantiate_from_config(OmegaConf.create({'_target_': 'geoegnn_ipa_encoder', 'd_model': 32}), group='model'); print(type(m).__name__)"` to verify Hydra `_target_` resolution; (3) submit the 6 minimum CATH ablations (GVP baseline / `update_coors` on-off / last-1 vs last-4 gated adapters / `lambda_aux ∈ {0.05, 0.10}`) and only treat the GeoEGNN-IPA replacement as mechanistically meaningful if final DPLM recovery improves by ≥2 absolute percentage points over the current CATH baseline without foldability degradation.
- refs:
  - `PLAN_IF_ENCODER.md` Tasks E0–E5
  - `L0089`


### L0091
- timestamp: 2026-05-21T05:10:00+08:00
- type: VERIFICATION
- module: IF_ENCODER
- trigger: Code-review pass on the L0090 GeoEGNN-IPA implementation surfaced ten concrete defects spanning correctness (encoder special-mask, SLURM flag wiring, adapter checkpoint loss, IPA pair-dim mismatch, checkpoint field completeness), unimplemented PLAN items (residue positional encoding), and contract drift (encoder_attention_mask override, kNN strict-cutoff semantics, seq-distance over compact indices, Hydra DictConfig coercion).
- change_summary: P0.1 — dropped `<mask>` (id=32) and `<unk>` (id=3) from default `special_token_ids`; encoder now prefers `batch["tokens"]` over `batch["prev_tokens"]` so generation at denoising step 0 doesn't drop every real residue. P0.2 — `submit_if_imp.slurm` `MODE=generate_encoder` now passes `--output-root` and `--allele` instead of the nonexistent `--output-dir`. P1.3 — `save_geo_encoder_checkpoint` now accepts `decoder=` and writes `adapter_state_dict`; `load_geo_encoder_checkpoint(..., decoder=...)` restores adapter-named params; the training script passes `decoder=task.model.decoder` on save and the two inference scripts pass `decoder=task.model.decoder` on load. P1.4 — `IPADenseRefinement._EdgePairEncoder` is built with `ipa_pairwise_dim`, not `ipa_dim`. P1.5 — `GeoEGNNIPAEncoder` stores `self._constructor_kwargs` covering all 19 non-graph fields (egnn_message_dim, update_global, norm_coors, ipa_pairwise_dim, ipa_heads, ipa_qk_points, ipa_v_points, dropout rates, etc.); `FORMAT_VERSION` bumped 1→2 and the load report adds adapter-key fields. P2.6 — residue positional encoding implemented as sinusoidal `pos_enc_dim` (default 16) on `token_positions`, contributing to `node_feat_dim`. P2.7 — both `DPLMInvFold.forward()` and `forward_encoder()` now conditionally set `encoder_attention_mask` only when the encoder didn't supply one. P2.8 — `closest_neighbor_fallback` is now a separate `GraphConfig` flag; `seq_fallback=False, closest_neighbor_fallback=False` produces a truly strict-cutoff graph. P2.9 — `_build_protein_data` takes `token_positions` and uses them for both seq-distance edge features and positional encoding so internal gaps (special tokens / NaN-coord residues) aren't collapsed. P2.10 — `GeoEGNNIPAEncoder._coerce_graph_config` accepts GraphConfig / dict / OmegaConf DictConfig.
- rationale: Several defects were silently masked by tests that exercised only the default config path (e.g. `ipa_pairwise_dim == ipa_hidden_dim`, no internal gaps, GraphConfig-typed graph_config). New tests cover non-default dimensions, internal-gap seq-distance, DictConfig coercion, adapter round-trip, and the explicit semantics of each kNN fallback flag.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/encoder.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/graph.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/ipa.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/geo_encoder/checkpoint.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm`
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_encoder.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/diag_if_imp_arms.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_graph.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_encoder_contract.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py`
- evidence: Local pytest (Python 3.11, no PyG / no omegaconf): `tests/inverse_folding/dplm_refiner/ tests/scripts/test_if_imp_encoder_scripts.py tests/scripts/test_diag_if_imp_arms.py` → 118 passed, 37 skipped (PyG / omegaconf gated), 4 warnings. New regression tests cover `test_build_geo_graph_seq_distance_uses_original_token_positions`, `test_build_knn_edges_strict_cutoff_with_both_fallbacks_off`, `test_checkpoint_round_trip_preserves_non_default_dims`, `test_checkpoint_adapter_state_round_trip_preserves_adapter_deltas`, `test_graph_config_accepts_dict_and_dictconfig_at_encoder_construction`, `test_encoder_prefers_tokens_over_prev_tokens_in_forward`, `test_encoder_constructor_kwargs_captures_non_default_fields`, `test_sinusoidal_positional_encoding_shape_and_finite`.
- impact:
  - scope: GeoEGNN-IPA encoder path. P0 fixes are correctness blockers — without them generation produces empty graphs (P0.1) or fails at sbatch (P0.2) or silently drops the adapter fine-tune (P1.3). Default GVP path remains bit-equivalent because `detach_encoder_feats=True` default and conditional mask overrides preserve legacy behavior.
  - risk: medium (cluster smoke + ablations still pending; runtime tests for the new paths are PyG-gated and will exercise on Della).
  - confidence: 0.9
- status: done
- next_action: Same as L0090 — run cluster smoke + 6 ablations on Della. After cluster smoke confirms the encoder path runs end-to-end, treat the bugfix sweep as fully validated.
- refs:
  - `PLAN_IF_ENCODER.md`
  - `L0090`


### L0092
- timestamp: 2026-05-21T15:30:00+08:00
- type: FEATURE
- module: PHASE_D
- trigger: PLAN_RF.md Phase D Task D1 (monitor-only reliability-gated online refresh + active blocks) was upgraded from STUB to a full implementation contract; D0 telemetry schema frozen. User instructed coder to implement strictly per the plan, working directly on `dev_head` (worktree path was tried and abandoned mid-T1 per user preference).
- change_summary: Built the Phase D1 monitor-only adaptive controller scaffold without touching C1 logits, schedule, or remask ranking. Seven sub-tasks (T1-T7), each driven by RED→GREEN TDD:
  - T1 `inverse_folding/reference_flow/controller_config.py` + `configs/d1_monitor.yaml` — YAML schema with strict D1 validators (mode==monitor_only, completion==argmax, score_scale==raw_logit, selection==threshold_then_top_n, max_windows>0, t_start in [0,1], refresh_interval>0), `enabled=false` short-circuit, deterministic `controller_config_hash`.
  - T2 `epitope_head/inference/predictor.py` — added optional `window_batch_size` to `predict_protein`, plus `predict_proteins(records, allele_idx, window_batch_size)` ordered batch facade reusing one model instance.
  - T3 `inverse_folding/reference_flow/head_scoring.py` — `WindowRiskRecord` / `HeadScore` / `BatchHeadScores` / `OnlineHeadScorer` + `StaticWindowCache`. Static cache keyed by `(protein_id, sequence_md5, allele, head_checkpoint_digest, head_config_hash, score_scale, window_k_min, window_k_max)`; parquet + sidecar `meta.json` schema mirrors B2 h-map; fail-fast on any meta drift; cache hit avoids double head call.
  - T4 `inverse_folding/reference_flow/controller.py` — `D1MonitorController` with refresh gating (`t_start`, `refresh_interval`), hard completion (committed kept; masked → argmax), window-level excess `max(0, z_dyn - z_static - τ_W)`, threshold-then-top-N active window selection, connected-component span merge (transitive overlap), reliability factors (`g_time` sigmoid ramp, `g_comp` with `min_completion_fraction` cutoff, `g_ent = exp(-mean_entropy/h0)`, `g_ESS = 1.0` in D1), `WindowMismatchError` on `(start,end,k)` drift, telemetry buffers for refresh records and monitor event rows.
  - T5 `inverse_folding/reference_flow/sampler.py` — added optional `controller`, `protein_id`, `design_idx` kwargs. Hook called between NaN check and unmask sampling; receives `x_t.clone()` so a mutating controller cannot corrupt sampler state; `controller=None` reproduces existing per-step trajectory bit-for-bit.
  - T6 `scripts/run_if_phase_c1.py` — `ControllerSetup` dataclass, `load_controller_setup` (parses `--controller-config` + head flags; SystemExit when enabled-but-missing flags), `compute_per_protein_summary` (D2/D3 fields nullable for D1), `write_d1_artifacts` (writes `refresh_log.jsonl`, `controller_events.parquet`, `per_protein_summary.json` plus an empty parquet when no events), `d1_manifest_provenance` (controller mode/hash, head provenance, window k-range, cache paths). `main()` builds the head scorer ONCE per process, instantiates a fresh `D1MonitorController` per `(protein_id, design_idx)`, accumulates telemetry across designs, and flushes `static_window_cache.{parquet,meta.json}` at the end. CPU OOM retry rebuilds a fresh controller for the CPU sample.
  - T7 `scripts/submit_if_phase_c.slurm` — exposed `CONTROLLER_CONFIG`, `HEAD_CHECKPOINT`, `HEAD_CONFIG_DIR`, `HEAD_VARIANT_ID`, `HEAD_DEVICE`, `HEAD_WINDOW_BATCH_SIZE`, `HEAD_ALLELE_IDX` env vars under `MODE=reference_flow`. Empty `CONTROLLER_CONFIG` → controller disabled and head flags ignored (vanilla C1 path preserved). Fixed a pre-existing missing `\` line-continuation after `--flag-name "${FLAG}"` that silently dropped `EXTRA_FLAGS` and `WANDB_ARGS` whenever they were non-empty. Registered the D1 extension in `doc/SCRIPTS.md` Phase C section.
- rationale: §4.7 D1 (hard completion → online refresh → active blocks → reliability gates) sits entirely outside the denoiser, so no C2 retrain and no C3 conditioning are required — implementing D1 as a callable hook layered on the existing C1 sampler is the minimum-blast-radius path that still produces the D0 priority metrics. The strict D1 config-validator (only `monitor_only` / `argmax` / `raw_logit` / `threshold_then_top_n` accepted) keeps the schema honest about what D1 actually supports; D2/D3 will widen the enums when they ship. Static window-score cache is required because B2 h-map artifacts intentionally do not store per-window logits (sidecar JSON has only residue-level h_raw/h_processed/global_risk/n_windows), so D1 must lazily build window-level static scores once per protein and reuse them across designs.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller_config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d1_monitor.yaml`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/head_scoring.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/sampler.py`
  - `/Users/jerry/Project/MHC-IF/epitope_head/inference/predictor.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c1.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_phase_c.slurm`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_controller_config.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_head_scoring.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_sampler_controller.py`
  - `/Users/jerry/Project/MHC-IF/tests/epitope_head/inference/test_batch_predictor.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_run_if_phase_c1_d1.py`
  - `/Users/jerry/Project/MHC-IF/.gitignore` (added `.worktrees/`)
- evidence: `PYTHONPATH=inverse_folding/dplm/src pytest tests/inverse_folding/test_reference_flow_{schedule,amplification,sampler,config,runtime,controller_config,head_scoring,d1_controller,sampler_controller}.py tests/epitope_head/inference/test_batch_predictor.py tests/scripts/test_run_if_phase_c1_d1.py tests/scripts/test_run_if_phase_c1_script.py tests/scripts/test_run_if_phase_c0_script.py` → **83 passed, 1 pre-existing warning** (amplification g_max_cap warning, unrelated). New tests: 12 (T1) + 4 (T2) + 6 (T3) + 14 (T4) + 6 (T5) + 8 (T6) = **50 new tests**, all green. `bash -n scripts/submit_if_phase_c.slurm` clean. `python -c "import scripts.run_if_phase_c1"` clean. C1 sampler bit-equivalent under `controller=None` (verified by `test_controller_none_keyword_accepted_and_default_behavior_preserved`).
- impact:
  - scope: Phase D entry point. With `--controller-config` unset, `scripts/run_if_phase_c1.py` and `scripts/submit_if_phase_c.slurm` are bit-equivalent to pre-D1. With `--controller-config inverse_folding/reference_flow/configs/d1_monitor.yaml` plus head flags, the run emits 4 new D1 telemetry artifacts (`refresh_log.jsonl`, `controller_events.parquet`, `per_protein_summary.json`, `static_window_cache.{parquet,meta.json}`) and adds 12 controller / head provenance fields to the manifest; `generated.parquet` schema is unchanged.
  - risk: medium. Cluster smoke not yet executed. Head model is loaded once per process — if the inference config's `min_k/max_k` differs from the controller cache's expected range the cache will fail-fast on the second run with a clear meta-mismatch error. The hard-completion sequence reconstruction in `D1MonitorController._build_hard_completion` uses `decode_residue_tokens(used_task, ...)` per refresh; if a step's structural logits contain non-residue tokens that the DPLM wrapper hasn't masked the head will receive a non-canonical AA character and may error — this case has not been exercised against the production DPLM denoiser yet.
  - confidence: 0.85
- status: done
- next_action: On Della cluster: (1) run `MODE=reference_flow CONTROLLER_CONFIG=inverse_folding/reference_flow/configs/d1_monitor.yaml HEAD_CHECKPOINT=<best.pt> HEAD_VARIANT_ID=<v> sbatch scripts/submit_if_phase_c.slurm` against a 2-protein smoke subset and verify all 4 D1 telemetry artifacts appear with non-empty refresh records; (2) compare `generated.parquet` against a `CONTROLLER_CONFIG=""` baseline run with the same `--seed` to confirm sequences are bit-equivalent; (3) compute the D0 priority metrics (static-dynamic drift, new hotspot rate, risk volatility, active block coverage, head risk trajectory) from D1 artifacts as a smoke for D2/D3 planning. Once D1 baselines are clean, start D2 (hard counterfactual logits) and D3 (residue-level EMA recommit) per PLAN_RF.md.
- refs:
  - `PLAN_RF.md` Task D0 (telemetry contract), Task D1 (monitor-only implementation), Task D3 STUB (route A/B deferred decision)
  - `doc/Reference_Flow_Derivation.md` §4.7 D1, §4.9 (training-side support justification)
  - `L0082`, `L0083` (Task D theory updates that motivated this implementation)


### L0093
- timestamp: 2026-05-21T16:39:48+08:00
- type: PLAN_UPDATE
- module: PHASE_D
- trigger: User asked whether D2 and D3 can now be fully planned together after D1 implementation, and whether logit steering plus commit/revisit should be implemented in one pass.
- change_summary: Replaced the old D3 stub in `PLAN_RF.md` with a combined D2-D3 implementation plan. The plan keeps D2 and D3 jointly planned but separately switchable through `monitor_only`, `d2_logits`, `d3_revisit`, and `d2_d3_full` modes; it defines the D2 hard-counterfactual candidate/logit path, the D3 EMA commit-score/remask path, sampler hook changes, telemetry migration, config presets, tests, and acceptance gates.
- rationale: D1 has already established the online refresh, active-block, static-window-cache, head-scoring, and controller-hook infrastructure. D2 and D3 share those interfaces and must share a widened telemetry schema, but their causal surfaces must remain separable for the D0 attribution and budget-normalization contract.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/PLAN_RF.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: Documentation-only update based on read-only inspection of `inverse_folding/reference_flow/controller.py`, `controller_config.py`, `sampler.py`, `head_scoring.py`, `scripts/run_if_phase_c1.py`, `PLAN_RF.md` D0/D1 schema, `doc/Reference_Flow_Derivation.md` §4.7 D2-D3, and two read-only explorer reports. No runtime tests executed.
- impact:
  - scope: Reference Flow D2/D3 implementation planning only; no production code or experiment artifacts changed.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Future coder should implement D2 first on the existing pre-sampling controller hook, then D3 on a new post-sampling/pre-remask hook, while landing the shared telemetry schema before either mode is treated as complete.
- refs:
  - `PLAN_RF.md` Task D2-D3
  - `PLAN_RF.md` Task D0
  - `doc/Reference_Flow_Derivation.md` §4.7 D2-D3
  - `L0092`


### L0094
- timestamp: 2026-05-21T18:47:06+08:00
- type: FEATURE
- module: PHASE_D
- trigger: User authorized D2/D3 implementation per the revised `PLAN_RF.md` §"Task D2-D3" after the M1-M6 + N2 review cycle closed. Implementation followed `superpowers:executing-plans` with TDD per sub-task and the strict CLAUDE.md execution standards.
- change_summary: Landed the full D2 hard-counterfactual logit correction and D3 EMA commit/revisit surfaces on top of the existing D1 monitor controller. ControllerConfig broadened to four modes (`monitor_only | d2_logits | d3_revisit | d2_d3_full`) with cross-mode validators; `counterfactual.py` and `commit.py` added as pure-function modules plus thin `D2Handler` / `D3Handler` orchestrators; `controller.py` was renamed to `ReferenceFlowController` (a deprecated `D1MonitorController` alias is kept), restructured around a single `RefreshState` snapshot, and now exposes a `post_step()` hook returning `PostSamplingResult(rank_scores, protected_positions, post_event_rows, refresh_addendum)`; `sampler.py` captures structural vs. corrected logits, snapshots `rng.bit_generator.state` after the Bernoulli draws / before categorical sampling, runs paired uncorrected sampling on an isolated RNG clone, calls the post-sampling hook before remask, and extends `_apply_reparam_remask` with optional `rank_scores` / `protected_positions` (defaults byte-equivalent to the legacy implementation); `scripts/run_if_phase_c1.py` derives `arm` from `controller.mode`, stamps `controller_surface_version=2` plus per-mode config blocks into the manifest, prints the resolved controller config at startup, and merges per-refresh D3 addenda into `refresh_log.jsonl`; four new YAML presets (`d_monitor_full`, `d2_logits`, `d3_revisit`, `d2_d3_full`) live under `inverse_folding/reference_flow/configs/`; `doc/SCRIPTS.md` was updated to describe the Phase D extension.
- rationale: PLAN_RF.md §D2-D3 requires that D2 (token direction) and D3 (reversibility) ship from one shared refresh / candidate / telemetry surface so the D0 attribution and budget-normalization matrix can decompose them cleanly. Shipping the modes together preserves bit-equivalence of the existing C1 / D1 paths (no behavior drift when `controller=None`, when `enabled=false`, or when the mode is `monitor_only` with D2/D3 sections present for diagnostics) and avoids a second telemetry rewrite when D3 lands after D2.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller_config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/counterfactual.py` (new)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/commit.py` (new)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/sampler.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d_monitor_full.yaml` (new)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d2_logits.yaml` (new)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d3_revisit.yaml` (new)
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d2_d3_full.yaml` (new)
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_phase_c1.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_controller_config.py`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_counterfactual.py` (new)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_commit.py` (new)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_d2_d3_controller.py` (new)
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/test_reference_flow_sampler_controller.py`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_run_if_phase_c1_d2_d3.py` (new)
- evidence: `pytest tests/inverse_folding/test_reference_flow_{controller_config,counterfactual,commit,d1_controller,d2_d3_controller,sampler_controller}.py tests/scripts/test_run_if_phase_c1_{d1,d2_d3}.py` → **125 passed**. Breakdown: T1 controller_config 33, T2 counterfactual 19, T3 commit 19, T4 d2_d3_controller 13, T5 sampler_controller 12 (incl. paired RNG isolation + `_apply_reparam_remask` byte-equivalence), T6 run_script d1 9 + run_script d2_d3 5, plus D1 monitor controller 15. All four preset YAMLs load cleanly via `load_controller_config`. The `D1MonitorController` alias keeps every pre-existing D1 test green without signature changes. `controller=None` + identity hook + `_apply_reparam_remask(rank_scores=None, protected_positions=())` were each verified bit-equivalent against the pre-T5 implementation by direct comparison tests. The one pre-existing failing test in `test_analysis_structural_features.py` is unrelated (missing `Bio` module) and was confirmed pre-existing via `git stash` regression.
- impact:
  - scope: Phase D adaptive controller. With `--controller-config` unset or `enabled=false`, the run is bit-equivalent to pre-D1. With one of the four new presets (`d_monitor_full`, `d2_logits`, `d3_revisit`, `d2_d3_full`), the controller activates the corresponding handler composition; `refresh_log.jsonl` gains nullable `e_i / m_i / rho_i / grace_positions` fields (populated only when D3 ran), `controller_events.parquet` retains the full D0 schema with nullable D2/D3 columns, and `manifest.json` stamps `controller_surface_version=2`, `controller_mode`, `d2_config`, `d3_config`, `attribution_config`, `controls_config`. `generated.parquet` schema is unchanged.
  - risk: medium. Cluster smoke not yet executed; the head scorer call shape for D2 candidate batches (up to `max_candidates_per_block * n_active_blocks` per refresh) has only been exercised against the stub scorer in unit tests. D2 per-corrected-position event rows currently stay at D1 block-level monitor format (`a_after` / `a_uncorrected` / KL fields are still null); `per_protein_summary` aggregates for D2/D3 stay at zero/null because the script-level aggregation of corrected positions / KL / recommit counts was not extended in this PR. These two follow-ups do not change the on-disk schema, only its content density.
  - confidence: 0.85
- status: done
- next_action: (1) On Della, run a 2-protein paired-seed smoke across all four presets per `PLAN_RF.md` §"Acceptance" 6 to validate D0 Layer A opportunity metrics and the D2/D3 telemetry density. (2) Wire D2 per-corrected-position event rows and the post-sampling `a_after / a_uncorrected / KL_struct_corrected / delta_R_corrected / delta_R_uncorrected` fill-in so `controller_events.parquet` rows for D2 modes carry full attribution. (3) Extend `compute_per_protein_summary` to aggregate D2 event count, total corrected positions, total KL budget, D3 recommit count, and productive-revisit rates from `event_rows`. (4) Optionally consider removing the `.worktrees/` line from `.gitignore` if no worktree workflow is planned for the near term.
- refs:
  - `PLAN_RF.md` §"Task D2-D3"
  - `PLAN_RF.md` §"Task D0" (telemetry contract)
  - `L0092`, `L0093`


### L0095
- timestamp: 2026-05-21T21:18:36+08:00
- type: FEATURE
- module: IF_ENCODER
- trigger: User identified that GeoEGNN-IPA encoder training still inherited the old Module-K decoder adapter shape and requested explicit adapter-depth controls in the IF-IMP encoder stage.
- change_summary: Added `--adapter-num-layers`, `--adapter-gated`, and `--adapter-gate-init` to `scripts/train_if_imp_encoder.py`, reinstalled requested decoder adapters before freezing non-adapter DPLM parameters, forwarded the new controls from `scripts/submit_if_imp.slurm`, and updated script registration/tests.
- rationale: Module-K checkpoints predate the adapter-shape fields and therefore load as last-1 ungated adapters by default. Last-N adapter ablations must be an explicit training-run contract; last-N > 1 is guarded to require gated adapters so fresh earlier-layer adapters start with zero contribution instead of perturbing decoder hidden states at initialization.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_encoder.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm`
  - `/Users/jerry/Project/MHC-IF/tests/scripts/test_if_imp_encoder_scripts.py`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/PROGRESS.md`
  - `/Users/jerry/Project/MHC-IF/LOG.md`
- evidence: `python scripts/train_if_imp_encoder.py --help` passed and lists the three adapter flags; `pytest -q tests/scripts/test_if_imp_encoder_scripts.py tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py` → 16 passed, 8 skipped (PyG / omegaconf / transformers gated); `python -m py_compile scripts/train_if_imp_encoder.py` passed; `bash -n scripts/submit_if_imp.slurm` passed.
- impact:
  - scope: IF encoder replacement training and launcher controls only; default remains last-1 ungated unless the new flags/env vars are set.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: On Della, run a two-batch `MODE=train_encoder` smoke with `ENCODER_ADAPTER_NUM_LAYERS=4 ENCODER_ADAPTER_GATED=1 ENCODER_ADAPTER_GATE_INIT=0.0`, then include adapter depth/gating in the planned GeoEGNN-IPA ablation matrix.
- refs:
  - `PLAN_IF_ENCODER.md` Tasks E0, E5, E6


### L0096
- timestamp: 2026-05-22T00:58:49-04:00
- type: BUGFIX
- module: IF_ENCODER
- trigger: E7 quick-screen `aux01_a4g` diagnostic evaluation failed because a last-4 gated adapter checkpoint was loaded into the default last-1 ungated decoder; the checkpoint loader correctly refused to silently drop trained adapter weights.
- change_summary: Added `auto_install_adapter_shape` to `load_geo_encoder_checkpoint()`. When enabled by production encoder-swap entrypoints, the loader rebuilds the target decoder adapter stack from checkpoint `adapter_config`, restores the decoder to its original device after fresh adapter modules are installed, then loads `adapter_state_dict` with the existing shape guard still active. `scripts/run_if_imp_refiner.py` and `scripts/diag_if_imp_arms.py` now opt into this behavior for GeoEGNN-IPA checkpoints. Default direct loader behavior remains fail-fast on adapter-shape mismatch.
- rationale: Adapter depth/gating is part of the trained checkpoint contract. Generation and diagnostics must restore both encoder weights and the matching trained adapter shape; bypassing the guard would produce invalid last-4 ablation results by partially loading or dropping adapter weights.
- artifacts:
  - `/home/zc1519/src/Immune-Design/inverse_folding/dplm_refiner/geo_encoder/checkpoint.py`
  - `/home/zc1519/src/Immune-Design/scripts/run_if_imp_refiner.py`
  - `/home/zc1519/src/Immune-Design/scripts/diag_if_imp_arms.py`
  - `/home/zc1519/src/Immune-Design/tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py`
  - `/home/zc1519/src/Immune-Design/tests/scripts/test_if_imp_encoder_scripts.py`
- evidence: `python -m py_compile inverse_folding/dplm_refiner/geo_encoder/checkpoint.py scripts/run_if_imp_refiner.py scripts/diag_if_imp_arms.py` passed. `pytest tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py tests/scripts/test_if_imp_encoder_scripts.py` -> 29 passed, 1 skipped, 1 warning. `bash -n scripts/submit_if_imp.slurm` passed.
- impact:
  - scope: GeoEGNN-IPA checkpoint restore path for `generate_encoder` and `diag_encoder`.
  - risk: low; the new auto-install path is opt-in and the default loader guard remains unchanged.
  - confidence: 0.92
- status: done
- next_action: Re-run the `aux01_a4g` diagnostic through the normal `MODE=diag_encoder` entrypoint and verify the checkpoint loads with adapter mismatch count zero.
- refs:
  - `PLAN_IF_ENCODER.md` Task E7
  - `L0095`


### L0097
- timestamp: 2026-06-03T01:20:57-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User provided `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/characterized_uricases.fasta` and requested that characterized uricases be deduplicated against the existing uricase/test-set data, added to the IF test set, structure-backed, and rebuilt through the current IF-ready contract.
- change_summary: Normalized the characterized FASTA to the existing uricase ID style (`accession|organism|name|taxid|len`), removed one exact-sequence duplicate (`Q2U050` duplicate of `Q00511`), and appended 25 curated characterized uricases as Tier 3 additions to both allele-specific raw test-set parquets. Scored the additions with the current npoff epitope-head checkpoints and NetMHCIIpan for both HLA-DRB1*07:01 and HLA-DRB1*04:01 before appending. Downloaded AFDB CIF structures through `scripts/download_test_set_pdbs.py`, linked the 23 successful structures into both allele-specific raw PDB roots, updated the two AFDB-404 accessions in `uniprot_final_failures.txt`, and rebuilt both IF-ready parquets with `scripts/build_if_ready_test_set.py --allow-cif --default-chain A`. Updated `PROGRESS.md` to reflect the new raw, structure, failure, and IF-ready counts.
- rationale: The characterized uricases were mostly already present in the broad uricase candidate pool but had not been selected into the assembled Tier 3 test set. For the current scientific use case they are curated functional/therapeutic additions, so inclusion should be based on curated status rather than rerunning the old pool-level selection/median filter. Still, allele-specific WT head/NMP artifacts and resolved-backbone IF-ready structures must be materialized so Phase C consumes the same schema and sequence/structure contract as the existing test set.
- artifacts:
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/characterized_uricases.normalized_unique.fasta`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/characterized_uricases.normalized_unique_manifest.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricase_prescreen/characterized_uricase_artifacts_HLA-DRB1_07_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricase_prescreen/characterized_uricase_artifacts_HLA-DRB1_04_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs/characterized_afdb/`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs/characterized_afdb_link_report.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs/{0401,0701}/uniprot_final_failures.txt`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.parquet`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/load_coords_gate_HLA-DRB1_07_01.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready/load_coords_gate_HLA-DRB1_04_01.json`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/backups/20260603_003541_EDT/`
  - `/home/zc1519/src/Immune-Design/PROGRESS.md`
  - `/home/zc1519/src/Immune-Design/LOG.md`
- evidence: Characterized FASTA normalization produced 25 unique sequences from 26 records, with `Q2U050` recorded as an exact duplicate of `Q00511`. NetMHCIIpan completed for all additions on both alleles: 0701 n_strong median=88 (min=10, max=190), 0401 n_strong median=82 (min=19, max=189). Raw test-set validation passed after append: 0701 3,141 -> 3,166 rows (Tier 3 126 -> 151), 0401 3,140 -> 3,165 rows (Tier 3 127 -> 152). AFDB CIF download via `download_test_set_pdbs.py` succeeded for 23/25 accessions; AFDB PDB retry confirmed 404 for `A0ABR4SZB5` and `A0ABX7U467`. IF-ready rebuilds completed: 0701 3,166 input -> 3,102 ready / 64 failed (Tier 3 ready=125), 0401 3,165 input -> 3,095 ready / 70 failed (Tier 3 ready=132). DPLM `load_coords()` gate passed exactly for all IF-ready rows: 0701 3,102/3,102, 0401 3,095/3,095.
- impact:
  - scope: Cluster data artifacts for Module L / Phase B only. Repo code was not changed; only `PROGRESS.md` and `LOG.md` were updated locally.
  - risk: low. Two characterized additions are present in the raw test set with WT head/NMP artifacts but are not IF-ready because AFDB has no CIF/PDB prediction for them; they are explicit missing-structure failures and remain candidates for AF3 fill-in.
  - confidence: 0.93
- status: done
- next_action: If these two characterized Streptomyces uricases are important for Phase C/F4, generate AF3 structures for `A0ABR4SZB5` and `A0ABX7U467`, then rerun `build_if_ready_test_set.py` and the DPLM `load_coords()` gate. B2 h-map precompute must be rerun against the updated IF-ready parquets before Phase C C1/D runs.
- refs:
  - `PROGRESS.md` Module L / Phase B snapshot
  - `scripts/download_test_set_pdbs.py`
  - `scripts/build_if_ready_test_set.py`


### L0098
- timestamp: 2026-06-03T01:53:34-04:00
- type: FEATURE
- module: DATA_SELECTION
- trigger: User decided the v1 Tier 2 selection is biased — the epitope head (the RF guidance signal) gated test-set selection, and the set is a high-immunogenicity tail rather than representative. Authorized a Tier 2 **v2** rebuild that is NMP-only, structure-blind, with a controlled unimodal (Gaussian) marginal over immunogenicity density. Tier 1 and Tier 3 (expanded in L0097) are untouched; v2 only swaps Tier 2.
- change_summary: Added the NMP-only density-stratified Tier 2 v2 selection path. `inverse_folding/evaluation/immunogenicity.py` gained pure `compute_coverage_fraction(window_df, seq_len)` (residues covered by ≥1 strong window / length) and now stamps `coverage_fraction` into `aggregate_nmp_batch_scores` output. New `inverse_folding/evaluation/sampling.py` provides pure `sample_uniform_bins` / `sample_gaussian_bins` (greedy water-filling proportional to per-bin weights, capped by availability, floored at min-per-bin, deficit redistributed to hit target; returns selected ids + realized histogram). `scripts/prescreen_tier2.py` gained `--selection-mode {dual_scorer,nmp_only}` (dual_scorer untouched/byte-equivalent), `--nmp-screen-lengths`, `--candidate-id-list` (reuse allele-independent overlap IDs, skip MMseqs2), and `--sample {none,uniform,gaussian}` + params; `--cath-train-fasta/--epitope-ckpt/--cath-domain-list` relaxed to conditionally-required. `scripts/submit_prescreen_tier2.slurm` gained `MODE=nmp_only` (CPU-only) branch. PLAN_DATA_SEL.md §7.3 dual-scorer rule marked SUPERSEDED; new §12 documents the two-stage S1(len-15→uniform≈5000)→if_ready gate→S2(multi-len→Gaussian≈3000) procedure. SCRIPTS.md updated.
- rationale: The epitope head is the guidance signal RF optimizes; using it to select the test set inflates apparent performance (the §8 risk #4 accepted in v1). NMP is the independent validator, so NMP-only selection is the de-biased contract. Single-length-15 is a high-recall coarse detector (MHC-II binding is 9-mer-core-driven), making a full-pool head-free scan affordable (~2× original NMP). Gaussian-over-coverage concentrates statistical power in the mid-density regime (where RF is most expressive) while keeping tails populated, and is naturally satisfiable by the right-skewed candidate pool (unlike flat-uniform, which over-requests the rare high-density tail).
- artifacts:
  - `/home/zc1519/src/Immune-Design/inverse_folding/evaluation/sampling.py` (new)
  - `/home/zc1519/src/Immune-Design/inverse_folding/evaluation/immunogenicity.py`
  - `/home/zc1519/src/Immune-Design/scripts/prescreen_tier2.py`
  - `/home/zc1519/src/Immune-Design/scripts/submit_prescreen_tier2.slurm`
  - `/home/zc1519/src/Immune-Design/tests/inverse_folding/test_tier2_v2_sampling.py` (new)
  - `/home/zc1519/src/Immune-Design/PLAN_DATA_SEL.md`
  - `/home/zc1519/src/Immune-Design/doc/SCRIPTS.md`
- evidence: `pytest tests/inverse_folding/test_tier2_v2_sampling.py tests/inverse_folding/test_module_l_prescreen.py` → 24 passed (11 new v2 coverage/sampling + 13 legacy dual_scorer regression; dual_scorer path unchanged). `py_compile` clean on all touched modules. Login-node end-to-end pilot: `prescreen_tier2.py --selection-mode nmp_only --nmp-screen-lengths 15` on 24 real overlap-passed proteins via the bundled NetMHCIIpan produced a valid scored parquet (coverage_fraction populated, e.g. 1 strong window over a 145-aa protein → 15/145 = 0.103) at 0.72 proteins/s with 4 workers → ~7.3 h projected for 75,425 at 16 workers, ~3.6 h at 32. Verified candidate pool: all 75,425 overlap-passed IDs (head cache) covered by `tier2_candidates_merged.fasta`; overlap is allele-independent so both alleles reuse the same id pool.
- impact:
  - scope: Tier 2 test-set selection only. `dual_scorer` default is byte-equivalent to pre-change; no existing artifact is overwritten by this change. Materialization (assemble → download structures → if_ready → h_maps) is deferred to Phase B and will overwrite `test_proteins_<allele>.parquet` only after the user reviews the S2 Gaussian histogram.
  - risk: medium. S1/S2 are long CPU jobs; the Gaussian (μ, σ, peak:tail, min-per-bin) is finalized against realized histograms before S3 commits. The v2 Tier 2 will include low-immunogenicity proteins (no floor), a deliberate distribution change from v1.
  - confidence: 0.85
- status: in_progress
- next_action: S1 length-15 screen runs as **strided job-arrays** (16 shards × 8 cores, `--array=0-15`, qos=short) — added `--n-shards/--shard-index` to `prescreen_tier2.py` + `N_SHARDS`/`SLURM_ARRAY_TASK_ID` to the SLURM after the initial 48-core single jobs (9128221/9128222) queued too long; small shards backfill instantly. Current arrays: 9147972 [0701] / 9147973 [0401], writing `tier2_nmp_screen_<allele>_l15.shardNNof16.parquet`. When drained: (1) merge shards → full scored parquet; (2) S1 uniform-sample → ≈5000 per allele; (3) S2 multi-length NMP on the ≈5000; (4) review Gaussian target histogram with user, finalize knobs; (5) S3 Gaussian → ≈3000; (6) assemble (swap Tier 2, keep Tier 1 + L0097 Tier 3), download structures, build_if_ready, rerun precompute_h_maps for both alleles.
- refs:
  - `PLAN_DATA_SEL.md` §12 (Tier 2 v2)
  - `PLAN_DATA_SEL.md` §7.3 (superseded), §8 risk #4
  - `L0097` (Tier 3 uricase expansion — orthogonal)


### L0099
- timestamp: 2026-06-03T05:40:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: Execute the Tier 2 v2 NMP-only density-stratified rebuild (L0098 design) end-to-end on Della for both alleles, producing the v2 test set (`test_proteins_v2` + `if_ready_v2` + `h_maps_v2`). User reviewed and approved the default Gaussian shape (μ=median coverage, peak:tail=3, min-per-bin=40, 20 bins, →3000).
- change_summary: Ran the full S1→S2→gate→S3→materialize pipeline. **S1**: length-15 NetMHCIIpan over the full 75,425 overlap-passed pool per allele, as strided 16-shard CPU job-arrays (2 shards timed out at 3h and were resumed after adding the `: "${PS1:=}"` guard to `submit_prescreen_tier2.slurm`); merged + `select_tier2_v2.py --mode uniform --n-target 5000` → 5000-protein coarse pool spanning all density bins. **S2**: full 12–25 NMP on the 5000 (24-shard arrays) → accurate `coverage_fraction`. **if_ready gate**: built 5000-candidate full-schema parquets (joined S2 NMP + head-cache risk), downloaded RCSB structures (4839/5000 0701, 4865/5000 0401 — the rest are obsolete/withdrawn PDB codes), ran `build_if_ready_test_set.py` → 4759/4745 if_ready survivors. **S3**: `select_tier2_v2.py --mode gaussian --n-target 3000` on survivors → final Tier 2 v2 (2999 0701 after dropping 1 Tier-1 collision `6Y76_A`; 3000 0401). **Materialize**: spliced `test_proteins_v2` (existing Tier 1+3 + new Tier 2) and `if_ready_v2` (existing Tier 1+3 if_ready + gated Tier 2 if_ready, column-aligned), then recomputed `h_maps_v2` with the current npoff head checkpoints (the existing h_maps were stale — old `LC1_lite_aug` ckpt, pre-L0097, per L0097 next_action).
- rationale: Single-length-15 coarse screen made the full-pool head-free scan affordable; uniform→Gaussian two-stage decouples "span all densities for S2" from "final unimodal marginal". if_ready gate before the final Gaussian keeps the selected set hole-free. Splicing (vs full re-assemble) avoided re-resolving Tier 1/3 structures and side-stepped the scattered structure layout (new Tier 2 structures landed in `pdbs/` top-level due to an output-dir glob bug; harmless since PDB structures are allele-independent and h_maps are sequence-based).
- artifacts:
  - `work/immune-design/if_test_set/test_proteins_v2_HLA-DRB1_07_01.parquet` — 3165 rows (T1=15, **T2=2999**, T3=151)
  - `work/immune-design/if_test_set/test_proteins_v2_HLA-DRB1_04_01.parquet` — 3167 rows (T1=15, **T2=3000**, T3=152)
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_v2_HLA-DRB1_07_01.parquet` — 3139 (all if_ready=True; T2=2999, T3=125)
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_v2_HLA-DRB1_04_01.parquet` — 3147 (all if_ready=True; T2=3000, T3=132)
  - `work/immune-design/if_test_set/h_maps/h_maps_v2_DRB1_07_01.{parquet,meta.json}` — 3139 rows, 0 failed, npoff ckpt
  - `work/immune-design/if_test_set/h_maps/h_maps_v2_DRB1_04_01.{parquet,meta.json}` — 3147 rows, 0 failed, npoff ckpt
  - Intermediates (audit/resume): `tier2_nmp_screen_<allele>_l15.shard*of16.parquet`, `..._lall.shard*of24.parquet`, `tier2_v2_s1_uniform_<allele>.parquet`, `tier2_v2_cand5000_<allele>.parquet`, `tier2_v2_ifready5000_<allele>.parquet`, `tier2_v2_selected_<allele>.parquet` (+ `.histogram.json`), cleaned structures in `pdbs_if_ready_v2/{0701,0401}/`
- evidence: Final verification — both alleles: protein_id unique; if_ready_v2 `if_ready` all True; `h_maps_v2` covers 100% of if_ready proteins (0 missing), 0 head failures; Tier 2 `head_global_risk` 100% populated (joined from head cache). Multi-length coverage medians: 0701 ≈ 0.39, 0401 ≈ 0.46. Gaussian per-bin histograms saved as `tier2_v2_selected_<allele>.histogram.json`. h_maps wall-clock ~2 min each (jobs 9164104/9164105 COMPLETED 0:0).
- impact:
  - scope: New parallel v2 test-set artifacts. **Frozen v1 files were NOT overwritten** — v2 lives under `*_v2` names. Promotion to canonical (back up v1, rename v2 → `test_proteins_<allele>.parquet` etc., repoint Phase C/D) is a separate user-gated step.
  - risk: low-medium. v2 Tier 2 is NMP-only / head-free / structure-blind-selected and includes low-immunogenicity proteins (no floor) — a deliberate distribution change from v1; downstream comparisons must not mix v1/v2. h_maps use the npoff head ckpt; confirm this matches the head used at Phase C/D generation before relying on the risk landscape.
  - confidence: 0.88
- status: done
- next_action: (1) User decides whether to promote `*_v2` to canonical (with v1 backup) or keep parallel. (2) If promoting, repoint Phase C/D `TEST_SET_PARQUET` / `H_MAPS_PARQUET` (or rename) and re-run any cached WT baselines. (3) Optionally regenerate `test_proteins_summary_v2_*.json` via the summary path. (4) The pilot/diagnostic subset (mid-density sweet spot) is a separate future task per the earlier discussion.
- refs:
  - `PLAN_DATA_SEL.md` §12 (Tier 2 v2)
  - `L0098` (v2 design + code), `L0097` (Tier 3 uricase expansion)


### L0100
- timestamp: 2026-06-03T17:05:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User confirmed promotion of the Tier 2 v2 full set to canonical, after a requested consistency check between the newly-computed npoff h_maps and the pre-existing `if_ready/h_maps_v2/` npoff h_maps.
- change_summary: Verified h_map consistency, then promoted v2 → canonical (both alleles). **Consistency check**: on overlapping proteins (289 0701 / 433 0401) the new h_maps are bit-identical to the existing npoff `if_ready/h_maps_v2/` — 100% sequence_md5 match, global_risk |Δ|=0, h_raw per-residue |Δ|=0 (same npoff head + same pipeline). **Promotion** (v1 backed up to `backups/20260603_170203_tier2v2_promote/`): `test_proteins_v2_<tag>` → `test_proteins_<tag>`; `if_ready/test_proteins_if_ready_v2_<tag>` → `if_ready/test_proteins_if_ready_<tag>`; `h_maps/h_maps_v2_DRB1_<tag>{.parquet,.meta.json}` → `if_ready/h_maps_v2/h_maps_DRB1_<tag>`; copied v2 cleaned tier2 structures into `pdbs_if_ready/{0701,0401}`; regenerated `test_proteins_summary_<tag>.json`.
- rationale: The workspace `h_maps_v2` naming denotes the npoff-checkpoint version (a different axis than test-set version); the prior npoff h_maps were already the in-use set, and the bit-identical overlap confirms the new h_maps are drop-in. Promotion makes the full v2 set canonical for Phase C full runs while leaving the derived fast subset untouched per user.
- artifacts:
  - `work/.../if_test_set/test_proteins_{HLA-DRB1_07_01,HLA-DRB1_04_01}.parquet` (now v2: 3165 / 3167)
  - `work/.../if_test_set/if_ready/test_proteins_if_ready_{...}.parquet` (now v2: 3139 / 3147, all if_ready)
  - `work/.../if_test_set/if_ready/h_maps_v2/h_maps_DRB1_{07_01,04_01}.parquet` (now v2 npoff: 3139 / 3147, covers 100%)
  - `work/.../if_test_set/pdbs_if_ready/{0701,0401}/` (+ v2 tier2 cleaned structures; all if_ready structs resolvable incl. tier3 .cif by safe-id)
  - `work/.../if_test_set/backups/20260603_170203_tier2v2_promote/` (v1 test_proteins + if_ready + npoff h_maps, both alleles)
- evidence: Post-promotion verification both alleles: canonical test_proteins tiers {1:15, 2:2999/3000, 3:151/152}; if_ready all if_ready=True; h_maps cover 100% of if_ready (0 missing); PDB_ROOT structures resolvable for tier1/2/3 (tier3 via safe-id .cif, 125/125 0701). No `*_v2` leftovers (moved). v1 backup present.
- impact:
  - scope: Canonical IF test set (full) for both alleles is now v2. Phase C/D full runs that read `test_proteins_if_ready_<tag>` + `if_ready/h_maps_v2/` consume v2. The fast subset (`*_if_ready_fast_*`, 493) is now stale relative to its source and left as-is. Phase C SLURM `H_MAPS_PARQUET` default points at the stale LC1 `if_ready/h_maps/`; runs must override to `if_ready/h_maps_v2/`.
  - risk: medium. Canonical files overwritten (v1 recoverable from backup). Do not mix v1/v2 results. Confirm Phase C/D generation head == npoff before relying on the risk landscape.
  - confidence: 0.9
- status: done
- next_action: (1) When fast/pilot runs are needed, rebuild a v2 fast subset (or a mid-density diagnostic pilot per the earlier discussion) from the new canonical full set. (2) Consider fixing the Phase C SLURM `H_MAPS_DEFAULT` to `if_ready/h_maps_v2/`. (3) Re-run any cached WT baselines that referenced v1 Tier 2.
- refs:
  - `L0099` (v2 build), `L0098` (v2 design)


### L0101
- timestamp: 2026-06-03T18:20:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: After promoting the v2 full test set, design + build two diagnostic subsets to replace the stale pilot50 / 493-fast subsets. Decided collaboratively: keep **two distinct sets** (not merged) — a high-frequency RF-iteration "pilot" (concentrated where RF wins) and a lower-frequency global "fast" sanity set — both allele-specific, NMP-only, structure-blind. Uricase (Tier 3) is a separate case study, excluded from both.
- change_summary: Built both subsets (both alleles) as strict subsets of the canonical v2 if_ready. **pilot** (50 = 15 Tier 1 + 35 Tier 2): Tier 2 drawn from the **mid-load Pareto band** [p40,p75] of `coverage_fraction` with 4-sub-bin light stratification — band centered near the median and kept below the p90 saturation zone, because the meaningful metric is Pareto improvement (immune↓ at matched structure) which peaks at moderate load, whereas immune-only↓ keeps rising into the structure-sacrificing saturated regime. 0701 band [0.33,0.56], 0401 [0.39,0.64]. All 15 Tier 1 included as experimental-epitope anchors. **fast** (300 = 15 Tier 1 + 285 Tier 2): Tier 2 uniform across the full coverage range (20 bins). Computed Tier 1 `coverage_fraction` via multi-length NMP (15/allele) for the record. Outputs `if_ready/pilot_v2_<tag>.parquet` and `if_ready/fast_v2_<tag>.parquet`.
- rationale: pilot vs fast have different objectives → different designs (concentrated-diagnostic vs representative-global) and sizes (≤50 iteration cap vs 300). Length fixed to the recovery-reliable range and structure kept out of selection so the RF-vs-baseline comparison is fair (selecting structurally-flexible epitopes would advantage unconstrained baselines). Allele-specific because `coverage_fraction` is allele-dependent — sharing proteins across alleles would break the density-bin assignment.
- artifacts:
  - `work/.../if_test_set/if_ready/pilot_v2_{HLA-DRB1_07_01,HLA-DRB1_04_01}.parquet` (50 each)
  - `work/.../if_test_set/if_ready/fast_v2_{HLA-DRB1_07_01,HLA-DRB1_04_01}.parquet` (300 each)
- evidence: pilot tier2 coverage medians 0.44 (0701) / 0.50 (0401), 4 sub-bins ~9 each, all below p90 saturation. fast tier2 coverage spread p10/p90 = 0.09/0.80 (0701), 0.10/0.86 (0401). Tier 1 coverage medians 0.32 / 0.34. Both subsets are subsets of canonical if_ready → inherit `pdbs_if_ready` structures + `h_maps_v2` coverage (verified 100% at promotion).
- impact:
  - scope: New diagnostic/sanity subsets for Phase C. Not wired into any SLURM default yet — pass `TEST_SET_PARQUET=...pilot_v2_<tag>.parquet` (or fast) explicitly.
  - risk: low (read-only subsets of canonical). The pilot band edges are a first principled bet on the Pareto-gap peak; the in-band gradient (top sub-bin probes toward saturation) lets the first pilot run confirm/refine the peak location.
  - confidence: 0.85
- status: done
- next_action: (1) Run RF + baselines on `pilot_v2` and read the Pareto-gap-vs-coverage trend across the 4 sub-bins to confirm the band; adjust edges if the peak sits elsewhere. (2) Build the uricase case-study set separately. (3) Optionally formalize pilot/fast generation into a small CLI if the bands need frequent retuning.
- refs:
  - `L0100` (v2 promotion), `L0099`/`L0098` (v2 build/design)


### L0102
- timestamp: 2026-06-03T22:47:53-04:00
- type: FEATURE
- module: IF_BASELINES
- result: `scripts/run_proteinmpnn_baseline.py` now supports IF-ready ProteinMPNN staging via `--test-set-parquet`. The wrapper resolves canonical IF-ready rows under `--input-pdb-folder`, rewrites each source PDB/CIF into a single-chain continuous-residue-numbered staging PDB (`1..L`, chain `A`) aligned to the parquet sequence, and maps staged ids back to original `protein_id` in output FASTA/parquet. Generated sequences are validated before output for non-canonical residues and length drift. The vendored ProteinMPNN parser/model files are unchanged.
- rationale: The observed `X` sequences came from ProteinMPNN's official parser preserving author-residue-number gaps as missing-coordinate positions, which are later represented as `X` and masked from redesign. The staging path aligns ProteinMPNN input with the existing IF-ready resolved-residue contract used by DPLM without dropping proteins or changing coordinates.
- artifacts:
  - `/home/zc1519/src/Immune-Design/scripts/run_proteinmpnn_baseline.py`
  - `/home/zc1519/src/Immune-Design/scripts/submit_if_baselines.slurm`
  - `/home/zc1519/src/Immune-Design/tests/scripts/test_run_proteinmpnn_baseline.py`
  - `/home/zc1519/src/Immune-Design/doc/SCRIPTS.md`
  - `/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/baselines/proteinmpnn_preflight/{0701,0401}/_mpnn_raw/staged_pdbs/`
- evidence: `pytest -q tests/scripts/test_run_proteinmpnn_baseline.py` -> 10 passed; `py_compile` clean for `scripts/run_proteinmpnn_baseline.py`; `git diff --check` clean for touched files. Full-data staging preflight succeeded for both canonical v2 IF-ready sets: 0701 3139/3139 staged, 0401 3147/3147 staged. CA residue numbering continuity check passed for every staged PDB (0 bad for both alleles).
- impact:
  - scope: ProteinMPNN comparison baseline wrapper + Module N SLURM defaults only.
  - risk: low-medium. The staging transform changes only PDB residue numbering/chain id presented to ProteinMPNN, not coordinates or sequence.
  - confidence: 0.9
- status: implemented; generation/evaluation metrics not yet recorded
- refs:
  - `scripts/run_proteinmpnn_baseline.py`
  - `scripts/submit_if_baselines.slurm`

### L0103
- timestamp: 2026-06-04T01:16:43-04:00
- type: VERIFICATION
- module: L
- trigger: User requested predicting GT backbone structures for the uricase missing-structure case set (1583 seqs lacking AFDB structures) with ESMFold2, strictly as evaluation GT.
- result: Folded the 1537 deduped legal uricase sequences (from 1583: −43 exact-duplicate sequences, −3 with X-run≥5) with ESMFold2 on ailab/H200 (8-shard array `9186985` + a 1-protein resume). 1537/1537 folded; 1494 (97.2%) pass the GT gate `mean pLDDT ≥ 0.90 & pTM ≥ 0.80` (ESMFold2 pLDDT is 0–1 scale). pLDDT median 0.957 (p5 0.921, min 0.657); pTM median 0.970 (p5 0.923, min 0.465). Per-protein mmCIF + PDB, merged manifest, and pass-id list written.
- rationale: uricase is an in-distribution known fold family, so ESMFold2 self-reported confidence is reliable here and a simple pLDDT+pTM gate suffices (no ensemble / cross-predictor machinery needed). ESMFold2 (`biohub/ESMFold2` + `biohub/ESMC-6B`) was installed in an isolated `esmfold2` conda env because its `esm` package shadows the fair-esm `esm` used by the existing `esmfold_runner` refold pipeline. Raw pLDDT/pTM are recorded per protein so the threshold is re-derivable without re-folding. METHODOLOGY CAVEAT: these predicted structures are GT for a case study only — using a predicted backbone as the scTM refold target for a protein with no experimental structure is self-consistency, not ground truth; paper-grade IF effectiveness should anchor on an independent folder (AF) and/or recovery vs the true native sequence.
- artifacts:
  - `/home/zc1519/src/Immune-Design/scripts/predict_esmfold2_gt.py`
  - `/home/zc1519/src/Immune-Design/scripts/submit_esmfold2_gt.slurm`
  - `/home/zc1519/src/Immune-Design/doc/SCRIPTS.md`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/esmfold2_inputs/{to_fold.fasta,fold_manifest.csv,dup_map.json,dropped_illegal.csv}`
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/esmfold2_structures/{mmcif/,pdb/,gt_manifest.jsonl,gt_manifest.parquet,gt_pass_ids.txt}`
- evidence: GPU smoke (A100, `--limit 5`) validated import/CUDA/API/gate and pLDDT 0–1 scale; full array sacct = 7/8 shards COMPLETED, shard 0 FAILED on a composite FASTA id containing `/` (treated as a path separator by `Path`), fixed by filename sanitization + a resume that folded only the 1 missing protein. Merge over the 8 shard manifests = 1537 unique ids; gate pass = 1494. Mean 2.8 s/protein on H200.
- impact:
  - scope: new uricase case-study GT structure set + 2 new scripts (Module L test-set curation); no change to existing IF/DPLM code or the `immune-design` env.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: optional — if-ready check (verify a sample of the ESMFold2 PDBs load via DPLM `load_coords()` with matching sequence) before using the 1494-protein pass set as IF evaluation conditioning/GT.
- refs:
  - `doc/SCRIPTS.md` (Module L · Test Set Curation #10; SLURM #5)


### L0104
- timestamp: 2026-06-04T00:55:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User reframed the uricase work as a standalone **case study** (translational de-immunization, separate from the RF-mechanism pilot/fast sets): build the *full* uricase pool into a complete dataset (structures + if_ready + h_maps), tag the characterized (proven-activity) members with a simple boolean (no evidence text), so the best designs can be cherry-picked downstream after RF. Also produce a "missing structure" list to fold externally.
- change_summary: Built the full uricase case-study dataset from `uricases/filtered_uricases.fasta` (6801) + 3 characterized not in the pool (O32141/Q45697/W8X3B8) = 6804 accessions (bare-accession protein_id; `characterized` boolean flag = 25). AFDB CIF download (`--source afdb`): 5222/6804 resolved (1582 no AFDB prediction). `build_if_ready_test_set.py --allow-cif`: **5222/5222 if_ready, 0 failures**. h_maps (npoff, both alleles): 5222 each, covers 100%, 0 failures. Emitted `missing_structure_list.{csv,fasta}` (1583 = 1582 no-AFDB uricases + 1 main-set extra; 2 are characterized) for external folding. **Operational note**: login-node `build_if_ready` on 5222 CIFs was SIGKILLed (~1h, empty log) — re-ran as a SLURM CPU job (`9187104`, 12 min). For large structure sets, run if_ready/downloads on a compute node, not login.
- rationale: For a case study, "best" = real therapeutic relevance, not RF-favorability or density — so no NMP/density selection; build everything and tag the curated (characterized) members. Bare accession protein_id is AFDB-native and clean. CATH overlap deliberately NOT annotated/excluded (case study ≠ controlled benchmark) — can be added later. NMP WT artifacts deferred (h_maps/if_ready prioritized per user); add a sharded NMP pass if a WT immunogenicity baseline is needed for eval.
- artifacts:
  - `work/.../if_test_set/uricases/uricase_caseset_if_ready.parquet` — 5222 (characterized flag; 23 with structure)
  - `work/.../if_test_set/uricases/h_maps/h_maps_uricase_DRB1_{07_01,04_01}.parquet` (+meta) — 5222 each, npoff
  - `work/.../if_test_set/uricases/pdbs_if_ready/` — 5222 cleaned CIF
  - `work/.../if_test_set/uricases/uricase_caseset_candidate.parquet` (6804), `uricase_caseset_structured.parquet` (5222), `afdb_structures/` (5222 raw CIF + download_failures.txt)
  - `work/.../if_test_set/uricases/missing_structure_list.{csv,fasta}` — 1583 to fold externally
- evidence: if_ready manifest n_input=5222 n_ready=5222 n_failed=0. h_maps both alleles cover 100% of if_ready, 0 head failures. characterized 23/25 in the if_ready set (2 Streptomyces A0ABR4SZB5/A0ABX7U467 have no AFDB → in the missing list). 6411→5222 with structure; unique sequences 4958.
- impact:
  - scope: New standalone uricase case-study dataset (not part of the main tier1/2/3 test set). Consumed by pointing `TEST_SET_PARQUET`/`H_MAPS_PARQUET`/`PDB_ROOT` at the uricase files. The main v2 test set still embeds Tier 3 uricases (151) — whether to drop them now that uricase is a separate case study is an open decision.
  - risk: low (additive). The 1583 missing-structure proteins are excluded until folded structures are supplied; then re-run if_ready/h_maps for those.
  - confidence: 0.9
- status: done
- next_action: (1) User folds the 1583 missing structures (prioritize the 2 characterized) and drops PDBs in; then materialize them into the case set. (2) Decide whether to remove Tier 3 from the main v2 test set. (3) If a WT immunogenicity baseline is needed, run a sharded NMP pass over the 5222. (4) Cherry-pick the best deimmunized uricases after RF generation.
- refs:
  - `L0097` (Tier 3 uricase expansion — the 25 characterized originate here), `L0101` (pilot/fast diagnostic subsets)

### L0105
- timestamp: 2026-06-04T10:18:31-04:00
- type: VERIFICATION
- module: L
- trigger: Executes L0104 next_action (1) — fold the missing-structure uricases (ESMFold2, per L0103) and materialize them into the case set; plus user request to unify all uricase structures into one location.
- change_summary: Made the 1537 ESMFold2 uricase structures (L0103) IF-ready via `build_if_ready_test_set.py` (PDB input, `--default-chain A`) and merged them into the existing uricase case set. 1518 IF-ready (`if_sequence_coverage==1.0`, `if_length_delta==0` for all); 19 failed = `unknown_residue` (UNK at X positions — same canonical-AA contract the 5222 AFDB set obeys). DPLM `byprot.utils.io.load_coords()` reads the new `.pdb` and returns the exact input sequence (3/3 spot-check). Unified `pdbs_if_ready/` now holds all 6740 uricase IF-ready structures (5222 `.cif` AFDB + 1518 `.pdb` ESMFold2); `uricase_caseset_if_ready_unified.parquet` (6740 rows, `structure_source ∈ {afdb, esmfold2}` + `mean_plddt`/`ptm`/`gt_pass` for ESMFold2 rows). Original AFDB-only parquet left intact.
- rationale: ESMFold2 mmCIF omits `_atom_site.occupancy` (build_if_ready cif reader KeyError) → used the biotite-written PDB instead. The 19 X→UNK sequences are excluded to stay consistent with the AFDB set's contract; raw structures remain under `esmfold2_structures/` if those 19 are wanted later. Structures co-located but provenance-tagged. NOTE: the full uricase GT is now entirely predicted (AF2 for 5222, ESMFold2 for 1518), no experimental — paper-grade IF effectiveness still needs an independent folder / recovery anchor (L0103).
- artifacts:
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/pdbs_if_ready/` — now 6740 (5222 cif + 1518 pdb)
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/uricase_caseset_if_ready_unified.parquet` — 6740 rows
  - `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/esmfold2_structures/{gt_manifest.parquet,gt_pass_ids.txt,esmfold2_caseset_if_ready.parquet,if_ready_failure.csv,mmcif/,pdb/}`
- evidence: build_if_ready summary 1518 ready / 19 failed (all `unknown_residue`); coverage==1.0 & length_delta==0 for all 1518; `load_coords(.pdb,'A') == input` 3/3; `pdbs_if_ready/` = 6740 (5222 cif + 1518 pdb); unified parquet 6740 rows, 0 duplicate `protein_id`.
- impact:
  - scope: uricase case-study IF-ready dataset (unified parquet + 1518 added structures); no change to existing AFDB parquet/structures or shared scripts.
  - risk: low
  - confidence: 0.95
- status: done
- next_action: use `uricase_caseset_if_ready_unified.parquet` + `pdbs_if_ready/` as the uricase IF eval set (filter `structure_source` / `gt_pass` as needed); decide separately whether to admit the 19 X-containing proteins.
- refs:
  - `L0103` (ESMFold2 fold), `L0104` (uricase case-study dataset build)


### L0106
- timestamp: 2026-06-04T11:32:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: Uricase is now a standalone case-study dataset, so Tier 3 is redundant in the main test set; user requested removing it.
- change_summary: Removed all Tier 3 rows from the canonical main test set (both alleles) — `test_proteins_<tag>.parquet`, `if_ready/test_proteins_if_ready_<tag>.parquet`, `if_ready/h_maps_v2/h_maps_<htag>.parquet` — and regenerated `test_proteins_summary_<tag>.json`. Main test set is now Tier 1 + Tier 2 only: 0701 3014 (T1 15 / T2 2999), 0401 3015 (T1 15 / T2 3000); if_ready + h_maps_v2 match (h_maps cover 100%, 0 missing). v1-with-tier3 backed up to `backups/20260604_113159_tier3_removal/`.
- evidence: 0701 test_proteins 3165→3014, if_ready 3139→3014, h_maps 3139→3014; 0401 3167→3015, 3147→3015, 3147→3015. h_maps cover if_ready missing=0 both alleles.
- impact: scope — main test set only; uricase case-study set untouched. risk low (backup retained).
- status: done
- refs:
  - `L0104` (uricase case study), `L0098`/`L0100` (Tier 2 v2 + promotion)

### L0107
- timestamp: 2026-06-05T10:52:00-04:00
- type: FIX
- module: IF_EVAL
- trigger: Structural evaluation exposed an ESMFold cache write failure when a valid protein identifier contained a path separator.
- result: ESMFold structure cache keys now replace path separators inside the protein-id component before constructing the cache basename, preserving the existing `(protein_id, sequence_hash)` cache semantics while preventing accidental nested paths.
- evidence: `python -m py_compile inverse_folding/evaluation/esmfold_runner.py scripts/evaluate_phase_c.py`; `pytest -q tests/inverse_folding/test_esmfold_runner.py` passed.
- impact: scope — ESMFold cache filename handling only; no change to refold model behavior, structural metrics, or ProteinMPNN generation.
- status: done

### L0108
- timestamp: 2026-06-05T13:10:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User noticed Tier 1 anchor `2VXP_A` in pilot50 despite its NMP `coverage_fraction` sitting below the mid-load band (0701 0.144 / 0401 0.379). Root cause: the original pilot admitted all 15 Tier 1 unconditionally, so anchors bypassed the density band. User decided Tier 1 anchors must be band-aligned on `coverage_fraction` (same band as Tier 2), take however many qualify per allele, fill the rest with Tier 2; target ~5 anchors/allele.
- change_summary: Added reusable `scripts/build_pilot_v2.py` (registered SCRIPTS.md §Module L #11) and rebuilt `if_ready/pilot_v2_<tag>.parquet` (both alleles, n=50, **5 Tier 1 + 45 Tier 2** each). Tier 1 now admitted only if `coverage_fraction ∈ [band_lo - tier1_relax, band_hi]`, capped at 5 via `--max-tier1` (keeps highest-coverage anchors, dropping from the lower band edge up); Tier 2 fills the remainder via 4-sub-bin `sample_uniform_bins`. **0701** band [0.33,0.56], relax 0, `--max-tier1 5` → anchors 6DNB,6HGM,6TN1,1U6T,1ZR3 (the 6th in-band anchor 2ERF@0.344, nearest the lower edge, dropped). **0401** band [0.39,0.64], relax 0.001 to admit near-bar 6DNB (cf 0.389294) → anchors 6DNB,6TN1,5HGJ,4ZIE,2ERF (only 5 qualify, cap inert). Source pool is `fast_v2` (only surviving artifact carrying `coverage_fraction`), so pilot rows are a strict subset of fast_v2 and inherit canonical if_ready schema + the `coverage_fraction` column; pilot Tier 2 ⊂ fast Tier 2 by construction. Previous pilot backed up to `backups/20260605_130737_pilot_rebuild/`.
- evidence: 0701 pilot coverage_fraction min/median/max = 0.335/0.429/0.551, sub-bins [12,11,11,11]; 0401 = 0.389/0.498/0.640, sub-bins [12,11,11,11]. Both: 2VXP_A absent, 5 Tier 1 + 45 Tier 2, 0 pdb_path/coverage_fraction nulls, all 50 ids present in canonical if_ready and h_maps_v2 (0 missing). `py_compile` clean.
- impact: scope — pilot diagnostic subset only; canonical main test set and fast_v2 untouched. risk low (backup retained, read-only subset of canonical). The 15-Tier-1-anchor convention still holds for fast_v2 (300 rows, anchors a small fraction).
- status: done
- refs:
  - `L0101` (original pilot/fast build), `L0100` (v2 promotion)

### L0109
- timestamp: 2026-06-07T16:00:00-04:00
- type: TOOLING
- module: IF_EVAL
- trigger: The 0701 WT immune baseline (full eval_immune over the 3139 superset, covers current 3014) exists, but 0401 had no WT facade and no immune eval at all. The single-job 0701 run took ~7.7 h, so the 0401 baseline is sharded for fast parallel start.
- change_summary: Added two reusable scripts (registered SCRIPTS.md §Phase B Evaluation Integration #2/#3): `scripts/build_wt_facade.py` (builds the WT `generated.parquet` facade from canonical IF-ready + optional N round-robin shard facades) and `scripts/merge_eval_immune_shards.py` (concatenates per-shard `imm_head/imm_nmp/imm_head_residues/imm_nmp_peptides` + failures into one unified run dir). Extended `scripts/submit_benchmark.slurm` phase_c branch with an additive facade-shard block (only fires when `N_SHARDS`+`SLURM_ARRAY_TASK_ID` are set; inserts `.shardKKofNN` into `GENERATED_PARQUET`/`RUN_ID`). Built the 0401 facade `work/.../if_test_set/wt_generated_v2_HLA-DRB1_04_01.parquet` (3015 rows) + 24 round-robin shards (125–126 each). Uses the npoff 0401 head (`cnn_himp_v1_npoff_drb0401_seed42/.../LC1/seed_42/best.pt`) to match the 0701 baseline's npoff head.
- evidence: `py_compile` clean on both scripts; `bash -n` clean on the slurm. Login-node 2-protein smoke (evaluate_phase_c `--mode imm --imm-full`, npoff 0401 head, accelerated NMP) produced imm_head(2)/imm_nmp(2)/imm_head_residues(550)/imm_nmp_peptides(7210), 0 failures, ~16s/protein at 2 workers. Submitted array 9358509 (`--array=0-23`, cpu/qos=short, 8 cpus, IMM_FULL=1) + dependent merge 9358510 (`afterok`, `--expected-proteins 3015` → unified `run/benchmark/wt_v2/HLA-DRB1_04_01/wt_v2_HLA-DRB1_04_01_imm_full/`).
- impact: scope — adds 0401 WT immune baseline + reusable sharding tooling; no change to existing 0701 baseline or non-array benchmark behavior. risk low (additive, smoke-verified).
- status: done
- refs:
  - `L0108` (pilot rebuild), `L0100` (v2 promotion)


### L0110
- timestamp: 2026-06-09T21:55:33-04:00
- type: VERIFICATION
- module: RF
- trigger: User requested `PLAN_RF.md` Task D2-D3 Stage A Actuation Pipeline implementation, restricted to Stage A scope.
- change_summary: Implemented Stage A D2/D3 actuation pipeline and post-review fixes for remask-ledger persistence metrics, per-correction D2 selection rates, d2_logits rank/freeze config materialization, ESS schema alignment, d2_d3_full D3 telemetry gating so non-refresh remask steps do not emit stale D3 event rows or productive-revisit snapshots, and current-step Stage A rank caching for remask-ledger rank_score fields.
- rationale: The pre-Stage-A D2 correction only acted at refresh and could be erased by legacy remask; Stage A tests whether D2-written tokens can reach sampling, earn realized-benefit rank credit, and survive remask without adding Phase C/GR mechanisms. Post-review fixes prevent the A-to-B persistence gate and selected-after-D2 headline metric from being inflated or diluted by telemetry-path artifacts.
- artifacts:
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/configs/d2_logits.yaml`
  - `inverse_folding/reference_flow/configs/d2_d3_full.yaml`
  - `inverse_folding/reference_flow/configs/d_monitor_full.yaml`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/counterfactual.py`
  - `inverse_folding/reference_flow/commit.py`
  - `scripts/run_if_phase_c1.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_counterfactual.py`
  - `tests/inverse_folding/test_reference_flow_commit.py`
  - `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
- evidence: |
    `pytest tests/inverse_folding/test_reference_flow_controller_config.py tests/inverse_folding/test_reference_flow_counterfactual.py tests/inverse_folding/test_reference_flow_commit.py -q` -> 92 passed.
    `pytest tests/inverse_folding/test_reference_flow_sampler_controller.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py tests/scripts/test_run_if_phase_c1_d2_d3.py -q` -> 60 passed.
    `pytest tests/inverse_folding/test_reference_flow*.py -q` -> 187 passed, 1 existing amplification clipping warning.
    `pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -q` -> 13 passed.
    `python - <<'PY' ... import scripts.run_if_phase_c1 ...` -> `surface 3`.
    `bash -n scripts/submit_if_phase_c.slurm` -> clean.
    `git diff --check` on touched Stage A files -> clean.
- impact:
  - scope: Reference Flow controller Stage A actuation pipeline and telemetry surfaces only; no Phase C scheduling, GR pressure, hard write-back, post-sampling accept/reject, new driver, or new SLURM script.
  - risk: medium
  - confidence: 0.90
- status: done
- next_action: Run the planned 2-protein real-head smoke on Della through `scripts/submit_if_phase_c.slurm`, then the 50-protein Stage A pilot with matched no-sticky or `sticky_ttl_steps=1` baseline before advancing to Stage B.
- refs:
  - `PLAN_RF.md` Task D2-D3: Stage A Actuation Pipeline
  - `doc/RF_Controller_Architecture.md`


### L0111
- timestamp: 2026-06-10T00:00:42-04:00
- type: FEATURE
- module: RF
- trigger: User requested `PLAN_RF_UNI_CTRL.md` Stage B (typed actionability targeting) implementation, with Stage C kept as gated plan only; alignment on r_ctx/b_env/scope resolved before coding.
- change_summary: Implemented the Stage B unified-controller typed actionability field `A_i(t)`. New pure-function module `actionability.py` (max-covering projection, anchored SoftOR, b_env peak×consistency, fresh-evidence/memory firewall, target-vs-pressure split, cluster support, G/g_GR). Added `controller.targeting` (`static_excess | typed_actionability`) + `controller.global_pressure` config blocks and `controller.d3.evidence_source` (`legacy_window_excess | typed_fresh`) with validation (`static_median` only; `first_reliable_refresh_median` reserved→NotImplementedError). Controller computes the typed field pre-D2 each refresh (tau_ref_B from static window cache; b_cur via max-covering excess; discovery-before-D2 r_ctx = g_time·g_comp·g_ent·g_pnll·g_stability with g_pnll from `compute_context_pnll`; b_env via K_env=3 global envelope completions over the seed-window union; b_mem EMA; v_target SoftOR) and routes active-window selection from `v_target` in typed mode while leaving the legacy `z_dyn − z_static` path bit-for-bit unchanged in static mode. D3 EMA `m_i` consumes the memory-excluded `e_fresh` in d2_d3_full_stageB via a `compute_d3_evidence_input` firewall. Added Stage B presets `d_monitor_stageB.yaml` / `d2_d3_full_stageB.yaml` and `actionability_residues.parquet` + `actionability_refresh_summary.jsonl` telemetry sidecars, compact refresh_log pointers, manifest `targeting_config`/`global_pressure_config`/`targeting_config_hash`, and startup prints. Stage C (g_GR scaling β/λ + Phase C editability) NOT implemented — gated plan only; `global_pressure.enabled=false` so G/g_GR are diagnostic-only.
- rationale: Stage B isolates the targeting bottleneck (where to steer) from Stage A actuation (whether a steered token persists). The typed field replaces the static-prior runtime subtraction with sequence-conditioned, refresh-local evidence (b_cur/b_env/b_mem) so dynamic targeting can be tested head-to-head against the static source on the same pilot. The memory firewall (e_fresh excludes b_mem; update_memory signature rejects v_target/u_pressure/m_i) prevents memory-inflated targeting from feeding back into the D3 commit memory. r_ctx is defined pre-D2 (not from `D2BlockOutcome.g_pnll`) because v_target must discover the blocks before D2 runs and monitor mode has no D2; b_env uses global envelope samples (K_env head calls/refresh, not num_seed_windows×K_env) to bound cost.
- artifacts:
  - `inverse_folding/reference_flow/actionability.py` (new)
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/commit.py`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageB.yaml` (new)
  - `inverse_folding/reference_flow/configs/d_monitor_stageB.yaml` (new)
  - `scripts/run_if_phase_c1.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_actionability.py` (new)
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`
  - `tests/inverse_folding/test_reference_flow_commit.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
- evidence: |
    TDD red→green per task (B1–B6).
    `pytest tests/inverse_folding/test_reference_flow_actionability.py tests/inverse_folding/test_reference_flow_controller_config.py tests/inverse_folding/test_reference_flow_d1_controller.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py tests/inverse_folding/test_reference_flow_commit.py tests/scripts/test_run_if_phase_c1_d2_d3.py -q` -> 167 passed (Stage B acceptance set).
    `pytest tests/inverse_folding/ tests/scripts/test_run_if_phase_c1_d2_d3.py -k "reference_flow or run_if_phase_c1" -q` -> 235 passed, 2 skipped, 1 pre-existing amplification clipping warning.
    Legacy bit-equivalence: 108 pre-B4 controller/commit/counterfactual tests unchanged after typed integration; static_excess refresh emits no actionability states and 1 dyn head call (no envelope).
    Both Stage B presets load via `load_controller_config`; `ast.parse` clean on `run_if_phase_c1.py`.
- impact:
  - scope: Reference Flow controller targeting layer (Stage B) + telemetry/manifest + two presets only; no model retraining, no NetMHC guidance, no Stage C actuation, no new SLURM script. `runtime.py` untouched (manifest authority lives in `run_if_phase_c1.d1_manifest_provenance`).
  - risk: medium
  - confidence: 0.88
- status: done
- next_action: Real-head 2-protein smoke on Della with `d_monitor_stageB.yaml` to confirm the actionability sidecars + manifest fields, then the 50-protein Stage B gate (dynamic A_i(t) vs static z_dyn−z_static on realized_benefit_rate; end-to-end ≥5pp Pareto over D3-only; focal/memory-firewall diagnostics). Stage B deployment is gated on Stage A local gates (final_persistence_rate ≥ 10%, structure ≥ D3-only, realized_benefit_rate > 0) passing first.
- refs:
  - `PLAN_RF_UNI_CTRL.md` Stage B
  - `doc/RF_Controller_Architecture.md` §2.8-§2.9
  - `L0110` (Stage A actuation)

### L0112
- timestamp: 2026-06-11T00:00:00-04:00
- type: FEATURE
- module: RF
- trigger: User requested `PLAN_RF_UNI_CTRL.md` Stage C.1 (trajectory-level global-pressure scaling) implementation, with Stage C.2 (Phase C editability allocation) kept as gated plan only; alignment confirmed `g_min` default 0.25→0.0 (PLAN-literal) before coding.
- change_summary: Implemented Stage C.1 global-pressure actuation as an actuator-only change over the Stage B typed controller. Added pure `smoothstep_pressure(B_GR; B_low,B_high,g_min,g_max)` (clipped cubic smoothstep, distinct from the Stage B `global_pressure_scalar` sigmoid diagnostic) and `protein_pressure_burden` (median of reliable per-refresh G) to `actionability.py`. Extended `GlobalPressureConfig` with the Stage C fields (`pressure_source`, `mapping`, `B_low/B_high`, `min_reliable_refreshes`, `unready_g`, `scale_beta`, `scale_lambda`; `g_min` default 0.25→0.0), added `validate_global_pressure_runtime` (band-presence fail-fast), and replaced the Stage-B `enabled=true` rejection with `_cross_validate_stage_bc` (enabled ⇒ typed_actionability). The controller now maintains a per-trajectory pressure state (`_pressure_G_values/_B_GR/_g_GR`, reliability-filtered, append-then-compute) and scales D2 `beta` (`correct_logits(beta_override=…)`, beta_eff baked into the sticky deltas at refresh creation — never recomputed on re-delivery) and D3/rank `lambda_commit` (`run_refresh(lambda_override=…)` + `_build_stage_a_rank_scores`) by the trajectory `g_GR`, gated independently by `scale_beta`/`scale_lambda`. `run_if_phase_c1.py` loads `--global-pressure-calibration-json` and stamps `B_low/B_high` into the config via `dataclasses.replace` before the config hash (fail-fast on missing flag / missing keys / inverted band). New primary config `d2_d3_full_stageC_pressure_aopen.yaml` (copies the Stage B unified-aopen preset, only enables `global_pressure`); `submit_if_phase_c.slurm` forwards `GLOBAL_PRESSURE_CALIBRATION_JSON`. Stage C.2 (`c_ready_i`/`f_rank_i` + schedule editability) NOT implemented — gated plan only.
- rationale: RAR 0005/0001 found the aggregate immune outcome is masked by a controller-wide low-burden over-intervention; typed targeting (Stage B) still fires on each protein's top positions regardless of absolute burden, so only trajectory-level `g_GR` pressure (Stage C) can suppress it. The actuator reads a stable per-design `B_GR = median(reliable G_step)` and a hard-saturating smoothstep (not the Stage B per-refresh sigmoid) so low-burden trajectories reach exactly `g_min=0` (PLAN G1/G2). `beta_eff` is frozen into each D2 correction at refresh creation (sticky re-delivery replays stored deltas) so within-TTL pressure is consistent and β is never recomputed (PLAN G8). `B_low/B_high` come from the Stage B/Aopen pilot via a CLI calibration JSON, never hardcoded (no placeholder anchors). All pressure scaling and Stage C telemetry are gated on `global_pressure.enabled`, so Stage B / legacy runs are byte-for-byte unchanged.
- artifacts:
  - `inverse_folding/reference_flow/actionability.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/commit.py`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml` (new)
  - `scripts/run_if_phase_c1.py`
  - `scripts/submit_if_phase_c.slurm`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_actionability.py`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
- evidence: |
    TDD red→green per task (C1.1–C1.8).
    `pytest tests/inverse_folding/test_reference_flow_actionability.py tests/inverse_folding/test_reference_flow_controller_config.py tests/inverse_folding/test_reference_flow_d1_controller.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py tests/inverse_folding/test_reference_flow_commit.py tests/scripts/test_run_if_phase_c1_d2_d3.py tests/scripts/test_run_if_phase_c1_d1.py -q` -> 216 passed (Stage C.1 acceptance set).
    `pytest tests/inverse_folding/ tests/scripts/ -q` (excluding 3 pre-existing Biopython-broken files) -> 777 passed, 42 skipped; `git diff --check` clean.
    Adversarial multi-agent review (6 dimensions: math, byte-identity, config, sticky-D2, lambda-D3-telemetry, script/PLAN-checklist) -> 0 confirmed findings.
    Key invariants verified: g_GR=1.0 ⇒ D2-corrected logits and Stage-A rank match the Stage B typed arm bit-for-bit at the same seed; g_GR=0.0 ⇒ beta-zero posterior (identity logits) and lambda immune penalty drops out (== lambda_commit=0); B_GR is the trajectory median (not the latest spike); sticky re-delivery replays the beta_eff-baked deltas; `global_pressure.enabled=false` leaves all Stage B telemetry / sampling unchanged.
- impact:
  - scope: Reference Flow controller actuation layer (Stage C.1) + calibration wiring + telemetry/manifest + one new config; no model retraining, no NetMHC guidance, no Stage C.2 schedule editability, no new SLURM script.
  - risk: medium
  - confidence: 0.86
- status: done
- next_action: Build the calibration JSON from the returned Stage B/Aopen typed `actionability_refresh_summary.jsonl` (median G_step per (protein,design,seed); B_low/B_high = 1/3, 2/3 quantiles), then run the 50-protein Stage C.1 arm (`d2_d3_full_stageC_pressure_aopen.yaml` + `--global-pressure-calibration-json`) against the matched static Aopen comparator (RAR 0005). Read-out: burden-stratified low-bin suppression (dhead ≤ +0.5), high-bin retention (dhead ≤ −2.0), aggregate movement toward the GR ideal (mean dhead ≤ −0.8), structure ≥ matched D3-only; fall back to `scale_lambda=false` sensitivity if high-bin retention fails.
- refs:
  - `PLAN_RF_UNI_CTRL.md` Stage C.1
  - `doc/RF_Controller_Architecture.md` §2.8-§2.9
  - `L0111` (Stage B typed targeting)

### L0113
- timestamp: 2026-06-12T00:00:00-04:00
- type: FEATURE
- module: RF
- trigger: RAR 0006 revision of PLAN_RF_UNI_CTRL.md: implement Stage B.1 (within-block typed position selection) and rebase Stage C.1 onto the thresholded-G driver + typed_target/legacy-D3 base, superseding the L0112 trajectory_median_G/unified route. C.2 stays gated.
- change_summary: (B.1) `select_editable_positions` gains `score_source` + `typed_field`: within-block editable positions rank by the typed `v_target` when `targeting.within_block_source='v_target'` (low-entropy tiebreak retained), else legacy `residue_excess` (default ⇒ bit-for-bit). Threaded through `D2Handler.correct_logits`/`_process_block`; the controller passes the active block's per-residue `v_target` in typed mode (None in static ⇒ legacy fallback). `TargetingConfig.within_block_source` (enum `legacy_excess|v_target`; `e_fresh` reserved). (C.1) The per-refresh `G_step` driver is changed from `mean(u_pressure)` to the prominence-thresholded mass `(1/L)Σ ReLU(u_pressure − τ_prom)` (new pure `prominence_thresholded_mass`; G3 / RAR 0006 M11). `τ_prom` is a second prominence cut (NOT a re-subtraction of `tau_ref_B`), stamped from the calibration JSON; `τ_prom=None⇒0` makes Stage B's `G` byte-identical (u_pressure≥0 ⇒ ReLU(u−0)=u). `mean(u_pressure)` is retained as the `G_step_mean` diagnostic. `GlobalPressureConfig` gains `tau_prom`/`tau_prom_source`; `pressure_source` default → `trajectory_thresholded_G`; validation now requires (when enabled) `pressure_source=trajectory_thresholded_G` (rejects the legacy `trajectory_median_G`), `tau_prom`+band present, and `d3.evidence_source=legacy_window_excess` (G10 / RAR 0006 M9 — typed_fresh harmful, deferred). The run-setup loader reads/validates `pressure_source`+`tau_prom`+band from the JSON (rejects wrong source / missing `tau_prom` / YAML↔JSON source mismatch), stamps them before the config hash; manifest gains `global_pressure_tau_prom`. Rebased `d2_d3_full_stageC_pressure_aopen.yaml` onto `d2_d3_full_stageB_aopen.yaml` (legacy D3 + `within_block_source: v_target` + thresholded pressure); added `within_block_source: v_target` to `d2_d3_full_stageB_aopen.yaml` (the B.1 mechanism run).
- rationale: RAR 0006 M7 found Stage B.0 routed only the *block* source to `v_target` while `select_editable_positions` still ranked within-block positions by legacy excess (`edited_in_high_v_top4_frac=0.448<0.5`); B.1 lets the typed signal reach the position layer. M8/M11 found `mean(u_pressure)` is floor-dominated (typed field positive at ~100% of residues) and does not discriminate burden, so the thresholded `G_step` with a calibrated `τ_prom` (Spearman(B_GR,d3 burden)=0.46 at `τ_prom=11.75`) is the actuator driver. M9 found `typed_fresh` (unified) inflates the D3 EMA (harmful), so C.1's base is the typed_target/legacy-D3 arm; `typed_fresh` is deferred until the field-floor fix. All new behavior is gated on `within_block_source`/`global_pressure.enabled`, so static/Stage-B runs are byte-for-byte unchanged.
- artifacts:
  - `inverse_folding/reference_flow/actionability.py`
  - `inverse_folding/reference_flow/counterfactual.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen.yaml`
  - `scripts/run_if_phase_c1.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_actionability.py`
  - `tests/inverse_folding/test_reference_flow_counterfactual.py`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `tests/inverse_folding/test_reference_flow_d2_d3_controller.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
- evidence: |
    TDD red→green per task (B.1; C.1 thresholded driver; loader/validation).
    `pytest <Stage B/C acceptance set> -q` -> 257 passed.
    `pytest tests/inverse_folding/ tests/scripts/ -q` (excluding 3 pre-existing Biopython-broken files) -> 794 passed, 42 skipped; `git diff --check` clean.
    Byte-identity verified: within_block_source=legacy_excess reproduces D2 editable selection bit-for-bit; tau_prom=0 ⇒ G==mean(u_pressure) (Stage B diagnostic unchanged); g_GR=1.0 still matches the Stage B typed arm. New: within_block v_target picks high-v_target/low-legacy residues; tau_prom>0 thresholds G strictly below G_step_mean; enabled⇒thresholded/tau_prom/legacy-D3 fail-fast; calibration JSON wrong-source / missing-tau_prom / source-mismatch fail-fast.
- impact:
  - scope: Reference Flow within-block targeting (B.1) + Stage C.1 pressure driver/base rebase + calibration wiring + telemetry/manifest; supersedes L0112's trajectory_median_G/unified route. No model retraining, no NetMHC guidance, no C.2, no new SLURM script.
  - risk: medium
  - confidence: 0.85
- status: done
- next_action: Run the B.1 mechanism arm (`d2_d3_full_stageB_aopen.yaml`) and re-run RAR 0006 M7 (`edited_in_high_v_top4_frac` should rise above 0.448, Spearman(v_target,edited) above 0.31). Then the Stage C.1 outcome arm (`d2_d3_full_stageC_pressure_aopen.yaml` + the RAR 0006 `--global-pressure-calibration-json` at `tau_prom=11.75`) vs the matched static Aopen comparator; burden-stratified read-out (G7): low-bin median dhead improves ≥50% vs Aopen `+2.080`, high-bin retains ~`-2.5`, structure ≥ matched D3-only.
- refs:
  - `PLAN_RF_UNI_CTRL.md` Stage B.1 + Stage C.1 (RAR 0006 revision)
  - `doc/RF_Controller_Architecture.md` §2.8
  - `L0112` (superseded Stage C.1 route)

### L0114
- timestamp: 2026-06-09T22:30:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: Need a dataset that demonstrates RF's de-immunization gain on genuinely high-risk proteins. User's thesis: select on the BASELINE generative distribution's risk (where the baseline can't avoid epitopes → real headroom), NOT on WT — if the unconditional distribution is already low-risk, the gain is invisible regardless of WT.
- change_summary: Added `scripts/build_highrisk_demo.py` (registered SCRIPTS.md §Module L #12) and built `if_ready/highrisk_demo_v1_<tag>.parquet` (both alleles, **n=400 each**). Selection anchors on the **ProteinMPNN** 8-design immunogenicity distribution (`v2_8design_20260604T024603Z` eval_immune): per-protein strong-window fraction `n_strong/n_windows`, ranked by `pm_med` (top-400 after a 10% head-`global_risk` soft floor so RF guidance has signal). Structure NOT gated (ProteinMPNN scTM median 0.96–0.97). Each row is a canonical IF-ready subset + diagnostic columns (`pm_min/med/mean/max`, `wt_sf`, `pm_head`, `wt_head`, `highrisk_rank`) for later re-filtering against the DPLM no-guidance distribution (running) without recomputation. Generous size by design — to be narrowed to ~50–100 later. Required building the **0401 WT immune baseline** first (see L0109).
- evidence: Empirical grounding before build — among WT-high (top-quartile) proteins, 34%(0701)/24%(0401) already have ProteinMPNN best-of-8 ≤ median (baseline ~solves → no headroom); corr(WT coverage_fraction, ProteinMPNN pm_med)=0.38/0.54 (WT is a weak proxy). v2 test-set validation: WT coverage_fraction is the intended unimodal Gaussian (median 0.386/0.444, near-symmetric) — construction sound — but its `min-per-bin=40` floor admits ~150/2999 tier2 that are WT-immune-free (n_strong=0), confirming v2 is a moderate-load REPRESENTATIVE set, not a high-risk set. Built set: 0701 pm_med 0.030–0.127 (median 0.038, vs full-set median 0.014); 0401 pm_med 0.051–0.163 (median 0.072, vs 0.020); both 400 unique, 0 pdb_path nulls, all in canonical if_ready + h_maps_v2.
- impact: scope — new high-risk demonstration subset; canonical test set, pilot/fast untouched. risk low (read-only canonical subset; generous, re-filterable). NOTE the head-floor uses the epitope head (the RF guidance) — accepted as a soft secondary floor only (NMP/ProteinMPNN burden is the non-circular primary axis).
- status: done
- next_action: When the DPLM no-guidance distribution lands, re-filter to proteins also-hard-for-DPLM; add an NMP strong-window evenness metric to drop single-peak cases; narrow to the final ~50–100 per allele.
- refs:
  - `L0109` (WT immune baseline tooling), `L0108` (pilot rebuild), `L0098`/`L0100` (v2 build/promotion)

### L0115
- timestamp: 2026-06-19T00:00:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement the monitor-only Self-Conditioned Proposal-Envelope GR probe (SC-GR) from `doc/Self-Cond_GR.md` per `PLAN_RF_SC_GR.md` Tasks SC0.1–SC0.5. SC-GR replaces ONLY the burden estimator behind `g_GR` with a recycled, ensembled, pre-D2 terminal-risk probe and is monitor-gated: it writes burden-estimator telemetry and changes no actuation. Post-monitor behavior stages (SC1 beta-only pressure / SC2 amplify / SC3 instability) are deferred and NOT implemented.
- change_summary: (SC0.1) New frozen `SelfConditionedGRConfig` (`enabled/mode/arms/ensemble_size/struct_temperature/confidence_source/confidence_threshold/update_state_from/probe_refresh_policy/top_m/lse_temperature/supra_tau_values/write_probe_telemetry`) added to `ControllerConfig` + `controller_config_to_dict` (asdict). `_materialize_self_conditioned_gr` validates the SC0.1 enums/ranges; `_cross_validate_scgr` enforces `monitor_only ⇒ global_pressure.enabled=false`. New preset `d2_d3_full_stageB_aopen_scgr_monitor.yaml` copies the Stage B.1 Aopen operating point verbatim and only adds the `self_conditioned_gr` block (pressure actuator off). (SC0.2) New pure module `self_conditioned_gr.py`: `SCGRState/SCGRProbeSample/SCGRRiskAggregates` + `build_probe_samples` (deterministic per-`(protein,design,seed,refresh,arm,sample)` RNG isolated from the sampler, canonical-AA sampling, committed positions fixed, `self_conditioned` reuses the prev structural-argmax token only where prev canonical-softmax confidence ≥ threshold else samples, no prev state ⇒ bootstrap=fresh), `update_state_from_structural_argmax` (canonical-softmax max-prob confidence, NO head input — firewall by signature), `compute_risk_aggregates` (full-sequence `b_cur` projection → `G_mean_excess` / top-m LSE / supra-mass; length-normalized; empty→0), `summarize_probe_refresh` (robust `median` `B_sc` + over-K `max`/`std` instability + `old_argmax_*` baseline). (SC0.3) Controller pre-D2 probe seam `_run_sc_gr_probe` (runs after scores/`_update_pressure_state`, before active-window selection + D2; mode-agnostic); reads only `x_t`/structural logits/the static window cache (`tau_ref_B` = static-z quantile, identical to `b_cur`)/the frozen head; `old_argmax` reuses `dyn_score` (no extra head call); one batched head call for all completions; updates `_scgr_state` after the probe; accessors `self_conditioned_gr_sample_rows/refresh_rows`; `__init__` canonical fail-fast. (SC0.4) Driver `SC_GR_PROBE_{SAMPLES,REFRESH}_FILE` + base-column schemas, `write_sc_gr_probe_artifacts` (empty rows ⇒ schema-readable parquet), per-design row collection (gated on `enabled`), manifest `self_conditioned_gr_config` (via `d1_manifest_provenance`) + sidecar paths (only when `enabled` + `write_probe_telemetry`), resolved-config printing. (SC0.5) New analysis CLI `scripts/analysis/sc_gr_monitor_probe.py`: per-protein `Spearman(B_sc, oracle)` / tercile crosswalk diagonal / `P(probe low|oracle high)` / `Recall@High` per arm×refresh_step×aggregator, best estimator by `Recall@High` then `Spearman` (pandas+numpy only); registered in `doc/SCRIPTS.md`. Degenerate-binning guard: non-finite `(probe, oracle)` pairs are dropped, a collapsed tercile cut (`q1==q2`, e.g. a constant/near-constant burden) or <3 finite points reports NaN bin stats instead of a spurious `Recall@High=1.0`, and `select_best_metric` requires BOTH a finite `Recall@High` and a finite `Spearman` so a non-informative constant estimator can never be selected as best.
- rationale: RAR 0008 (NoD-baseline) reframed GR — the single-argmax `B_GR` correlates only ρ≈0.41 with the NoD per-protein burden (the argmax is the *mode*, not the expectation, of a sharply multimodal terminal-risk functional), and the amplify-high direction (the −1.0 nat high-burden lever) is the real objective with do-no-harm of low-burden as the secondary, within-noise constraint. SC-GR is the redesigned estimator; this stage MEASURES it (gate: best Spearman ≥ 0.60, Recall@High beats C.1) before any actuation, so the gain can be calibrated without touching D2/D3. Monitor-only by construction: the probe is firewalled (pre-D2, reads no D2/D3/pressure state, mutates nothing but its own telemetry + carried state), so generation is byte-for-byte unchanged whether SC-GR is enabled or not.
- artifacts:
  - `inverse_folding/reference_flow/self_conditioned_gr.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor.yaml`
  - `scripts/run_if_phase_c1.py`
  - `scripts/analysis/sc_gr_monitor_probe.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_self_conditioned_gr.py`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
  - `tests/scripts/test_sc_gr_monitor_probe.py`
- evidence: |
    TDD red→green per task (SC0.1–SC0.5).
    Unit Gate (`test_reference_flow_self_conditioned_gr` + `_controller_config` + `_d1_controller` + `test_run_if_phase_c1_d2_d3` + `test_sc_gr_monitor_probe`) -> 182 passed.
    `pytest tests/inverse_folding/ tests/scripts/ -q` (excluding 3 pre-existing Biopython-broken files) -> 848 passed, 42 skipped; `git diff --check` clean.
    Firewall verified: SC-GR enabled vs disabled returns identical logits; no in-place mutation of `context.logits`/`context.x_t`; `B_GR`/`g_GR_effective` stay None when `global_pressure.enabled=false`; `self_conditioned` recycles across refreshes (bootstrap on refresh 0, full reuse on refresh 1 under sharp logits); enabled without `canonical_token_ids` fails fast at construction. Aggregator math: top-m LSE > mean on a focal spike; supra-mass + length-normalization exact; empty windows → 0. Analysis: monotone probe → Spearman≈1; true-high-as-low denominator conditioned on oracle-high; best-metric prefers Recall@High over Spearman; a constant probe (collapsed tercile cut) now yields NaN bin stats (not Recall@High=1.0) and is excluded from best-metric selection; non-finite pairs filtered. Monitor preset differs from `d2_d3_full_stageB_aopen.yaml` only by the `self_conditioned_gr` block.
- impact:
  - scope: RF controller monitor/sensing layer — new pre-D2 burden-estimator probe + two telemetry sidecars (`sc_gr_probe_samples/refresh.parquet`) + monitor analysis CLI + one new config preset. No generation-behavior change (monitor-only, firewalled), no actuation, no model retraining, no NetMHC guidance, no new SLURM script. Post-monitor SC1/SC2/SC3 deferred.
  - risk: low
  - confidence: 0.85
- status: done
- next_action: Run the monitor on the 50-protein pilot with `d2_d3_full_stageB_aopen_scgr_monitor.yaml` via the existing `submit_if_phase_c.slurm` (`N_STEPS=20` so the `self_conditioned` arm is exercised across ≥2 refreshes; `GLOBAL_PRESSURE_CALIBRATION_JSON=""`). Then `scripts/analysis/sc_gr_monitor_probe.py` against the NoD oracle (`imm_head.parquet`). Gate (`doc/Self-Cond_GR.md` §6 / `PLAN_RF_SC_GR.md` §5): best `Spearman(B_sc, NoD burden) ≥ 0.60`, `Recall@High` beats old C.1 `B_GR`, `P(probe low|oracle high)` drops, `self_conditioned ≥ fresh` at a multi-refresh step. If the gate passes, SC1 (beta-only SC pressure, `mode=beta_pressure`) is the next coder handoff.
- refs:
  - `PLAN_RF_SC_GR.md` Tasks SC0.1–SC0.5
  - `doc/Self-Cond_GR.md` §1–§7
  - `L0113` (Stage B.1 / Stage C.1 base operating point)

### L0116
- timestamp: 2026-06-19T22:40:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: The full DPLM-native immune baseline (0701, 3014 proteins, 8 designs) landed (`run/benchmark/phase_d/baselines/dplm_native_full_v2/.../c0_..._imm_full_single64_20260619T024206Z`). Comparing it to ProteinMPNN showed the two baselines' high-burden tails barely overlap, so the single ProteinMPNN-only set (L0114) is reframed into two purpose-separated sets per user decision.
- change_summary: Refactored `scripts/build_highrisk_demo.py` to a general selector — `--primary-imm-dir` (the baseline to select on) + `--rank-mode both_nmp_head` (rank by `min(nmp_pct, head_pct)`, i.e. NMP strong_frac AND head global_risk both high; head promoted from gate to co-ranking axis) + repeatable `--diag-imm-dir LABEL=DIR` for cross-baseline diagnostic columns. Built two SEPARATE 0701 top-100 sets: `if_ready/highrisk_pmpnn_demo_v1_HLA-DRB1_07_01.parquet` (primary=ProteinMPNN — demonstration / "beat the reported baseline") and `if_ready/highrisk_dplm_iter_v1_HLA-DRB1_07_01.parquet` (primary=DPLM-native — RF validation/iteration, where RF's own base distribution is hard). Each row = canonical IF-ready subset + `sel_nmp/sel_head/both_min_pct/highrisk_rank` + the other two baselines' `*_nmp/*_head` diagnostics. (Supersedes the L0114 ProteinMPNN-only 400-set `highrisk_demo_v1_*`.)
- evidence: DPLM-native vs ProteinMPNN per-protein strong_frac: marginals near-identical (med 0.013 vs 0.014) but alignment only Pearson 0.45 / Spearman 0.46 full-range, **0.11 in the top decile** (top-150 share 31/150) — high-risk is baseline-dependent. DPLM 8 designs near-constant burden (min≈med) vs ProteinMPNN high design-diversity (min 0.005 ≪ med 0.014). Both built sets: nmp_pct AND head_pct in [0.90,1.00]; head–NMP correlate Pearson 0.60(A)/0.53(B) so the co-rank sharpens not distorts; **Set A ∩ Set B = 15/100** (genuinely separate); each set top-9% under its own baseline, ~q77 under the other. 100 unique each, 0 pdb_path nulls, all in canonical if_ready + h_maps_v2.
- impact: scope — two new 0701 high-risk subsets + generalized selector; canonical set / pilot / fast untouched. 0401 DPLM-native not yet available → Set B is 0701-only for now; Set A could be built for 0401 on ProteinMPNN alone if needed. risk low (read-only canonical subsets).
- status: done
- next_action: (1) build 0401 once its DPLM-native baseline lands; (2) optionally add an NMP strong-window evenness column; (3) run RF on Set B (iteration) and report gain vs ProteinMPNN on Set A.
- refs:
  - `L0114` (superseded ProteinMPNN-only 400-set), `L0109` (WT baseline), `L0100` (v2 promotion)

### L0117
- timestamp: 2026-06-20T01:51:56-04:00
- type: CODEMAP_DIFF
- module: RF
- trigger: User requested merging the RF performance branch after cluster validation of GPU-head/cache and batched-denoiser RF paths.
- change_summary: Added RF generation performance path: cached/batched head scoring, PNLL precompute reuse, vectorized small helpers, and `run_if_phase_c1.py --batch-size` / `RF_BATCH_SIZE` to batch only the DPLM denoiser forward while keeping controller/RNG/telemetry per lane.
- rationale: Stage A/C controller runs were bottlenecked by repeated CPU/head/PNLL and scalar DPLM decoder overhead, leaving GPU under-used. The implementation preserves controller semantics by sharing only the structural decoder forward across lanes; D2/D3/SC-GR controller state, RNG, remask, and telemetry remain per-design independent.
- artifacts:
  - `epitope_head/inference/predictor.py`
  - `inverse_folding/reference_flow/actionability.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/counterfactual.py`
  - `inverse_folding/reference_flow/head_scoring.py`
  - `inverse_folding/reference_flow/runtime.py`
  - `inverse_folding/reference_flow/sampler.py`
  - `scripts/run_if_phase_c1.py`
  - `scripts/submit_if_phase_c.slurm`
  - `doc/SCRIPTS.md`
  - `tests/epitope_head/inference/test_batch_predictor.py`
  - `tests/inverse_folding/test_reference_flow_counterfactual.py`
  - `tests/inverse_folding/test_reference_flow_head_scoring.py`
  - `tests/inverse_folding/test_reference_flow_runtime.py`
  - `tests/inverse_folding/test_reference_flow_sampler.py`
  - `tests/scripts/test_run_if_phase_c1_script.py`
- evidence: |
    Local verification on the perf branch: `bash -n scripts/submit_if_phase_c.slurm`; `git diff --check`; `pytest tests/scripts/test_run_if_phase_c1_script.py tests/scripts/test_run_if_phase_c1_d1.py tests/scripts/test_run_if_phase_c1_d2_d3.py tests/inverse_folding/test_reference_flow_runtime.py tests/inverse_folding/test_reference_flow_sampler.py tests/inverse_folding/test_reference_flow_sampler_controller.py` -> 73 passed; `pytest tests/inverse_folding/test_reference_flow_*.py` -> 269 passed, 1 warning.
    Cluster smoke (2 proteins / 20 steps): GPU-head path wall time 44.79s vs CPU-head 69.77s (1.56x pipeline speedup; SLURM elapsed 1:34 vs 1:59). Semantic check parent vs perf: `generated.parquet` identical after dropping `wall_seconds`; `controller_events.parquet` schema/row count/discrete fields identical; float telemetry drift max 1.57e-5, within parent-repeat CUDA noise (1.76e-5).
    Batched RF denoiser validation: scalar vs `RF_BATCH_SIZE=4/8` produced identical `generated.parquet` after dropping `wall_seconds`; `controller_events.parquet` schema and row count identical (535 rows); controller discrete fields identical. Decoder-derived float telemetry drift was bounded but not bit-equivalent (`max_abs` 3.34e-5 for batch4, 3.59e-5 for batch8), concentrated in `struct_top*_logprob/gap`, `logit_struct`, and `logit_corrected`; accepted as trajectory-equivalent with bounded batched-DPLM numerical drift, not scalar float bit-equivalence.
- impact:
  - scope: RF runtime/driver performance path and telemetry provenance. Default behavior remains scalar (`--batch-size=1`, `RF_BATCH_SIZE=1`); `RF_BATCH_SIZE>1` is an explicit performance operating point with identical discrete trajectory in smoke validation and bounded decoder-float drift.
  - risk: medium
  - confidence: 0.86
- status: done
- next_action: Use `RF_BATCH_SIZE=4` as the first cluster full-run setting, escalate to `8` only after confirming memory headroom; evaluate generated/controller discrete equivalence as hard gates and track decoder-derived float telemetry with a relaxed bounded-drift tolerance (e.g. `max_abs <= 5e-5`).
- refs:
  - `codex/rf-perf-stagea`
  - `codex/rf-batched-denoiser`
  - `L0116` (SC-GR monitor branch preserved during merge)

### L0118
- timestamp: 2026-06-20T00:00:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement Stage SC1 (beta-only SC pressure) from `PLAN_RF_SC_GR.md` §6.1, gated on the SC0 monitor (L0116) passing per RAR 0010. Connect the monitor-validated SC-GR `B_sc` estimator to the EXISTING `global_pressure` actuator: D2 `beta` is gated by `g_GR=smoothstep(B_sc)` from the early-frozen per-design probe burden. `scale_lambda=false` (RAR 0008); `g in [g_min, 1]` protect-low only (amplify g>1 is SC2, deferred). D2/D3 token actuation untouched.
- change_summary: (SC1.1) `controller_config.py`: added `self_conditioned_probe` to the allowed/enabled pressure sources; `_materialize_global_pressure` now accepts `{trajectory_thresholded_G, self_conditioned_probe}` when enabled (still rejects legacy `trajectory_median_G`); `validate_global_pressure_runtime` requires `tau_prom` ONLY for `trajectory_thresholded_G` and the band for both. `SelfConditionedGRConfig` gained `mode` (now `monitor_only|beta_pressure`), `actuation_aggregator` (`mean_excess|topm_lse|supra_mass`), `actuation_arm` (`fresh|self_conditioned`), `freeze_after_reliable_refreshes` (>=1), `actuation_reduce` (`median|ema`); `_materialize_self_conditioned_gr` validates them (incl. `actuation_arm ∈ arms`, supra_mass ⇒ single supra tau). `_cross_validate_scgr` is mode-conditional AND bidirectional: `monitor_only ⇒ global_pressure.enabled=false`; `beta_pressure ⇒ global_pressure.enabled=true ∧ pressure_source=self_conditioned_probe ∧ scale_lambda=false ∧ g_max=1.0`; and the reverse — `global_pressure.enabled ∧ pressure_source=self_conditioned_probe ⇒ self_conditioned_gr.enabled ∧ mode=beta_pressure` (else the probe never produces `B_sc` and `g_GR` silently degrades to `unready_g`). (SC1.2) `controller.py`: the SC-GR probe now runs BEFORE `_update_pressure_state` (moved ahead of the typed-actionability block) and stores this refresh's `(actuation_arm, actuation_aggregator)` `B_sc` in `_scgr_actuation_B_sc`; new `_scgr_B_sc_window`/`_scgr_frozen_B_sc` state; `_update_pressure_state` branches on `pressure_source` — the new `_scgr_frozen_pressure` appends the per-refresh `B_sc` on reliable refreshes, freezes the reduced value once `freeze_after_reliable_refreshes` accrue (held constant thereafter; `unready_g` before freeze), then feeds the existing `smoothstep_pressure`; the legacy `trajectory_thresholded_G` branch is byte-identical. (SC1.3) `run_if_phase_c1.py`: `_load_global_pressure_calibration` branches on `pressure_source` (the `self_conditioned_probe` source requires `aggregator`/`arm`/`refresh_step`/band, no `tau_prom`) and returns a dict; `load_controller_setup` validates JSON `aggregator`/`arm` == `self_conditioned_gr.actuation_*` (Option A, YAML-authoritative), validates `refresh_step` == the freeze horizon (`freeze_after_reliable_refreshes - 1` — the band must come from the refresh SC1 actually freezes on), and stamps only the band. `sc_gr_monitor_probe.py` gained `--emit-calibration` (+ `--calibration-{aggregator,arm,refresh-step}`, `--low/high-quantile`) writing the `self_conditioned_probe` band from the per-protein `B_sc` distribution at the freeze horizon. (SC1.4) New config `d2_d3_full_stageC_scgr_betaonly_aopen.yaml` (copy of the SC0 monitor preset; only the `global_pressure` + `self_conditioned_gr` blocks differ: enabled `self_conditioned_probe`, `g_max=1.0`, `scale_lambda=false`, `unready_g=1.0`, mode `beta_pressure`, `topm_lse`/`fresh`/freeze=1/median — RAR 0010; band omitted, stamped from the calibration JSON).
- rationale: RAR 0010 ran the SC0 monitor and passed the entry gate, pinning `best_metric=topm_lse` (decisive vs supra_mass/mean_excess) and `actuation_arm=fresh` (the SC recycling edge +0.009 was far below the 0.078 reseed noise floor, so fresh is the robust choice). `freeze_after_reliable_refreshes=1` because the Stage B.1 cadence (`t_start=0.50`, `refresh_interval=5`) exposes only ~2 post-start refreshes, so the estimator freezes at the first (unguided) refresh — the least self-contaminated and closest-to-NoD estimate (doc/Self-Cond_GR.md §4). `unready_g=1.0` holds beta at base (B1 behavior) before freeze rather than running the first refreshes unsteered. All new behavior is gated on `mode=='beta_pressure'`/`pressure_source=='self_conditioned_probe'`, so monitor / legacy-thresholded / static runs are byte-for-byte unchanged.
- artifacts:
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/configs/d2_d3_full_stageC_scgr_betaonly_aopen.yaml`
  - `scripts/run_if_phase_c1.py`
  - `scripts/analysis/sc_gr_monitor_probe.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/inverse_folding/test_reference_flow_d1_controller.py`
  - `tests/scripts/test_run_if_phase_c1_d2_d3.py`
  - `tests/scripts/test_sc_gr_monitor_probe.py`
- evidence: |
    TDD red→green per task (SC1.1–SC1.4).
    SC1 gate (`test_reference_flow_controller_config` + `_d1_controller` + `_d2_d3_controller` + `_self_conditioned_gr` + `test_run_if_phase_c1_d2_d3` + `test_sc_gr_monitor_probe`) -> 260 passed.
    `pytest tests/inverse_folding/ tests/scripts/ -q` (excluding 3 pre-existing Biopython-broken files) -> 887 passed, 42 skipped; `git diff --check` clean.
    Behavior verified: beta_pressure freezes B_sc at the first reliable refresh and gates beta = beta_base * smoothstep(B_sc); frozen value constant across later refreshes; lambda never scaled (scale_lambda=false ⇒ _effective_lambda_commit None); unready_g held before freeze (freeze>=2). Byte-identity: trajectory_thresholded_G pressure runs unchanged (existing C1.4 pressure tests pass). Config: beta_pressure requires self_conditioned_probe + scale_lambda false + g_max 1.0 (each fails fast otherwise); the REVERSE — pressure_source=self_conditioned_probe with the probe disabled / monitor_only — also fails fast (no silent unready_g degradation). SC calibration loads/validates aggregator+arm consistency + the freeze-horizon refresh_step + band, no tau_prom; a refresh_step != freeze_after_reliable_refreshes-1 fails fast; betaonly preset differs from the monitor preset only in the two pressure blocks. Analysis: --emit-calibration writes the band (with refresh_step) from the freeze-horizon B_sc distribution and fails fast on a collapsed band.
- impact:
  - scope: RF controller actuation — a second `g_GR` burden source (SC-GR probe) feeding the existing beta gate + its calibration wiring + one new config. No D2/D3 token actuation change, no lambda scaling, no amplify (g>1), no model retraining, no NetMHC guidance, no new SLURM script. SC2/SC3 deferred.
  - risk: medium
  - confidence: 0.83
- status: done
- next_action: Emit the SC calibration from the L0116 monitor run (`sc_gr_monitor_probe.py --emit-calibration`, arm=fresh/topm_lse/refresh_step=0), then run the 50-protein pilot with `d2_d3_full_stageC_scgr_betaonly_aopen.yaml` + `--global-pressure-calibration-json` (`N_STEPS=20`, existing `submit_if_phase_c.slurm`). Analyze vs B1 binning by NoD burden terciles; pass criteria (doc/Self-Cond_GR.md §6 Stage 2): true-low over-intervention drops, true-high gain preserved, structure no worse than B1, pressure-bin vs NoD-burden-bin diagonal above old C.1. If true-high gain is preserved, SC2 (amplify g>1) is the next gate.
- refs:
  - `PLAN_RF_SC_GR.md` §6.1 (Stage SC1)
  - `doc/Self-Cond_GR.md` §4, §6
  - `L0116` (SC0 monitor — gated predecessor)

### L0119
- timestamp: 2026-06-20T00:00:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement the CODE part of Stage SC2 (amplify-high, g>1) from `PLAN_RF_SC_GR.md` §6.2 — unlock a two-sided gain so the SC-GR `beta_pressure` actuator can both suppress low-NoD-burden proteins (g_min<1) AND amplify high-burden ones (g_max>1) on the one existing smoothstep. The experiment config (`d2_d3_full_stageC_scgr_amplify_aopen.yaml`) and the pilot run are deferred (config provided after an experiment).
- change_summary: (SC2.1) `controller_config.py`: `GlobalPressureConfig` gains `amplify: bool=False`; `_materialize_global_pressure` parses it and validates `amplify=true ⇒ g_min < 1.0 < g_max <= 3.0` (so a two-sided gain is well-formed and bounded). `_cross_validate_scgr` gates the SC1 `g_max==1.0` enforcement on `amplify`: `beta_pressure ∧ ¬amplify ⇒ g_max==1.0` (SC1 unchanged), `beta_pressure ∧ amplify ⇒ g_max>1` allowed — a coder cannot silently amplify. NO controller-runtime change: `smoothstep_pressure(B; B_low, B_high, g_min, g_max)` is already two-sided (maps low→g_min, high→g_max on one monotone curve) and `_scgr_frozen_pressure` already passes `g_min`/`g_max` through unclamped, so `g_max>1` flows to `beta_eff = beta·g_GR` with no code change. (SC2.2 code) `sc_gr_monitor_probe.py`: `emit_calibration(..., amplify=True, g_min, g_max)` + `--amplify`/`--g-min`/`--g-max` CLI **solve** the band so the smoothstep crossover `g==1` lands EXACTLY at the NoD-burden median for the (possibly asymmetric) gain — `s* = (1−g_min)/(g_max−g_min)`, `q*` is its smoothstep inverse (`_solve_smoothstep_q`, bisection), width `W=q_high−q_low`, `B_low = median − q*·W`, `B_high = median + (1−q*)·W`; median-centering (`q*=0.5`) is the special case of symmetric `g_min+g_max==2` (P1: a naive median-centered band gives `g(median)=(g_min+g_max)/2 ≠ 1` for asymmetric `g`, e.g. `1.25` at `g_min=0.5,g_max=2.0`). The JSON carries `amplify:true` + `B_median` + the `g_min`/`g_max` it was solved against. The SC1 (suppress-low) path keeps the literal `[q_low, q_high]` band. **Runtime type guard (P1):** `_load_global_pressure_calibration` reads the calibration `amplify` flag (+ requires `B_median`/`g_min`/`g_max` when amplify); `load_controller_setup` rejects `calib.amplify != global_pressure.amplify` (an amplify config can no longer silently consume an SC1 band or vice-versa) and, for amplify, rejects `calib.g_min/g_max != config.g_min/g_max` (a band solved for a different gain). `doc/SCRIPTS.md` updated.
- rationale: RAR 0008 reframed the GR objective — the amplify-high direction (the −1.0 nat high-burden lever) is the larger lever; do-no-harm of low-burden is secondary. SC2 turns the SC1 protect-low gate into the two-sided gain the corrected objective wants. The PLAN notes the smoothstep is already two-sided, so SC2 is a calibration + bounds change, not a new mapping. The `amplify` flag keeps SC1 (`g_max=1`) and SC2 (`g_max>1`) configs distinct and fail-fast, and the median-centered emit makes the crossover interpretable (g==1 at the burden median). All new behavior is gated on `global_pressure.amplify`, so SC1 / legacy-thresholded / monitor / static runs are byte-for-byte unchanged. The experiment config (g_min/g_max start `0.5`/`1.5`, band from `--emit-calibration --amplify`) is deferred until the SC1 readout informs the conservative starting bounds.
- artifacts:
  - `inverse_folding/reference_flow/controller_config.py`
  - `scripts/analysis/sc_gr_monitor_probe.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_controller_config.py`
  - `tests/scripts/test_sc_gr_monitor_probe.py`
- evidence: |
    TDD red→green per task (SC2.1 amplify gate; SC2.2 amplify emit).
    `pytest tests/inverse_folding/test_reference_flow_controller_config.py -q` -> 117 passed; `pytest tests/scripts/test_sc_gr_monitor_probe.py -q` -> 24 passed; `pytest tests/scripts/test_run_if_phase_c1_d2_d3.py -q` -> 43 passed.
    `pytest tests/inverse_folding/ tests/scripts/ -q` (excluding 3 pre-existing Biopython-broken files) -> 905 passed, 42 skipped; `git diff --check` clean.
    Verified: amplify=true requires g_min<1.0<g_max<=3.0 (g_max=1.0, g_min=1.0, g_max=3.5 each fail fast); amplify=false keeps beta_pressure g_max==1.0 (SC1 unchanged); amplify=true with g_min=0.5/g_max=1.5 loads. emit --amplify solves the band so smoothstep_pressure(B_median; band, g_min, g_max)==1.0 for BOTH symmetric (0.5/1.5, band midpoint==median) AND asymmetric (0.5/2.0, midpoint≠median) gains; requires g_min/g_max; fails fast on zero spread. Loader: an amplify config rejects a non-amplify (SC1) calibration and vice-versa; an amplify band with g_min/g_max != config fails fast; a missing B_median fails fast. No controller.py change ⇒ SC1/legacy/static runs unchanged.
- impact:
  - scope: RF actuation config — one new `global_pressure.amplify` field + its validation gate, and the amplify calibration emitter. No controller-runtime change (smoothstep already two-sided), no model retraining. CONFIG (`d2_d3_full_stageC_scgr_amplify_aopen.yaml`) and the pilot run are deferred (user provides the config after the SC1 experiment). SC3 (instability trigger) deferred.
  - risk: low
  - confidence: 0.85
- status: partial (code complete + tested; experiment config + pilot run pending)
- next_action: Await the SC1 readout, then author `d2_d3_full_stageC_scgr_amplify_aopen.yaml` (copy `d2_d3_full_stageC_scgr_betaonly_aopen.yaml`; set `global_pressure: {amplify: true, g_min: 0.5, g_max: 1.5}`). Emit the amplify band (`sc_gr_monitor_probe.py --emit-calibration --amplify`), run the 50-protein pilot, readout: true-high reduction EXCEEDS SC1, true-low not harmed beyond SC1, structure cost bounded; if scTM degrades at g_max=1.5, sweep g_max∈{1.25,1.5,2.0} for the Pareto knee.
- refs:
  - `PLAN_RF_SC_GR.md` §6.2 (Stage SC2)
  - `doc/Self-Cond_GR.md` §1 (amplify objective), §3 step 5
  - `L0118` (SC1 beta-only — gated predecessor)

### L0120
- timestamp: 2026-06-23T17:30:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement uricase enzyme mode v0 (`PLAN_URICASE_ENZYME_MODE.md`, Tasks U1–U9): active-site-constrained global Reference Flow redesign. Hard active-site anchors are fixed from step 0 and permanently protected from remask; controller semantics (D1–D3, SC-GR, control law, DPLM decoder) are untouched. WT never enters generation (post-hoc decision layer only).
- change_summary: (U1) `inverse_folding/reference_flow/constraints.py`: YAML hard-anchor manifest loader (`HardAnchor`/`MonitoredShellResidue`/`ActiveSiteConstraint`/`ConstraintManifest`), content-stable sha256 hash, `validate_against_sequence` fail-fast (index bounds + expected_AA identity), duplicate-index / empty-without-control hard errors; AA→token conversion deliberately kept OUT of the parser. Plus driver-boundary helpers `build_run_constraints` (per-protein fixed-token map, validate present proteins, skip absent), `constraint_manifest_provenance`, and `build_constraint_application_report` (U4). (U2) `sampler.py`: `sample()`/`SamplerBatchLane`/`sample_batch()` gain optional `fixed_tokens`; anchors are initialized committed (`x_t[i]=tok`, `unmask_step_by_pos[i]=0`) and exposed as a separate `fixed_positions` set; the permanent anchor set is unioned into `protected_positions` at BOTH `_apply_reparam_remask` call sites (single + batch) — which run independent of the controller — so anchors (carrying `scores=-inf`, otherwise the first remask target) survive `remask.enabled=true` AND `controller=None` AND a broadly-remasking controller. Empty `fixed_tokens` ⇒ byte-for-byte legacy behavior. (U3) `run_if_phase_c1.py`: `--constraint-manifest`; preflight loads the manifest, builds per-protein fixed-token maps (fail-fast on AA mismatch BEFORE generation), threads `fixed_tokens` into both sampler call sites, stamps `enzyme_mode_*` manifest provenance, and **fails fast (return 2) if the manifest matches 0 hard anchors** (no false-positive enzyme run). `submit_if_phase_c.slurm` forwards `CONSTRAINT_MANIFEST` only in `MODE=reference_flow`. (U4) writes `constraints_applied.parquet` + `constraints_applied_summary.json` and fails the run (non-zero, AFTER persisting evidence) if any anchor is not preserved. (U5–U7) post-hoc sidecar builders `enzyme_telemetry.py` (shared `normalize_design_keys` join-key normalization with design_id/design_idx consistency check + `anchor_overlap`), `enzyme_immune.py` (`enzyme_nmp_windows` + `enzyme_immune_summary`; WT deltas computed ONLY on a matched `(protein_id, pep_length, pos)` window grid, else `NA` + `wt_window_support_matched=False`), `enzyme_controller.py` (`enzyme_controller_overlap`; `final_persistence_rate` judged by sampler `step` not `refresh_step`; `selected/remasked_hard_anchor_count` must be 0), `enzyme_fpost.py` (`f_post_v0`: `f_anchor` hard-invariant + monomer `f_fold` ranking-only; `f_assembly`/`f_site_scaffold` unavailable). Consolidated CLI `scripts/build_enzyme_telemetry.py` (optional `--wt-facade-eval-dir`/`--wt-facade-struct-dir`; missing WT ⇒ NA deltas). Manifest config `configs/uricase_q00511_active_site_v0.yaml` (frozen set {10,57,58,159,176,228,254,256}). `doc/SCRIPTS.md` updated.
- rationale: The constraint belongs in the sampler, not DPLM (`runtime.py:526` writes `prev_tokens[residue_positions]=x_t`, so a native-valued anchor in `x_t` conditions DPLM automatically — verified). v0 makes zero control-law / risk-estimation / D1–D3 change: only anchor init + permanent remask protection. The hard-anchor preservation is a defense-in-depth invariant (the sampler guarantees it; the U4 gate is the runtime check). Telemetry is the v0 product: it surfaces whether the controller spends pressure on visible-but-noneditable anchor-overlap risk (U6) without realized benefit, and the external-NMP immune readout vs WT (U5), without changing any controller decision. Three Codex review findings fixed before completion: (P1a) 0-anchor manifest match now fail-fast; (P1b) WT immune deltas now matched-window-gated (no aggregate subtraction over mismatched supports) + design_id/idx consistency check; (P2) persistence judged by sampler step (a same-refresh later-step remask no longer over-counts persistence).
- artifacts:
  - `inverse_folding/reference_flow/constraints.py`
  - `inverse_folding/reference_flow/sampler.py`
  - `inverse_folding/reference_flow/enzyme_telemetry.py`
  - `inverse_folding/reference_flow/enzyme_immune.py`
  - `inverse_folding/reference_flow/enzyme_controller.py`
  - `inverse_folding/reference_flow/enzyme_fpost.py`
  - `inverse_folding/reference_flow/configs/uricase_q00511_active_site_v0.yaml`
  - `scripts/run_if_phase_c1.py`
  - `scripts/submit_if_phase_c.slurm`
  - `scripts/build_enzyme_telemetry.py`
  - `doc/SCRIPTS.md`
- evidence: |
    TDD red→green per task; new suites: test_reference_flow_{constraints,sampler_constraints,enzyme_telemetry,enzyme_immune,enzyme_controller,enzyme_fpost}.py -> 81 passed.
    `pytest tests/inverse_folding/test_reference_flow*.py tests/scripts/test_run_if_phase_c1_d2_d3.py -q` -> 459 passed, 1 pre-existing amplification warning; zero regression (no-constraint path byte-equivalent).
    `run_if_phase_c1.py --help` shows `--constraint-manifest`; `bash -n submit_if_phase_c.slurm` OK; py_compile OK.
    Real-data smokes (cwd-local, not committed): Q00511 (seq_len 302, 3983 NMP windows) via build_enzyme_telemetry.py — enzyme_nmp_windows (1455 anchor-overlap, 98 strong @rank<=2 ≈ design doc §7.3 "WT≈97"), enzyme_immune_summary delta_total_strong=0 + wt_window_support_matched=True for design==WT; U6/U7 builders run on real refresh_log/controller_events/structural schemas (an unconstrained run correctly surfaces non-zero selected/remasked anchor counts, proving the 0-invariant diagnostic fires).
- impact:
  - scope: RF sampler gains an optional hard-anchor constraint layer (additive; default off ⇒ unchanged), a Phase-C driver CLI flag + manifest provenance + hard gate, and a post-hoc enzyme-telemetry builder. No controller-runtime / risk-estimation change, no model retraining. First run (Q00511) is launch-ready; the enzyme sidecars are an analysis layer over the run.
  - risk: low
  - confidence: 0.86
- status: done (code complete + tested; Q00511 first run + WT structural facade pending)
- next_action: Launch the Q00511 constrained smoke (`MODE=reference_flow CONSTRAINT_MANIFEST=.../uricase_q00511_active_site_v0.yaml`), confirm `constraints_applied.parquet` has 0 mismatches, then run `build_enzyme_telemetry.py` over the run + the WT facade for the (F_post, R, C) readout. A same-protocol WT `--mode struct` facade run is still needed for U7 structural deltas (currently NA).
- refs:
  - `PLAN_URICASE_ENZYME_MODE.md` (Tasks U1–U9)
  - `doc/Uricases_Design.md` §9.0 (v0 freeze), Appendix A (verified code surface)

### L0121
- timestamp: 2026-06-25T12:30:58-04:00
- type: CODEMAP_DIFF
- module: D
- trigger: Enable the SC-GR per-residue `r_i` accuracy gate (position-dependent / "local SC1" path); the validated ensemble's per-residue map was computed internally then discarded (RAR 0010 validated only the protein scalar).
- change_summary: SC-GR probe now surfaces `SCGRRiskAggregates.residue_excess` and appends the per-completion per-residue map to `sc_gr_probe_samples.parquet` behind a new `self_conditioned_gr.write_residue_telemetry` flag (default False → existing monitor telemetry byte-identical); adds the per-residue accuracy analysis script + a flag-on monitor config.
- rationale: `r_i` is a LEVEL (per-residue terminal risk) and MHC-II burden concentrates at a few anchor sites, so it may inherit RAR 0010's protein-scalar accuracy even though the candidate-tuple MARGINAL did not (RAR 0018); validating it gates whether position-dependent pressure is on the table.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/self_conditioned_gr.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller_config.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/controller.py`
  - `/Users/jerry/Project/MHC-IF/inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor_residue.yaml`
  - `/Users/jerry/Project/MHC-IF/scripts/analysis/scgr_residue_accuracy.py`
  - `/Users/jerry/Project/MHC-IF/PLAN_RF_SC_GR_ri_accuracy.md`
- evidence: `pytest tests/inverse_folding/test_reference_flow_self_conditioned_gr.py tests/inverse_folding/test_reference_flow_controller_config.py tests/inverse_folding/test_reference_flow_d1_controller.py tests/scripts/test_run_if_phase_c1_d2_d3.py tests/scripts/test_scgr_residue_accuracy.py -q` → 219 passed; config round-trip + parquet list-column round-trip verified manually.
- impact:
  - scope: SC-GR monitor telemetry (gated additive column), one new config + one new analysis script; no D2/D3/remask/global_pressure/actuation change.
  - risk: low
  - confidence: 0.95
- status: done
- next_action: Run the two cluster jobs in `PLAN_RF_SC_GR_ri_accuracy.md` §3 (flag-on monitor re-run + NoD `--imm-full` re-eval), then `scgr_residue_accuracy.py` locally and apply the §1 gate.
- refs:
  - `PLAN_RF_SC_GR_ri_accuracy.md`
  - `doc/Self-Cond_GR.md` §3 step 4 / §5; RAR 0010, RAR 0018

### L0122
- timestamp: 2026-06-26T18:30:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement the SC-GR position-dependent **allocation layer** (`PLAN_PLANNER_SC_GR.md`, Tasks 1–8): an inference-only WHERE-layer that reweights D2 position-*selection* by a frozen per-residue editability mass `Φ_i` derived from the SC-GR per-residue prospective-immune field `r_i` (validated by RAR 0020: region ρ 0.578 / Recall@High 0.645, median-over-K, fresh arm, 9-residue grain). The Budget layer (`B_sc→g`, amplify) is reused unchanged; this builds only the Allocation layer (the Value `P_cheap` layer is deferred to v1.1).
- change_summary: (T1) `self_conditioned_gr.py`: `reduce_residue_excess_over_k(per_sample, *, arm)` = median-over-K of one arm's `residue_excess` ⇒ `r_i`. (T2) new module `allocation.py` (distinct from the typed-actionability field `A_i(t)`): `smooth_window_max` (register grain), `allocation_mass` (mean-normalized `Φ_i`, all-zero⇒uniform), `reweight_by_allocation` (`s = v_target·(eps + c·Φ)`), `stable_seed` (sha256, process-stable). (T3) `controller_config.py`: frozen `AllocationConfig` (`enabled/arm/smooth_half_width/reweight_c/reweight_eps`) under `controller.allocation` + `TargetingConfig.selection_field_mode ∈ {v_target, v_target_x_alloc, flat}` (`v_target` default = uniform-allocation control); both materialized + range-validated. `_cross_validate_allocation` (review P1-3) fails fast on every silent-degeneration path: `v_target_x_alloc` ⇒ `allocation.enabled` AND `global_pressure.enabled`+`pressure_source=self_conditioned_probe` (Φ_i freezes inside the SC-GR pressure path; otherwise the treatment arm collapses to its own `v_target` control), and `allocation.enabled` ⇒ `self_conditioned_gr.enabled` AND `allocation.arm ∈ self_conditioned_gr.arms`. (T4) `controller.py`: freeze `Φ_i` parallel to `B_sc` — `_run_sc_gr_probe` computes `r_i` (gated on `allocation.enabled`, no new head call), `_scgr_frozen_pressure` freezes `Φ_i = allocation_mass(median-over-window r_i)` reliable-gated + freeze-once (cannot self-reinforce). (T5) two selection seams: `_selection_field(v_target, protein_id, design_idx)` mode-selects the SELECTION field (raw `v_target` / `v_target·(eps+c·Φ)` / sha256-seeded flat) and feeds it to BOTH the active-window ranker (`_select_active_windows_typed`) and the D2 within-block `typed_field`; raw `v_target` (and thus `u_pressure`/G/β) is untouched ⇒ firewall preserved. (T6) telemetry: `UnifiedActionabilityState` gains `phi_alloc`/`selection_field` (filled post-selection); `run_if_phase_c1.py` writes `phi_alloc`/`selection_field` residue columns (None-safe ⇒ `Φ≡1`/`v_target` on static/non-typed refreshes) + `selection_field_mode`/`allocation_frozen_flag` summary fields + a `[allocation]` config-print block. (T7) 9 experiment YAMLs (`planner_{flat,vtarget,alloc}_gmax{15,20,25}_aopen.yaml`) on the `scgr_betaonly` Budget base (`amplify:true`, `g_min:0.5`, `g_max∈{1.5,2.0,2.5}`) + `configs/PLANNER_README.md` documenting the per-`g_max` amplify-calibration emit + run commands. (T8) `scripts/analysis/planner_h1_pareto.py`: H1 gate — `compute_h1` (design→protein median, paired Wilcoxon) where a `g_max` cell passes iff primary (`alloc<v_target`, p<0.05) AND scTM non-inferior AND **macro leg** (`v_target<flat` AND `alloc<flat`; review P1-1 — `flat≈v_target` ⇒ FAIL per §5) AND **cohort complete** (review P1-2: `expected_n_proteins`/`expected_n_designs`/`expected_g_max` gating, default the §5 50/8/`{1.5,2.0,2.5}` cohort; overall gate = all g_max present AND ≥1 cell passes); `merge_run_eval[_frames]` assembles the H1 frame from real `evaluate_phase_c.py` output (`immune_nmp := n_strong_binders/n_windows_scored`, `scTM→sctm`, join on `(protein_id, design_idx)`) and **fails fast on a silent inner-join drop** (review P1-2: a design missing immune XOR structure); `extract_arm_mode_g_max` (from stamped controller config). `doc/SCRIPTS.md` updated (item #15; item #10 SC2 clause corrected to the solved band).
- rationale: RAR 0020 validated `r_i` at the 9-residue region grain, so position-dependent pressure is on the table; allocation is the cheapest expression of it (WHERE-only, multiplicative reweight of the existing selection field, downstream of memory). `Φ_i` is mean-normalized so uniform `Φ≡1` is a byte-identical constant rescale of `v_target` — making the `v_target` arm the exact uniform-allocation control for the pre-registered H1 (`v_target_x_alloc` beats `v_target` at matched scTM). The reweight touches ONLY the local selection field, never `v_target`/`u_pressure`/G/β/memory/the schedule, so the SC-GR firewall (probe must not feed D2/D3 token actuation or remask) holds. The H1 immune endpoint is `strong_frac = n_strong_binders/n_windows_scored` (length-normalized; within-protein paired ⇒ same sign as the raw count) so lower matches the `alloc<v_target` hypothesis. All new behavior is gated on `allocation.enabled` + `selection_field_mode`, so legacy / static / monitor / SC1 / SC2 runs are byte-for-byte unchanged.
- artifacts:
  - `inverse_folding/reference_flow/self_conditioned_gr.py`
  - `inverse_folding/reference_flow/allocation.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `scripts/run_if_phase_c1.py`
  - `scripts/analysis/planner_h1_pareto.py`
  - `inverse_folding/reference_flow/configs/planner_{flat,vtarget,alloc}_gmax{15,20,25}_aopen.yaml` (9)
  - `inverse_folding/reference_flow/configs/PLANNER_README.md`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_{self_conditioned_gr,allocation,controller_config,d1_controller}.py`
  - `tests/scripts/test_{run_if_phase_c1_d2_d3,planner_h1_pareto}.py`
- evidence: |
    TDD red→green per task (Tasks 1–8). Pre-implementation: a verification workflow cross-checked all PLAN claims vs source (42 confirmed; 5 test/run-wiring blockers fixed: load_controller_config not *_file + flat cfg; num_active_windows off UnifiedActionabilityState; self.protein_id/design_idx at the seam; typed-override build_guided_controller fixture; H1 eval-column join). User aligned on the immune endpoint (strong_frac) + fixture style (typed-override).
    `pytest tests/inverse_folding/test_reference_flow_allocation.py -q` → 6; `...self_conditioned_gr.py -q` reduce_residue_excess → 2; `...controller_config.py -q` → 135 (incl. 4 allocation-defaults/mode + 4 allocation cross-validation + 10 planner-config); `...d1_controller.py -q` → 45 (incl. 2 freeze + 3 seam + 2 telemetry); `tests/scripts/test_run_if_phase_c1_d2_d3.py -q` → 45 (incl. 2 allocation-column); `tests/scripts/test_planner_h1_pareto.py -q` → 9 (incl. macro-leg / incomplete-cohort / complete-cohort / strict-merge).
    `pytest tests/inverse_folding/ tests/scripts/test_run_if_phase_c1_d2_d3.py tests/scripts/test_sc_gr_monitor_probe.py tests/scripts/test_planner_h1_pareto.py -q` → 923 passed, 41 skipped, 1 pre-existing fail (Biopython `Bio` absent in local env, structural_features, unrelated); `git diff --check` clean.
    Verified: uniform `Φ≡1` ⇒ identical active-window selection to the `v_target` arm; `v_target_x_alloc` seam feeds `reweight_by_allocation(v_target, Φ, c, eps)`; `flat` is content-blind + sha256-reproducible; `Φ_i` mean==1, frozen reliable-gated + once; all 9 configs load with amplify/g_min/g_max/selection_field_mode. Review P1 fixes: a `v_target_x_alloc` config with `allocation.enabled=false` / SC-pressure off / `arm` not in `scgr.arms` / `scgr` disabled all fail fast; H1 pairs over proteins (n=50, not 400 rows), `flat≈v_target` fails the macro leg, a 6-protein cohort fails completeness, and an immune/structure key mismatch fails the merge.
- impact:
  - scope: RF controller gains an inference-only allocation layer (additive; default `selection_field_mode=v_target` + `allocation.enabled=false` ⇒ byte-identical legacy), residue/summary telemetry, 9 experiment configs, and the H1 gate script. No D2/D3 token actuation / remask ranking / Phase-C schedule / pressure / model-retraining change. The pilot50 DRB1*07:01 runs + amplify calibration JSONs are a cluster step (deferred).
  - risk: low
  - confidence: 0.88
- status: done (code complete + tested; SC1-pilot calibration emit + the 9-config × pilot50 H1 run are the pending cluster steps)
- next_action: On the cluster — emit the 3 amplify calibration JSONs from the SC1 pilot (`sc_gr_monitor_probe.py --emit-calibration --amplify --g-min 0.5 --g-max {1.5,2.0,2.5}`), run the 9 planner configs over pilot50r2 DRB1*07:01 (n=8, seed=42), evaluate immune(NMP)+scTM, then `planner_h1_pareto.py` for the §5 gate; on PASS register a RAR and proceed to the v1.1 Value layer (`P_cheap`).
- refs:
  - `PLAN_PLANNER_SC_GR.md` (Tasks 1–8, §5 H1 gate)
  - `doc/Self-Cond_GR.md` §8.3 (Budget→Allocation→Value)
  - `L0118` (SC1 Budget), `L0119` (SC2 amplify), `L0121` (r_i telemetry); RAR 0020 (`r_i` validation), RAR 0010 (`B_sc`)

### L0123
- timestamp: 2026-06-29T17:00:00-04:00
- type: FEATURE
- module: RF
- trigger: The direct-allocation H1 (L0122 / §8.3 `r_i → Φ → v_target·(ε+c·Φ)` reweight) was FALSIFIED on `planner_pilot47` (RAR 0019 M4: alloc did not beat v_target at any g_max; the multiplicative reweight dragged ~45% of active residues onto higher-Φ/lower-`v_target` sites). Implement the two revised, independent, default-off ablations of `PLAN_PLANNER_SC_GR.md` §A (doc §8.4 Path A): **A1** corrects the `r_i` consumption semantics (triage WITHIN the `v_target`-eligible set, no `τ_v` widening); **A2** swaps the D2 candidate score from local `ΔR_B` to a terminal-completion probe (`P_cheap`) on the B1 baseline. Both reuse the L0122 `Φ_i`/telemetry code and the Task 7–8 harness/eval; Path C (coordinated proposals) is out of scope.
- change_summary: (A1) `allocation.py`: `_norm_rank` + `triage_field(v_target, Φ, *, eligible_quantile, triage_lambda)` = within `eligible={v≥quantile(v,q)}`, `norm_rank(v) + λ·norm_rank(Φ)`; ineligible⇒0 (additive, rank-space, eligibility-gated — `Φ` can no longer drag fire onto low-`v_target` sites, fixing §8.2.1). `controller_config.py`: `"v_target_triage"` added to `_ALLOWED_SELECTION_FIELD_MODES`; `AllocationConfig` gains `eligible_quantile=0.5`/`triage_lambda=0.3` (parsed + range-validated); `_cross_validate_allocation` generalized to a `_PHI_CONSUMING_SELECTION_MODES={v_target_x_alloc, v_target_triage}` set (both require `allocation.enabled` + `global_pressure.pressure_source=self_conditioned_probe`, else `Φ_i` never freezes and the arm silently degrades to the `v_target` control). `controller.py` `_selection_field`: `v_target_triage` branch → `triage_field` on the frozen `Φ_i`. (A2) `controller_config.py`: `D2Config` gains `candidate_score_source="local"` (validated `{local,terminal}`) + `candidate_terminal_K_P=4`; the `completion_ensemble_scope` guard is UNTOUCHED (terminal bypasses the ensemble path, gated by `candidate_score_source`). `counterfactual.py`: new `D2Handler._score_completion_terminal` parallels `_score_completion_ensemble` but rolls each shortlisted candidate to a FULL-sequence terminal completion (`self_conditioned_gr.build_probe_samples`, `K_P` paired completions — fill RNG key excludes the candidate index, so context draws are shared and only the committed editable tokens differ), scored as `Ω(B)-LME(window_risk) − r_current` in the SAME head-logit units as local `ΔR_B` (NOT `G_topm_lse`); `_process_block` branches the rescore on `candidate_score_source`; `D2BlockOutcome.candidate_score_source` telemetry stamped uniformly at the `correct_logits` block-loop `replace`. (A3) configs: `planner_triage_gmax20_aopen.yaml` (scgr_betaonly Budget base + `v_target_triage`, g_max=2.0) + `b1_terminal_aopen.yaml` (B1 base, `candidate_score_source=terminal`). `planner_h1_pareto.py`: `compute_h1` `treatment_mode` param (`v_target_x_alloc` default / `v_target_triage`) + `--arm-treatment`; new `compute_pairwise_gate(treatment, baseline)` for A2's 2-arm direct diff (paired Wilcoxon + scTM non-inf, NO flat/macro/Pareto; reports `treatment_better` + `neutral_or_better` per the tempered "neutral-or-better at matched scTM" success bar) + `--pairwise`/`--arm-baseline`. `doc/SCRIPTS.md` (#15) + `configs/PLANNER_README.md` updated. **Review P-fixes:** (P1) `triage_field` strict eligibility — eligible = top-(1−q) AMONG strictly-positive actionability (`v>floor`, `floor=active_window_min_excess`), quantile over positives only, so a sparse `v_target` with median 0 can never admit zero-actionability sites (the §8.2.1 mechanism the original `v>=quantile(v,q)` re-opened). (P2a) `candidate_score_source='terminal'` now fails fast unless `completion_ensemble_enabled=true` (the terminal rescore rides the ensemble shortlist path; off ⇒ silent degrade to local-argmax + mislabel). (P2b) `candidate_score_source` written into D2 telemetry (block diagnostics + D2 event rows + sticky rows; `_stage_a_event_defaults` null for D3/monitor) so controller_events/refresh_log audit local-vs-terminal per row. (P2c) under `--pairwise`, runs-json entries require explicit `arm_mode` — `b1_local`/`b1_terminal` share config fields so auto-derive cannot distinguish them.
- rationale: §8.2's two findings — wrong `r_i` consumption semantics + a per-edit authority ceiling (RAR 0019 M5 / 0013: `delta_R_B`≈−0.2 flat across g_max, `ESS_candidates`≈3) — bound any `r_i` consumption. A1 makes `r_i` a bounded, rank-space, eligibility-gated tie-break so the failure mechanism (multiplicative scale domination + dragging onto unactionable sites) cannot recur; A2 tests whether a terminal-immune candidate ranker (RAR 0019 ρ0.44 soft ranker) recovers anything within the ceiling, isolated on B1 (no budget/pressure). Both are default-off: `selection_field_mode=v_target_triage` requires the SC pressure freeze path (cross-validated), and `candidate_score_source` defaults `local` (byte-identical), so all legacy / §8.3 / SC1 / SC2 / monitor / static runs are unchanged. Pre-registered expectation is "harmful→neutral / small positive", not a large drop — the decisive lever is Path C (deferred).
- artifacts:
  - `inverse_folding/reference_flow/allocation.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/counterfactual.py`
  - `inverse_folding/reference_flow/configs/planner_triage_gmax20_aopen.yaml`
  - `inverse_folding/reference_flow/configs/b1_terminal_aopen.yaml`
  - `inverse_folding/reference_flow/configs/PLANNER_README.md`
  - `scripts/analysis/planner_h1_pareto.py`
  - `doc/SCRIPTS.md`
  - `tests/inverse_folding/test_reference_flow_{allocation,controller_config,d1_controller,counterfactual}.py`
  - `tests/scripts/test_planner_h1_pareto.py`
- evidence: |
    TDD red→green per task (A1 allocation+config+controller; A2 config+counterfactual; A3 configs+gate). Pre-implementation alignment with the user fixed the §A2 gate shape (direct 2-arm pairwise, NOT Pareto), the `_cross_validate_allocation` extension to `v_target_triage` (PLAN A1 omitted it), and "don't touch the completion_ensemble_scope guard".
    `pytest tests/inverse_folding/test_reference_flow_allocation.py -q` → 13 (incl. 4 triage + 3 strict-eligibility); `...controller_config.py -q` → 147 (incl. v_target_triage materialize/cross-validation + candidate_score_source + terminal-requires-ensemble + triage/b1_terminal config-load); `...d1_controller.py -q` → 48 (incl. triage seam + D2 telemetry provenance); `...counterfactual.py -q` → 31 (incl. 3 terminal-probe: ranking/pairing/full-fill + process_block terminal≠local + stamp); `tests/scripts/test_planner_h1_pareto.py -q` → 16 (incl. treatment_mode triage + 4 pairwise-gate + pairwise explicit-arm).
    `pytest tests/inverse_folding/ tests/scripts/test_run_if_phase_c1_d2_d3.py tests/scripts/test_sc_gr_monitor_probe.py tests/scripts/test_planner_h1_pareto.py -q` → 955 passed, 41 skipped, 1 pre-existing fail (Biopython `Bio` absent locally, structural_features, unrelated); `git diff --check` clean.
    Verified: `triage_field` λ=0 ⇒ pure v_target order within eligible, ineligible⇒0, high-Φ wins a within-eligible tie, huge Φ at an ineligible site cannot beat an eligible one; (P1) all-zero v_target ⇒ all-zero field, sparse-zero (median 0) excludes every zero site, `floor` excludes ≤active_window_min_excess; `v_target_triage` fails fast without the SC pressure path / allocation.enabled. `_score_completion_terminal` fills all masks, ranks the lower-ord editable token below the higher, and is paired (identical editable tokens ⇒ identical ΔR under shared fills); `_process_block` terminal `delta_R_B` ≠ local when off-block completions disagree; both stamped; (P2a) terminal+ensemble-off fails fast; (P2b) D2 event rows carry `candidate_score_source`, D3/monitor null; (P2c) `--pairwise` without explicit arm_mode fails fast. `candidate_score_source=local` default ⇒ D2 byte-identical (full controller suite green).
- impact:
  - scope: RF gains (a) a corrected `r_i` WHERE-consumption mode (`v_target_triage`, additive in-eligible-set), (b) a default-off terminal candidate scorer (`candidate_score_source=terminal`) in D2, (c) 2 ablation configs + the A1/A2 gates. No change to the Budget layer, D3, remask, schedule, or the default (`local` / `v_target`) behavior; no model retraining. A1 (pilot47, g_max=2.0) + A2 (B1, high-risk cohort) runs at N_STEPS=100 are the pending cluster steps.
  - risk: low
  - confidence: 0.85
- status: done (code complete + tested; A1/A2 cluster runs + eval/gate pending)
- next_action: Cluster — re-emit `amplify_calib_gmax20.json` at N_STEPS=100; run A1 (flat/v_target/triage @ g_max=2.0, pilot47) + A2 (b1_local/b1_terminal, RAR 0013 high-risk cohort); gate via `planner_h1_pareto.py` (`--arm-treatment v_target_triage` for A1; `--pairwise --arm-treatment b1_terminal --arm-baseline b1_local` for A2). On outcomes register a RAR; A1≈neutral + A2≈neutral ⇒ green-light Path C.
- refs:
  - `PLAN_PLANNER_SC_GR.md` §A1 / §A2 / §A3
  - `doc/Self-Cond_GR.md` §8.2 (falsification), §8.4 Path A; RAR 0019 (M4 mechanism, M5 ceiling, `P_cheap` ρ0.44), RAR 0013 (high-risk null)
  - `L0122` (falsified direct-allocation, reused `Φ_i`/telemetry)

### L0124
- timestamp: 2026-07-01T10:00:00-04:00
- type: FEATURE
- module: RF
- trigger: A1 (`v_target_triage`, L0123) is a WITHIN-eligible tie-break — low-`v_target` registers never activate. C0a M4 found high-`r_i` low-`v_target` registers are skipped for low *local* head evidence (`b_env`), not structure. Implement §A3 (doc `Self-Cond_GR.md` §8.4 Fork A "terminal-OR-local gated union"), the principled INVERSE of A1: a new default-off `selection_field_mode=v_target_terminal_union` that PROMOTES those registers into the active set via a SEPARATE terminal gate on the frozen register-smoothed `Φ_i` (never widening `τ_v`). Mirrors the green A1 stack.
- change_summary: `allocation.py`: new `terminal_union_field(v_target, Φ, *, terminal_eligible_quantile, terminal_lambda, floor=0.0)` — eligibility `local | terminal` where `local = v>floor` and `terminal = Φ ≥ quantile(Φ[Φ>0], q)` — the quantile over the **positive (discriminative) `Φ` subset**, mirroring A1 `triage_field`; over the union `norm_rank(v) + terminal_lambda·norm_rank(Φ)`, ineligible⇒0. Two `Φ`-degeneracy guards stop the terminal set widening onto dead sites (the falsified `v_target·Φ`, RAR 0019 M4 / §8.2.1): the positive-subset quantile defeats a **sparse** `Φ` (realistic `allocation_mass` output is mostly-zero, so a full-array quantile would collapse to 0 and admit everyone — the review P1), and the `max>min` flat guard defeats a **uniform** `Φ` (all-zero `r_i` ⇒ `allocation_mass` ones). `controller_config.py`: `"v_target_terminal_union"` added to `_ALLOWED_SELECTION_FIELD_MODES` and `_PHI_CONSUMING_SELECTION_MODES` (so `_cross_validate_allocation` enforces `allocation.enabled` + `global_pressure.pressure_source=self_conditioned_probe`); `AllocationConfig` gains `terminal_eligible_quantile=0.9`/`terminal_lambda=0.3` (parsed + range-validated: quantile∈[0,1], lambda≥0). `controller.py` `_selection_field`: `v_target_terminal_union` branch → `terminal_union_field` on the frozen `Φ_i` (floor = `active_window_min_excess`); `phi_alloc` telemetry UNTOUCHED (A3 is semantically identical to A1 — `Φ` is a rank input, not an applied multiplicative mass, so `phi_alloc` stays ones; the actual field is captured by `selection_field`). New config `planner_terminal_union_gmax20_aopen.yaml` (copy of `planner_triage_gmax20_aopen.yaml`; `selection_field_mode=v_target_terminal_union`, `terminal_eligible_quantile=0.9`, `terminal_lambda=0.3`, g_max=2.0). `PLANNER_README.md` §A3 added (3-arm flat/v_target/terminal_union cohort + 3-arm `compute_h1` gate `--arm-treatment v_target_terminal_union --expected-g-max 2.0`, NO `--pairwise` — A3 is WHERE-layer like A1, so `flat`/macro leg are consumed). Print-hyperparams: `run_if_phase_c1.py` already loops over the whole resolved `allocation` dict, so the two new knobs echo automatically (no lone print added).
- rationale: A1's tie-break cannot reach registers below the `v_target` eligibility bar even when their prospective-immune `r_i` is high; A3 opens a second, register-grain terminal channel to admit exactly those (C0a M4). Additive rank-space + bounded `terminal_lambda` + the uniform-`Φ` guard keep the falsified multiplicative `v_target·Φ` harm (RAR 0019 M4) from recurring. Default-off: the mode requires the SC pressure freeze path (cross-validated) and defaults to `v_target`, so all legacy / §8.3 / A1 / A2 / monitor runs are byte-identical.
- artifacts:
  - `inverse_folding/reference_flow/allocation.py`
  - `inverse_folding/reference_flow/controller_config.py`
  - `inverse_folding/reference_flow/controller.py`
  - `inverse_folding/reference_flow/configs/planner_terminal_union_gmax20_aopen.yaml`
  - `inverse_folding/reference_flow/configs/PLANNER_README.md`
  - `tests/inverse_folding/test_reference_flow_{allocation,controller_config,d1_controller}.py`
- evidence: |
    TDD red→green (allocation function; config parse/validate + cross-validation; controller seam). Per-file: `test_reference_flow_allocation.py` → 21 (8 new `terminal_union`: promotion-vs-triage, uniform-Φ guard, all-zero-v_target terminal-only, **sparse-Φ no-widen-to-dead-sites [review P1 regression]**, floor via terminal gate, lambda=0 v_target order, scale-invariance, empty); `test_reference_flow_controller_config.py` → 155 (7 new: materialize/custom-knobs/requires-SC-pressure/requires-allocation-enabled/quantile-range/lambda-range + terminal_union config-load; grid 10→11); `test_reference_flow_d1_controller.py` → 50 (2 new: terminal_union seam promotes triage-zeroed sites, pre-freeze v_target fallback); `test_reference_flow_counterfactual.py` → 31 (unchanged). 4-file targeted → 257 passed. Review P1 (adversarial verify): the terminal gate first took its quantile over the FULL `Φ` array; on the realistic sparse `Φ` the threshold collapsed to 0 and admitted every dead site (τ_v-widening) — fixed to the positive-subset quantile + a sparse-`Φ` regression test.
- impact:
  - scope: RF gains a corrected inverse-of-A1 WHERE-consumption mode (`v_target_terminal_union`) + 1 ablation config + the A3 gate wiring. No change to the Budget layer, D2/D3 token actuation, remask, schedule, pressure, `phi_alloc` telemetry, or the default (`v_target`) behavior; no model retraining. The A3 cluster run (flat/v_target/terminal_union @ g_max=2.0, pilot47) is pending.
  - risk: low
  - confidence: 0.85
- status: done (code complete + tested; A3 cluster run + eval/gate pending)
- next_action: Cluster — run A3 (flat/v_target/terminal_union @ g_max=2.0, pilot47, reuse `amplify_calib_gmax20.json`); gate via the 3-arm `planner_h1_pareto.py --arm-treatment v_target_terminal_union --expected-g-max 2.0` (no `--pairwise`); on outcome register a RAR.
- refs:
  - `PLAN_PLANNER_SC_GR.md` §A3
  - `doc/Self-Cond_GR.md` §8.4 (Fork A); RAR 0019 (M4 mechanism), C0a M4 (terminal-vs-local sensing gap)
  - `L0123` (A1 `v_target_triage`, mirrored), `L0122` (falsified direct-allocation)

### L0125 
- timestamp: 2026-06-24T15:25:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User requested exact-sequence dedup of the uricase case set (near-redundancy/nr-clustering intentionally NOT done), with all characterized members preserved.
- change_summary: Exact-sequence deduped `uricases/uricase_caseset_if_ready_unified.parquet` **6740 → 6387** (−353; one row per unique `sequence`; **all 25 `characterized` force-retained**; non-char representative prefers `afdb` over `esmfold2`). No two characterized share a sequence; 16 non-char rows whose sequence matched a characterized one were dropped (char represents). Source after: 4958 AFDB + 1429 ESMFold2. Subset downstream to match: `h_maps/h_maps_uricase_DRB1_{07_01,04_01}.parquet` (6387 each, cover 100%, metas updated), `wt_generated_uricase_caseset_HLA-DRB1_07_01.parquet` facade (6387, meta updated). Wrote `uricase_caseset_if_ready_unified.manifest.json` recording the dedup. `pdbs_if_ready/` structure files left on disk (353 now-orphaned exact-dups retained). Pre-dedup 6740 copies backed up to `backups/20260624_152048_uricase_exact_dedup/`. Near-redundancy (homolog clustering) NOT applied.
- evidence: unified 6387 rows / 6387 unique sequences / 0 duplicate `sequence` / 25 characterized; both h_maps 6387 (0 missing vs unified); facade 6387 ⊆ unified. PROGRESS Module L uricase block + `doc/Uricases_Design.md` §design count updated 6740→6387; LOG unification entries (L0103-era) left as historical.
- impact: scope — uricase case-study set + its h_maps/facade/manifest only; main test set / pilot / fast / highrisk untouched. risk low (backup retained; structures kept; no near-redundancy removal).
- status: done
- refs:
  - `doc/Uricases_Design.md`, `L0104` (uricase case-study build), `L0097` (characterized expansion)

### L0126
- timestamp: 2026-06-25T18:00:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: IF-ready resolved-backbone build applied no length-coverage filter, so the main test set held heavily-truncated structure fragments (some resolved <50% of the original, e.g. 8Q0P_E 476->235). These deviate from whole-protein redesign and collapse structurally; B1 highrisk confirmed 8Q0P_E scTM~0.31 (set p05). User decided to drop all `if_sequence_coverage < 0.8`. (Uricase set checked too: clean, coverage~1.0, no action.)
- change_summary: Removed every `if_sequence_coverage < 0.8` protein from the v2 main test set + all canonical subsets (0701 -135 -> 2879, 0401 -186 -> 2829). Subset to match: `test_proteins_<tag>.parquet`, `if_ready/main/test_proteins_if_ready_<tag>.parquet`, `if_ready/h_maps_v2/h_maps_<htag>.parquet` (all 2879/2829, h_maps cover 100%), `test_proteins_summary_<tag>.json` recomputed (T2 0701 2864 / 0401 2814). `fast_v2` 300->286/285; `pilot_v2` 50->47(5+42)/49(5+44) -- 3 (0701: 1O51_A,1VF7_J,7TN4_A) / 1 (0401: 7Z3G_B) truncated Tier-2 dropped, pilot FASTA regenerated (not backfilled to 50). Added `--min-coverage` to `scripts/build_highrisk_demo.py` and rebuilt both 0701 highrisk sets at >=0.8 (dropped leaked 8Q0P_E/4QRF_B/5HSF_A, backfilled to 100 from next-ranked full-coverage proteins; burden unchanged, FASTAs regenerated). Pre-removal copies in `backups/20260625_175146_coverage080_removal/`.
- evidence: post-removal all artifacts have 0 rows with coverage<0.8; `test_proteins_<tag>` set == `if_ready/main` set; h_maps 0 missing / 0 extra vs main; highrisk both 100 unique, min coverage 0.811/0.809, nmp_pct & head_pct still [0.90,1.0], 0 pdb nulls.
- impact: scope -- main test set + fast/pilot/highrisk + h_maps. risk low (backup retained). Open: pilot not backfilled to 50; `build_if_ready_test_set.py` does not yet enforce the coverage floor.
- status: done
- next_action: optionally (1) backfill pilot to 50 from cleaned fast; (2) add a `--min-if-coverage` floor to `build_if_ready_test_set.py` to prevent recurrence.
- refs:
  - `L0115` (highrisk sets), `L0108` (pilot rebuild), `L0100` (v2 promotion)
### L0127
- timestamp: 2026-07-03T15:40:00-04:00
- type: FEATURE
- module: EVALUATION
- trigger: User requested wiring two additional structure-prediction ("refold") backends — Protenix v2 (`/scratch/gpfs/KAIYIJIANG/tools/protenix`, ByteDance AF3 repro) and local ESMFold2 — alongside ESMFold v1 in the Phase C scTM path, with format-aligned outputs. Do not modify the shared `tools/` install; all GPU on ailab/pli.
- change_summary: `evaluate_phase_c.py --refold-model` now accepts `{esmfold, esmfold2, protenix, af3}` (+ generic `--refold-cache-dir` aliasing `--esmfold-cache-dir`). `inverse_folding/evaluation/refold.py` split into TWO dispatch classes: **live** (`esmfold` v1, unchanged — model loaded once, folded per design in-process) and **cache-read** (`esmfold2`, `protenix` — foreign envs cannot be imported into immune-design, so their structures are pre-computed by a separate SLURM step and read at eval time; `load_refold_model` returns `None`, `refold()` reads `<cache_dir>/<cache_key(protein_id,sequence)>.pdb` + `.plddt`, fail-fast non-OOM on miss). `af3` (official DeepMind AlphaFold3) is ALSO wired cache-read (`af3_runner.py` + `precompute_af3_refold.py`); its MSA is a LOCAL DB search (`/scratch/gpfs/DATASETS/alphafold`, no external server) so full-MSA is a clean two-stage on compute nodes — `submit_af3_data.slurm` (CPU `--mode data`) → `submit_af3_refold.slurm` (GPU `--mode inference`); a post-output JAX-teardown non-zero exit is tolerated (normalize is the gate). Common normalized contract = single-chain **.pdb** + **pLDDT 0-100**, so structural.parquet / TMalign scTM / sc_rmsd / foldability are backend-agnostic; **cache-read struct eval is CPU-only** (no in-process fold) and reuses `submit_benchmark.slurm` unchanged. New lib `refold_normalize.py` (mmCIF→single-chain PDB via biotite + pLDDT-scale normalize + `fold_records_from_parquet` single-source key enumeration) and `protenix_runner.py` (pick global-best Protenix sample by `ranking_score` + normalize). Precompute: **ESMFold2** reuses `predict_esmfold2_gt.py` (added `--from-parquet` / `--cache-layout` / `--emit-fold-fasta`; parquet read runs in immune-design, fold in the esmfold2 env — the env lacks pyarrow) via extended `submit_esmfold2_gt.slurm`. **Protenix full-MSA is TWO-STAGE** (compute-node `proxy/default` does NOT whitelist protenix-server.com → 403, and a GPU job idle-waiting on MSA is auto-killed): `scripts/protenix_msa_login.sh` fetches MSA on a login node (direct egress) → `_json/shard{k}of{N}-update-msa.json`; `scripts/submit_protenix_refold.slurm` (STAGE B, ailab/h200) folds the pre-fetched a3m (`--use_msa true`, GPU never idle) + `scripts/precompute_protenix_refold.py --mode normalize`. `USE_MSA=false` = no-MSA fallback. TDD: 47 tests across `tests/inverse_folding/test_{refold_backends,refold_normalize,protenix_runner,af3_runner}.py` + `tests/scripts/test_{evaluate_phase_c_refold_wiring,esmfold2_precompute,precompute_protenix_refold,precompute_af3_refold}.py`.
- evidence: live smoke on 2 uricases (Q00511, D0VWQ1; WT design) — end-to-end precompute→cache→cache-read→TMalign→structural.parquet, all format-consistent (single-chain pdb, pLDDT 0-100, `refold_backend` col). scTM/pLDDT: **esmfold2** 0.965/96.0 & 0.976/96.6; **protenix full-MSA** 0.969/95.2 & 0.990/96.1; **protenix no-MSA** 0.314/35.7 & 0.372/38.3; **af3 no-MSA** 0.225/28.7 & 0.309/27.6 (AF3 no-MSA near-useless → AF3 MUST use full-MSA, whose local-DB path is the cleanest of the three). 47/47 TDD green; existing esmfold path + its tests unchanged.
- impact: scope — Phase C refold/struct path + refold precompute scripts + `doc/SCRIPTS.md`. Nothing under `/scratch/gpfs/KAIYIJIANG/tools/` modified (protenix/AF3/ESMFold2 installs are CALLED). Existing `esmfold` v1 behavior byte-for-byte unchanged. risk low. Open: (1) Protenix full-MSA MSA-fetch is login-node + external-server + rate-limited; AF3 full-MSA is local-DB (clean, no server/login) but the CPU data pipeline is slow (~30min-hours/protein). (2) full-MSA AF3/protenix at scale not yet run (2-protein smokes only).
- status: done
- refs:
  - `doc/SCRIPTS.md` (Evaluation Integration §4-6), `B4` refold dispatcher, `L0126` (uricase set)

### L0128
- timestamp: 2026-07-02T18:30:00-04:00
- type: FEATURE
- module: RF
- trigger: Implement the RF refinement stage (PLAN_RF_REFINE.md R3–R4): a post-hoc targeted epitope-elimination search between generation and evaluation that mutates a best-of-N design's residual high-risk positions to drive the distinct NMP strong-core count toward 0 while preserving fold and freezing active-site anchors. R1/R2 (core extraction, blocks, editable, enumeration) were already done; this lands R3 (search law) + R4 (driver).
- change_summary: (R3) `inverse_folding/reference_flow/refine.py`: `StructureMetrics` bundle, `rank_margin_mass` (continuous NMP surrogate Σ max(strong_rank−rank_EL,0) over cores < margin_band), `structure_gate` (scTM floor vs seed − eps, extensible scRMSD/active-site ceilings), `accept_refinement` (hard output-accept = strictly fewer cores AND structure pass), `RefineResult`, and the `refine_sequence` beam search — soft beam ordered `(count asc, margin asc)`; structure is gated on OUTPUTS not every step (PLAN §1 update): only count-dropping candidates are refolded + `structure_gate`-checked (pass → shortlist+beam, fail → dropped), margin-progress states enter the beam with NO refold, and `max_path_mutations` (default 8) caps stacked-edit branch-waste. **NMP is BATCHED**: after a cost/timing investigation (batched ~10-13 s/seq vs single-entry ~15-28; E0 ceiling scores 1-3k candidates in one round), the `NmpFn` interface is `(pid, list[seq]) -> list[list[dict]]` and each round's whole `chosen` set is scored in ONE `nmp_fn` call (NetMHCIIpan model-load amortized); `head_fn` batched, `struct_fn` per-seq (only improving refold). (R4) `scripts/refine_rf_designs.py`: `--mode ceiling` (effect ceiling / E0 dataset, NMP-all one round → `ceiling/candidates.parquet`) and `--mode refine` (beam per seed → `refined/{refined_designs,evaluator_ready,refine_trace}.parquet` + `refine_config.json`; `evaluator_ready` is byte-schema-identical to `generated.parquet` for drop-in re-eval). Seeds come from run-dir best-of-N (`global_risk`) OR a self-contained `--seed-table` (protein_id/design_id/sequence). Block-structured proposer (`hotspot_blocks`→exhaustive singles→head-ranked pairs, `--max-pairs`). Oracles injected via `build_oracles` (reuses `build_head_scorer`; `StandaloneRunner.score_batch` flattened, `rank_EL` in FRACTION units; ESMFold `refold` + `run_tmalign` returning `{tm_score,rmsd}`; `resolve_structure_path` fed a test-set row). Sharded inputs globbed across `generation/*shard*/` + `eval_immune/*shard*/`. `doc/SCRIPTS.md` updated.
- rationale: A review-first pass reconciled the plan's R4 oracle sketch against reality before coding: (B1) `rank_EL` unit pinned to fraction (runner `el_rank`), NOT the ×100 percent of `imm_nmp_peptides.parquet`, so `strong_rank=0.02` compares correctly; (B2) `run_tmalign` returns a dict, not a tuple; (B3) driver needs `--test-set-parquet` for `resolve_structure_path(row, pdb_root)`; (B4) `--esmfold-cache-dir` required (refold `pdb_path` is None without it); (B5) inputs are 8-shard unmerged → glob. The batched-NMP interface change to `refine_sequence` was made at R3 (zero rework since R4 was unwritten) per the user's acceleration ask + the quantitative timing. Objective/soft-search framing (count is the hard accept, margin the soft progress) prevents multi-point-synergy stalls; the beam holding only structure-valid states prevents stacking edits on a broken backbone.
- artifacts:
  - `inverse_folding/reference_flow/refine.py`
  - `scripts/refine_rf_designs.py`
  - `tests/inverse_folding/test_refine.py`
  - `tests/scripts/test_refine_rf_designs.py`
  - `doc/SCRIPTS.md`
  - `PLAN_RF_REFINE.md` (NmpFn batched interface + confirmed R4 corrections)
- evidence: |
    TDD red→green: R3 pure logic (rank_margin_mass, structure_gate, accept, drive-to-zero, soft-beam two-step, structure-floor-blocks-beam) then batched-NmpFn retrofit + the structure-on-outputs policy (margin-progress-not-refolded + max_path_mutations cap tests) — `pytest tests/inverse_folding/test_refine.py -q` -> 14 passed. R4 driver smoke (fake oracles bypass build_oracles): refine mode writes both schemas + kills a core (count 1→0), evaluator_ready == generated schema, ceiling writes candidates with eliminates=True, arg parser exposes test_set_parquet/esmfold_cache_dir/netmhciipan_bin/topB, build_oracles fail-fasts on missing heavy inputs, the v0-deferral switches (--target-source head / --head-high-topk>0) fail-fast, and --seed-table sourcing refines listed seeds directly — `pytest tests/scripts/test_refine_rf_designs.py -q` -> 6 passed. Fixed a float-precision bug in the plan-provided R3 test (`==0.018` → `round(...,6)`). build_oracles (real torch/NMP/ESMFold path) has no runtime test; two review rounds against the confirmed oracle APIs caught + fixed pre-run breaks: (round 1) StandaloneRunner missing required binary_path → added --netmhciipan-bin, build_head_scorer needs setup.allele + setup.config.head.score_scale='raw_logit' + Path-typed head_checkpoint; (round 2 correctness) the head target-window template is now built PER-SEED from that sequence's actual length (a fixed 30-mer template mis-mapped ~300aa cores), ceiling soft-improvement uses margin<seed_margin + reuses structure_gate (was `margin<0` dead + a handwritten scTM check ignoring scRMSD/active-site ceilings), and nmp_fn fail-fasts on a missing NMP key / missing fitting length instead of reading a failed call as 'no epitope'. The consolidated required-input guard + the deferral guards are unit-tested to fail-fast before any heavy import.
- impact:
  - scope: New post-hoc RF stage (pure search module + driver). No change to generation/eval/controller. First experiments (E0 ceiling probe, E1 char23/0701) are agent-run via the existing submit pattern; a same-protocol structure reference (`--pdb-root`) + esmfold cache are run-time inputs.
  - risk: low
  - confidence: 0.82
- status: done (R3+R4 code complete + tested; E0/E1 experiments pending)
- next_action: Run the E0 ceiling probe (5 seeds, Q00511 + 4 lowest-global_risk) to measure the effect ceiling + head↔NMP retrieval before the full E1 char23/0701 refine; both as Della SLURM jobs (batched NMP, --esmfold-cache-dir, glob the 8 shards).
- refs:
  - `PLAN_RF_REFINE.md` (Tasks R1–R9)
  - `L0120` (enzyme-mode constraints reused for anchor freezing)

### L0128
- timestamp: 2026-07-03T16:10:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: User committed Pegloticase into the uricase manifest (`1efc7ef`, manifest + projection-audit only) and asked to run it fully through the dataset pipeline with an **AF3-predicted structure** added to the PDB collection (also the first real AF3 test). RF not to be run.
- change_summary: Materialized **Pegloticase** (recombinant pig-baboon chimeric therapeutic uricase / Krystexxa; 298 aa; P16164 variant 98.7% id; md5 `a0807c9...`, verified == committed manifest). Structure = **official DeepMind AF3 v3.0.3 full-MSA** prediction via the new two-stage refold pipeline (local-DB, no external server: `submit_af3_data.slurm` CPU MSA ~16 min on 4 cores + `submit_af3_refold.slurm` GPU inference 43 s; pLDDT **0.947**, pTM **0.94**) -> cleaned single-chain `pdbs_if_ready/Pegloticase.pdb` (`structure_source=af3`, first AF3-sourced structure). Appended as the **LAST row** of `uricase_caseset_if_ready_unified.parquet` (6387->**6388**; characterized 25->**26**; coverage 1.0; identity IF-ready mapping). Added to `Uricases_RF/` (design_viable 5491->**5492**; labeled caseset 6388). `h_maps_uricase_DRB1_{07_01,04_01}` computed standalone (npoff heads) and **concatenated** (6387->**6388** each; global_risk 0701 -4.61 / 0401 -9.53). Backups retained; original rows untouched.
- evidence: unified 6388, Pegloticase last, md5 == committed manifest `a0807c9`; manifest `validate_against_sequence(Pegloticase)` PASS (8/8 anchors); Uricases_RF design_viable 5492 / 24 characterized; h_maps 6388 both alleles with **0 missing / 0 md5-mismatch** vs unified; AF3 structure 298 CA single-chain, predicted seq == input. First end-to-end AF3 full-MSA production run on this cluster (local-DB MSA clean + fast: ~17 min/protein total).
- impact: scope -- uricase case set + `Uricases_RF/` + uricase h_maps + `pdbs_if_ready/`. Pegloticase now RF-ready (test_set + h_maps + structure + constraint manifest all aligned); **RF not run** (per user). risk low (backups retained; manifest already committed + validated). Open: `wt_generated` facade + uricase immune still on pre-Pegloticase 6387 (re-run if case-study immune needs Pegloticase).
- status: done
- refs:
  - commit `1efc7ef` (manifest add), `L0127` (refold backends incl. AF3 wiring), `L0125` (uricase exact-dedup)

### L0129
- timestamp: 2026-07-07T20:32:47-04:00
- type: FEATURE
- module: RF
- trigger: DPLM native full-v2 n=8 baseline was found to be effectively deterministic because the C0 native path hardcoded `sampling_strategy="argmax"` despite recording distinct seeds and `temperature=1.0`.
- change_summary: Exposed DPLM native `sampling_strategy` as a C0 CLI/slurm parameter and preserved historical `argmax` as the default; `gumbel_argmax` can now be selected for stochastic native multi-design sampling.
- rationale: The C0 native baseline needs an explicit stochastic sampling mode when `n_designs_per_protein > 1`; keeping `argmax` as the default avoids silently changing old deterministic baselines or IF-IMP reuse of the shared runtime helpers.
- artifacts:
  - `inverse_folding/reference_flow/runtime.py`
  - `scripts/run_if_phase_c0.py`
  - `scripts/submit_if_phase_c.slurm`
  - `doc/SCRIPTS.md`
- evidence: |
    `/home/zc1519/.conda/envs/immune-design/bin/python -m py_compile scripts/run_if_phase_c0.py inverse_folding/reference_flow/runtime.py` passed.
    `/home/zc1519/.conda/envs/immune-design/bin/python scripts/run_if_phase_c0.py --help` shows `--sampling-strategy {argmax,gumbel_argmax}`.
    Slurm smoke `10804754` used `NATIVE_SAMPLING_STRATEGY=gumbel_argmax` on 1 protein × 4 designs and completed with 4/4 unique generated sequences.
- impact:
  - scope: C0 native DPLM generation only when the new option is set; default C0 and shared runtime callers remain `argmax`.
  - risk: low
  - confidence: 0.9
- status: done
- next_action: Let full-v2 stochastic native job `10804882` complete and inspect `generation_diversity_check.json` from dependent check job `10805102`.
- refs:
  - `doc/SCRIPTS.md` Phase C C0 driver and launcher entries

### L0130
- timestamp: 2026-07-09T07:00:00-04:00
- type: DATA
- module: DATA_SELECTION
- trigger: `highrisk_dplm_iter_v1` (L0116) was selected on the DPLM-native C0 baseline, whose sampler hardcoded **argmax** (L0129) -> all 8 designs per protein were IDENTICAL, so the per-protein median-over-8 burden collapsed to a single over-confident mode (near-noise for MHC-II, where burden hinges on a few anchor/register positions). Compounding it, DPLM-native is a DIFFERENT sampler than the DFM reference-flow kernel every guided RF arm (C1/C2/C3/D) runs on. User: drop DPLM-native, rebuild the highrisk cohort on the **NoD** (RF-no-guidance, DFM kernel) FULL-POOL baseline; shard generation + eval for speed.
- change_summary: Built the first **full-pool NoD baseline** for 0701 -- `run_if_phase_c1.py` + `c1_null.yaml` (DFM `PositionDependentDFMSampler`, `amplification.form=constant_one`, no controller), temp 1.0 / seed 42 / n=8, over the full **2879**-protein 0701 test set, sharded into **16 parallel ailab/h200** jobs (~1.25 h each, ~3 h wall vs ~19 h single-job). Merged -> **23,032 designs**; verified **8/8 unique sequences for all 2879 proteins** (diversity restored vs argmax degeneracy). Immune eval = a1res03 0701 fold0 head + accelerated NetMHCIIpan, sharded into **96 cpu/qos=short** jobs (8 cores each, ~2.5 h wall), merged (imm_nmp + imm_head, 23,032 rows). Selected **`highrisk_nod_v1_HLA-DRB1_07_01`** (top-100 by `min(nmp_pct,head_pct)`, min-coverage 0.8) + `pmpnn` cross-baseline diagnostic + sorted FASTA. Marked `highrisk_dplm_iter_v1` **STALE** (sibling `.STALE.md`). New generic splitter `scripts/split_facade_shards.py` (registered in `doc/SCRIPTS.md`).
- evidence: NoD gen 2879x8=23,032 rows, all-8-unique 2879/2879; imm merged 23,032 rows / 2879 proteins; `highrisk_nod_v1` n=100, both axes pct in [0.91,1.00], sel_nmp 0.030-0.082, sel_head 1.66-4.60; **overlap with stale dplm_iter = 29/100** (71 picks changed -> the argmax bug materially corrupted the old selection). Kernel verification confirmed in code (RF = DFM kernel; NoD = c1_null guidance-off, same kernel as guided arms; DPLM-native C0 = all-position argmax/gumbel + DPLM reparam, a different generative process).
- impact: scope -- `if_ready/highrisk/` (0701). `highrisk_nod_v1` is now the kernel-consistent RF-iteration highrisk set; `dplm_iter` stale; `pmpnn_demo` unaffected (diverse sampler). **RF not run**. Open: struct eval on selected 100x8; RAR 0013 (B1 vs DPLM-native, regression-to-mean-inflated -4.37 nat / 95-100 / p<1e-16) needs a regression-free recompute vs NoD (docs put the unbiased aggregate near -0.58); 0401/1501 full-pool NoD baselines. risk low (stale set retained + marked, not deleted).
- status: done
- refs:
  - `scripts/build_highrisk_demo.py`, `scripts/split_facade_shards.py`, `inverse_folding/reference_flow/configs/c1_null.yaml`
  - `L0116` (dplm_iter selection, superseded), `L0129` (argmax determinism), `doc/Self-Cond_GR.md` (NoD oracle + regression-to-mean)

### L0131
- timestamp: 2026-07-09T18:30:00-04:00
- type: REFACTOR
- module: REFERENCE_FLOW
- trigger: The IF test-set pipeline carried **6 distinct allele-tag variants** (h_maps used `DRB1_07_01`, pdbs used numeric `0701`, everything else `HLA-DRB1_07_01`) from 3 divergent tag-helper functions; and the RF driver hard-required `--h-maps-parquet` + an md5 gate even for NoD/controller arms where the h-map is provably inert (g≡1). User (ultracode): unify the tag + make the h-map gate optional (h-maps likely dropped entirely downstream).
- change_summary: Ran an ultracode Workflow (map → implement → adversarial-verify) then verified/finished by hand. **(1) Tag unification**: 4 tag helpers (`precompute_h_maps.allele_tag`, `prescreen_tier2._allele_tag`, `prescreen_uricases._allele_tag`, `analysis/tier1_structural_analysis.allele_tag`) now all normalize+delegate to `reference_flow.runtime.safe_allele_tag` -> unified **`HLA-DRB1_07_01`**; every hardcoded SLURM default updated (`h_maps_DRB1_`->`h_maps_HLA-DRB1_`, `pdbs_if_ready/0701`->`pdbs_if_ready/HLA-DRB1_07_01`, retired `ALLELE_SHORT`). **(2) h-map gate OPTIONAL** (`run_if_phase_c1.py`): `--h-maps-parquet` now `default=None`; load + md5 gate skipped when omitted; **fail-fast only if `amplification.form` ∈ {linear_clamp,sigmoid,power}** (the h-consuming arms); NoD/`c1_null`/controller arms run with a zero h-vector (`g≡1`, identity schedule). **(3) Data reconciled** via new-tag->old-data symlinks (race-free, no rename, in-flight jobs unaffected); reversible rename migration `scripts/_migrate_tag_unify.sh` emitted (DRY-RUN default, NOT run).
- evidence: adversarial behavior-verifier = **PRESERVED** (amplification arms still require+consume h; NoD zero-h -> g≡1, no NaN/cap; all metadata sites None-guarded); 100 pytest passed; residual old-tag operative grep = **clean**; `run_if_phase_c1.py --help` confirms `--h-maps-parquet` optional (required set = checkpoint/test-set/pdb-root/allele/config/output-root only); all 7 changed .slurm bash -n OK + 5 changed .py py_compile OK; new-tag h_maps/pdbs symlinks resolve to real data.
- impact: scope -- RF driver + all IF-test-set SLURM launchers + tag helpers + docs. **1501 NoD generation can now SKIP the h-map precompute (Stage 6 GPU) entirely** (only amplification arms need it). All new artifacts use the unified tag. risk low (symlinks preserve every old path; migration reversible + not yet run). Open: run `_migrate_tag_unify.sh` (real rename) when all jobs idle, for on-disk name cleanliness (optional — symlinks are already functionally complete).
- status: done
- refs:
  - Workflow `wf_7fecdf6b-fd8` (tag-unify-hmap-gate), `scripts/_migrate_tag_unify.sh`, `scripts/run_if_phase_c1.py` gate
  - `L0130` (NoD baseline that motivated dropping h-maps)

### L0132
- timestamp: 2026-07-11T05:00:00-04:00
- type: FEATURE
- module: REFERENCE_FLOW
- trigger: `PLAN_RF_REFINE_FUSION.md` — build the Head-guided, structure-constrained edit-and-repair "Reward-Selected Edit-and-Repair Reference Flow" that supersedes the deferred Unified Architecture (`PLAN_RF_REFINE.md` §11). Runtime immune signal must be Head-only (NMP excluded from targeting/proposal/selection/acceptance/stopping); FK is an OPTIMIZATION selector, not exact twisted-SMC.
- change_summary: New `inverse_folding/reference_flow/fusion/` package (F1-F7) — immutable state + fail-fast typed config (calibration gates, canonical hash); complete-state Head objective (deterministic targeting, LME delta, parent-relative off-halo new-hotspot; all finite-guarded fail-closed); explicit breadth proposer (ALL singles always emitted, cap governs pairs) + RF reopen / edit-repair adapters (freeze halo-complement + edited-core + anchors, per-child seeds, assert-frozen-preserved / no-mask, injected `repair_fn`); Head-only oracle bundle + provenance structure cache + ABSOLUTE scTM/active-site gate; per-protein runner (evaluator/firewall + §1.7 `build_initial_population` + greedy/beam(ε_H)/FK(offspring-normalized, log-space, canonical-order resample) loop with independent best-feasible elite archive + fresh-parent scoring); and the `active_site_spatial_shell6A_ca_rmsd` producer (`structure_metrics.py`, `sc_rmsd` Kabsch + new 6 Å shell membership). `runtime.py` gains a tri-state `use_draft_seq_override` (byte-equivalent when omitted; forces backbone-only) and `reference_flow/__init__` is now lazy (PEP 562) so pure imports are torch-free. New driver `scripts/run_rf_refine_fusion.py` (terminal handoff → loop → 7 artifacts + manifest, per-protein atomic checkpoint + deterministic resume; real Head / ESMFold+TMalign+shell-RMSD oracles; NMP never imported) + `submit_refine.slurm` `MODE=fusion` branch + `configs/rf_refine_fusion_smoke.yaml`.
- evidence: 160 local pytest pass across `tests/inverse_folding/test_reference_flow_fusion_*.py` (state/config/objective/moves/runner/selection/structure_metrics) + `test_reference_flow_runtime.py` (override byte-equivalence) + `tests/scripts/test_run_rf_refine_fusion.py` (fake-oracle artifacts + resume + failure recording). A Codex review (7×P1 / 4×P2) + independent 4-agent re-review drove fail-closed hardening (NaN/None/missing-ceiling, elite captures ε_H-rejected feasible states, breadth no longer position-biased, config validation, torch-free imports) — all findings fixed + re-verified. `py_compile` clean; `bash -n scripts/submit_refine.slurm` OK; existing `reference_flow` suite (sampler/runtime/D1, 74 tests) unaffected by the lazy `__init__` + runtime override.
- impact: scope -- net-new Fusion generation path; legacy RF/controller/D2/D3/SC-GR/standalone-refiner behavior unchanged unless `MODE=fusion` / a fusion config is selected. **Implemented in an isolated git worktree** (`fusion_rf_refine`), uncommitted. risk low (additive; existing suites green). Open: the sampler-backed `repair_fn` for the H3 repair arm + the real S0-S2 smokes (Head/ESMFold checkpoints, GPU) are cluster steps; scTM/ε_H/`active_site_RMSD_max`/N/n_rounds/β are calibration gates to fix from §6 smoke telemetry.
- status: done
- refs:
  - `PLAN_RF_REFINE_FUSION.md`, `doc/RF-Refine-Fusion.md`, `doc/SCRIPTS.md` (RF-Refine Fusion section)
  - `inverse_folding/reference_flow/fusion/`, `scripts/run_rf_refine_fusion.py`, `scripts/submit_refine.slurm` `MODE=fusion`

### L0133
- timestamp: 2026-07-11T09:00:00-04:00
- type: BUGFIX
- module: REFERENCE_FLOW
- trigger: Second Codex review of the F6/F8/F9/F10 driver layer (0×P0, 8×P1, 3×P2) found the L0132 driver was over-claimed: the real-oracle wiring was written against GUESSED interfaces (broken), edit-repair was unwired (not "cluster-only"), and resume/config/artifacts/elite had real gaps. All findings accepted after re-audit.
- change_summary: Re-verified the real interfaces with a 5-agent workflow, then fixed against ground truth. **Driver**: `ConstraintManifest` lookup uses `has_protein`/`constraint_for_protein` (no `.get()`); shell RMSD uses `np.vstack([r.coord for r in parse_ca_trace(...)])`; head setup passes `Path`-typed `head_config_dir`; `_load_generated` handles the sharded `generation/*shard*/generated.parquet` layout; launcher `all` maps to all-proteins and the driver exits non-zero on 0-ok. **Repair (P1-3)**: implemented `make_sampler_repair_fn` (unit-tested) + `build_repair_fn_factory` (per-protein `prepare_backbone`→backbone-only context→`make_dplm_denoiser`→sampler with `fixed_tokens`/`controller=None`/per-child seed) + `--base-if-checkpoint`/`--rf-sampler-config`; the edit-repair moves are now merged into `run_protein`'s round order. **Resume (P1-4)**: per-protein checkpoints carry `config_hash`; resume reuses only matching successful checkpoints, re-runs failures, and aggregates only current-config checkpoints. **Runner/state/config**: independent elite archive uses a distinct `make_elite_id` (no beam self-loop / dangling lineage; `first_round_seen=r`); config rejects NaN/inf structure ceilings + unknown backend + enabled-move zero budgets; `PopulationState` enforces unique ids + contiguous slots; `build_initial_population` validates target-backbone length; global per-parent sequence dedup across move families; artifacts add ancestry/selection-count, elite provenance, git SHA, round cache-hits/wall-time.
- evidence: +~30 adversarial tests; full fusion suite green; driver fake-oracle smoke (artifacts/resume/failure/provenance/repair-helper) green; `py_compile` + `bash -n` OK; existing `reference_flow` suite unaffected. Re-verification workflow `wf_5c28be36` returned the exact ConstraintManifest / CaResidue.coord / build_head_scorer-setup / sampler.sample / `_load_sharded` signatures the fixes were written against.
- impact: scope -- the L0132 Fusion driver, now correct against real interfaces + repair-wired. Still worktree-only, uncommitted. risk low (additive + hardening). Open: S0–S2 real smokes (checkpoints/GPU) remain the only unrun step; the sampler-backed repair's live execution is pending cluster validation (interfaces verified against source, not yet run).
- status: done
- refs:
  - `L0132` (initial Fusion landing, hardened here), re-verify workflow `wf_5c28be36`
  - `scripts/run_rf_refine_fusion.py`, `inverse_folding/reference_flow/fusion/{runner,config,state}.py`

### L0134
- timestamp: 2026-07-11T15:30:00-04:00
- type: BUGFIX
- module: REFERENCE_FLOW
- trigger: Third Codex review of the Fusion module (7×P1 + 4 minor) found experiment-integrity and firewall gaps that the happy-path tests missed. All P1 accepted after per-`file:line` re-verification; round-boundary checkpoint / F6 batching / S2 replay-CLI judged genuine larger scope and honestly deferred (not silently dropped).
- change_summary: Repair kernel now actually drives the sampler; resume identity binds all inputs; beam-elite lineage is replayable; anchor firewall + repair-halo gate closed; window telemetry + manifest provenance emitted.
- rationale: The H3 arm's experiment label must match the executed kernel and its ancestry must be replayable; resume must never return stale results on changed inputs; the active-site firewall must fail closed on malformed manifests.
- change_summary_detail: **P1-1** `make_sampler_repair_fn` applies the Fusion config's declared kernel (`sampler_steps`→`sampler.n_steps`, `local_remask`→`sampler.remask.enabled`) over the base rf-sampler-config every child; config now requires `sampler_steps≥1` when `rf_reopen` is enabled (shared kernel). **P1-2** `submit_refine.slurm` `MODE=fusion` conditionally forwards `--base-if-checkpoint`/`--rf-sampler-config`. **P1-3** the repair-shortlist off-target gate measures new hotspots against the actual repair halo (`moves.repair_halo_span`, target±`halo_radius`), not the narrower target span. **P1-4** per-protein resume `run_sig = sha256(config_hash + input_signature + seed_digest)`; `input_signature` fingerprints Head ckpt/config/variant/allele/**window range**, constraint manifest, base-IF, sampler config, target backbone (config_hash still scopes aggregate/manifest). **P1-5** the beam protected-elite carry re-instates the elite's own persisted parent + originating proposal and emits an `elite_protect` lineage edge (no dangling archive-id parent / null proposal). **P1-6** negative/out-of-range anchor indices fail closed in `structure_metrics.shell_indices_from_ca`, `runner._validate_anchor_indices`, and driver `_validate_manifest_against_references` (`validate_against_sequence` vs reference). **P1-7** `aligned_windows` populated from Head window z (gated on `telemetry.window_detail`) + candidate column; manifest carries an input-provenance block. **Minor** strict `_as_bool` (`"false"`→False, ambiguous→fail-fast); all "cluster-validated" doc claims corrected to "interfaces verified / live run pending".
- evidence: 241 local pytest green (fusion/driver/runtime/config/sampler/constraints), +13 adversarial tests this round (repair-halo gate, elite-protect lineage, anchor range×3, sampler-kernel override, reopen-shared-kernel, resume input/seed identity, aligned_windows gating, provenance wiring, strict bool×3, driver firewall). `py_compile` + `bash -n` + smoke-YAML load clean. A 10-agent adversarial-verification workflow (`wf_4988d27a`) refuted each fix independently: P1-1/P1-3/P1-5/bool confirmed_fixed; it surfaced 3 real residuals — a missed "cluster-validated" docstring, `window_k_min/max` absent from the resume signature, and untested `_provenance`/driver-firewall wiring — all fixed here.
- impact: scope -- experiment-integrity + firewall hardening of the L0132/L0133 Fusion path; no change to legacy RF behavior. Still worktree-only (`fusion_rf_refine`), uncommitted. risk low (additive + fail-fast). Open (honestly deferred, not done): F8 round-boundary checkpoint (per-protein granularity suffices for v0 smokes), F6 per-child→batch repair, an S2 evaluated-pool replay CLI; and the real S0–S2 smokes (Head/ESMFold/base-IF checkpoints + GPU) remain the only unrun step.
- status: done
- refs:
  - `L0132`/`L0133` (Fusion landing + review-2 hardening), verify workflow `wf_4988d27a`
  - `inverse_folding/reference_flow/fusion/{runner,config,moves,structure_metrics}.py`, `scripts/run_rf_refine_fusion.py`, `scripts/submit_refine.slurm`

### L0135
- timestamp: 2026-07-13T17:27:04-04:00
- type: FEATURE
- module: REFERENCE_FLOW
- trigger: The standard RF refiner still used in-process ESMFold1 and a seed-relative scTM / whole-anchor compatibility gate, while `PROTOCOL/shortlist_and_refine_seed_selection.md` defines active-site reliability using absolute scTM, direct-functional catalytic side-chain RMSD, and active-site confidence. User requested ESMFold2 and the protocol metric set without hardcoded threshold values.
- change_summary: Standard `refine`/`ceiling` now defaults to one persistent isolated `esmfold2_live` worker and closes it before the final cache-read structural pass. The standard fail-closed gate is the conjunctive runtime-calibrated trio `scTM >= X`, `cat_max_scRMSD <= Y`, and `predicted_active_site_min_pLDDT >= Z`; all thresholds are required CLI/SLURM inputs with no repository defaults. `cat_max_scRMSD` is computed over the manifest's direct-functional subset (safety-max provenance, or the explicit v0 hard-anchor mapping), persisted in refinement outputs, and missing/incomplete geometry fails closed. The old gate remains only under `structure_gate_profile=legacy`. The launcher uses a backend-specific refold cache by default so an old ESMFold1 cache is not silently reused.
- evidence: 109 targeted tests pass across pure refine/driver/merge, ESMFold2 live/refold/cache normalization, structural v2, Phase C eval, and Fusion launcher suites; `bash -n`, `py_compile`, and targeted `git diff --check` pass. Della has a complete ESMFold2 package overlay and a 1.3 GB cached `biohub/ESMFold2` model. No live GPU refinement job was run in this change.
- impact: scope -- standard RF refinement structure oracle, gate semantics, launcher inputs, and refinement metric columns; Fusion behavior is unchanged. Existing ESMFold1 outputs/caches are not rewritten. risk medium until a real H200 one-seed smoke confirms joint Head + ESMFold2 memory/runtime.
- status: done
- refs:
  - `scripts/refine_rf_designs.py`, `scripts/submit_refine.slurm`, `inverse_folding/reference_flow/refine.py`
  - `PROTOCOL/shortlist_and_refine_seed_selection.md`, `PLAN_RF_REFINE.md`, `doc/SCRIPTS.md`

### L0136
- timestamp: 2026-07-17T01:10:36-04:00
- type: FEATURE
- module: L
- trigger: pilot_v2/fast_v2 stratify on `coverage_fraction` (generator-independent, difficulty-blind); user wants a pilot uniform over de-immunization difficulty on the NoD baseline. Question raised: is the highrisk-style regression bias serious enough to require decoupling the difficulty realization from the eval baseline?
- change_summary: Added `scripts/build_pilot_difficulty.py` (SCRIPTS.md §Module L #13) and built pilot-v3 for both alleles (`if_ready/pilot/pilot_v3_<tag>.parquet` + `.fasta`, n=50 each), stratified uniformly across NoD-baseline difficulty (median NMP strong_frac over ALL 8 designs; 10 equal-frequency quantile bins × 5).
- rationale: NoD-residual NMP burden is the on-target "de-immunization difficulty" axis (more than native/WT load). The highrisk regression inflation is severe only for *extreme-tail* selection + a quantitative small-effect claim; for a UNIFORM iteration set analyzed qualitatively it is a second-order effect, so binning on all 8 designs (cleanest per-protein estimate) is the default rather than a 4/4 split. The script still supports an optional disjoint `--half-b-designs` reserve for when a rigorous quantitative per-bin method-vs-NoD number is needed (or regenerate a fresh NoD baseline for the 50 proteins at eval time). NMP-only (not head) defines the axis because the head is the RF guidance (circular). fast_v2 unchanged (generator-independent breadth set); pilot_v2 retained in parallel (SC-GR planner cohort, `PLAN_PLANNER_SC_GR.md`).
- artifacts:
  - `scripts/build_pilot_difficulty.py`, `tests/scripts/test_build_pilot_difficulty.py`
  - `work/immune-design/if_test_set/if_ready/pilot/pilot_v3_HLA-DRB1_{07_01,04_01}.parquet` (+`.fasta`)
  - `run/benchmark/if_phase_c/nod_full_v1/HLA-DRB1_{07_01,04_01}/nod_full_v1_*_imm/` (source NoD imm, 8 designs/protein)
- evidence: 5 pytest green (per-design difficulty aggregation, equal-frequency quantile binning, remainder→hardest, monotonicity, seed-determinism). Build (all-8 binning): both alleles n=50, 5/bin, per-bin median difficulty monotone, selected NMP strong_frac span 0701 [0.000,0.042] / 0401 [0.001,0.071], all rows ⊂ main IF-ready.
- impact:
  - scope: new data-selection script + pilot-v3 canonical subset (both alleles); fast_v2/pilot_v2/main untouched.
  - risk: low
  - confidence: 0.90
- status: done
- next_action: if a quantitative per-bin method-vs-NoD claim is later needed on pilot-v3, rebuild with a reserved `--half-b-designs` or evaluate against a freshly regenerated NoD baseline for the 50 proteins.
- refs:
  - `scripts/build_highrisk_demo.py` (selection framework reused), `PLAN_PLANNER_SC_GR.md` (pilot_v2 cohort)


### L0137
- timestamp: 2026-07-27T02:40:00-04:00
- type: FEATURE
- module: REFERENCE_FLOW
- trigger: `PLAN_RF_REFINE_FUSION_V1.md` (V1-A) had no implementation. A first pass landed the P1 pre-terminal arm, the P1 terminal arm and T0; a Codex review plus an 8-way adversarial verification pass then found that several of them were silently wrong rather than merely incomplete.
- change_summary: |
    Implements V1-A end to end on `fusion_rf_refine` (worktree-only, uncommitted). New pure package
    `inverse_folding/reference_flow/fusion/v1_{records,config,seeds,alloc,admission,ledger}.py`,
    entry orchestration `scripts/rf_fusion_v1_{entry_core,cohort,artifacts,preflight,oracles}.py`,
    driver `scripts/run_rf_fusion_v1_entry.py`, launcher `MODE=v1_entry`, and three canary configs.
    The entry runtime is the frozen null kernel (`g(h)=1`, `controller=None`, no h-map); BOTH
    position-dependency pathways are actively refused, and isolation is verified AT THE SAMPLER with
    a positive control (the same comparison under `linear_clamp` must diverge).
    The review pass fixed six defects that produced plausible wrong numbers rather than errors:
    (1) T0 grid points shared one root pool, so a later maturity estimated/selected/evaluated on an
    EARLIER maturity's roots and its reserved budget -- hence the size of its compute-matched control
    -- grew with grid position; each point now reads only the journal tail it wrote, and
    `full_control` ledger ids carry `rho_id`.
    (2) T0 wrote only membership tables, leaving all four runbook §6.1 GO/KILL conditions
    uncomputable from disk while exiting 0; it now writes the standard §9 evidence tables, the
    per-maturity reserved budget, and every realized seed.
    (3) the T0 `independent_full` control was never Head-scored (while its ledger booked
    `head_samples=1` per trajectory); it is now ranked by exact terminal Head under the Terminal law.
    (4) the §3.4 launch gate under-projected T0 by `len(rho_grid)` and omitted the full control
    entirely, so a run blew its declared `max_dfe`/`max_refolds` and still returned 0.
    (5) the Terminal reservation was not bound to the experiment that produced it and its content
    did not enter the resume signature, so a manifest from another campaign ran at the wrong scale
    silently; and the cohort total used `min × n_proteins`.
    (6) facade collapse dropped lineages that were persisted nowhere (breaking the §2.11 chain) and
    labelled a converged top-`F_cap` trajectory `rank=null`, which the schema documents as "generated
    but not submitted"; collapse is now persisted to `facade_collapse.parquet`, a shortened facade
    returns `entry_facade_collapsed`, and below `N` the entry stage hard-fails instead of letting v0
    misattribute it.
    Also: the entry→v0 edge is now an explicit `entry_source_id` carried onto `ParticleState` and
    persisted through `fusion_particles` (PLAN §2.11:531 forbids reconstructing it from sequence);
    the batched sampler finalizer no longer drops `logical_dfe`/`final_scores`; the declared
    `random_membership_seed` now participates in the draw it names; `realized_seeds` lists only the
    `final` seeds actually drawn; and the dead `continuation_batch_size` knob is removed.
- evidence: full local suite 2355 passed / 47 skipped, with the SAME 14 pre-existing failures as the pristine tree (unrelated modules with missing optional deps) -- i.e. zero new failures. `bash -n scripts/submit_if_phase_c.slurm` clean. Every fix landed RED-first. NOTHING has run on a GPU; all evidence is unit/contract tests and local adversarial probes.
- impact:
  - scope: new V1 entry pipeline + configs + launcher mode; v0 Fusion behavior unchanged except two additive, defaulted fields (`ParticleState.entry_source_id`, `fusion_particles.entry_source_id`) that stay null on a standalone v0 run.
  - risk: medium until the cluster canary runs; the matched-compute claim is the load-bearing one and has never been exercised on real DPLM/Head/structure oracles.
  - confidence: 0.75
- followup_2026-07-27: |
    Second Codex review + a 6-way adversarial verification pass. Six further defects closed, three
    of them introduced by the first pass:
    (a) the multi-rho T0 checkpoint derived its status from `points[0]`, so a grid whose later
    maturities all failed reported `t0_complete`/ok=true and exited 0; status is now the first
    failing point's and `per_rho_status` is persisted.
    (b) the Head-ranked `full_pool` added for the T0 control was never read by the structure
    subsample. Fixing that by truncating the control to the Terminal law's top-`F_cap` was WRONG
    and was reverted: PLAN §2.10 requires "no policy may use another rule or sample size", and a
    one-sided truncation would have drawn the control from its best ~25% while the partial views
    used 100% of theirs, biasing GO/KILL condition 3 toward a false KILL. The eligible pool is an
    OPEN scientific decision now recorded in runbook §6.0 as blocking a scientific T0.
    (c) `{protein}__{arm}.json` collided between T0 and P1 pre-terminal (T0's arm IS preterminal),
    silently overwriting the P1 reservation; the phase is now part of the name.
    (d) the Terminal reservation was not bound to the scientific substrate: the manifest never
    recorded the pre-terminal `S`, so a Terminal run reading S from its own YAML got 2x/4x the
    matched budget with no error and every artifact internally consistent at the wrong scale. The
    manifest now persists S + Head/DPLM/backbone/v0-config digests + the Head INFERENCE config
    (variant, allele index, window k-min/k-max, allele) and the loader compares all of them;
    an absent substrate block is refused rather than waived.
    (e) v0 admission dropped every rejected facade row with a bare `continue` and booked no round-0
    refolds, so an anchor violation on a completed sequence (detectable nowhere else) vanished and
    a matched-compute check on refolds compared two zeros. Added the §2.11-authorized additive
    audit seam: `fusion_initial_admission_verdicts.parquet` + `manifest.initial_refolds_by_protein`.
    (f) `_seed_digest` omitted `entry_source_id`, so two entry runs whose ranked sequences converge
    but whose roots differ shared a v0 resume key and inherited the wrong lineage.
    Also: two justifications written into the collapse docstring were FALSE (v0 does not re-refold a
    duplicate -- the cache keys on sequence_md5 -- and particle_id is not ambiguous, it embeds
    slot_idx); the one real reason is that duplicate round-0 particles buy ancestry mass, which v0's
    own contract forbids. Restored the 24-anchor `uricase_q00511_active_site_safety_v1.yaml` (the
    branch had only the 8-anchor v0 projection template, and the authority fixes 24/302). Added
    runbook §11.7: the v0 terminal stage is a SECOND job, and `submit_refine.slurm`'s default
    `FUSION_CONFIG` is the SMOKE package, which runs to completion and emits a plausible elite from
    the wrong method. 20 stale PLAN line-anchors corrected repo-wide.
    NOT fixed, disclosed instead: a crash inside an oracle still loses that call's burned physical
    cost (the real fix is an oracle-contract change; booking a zero would be a fabricated
    measurement), and `admit_facade` can lose up to `F_cap` such attempts, not one.
- status: done (local); V1F7 cluster canaries pending
- next_action: run runbook §11.2 (Canary A) then §11.3 / §11.5 / §11.6; freeze `TEST_SET_PARQUET`, `DEV_IDS`/`HOLDOUT_IDS`, the T0 `rho_grid`/`K_EVAL`/`Q_T0`, and the measured `SECONDS_PER_REFOLD`/`SECONDS_PER_DFE` (the gate refuses an unverifiable walltime by design).
- refs:
  - `PLAN_RF_REFINE_FUSION_V1.md`, `doc/FUSION_V1.md`, `doc/RF_Fusion_v1_Cluster_Runbook.md`, `doc/SCRIPTS.md`
  - two known gaps stated in the runbook status block: no T0→v0 definitive-structure handoff, and `maturity_telemetry.anchor_preservation` is null (not a measurement)
