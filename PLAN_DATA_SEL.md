# Data Selection & Evaluation Pipeline Plan

> **STATUS (2026-08-28): historical design record, not the operative document.**
> This plan was last revised 2026-06-04 and stops before three changes that define the shipped
> set: Tier 3 removal (`LOG.md` L0106), the `if_sequence_coverage >= 0.8` truncation filter
> (L0126), and the move to the full-data production Head. For what the cohort must satisfy today
> — including which constraints here are retired — read
> **`PROTOCOL/if_benchmark_test_set_construction.md`**. Cite this plan only for the provenance of
> a frozen value. Individual superseded decisions are marked inline in §7.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For Claude:** REQUIRED WORKFLOWS: use `superpowers:writing-plans` for planning updates and `superpowers:test-driven-development` for key module/function/pipeline checks.

**Goal:** Curate a three-tier test protein set for the Immune-Design inverse-folding pipeline, generate pre-computed evaluation artifacts for each protein, and verify the shared evaluation stack runs end-to-end on the curated set.

**Architecture:** The test set serves all downstream IF modules (guidance sweep, comparison baselines, Phase 2-3). Tier 1 (gold standard) provides experimental ground truth; Tier 2 (computationally pre-screened) provides statistical power; Tier 3 (therapeutic) provides translational narrative. Each tier has different selection criteria but shares a common artifact schema. Selection is a mix of manual curation (IEDB queries, literature search, manual review) and automated pipelines (MMseqs2 overlap, NetMHCIIpan batch screening, epitope head scoring, CATH topology sampling). Manual steps are gated by scripted validators that refuse to proceed without required intermediate artifacts.

**Tech Stack:** Python, pandas/Parquet/JSON, MMseqs2, NetMHCIIpan 4.3, epitope-head `InferencePredictor`, ESMFold, TM-align, BioPython (PDB extraction), pytest.

---

## 1. Planning Rules (This File)

1. This file is the execution-facing plan for data selection and evaluation pipeline integration, absorbing Module L from `PLAN_IF.md`.
2. This file records curation workflow, frozen contracts, manual-step checklists, risk controls, and verification gates.
3. This file does not contain production code snippets.
4. Any change to test-set curation assumptions must be reflected here before implementation proceeds.
5. Every material planning update must be appended to `LOG.md`.

## 2. Source of Truth

1. Three-tier test set design and scientific rationale: `doc/Data_Selection_v1.md`
2. Evaluation metric definitions and pipeline spec: `doc/Inverse_Folding_v1.md` Section 2
3. Inverse-folding stage plan (Module K/M/N, this file absorbs Module L): `PLAN_IF.md`
4. Epitope-head inference contract: `doc/Epitope_Head_v1.md`
5. High-level architecture: `doc/Immune_Design_Architecture_v2.md`

## 3. Global Milestones

### Milestone M0: Governance Green

- Outcome:
  - tier schema and curation contract are frozen
  - overlap policy is machine-enforced
  - artifact schema per test protein is defined

### Milestone M1: Tier 1 Green

- Outcome:
  - 5-10 proteins with experimentally validated DRB1\*07:01 epitopes are curated
  - CATH training overlap is excluded
  - epitope-head training overlap is annotated

### Milestone M2: Tier 2 Green

- Outcome:
  - 30-50 proteins pass automated pre-screening (NetMHCIIpan + epitope head agreement)
  - CATH topology diversity is maximized (≥ 10 topologies)
  - all overlap flags are computed

### Milestone M3: Tier 3 Green

- Outcome:
  - 5-10 therapeutic proteins with literature immunogenicity evidence are selected
  - NetMHCIIpan signal is confirmed

### Milestone M4: Assembly Green

- Outcome:
  - `test_proteins.parquet` is assembled with all pre-computed artifacts
  - per-protein PDB and FASTA files are stored
  - WT hotspot maps and NetMHCIIpan baselines are pre-computed

### Milestone M5: Evaluation Integration Green

- Outcome:
  - shared evaluation pipeline runs end-to-end on the curated test set
  - downstream modules (M, N) can consume the test set without modification

## 4. Completed Infrastructure (From PLAN_IF.md Module L)

The following evaluation pipeline components are already implemented and tested. They are not re-planned here but are listed for reference and dependency tracking.

| Component | File | Status | Tests |
|-----------|------|--------|-------|
| Frozen metric schema | `inverse_folding/evaluation/schema.py` | Done | `test_module_l_eval_contract.py` |
| Basic curation logic | `inverse_folding/evaluation/test_set.py` | Done (needs extension) | `test_module_l_testset_contract.py` |
| ESMFold wrapper | `inverse_folding/evaluation/esmfold_runner.py` | Done | `test_module_l_eval_contract.py` |
| TM-align wrapper | `inverse_folding/evaluation/tmalign.py` | Done | `test_module_l_eval_contract.py` |
| Immunogenicity scorers | `inverse_folding/evaluation/immunogenicity.py` | Done | `test_module_l_eval_contract.py` |
| Evaluation aggregator | `inverse_folding/evaluation/aggregate.py` | Done | `test_module_l_eval_contract.py` |
| Evaluation CLI | `scripts/evaluate_if.py` | Done | — |
| IEDB source extractor | `scripts/identify_sources.py` | Done | — |

## 5. File Structure Map

### New Files

| File | Responsibility |
|------|---------------|
| `inverse_folding/evaluation/overlap.py` | MMseqs2-based sequence identity overlap detection |
| `inverse_folding/evaluation/prescreen.py` | Batch NetMHCIIpan + epitope head pre-screening for Tier 2 |
| `inverse_folding/evaluation/cath_topology.py` | CATH topology assignment and diversity sampling |
| `inverse_folding/evaluation/assembly.py` | Test set assembly, artifact generation, and validation |
| `scripts/run_overlap_filter.py` | CLI: run MMseqs2 overlap against CATH train sequences |
| `scripts/prescreen_tier2.py` | CLI: batch pre-screen candidates for Tier 2 |
| `scripts/assemble_if_test_set.py` | CLI: final assembly of `test_proteins.parquet` |
| `scripts/submit_prescreen_tier2.slurm` | SLURM job for Tier 2 batch pre-screening |
| `scripts/submit_assemble_test_set.slurm` | SLURM job for artifact generation (ESMFold, head, NMP) |
| `tests/inverse_folding/test_module_l_overlap.py` | Tests for MMseqs2 overlap detection |
| `tests/inverse_folding/test_module_l_prescreen.py` | Tests for pre-screening logic |
| `tests/inverse_folding/test_module_l_assembly.py` | Tests for assembly and validation |

### Files to Modify

| File | Change |
|------|--------|
| `inverse_folding/evaluation/test_set.py` | Extend `CurationConfig` with tier field, add MMseqs2-aware overlap |
| `PLAN_IF.md` | Remove Module L (absorbed here), update cross-references |

## 6. Module L: Data Selection & Shared Evaluation Pipeline

**Objective**
- Curate a fixed three-tier test protein set, generate pre-computed WT artifacts per protein, and verify the shared evaluation pipeline runs end-to-end on the curated set.

**Inputs**
- CATH 4.3 training split chain names (`chain_set_splits.json`)
- CATH 4.3 training sequences (for MMseqs2 clustering)
- IEDB export for DRB1\*07:01-restricted assays
- PDB candidate pool (single chain, 100-500 AA, X-ray < 2.5 Å)
- Epitope-head checkpoint for `HLA-DRB1*07:01`
- NetMHCIIpan 4.3 runtime

**Outputs**
- `outputs/if/test_set/test_proteins.parquet`
- `outputs/if/test_set/test_proteins_summary.json`
- `outputs/if/test_set/curation_ledger.json`
- `outputs/if/test_set/pdbs/<protein_id>.pdb`
- `outputs/if/test_set/fastas/<protein_id>.fasta`
- `outputs/if/test_set/wt_hotspot_maps.parquet`
- `outputs/if/test_set/wt_netmhciipan.parquet`
- `outputs/if/test_set/tier1_candidates.json` (manual curation intermediate)
- `outputs/if/test_set/tier2_prescreened.parquet` (automated intermediate)
- `outputs/if/test_set/tier3_candidates.json` (manual curation intermediate)

**Scope Boundaries**
1. The test set must serve Module M (guidance), Module N (baselines), and future Phase 2-3.
2. CATH training overlap is excluded by MMseqs2 sequence identity > 30%, not just PDB code matching.
3. Epitope-head training overlap is annotated but not excluded — NetMHCIIpan provides independent validation.
4. The test set is frozen after curation. Future expansion uses separate cohorts.
5. Manual steps produce intermediate JSON/CSV artifacts that are validated by scripts before proceeding.

**Overlap Policy (Frozen)**

| Overlap type | Policy | Rationale |
|---|---|---|
| Test protein ∩ CATH training (IF model) | **Strict exclude** (seq id > 30% via MMseqs2) | Fair IF evaluation |
| Test protein ∩ Epitope head training | **Annotate, split-report** | Head is guidance signal, not evaluator |
| Tier 1 epitope data ∩ head training positives | **Annotate** | Flag for transparency |

---

### Task L0: Freeze Tier Schema and Curation Contract

**Goal**
- Define the per-protein schema that all three tiers must produce, so downstream assembly can merge them without schema drift.

**Actions**
1. Freeze the test protein schema with these required fields:
   - `protein_id`: unique identifier (PDB chain ID for Tier 1/2, custom for Tier 3)
   - `tier`: integer `1`, `2`, or `3`
   - `sequence`: WT amino acid sequence
   - `sequence_length`: integer
   - `pdb_path`: relative path to PDB file
   - `resolution`: X-ray resolution in Å (null for AF2 predictions)
   - `cath_overlap_flag`: boolean
   - `cath_overlap_id`: matched CATH ID or null
   - `head_train_overlap_flag`: boolean
   - `netmhciipan_n_strong`: WT strong binder count
   - `netmhciipan_mean_best_rank`: WT mean best rank
   - `head_global_risk`: WT epitope head global risk R
   - `head_n_hotspot`: WT hotspot positions above threshold
   - `cath_topology`: CATH topology code (Tier 2 only, null otherwise)
   - `experimental_epitopes_json`: JSON array of experimentally validated epitope spans (Tier 1 only, null otherwise)
   - `literature_evidence`: citation or brief evidence string (Tier 3 only, null otherwise)
   - `selection_reason`: free-text selection rationale
2. Freeze the curation config extensions:
   - Tier 1: require experimental DRB1\*07:01 epitopes from IEDB
   - Tier 2: require NetMHCIIpan ≥ 5 windows %Rank_EL < 2% AND head global risk in top 50%
   - Tier 3: require NetMHCIIpan ≥ 3 windows %Rank < 5% AND literature immunogenicity evidence
3. Freeze MMseqs2 overlap parameters: `--min-seq-id 0.3 --cov-mode 0 -c 0.8`

**TDD Gate**
1. RED: schema validator accepts a protein entry missing required fields.
2. GREEN: validator rejects incomplete entries and accepts only fully populated ones, with tier-specific nullable rules enforced.

**Acceptance**
- Module L has one frozen per-protein schema shared across all tiers.

---

### Task L1: Build MMseqs2 Overlap Detection

**Goal**
- Replace the PDB-code-level overlap check with proper sequence identity clustering, so that structurally similar but differently-named proteins are caught.

**Actions**
1. Build a wrapper that:
   - Takes candidate FASTA + CATH training FASTA as inputs
   - Runs `mmseqs easy-search` with frozen parameters
   - Parses results into a set of `(query_id, target_id, seq_identity)` tuples
   - Returns overlap decisions: exclude if any match > 30% identity
2. Handle edge cases:
   - Candidate with no CATH hits → passes
   - Candidate matching multiple CATH entries → report highest identity match
   - MMseqs2 binary not found → fail fast with clear error
3. Build a CLI script `scripts/run_overlap_filter.py` that:
   - Takes `--candidate-fasta`, `--cath-train-fasta`, `--output-json`
   - Outputs per-candidate overlap decision JSON

**TDD Gate**
1. RED: overlap checker misses a 35% identity match that should be excluded.
2. GREEN: fixture with known identity pairs produces correct exclude/pass decisions at boundary values (29%, 30%, 31%).

**Acceptance**
- Overlap detection uses sequence identity, not just PDB code matching.

---

### Task L2: Tier 1 Gold Standard Curation

**Goal**
- Curate 5-10 proteins with experimentally validated DRB1\*07:01-restricted epitopes.

**Manual Steps (User-Executed)**

This task involves manual scientific curation. The user performs the manual steps and produces intermediate artifacts. Scripts validate those artifacts before proceeding.

- [ ] **Step 1: IEDB query**
  - Query IEDB for: `MHC restriction = HLA-DRB1*07:01`, positive assay only, T-cell assay OR eluted ligand
  - Download the full export as TSV
  - Place at: `data/iedb_drb1_0701_positive.tsv`

- [ ] **Step 2: Extract source protein accessions**
  - Run: `python scripts/identify_sources.py` (already exists, may need minor path adjustment)
  - Review the output `data/accessions_sources.csv` for UniProt/PDB-mappable proteins

- [ ] **Step 3: Cross-reference with PDB**
  - For each source protein accession, check PDB for available structures
  - Filter: X-ray or cryo-EM < 2.5 Å, 100-500 AA, single chain preferred
  - Require: ≥ 2 experimentally confirmed epitope regions per protein
  - Produce candidate list as JSON at: `outputs/if/test_set/tier1_candidates.json`
  - Required JSON schema per entry:
    ```
    {
      "protein_id": "1XYZ_A",
      "uniprot_id": "P12345",
      "pdb_id": "1XYZ",
      "chain": "A",
      "sequence_length": 250,
      "resolution": 1.8,
      "experimental_epitopes": [
        {"start_0b": 45, "end_0b": 60, "assay_type": "T-cell", "iedb_ref": "..."},
        {"start_0b": 120, "end_0b": 135, "assay_type": "EL", "iedb_ref": "..."}
      ],
      "selection_reason": "..."
    }
    ```

- [ ] **Step 4: Validate candidate list (scripted)**
  - Run: `python scripts/validate_tier1.py --candidates outputs/if/test_set/tier1_candidates.json --cath-train-fasta <path>`
  - Script checks:
    - JSON schema completeness
    - Sequence length within 100-500
    - Resolution ≤ 2.5 Å
    - ≥ 2 experimental epitopes per protein
    - MMseqs2 overlap against CATH training → flags or excludes
  - Output: validated candidate list + overlap annotations

- [ ] **Step 5: Manual review of validation results**
  - Review overlap exclusions: are they correct?
  - Review remaining candidates: are epitopes in structurally interesting regions?
  - Finalize Tier 1 selection (5-10 proteins)
  - Update `tier1_candidates.json` with `"accepted": true/false` per entry

**TDD Gate**
1. RED: Tier 1 validator accepts a candidate missing experimental epitopes or with sequence length outside range.
2. GREEN: validator rejects incomplete candidates and flags CATH overlaps correctly.

**Acceptance**
- 5-10 gold standard proteins with traceable experimental epitope provenance.

---

### Task L3: Tier 2 Automated Pre-screening Pipeline

**Goal**
- Select 30-50 structurally diverse proteins where both the epitope head and NetMHCIIpan independently agree on significant DRB1\*07:01 presentation risk.

**Actions**
1. Build `inverse_folding/evaluation/prescreen.py`:
   - `prescreen_netmhciipan(protein_id, sequence, netmhciipan_binary, allele)`:
     - Run sliding window k ∈ [12, 25]
     - Count windows with %Rank_EL < 2%
     - Return `(n_strong_windows, mean_best_rank)`
   - `prescreen_epitope_head(protein_id, sequence, predictor)`:
     - Call `predict_protein(sequence, allele_idx=0)`
     - Return `(global_risk, n_hotspot_positions)`
   - `apply_tier2_filters(nmp_result, head_result, config)`:
     - Pass if `n_strong_windows >= 5` AND `global_risk >= risk_median_threshold`
     - The `risk_median_threshold` is computed as the median of all candidates' global risk in the current batch

2. Build `inverse_folding/evaluation/cath_topology.py`:
   - `assign_topology(protein_id, cath_domain_list_path)`:
     - Look up CATH topology code for the protein's PDB code
     - Return topology string (e.g., "1.10.490") or null
   - `sample_diverse(candidates_df, max_per_topology=2, target_total=50)`:
     - Group candidates by CATH topology
     - Sample up to `max_per_topology` per topology, prioritizing higher NMP signal
     - Ensure coverage across architecture classes (all-alpha, all-beta, alpha/beta, alpha+beta)

3. Build CLI `scripts/prescreen_tier2.py`:
   - Input: `--candidate-fasta-dir`, `--cath-train-fasta`, `--epitope-ckpt`, `--netmhciipan-bin`, `--cath-domain-list`, `--output-dir`
   - Pipeline: load candidates → MMseqs2 overlap filter → NetMHCIIpan batch screen → epitope head batch screen → CATH topology sampling → write `tier2_prescreened.parquet`

4. Build SLURM script `scripts/submit_prescreen_tier2.slurm` following `submit_cnn_enhance.slurm` template:
   - GPU partition (for epitope head inference)
   - Standard sections: Paths, Environment, Diagnostics, Launch

**Manual Steps (User-Executed)**

- [ ] **Step 1: Prepare PDB candidate pool**
  - Download PDB entries: single chain, 100-500 AA, X-ray < 2.5 Å, deposited before 2023
  - Exclude membrane proteins, disordered proteins
  - Extract sequences to FASTA files in a candidate directory
  - Place at: cluster `work/if/test_set/tier2_candidates/`

- [ ] **Step 2: Run automated pre-screening (on cluster)**
  - Submit: `sbatch scripts/submit_prescreen_tier2.slurm`
  - Monitor: check `logs/` for completion

- [ ] **Step 3: Manual review of diversity-sampled candidates**
  - Review `tier2_prescreened.parquet`: are the selected proteins scientifically interesting?
  - Check topology coverage: are major CATH classes represented?
  - Manually override if needed (add/remove specific proteins with documented reason)
  - Finalize selection (30-50 proteins)

**TDD Gate**
1. RED: pre-screener passes a protein with only 3 strong NMP windows (threshold is 5).
2. GREEN: filter correctly applies both NMP and head thresholds, and topology sampler respects `max_per_topology` and total target.

**Acceptance**
- 30-50 structurally diverse proteins with dual-scorer agreement on immunogenicity signal.

---

### Task L4: Tier 3 Therapeutic Protein Selection

**Goal**
- Select 5-10 therapeutic proteins with known or suspected clinical immunogenicity.

**Manual Steps (User-Executed)**

- [ ] **Step 1: Literature search**
  - Search for therapeutic proteins with:
    - Published clinical immunogenicity data (ADA incidence)
    - Available PDB structure (or high-confidence AF2 prediction)
    - Relevance to the de-immunization narrative
  - Candidate categories: antibody scaffolds, enzyme therapeutics, cytokines
  - Specific candidates to consider:
    - Adalimumab framework (known immunogenicity)
    - Asparaginase (bacterial origin, highly immunogenic)
    - Interferon-beta (clinical ADA data available)
    - IL-2 (aldesleukin, well-characterized)
    - Streptokinase (if PDB available)

- [ ] **Step 2: Produce candidate list**
  - Place at: `outputs/if/test_set/tier3_candidates.json`
  - Required JSON schema per entry:
    ```
    {
      "protein_id": "1D6R_A",
      "name": "adalimumab Fab",
      "pdb_id": "1D6R",
      "chain": "A",
      "sequence_length": 214,
      "resolution": 2.0,
      "structure_source": "xray",
      "literature_evidence": "Clinical ADA incidence 5-12%, Bartelds et al. 2011",
      "selection_reason": "..."
    }
    ```

- [ ] **Step 3: Run NetMHCIIpan screen**
  - For each candidate, run NetMHCIIpan DRB1\*07:01, k ∈ [12,25]
  - Require: ≥ 3 windows at %Rank < 5%
  - Can reuse existing `immunogenicity.py` wrapper

- [ ] **Step 4: Validate candidates (scripted)**
  - Run: `python scripts/validate_tier3.py --candidates outputs/if/test_set/tier3_candidates.json`
  - Script checks: schema completeness, NMP signal, PDB availability
  - Finalize selection (5-10 proteins)

**TDD Gate**
1. RED: Tier 3 validator accepts a candidate without literature evidence or with insufficient NMP signal.
2. GREEN: validator rejects incomplete candidates and correctly applies NMP signal threshold.

**Acceptance**
- 5-10 therapeutic proteins with literature immunogenicity evidence and NMP-confirmed DRB1\*07:01 signal.

---

### Task L5: Artifact Generation and Test Set Assembly

**Goal**
- For every accepted test protein across all tiers, generate the full set of pre-computed WT artifacts and assemble the final `test_proteins.parquet`.

**Actions**
1. Build `inverse_folding/evaluation/assembly.py`:
   - `extract_wt_sequence(pdb_path)`:
     - Extract single-chain sequence from PDB file
     - Return FASTA string
   - `extract_backbone(pdb_path)`:
     - Extract N, Cα, C, O coordinates
     - Write clean backbone PDB
   - `generate_wt_artifacts(protein_entry, predictor, netmhciipan_binary)`:
     - Run epitope head → hotspot map + global risk
     - Run NetMHCIIpan → window scores + aggregated metrics
     - Return artifact dict matching the frozen schema
   - `assemble_test_set(tier1_entries, tier2_entries, tier3_entries)`:
     - Merge all tiers into one DataFrame
     - Validate every row against the frozen schema
     - Write `test_proteins.parquet` + summary JSON
   - `validate_test_set(parquet_path)`:
     - Load and re-validate schema
     - Check no duplicate `protein_id`
     - Check tier counts match expectations
     - Check no CATH overlap flags are True (they should have been excluded)

2. Build CLI `scripts/assemble_if_test_set.py`:
   - Input: `--tier1-json`, `--tier2-parquet`, `--tier3-json`, `--pdb-dir`, `--epitope-ckpt`, `--netmhciipan-bin`, `--output-dir`
   - Pipeline: load all tiers → extract WT sequences + backbones → generate WT artifacts → assemble → validate → write

3. Build SLURM script `scripts/submit_assemble_test_set.slurm`:
   - GPU partition (for epitope head inference)
   - Standard sections following `submit_cnn_enhance.slurm` template

**TDD Gate**
1. RED: assembly accepts a protein with missing WT hotspot map or duplicate `protein_id`.
2. GREEN: assembler produces schema-valid parquet, rejects duplicates, and summary counts match input tier sizes.

**Acceptance**
- `test_proteins.parquet` is assembled with complete WT artifacts for all accepted test proteins.

---

### Task L6: End-to-End Evaluation Pipeline Verification

**Goal**
- Verify that the existing evaluation pipeline (`scripts/evaluate_if.py`) runs end-to-end on the curated test set, producing all required artifact tables.

**Actions**
1. Run a smoke evaluation using a small subset (2-3 proteins) of the curated test set:
   - Use the Module K baseline checkpoint (or a mock if K is not yet complete)
   - Execute full pipeline: PDB → generate → ESMFold → TM-align → epitope head → NetMHCIIpan → aggregate
2. Verify output artifact bundle:
   - `structural_metrics.csv` matches frozen schema
   - `immunogenicity_head.csv` matches frozen schema
   - `immunogenicity_nmp.csv` matches frozen schema
   - `comparison.csv` matches frozen schema
   - `summary.json` contains all required counters
3. Verify WT baselines are consumed correctly:
   - Comparison deltas reference the pre-computed WT values from `test_proteins.parquet`

**TDD Gate**
1. RED: evaluate_if.py crashes on the curated test set or produces schema-violating outputs.
2. GREEN: smoke evaluation produces complete, schema-valid artifact bundle with aligned row counts.

**Acceptance**
- The evaluation pipeline is verified on the curated test set and ready for Module M/N consumption.

---

### Task L7: Release Gates and Test Set Freeze

**Goal**
- Prove the test set is complete, clean, and ready for freezing.

**Actions**
1. Produce final curation summary:
   - Total proteins per tier
   - CATH overlap exclusion count
   - Epitope-head training overlap annotation count
   - NetMHCIIpan signal statistics per tier
   - CATH topology coverage (Tier 2)
   - Literature evidence completeness (Tier 3)
2. Run release checks:
   - No CATH-overlapping proteins remain in the accepted set
   - Every accepted protein has complete WT artifacts
   - No duplicate protein IDs across tiers
   - Tier 2 covers ≥ 10 CATH topologies
   - Tier 1 has ≥ 5 proteins with ≥ 2 experimental epitope regions each
3. Write `outputs/if/test_set/curation_ledger.json` with full provenance
4. Record in `LOG.md`

**TDD Gate**
1. RED: release summary passes with missing tier counts or incomplete artifact coverage.
2. GREEN: release validator requires all frozen counters and rejects partial test sets.

**Acceptance**
- Test set is frozen, documented, and ready for downstream consumption.

---

## 7. Current Open Decisions (Must Be Frozen Before/At Task Start)

1. Frozen:
   - CATH overlap uses MMseqs2 sequence identity > 30%, not PDB code matching
2. Frozen, but **never implemented as written** (see PROTOCOL C2):
   - epitope-head training overlap is annotated but not excluded
   - `head_train_overlap_flag` was written as a hardcoded `False` (`prescreen_tier2.py:683`) and no
     search was ever run, so the shipped column carries no information. Under the full-data Head
     the real rate is 14.8 % (0701) / 18.1 % (0401) of the main set at 30 % identity / 80 % query
     coverage, and Tier 1 is 15/15 (0701) by construction. The policy (annotate, split-report, do
     not exclude) still stands; the measurement is now mandatory.
3. **SUPERSEDED by §12 (Tier 2 v2, 2026-06-03)**:
   - ~~Tier 2 requires dual-scorer agreement (NMP ≥ 5 strong windows AND head risk in top 50%)~~
   - The dual-scorer rule is retired for the Tier 2 v2 rebuild. The epitope head is the guidance signal and must not gate test-set selection (selection bias inflates RF; see §8 risk #4). Tier 2 v2 selection is **NMP-only**. See §12.
4. **SUPERSEDED by `LOG.md` L0106 (2026-06-04)** — Tier 3 was removed from the main test set
   entirely; uricase is now a standalone case study under `if_test_set/uricases/`. The main set is
   Tier 1 + Tier 2 only.
   - ~~Tier 3 requires NMP ≥ 3 windows at %Rank < 5% AND literature evidence~~
5. Frozen:
   - MMseqs2 parameters: `--min-seq-id 0.3 --cov-mode 0 -c 0.8`
6. Frozen:
   - artifact storage under cluster path `work/if/test_set/`
7. Open:
   - exact PDB candidate pool source for Tier 2 (RCSB query vs existing curated list)
8. Open:
   - whether Tier 3 proteins with only AF2 structures (no X-ray) are acceptable
9. **RETIRED by §12 (Tier 2 v2)** — CATH-topology diversity sampling is incompatible with the
   structure-blind selection rule; `cath_topology` is null for every shipped Tier 2 row and the
   column is vestigial. The realized Tier 2 size is ~2.8k per allele, not 50.
   - ~~Tier 2 max_per_topology (proposed: 2) and target total (proposed: 50)~~

10. Added after this plan was frozen — see `PROTOCOL/if_benchmark_test_set_construction.md`:
    - `if_sequence_coverage >= 0.8` truncation filter (L0126)
    - `pdb_path` must resolve against disk rather than be constructed (L0170)
    - the epitope Head is the full-data production fixed-epoch checkpoint; `best.pt` is banned
    - h-maps are no longer a product of this pipeline

## 8. Risk Register (Execution-Level)

1. Tier 1 yield risk:
   - IEDB may have few DRB1\*07:01-restricted assays with PDB structures; fallback is to relax resolution or accept cryo-EM
2. Tier 2 compute risk:
   - Batch NetMHCIIpan scoring of ~1000+ candidates is expensive; pilot on 50 candidates first to estimate throughput
3. MMseqs2 over-exclusion risk:
   - 30% identity threshold may exclude distant homologs that are scientifically interesting; monitor exclusion rate
4. Head pre-screening bias risk:
   - Using the epitope head for Tier 2 selection means head-easy proteins are favored; NetMHCIIpan serves as independent check
5. Manual curation consistency risk:
   - Tier 1 and 3 require manual judgment; intermediate JSON artifacts + scripted validation gates mitigate but do not eliminate subjectivity
6. Topology coverage risk:
   - If candidate pool is biased toward common folds, Tier 2 diversity sampling may not reach 10 topologies; widen PDB query if needed

## 9. Codemap Annotation Policy

When a blocking ambiguity or behavior change appears:
1. If the change touches shared evaluation interfaces, update `inverse_folding/evaluation/schema.py` and record in `LOG.md`.
2. If the change affects downstream Module M/N contracts, update `PLAN_IF.md` cross-references.
3. Do not silently change curation criteria without reflecting them in this plan's frozen decisions.

## 10. Log Governance (`LOG.md`)

Follows the same schema and rules as `PLAN_IF.md` Section 9. Module identifier for this plan's entries is `L`.

## 11. Cross-References

- Module L in `PLAN_IF.md` is superseded by this file. `PLAN_IF.md` should be updated to reference `PLAN_DATA_SEL.md` for all test set curation and evaluation pipeline tasks.
- Module M (guidance sweep) depends on this plan's outputs: `test_proteins.parquet`, per-protein PDB/FASTA, WT hotspot maps.
- Module N (comparison baselines) depends on the same outputs.
- The test set persists across Phase 2-3 without modification. New cohorts (e.g., multi-allele) are additive.

---

## 12. Tier 2 v2 Rebuild — NMP-only, Density-Stratified (2026-06-03)

**Status:** active. Supersedes the §7.3 dual-scorer rule for Tier 2 only. Tier 1 (manual gold) and Tier 3 (uricase case study, expanded per `LOG.md` L0097) are **not** rebuilt; v2 only swaps the Tier 2 rows.

### 12.1 Motivation

The v1 Tier 2 selection used the epitope head twice — a `head_global_risk` top-K prefilter before NMP, and a `head_global_risk >= median` dual-scorer gate. The head is the **guidance signal** the reference flow optimizes against; selecting the test set with it biases the pool toward exactly where RF acts, inflating apparent performance (the §8 risk #4 that was accepted in v1). The v1 set is also a high-immunogenicity tail (`n_strong >= 5` + head gate), not representative across the immunogenicity spectrum.

**v2 goal:** a Tier 2 set whose selection uses **only NetMHCIIpan** (the independent validator), structure-blind, with a controlled **unimodal (Gaussian) marginal over immunogenicity density** so the bulk of statistical power sits in the mid-density regime while the tails are still populated. The diagnostic "pilot" subset is a separate, later concern (not this plan).

### 12.2 Frozen v2 decisions

1. **Selection signal:** NetMHCIIpan only. The epitope head must not enter Tier 2 selection (neither prefilter nor gate). Structure (burial/conservation/CATH) must not enter selection either — structure is handled as a shared task constraint and an evaluation readout, not a selection filter.
2. **Density axis:** `coverage_fraction` = (number of residues covered by ≥1 strong window with %Rank_EL < 2%) / `sequence_length`. Bounded [0,1], length-normalized. This is the binning axis.
3. **No floor:** sample from lowest to highest density; the near-zero-density bin is included (characterizes RF behavior across the full spectrum, including already-safe proteins).
4. **Target marginal:** unimodal Gaussian over `coverage_fraction`, centered at the density median (μ), modest peak (central:tail bin-count ratio ≈ 3:1), with a hard `min-per-bin` floor (≈ 40) so every bin has an objective count. Each bin target is capped by its availability in the pool (rare high-density bins take all available); capping is logged, never silent.
5. **Target size:** ≈ 3000 (current Tier 2 magnitude).
6. **Overlap reuse:** the v1 MMseqs2 CATH-overlap decision (the 75,425 overlap-passed candidate IDs in `_prescreen_head_results_*.parquet`) is reused as-is — overlap is train/test separation, independent of head/NMP. No MMseqs2 re-run.

### 12.3 Two-stage NMP procedure (cost control)

Running full multi-length NMP (k∈[12,25], 14 lengths) on all 75,425 candidates is prohibitive. Single-length-15 is a high-recall coarse detector (MHC-II binding is core-driven; a strong core almost always shows in a centered 15-mer), so:

| Stage | Operation | Scale | Cost |
|---|---|---|---|
| **S1 screen** | length-15-only NMP on the full overlap-passed pool (75,425) → coarse `coverage_fraction_15` | 75k × 1 length | ≈ 1× v1 NMP |
| **S1 sample** | uniform across `coverage_fraction_15` bins (no floor, lowest→highest), down to ≈ 5000 | →≈5000 | cheap |
| **S2 screen** | full k∈[12,25] NMP on the ≈5000 → accurate `coverage_fraction` + final artifact fields (`n_strong`, `mean_best_rank`) | 5k × 14 | ≈ 1× v1 NMP |
| **if_ready gate** | structure download + `build_if_ready_test_set` + `load_coords` gate on the ≈5000 **before** final sampling, so failures don't punch holes in the target distribution | — | — |
| **S3 sample** | Gaussian over accurate `coverage_fraction` (μ=median, peak:tail≈3:1, min-per-bin), down to ≈ 3000 → Tier 2 v2 | →≈3000 | — |

Total ≈ 2× the original NMP cost, for full-pool, head-free coverage. The double sampling (coarse uniform → accurate Gaussian) absorbs the length-15-vs-multi-length bin jitter: final bins are assigned on accurate multi-length density.

### 12.4 Materialization (after selection)

Tier 2 v2 selection → replace Tier 2 rows in the assembled `test_proteins_<allele>.parquet` (keep existing Tier 1 + updated Tier 3) → download structures for any new Tier 2 protein_ids → `build_if_ready_test_set.py` → `precompute_h_maps.py` rerun against the updated IF-ready parquet (B2). Both alleles (HLA-DRB1*07:01 and HLA-DRB1*04:01).

> **Amended after execution.** Tier 3 was removed from the main set the day after this section
> ran (L0106), so the materialization is Tier 1 + Tier 2 only. The `precompute_h_maps.py` step is
> also no longer part of the release — RF Fusion V2 does not read an h-map. See
> `PROTOCOL/if_benchmark_test_set_construction.md` §"What is no longer a product".

### 12.5 Implementation surface

- `inverse_folding/evaluation/immunogenicity.py`: already supports `pep_lengths`; add a pure `compute_coverage_fraction(window_df, seq_len)`.
- `inverse_folding/evaluation/prescreen.py` (or a new `sampling.py`): pure `sample_uniform_bins(...)` and `sample_gaussian_bins(...)` returning selected ids + realized histogram (TDD).
- `scripts/prescreen_tier2.py`: extend with `--nmp-screen-lengths`, `--selection-mode {dual_scorer,nmp_only}`, `--reuse-overlap-ids`, and `--sample {none,uniform,gaussian}` + params; the dual_scorer default path stays byte-equivalent for backward compatibility.
- SLURM: extend `submit_prescreen_tier2.slurm` with env overrides for the new mode (no new file).

### 12.6 Open knobs (tune at runtime, log realized values)

- Exact `n_bins`, μ, peak:tail ratio, min-per-bin — finalized against the realized S1/S2 density histograms; the realized target histogram is logged and (per user) reviewed before S3 commits.
- if_ready yield buffer: S1 target (≈5000) carries margin so post-if_ready survivors still comfortably exceed the ≈3000 S3 target.
