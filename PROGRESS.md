# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-04-27T00:47:26+08:00
> **Branch**: dev_head

---

## Epitope Head (Modules A-J)

- **Code**: complete
- **Cluster**: trained, checkpoint frozen
  - best run: `LC1_lite_aug` (runtime mutation augmentation, p_aug=0.20)
  - checkpoint: `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt`
  - config: `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/resolved_config.yaml`
  - allele: HLA-DRB1*07:01 (single allele, v1 scope)
- **Key Data** (LC1_lite_aug, best epoch 21 / 36 total, early-stopped):
  - held-out pp_ap (PR-AUC): **0.3027**
  - held-out pp_auc (ROC-AUC): **0.9636**
  - per_protein_auc: 0.9492
  - pp_recall@50: 0.7133, pp_recall@100: 0.7880
  - encoder ablation: not run yet
  - NetMHCIIpan benchmark comparison: not run yet
- **Run comparison** (all LC1 / seed_42):
  | Run | pp_ap | pp_auc | Epochs | Date |
  |-----|-------|--------|--------|------|
  | LC1_lite_aug | **0.3027** | 0.9636 | 21/36 | 2026-03-23 |
  | cnn_strict_full | 0.2874 | 0.9571 | 32/47 | 2026-03-05 |
  | LC1_lite | 0.2520 | — | —/— | 2026-03-19 |
  | cnn_balanced_full | 0.1359 | — | —/— | 2026-03-05 |
- **Artifacts**: `InferencePredictor` verified, `predict_protein()` API stable
- **Multi-allele extension**: **done** (training complete)
  - DRB1*04:01: checkpoint at `run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt` — pp_ap=**0.2962**, pp_auc=**0.9492** (epoch 26/48, p_aug=0.20, 1490 training entries)
  - DRB1*15:01: checkpoint at `run/epitope_head/LC1_drb1501_aug/runs/LC1/seed_42/best.pt` — pp_ap=**0.1721**, pp_auc=**0.8492** (epoch 32/18, p_aug=0.0, 1236 training entries)
  - manifests: `outputs/manifests/drb0401/` (1865 proteins), `outputs/manifests/drb1501/` (1536 proteins)
  - Note: DRB1501 performance significantly lower — smaller training set (1236 vs 1490 vs 1056) and augmentation was off
- **Open**: encoder ablation (Module H, PLAN_enco_abl.md) — not started
- **final_scripts**: `scripts/submit_cnn_enhance.slurm`, `scripts/submit_train_v2_cnn.slurm`, `scripts/submit_mutation_augmentation.slurm`

---

## Module K: DPLM Baseline

- **Code**: complete (adapter training pipeline, config schema, launcher)
- **Cluster**: trained + validated
  - experiment: `/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244`
  - checkpoint: `checkpoints/best.ckpt` → `step_10383.0-ppl_1.67-acc_median_55.74.ckpt`
  - config: `resolved_config.yaml`
  - training: 46 epochs (0–45), 2× GPU, ~21h wall time (2026-03-19 05:42 → 2026-03-20 02:45)
  - best val: acc_median=55.74, ppl=1.651, acc=55.19
  - CATH split: train=16,631 / val=1,516 / test=1,864
- **Key Data** (CATH test, 1,864 proteins, T=1.0, max_iter=10, argmax):
  - recovery: mean=0.5146, median=0.55
  - scTM: mean=0.8114, median=0.8702
  - scTM > 0.5 (foldability): **91.95%** (1,714/1,864) — smoke gate PASS
  - scTM > 0.8 (target): **68.24%** (1,272/1,864) — target gate PASS
  - pLDDT: mean=76.62
  - scRMSD: mean=7.15
  - structural collapse (scTM ≤ 0.5): 150 proteins (8.05%)
- **Artifacts**: baseline_validation.json, per_protein_metrics.csv, test_tau1.0.fasta, native.fasta
- **Note**: 集群源码已同步 (2026-04-01)
- **Open**: multi-seed panel (decision pending — PLAN_IF.md §6.5)

---

## Module L → Data Selection (PLAN_DATA_SEL.md)

- **Code**: L0-L5 implemented + tested (139 tests passing), 6 code review issues fixed
  - schema (`validate_test_protein_entry`), overlap (MMseqs2), prescreen (dual-scorer + CATH diversity), assembly, tier validators
  - CLIs: `run_overlap_filter.py`, `validate_tier1.py`, `validate_tier3.py`, `prescreen_tier2.py`, `assemble_if_test_set.py`, `download_tier2_candidates.py`
  - SLURM: `submit_prescreen_tier2.slurm`, `submit_assemble_test_set.slurm`, `submit_download_tier2.slurm`
  - Key fixes: validate_tier1 now runs MMseqs2 overlap; validate_tier3 enforces NMP signal; prescreen uses correct epitope_head API; assembly generates WT artifacts; overlap CLI includes no-hit candidates; sample_diverse guarantees cross-architecture-class coverage
- **Cluster**: test sets **assembled** for two alleles
- **Key Data**:
  - **HLA-DRB1\*07:01 test set** — `test_proteins_HLA-DRB1_07_01.parquet` — **3,141 proteins**
    | Tier | Count | mean_len | mean_risk | mean_nmp_strong |
    |------|-------|----------|-----------|-----------------|
    | 1 | 15 | 267.5 | -0.433 | 64.5 |
    | 2 | 3,000 | 250.3 | 3.649 | 85.1 |
    | 3 | 126 | 282.9 | 0.452 | 64.5 |
  - **HLA-DRB1\*04:01 test set** — `test_proteins_HLA-DRB1_04_01.parquet` — **3,140 proteins**
    | Tier | Count | mean_len | mean_risk | mean_nmp_strong |
    |------|-------|----------|-----------|-----------------|
    | 1 | 15 | 267.5 | -4.207 | 49.9 |
    | 2 | 2,998 | 207.9 | 3.128 | 113.4 |
    | 3 | 127 | 281.0 | -2.719 | 80.2 |
  - Prescreen pipeline (Tier 2): 225,821 → 75,425 (MMseqs2 CATH overlap) → 3,000 (head prefilter) → dual predictor filter (head + NMP) → assembly
  - NMP scoring: 3,000/3,000 complete (both alleles), no timeout
  - CATH topology filter: **frozen** (舍弃) — 最后一步只做 dual predictor (head + NMP) 筛选，不再做 CATH topology diversity sampling
- **Artifacts**:
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet` — 3,141 rows, 17 columns
  - `work/immune-design/if_test_set/test_proteins_summary_HLA-DRB1_07_01.json`
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet` — 3,140 rows, 17 columns
  - `work/immune-design/if_test_set/test_proteins_summary_HLA-DRB1_04_01.json`
  - `work/immune-design/if_test_set/fastas/` — 6,166 FASTA files
  - `work/immune-design/if_test_set/pdbs/0401/` — **2,943 .pdb** (RCSB experimental) + **192 .cif** (local AF snapshot rescue + RCSB .cif Track A + AFDB Track B) = 3,135 structures; **18 unresolved** uricase UniProts listed in `uniprot_final_failures.txt` (no AFDB prediction)
  - `work/immune-design/if_test_set/pdbs/0701/` — **2,874 .pdb** (RCSB experimental) + **244 .cif** (18 local AF + 123 RCSB cif + 102 AFDB + 1 D3BGR1 fix) = 3,118 structures; **23 unresolved** uricase UniProts listed in `uniprot_final_failures.txt` (no AFDB prediction)
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.parquet` — **3,072 IF-ready proteins** (Tier 1:15, Tier 2:2,948, Tier 3:109; 2,881 `.pdb` + 191 `.cif`); cleaned structures in `work/immune-design/if_test_set/pdbs_if_ready/0401/`; DPLM `load_coords()` gate: **3,072/3,072 exact sequence+length match**
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.parquet` — **3,079 IF-ready proteins** (Tier 1:15, Tier 2:2,962, Tier 3:102; 2,839 `.pdb` + 240 `.cif`); cleaned structures in `work/immune-design/if_test_set/pdbs_if_ready/0701/`; DPLM `load_coords()` gate: **3,079/3,079 exact sequence+length match**
  - IF-ready failure manifests: `if_ready/test_proteins_if_ready_HLA-DRB1_04_01.failures.csv` (68 failures: 50 unknown residues, 18 missing structures), `if_ready/test_proteins_if_ready_HLA-DRB1_07_01.failures.csv` (62 failures: 38 unknown residues, 24 missing structures)
  - Final-failure manifests (absolute paths):
    - `work/immune-design/if_test_set/pdbs/0401/uniprot_final_failures.txt` (18 UniProts, mostly Streptomyces / 小众真菌 / Uncharacterized TrEMBL entries)
    - `work/immune-design/if_test_set/pdbs/0701/uniprot_final_failures.txt` (23 UniProts, same pattern)
    - `pdb_final_failures.txt` — 0 entries on both (all RCSB 404s recovered via .cif tracks)
    - `download_failures_remaining.txt` — 45 (0401) / 98 (0701) historical chain-level RCSB 404 lines, superseded by rescue via .cif; kept for audit
  - intermediate: `_prescreen_{head,nmp}_results*.parquet`, `tier2_prescreened*.parquet`
- **Feeds**: all downstream evaluation (M, N, and all F3/F4 subfigures)
- **Open**:
  - 41 UniProts (18 + 23) have no AFDB prediction — planned to run **AF3** locally to fill these gaps before Phase C
  - L6 E2E verification + L7 release gates not yet run

---

## Module M: Classifier Guidance — SUPERSEDED (2026-04-07)

- **Status**: **SUPERSEDED**. Resampling-based "guidance" is post-hoc selection, not generation-time steering. Replaced by Phase C (reference flow).
- **Code**: M0-M3 preserved in `inverse_folding/guidance/`. Scoring bridge may be reusable for Phase C.
- **Cluster**: never run. No data produced.

---

## Module N: Comparison Baselines (Simplified)

- **Scope change (2026-04-04)**: Level 3 (DRAKES/DPO) **dropped** — focus shifted entirely to core reference flow contribution. Only Level 1 post-hoc filter retained as baseline.
- **N1 (Level 1 post-hoc filter)**: implemented (2026-04-02)
  - Code: `inverse_folding/baselines/level1_filter.py`
  - CLI: `scripts/run_if_level1_filter.py`
  - SLURM: `scripts/submit_if_level1_filter.slurm`
  - Reads M3 eta=0 candidates, picks argmin global_risk
- **N2 (Level 3 DRAKES)**: **dropped** — not needed for paper; can be added as reviewer response if requested
- **Blocked by**: Phase C (need unguided baseline candidate pool from C0)
- **Comparison structure for paper**:
  1. Level 1 post-hoc filter (N1) — from C0 candidates
  2. Sampling-only position-dependent schedule (C1 / Tier 0 ablation)
  3. Full position-dependent reference flow (C2-C3 / our method)

---

## Phase B: Experiment Preparation (NEW, 2026-04-07)

- **B1 (PDB download)**: **near-complete** — 0401: 2,943 .pdb + 192 .cif (AF rescue + RCSB cif + AFDB); 0701: 2,874 .pdb + 244 .cif. All RCSB 404s rescued via .cif tracks (`pdb_final_failures.txt` = 0 on both). Residual **41 UniProts** (18 on 0401 / 23 on 0701) have no AFDB prediction — listed in `pdbs/{0401,0701}/uniprot_final_failures.txt`; will be filled by **AF3 local prediction**
- **B1.5 (IF-ready structure/sequence cleaning)**: **complete for PDB/mmCIF-backed rows** — `scripts/build_if_ready_test_set.py` materialized resolved-backbone parquets and cleaned single-chain PDB/mmCIF structures. 0401: 3,072 ready / 68 failed; 0701: 3,079 ready / 62 failed. DPLM `load_coords()` exact sequence+length gate passes for all ready rows. Tier 3 uricases are now partially recovered via default chain `A` (0401:109, 0701:102). Remaining failures are missing structures or unknown residue symbols not covered by BioPython's extended PDB residue map.
- **B2 (h_i maps for test set)**: **must rerun on IF-ready parquets** — shared CLI `scripts/precompute_h_maps.py`, loader/validator `inverse_folding/evaluation/h_maps.py`, test-set SLURM `scripts/submit_precompute_h_test.slurm`; previous biological-sequence h-maps are invalid for Phase C if their length differs from the resolved backbone sequence. Resume is guarded by sidecar metadata, allele/checkpoint/config/source matching, and sequence-length checks.
- **B3 (h_i maps for CATH training set)**: **code ready, cluster not run** — same CLI with JSONL split filtering, CATH-specific sequence policy, guarded resume support, and required corpus-level `h_raw` stats; SLURM `scripts/submit_precompute_h_cath.slurm`
- **B4 (eval pipeline integration)**: **code ready, cluster not run** — new `scripts/evaluate_phase_c.py` consumes Phase C `generated.parquet` directly, supports `imm | struct | all` mode separation, writes schema-aligned `imm_head.parquet`, `imm_nmp.parquet`, `structural.parquet`, uses `scripts.infer_v1.build_predictor` for head inference, dispatches refold via `inverse_folding/evaluation/refold.py` (`esmfold` active, `af3` stub), and is wired into `scripts/submit_benchmark.slurm` as `MODE=phase_c`. Legacy `scripts/evaluate_if.py` removed.

---

## Phase C: Core Contribution — Reference Flow (NEW, 2026-04-07)

- **Status**: C0/C1 **code ready, cluster not run**. C2/C3 still stubbed.
- **Math foundation**: `doc/Reference_Flow_Derivation.md` — Tasks 0, A, C complete; Task B framework complete
- **Planned tiers**: C0 (unguided baseline) → C1 (sampling-only) → C2 (retrain) → C3 (FiLM+CFG)
- **Implemented now**:
  - `inverse_folding/reference_flow/` package: YAML-validated config schema, base schedules + derivatives, amplification forms (`constant_one`, `linear_clamp`, `sigmoid`, `power`), per-protein h-shuffle control, and denoiser-agnostic `PositionDependentDFMSampler`
  - preset YAML scaffolds: `c1_null`, `c1_linclamp`, `c1_sigmoid`, `c1_power`, `c1_shuffle`
  - C0/C1 drivers: `scripts/run_if_phase_c0.py`, `scripts/run_if_phase_c1.py`
  - shared Phase C SLURM: `scripts/submit_if_phase_c.slurm` (`MODE=native|reference_flow`)
  - tests: schedule / amplification / sampler / config + C0 artifact contract (**16 passing**)
- **Acceptance status**:
  - TDD gates for C0/C1 contract layer: **pass**
  - Existing Phase B + Module L contract suites: still green after Phase C changes (**46 passing**)
  - End-to-end allele run on Della: **pending**
- **Operational note**:
  - C0 can run immediately on the assembled IF test set + B1 structures.
  - C1 actual experiment runs still depend on B2 test-set h-maps being materialized for the target allele; `h_normalized_corpus` arms additionally need B3 corpus stats sidecar.

### Experimental results (TBD — fill per run after cluster completes)

> Evaluation pipeline reference: `doc/Reference_Flow_Derivation.md §6` for hypothesis framing; Module L §L0 for metric schema. Δrisk and Δn_strong are computed against WT (the input test protein), not against C0. One row per `run_id`; append rows as sweeps grow.

**C0 baseline** (DPLM native sampler on frozen Module K checkpoint)

| allele | run_id | n_proteins | n_designs | scTM mean | scTM median | % scTM > 0.5 | % scTM > 0.8 | recovery mean | pLDDT mean | head Δrisk vs WT | NMP Δn_strong vs WT | mean mutation_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HLA-DRB1*07:01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| HLA-DRB1*04:01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**C1 sampling-only** (position-dependent DFM; one row per `run_id`; record the full resolved config fields so future-you can trace what was run)

| allele | config preset | form | c | mu | kappa / p | base schedule | n_steps | h_source | shuffle? | seed | run_id | scTM mean | % scTM > 0.5 | recovery mean | head Δrisk | NMP Δn_strong | mean mutation_count | edit ratio ρ (H1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TBD | c1_null | constant_one | — | — | — | TBD | TBD | h_processed | no | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a |
| TBD | c1_linclamp | linear_clamp | TBD | 0.0 | — | TBD | TBD | TBD | no | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | c1_sigmoid | sigmoid | TBD | 0.0 | κ=TBD | TBD | TBD | TBD | no | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | c1_power | power | TBD | 0.0 | p=TBD | TBD | TBD | TBD | no | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | c1_shuffle | linear_clamp | TBD | 0.0 | — | TBD | TBD | TBD | yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**N1 post-hoc filter baseline** (consumes C0 candidate pool; required for H2 Pareto)

| allele | source C0 run_id | K candidates | n_proteins | scTM mean | head Δrisk vs WT | NMP Δn_strong vs WT | mean mutation_count |
|---|---|---|---|---|---|---|---|
| HLA-DRB1*07:01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| HLA-DRB1*04:01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Hypothesis readout (TBD — fill after sweep)

> `doc/Reference_Flow_Derivation.md §6`. Each hypothesis needs a direction (pass / fail / inconclusive) and one quantitative anchor. Blocked rows stay blocked until C2 exists.

| Hypothesis | Measurement | Arms compared | Quantitative anchor | Status |
|---|---|---|---|---|
| H1 edit localization | ρ = (edits@hotspot / N_hotspot) / (edits@non-hotspot / N_non-hotspot), stratified by `c` | c1_null vs c1_linclamp / c1_sigmoid / c1_power at matched scTM | TBD (expect ρ > 1, increasing in c) | pending |
| H2 Pareto dominance | (scTM, head Δrisk) and (scTM, NMP Δn_strong) Pareto fronts | N1 post-hoc filter vs each c1 arm | TBD (non-trivial region of dominance) | pending |
| H3 shuffle control | Targeted-advantage gap between real and shuffled h on (scTM, Δrisk, ρ) | c1_linclamp vs c1_shuffle (matched c) | TBD (expect targeted advantage collapses toward shuffle baseline) | pending |
| H4a / H4b retrain vs sample-only | same pipeline on c1 vs C2 retrained checkpoint | c1 arms vs C2 | — | blocked on C2 |
| H5 hotspot entropy gap | mean `H(p_θ(x_1^i \| x_t))` at hotspot − non-hotspot, matched (t, x_t) | c1 denoiser vs C2 denoiser | — | blocked on C2 |

### Failure / anomaly log (fill as encountered)

- NaN designs (protein_id, design_idx, step, t): TBD
- Structural collapse (scTM ≤ 0.5) proteins: TBD
- OOM-retried-on-CPU: TBD
- `g_max_cap` triggered (protein_id, max g_i): TBD

---

## Data Inventory

### Local (this repo)

| Artifact | Path | Status |
|----------|------|--------|
| Epitope head training data (strict) | `outputs/data/span_records_strict.parquet` | 23,988 rows |
| Epitope head training data (balanced) | `outputs/data/span_records_balanced.parquet` | 54,945 rows |
| DPLM vendor code | `inverse_folding/dplm/` | vendored, .git removed |
| Guidance code (M0-M3) | `inverse_folding/guidance/` | config + scoring_bridge + reweighting |
| Guidance sweep script | `scripts/run_if_guidance_sweep.py` | committed |
| Guidance SLURM | `scripts/submit_if_guidance_sweep.slurm` | committed |
| Data selection code (L0-L5) | `inverse_folding/evaluation/` | overlap, prescreen, cath_topology, assembly, tier_validators (139 tests) |
| Phase B h-map precompute | `scripts/precompute_h_maps.py`, `inverse_folding/evaluation/h_maps.py` | code ready; cluster B2/B3 runs pending |
| Tier 1 candidates | `outputs/if/test_set/tier1_candidates.json` | 15 proteins, 10-50% epitope coverage |
| Tier 2 candidate pool | `outputs/if/test_set/tier2_candidates_merged.fasta` | 226k seqs, transferred to cluster |
| Tier 2 entity IDs | `outputs/if/test_set/tier2_all_entity_ids.txt` | 113k RCSB entity IDs |
| Multi-allele manifests | `outputs/manifests/{drb0401,drb1501}/` | complete |

### Cluster (`/scratch/gpfs/KAIYIJIANG/zijie/`)

| Artifact | Path | Status |
|----------|------|--------|
| Epitope head checkpoint (best) | `run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.3027 |
| Epitope head checkpoint (strict) | `run/epitope_head/cnn_strict_full/runs/LC1/seed_42/best.pt` | pp_ap=0.2874 |
| DPLM adapter checkpoint | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt` | validated |
| DPLM validation results | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/baseline_validation.json` | complete |
| CATH dataset (4.3) | `work/immune-design/cath_4.3/` | train/val/test |
| Augmentation registry | `work/immune-design/augmentation/mutation_registry_strict.parquet` | used by LC1_lite_aug |
| Manifests | `work/immune-design/manifests/` | epitope head training |
| Epitope head DRB0401 | `run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.2962, pp_auc=0.9492 |
| Epitope head DRB1501 | `run/epitope_head/LC1_drb1501_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.1721, pp_auc=0.8492 |
| Tier 1 candidates | `work/immune-design/if_test_set/tier1_candidates.json` | 15 candidates, user review pending |
| Tier 2 merged FASTA | `work/immune-design/if_test_set/tier2_candidates_merged.fasta` | 226k chains |
| IF test set (0701) | `work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet` | 3,141 proteins (T1:15 T2:3000 T3:126) |
| IF test set (0401) | `work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet` | 3,140 proteins (T1:15 T2:2998 T3:127) |
| IF test set FASTAs | `work/immune-design/if_test_set/fastas/` | 6,166 files |
| IF test set structures | `work/immune-design/if_test_set/pdbs/{0401,0701}/` | 0401: 2,943 .pdb + 192 .cif; 0701: 2,874 .pdb + 244 .cif. RCSB 404s fully rescued. 18 / 23 UniProts still missing (no AFDB) — `uniprot_final_failures.txt`; AF3 planned |
| IF test set h-maps (B2) | `work/immune-design/if_test_set/h_maps/h_maps_DRB1_{07_01,04_01}.parquet` + `.meta.json` | TBD — cluster run pending |
| CATH training h-maps (B3) | `work/immune-design/cath_4.3/h_maps/h_maps_cath_DRB1_{07_01,04_01}.parquet` + `.meta.json` | TBD — cluster run pending |
| Phase C0 outputs | `work/immune-design/if_phase_c/c0/<allele_tag>/<run_id>/{generated.{parquet,fasta},run_config.yaml,manifest.json}` | TBD — cluster run pending |
| Phase C1 outputs | `work/immune-design/if_phase_c/c1/<allele_tag>/<run_id>/{generated.{parquet,fasta},run_config.yaml,manifest.json,trajectories/*.parquet}` | TBD — cluster run pending |
| Guidance sweep results | N/A | superseded by Phase C (Module M superseded 2026-04-07) |

---

## → Paper Readiness (derived from above)

| SubFigure | Category | Blocked by | Data Ready? | Notes |
|-----------|----------|-----------|-------------|-------|
| Allele Distribution | F1 | MA strategy | partial | pure analysis done |
| Model overview | F1 | — | partial | architecture doc + mermaid exist, formal figure not made |
| Clinics immunogenicity plot | F1 | — | no | concept only |
| EL Ability | F2 | NetMHCIIpan benchmark | partial | pp_ap=0.3027, comparisons pending |
| Synthetic point mutation | F2 | — | partial | hard negative code committed |
| Multi-allele analysis | F2 | multi-allele eval | partial | DRB0401 + DRB1501 trained, metrics TBD |
| Structural metrics on epitope | F2 | Tier 1 PDBs + h_i maps | no | 4 analyses: RSA vs h_i, SS distribution, B-factor, contact number |
| IF Benchmark | F3 | L (test set) | partial | DPLM baseline validated on CATH (scTM=0.87 median) |
| Structure Self-Consistency | F3 | L (ESMFold wrapper) | no | |
| Pareto Frontier | F3 | Phase C | no | reference flow vs post-hoc: scTM vs Δrisk |
| Emergent Ordering | F3 | Phase C | no | unmasking order correlates with h_i (core claim) |
| Ablation (Tier 0 vs 1) | F3 | Phase C | no | sampling-only vs full training |
| Uricase schematic | F4 | F3 pipeline | no | future |
| Phylogenetic Tree | F4 | uricase data | no | future |
| T cell assay | F4 | wet lab | no | future |
| Functionality | F4 | wet lab | no | future |
