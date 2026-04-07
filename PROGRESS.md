# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-04-07T22:00:00-04:00
> **Branch**: dev_head

---

## Epitope Head (Modules A-J)

- **Code**: complete
- **Cluster**: trained, checkpoint frozen
  - best run: `LC1_lite_aug` (runtime mutation augmentation, p_aug=0.20)
  - checkpoint: `/scratch/network/zc1519/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt`
  - config: `/scratch/network/zc1519/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/resolved_config.yaml`
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
  - experiment: `/scratch/network/zc1519/run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244`
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
  - `work/immune-design/if_test_set/pdbs/` — **empty** (PDB structures not downloaded yet)
  - intermediate: `_prescreen_{head,nmp}_results*.parquet`, `tier2_prescreened*.parquet`
- **Feeds**: all downstream evaluation (M, N, and all F3/F4 subfigures)
- **Open**:
  - PDB structure files not yet downloaded (`pdbs/` empty) — needed for inverse folding input
  - L6 E2E verification + L7 release gates not yet run

---

## Module M: Classifier Guidance

- **Code**: M0-M3 complete, committed and synced to cluster
  - M0 (guidance contract): `inverse_folding/guidance/config.py`
  - M1 (scoring bridge): `inverse_folding/guidance/scoring_bridge.py`
  - M2 (reweighting): `inverse_folding/guidance/reweighting.py`
  - M3 (sweep runner): `scripts/run_if_guidance_sweep.py`
  - M4 (failure analysis): skipped (non-critical-path)
  - SLURM: `scripts/submit_if_guidance_sweep.slurm`
  - Tests: `test_module_m_guidance_contract.py` + `test_module_m_scoring_bridge.py`
- **Cluster**: not run — test set assembled, **blocked by PDB download + sweep execution**
- **Key Data**: N/A (sweep not executed)
  - expected outputs per eta: scTM, delta_risk (head), delta_risk (NetMHCIIpan), mutation_count
  - eta grid: {0, 0.5, 1, 2, 5, 10}, K=8 candidates
- **Artifacts needed**: per-eta FASTA + eval CSV + pareto_summary.json

---

## Module N: Comparison Baselines

- **Code**: not started
- **Cluster**: N/A
- **Key Data**: N/A
- **Blocked by**: Module M (need shared eval substrate first)
- **Priority**: lowest in v1

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
| Tier 1 candidates | `outputs/if/test_set/tier1_candidates.json` | 15 proteins, 10-50% epitope coverage |
| Tier 2 candidate pool | `outputs/if/test_set/tier2_candidates_merged.fasta` | 226k seqs, transferred to cluster |
| Tier 2 entity IDs | `outputs/if/test_set/tier2_all_entity_ids.txt` | 113k RCSB entity IDs |
| Multi-allele manifests | `outputs/manifests/{drb0401,drb1501}/` | complete |

### Cluster (`/scratch/network/zc1519/`)

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
| IF test set PDBs | `work/immune-design/if_test_set/pdbs/` | **empty** — not downloaded yet |
| Guidance sweep results | N/A | not run yet |

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
| IF Benchmark | F3 | L (test set) | partial | DPLM baseline validated on CATH (scTM=0.87 median) |
| Structure Self-Consistency | F3 | L (ESMFold wrapper) | no | |
| Pareto Frontier | F3 | L + M sweep | no | M0-M3 code ready (uncommitted), blocked by test set |
| Local Resampling | F3 | L + M | no | |
| Diversity ablation | F3 | L + M | no | |
| Uricase schematic | F4 | F3 pipeline | no | future |
| Phylogenetic Tree | F4 | uricase data | no | future |
| T cell assay | F4 | wet lab | no | future |
| Functionality | F4 | wet lab | no | future |
