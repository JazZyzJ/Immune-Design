# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-03-26T22:30:00-04:00
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
- **Open**: multi-allele extension (DRB1*15:01) — not v1 scope

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
- **Note**: 集群源码未同步——验证用的是旧版代码，需要 `git pull` 后重新确认
- **Open**: multi-seed panel (decision pending — PLAN_IF.md §6.5)

---

## Module L → Data Selection (PLAN_DATA_SEL.md)

- **Code**: plan drafted (`PLAN_DATA_SEL.md`), no implementation yet
- **Cluster**: N/A
- **Key Data**: N/A (no test set curated yet)
- **Blocker**: this is the current critical path
  - Tier 1 (gold standard): needs IEDB query + PDB cross-reference
  - Tier 2 (computational): needs NetMHCIIpan batch screening + epitope head scoring
  - Tier 3 (therapeutic): needs literature search
- **Artifacts needed**:
  - `outputs/if/test_set/test_proteins.parquet`
  - `outputs/if/test_set/wt_hotspot_maps.parquet`
  - per-protein PDB/FASTA files
- **Feeds**: all downstream evaluation (M, N, and all F3/F4 subfigures)

---

## Module M: Classifier Guidance

- **Code**:
  - M0 (guidance contract): done — `inverse_folding/guidance/config.py`
  - M1 (scoring bridge): **not found** — `inverse_folding/guidance/scoring_bridge.py` does not exist
  - M2 (reweighting): done — `inverse_folding/guidance/reweighting.py`
  - M3 (sweep runner): **not found** — `scripts/run_if_guidance_sweep.py` does not exist
  - M4 (failure analysis): not started
  - SLURM script: **not found** — `scripts/submit_if_guidance_sweep.slurm` does not exist
  - Tests: `tests/inverse_folding/test_module_m_guidance_contract.py` only (scoring_bridge test does not exist)
- **Cluster**: not run (blocked by Module L — no test set)
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
| Guidance code (M0, M2) | `inverse_folding/guidance/` | config.py + reweighting.py |

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
| IF test set | N/A | not curated yet |
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
| Multi-allele analysis | F2 | multi-allele head | no | blocked by v1 single-allele scope |
| IF Benchmark | F3 | L (test set) | partial | DPLM baseline validated on CATH (scTM=0.87 median) |
| Structure Self-Consistency | F3 | L (ESMFold wrapper) | no | |
| Pareto Frontier | F3 | L + M sweep | no | M1/M3 code missing |
| Local Resampling | F3 | L + M | no | |
| Diversity ablation | F3 | L + M | no | |
| Uricase schematic | F4 | F3 pipeline | no | future |
| Phylogenetic Tree | F4 | uricase data | no | future |
| T cell assay | F4 | wet lab | no | future |
| Functionality | F4 | wet lab | no | future |
