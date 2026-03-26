# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-03-26T14:00:00-04:00
> **Branch**: dev_head

---

## Epitope Head (Modules A-J)

- **Code**: complete
- **Cluster**: trained, checkpoint frozen
  - checkpoint: `/scratch/network/zc1519/run/epitope_head/best.pt` <!-- verify path -->
  - allele: HLA-DRB1*07:01 (single allele, v1 scope)
- **Key Data**:
  - training: TBD (fill from cluster logs — final loss, epochs, learning rate)
  - held-out EL AUC: TBD
  - held-out EL PR-AUC: TBD
  - encoder ablation: not run yet
  - NetMHCIIpan benchmark comparison: not run yet
- **Artifacts**: `InferencePredictor` verified, `predict_protein()` API stable
- **Open**: multi-allele extension (DRB1*15:01) — not v1 scope

---

## Module K: DPLM Baseline

- **Code**: complete (adapter training pipeline, config schema, launcher)
- **Cluster**: trained
  - checkpoint: TBD <!-- fill exact path from cluster -->
  - config: TBD
  - training: TBD (fill — epochs, final loss, wall time)
- **Key Data**:
  - CATH test recovery rate: TBD
  - CATH test scTM distribution: TBD (mean, median, % > 0.5, % > 0.8)
  - CATH test pLDDT: TBD
  - failure cases: TBD (proteins with structural collapse)
- **Artifacts**: baseline release package (checkpoint + resolved config)
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
  - M1 (scoring bridge): done — `inverse_folding/guidance/scoring_bridge.py`
  - M2 (reweighting): TBD (check if `inverse_folding/guidance/reweighting.py` exists)
  - M3 (sweep runner): done — `scripts/run_if_guidance_sweep.py`
  - M4 (failure analysis): not started
  - SLURM script: done — `scripts/submit_if_guidance_sweep.slurm`
  - Tests: `tests/inverse_folding/test_module_m_scoring_bridge.py`, `test_module_m_guidance_contract.py`
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
| Guidance code (M0-M3) | `inverse_folding/guidance/` | uncommitted on dev_head |

### Cluster (`/scratch/network/zc1519/`)

| Artifact | Path | Status |
|----------|------|--------|
| Epitope head checkpoint | `run/epitope_head/...` | TBD — verify exact path |
| DPLM adapter checkpoint | `run/if/dplm_v1_adapter/...` | TBD — verify exact path |
| CATH training data | `work/...` | TBD — verify |
| IF test set | N/A | not curated yet |
| Guidance sweep results | N/A | not run yet |

> **TBD items**: fill from cluster session — run `ls`, read SLURM logs, extract key metrics.

---

## → Paper Readiness (derived from above)

| SubFigure | Category | Blocked by | Data Ready? | Notes |
|-----------|----------|-----------|-------------|-------|
| Allele Distribution | F1 | MA strategy | partial | pure analysis done |
| Model overview | F1 | — | partial | architecture doc + mermaid exist, formal figure not made |
| Clinics immunogenicity plot | F1 | — | no | concept only |
| EL Ability | F2 | NetMHCIIpan benchmark | partial | EL AUC done, comparisons pending |
| Synthetic point mutation | F2 | — | partial | hard negative code committed |
| Multi-allele analysis | F2 | multi-allele head | no | blocked by v1 single-allele scope |
| IF Benchmark | F3 | L (test set) | no | DPLM trained but no formal eval |
| Structure Self-Consistency | F3 | L (ESMFold wrapper) | no | |
| Pareto Frontier | F3 | L + M sweep | no | code ready, not run |
| Local Resampling | F3 | L + M | no | |
| Diversity ablation | F3 | L + M | no | |
| Uricase schematic | F4 | F3 pipeline | no | future |
| Phylogenetic Tree | F4 | uricase data | no | future |
| T cell assay | F4 | wet lab | no | future |
| Functionality | F4 | wet lab | no | future |
