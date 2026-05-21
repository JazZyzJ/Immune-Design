# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-05-20T00:00:00+08:00
> **Branch**: dev_head

---

## Epitope Head (Modules A-J)

- **Code**: complete
- **Cluster**: trained, checkpoint frozen
  - best run: `cnn_himp_v1_npoff_drb0701_seed42` (himp v1 variant, `npoff` mechanism)
  - checkpoint: `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_v1_npoff_drb0701_seed42/runs/LC1/seed_42/best.pt`
  - config: `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_v1_npoff_drb0701_seed42/runs/LC1/seed_42/resolved_config.yaml`
  - allele: HLA-DRB1*07:01 (single allele, v1 scope)
- **Key Data** (cnn_himp_v1_npoff_drb0701_seed42, best epoch 45 / 50 total, no early stopping):
  - held-out pp_ap (PR-AUC): **0.3497**
  - held-out pp_auc (ROC-AUC): **0.9532**
  - per_protein_auc: 0.9371
  - pp_recall@50: 0.7017, pp_recall@100: 0.7849
  - training data: 1,056 proteins / 1,537 chunks; val: 126 proteins / 204 chunks
  - encoder ablation: **done** (Module H, E1 winner — see ablation block below)
  - NetMHCIIpan benchmark comparison: **done** on prior `cnn_enhance_v2/LC1` checkpoint (see Benchmark block below); not re-run on npoff variant yet
- **Run comparison** (all LC1 / seed_42, sorted by pp_ap):
  | Run | pp_ap | pp_auc | Epochs | Date |
  |-----|-------|--------|--------|------|
  | cnn_himp_v1_npoff_drb0701 | **0.3497** | 0.9532 | 45/50 | 2026-04-25 |
  | LC1_lite_aug | 0.3027 | 0.9636 | 21/36 | 2026-03-23 |
  | cnn_strict_full | 0.2874 | 0.9571 | 32/47 | 2026-03-05 |
  | LC1_lite | 0.2520 | — | —/— | 2026-03-19 |
  | cnn_balanced_full | 0.1359 | — | —/— | 2026-03-05 |
- **Encoder ablation** (Module H, `run/ablation/encoder_v2/`, 3 seeds, strict val):
  | Encoder | pp_AUC | pp_AP | Recall@50 | Recall@100 | Latency L=512 |
  |---------|--------|-------|-----------|------------|---------------|
  | E0 | 0.853±0.008 | 0.041±0.011 | 0.239 | 0.313 | 150 ms |
  | **E1 (CNN, winner)** | **0.972±0.002** | **0.272±0.018** | **0.705** | **0.773** | 2.9 ms |
  | E2 | 0.763±0.011 | 0.018±0.004 | 0.104 | 0.158 | 4.5 ms |
- **NetMHCIIpan benchmark** (`run/benchmark/epi/`, HLA-DRB1*07:01, strict val, 126 proteins, on prior `cnn_enhance_v2/LC1` checkpoint):
  - head (ours): pp_auc=0.9587, pp_ap=0.3033, recall@50=0.706, recall@100=0.742
  - NMP (NetMHCIIpan): pp_auc=0.9802, pp_ap=0.3450, recall@50=0.777, recall@100=0.833
  - NMP > head on this checkpoint; npoff variant (pp_ap=0.3497) now exceeds NMP — re-run NMP comparison on npoff before claiming
- **Artifacts**: `InferencePredictor` verified, `predict_protein()` API stable
- **Multi-allele extension**: **done** (npoff alignment for DRB0401; DRB1501 pending npoff re-train)
  - DRB1*07:01: see above — `cnn_himp_v1_npoff_drb0701_seed42`, pp_ap=**0.3497**, pp_auc=**0.9532** (epoch 45/50, 1,056 training proteins)
  - DRB1*04:01: checkpoint at `run/epitope_head/cnn_himp_v1_npoff_drb0401_seed42/runs/LC1/seed_42/best.pt` — pp_ap=**0.3165**, pp_auc=**0.9627**, per_protein_auc=0.9714 (epoch 42/50, 1,490 training proteins / 2,191 chunks, p_aug=0.20 effective_fraction≈0.097)
  - DRB1*15:01: **still on `LC1_drb1501_aug`** — pp_ap=**0.1721**, pp_auc=**0.8492** (epoch 32, p_aug=0.0, 1,236 training entries); npoff variant **not yet trained**, methodology misaligned with 0701/0401
  - manifests: `outputs/manifests/drb0401/` (1,865 proteins), `outputs/manifests/drb1501/` (1,536 proteins)
  - Note: DRB1501 lag likely driven by smaller training set + augmentation off; npoff re-train should close at least the methodology gap
- **Open**:
  - DRB1501 `cnn_himp_v1_npoff_drb1501_seed42` not yet trained — needed to keep three-allele comparison apples-to-apples
  - NetMHCIIpan benchmark re-run on `cnn_himp_v1_npoff_drb0701` to update head-vs-NMP comparison with the new SOTA head
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
  - SLURM: `MODE=level1 sbatch scripts/submit_if_baselines.slurm`
  - Reads M3 eta=0 candidates, picks argmin global_risk
- **N2 (ProteinMPNN baseline + optional NMP filter)**: implemented (2026-05-08)
  - CLI: `scripts/run_proteinmpnn_baseline.py` (wraps vendored `DRAKES/drakes_protein/ProteinMPNN/`)
  - SLURM: `MODE=proteinmpnn ALLELE=... [APPLY_NMP_FILTER=1] sbatch scripts/submit_if_baselines.slurm`
  - Generates ``--num-seq-per-target`` designs per PDB, optional argmin-NMP-risk selection per protein
- **N3 (Level 3 DRAKES)**: **dropped** — not needed for paper; can be added as reviewer response if requested
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

- **Status**: C0/C1 **code ready, cluster not run**. C2/C3 still stubbed. **D1 monitor-only controller code ready (2026-05-21)**, cluster smoke pending.
- **Math foundation**: `doc/Reference_Flow_Derivation.md` — Tasks 0, A, C complete; Task B framework complete; §4.3.1 added 2026-05-08 documenting that the inference-time reparam refinement is *not* part of the DFM derivation; §4.7 D1-D5 reworked 2026-05-11/12 (L0082, L0083) so Task D is a continuous controller design (active blocks + hard counterfactual logits + residue-level EMA recommit).
- **Planned tiers**: C0 (unguided baseline) → C1 (sampling-only) → C2 (retrain) → C3 (FiLM+CFG). **Phase D** layers on top: D1 monitor-only (now implemented) → D2 hard counterfactual logits → D3 EMA commit/revisit → D4 dyn-schedule (deferred).
- **Implemented now**:
  - `inverse_folding/reference_flow/` package: YAML-validated config schema, base schedules + derivatives, amplification forms (`constant_one`, `linear_clamp`, `sigmoid`, `power`), per-protein h-shuffle control, denoiser-agnostic `PositionDependentDFMSampler`, and (2026-05-08) optional DPLM-style reparam re-mask refinement (`SamplerConfig.remask`)
  - preset YAML scaffolds: `c1_null`, `c1_linclamp`, `c1_sigmoid`, `c1_power`, `c1_shuffle` — all on n_steps=100 + remask=true (2026-05-08, aligned with DPLM paper inverse-folding eval default)
  - C0/C1 drivers: `scripts/run_if_phase_c0.py`, `scripts/run_if_phase_c1.py` (C1 now does h_maps↔test-set md5 alignment check at startup, fail-fast)
  - shared Phase C SLURM: `scripts/submit_if_phase_c.slurm` (`MODE=native|reference_flow`); MAX_ITER default 100 (was 50)
  - h_maps schema includes `sequence_md5` column (2026-05-08); old h_map parquets must be regenerated
  - **D1 monitor-only controller (2026-05-21, L0092)**:
    - `inverse_folding/reference_flow/controller_config.py` — strict D1 YAML schema with `controller_config_hash` provenance helper; preset `configs/d1_monitor.yaml`
    - `inverse_folding/reference_flow/head_scoring.py` — `OnlineHeadScorer` + `StaticWindowCache` (parquet + sidecar meta), fail-fast on `(seq_md5, allele, checkpoint_digest, head_config_hash, score_scale, window_k_min, window_k_max)` drift
    - `inverse_folding/reference_flow/controller.py` — `D1MonitorController` with refresh gating (`t_start`, `refresh_interval`), hard completion, window→block geometry (single / overlap / transitive merge), reliability factors (`g_time × g_comp × g_ent × g_ESS`)
    - `inverse_folding/reference_flow/sampler.py` — accepts optional `controller`; hook called between NaN check and unmask sampling; `controller=None` reproduces existing per-step trajectory bit-for-bit
    - `epitope_head/inference/predictor.py` — added `predict_proteins(records, allele_idx, window_batch_size)` ordered batch facade reusing one model
    - `scripts/run_if_phase_c1.py` extended with `--controller-config` + `--head-*` flags; emits `refresh_log.jsonl`, `controller_events.parquet`, `per_protein_summary.json`, `static_window_cache.{parquet,meta.json}` when D1 enabled; manifest gains 12 controller + head provenance fields. C1 generated.parquet schema unchanged.
    - `scripts/submit_if_phase_c.slurm` exposes `CONTROLLER_CONFIG`, `HEAD_CHECKPOINT`, `HEAD_CONFIG_DIR`, `HEAD_VARIANT_ID`, `HEAD_DEVICE`, `HEAD_WINDOW_BATCH_SIZE`, `HEAD_ALLELE_IDX` (also fixed a pre-existing missing line-continuation that was silently dropping EXTRA_FLAGS / WANDB_ARGS in reference_flow mode)
  - tests: schedule / amplification / sampler / config / remask / md5 alignment / C0 artifact contract + **D1 contract suites (controller_config, head_scoring, d1_controller, sampler_controller, batch_predictor, run_if_phase_c1_d1)** — **83 passing in reference_flow + epitope_head + scripts suites, 50 of them new for D1**
- **Acceptance status**:
  - TDD gates for C0/C1 contract layer: **pass**
  - TDD gates for D1 monitor-only controller (T1-T6): **pass** (50/50)
  - Existing Phase B + Module L contract suites: still green after D1 changes (**83 passing across reference_flow + epitope_head + script tests**)
  - End-to-end allele run on Della: **pending — must regenerate h_maps with sequence_md5 first; D1 also needs cluster smoke against a real epitope-head checkpoint to confirm the head decode path under DPLM tokenizer**
- **Operational note**:
  - C0 can run immediately on the assembled IF test set + B1 structures.
  - C1 actual experiment runs still depend on B2 test-set h-maps being materialized for the target allele; `h_normalized_corpus` arms additionally need B3 corpus stats sidecar.
  - **D1 monitor-only is enabled by passing `CONTROLLER_CONFIG=inverse_folding/reference_flow/configs/d1_monitor.yaml` plus `HEAD_CHECKPOINT=<best.pt>` / `HEAD_VARIANT_ID=<v>` to `scripts/submit_if_phase_c.slurm` under `MODE=reference_flow`. Empty `CONTROLLER_CONFIG` falls back to vanilla C1 (bit-equivalent to pre-D1).**
  - **2026-05-08 schema bump**: every existing h_maps parquet must be re-generated to include `sequence_md5` before C1 will load it; the c1 driver fails fast on missing or mismatching md5.

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
| Epitope head checkpoint (best, DRB0701) | `run/epitope_head/cnn_himp_v1_npoff_drb0701_seed42/runs/LC1/seed_42/best.pt` | pp_ap=0.3497, pp_auc=0.9532 |
| Epitope head checkpoint (legacy DRB0701 aug) | `run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.3027 (superseded by npoff) |
| Epitope head checkpoint (strict) | `run/epitope_head/cnn_strict_full/runs/LC1/seed_42/best.pt` | pp_ap=0.2874 |
| DPLM adapter checkpoint | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt` | validated |
| DPLM validation results | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/baseline_validation.json` | complete |
| CATH dataset (4.3) | `work/immune-design/cath_4.3/` | train/val/test |
| Augmentation registry | `work/immune-design/augmentation/mutation_registry_strict.parquet` | used by LC1_lite_aug |
| Manifests | `work/immune-design/manifests/` | epitope head training |
| Epitope head DRB0401 (npoff, current) | `run/epitope_head/cnn_himp_v1_npoff_drb0401_seed42/runs/LC1/seed_42/best.pt` | pp_ap=0.3165, pp_auc=0.9627 |
| Epitope head DRB0401 (legacy aug) | `run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.2962, pp_auc=0.9492 (superseded) |
| Epitope head DRB1501 | `run/epitope_head/LC1_drb1501_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.1721, pp_auc=0.8492 — npoff variant not yet trained |
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
