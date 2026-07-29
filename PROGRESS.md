# Progress Snapshot

> **Purpose**: Live status of each workstream. Overwritten (not append-only).
> Read this first every session. For event history see `LOG.md`.
>
> **Last synced**: 2026-07-29T01:04:17-04:00
> **Branch**: fusion_rf_refine

---

## Epitope Head (Modules A-J)

> **Wave-4 CONVERGED (2026-06-28) — STANDALONE head branch (`epi-head-wave3`), NOT yet wired downstream.**
> New best head **`a1res03`** = CNN + `mixed_margin` + **`lambda_iou_rank=1.0`** (IoU-graded window ranking
> as the MAIN span objective) + **`lambda_residue=0.3`** + log_mean_exp β=4. Config
> `epitope_head/configs/cnn_himp_a1_res03.yaml`. Run loop migrated to **Modal** (`scripts/modal_epitope.py`).
> Full writeup: `report/epi_wave4_session_report.md`. Frozen `cnn_himp_v1_npoff_*` still drive IF h-maps until this is wired in.
> - **0701 5-fold cluster CV (held-out test):** a1res03 IoU50 **0.581**, ExAP 0.262, **Pearson 0.508** (vs NMP 0.682 / 0.336 / **0.170** — landscape carries). Progression A0 0.493 → A1 (+λ_iou_rank) 0.578 → a1res03 (+λ_residue=0.3) 0.581; Δ A1−A0 = **+0.085 IoU50, +0.089 Pearson, zero regression**.
> - **The lever**: IoU-graded ranking promoted from Wave-3 aux (λ=0.3, +0.007 null) to main objective (λ=1.0) **after fixing the starved hard-negative candidate spectrum** (`train.yaml` 0.8/5 → 0.95/5/hard_frac0.5).
> - **0401 (now 5-fold cluster CV, mean±std):** landscape Pearson **0.429±0.036 ≫ NMP 0.146** (~2.9×) holds; region/exact/ResAP trail NMP under proper CV (IoU50 0.513 vs 0.556, ExAP 0.239 vs 0.280, ResAP 0.528 vs 0.562, ResAUC 0.851 vs 0.870). The earlier single split (IoU50 0.544, ResAP 0.562>NMP) was **optimistic split-luck**; CV is the defensible number. ⇒ 0401's claim is the **landscape dominance**, not a region tie.
> - **1501 (single split, live NetMHCIIpan):** a1res03 IoU50 0.567 vs NMP 0.623, **Res_Pearson 0.461 ≫ NMP 0.154**, Exact-AP **0.251 vs old `LC1_drb1501_aug` 0.172 (+46%)** — the poor-1501 problem is fixed. ⇒ **a1res03 unified across all 3 alleles**; landscape Pearson dominates NMP on every allele (0.508/0.450/0.461 vs 0.170/0.139/0.154).
> - **Two architecture levers, both NEGATIVE** (→ convergence): dual-head exact readout (`z_exact` exact-AP 0.249 ≈ z_region 0.255 — exact-15mer boundary is the IEDB peptide convention, not sequence-recoverable); core-aware 9-mer scorer (IoU50 +0.006, within noise). Cheap levers (λ_iou {0.5,2}, λ_residue {0.3,0.5}, β {2,8}, near-exact negs) all swept → converged.
> - Goal priority (frozen): **IoU≥0.5/0.7 region AP**, then **residue density Pearson/Spearman**; exact-AP loose guardrail; downstream consumes ONLY the per-residue landscape (NMP = reference).

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
- **Multi-allele a1res03 deliverables (Wave-4, unified config `epitope_head/configs/cnn_himp_a1_res03.yaml`)**: a1res03 is the unified best across all 3 alleles. **All three are EVAL-SPLIT-trained** (held out val+test for honest NMP comparison) — **none is trained on full data** (see Full-data status). Reported test metrics ↔ the exact ckpt below:
  - DRB1*07:01: 5-fold cluster CV — now on **Della** `run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold{0..4}/runs/LC1/seed_42/` (each fold: goal-epoch ckpt + best.pt + configs/logs; also on Modal `immune-design-runs`). Per-fold goal epochs (argmax val IoU50, reproduce the reported mean exactly): **f0=e29, f1=e24, f2=e39, f3=e29, f4=e34**; best.pt = val-pp_ap-selected (downstream-leaning). No single full-data model (CV by construction; each fold trains on 4/5). Test mean: IoU50 **0.581**, ExAP 0.262, Pearson **0.508**.
  - DRB1*04:01: **5-fold cluster CV** (mean±std, goal-selected epoch/fold: f0=e9,f1=e34,f2=e24,f3=e39,f4=e44) — 5 fold ckpts on **Della** `run/epitope_head/cnn_himp_a1_res03_drb0401_seed42_cv5_fold{0..4}/runs/LC1/seed_42/`. Test mean: IoU50 **0.513±0.012**, IoU70 0.487±0.014, ExAP 0.239±0.014, **Pearson 0.429±0.036**, ResAUC 0.851±0.009, ResAP 0.528±0.015 (NMP 0.556/0.533/0.280/0.146/0.870/0.562). Aggregate: `run/benchmark/w4_drb0401_cv/cv_summary_w4.json`; NMP reused from full-pool cache `run/benchmark/epi_himp_sweep/cache_0401_all1865.parquet`. *(The earlier single-split epoch_34 ckpt at `cnn_himp_a1_res03_drb0401_seed42/` is retained but the CV supersedes it as the reported basis.)*
  - DRB1*15:01: single split, **best.pt** — deliverable on **Della** `run/epitope_head/cnn_himp_a1_res03_drb1501_seed42/runs/LC1/seed_42/best.pt`. Trained on **1,236/1,536** (val 153 + test 147 held out). Test: IoU50 **0.567**, ExAP 0.251, Pearson **0.461** (vs old `LC1_drb1501_aug` exact-AP 0.172 → **+46%**).
  - manifests: `work/immune-design/manifests/{drb0701,drb0401,drb1501}/splits/strict/`.
- **0401 full 5-fold CV — DONE** (2026-07-07). cv5 splits generated; 5 folds trained on Modal then pulled to Della; full-pool NMP cache `cache_0401_all1865.parquet` built on Della (16h41m); **eval + aggregate run LOCALLY on Della** (ailab GPU, reuse cache, no NetMHCIIpan). Result: landscape Pearson **0.429±0.036 ≫ NMP 0.146**; region/exact/ResAP trail NMP (see blockquote). Single-split 0401 was optimistic → CV is the reported basis now. report + this file backfilled. (Fixed a `cv_select_and_aggregate.py` single-arm IndexError along the way.)
- **Full-data (production) checkpoints**: **NONE trained yet** for a1res03 on any allele. Every a1res03 ckpt holds out ~20% (val+test). A full-data refit (train on train+val+test at the goal epoch) would be the downstream-deployment checkpoint — not built (a1res03 downstream deployment not yet requested; for the paper, eval-split is the correct basis).
- **Prior production heads (npoff/aug — superseded by a1res03 on span+landscape goal, retained; these are what RF downstream currently points to)**: 0701 `cnn_himp_v1_npoff_drb0701_seed42` (pp_ap 0.3497); 0401 `cnn_himp_v1_npoff_drb0401_seed42` (pp_ap 0.3165); 1501 `LC1_drb1501_aug` (pp_ap 0.1721).
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

- **`if_ready/` is organized by target** (each in its own folder): `main/` (canonical test set + sidecars + `load_coords_gate_<tag>.json`), `pilot/`, `fast/`, `highrisk/`, `h_maps_v2/` (shared head-risk maps), `_ARCHIVED/` (`fast_v1`, `h_maps_v1`, `highrisk_demo_v1_pmpnn_only`).
- **Main test set = v2 (Tier 1 + Tier 2 only; Tier 3 removed).** Tier 2 selection: NMP-only, structure-blind, unimodal-Gaussian over `coverage_fraction` (μ=median, peak:tail=3, min-per-bin=40). **IF-ready truncation filter applied: all rows have `if_sequence_coverage ≥ 0.8`** (heavily-truncated structure fragments removed — they deviate from whole-protein redesign and collapse structurally; 0701 −135, 0401 −186). `build_if_ready_test_set.py --min-coverage` now defaults to 0.8. Canonical files (npoff h_maps):
  | allele | `test_proteins_<tag>.parquet` | `if_ready/main/test_proteins_if_ready_<tag>.parquet` | `if_ready/h_maps_v2/h_maps_<tag>.parquet` |
  |--------|-------------------------------|--------------------------------------------------|--------------------------------------------|
  | DRB1\*07:01 | 2879 (T1 15 / T2 2864) | 2879 | 2879 (covers 100%) |
  | DRB1\*04:01 | 2829 (T1 15 / T2 2814) | 2829 | 2829 (covers 100%) |
  - Phase C: `TEST_SET_PARQUET=if_ready/main/test_proteins_if_ready_<tag>.parquet`, `H_MAPS_PARQUET=if_ready/h_maps_v2/h_maps_<tag>.parquet` (optional; omit for constant_one / controller-only arms), `PDB_ROOT=pdbs_if_ready/<tag>`. Unified allele tag = full `HLA-DRB1_07_01` form on disk everywhere. SLURM `submit_if_phase_c.slurm` H_MAPS default = `h_maps_v2`; set `H_MAPS_PARQUET=` (empty) to run without an h-map.
  - CLIs: `select_tier2_v2.py` (merge shards + stratified sample); `prescreen_tier2.py --selection-mode nmp_only` (`--nmp-screen-lengths/--n-shards/--sample`); lib `inverse_folding/evaluation/{sampling.py,immunogenicity.compute_coverage_fraction}`.
- **Diagnostic / demonstration subsets (allele-specific, subsets of canonical if_ready):**
  - **pilot** — two parallel variants (different stratification axes; both n≈50, high-freq RF iteration):
    - **pilot_v3 (CURRENT, 2026-07-17)** `if_ready/pilot/pilot_v3_<tag>.parquet` (+`.fasta`) — **0701 50 / 0401 50**, stratified UNIFORMLY over de-immunization difficulty on the unguided **NoD** baseline. Difficulty = median NMP `strong_frac` over **all 8** NoD designs (cleanest estimate); axis = **NMP-only** (head is RF guidance → diagnostic only). 10 equal-frequency quantile bins × 5, difficulty per-bin monotone, selected span [0.000,~0.07]. `build_pilot_difficulty.py`. Regression note: the highrisk NoD-selection bias is a *second-order* effect for a uniform set + qualitative iteration (severe only for extreme-tail + quantitative claims), so all-8 binning is fine; for a rigorous quantitative per-bin method-vs-NoD number, reserve a disjoint `--half-b-designs` or regenerate a fresh NoD baseline for the 50 proteins at eval time.
    - **pilot_v2** `if_ready/pilot/pilot_v2_<tag>.parquet` — **0701 47 / 0401 49**, `coverage_fraction` mid-load band [p40,p75] + 5 Tier-1 anchors. Generator-independent but difficulty-blind. Retained: it is the cohort of the SC-GR planner (`PLAN_PLANNER_SC_GR.md`). `build_pilot_v2.py`.
  - **fast** `if_ready/fast/fast_v2_<tag>.parquet` — **0701 286 / 0401 285** (was 300; truncated removed). 15 Tier 1 + Tier 2 uniform across coverage. Global sanity. Carries a `coverage_fraction` column.
  - **highrisk (0701 + 0401 done; 1501 pending)** `if_ready/highrisk/` — top-100 by `min(nmp_pct, head_pct)` (NMP strong-window burden AND epitope-head global_risk both high; `if_sequence_coverage ≥ 0.8`; `build_highrisk_demo.py`).
    - **`highrisk_nod_v1_HLA-DRB1_07_01`** (CURRENT, 2026-07-09) — **primary = NoD full-pool baseline** (RF's own DFM kernel with guidance OFF, `c1_null`; **2879 proteins × 8 diverse designs**, temp 1.0, seed 42; a1res03 fold0 head). Kernel-consistent baseline for the RF-validation/iteration highrisk set (NoD = same DFM sampler as every guided C1/C2/C3/D arm). sel_nmp 0.030–0.082 / sel_head 1.66–4.60; `pmpnn_nmp/head` diagnostic + sorted FASTA. NoD gen `run/inverse_folding/baselines/nod_full_v1/`, imm `run/benchmark/if_phase_c/nod_full_v1/`.
    - **`highrisk_nod_v1_HLA-DRB1_04_01`** (CURRENT, 2026-07-10) — same NoD full-pool recipe on 0401 (**2829 proteins × 8 diverse designs**, a1res03 0401 fold0 head; gen 16 ailab shards, imm 192 cpu/qos=short shards ~110 min). 100 proteins, sel_nmp 0.046–0.114 / sel_head 0.42–4.6. **0401 test set now complete** (main + fast + pilot + highrisk).
    - **`highrisk_dplm_iter_v1_*` — STALE** (see sibling `.STALE.md`): selected on the DPLM-native **argmax** C0 baseline (all 8 designs identical → degenerate median-over-8; also a different sampler than the DFM RF arms). Superseded by `nod_v1`; **overlap 29/100** (71 picks changed). Do not use.
    - `highrisk_pmpnn_demo_v1_<tag>` (primary=ProteinMPNN → demonstration / beat the reported baseline) — **unaffected** (ProteinMPNN sampling is diverse); still valid. Each row carries cross-baseline `nmp`/`head` diagnostics.
    - **Status (2026-07-10)**: 0701 + 0401 NoD baselines + highrisk done; NoD 0701 full struct eval done (scTM median 0.938, ≥0.5 90%). **B1-vs-NoD regression-aware analysis done**: after correcting for highrisk being a NoD-extreme set, B1 shows a *modest genuine* head de-immunization (raw head Δ −0.66/−1.37 is largely regression-inflated; guidance residual ≈ −0.07…−0.44) and **essentially no NetMHCIIpan benefit over regression**, structure preserved for ~80%. A clean de-immunization claim needs a non-NoD-selected eval set. 1501 baseline pending.
- **Uricase case study (standalone, not in main test set).** Files under `uricases/`:
  - `uricase_caseset_if_ready_unified.parquet` — **6388** (6387 exact-deduped + **Pegloticase** appended as the last row 2026-07-03; **all 26 `characterized` retained**; near-redundancy intentionally NOT removed). Source: 4958 AFDB + 1429 ESMFold2 + **1 AF3** (Pegloticase). Dedup recorded in `uricase_caseset_if_ready_unified.manifest.json`.
  - **Pegloticase** (recombinant pig–baboon chimeric therapeutic uricase / Krystexxa; 298 aa, P16164 variant 98.7% id): structure from **AF3 full-MSA** prediction (local-DB, pLDDT 0.947 / pTM 0.94; `pdbs_if_ready/Pegloticase.pdb`, `structure_source=af3`, coverage 1.0). Projects cleanly (8/8 Q00511 anchors, triad-complete, `viability=projected`, `design_viable=True`); `validate_against_sequence` passes vs the committed manifest.
  - `pdbs_if_ready/` — structure files retained on disk (6740: 5222 `.cif` + 1518 `.pdb`; 353 orphaned exact-dups kept) + `Pegloticase.pdb` (AF3); `h_maps/h_maps_uricase_HLA-DRB1_{07_01,04_01}.parquet` — **6388** each (Pegloticase concatenated; 0 missing / 0 md5-mismatch vs unified), covers 100%, npoff; `wt_generated_uricase_caseset_HLA-DRB1_07_01.parquet` facade — 6387 (pre-Pegloticase).
  - Enzyme-mode RF manifest `inverse_folding/reference_flow/configs/uricase_q00511_active_site_perprotein_v0.yaml` covers **24/26 characterized** proteins with validated hard anchors (incl. Pegloticase); `C5HDG5` and `W8X3B8` are intentionally excluded by `uricase_characterized_special_overrides.yaml`. Q00511 B1 highrisk-open inpainting smoke passed on A100 (`run/inverse_folding/phase_d/uricase_smoke/HLA-DRB1_07_01/uricase_q00511_b1_aopen_inpaint_smoke_seed42_gpu80_r4_20260630T013544Z`): 8/8 anchors preserved, 158 non-anchor edits.
  - **`Uricases_RF/`** (design_viable subset) — **5492** design_viable = 5489 Q00511-triad-projected + 2 own-annotated PucL fusions (O32141/Q45697) + Pegloticase; 24/26 characterized in-set (894 gated + C5HDG5/W8X3B8 excluded). Deterministically derived from the 6388 unified set (triad gate {10,57,256}). Holds `uricase_rf_designviable_if_ready.parquet` (IF-ready + viability labels), `.fasta`, `uricase_caseset_if_ready_dedup_labeled.parquet` (6388 labeled), `manifest.json`. RF-ready: structure / h_maps(0701+0401) / constraint-manifest all cover 5492/5492 (0 missing, 0 md5 mismatch). Original `uricases/` unchanged.
  - **WT immune baseline — a1res03 heads, 3 alleles** — the uricase design baseline (WT uricase sequences scored; what the uricase-RF designs are compared against). **Cluster** `run/benchmark/wt_v2/HLA-DRB1_{07_01,04_01,15_01}/wt_uricase_a1res03_HLA-DRB1_<tag>_imm_full/` (`imm_head`/`imm_nmp` + `_residues`/`_peptides` long tables; 0401/1501 also keep 48 per-allele shards). **Local (Mac)** `mhc-if-local:/Users/jerry/Project/MHC-IF/Results/IFStandalone/uricase_a1res03_immune/HLA-DRB1_{07_01,04_01,15_01}/` (imm-only; the uricase WT structures are the input PDBs, no separate struct baseline). **6388 each** — Pegloticase concatenated 2026-07-03: head global_risk/NMP strong_frac by allele 0701 −8.95/0.012 (pct 5/24), 0401 −9.25/0.020 (18/53), 1501 −8.72/0.008 (52/67); low-immune, strongly allele-dependent). Heads: 0701 = a1res03 cv5 fold0 best.pt (NMP reused from prior 6387 DRB1\*07:01 run — head-independent); 0401 = a1res03 epoch_34; 1501 = a1res03 best.pt. NMP strong_frac median **0701 0.021 / 0401 0.019 / 1501 0.005** (uricases far less 1501-immunogenic); a1res03-head↔NMP Spearman **0.70 / 0.38 / 0.46**; 1501 head healthy (left-skewed, std 2.58). Immune landscape strongly allele-dependent (e.g. Q9RV70 0701 pct 2 ↔ 1501 pct 93).
  - **Tetramer interface gate (2026-07-21, code + 1R51 reference data ready)** — torch-free reusable metrics in `inverse_folding/evaluation/tetramer_interfaces.py`; `eval_tetramer_gate.py`/`submit_tetramer_eval.slurm` score prediction cohorts, while new `eval_tetramer_reference.py` scores a static crystal/prediction and optionally runs per-residue Rosetta `DdGScan`. Complete-polymer standardization now removes 1R51's incomplete/free amino-acid crystal records and maps N-terminal `SAC→SER` before coordinate/Rosetta scoring. Corrected 1R51 coordinate BSA: interface_1 AB/CD 5.780k Å² total (2.890k/partner), interface_2 AD/BC 5.214–5.219k (2.607–2.609k/partner), diagonal 0.738–0.740k. InterfaceAnalyzer: AB/CD dSASA 5803.7/5878.8 Å², dG −155.37 REU; AD/BC dSASA 5284.6/5330.1 Å², dG −58.21 REU; full packing/H-bond/BUNS/SC fields retained. A-B/A-D fixed-backbone alanine scan is complete: 266 chain-side rows / 133 index rows (70 AB + 63 AD), exact chain symmetry and Ala→Ala zero controls; strongest interpretable side-chain truncations are Y46A on AB (+5.686 REU) and W106A on AD (+5.780 REU). Gly/Ala/Pro values remain available but are excluded from side-chain ranking. Persistent data = `work/immune-design/tetramer_gate/reference_metrics/1R51/`; Rosetta runtime = `run/benchmark/tetramer_gate/reference_metrics/1R51/rosetta/`; stdout/stderr = `logs/benchmark/tetramer_gate/reference_metrics/1R51/rosetta/`. The five Parquet tables are exactly unchanged after path separation. Threshold calibration remains pending assembly/activity anchors; protocol = `PROTOCOL/tetramer_prediction_gate.md`.
  - **Cluster path-layer debt (2026-07-21 audit)** — canonical policy is now enforced in `AGENTS.md`: experiments/runtime/predictions/MSAs/caches → `run/`, datasets/static/persistent products → `work/`, stdout/stderr → `logs/`. Future defaults were corrected in the tetramer, Phase C, and IF-IMP launchers. Legacy runtime trees still physically under `work/` include `af3_colab_test/`, `esmfold2_refold/`, `if_imp/`, `if_phase_c/`, `protenix_barrel_refold/`, `protenix_prasnase_docking/`, and tetramer `pred*/msa*` trees; they are retained in place because active jobs/manifests may reference absolute paths and must be migrated case-by-case, never by blind bulk move.
  - `missing_structure_list.{csv,fasta}` — uricases still without structure.
  - Phase C: `TEST_SET_PARQUET=uricases/uricase_caseset_if_ready_unified.parquet`, `H_MAPS_PARQUET=uricases/h_maps/h_maps_uricase_<tag>.parquet`, `PDB_ROOT=uricases/pdbs_if_ready`.
  - Caveat: ESMFold-predicted backbones as scTM refold targets = self-consistency, not true GT.

- **Code**: L0-L5 implemented + tested (139 tests passing), 6 code review issues fixed
  - schema (`validate_test_protein_entry`), overlap (MMseqs2), prescreen (dual-scorer + CATH diversity), assembly, tier validators
  - CLIs: `run_overlap_filter.py`, `validate_tier1.py`, `validate_tier3.py`, `prescreen_tier2.py`, `assemble_if_test_set.py`, `download_tier2_candidates.py`
  - SLURM: `submit_prescreen_tier2.slurm`, `submit_assemble_test_set.slurm`, `submit_download_tier2.slurm`
  - Key fixes: validate_tier1 now runs MMseqs2 overlap; validate_tier3 enforces NMP signal; prescreen uses correct epitope_head API; assembly generates WT artifacts; overlap CLI includes no-hit candidates; sample_diverse guarantees cross-architecture-class coverage
- **Cluster**: test sets **assembled** for two alleles
- **Key Data**:
  - **HLA-DRB1\*07:01 test set** — `test_proteins_HLA-DRB1_07_01.parquet` — **3,166 proteins**
    | Tier | Count | mean_len | mean_risk | mean_nmp_strong |
    |------|-------|----------|-----------|-----------------|
    | 1 | 15 | 267.5 | -0.433 | 64.5 |
    | 2 | 3,000 | 250.3 | 3.649 | 85.1 |
    | 3 | 151 | 289.1 | 0.006 | 68.4 |
  - **HLA-DRB1\*04:01 test set** — `test_proteins_HLA-DRB1_04_01.parquet` — **3,165 proteins**
    | Tier | Count | mean_len | mean_risk | mean_nmp_strong |
    |------|-------|----------|-----------|-----------------|
    | 1 | 15 | 267.5 | -4.207 | 49.9 |
    | 2 | 2,998 | 207.9 | 3.128 | 113.4 |
    | 3 | 152 | 287.5 | -2.859 | 80.7 |
  - Characterized uricase curated additions (2026-06-03): 26 input FASTA records → 25 unique sequences (`Q2U050` exact duplicate of `Q00511`, omitted); appended as Tier 3 to both allele test sets after allele-specific head/NMP artifact scoring. NMP status complete for all 25 per allele; NMP strong windows: 0701 median=88 (min=10, max=190), 0401 median=82 (min=19, max=189).
  - Prescreen pipeline (Tier 2): 225,821 → 75,425 (MMseqs2 CATH overlap) → 3,000 (head prefilter) → dual predictor filter (head + NMP) → assembly
  - NMP scoring: 3,000/3,000 complete (both alleles), no timeout
  - CATH topology filter: **frozen** (舍弃) — 最后一步只做 dual predictor (head + NMP) 筛选，不再做 CATH topology diversity sampling
- **Artifacts**:
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet` — 3,166 rows, 17 columns
  - `work/immune-design/if_test_set/test_proteins_summary_HLA-DRB1_07_01.json`
  - `work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet` — 3,165 rows, 17 columns
  - `work/immune-design/if_test_set/test_proteins_summary_HLA-DRB1_04_01.json`
  - `work/immune-design/if_test_set/fastas/` — 6,191 FASTA files
  - `work/immune-design/if_test_set/pdbs/0401/` — **2,943 .pdb** (RCSB experimental) + **215 .cif** (local AF snapshot rescue + RCSB .cif Track A + AFDB Track B + characterized uricase AFDB) = 3,158 structures; **20 unresolved** uricase UniProts listed in `uniprot_final_failures.txt` (no AFDB prediction; includes `A0ABR4SZB5`, `A0ABX7U467`)
  - `work/immune-design/if_test_set/pdbs/0701/` — **2,874 .pdb** (RCSB experimental) + **267 .cif** (18 local AF + 123 RCSB cif + 102 AFDB + 1 D3BGR1 fix + characterized uricase AFDB) = 3,141 structures; **25 unresolved** uricase UniProts listed in `uniprot_final_failures.txt` (no AFDB prediction; includes `A0ABR4SZB5`, `A0ABX7U467`)
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_04_01.parquet` — **3,095 IF-ready proteins** (Tier 1:15, Tier 2:2,948, Tier 3:132; 23 characterized uricase additions ready); cleaned structures in `work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_04_01/`; DPLM `load_coords()` gate: **3,095/3,095 exact sequence+length match**
  - `work/immune-design/if_test_set/if_ready/test_proteins_if_ready_HLA-DRB1_07_01.parquet` — **3,102 IF-ready proteins** (Tier 1:15, Tier 2:2,962, Tier 3:125; 23 characterized uricase additions ready); cleaned structures in `work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_07_01/`; DPLM `load_coords()` gate: **3,102/3,102 exact sequence+length match**
  - IF-ready failure manifests: `if_ready/test_proteins_if_ready_HLA-DRB1_04_01.failures.csv` (70 failures: 50 unknown residues, 20 missing structures), `if_ready/test_proteins_if_ready_HLA-DRB1_07_01.failures.csv` (64 failures: 38 unknown residues, 26 missing structures)
  - Final-failure manifests (absolute paths):
    - `work/immune-design/if_test_set/pdbs/0401/uniprot_final_failures.txt` (20 UniProts, mostly Streptomyces / 小众真菌 / Uncharacterized TrEMBL entries; now includes two characterized additions with AFDB CIF/PDB 404)
    - `work/immune-design/if_test_set/pdbs/0701/uniprot_final_failures.txt` (25 UniProts, same pattern; now includes two characterized additions with AFDB CIF/PDB 404)
    - `pdb_final_failures.txt` — 0 entries on both (all RCSB 404s recovered via .cif tracks)
    - `download_failures_remaining.txt` — 45 (0401) / 98 (0701) historical chain-level RCSB 404 lines, superseded by rescue via .cif; kept for audit
  - intermediate: `_prescreen_{head,nmp}_results*.parquet`, `tier2_prescreened*.parquet`
- **Feeds**: all downstream evaluation (M, N, and all F3/F4 subfigures)
- **Open**:
  - 45 allele-specific UniProt structure entries (20 + 25; includes two characterized additions missing for both alleles) have no AFDB prediction — planned to run **AF3** locally to fill these gaps before Phase C
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
  - Current PMPNN v2 full-set run: output root `run/inverse_folding/baselines/proteinmpnn/v2_8design_20260604T024603Z/`; input staging preflight passed for 0701 3139/3139 and 0401 3147/3147 with continuous CA residue numbering. Generation/evaluation metrics are not yet recorded.
- **N3 (Level 3 DRAKES)**: **dropped** — not needed for paper; can be added as reviewer response if requested
- **Blocked by**: Phase C (need unguided baseline candidate pool from C0)
- **Comparison structure for paper**:
  1. Level 1 post-hoc filter (N1) — from C0 candidates
  2. Sampling-only position-dependent schedule (C1 / Tier 0 ablation)
  3. Full position-dependent reference flow (C2-C3 / our method)

---

## IF Encoder Replacement (PLAN_IF_ENCODER.md)

- **Status**: GeoEGNN-IPA implementation code ready; E7 quick encoder diagnostics rerun with `use_draft_seq=true` on 300 proteins using `encoder_last.pt` checkpoints (2026-05-22).
- **Latest code state**:
  - `scripts/train_if_imp_encoder.py` now exposes adapter shape controls: `--adapter-num-layers`, `--adapter-gated`, `--adapter-gate-init`.
  - Training reinstall logic runs before DPLM freezing: old Module-K last-1 ungated adapters can be rebuilt as last-N gated adapters, then only `encoder.*` and adapter-named decoder parameters remain trainable.
  - `--adapter-num-layers > 1` requires `--adapter-gated` fail-fast to avoid fresh ungated mid-layer adapters perturbing decoder hidden states at initialization.
  - `scripts/submit_if_imp.slurm` forwards `ENCODER_ADAPTER_NUM_LAYERS`, `ENCODER_ADAPTER_GATED`, and `ENCODER_ADAPTER_GATE_INIT` under `MODE=train_encoder`.
- **Validation**: local focused checks passed (`tests/scripts/test_if_imp_encoder_scripts.py` + `tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py` → 16 passed, 8 skipped; skipped tests are PyG / omegaconf / transformers gated). `python scripts/train_if_imp_encoder.py --help`, `python -m py_compile scripts/train_if_imp_encoder.py`, and `bash -n scripts/submit_if_imp.slurm` pass.
- **Latest cluster diagnostics** (legacy pre-separation location `work/immune-design/if_imp/diag_encoder/e7_quick_eval_draftseq_last/`, retained in place; future IF-IMP runs default to `run/inverse_folding/if_imp/`; 300 proteins, `use_draft_seq=true`, `encoder_last.pt`):
  | Arm | final recovery mean | final recovery median | draft-init recovery mean | draft-init recovery median |
  |-----|---------------------|-----------------------|--------------------------|----------------------------|
  | `aux05_a1` | **0.1736** | **0.1697** | 0.2549 | 0.2657 |
  | `pre2_aux01_a1` | 0.1389 | 0.1289 | 0.2546 | 0.2596 |
  | `aux01_a1` | 0.0975 | 0.0938 | 0.2102 | 0.2129 |
  | `aux01_a4g` | 0.0911 | 0.0833 | 0.2130 | 0.2200 |
- **Current interpretation**: draft initialization is active and non-trivial (`draft_init_recovery` ≈0.21-0.25), but iterative decoding does not preserve draft recovery by default. `aux05_a1` remains the best final-recovery arm; `pre2_aux01_a1` is second-best under last checkpoint and no longer shows the best-checkpoint collapse.
- **Operational note**: Della `immune-design` smoke should include one `ENCODER_ADAPTER_NUM_LAYERS=4 ENCODER_ADAPTER_GATED=1 ENCODER_ADAPTER_GATE_INIT=0.0 LIMIT_BATCHES=2 MODE=train_encoder` job before full ablations.

---

## Phase B: Experiment Preparation (NEW, 2026-04-07)

- **B1 (PDB download)**: **near-complete** — 0401: 2,943 .pdb + 215 .cif (AF rescue + RCSB cif + AFDB + characterized uricase AFDB); 0701: 2,874 .pdb + 267 .cif. All RCSB 404s rescued via .cif tracks (`pdb_final_failures.txt` = 0 on both). Residual **45 allele-specific UniProt structure entries** (20 on 0401 / 25 on 0701; includes two characterized additions with AFDB CIF/PDB 404 on both alleles) have no AFDB prediction — listed in `pdbs/{0401,0701}/uniprot_final_failures.txt`; will be filled by **AF3 local prediction**
- **B1.5 (IF-ready structure/sequence cleaning)**: **complete for PDB/mmCIF-backed rows** — `scripts/build_if_ready_test_set.py` materialized resolved-backbone parquets and cleaned single-chain PDB/mmCIF structures. 0401: 3,095 ready / 70 failed; 0701: 3,102 ready / 64 failed. DPLM `load_coords()` exact sequence+length gate passes for all ready rows. Tier 3 uricases are now partially recovered via default chain `A` (0401:132, 0701:125), including 23 characterized additions per allele. Remaining failures are missing structures or unknown residue symbols not covered by BioPython's extended PDB residue map.
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
  - **Standard RF refinement structure contract (2026-07-13)**: `scripts/refine_rf_designs.py` / `submit_refine.slurm` now default to the persistent isolated `esmfold2_live` worker (final metrics read the same normalized ESMFold2 cache). The standard gate is the protocol trio `scTM`, direct-functional `cat_max_scRMSD`, and `predicted_active_site_min_pLDDT`; all three thresholds are required per-run inputs with no repository defaults. The previous seed-relative/whole-anchor gate is available only via explicit `structure_gate_profile=legacy`.
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

**NoD baseline** (RF-no-guidance DFM sampler `c1_null`, temp 1.0 / seed 42 / n8) — **the current unguided baseline; SUPERSEDES the deleted DPLM-native C0** (argmax-degenerate `dplm_native_full_v2` removed from cluster + Mac, L0130/L0131). Status: 0701 = gen + imm + **struct** all done; 0401 = gen + imm done (struct not run); 1501 pending.

**Cluster locations** (base `/scratch/gpfs/KAIYIJIANG/zijie/`):

| allele | stage | path |
|---|---|---|
| 0701 | generation (23032 = 2879×8) | `run/inverse_folding/baselines/nod_full_v1/HLA-DRB1_07_01/nod_full_v1_n8_t1p0_seed42_merged/generated.parquet` |
| 0701 | eval_immune (a1res03) | `run/benchmark/if_phase_c/nod_full_v1/HLA-DRB1_07_01/nod_full_v1_HLA-DRB1_07_01_imm/{imm_head,imm_nmp}.parquet` |
| 0701 | eval_structure (esmfold) | `run/benchmark/if_phase_c/nod_full_v1/HLA-DRB1_07_01/nod_full_v1_HLA-DRB1_07_01_struct/{structural,structural_residues}.parquet` |
| 0401 | generation (22632 = 2829×8) | `run/inverse_folding/baselines/nod_full_v1/HLA-DRB1_04_01/HLA-DRB1_04_01/nod_full_v1_n8_t1p0_seed42_merged/generated.parquet` |
| 0401 | eval_immune (a1res03) | `run/benchmark/if_phase_c/nod_full_v1/HLA-DRB1_04_01/nod_full_v1_HLA-DRB1_04_01_imm/{imm_head,imm_nmp}.parquet` |

**Local (Mac)** `mhc-if-local:/Users/jerry/Project/MHC-IF/Results/` — 0701 returned to `RF/HLA-DRB1_07_01/nod_full_v1_n8_t1p0_seed42__20260709T215523Z/{generation,eval_immune,eval_structure,meta}/` (this dir REPLACES the deleted `dplm_native_full_v2_{n8_seed42,imm_full}__*`).

| allele | n×d | scTM mean/median (>=0.5 / >=0.7) | head global_risk median | NMP strong_frac median |
|---|---|---|---|---|
| HLA-DRB1*07:01 | 2879×8 | **0.854 / 0.938** (90.0% / 84.8%) | 2.56 | 0.036 |
| HLA-DRB1*04:01 | 2829×8 | struct not run | 0.99 | 0.059 |

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

## RF-Refine Fusion V1-A — pre-terminal entry (PLAN_RF_REFINE_FUSION_V1.md)

> **LOCAL IMPLEMENTATION COMPLETE; CANARIES A/B + TERMINAL-ARM SMOKE COMPLETE.** Branch
> `fusion_rf_refine`, worktree `/Users/jerry/Project/MHC-IF-fusion`. The deployment canaries ran
> on Della on 2026-07-29; no scientific T0/P1 has run. Do not read a treatment claim out of canary
> data.

**The question.** Same total DFE, different ALLOCATION POINT: does spending the reward signal at a
partial state (ρ_edit<1) beat spending it at the endpoint? "Matched compute" is the entire claim.

| Piece | State |
|---|---|
| P1 `preterminal` arm (continuation-value allocation) | implemented, canary-configured |
| P1 `terminal` arm (complete trajectories, exact terminal Head rank) | implemented, canary-configured |
| T0 calibration (3 policy views over ONE root pool + ONE held-out K_EVAL) | implemented, canary-configured |
| Null-runtime firewall (§2.12), verified at the SAMPLER with a positive control | done |
| §3.4 fail-closed launch gate, §4.2 content-bound resume | done |
| Cluster canaries (V1F7) | A/B + terminal-arm smoke complete; C blocked |

Canary A entry sizing (2026-07-29; deployment evidence only):

| protein | `B` | unique roots | facade offered to v0 | `C_reserved` |
|---|---:|---:|---:|---:|
| `5ZHV_B` | 16 | 14 | 6/6 | 1657 DFE |
| `9L2Q_A` | 16 | 16 | 6/6 | 1663 DFE |

Across both proteins, logical DFE was `prefix=3168`, `est=128`, `final=14`. Thus the root-yield
and facade-cap wiring passed (`30/32` unique roots; both full facades), but the maturity evidence
is scientifically cautionary: `3168/32=99` prefix DFE per attempt and
`128/(30*4)=1.07` estimator-tail DFE per root-continuation. Under this smoke,
`rho_target=.85` therefore captured states with only about one denoising step remaining on
average. This is not a failure of Canary A, but `.85` is not frozen as a meaningful
pre-terminal scientific handoff until `snapshot_step`, `rho_actual`, absolute unresolved mass and
descendant diversity are inspected.

Cluster closure (2026-07-29; deployment evidence only):

- Canary A and the terminal-arm smoke each produced 6-row facades per protein; their v0 jobs
  closed `entry_source_id -> admission verdict -> initial particle` without a sequence join.
- Canary B used the 24-anchor Q00511 safety manifest. All 24 WT identities were preserved across
  6 facade rows and 62 continuation rows; `anchor_preservation` remains null and is not a measured
  field.
- At `F_cap=6, N=4`, `5ZHV_B` produced an elite in both arms, `9L2Q_A` had `0<4` feasible initial
  parents in both arms, and Q00511 reached `2<4`. Thus Q00511 did not exercise the anchored v0
  repair loop, and the scientific P1 run requires a larger frozen `F_cap`.
- The tested environment requires H100/H200/A100-class compatible CUDA hardware. The Della
  RTX PRO 6000 Blackwell nodes are incompatible with the current torch 2.5.1+cu121 build.

**Blocking a scientific run** (not a canary): `TEST_SET_PARQUET`, `DEV_IDS`/`HOLDOUT_IDS`, the real
T0 `rho_grid`/`K_EVAL`/`Q_T0`, and measured `SECONDS_PER_REFOLD`/`SECONDS_PER_DFE` — the gate
refuses an unverifiable walltime by design, so the last one is a hard stop.

**Known gaps, stated so no canary over-claims them** (full list in the runbook status block):
1. no T0→v0 definitive-structure handoff — T0's `3·Q_T0` subset is charged and recorded but
   deferred, so runbook §6.1 GO/KILL condition 3 cannot be closed from a T0 run alone;
2. the T0 structure-eligibility law is frozen as policy-faithful B* but the code still uses
   all independent-full survivors; Canary C must exercise the corrected law;
3. `maturity_telemetry.anchor_preservation` is null and is NOT a measurement — root-level
   preservation is structurally enforced (so a value there would be 1.0 by construction) and the
   completed-sequence checksum is not computed.

**Local test state**: full suite 2388 passed / 47 skipped, with the same 14 pre-existing failures
as the pristine tree (unrelated modules, missing optional deps) — zero new failures.

---

## Backup Collection — wet-lab targets (NEW, 2026-07-13)

4 experimentalist-supplied proteins redesigned by **B1Aopen inpainting** (RF stage-B1,
amplification=`constant_one`, controller `d2_d3_full_stageB_aopen_beta5p0.yaml` hash 2c39f229;
original `run_if_phase_c1.py`, NOT refine/fusion) with active-site hard anchors frozen. Structures =
**AlphaFold3 v3.0.3 full-MSA monomer** from WT seq (crystals unusable — chromophore fused to CRO/CH6;
AF3 validated vs crystal EGFP 0.29Å / mCherry 0.44Å Cα). Manifest (committed):
`inverse_folding/reference_flow/configs/backup_active_site_v0.yaml` (EGFP 8 / mCherry 9 / PrASNase 7 /
NanoLuc 8 anchors). Collection (Della): `work/immune-design/if_test_set/backup/` (AF3 structures
`pdbs_if_ready/`, `backup_caseset_if_ready.parquet`).

- **4-protein B1Aopen 0401** (256 designs each, a1res03 0401 cv5-fold0 head): all 7168 anchors preserved.
  **PrASNase** = clean win (NMP 0.0075→0.0021 −73%, esmfold2 scTM 0.99, active-site 0.26Å).
  **FP EGFP/mCherry** — ESMFold2 scTM 0.4 was an ARTIFACT (ESMFold2 can't fold FP β-barrels; WT-EGFP
  self-consistency only 0.64). **AF3 refold (WT-MSA transplant): EGFP 0.973 / mCherry 0.962, 100%≥0.7** →
  FPs fold + active sites preserved (mean-anchor sc-RMSD 1.1–1.3Å). Immune: EGFP NMP flat, mCherry regressed.
- **PrASNase 3-allele** (256/allele, expanded 7 anchors incl catalytic T15/T94/D95): structure perfect
  all alleles (scTM 0.988, 7-anchor sc-RMSD 0.42–0.44Å). Immune allele-dependent — **0401 −73% (243/256
  viable), 1501 −18% (170), 0701 +132% regressed (40)**.
- **Method finding**: B1Aopen de-immunizes cleanly only when WT has real immune burden + is an enzyme;
  on already-low-immune WT (mCherry, NanoLuc, PrASNase-0701) NMP regresses (head always drops, NMP only
  improves with burden). Screen WT immunogenicity before designing.
- **Fixed (2026-07-13)**: af3 refold normalizer wrote the `.plddt` sidecar as the all-atom `atom_plddts`
  mean, ~2-3 pLDDT below `evaluate_phase_c`'s v2 `predicted_global_plddt` (CA/per-residue mean) → af3
  struct-eval failed closed (`_validate_v2_prediction_plddt`). `normalize_af3_to_cache` now writes the CA
  B-factor mean of the emitted cache PDB (`mean_ca_plddt_from_pdb`) so sidecar ≡ check; regression test
  `tests/inverse_folding/test_af3_runner.py`. (esmfold2/protenix unaffected.)
- **Return (Mac)**: `mhc-if-local:/Users/jerry/Project/MHC-IF/Results/RF/backup/backup_20260713/`
  (collection + per-run gen/imm/struct + WT baselines + `RESULTS_SUMMARY.md`).

### LuxSit-i (backup target, 2026-07-15) — reference-fold matrix + RF-ready

De-novo luciferase **LuxSit-i core117** (Baker Nature 2023 / Chem 2025; substrate DTZ). **Redesign target = i only**;
parent LuxSit kept as reference. No experimental structure exists (Rosetta theoretical only) → **AF3 apo = RF backbone**.
Manifest `inverse_folding/reference_flow/configs/luxsit_active_site_v0.yaml` (collaborator-supplied `luxsit_active_site_v1`,
**safety-max 22 hard anchors + 7 monitored**, validates through our loader). Collection (Della):
`work/immune-design/if_test_set/backup/luxsit/`.

- **Structure = 4-refold reference matrix ×2 predictors**: AF3 v3.0.3 + Protenix v2, each {parent,i}×{apo,holo}. Holo
  cofolds the **collaborator-verified DTZ anion** `O=c1c(Cc2ccccc2)nc2c(-c3ccccc3)[n-]c(-c3ccccc3)cn1-2` (C25H18N3O⁻).
  Code: `--ligand-smiles` added to both refold-cache build-json drivers (`precompute_{af3,protenix}_refold.py`, off by default).
- **AF3-vs-Protenix verdict (LuxSit-i)**: apo **TM 0.985 / 0.50 Å**; holo pocket catalytic distances within ~0.2 Å
  (Y14–H98 2.7, H98–DTZ O1 2.8, R65–DTZ N1 3.7); **DTZ pose COM 0.13 Å**, ligand intact (29 heavy atoms); both
  high-confidence (pTM 0.92/0.96, ipTM 0.93/0.965) clash-free. → **No significant difference; safe to switch downstream
  validation to Protenix** (holo fwd ≈6.5 s/seed). Parent apo looser (TM 0.94). Both sit ~1.45 Å from the Rosetta design
  model (= paper AF2-vs-design 1.35 Å). Shared model-vs-Rosetta note: D18–R65 salt bridge ~4.2 Å (both) vs Rosetta 2.65 Å.
  Full table: `if_test_set/backup/luxsit/AF3_vs_Protenix_comparison.md`.
- **RF-ready-i**: `luxsit_i_caseset_if_ready.parquet` + `pdbs_if_ready/LuxSit-i_core117.pdb` + the manifest;
  `if_sequence_offset==0`, resolved seq == WT-i, 22 anchors validate. (holo cache-normalize skipped by design — multi-chain
  guard; holo structures live as raw CIFs with DTZ.)

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
| **a1res03 DRB0701 (Wave-4 best, 5-fold CV)** | **Della** `run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold{0..4}/runs/LC1/seed_42/` (goal-epoch f0=e29/f1=e24/f2=e39/f3=e29/f4=e34 + best.pt; also Modal `immune-design-runs`) | test mean IoU50 0.581 / ExAP 0.262 / Pearson 0.508. **Eval-split (CV); no full-data model.** |
| **a1res03 DRB0401 (Wave-4 best, 5-fold CV)** | **Della** `run/epitope_head/cnn_himp_a1_res03_drb0401_seed42_cv5_fold{0..4}/runs/LC1/seed_42/` (also Modal; earlier single-split `_drb0401_seed42/epoch_34.pt` retained) | CV test mean IoU50 0.513±0.012 / ExAP 0.239±0.014 / **Pearson 0.429±0.036** (NMP 0.556/0.280/0.146). Aggregate `run/benchmark/w4_drb0401_cv/cv_summary_w4.json`. |
| **a1res03 DRB1501 (Wave-4 best)** | **Della** `run/epitope_head/cnn_himp_a1_res03_drb1501_seed42/runs/LC1/seed_42/best.pt` | test IoU50 0.567 / ExAP 0.251 / Pearson 0.461 (vs old aug exact-AP 0.172, +46%). **Eval-split: trained 1,236/1,536 (held out val+test).** |
| Epitope head checkpoint (npoff DRB0701, prior production / RF-wired) | `run/epitope_head/cnn_himp_v1_npoff_drb0701_seed42/runs/LC1/seed_42/best.pt` | pp_ap=0.3497, pp_auc=0.9532 (superseded by a1res03 on span+landscape goal) |
| Epitope head checkpoint (legacy DRB0701 aug) | `run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.3027 (superseded by npoff) |
| Epitope head checkpoint (strict) | `run/epitope_head/cnn_strict_full/runs/LC1/seed_42/best.pt` | pp_ap=0.2874 |
| DPLM adapter checkpoint | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt` | validated |
| DPLM validation results | `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/baseline_validation.json` | complete |
| CATH dataset (4.3) | `work/immune-design/cath_4.3/` | train/val/test |
| Augmentation registry | `work/immune-design/augmentation/mutation_registry_strict.parquet` | used by LC1_lite_aug |
| Manifests | `work/immune-design/manifests/` | epitope head training |
| Epitope head DRB0401 (npoff, prior production / RF-wired) | `run/epitope_head/cnn_himp_v1_npoff_drb0401_seed42/runs/LC1/seed_42/best.pt` | pp_ap=0.3165, pp_auc=0.9627 (superseded by a1res03) |
| Epitope head DRB0401 (legacy aug) | `run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.2962, pp_auc=0.9492 (superseded) |
| Epitope head DRB1501 (legacy aug, prior / RF-wired) | `run/epitope_head/LC1_drb1501_aug/runs/LC1/seed_42/best.pt` | pp_ap=0.1721, pp_auc=0.8492 (superseded by a1res03) |
| Tier 1 candidates | `work/immune-design/if_test_set/tier1_candidates.json` | 15 candidates, user review pending |
| Tier 2 merged FASTA | `work/immune-design/if_test_set/tier2_candidates_merged.fasta` | 226k chains |
| IF test set (0701) | `work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet` | 3,166 proteins (T1:15 T2:3000 T3:151) |
| IF test set (0401) | `work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet` | 3,165 proteins (T1:15 T2:2998 T3:152) |
| IF test set FASTAs | `work/immune-design/if_test_set/fastas/` | 6,191 files |
| IF test set structures | `work/immune-design/if_test_set/pdbs/{0401,0701}/` | 0401: 2,943 .pdb + 215 .cif; 0701: 2,874 .pdb + 267 .cif. RCSB 404s fully rescued. 20 / 25 UniProt structure entries still missing (no AFDB) — `uniprot_final_failures.txt`; AF3 planned |
| IF test set h-maps (B2) | `work/immune-design/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_{07_01,04_01}.parquet` + `.meta.json` | **done** (07:01 2879, 04:01 2829; npoff head). Old flat `if_test_set/h_maps/` archived → `if_ready/_ARCHIVED/h_maps_flat_preV2/` (2026-07-09). h-map values inert for NoD/controller arms (gate-only); consumed only by `c1_{linclamp,sigmoid,power}`. |
| CATH training h-maps (B3) | `work/immune-design/cath_4.3/h_maps/h_maps_cath_HLA-DRB1_{07_01,04_01}.parquet` + `.meta.json` | TBD — cluster run pending |
| Phase C0 outputs | `run/inverse_folding/if_phase_c/native/<allele_tag>/<run_id>/{generated.{parquet,fasta},run_config.yaml,manifest.json}` | TBD — cluster run pending |
| Phase C1 outputs | `run/inverse_folding/if_phase_c/reference_flow/<allele_tag>/<run_id>/{generated.{parquet,fasta},run_config.yaml,manifest.json,trajectories/*.parquet}` | TBD — cluster run pending |
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
