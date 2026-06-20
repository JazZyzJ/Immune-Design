# Scripts Overview

This file lists all scripts and SLURM submission files, organized by project module.

---

## Dataset Pipeline (V2)

Scripts that build the `data/mhc_if_v2.tsv` dataset from raw IEDB exports.

### End-to-End

1. `scripts/iedb_end_to_end.py` — Runs the full IEDB-only V2 pipeline from raw export to `data/mhc_if_v2.tsv`.

### IEDB Cleaning

1. `scripts/iedb/filter_mhc_ligand.py` — Filters raw IEDB export to Class II, MS, ligand presentation, human host. Output: `data/iedb_ligand_rows_filtered.tsv`.
2. `scripts/iedb/filter_mhc_allele.py` — Keeps only valid allele rows and normalizes allele names. Output: `data/iedb_ligand_with_allele.tsv`.
3. `scripts/iedb/build_iedb_clean.py` — Builds high-resolution IEDB clean schema. Output: `data/iedb_clean_v1.tsv`.
4. `scripts/iedb/extract_low_reso_alleles.py` — Extracts low-resolution allele rows. Output: `data/iedb_low_reso.tsv`.
5. `scripts/iedb/build_iedb_low_reso_clean.py` — Builds low-resolution clean schema with `resolution`. Output: `data/iedb_low_reso_clean.tsv`.

### V2 Assembly Utilities

1. `scripts/add_positions.py` — Adds `peptide_position_info` to IEDB clean files using raw export coordinates.
2. `scripts/collect_accessions.py` — Collects protein accessions for sequence download. Outputs: `data/uniprot_accessions.txt` and related lists.
3. `scripts/identify_sources.py` — Maps accessions to source databases from raw export. Output: `data/accessions_sources.csv`.
4. `scripts/fetch_sequences.py` — Downloads UniProt FASTA sequences. Output: `data/uniprot_sequences.fasta`.
5. `scripts/fetch_ncbi_sequences.py` — Downloads NCBI FASTA sequences. Output: `data/ncbi_sequences.fasta`.
6. `scripts/regenerate_combined_fasta.py` — Merges UniProt + NCBI into `data/all_sequences.fasta`.
7. `scripts/finalize_v2.py` — Verifies positions, fixes shifts, and adds flanks. Use `--iedb-only` for IEDB-only V2. Output: `data/mhc_if_v2.tsv`.
8. `scripts/update_resolution.py` — Recomputes the `resolution` column for `data/mhc_if_v2.tsv`.
9. `scripts/validate_consistency.py` — Validates peptide vs protein slices and removes mismatches in-place.

---

## Epitope Head

Scripts for training, inference, ablation, and visualization of the Epitope Head model.

### Training

1. `scripts/train_v1.py` — Train Epitope Head v1 scorer (wires Modules A-G into executable training pipeline).
2. `scripts/train_v2_ablation.py` — Train Epitope Head v2 with encoder ablation for a single (encoder_id, seed) combination.

### Inference & Visualization

1. `scripts/infer_v1.py` — Run Epitope Head inference with CNN encoder on single or batched sequences; export canonical JSON payloads.
2. `scripts/visualize_demo.py` — Visualize protein residue immunogenicity risk as a heatmap HTML.

### Ablation & Diagnostics

1. `scripts/flank_ablation_test.py` — Compare original vs flank-ablated pp_AUC in ablation experiment.
2. `scripts/eval_encoder_ablation.py` — Collect metrics from completed encoder ablation runs and produce summary artifacts (ranking, mutation sensitivity, convergence, latency).

### Benchmark

1. `scripts/benchmark_head_vs_nmp.py` — Window-level comparison of epitope head vs NetMHCIIpan on a FASTA (no ground truth). Aligns predictions by (position, peptide_length), uses NMP strong-binder as label, computes per-protein AUC/AP/Recall@K/Spearman. Accelerated NMP. Outputs JSON.
2. `scripts/benchmark_iedb_test.py` — IEDB test-set benchmark: evaluates **both** head and NMP against IEDB ground-truth EL-positive spans (from `protein_samples_*.parquet` + `test_ids.txt`). Default NMP mode is "original" (batch_size=1, max_lengths_per_call=14, n_workers=1) for a clean baseline wall-time. `--metric-suite {exact,residue,iou_ladder,emd,all}` (default `all`) selects the metric families from `doc/EL_new_evaluation.md`: exact-span primary/conditional (always on), M2a residue-level AUC/AP/Pearson/Spearman over the **shared multi-k span set k∈[min_k,max_k]** (head and NMP aggregated identically — no length restriction), M6 IoU ladder with COCO-style greedy 1-to-1 assignment over configurable `--iou-thresholds` (default `1.0 0.7 0.5 0.0`; length-agnostic since IoU compares spans of differing length directly), and M3 normalized 1D EMD using the same residue aggregation. Adds `--bootstrap-n` / `--bootstrap-seed` for 95% CI and a paired Wilcoxon signed-rank (head vs NMP) on every headline metric. **Windows cache** (`--windows-cache PATH`) writes the raw `(start, k) → score` dicts for both head and NMP into a parquet (with sidecar `.meta.json` capturing allele/k-range/ckpt) after inference; `--resume-from-cache PATH` reloads the parquet and skips steps 2 (head) and 3 (NMP) entirely, so subsequent metric tweaks reuse previous inference results. For head-iteration workflows where NMP is frozen but the head checkpoint changes often, `--reuse-nmp-from-cache PATH` loads only the NMP half of the cache and skips step 3 (NMP) while running step 2 (head) fresh from `--epitope-ckpt`; cache metadata (`allele`, `min_k`/`max_k`) is validated against the current run and a mismatch fails fast (`nmp_mode` mismatch is a soft warning since NMP outputs are deterministic across modes). `--reuse-nmp-from-cache` is mutually exclusive with `--resume-from-cache` but compatible with `--windows-cache` (the rewritten cache then captures the fresh head + reused NMP, with `nmp_source: reused_from:<prev>` in the metadata for provenance). Optional `--near-miss-analysis` adds exact-vs-overlap diagnostics (`exact_ap`, `overlap_ap`, `iou50_ap`), GT exact-rank vs best-overlap-rank summaries, top-FP near/far buckets, and grouped summaries for `nmp_better`, `head_better`, and `all`. All flag families are additive and leave default benchmark behavior unchanged when omitted. Outputs JSON with per-protein + macro metrics, `statistics` block, and detailed timing.

### SLURM

1. `scripts/submit_train_v1.slurm` — Epitope Head v1 training (24hr, 1 GPU, 64GB).
2. `scripts/submit_train_v1_balanced.slurm` — Balanced Epitope Head v1 training (24hr, 1 GPU, 64GB).
3. `scripts/submit_train_v2_cnn.slurm` — Epitope Head v2 CNN training (24hr, 1 GPU, 64GB).
4. `scripts/submit_cnn_enhance.slurm` — CNN Lite augmentation training (24hr, 1 GPU, 64GB).
5. `scripts/submit_encoder_ablation.slurm` — Encoder ablation evaluation (24hr, 1 GPU, 64GB).
6. `scripts/submit_flank_ablation.slurm` — Flank ablation experiment (12hr, 1 GPU, 64GB).
7. `scripts/submit_epitope_train.slurm` — Parameterized epitope-head training launcher (HIMP-aware, 24hr, 1 GPU, 64GB). Reads `ALLELE` (e.g. `HLA-DRB1*07:01`), `OVERRIDE_CFG` (path to a `cnn_himp_v1*.yaml` override file), `SEED`, `P_AUG` (default 0.20; set to 0.00 to disable augmentation), and optional `DATA_DIR` / `REGISTRY` / `OUTPUT_ROOT` / `PROFILE` / `PROJECT_ROOT`. Derives allele tag (`drb0701` / `drb0401` / `drb1501`) and run tag (`<cfg-basename>_<allele-tag>_seed<seed>`) automatically; constructs default DATA_DIR / REGISTRY / OUTPUT_ROOT under the cluster `work/`, `augmentation/`, and `run/` trees. Designed to replace the per-allele `submit_train_drb*.slurm` legacy scripts; a sweep loop over `ALLELE` × `OVERRIDE_CFG` × `SEED` is the canonical entry point for HIMP ablation experiments.
7. `scripts/submit_benchmark.slurm` — Head vs NMP benchmark (24hr, 1 GPU, 16 CPUs). `MODE=fasta` (default) runs `benchmark_head_vs_nmp.py`; `MODE=iedb` runs `benchmark_iedb_test.py` against IEDB ground truth. Shared overrides: `EPITOPE_CKPT`, `ALLELE`, `VARIANT_ID`, `DEVICE`. FASTA-mode: `INPUT_FASTA`, `NMP_WORKERS`, `NMP_BATCH_SIZE`. IEDB-mode: `PROFILE`, `ALLELE_SUBDIR`, `PROTEIN_SAMPLES_PARQUET`, `TEST_IDS`, `NMP_MODE`, `NMP_WORKERS`, `NMP_BATCH_SIZE`, `NMP_MAX_LENGTHS_PER_CALL`, `NMP_TIMEOUT`, `NEAR_MISS_ANALYSIS`, `WINDOWS_CACHE`, `RESUME_FROM_CACHE`. When `NEAR_MISS_ANALYSIS=1`, the launcher appends `--near-miss-analysis` and writes to the `_near_miss.json` output variant; default behavior is unchanged when the env var is omitted or `0`. `WINDOWS_CACHE=PATH` writes the head/NMP raw `(start, k) → score` parquet (with sidecar `.meta.json`) after inference for cheap metric iteration; `RESUME_FROM_CACHE=PATH` reloads that parquet and skips head + NMP entirely so subsequent runs are sub-second on a login node; `REUSE_NMP_FROM_CACHE=PATH` loads only the NMP half and reruns head fresh from `EPITOPE_CKPT` — the workflow for head iteration on a frozen NMP. `RESUME_FROM_CACHE` and `REUSE_NMP_FROM_CACHE` are mutually exclusive (alternative read paths); `WINDOWS_CACHE` is compatible with `REUSE_NMP_FROM_CACHE` (rewrites cache with fresh head + reused NMP) but not with `RESUME_FROM_CACHE`.

---

## Data Augmentation (PLAN.md)

Mutation-based augmentation pipeline for Epitope Head training data.

1. `scripts/build_mutation_registry.py` — Build full mutation registry for strict-train proteins with WT verification, mutation enumeration, scoring, and top-N selection.
2. `scripts/run_mutation_pilot.py` — Pilot feasibility run for NetMHCIIpan mutation augmentation with stratified subset validation.
3. `scripts/materialize_augmented_train_set.py` — Materialize augmented train-only `ProteinSample` artifacts from mutation registry and source parquet.

### SLURM

1. `scripts/submit_mutation_augmentation.slurm` — Mutation augmentation sharding (6hr, CPU, 32GB, array job).
2. `scripts/submit_mutation_merge.slurm` — Mutation registry merge (30min, CPU, 16GB).

---

## Inverse Folding — Module K: DPLM Baseline

Infrastructure and unconditional IF baseline (DPLM v1 + GVP adapter).

1. `scripts/train_if_v1.py` — CLI entry point for DPLM v1 adapter training with local smoke test and cluster support.
2. `scripts/validate_if_baseline.py` — Generate sequences and measure structural quality (scTM, scRMSD, pLDDT) for K3 baseline validation. Also supports `--generated-parquet PATH`, where `PATH` may be an IF_IMP run directory or `generated.parquet`; this skips DPLM generation and computes ESMFold/self-consistency metrics directly from existing generated sequences.

### SLURM

1. `scripts/submit_if_train.slurm` — DPLM v1 adapter training (48hr, 2 GPUs, 64GB).
2. `scripts/submit_if_validate.slurm` — K3 baseline validation (12hr, 1 GPU, 64GB). Set `GENERATED_PARQUET=/path/to/if_imp/run_or_generated.parquet` to validate existing IF_IMP CATH-generation results; outputs default to `run/inverse_folding/if_imp/validation/<run-name>/`.

---

## Inverse Folding — Module L: Data Selection & Evaluation (PLAN_DATA_SEL.md)

Test set curation (three-tier) and shared evaluation pipeline.

### Test Set Curation

1. `scripts/run_overlap_filter.py` — Run MMseqs2 overlap filter against CATH training sequences (L1).
2. `scripts/validate_tier1.py` — Validate Tier 1 (gold standard) candidate JSON for the IF test set (L2).
2a. `scripts/build_tier1_candidates.py` — Build a Tier 1 candidate JSON for one allele (L2 source-of-truth). Reads per-allele epitope-head test split, extracts matching EL spans from `data/mhc_if_v2.tsv`, queries PDBe SIFTS `/mappings/best_structures/{uniprot}` (endpoint: `https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}`) for X-ray chains, applies the Tier 1 selection band (coverage 10-50%, resolution ≤ 2.5 Å, chain 100-500 AA, ≥ 2 EL spans), ranks by `epitope_coverage` (configurable), emits a validated JSON. SIFTS responses cached under `outputs/.cache/sifts/{uniprot}.json` (lazy, per-UniProt); delete a cache file to force refetch when PDB entries update. Network 404 cached as `{}` to avoid retry storms. Backed by `inverse_folding/evaluation/tier1_builder.py`.
3. `scripts/prescreen_tier2.py` — Batch pre-screen PDB candidates for Tier 2 (L3). `--selection-mode dual_scorer` (default, frozen) runs the legacy MMseqs2 overlap → epitope-head prefilter → NMP → dual-scorer gate → CATH diversity path. `--selection-mode nmp_only` runs the **Tier 2 v2** path (PLAN_DATA_SEL §12): **no epitope head** (neither prefilter nor gate) and **no structure** in selection; scores the overlap-passed pool with NetMHCIIpan at configurable peptide lengths (`--nmp-screen-lengths '15' | '12-25' | '12,15,18'`), computes `coverage_fraction` (residues covered by ≥1 strong window / length), and optionally draws a density-stratified subsample over `coverage_fraction` via `--sample {uniform,gaussian}` (`--sample-n-target`, `--sample-n-bins`, `--sample-min-per-bin`, `--sample-peak-to-tail`, `--sample-seed`). `--candidate-id-list PATH` reuses an overlap-passed protein_id list (parquet/txt) to skip MMseqs2 (overlap is allele-independent). Writes `tier2_nmp_screen_<allele>_l<lentag>.parquet` (scored) and, when sampling, `tier2_prescreened_v2_<allele>.parquet` + a `.histogram.json` sidecar. Density-stratified sampling is backed by `inverse_folding/evaluation/sampling.py`; coverage by `inverse_folding/evaluation/immunogenicity.compute_coverage_fraction`. The two-stage v2 procedure: S1 = `nmp_only --nmp-screen-lengths 15 --sample uniform` on the 75k pool → ≈5000; S2 = `nmp_only` (all lengths) on the S1 ids `--sample gaussian` → ≈3000.
4. `scripts/validate_tier3.py` — Validate Tier 3 (therapeutic) candidate JSON for the IF test set (L4).
5. `scripts/assemble_if_test_set.py` — Assemble final `test_proteins.parquet` from three tiers.
6. `scripts/download_tier2_candidates.py` — Download and filter RCSB PDB sequence database for Tier 2 candidate pool (single-chain X-ray, 100-500 AA, ≤ 2.5 Å).
7. `scripts/download_test_set_pdbs.py` — Download PDB structure files for assembled IF test set proteins.
8. `scripts/prescreen_uricases.py` — Pre-screen uricase sequences for Tier 3 via MMseqs2 overlap, epitope head prefilter, and batched NetMHCIIpan.
9. `scripts/select_tier2_v2.py` — Tier 2 v2 selection step (PLAN_DATA_SEL §12). Merges one or more NMP-only scored parquets (`--scored-glob` for the length-15 shards or `--scored-parquet` for the multi-length S2 output), de-dups by protein_id, and draws a density-stratified subsample over `--value-col` (default `coverage_fraction`) via `--mode {uniform,gaussian}` (`--n-target`, `--n-bins`, `--min-per-bin`, `--peak-to-tail`, `--mu`, `--seed`). Writes the selected subset parquet + a `.histogram.json` sidecar and prints a per-bin (available, selected) ASCII histogram. Pure CPU, deterministic, rerunnable for tuning the Gaussian before the S3 selection commits. Backed by `inverse_folding/evaluation/sampling.py`. Pairs with `prescreen_tier2.py --selection-mode nmp_only`: S1 = uniform downsample of the length-15 shards → ≈5000; S3 = Gaussian downsample of the multi-length S2 pool → ≈3000.
10. `scripts/predict_esmfold2_gt.py` — Predict GT backbone structures for the uricase missing-structure case set with ESMFold2 (runs in the dedicated `esmfold2` conda env, which shadows fair-esm — NOT immune-design). Reads `to_fold.fasta`, folds each sequence (`ESMFold2InputBuilder().fold(..., num_loops, num_sampling_steps, num_diffusion_samples)`), writes per-protein mmCIF + best-effort PDB (biotite) under `<out>/{mmcif,pdb}/`, and appends one JSONL row per protein (`id, len, mean_plddt, ptm, sec, pass`). ESMFold2 pLDDT is 0–1 scale; GT-eligibility gate is `mean pLDDT ≥ --plddt-min` (default 0.90, AF "very high") AND `pTM ≥ --ptm-min` (default 0.80); raw values are always recorded so the threshold is re-derivable without re-folding. Resume-safe (skips ids already in the manifest) and shardable (`--num-shards`/`--shard-idx`, round-robin, per-shard manifest) for SLURM arrays; `--limit N` folds only the first N (smoke). Requires `biohub/ESMFold2` + `biohub/ESMC-6B` weights pre-cached under `HF_HOME` (compute nodes run offline via `HF_HUB_OFFLINE=1`).
11. `scripts/build_pilot_v2.py` — Build the **pilot** diagnostic subset band-aligned on `coverage_fraction`. pilot = (Tier 1 IEDB-gold anchors whose NMP `coverage_fraction` falls in the same mid-load band as Tier 2) + (Tier 2 drawn from that band with light `--n-sub-bins` stratification via `sample_uniform_bins`), to `--n-total` (default 50). Reads `coverage_fraction` (and the full canonical if_ready schema) from the `fast_v2` parquet (`--fast`) and writes a strict subset as the pilot parquet (`--output`). Tier 1 admitted iff `coverage_fraction ∈ [band_lo - tier1_band_relax, band_hi]` (only the Tier 1 **lower** edge may be widened, to catch a near-bar anchor; Tier 2 band is never relaxed); `--max-tier1 N` caps the anchor count, keeping the N highest-`coverage_fraction` ones (drops from the lower band edge up); Tier 2 fills the remainder. `--band-lo`/`--band-hi` are the Tier 2 mid-load band ([p40,p75] of canonical Tier 2 `coverage_fraction`: 0701 [0.33,0.56], 0401 [0.39,0.64]). Allele-specific, deterministic (`--seed`), pure CPU. Backed by `inverse_folding/evaluation/sampling.py`. Note: `fast_v2` is the source pool, so pilot Tier 2 ⊂ fast Tier 2 by construction.

### SLURM

1. `scripts/submit_assemble_test_set.slurm` — Assemble final IF test set (8hr, 1 GPU, 64GB).
2. `scripts/submit_prescreen_tier2.slurm` — Tier 2 prescreening pipeline. `MODE=dual_scorer` (default) runs the legacy GPU path. `MODE=nmp_only` runs the Tier 2 v2 NMP-only screen (PLAN_DATA_SEL §12) — **pure CPU, no GPU**; submit CPU-only, e.g. `MODE=nmp_only NMP_WORKERS=48 NMP_SCREEN_LENGTHS=15 SAMPLE=none ALLELE="HLA-DRB1*07:01" sbatch --partition=cpu --qos=short --gres=none --cpus-per-task=48 --time=12:00:00 scripts/submit_prescreen_tier2.slurm`. Env: `MODE`, `NMP_WORKERS`, `NMP_BATCH_SIZE`, `NMP_SCREEN_LENGTHS`, `CANDIDATE_ID_LIST` (defaults to `_prescreen_head_results_<allele>.parquet`), `SAMPLE`, `SAMPLE_N_TARGET`, `SAMPLE_N_BINS`, `SAMPLE_MIN_PER_BIN`, `SAMPLE_PEAK_TO_TAIL`, `SAMPLE_SEED`, `N_SHARDS`, `ALLELE`. For the S1 75k scan, submit as a **job array** with small shards (fast backfill, parallel): `N_SHARDS=16 ... sbatch --array=0-15 --partition=cpu --qos=short --gres=none --cpus-per-task=8 --mem=16G --time=3:00:00 ...` — each task scores `sorted(pool)[task_id::N_SHARDS]` → `tier2_nmp_screen_<allele>_l15.shardNNof16.parquet`; concat the shard parquets before S1 sampling.
3. `scripts/submit_download_tier2.slurm` — Download Tier 2 PDB candidate sequences (2hr, CPU, 16GB).
4. `scripts/submit_prescreen_uricases.slurm` — Uricase pre-screening for Tier 3 (24hr, 1 GPU, 64GB). Parameterized: `ALLELE`, `EPITOPE_CKPT`.
5. `scripts/submit_esmfold2_gt.slurm` — ESMFold2 GT prediction for the uricase case set (2hr, 1 GPU, 64GB; `--array=0-7` = 8 shards). Activates the `esmfold2` env, sets `HF_HOME` + `HF_HUB_OFFLINE=1`, runs `predict_esmfold2_gt.py` on shard `$SLURM_ARRAY_TASK_ID`. Overridable env: `FASTA`, `OUT_DIR`, `NUM_SHARDS` (keep == `--array` size), `NUM_LOOPS` (3), `NUM_SAMPLING_STEPS` (100), `NUM_DIFFUSION_SAMPLES` (1), `PLDDT_MIN` (0.90), `PTM_MIN` (0.80). Per-shard manifests (`gt_manifest.shard{k}of{N}.jsonl`) concat after.

## Inverse Folding — Phase B: Experiment Preparation (PLAN_IF.md)

Mechanical preparation jobs that bridge the fixed IF test set and Phase C reference-flow experiments.

### h-map Precompute

1. `scripts/build_if_ready_test_set.py` — B1.5 structure/sequence cleaning gate for Phase C. Reads the assembled biological-sequence test-set parquet plus downloaded PDB/mmCIF structures, extracts resolved single-chain backbone residues, canonicalizes modified amino acids via BioPython's extended PDB residue map, collapses duplicate author residue IDs and alternate conformers, writes cleaned single-chain PDB/mmCIF structures plus an IF-ready parquet whose `sequence`/`sequence_length`/`if_chain_id` exactly match DPLM `load_coords()` output. Original biological sequence, original length, source structure path, residue-number mapping, and failure ledgers are retained for audit.
2. `scripts/precompute_h_maps.py` — Shared B2/B3 CLI that computes wide per-protein h-map parquets from epitope-head `debug.h_raw` and `residue_hotspot`, with metadata sidecars, resume support, failure ledgers, and CATH corpus stats for JSONL training inputs.

### Evaluation Integration

1. `scripts/evaluate_phase_c.py` — B4 end-to-end evaluator for Phase C outputs. Reads `generated.parquet` + test-set parquet and writes schema-aligned `imm_head.parquet`, `imm_nmp.parquet`, and `structural.parquet` under a run directory; mode-switchable (`imm|struct|all`), idempotent per mode, WT-free, and backed by `inverse_folding/evaluation/refold.py` (`esmfold` active, `af3` stub). `--imm-full` opt-in flag additionally emits two long-tables for `imm/all` modes — `imm_head_residues.parquet` (columns: `protein_id, design_id, design_idx, residue_idx, residue_aa, hotspot`) and `imm_nmp_peptides.parquet` (columns: `protein_id, design_id, design_idx, pep_length, pos, peptide, core, rank_EL, el_score`) — leaving the aggregated parquets untouched. Use this when you need per-residue hotspot trajectories or per-peptide NMP landscapes (e.g. benchmarking ProteinMPNN or another external generator at full resolution); leave it off for routine Phase C method comparison where only the aggregates are needed.
2. `scripts/build_wt_facade.py` — Build a WT `generated.parquet` facade (`protein_id`, `design_idx`=0, `sequence`) from a canonical IF-ready test parquet so the WT test set can be scored by `evaluate_phase_c.py` as a single-design generation (the WT immune baseline). Writes the full facade + a `.meta.json` sidecar; `--n-shards N` ALSO writes N round-robin shard facades (`<base>.shardKKofNN.parquet`, split on `sorted(protein_id)[k::N]`) whose names match the shard block in `submit_benchmark.slurm`, so the immune eval can run as a SLURM job array. Pure CPU, deterministic.
3. `scripts/merge_eval_immune_shards.py` — Merge per-shard `evaluate_phase_c` immune outputs into one unified run dir. Concatenates the four immune tables (`imm_head`, `imm_nmp`, `imm_head_residues`, `imm_nmp_peptides`) + `failures.json` from `<output_root>/<allele_tag>/<run_id>.shardKKofNN/` into `<output_root>/<allele_tag>/<run_id>/` and writes a merge `manifest.json`. Each protein lives in exactly one shard → a duplicate `(protein_id, design_idx)` in an aggregated table is a hard error; `--expected-proteins N` asserts final coverage. Pairs with `build_wt_facade.py --n-shards` + `submit_benchmark.slurm` `N_SHARDS`.

### SLURM

1. `scripts/submit_precompute_h_test.slurm` — B2 test-set h-map precompute (24hr, 1 GPU, 64GB). Parameterized: `ALLELE`, `EPITOPE_CKPT`, `INPUT_PARQUET`, `OUTPUT_PARQUET`, `OUTPUT_META`, `WINDOW_BATCH_SIZE`, `RESUME_FROM`.
2. `scripts/submit_precompute_h_cath.slurm` — B3 CATH train h-map precompute (72hr, 1 GPU, 64GB). Parameterized: `ALLELE`, `EPITOPE_CKPT`, `CHAIN_SET`, `SPLITS_JSON`, `SPLIT`, `OUTPUT_PARQUET`, `OUTPUT_META`, `WINDOW_BATCH_SIZE`, `RESUME_FROM`.
3. `scripts/submit_benchmark.slurm` — Shared benchmark/evaluation launcher. `MODE=fasta` and `MODE=iedb` keep the epitope-head benchmark workflows; `MODE=phase_c` runs `evaluate_phase_c.py`. Phase-C overrides: `GENERATED_PARQUET`, `TEST_SET_PARQUET`, `ALLELE`, `EVAL_MODE`, `REFOLD_MODEL`, `PDB_ROOT`, `PHASE_C_OUTPUT_ROOT`, `RUN_ID`, `OVERWRITE`, `FAIL_PCT_THRESHOLD`, `NMP_MODE`, `NMP_WORKERS`, `NMP_BATCH_SIZE`, `NMP_MAX_LENGTHS_PER_CALL`, `NMP_TIMEOUT`, `IMM_FULL`. In Phase C, `NMP_MODE=accelerated` is the default and uses the explicit NMP knobs; `NMP_MODE=original` matches the IEDB benchmark baseline (`batch_size=1`, `max_lengths_per_call=14`, `workers=1`). Set `IMM_FULL=1` to pass `--imm-full` to `evaluate_phase_c.py` and emit per-residue / per-peptide long tables alongside the aggregated parquets. **Facade sharding (job array):** set `N_SHARDS` and submit with `--array=0-(N_SHARDS-1)`; the phase_c branch then treats `GENERATED_PARQUET` and `RUN_ID` as BASE names and inserts `.shardKKofNN` into both (each task scores one pre-split facade shard from `build_wt_facade.py` into its own run dir). Merge afterward with `merge_eval_immune_shards.py` (e.g. as a `--dependency=afterok` job). Only triggers when both `N_SHARDS` and `SLURM_ARRAY_TASK_ID` are set; non-array runs are unchanged.

---

## Inverse Folding — Phase C: Reference Flow (PLAN_IF.md)

Sampling-only and native-sampler generation drivers for the Phase C contribution.

### Drivers

1. `scripts/run_if_phase_c0.py` — C0 native DPLM sampler baseline on the assembled IF test set. Writes `generated.parquet`, `generated.fasta`, `run_config.yaml`, and `manifest.json` under `output_root/<allele_tag>/<run_id>/`.
2. `scripts/run_if_phase_c1.py` — C1 sampling-only position-dependent DFM driver. Consumes B2 h-maps plus a validated YAML config, supports optional trajectory dumps, guarded resume from prior partial/final runs, partial checkpoint writes, and CPU retry on per-design GPU OOM. **Phase D extension** (D1 monitor + D2 logits + D3 commit/revisit): pass `--controller-config <ctrl.yaml>` plus `--head-checkpoint`, `--head-config-dir`, `--head-variant-id`, `--head-device`, `--head-window-batch-size`, `--head-allele-idx` to layer the adaptive controller on top of C1. Controller mode is selected by the YAML's `controller.mode` field (`monitor_only | d2_logits | d3_revisit | d2_d3_full`); presets are `inverse_folding/reference_flow/configs/{d1_monitor,d_monitor_full,d2_logits,d3_revisit,d2_d3_full}.yaml`. Each run emits `refresh_log.jsonl` (per-refresh telemetry, schema widened with nullable `e_i/m_i/rho_i/grace_positions`), `controller_events.parquet` (full D0 event schema with nullable D2/D3/remask-ledger columns), `per_protein_summary.json` (D2/D3 aggregates; `arm` matches `controller.mode`), and `static_window_cache.{parquet,meta.json}` (lazy WT window-score cache). `generated.parquet` schema is unchanged; manifest gains `controller_surface_version` (=3), `controller_mode`, `controller_config`, `d2_config`, `d3_config`, `attribution_config`, `controls_config`, and head provenance fields. When `--controller-config` is omitted (or YAML has `enabled: false`) the driver runs vanilla C1. **Stage B extension** (PLAN_RF_UNI_CTRL.md, typed actionability targeting): the YAML's `controller.targeting` block selects the active-block source — `mode: static_excess` (default, legacy `z_dyn - z_static`) or `mode: typed_actionability`, which builds the typed `A_i(t)` field (`tau_ref_B`, `b_cur`/`b_env`/`b_mem`, `r_ctx`, `v_target`, `u_pressure`, `G`/`g_GR`) and selects active windows from `v_target`; `controller.d3.evidence_source: typed_fresh` is a typed-wiring sanity route, while the primary Aopen mechanism arm keeps `legacy_window_excess`. **Stage B.1** (PLAN_RF_UNI_CTRL.md, RAR 0006 M7): `controller.targeting.within_block_source` selects the *within-block* editable-position ranking — `legacy_excess` (default, bit-for-bit) or `v_target` (routes the typed signal to the D2 position layer); `d2_d3_full_stageB_aopen.yaml` and the Stage C.1 config set `v_target` (`e_fresh` is reserved/deferred). Stage B presets are `inverse_folding/reference_flow/configs/{d_monitor_stageB,d2_d3_full_stageB,d2_d3_full_stageB_aopen,d2_d3_full_stageB_unified_aopen}.yaml`. When `targeting.write_actionability_telemetry: true` and typed mode is active, the run additionally emits `actionability_residues.parquet` (one row per `protein_id × design_idx × seed × refresh_step × residue`; per-residue typed channels plus `legacy_residue_excess` and `active_block_id` for the B4.4 in-block confound diagnostic) and `actionability_refresh_summary.jsonl` (per-refresh `G`/`g_GR_diagnostic`, seed/active window counts incl. pre-cap `num_actionable_windows_pre_cap`, channel means); `refresh_log.jsonl` carries compact sidecar pointers, and the manifest gains `targeting_config`, `global_pressure_config`, `targeting_config_hash`, and the two sidecar paths. `global_pressure.enabled` stays `false` in Stage B — `G`/`g_GR` are logged but never scale beta/lambda (that is the gated Stage C). Static-excess runs keep the legacy artifact set byte-for-byte. **Stage C.1 extension** (PLAN_RF_UNI_CTRL.md §"Stage C.1", actuator-only over the Stage B typed-target Aopen controller): setting `controller.global_pressure.enabled: true` (with `mapping: smoothstep`, `pressure_source: trajectory_thresholded_G`, `g_min: 0.0`) scales D2 `beta` (when `scale_beta`) and the D3/rank `lambda_commit` (when `scale_lambda`) by the trajectory-level pressure `g_GR = smoothstep(B_GR; B_low, B_high, g_min, g_max)`, where `G_step = mean_i ReLU(u_pressure_i - tau_prom)` and `B_GR` is the median of reliable per-refresh `G_step`. The `tau_prom`/`B_low`/`B_high` band is never hardcoded: pass `--global-pressure-calibration-json PATH` (the current pilot artifact is `Results/Analysis/0006-stage-b-typed-targeting-mechanism-check/data/stageC1_global_pressure_calibration_typed_target_thresholded_tauprom11p75.json`); it is stamped into the config before the config hash and fails fast if missing/inverted or if `pressure_source` mismatches. The flag is **required** when pressure is enabled and ignored (manifest-provenance only) when disabled. Telemetry adds `G_step_mean`/`G_step`/`B_GR`/`g_GR_effective`/`beta_eff`/`lambda_eff`/`pressure_burden_bin` to the actionability summary and `beta_base`/`beta_eff` (D2) / `lambda_base`/`lambda_eff` (D3) / `g_GR_effective` to the event rows; the manifest gains `global_pressure_calibration_path`/`_hash`, `global_pressure_tau_prom`, `global_pressure_B_low`/`_B_high`, and `global_pressure_pressure_source`. The primary Stage C.1 config is `inverse_folding/reference_flow/configs/d2_d3_full_stageC_pressure_aopen.yaml` (copies `d2_d3_full_stageB_aopen.yaml`, keeps `d3.evidence_source=legacy_window_excess`, enables thresholded `global_pressure`). When `global_pressure.enabled: false` every output is byte-for-byte identical to the Stage B / legacy runs.

### SLURM

1. `scripts/submit_if_phase_c.slurm` — Shared Phase C launcher (24hr, 1 GPU, 64GB). `MODE=native` runs `run_if_phase_c0.py`; `MODE=reference_flow` runs `run_if_phase_c1.py`. Shared overrides: `ALLELE`, `CHECKPOINT`, `TEST_SET_PARQUET`, `PDB_ROOT`, `OUTPUT_ROOT`, `DEVICE`. Native-mode overrides: `N_DESIGNS_PER_PROTEIN`, `SEED`, `MAX_ITER`, `TEMPERATURE`. Reference-flow overrides: `H_MAPS_PARQUET`, `CONFIG_PATH`, `FAIL_PCT_THRESHOLD`, `SAVE_TRAJECTORIES`, `H_CORPUS_STATS`, `RESUME_FROM`, `N_DESIGNS_PER_PROTEIN`, `SEED`, `N_STEPS`. **Phase D1 overrides** (only consumed in `MODE=reference_flow`): `CONTROLLER_CONFIG`, `HEAD_CHECKPOINT`, `HEAD_CONFIG_DIR` (default `${PROJECT_ROOT}/epitope_head/configs`), `HEAD_VARIANT_ID`, `HEAD_DEVICE` (default `cpu`), `HEAD_WINDOW_BATCH_SIZE`, `HEAD_ALLELE_IDX` (default `0`). Empty `CONTROLLER_CONFIG` disables the controller and ignores the head flags. **Stage C.1 override**: `GLOBAL_PRESSURE_CALIBRATION_JSON` forwards to `--global-pressure-calibration-json` (required when the controller config has `global_pressure.enabled: true`; empty otherwise).

---

## Inverse Folding -- IF Improvement (PLAN_IF_IMP.md)

MapDiff-grounded DPLM inverse-folding improvement path: IPA refiner + IPA geometry sidecar, with 4-arm ablation infrastructure.

1. `scripts/train_if_imp_refiner.py` -- Train the DPLM-compatible IPA refiner on CATH batches. Uses DPLM/ESM alphabet and `[N, CA, C, O]` coordinates, converts to IPA `[N, CA, C, CB, O]`, and saves a refiner checkpoint with config.
2. `scripts/train_if_imp_sidecar.py` -- Train the DPLM-compatible IPA geometry sidecar on CATH backbones. The sidecar is wired into the GVP encoder via a residual on `feats`; objective is DPLM decoder masked-AA recovery with everything but the sidecar frozen. Saves `sidecar_last.pt` + config.
3. `scripts/run_if_imp_refiner.py` -- Generate inverse-folding designs with optional refiner and/or sidecar. Supports the 4 ablation arms (`baseline` / `refiner` / `sidecar` / `sidecar_refiner`) via `--arm`, auto-resolved from `--refiner-checkpoint` and `--sidecar-checkpoint`. `--ablation-mode` emits per-step refiner diagnostics + per-design `ablation_diagnostics.parquet`.
4. `scripts/compare_if_imp_ablation.py` -- Cross-arm ablation comparison consuming 2-4 run dirs from `run_if_imp_refiner.py --ablation-mode`. Computes the 5 PLAN-mandated indicators (refiner-selection shrinkage, fused-vs-base entropy quantile gain, sidecar effect on base entropy, combined-vs-best-single delta, compute cost per arm) into `comparison.parquet` + `comparison_summary.json`.
5. `scripts/diag_if_imp_arms.py` -- Paired-by-design 4-arm diagnostic: runs all enabled arms on the same ~200-protein CATH subset in one process with fixed seed, using `DPLMArmDiagnosticProcessor` to emit ground-truth-aware per-step records (base/refiner/fused top-1 recovery at selected positions, base-vs-refiner agreement, fusion flip rate, reparam preservation rate, draft-init recovery, base entropy mean). Outputs `per_protein_summary.parquet` + `per_step_summary.parquet` (+ optional `per_position_records.parquet` via `--emit-per-position`).

### SLURM

1. `scripts/submit_if_imp.slurm` -- Parameterized IF improvement launcher. `MODE=train_refiner|train_sidecar|generate_refiner|compare_ablation|diag_arms`. Generation overrides include `ARM`, `REFINER_CHECKPOINT`, `SIDECAR_CHECKPOINT`, `ABLATION_MODE=1`, plus inference-time refiner knobs `APPLY_STEPS`/`FUSION_MODE`/`FUSION_ALPHA`. Compare mode reads `BASELINE_RUN`/`REFINER_RUN`/`SIDECAR_RUN`/`COMBINED_RUN`. Diag mode reads `DIAG_N_PROTEINS`, `DIAG_ARMS`, `DIAG_EMIT_PER_POSITION`, `DIAG_RUN_ID`.
2. `scripts/submit_if_imp_refiner_sweep.sh` -- Refiner inference-time sweep orchestrator. Submits 12 sbatch jobs over `apply_steps ∈ {final_only, last_2} × mask_ratio_center ∈ {0.05, 0.10, 0.20} × fusion_mode=residual × fusion_alpha ∈ {0.10, 0.25}` against one trained refiner checkpoint (no retraining). Each submission gets a unique `TAG`; `DRY_RUN=1` prints commands without submitting.

---

## Inverse Folding -- IF Encoder Replacement (PLAN_IF_ENCODER.md)

GeoEGNN-IPA structure encoder that replaces DPLM-IF's frozen GVP encoder. The encoder builds a per-protein backbone kNN graph, runs sparse EGNN message passing (PyG `MessagePassing`), and refines residue hidden states with dense IPA over the original `N/CA/C` rigid frames. DPLM weights stay frozen; trainable parameters are the new encoder + draft head + configured adapter layers. Default behavior of all upstream scripts is unchanged unless `--encoder-kind=geoegnn_ipa` is set.

1. `scripts/train_if_imp_encoder.py` -- Train the GeoEGNN-IPA encoder + adapter parameters on CATH backbones. Loads DPLM via `load_if_task`, swaps `task.model.encoder` for a fresh `GeoEGNNIPAEncoder`, sets `cfg.detach_encoder_feats=False`, optionally reinstalls the decoder adapter shape before freezing, and optimizes `diff_loss + lambda_aux * encoder_loss`. Saves `encoder_last.pt` (via `save_geo_encoder_checkpoint`), `run_config.yaml`, `metrics.jsonl`, `manifest.json`. Knobs include `--adapter-num-layers`, `--adapter-gated`, `--adapter-gate-init`, `--egnn-depth`, `--egnn-hidden-dim`, `--ipa-depth`, `--ipa-hidden-dim`, `--update-coors`, `--use-updated-coord-bias`, `--lambda-aux`.

### SLURM

1. `scripts/submit_if_imp.slurm` (extended) -- Three new modes: `MODE=train_encoder` runs `train_if_imp_encoder.py` (env: `ENCODER_BATCH_SIZE`, `ENCODER_EPOCHS`, `ENCODER_LR`, `ENCODER_ADAPTER_NUM_LAYERS`, `ENCODER_ADAPTER_GATED`, `ENCODER_ADAPTER_GATE_INIT`, `ENCODER_D_MODEL`, `ENCODER_EGNN_DEPTH`, `ENCODER_EGNN_HIDDEN_DIM`, `ENCODER_IPA_DEPTH`, `ENCODER_IPA_HIDDEN_DIM`, `ENCODER_LAMBDA_AUX`, `ENCODER_UPDATE_COORS`, `ENCODER_USE_UPDATED_BIAS`). `MODE=generate_encoder` calls `run_if_imp_refiner.py` with `--encoder-kind=geoegnn_ipa --encoder-checkpoint=$ENCODER_CHECKPOINT` to generate using the trained encoder. `MODE=diag_encoder` calls `diag_if_imp_arms.py` with the same encoder swap so the 4-arm paired diagnostic compares the new encoder against baseline/refiner/sidecar.

---

## Inverse Folding — Module M: Classifier Guidance (PLAN_IF.md)

Inference-time epitope-aware steering with eta sweep.

1. `scripts/run_if_guidance_sweep.py` — Run classifier guidance eta sweep on fixed test set with DPLM IF generation, epitope head scoring, and risk-weighted resampling.

### SLURM

1. `scripts/submit_if_guidance_sweep.slurm` — M3 guidance eta sweep (24hr, 1 GPU, 64GB).

---

## Inverse Folding — Module N: Comparison Baselines

Post-hoc filter baselines for IF evaluation.

1. `scripts/run_if_level1_filter.py` — N1 level-1 post-hoc filter: reads M3 eta=0 candidates, selects argmin-global-risk per protein.
2. `scripts/run_proteinmpnn_baseline.py` — ProteinMPNN inverse-folding baseline with optional NetMHCIIpan post-hoc filter. Wraps the vendored `DRAKES/drakes_protein/ProteinMPNN/` (`parse_multiple_chains.py` + `assign_fixed_chains.py` + `protein_mpnn_run.py`). With `--test-set-parquet`, it resolves exactly the IF-ready parquet rows under `--input-pdb-folder`, rewrites each source PDB/CIF into a private continuous-residue-numbered staging PDB aligned to the parquet sequence, and maps staged file ids back to original `protein_id` in outputs; this keeps ProteinMPNN's official parser/model unchanged while avoiding author-numbering gaps being emitted as `X`. With `--apply-nmp-filter` (requires `--netmhciipan-bin` + `--allele`), each design is scored by `epitope_head.data.netmhciipan_runner.StandaloneRunner.score_batch` over `--nmp-pep-lengths` (default 15); aggregation splits multi-chain designs on `/`, sums `n_sb` (windows with `%Rank_EL ≤ --nmp-rank-threshold`, default 2.0%) and averages `el_rank` per design. Per-protein selection rule via `--nmp-rule {min_n_sb,min_mean_rank}`; ties tie-break with the other metric then `design_idx`. Outputs `generated.parquet` (schema-compatible with `run_if_phase_c0.py`: `protein_id, design_idx, sequence, mpnn_score, mpnn_global_score, seq_recovery, mpnn_temperature, mpnn_sample` + NMP columns when filtered), `generated.fasta`, `run_config.yaml`, and (NMP-on) `selection.parquet` + `selection.json`. ProteinMPNN audit artifacts are kept under `<output>/_mpnn_raw/{seqs,parsed_chains.jsonl,staged_inputs_manifest.json,stage_id_to_protein_id.json}` unless `--keep-mpnn-raw` is passed.

### SLURM

1. `scripts/submit_if_baselines.slurm` — Module-N parameterized baseline launcher. `MODE=level1` runs `run_if_level1_filter.py` (CPU-friendly: submit with `--partition=cpu --gres=none --time=01:00:00 --mem=16G` overrides for short CPU-only runs); `MODE=proteinmpnn` runs `run_proteinmpnn_baseline.py` on the canonical IF-ready test-set parquet and forwards `MPNN_*` knobs (`MPNN_MODEL_NAME`, `NUM_SEQ_PER_TARGET`, `SAMPLING_TEMP`, `MPNN_BATCH_SIZE`, `MPNN_SEED`, `MPNN_DEVICE`, `DESIGN_CHAINS`, `CA_ONLY`, `OVERWRITE`); when `APPLY_NMP_FILTER=1` it also forwards `NETMHCIIPAN_BIN` (default `${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan`), `NMP_PEP_LENGTHS`, `NMP_RANK_THRESHOLD`, `NMP_RULE`, `NMP_BATCH_SIZE`, `NMP_MAX_LENGTHS_PER_CALL`, `NMP_WORKERS`, `NMP_TIMEOUT`. Default GPU resources (1 GPU, 24h, 64GB, MIG partition); `INPUT_PDB_FOLDER` defaults to `${WORK_DIR}/if_test_set/pdbs_if_ready/{0701,0401}` and `TEST_SET_PARQUET` defaults to `${WORK_DIR}/if_test_set/if_ready/test_proteins_if_ready_<allele>.parquet` based on `ALLELE`; `OUTPUT_DIR` defaults to `${RUN_DIR}/baselines/{level1_filter,proteinmpnn}/...`. Replaces the legacy `submit_if_level1_filter.slurm` (deleted 2026-05-08; functionality fully subsumed by `MODE=level1`).

---

## Legacy SLURM Duplicates (Pre-Consolidation)

> These scripts predate the parameterization protocol in `scripts/CLAUDE.md`.
> They are allele-specific copies that should eventually be merged into their parent scripts.
> **Do not add new files in this pattern** — use env-var overrides instead.

| Script | Duplicates | Should use |
|--------|-----------|------------|
| `submit_aug_drb0401.slurm` / `submit_aug_drb1501.slurm` | `submit_mutation_augmentation.slurm` | `ALLELE=... sbatch submit_mutation_augmentation.slurm` |
| `submit_train_drb0401.slurm` / `submit_train_drb1501.slurm` | `submit_cnn_enhance.slurm` | Parameterize with `ALLELE`, `DATA_DIR`, `REGISTRY` |
| `submit_prescreen_uricases_0401.slurm` | `submit_prescreen_uricases.slurm` | `ALLELE=... sbatch submit_prescreen_uricases.slurm` |
| `submit_epi_benchmark.slurm` | `submit_benchmark.slurm` | Already parameterized, use `submit_benchmark.slurm` |

---

## Analysis (Optional)

1. `scripts/analysis/statistics_v2.py` — Full dataset statistics for `data/mhc_if_v2.tsv`.
2. `scripts/analysis/analyze_overlap.py` — Overlap analysis across sources in V2.
3. `scripts/analysis/analyze_hla_distribution.py` — HLA distribution analysis for V2 (plots + stats).
4. `scripts/analysis/analyze_clean_pids.py` — Protein accession statistics for IEDB clean files.
5. `scripts/analysis/check_alleles.py` — Spot-check allele JSON formatting.
6. `scripts/analysis/diagnose_e1.py` — Post-hoc diagnostics for E1 (Dilated CNN) ablation: per-protein AUC distribution and first-layer conv kernel visualization.
7. `scripts/analysis/window_boundary_bias.py` — Analyze window-level boundary bias to check if distance to chunk boundaries systematically biases window selection.
8. `scripts/analysis/tier1_structural_analysis.py` — Tier 1 IEDB structural analysis. Reads `tier1_candidates.json` + per-chain PDBs, computes per-residue (RSA, SS-3state, Cα B-factor z, contact number) via DSSP + BioPython, runs per-protein effect sizes (Cliff's δ, Mann–Whitney U) and pooled SS-coil log-OR via DerSimonian-Laird random-effects meta-analysis. Emits `per_residue.csv`, `per_protein_stats.json`, `meta_analysis.json`, and optional F2 supplementary PDF panels. Backed by `inverse_folding/analysis/structural_features.py`.
9. `scripts/analysis/imm_landscape.py` — Phase C immunogenicity landscape + sample-size variability plots. Consumes `imm_head.parquet` + `imm_nmp.parquet` from `scripts/evaluate_phase_c.py --eval-mode imm` (auto-falls back to `.partial/` if the run never reached manifest write). Emits: `fig1a_landscape.png` (6-panel histograms for head `global_risk`/`mean_hotspot`/`max_hotspot` and NMP `n_strong_binders`/`n_weak_binders`/`mean_best_rank`); `fig1b_head_vs_nmp.png` (hexbin of head `global_risk` vs NMP `mean_best_rank` with Pearson + Spearman); `fig2_std_vs_k.png` (bootstrap sample-std of each metric vs k=2..max designs/protein, band = 2.5–97.5% across proteins); plus `coverage_summary.csv` recording designs/protein. CLI: `--eval-dir`, `--output-dir` (default `{eval-dir}/analysis`), `--bootstrap-n` (default 200), `--seed`, `--total-proteins-expected` (optional, reports fully-failed protein count).
10. `scripts/analysis/sc_gr_monitor_probe.py` — SC-GR monitor analysis (PLAN_RF_SC_GR.md Task SC0.5 / doc/Self-Cond_GR.md §6). Reads a monitor run's `sc_gr_probe_refresh.parquet` (from `run_if_phase_c1.py` with `self_conditioned_gr.enabled`) plus an oracle burden parquet (`imm_head.parquet` from the NoD / D3 reference run) and scores every candidate burden estimator — `arm` (fresh / self_conditioned) × `refresh_step` × aggregator (`mean_excess`, `topm_lse`, `supra_mass_tau_<label>`) — against the oracle: per-protein `Spearman(B_sc, oracle)`, a tercile crosswalk diagonal, `P(probe low | oracle high)` (the true-high-as-low rate that cost C.1 its high-burden gain), and `Recall@High = P(probe high | oracle high)`. Picks the best estimator by Recall@High first, Spearman second (the tail, not the mean correlation, is the gate). Per-protein probe burden = median `B_sc` over the protein's designs; oracle aggregated per protein by `--oracle-aggregate` (median/mean). Pure functions (`compute_monitor_table`, `select_best_metric`, `spearman`, `tercile_bins`) + thin CLI: `--run-dir`, `--oracle-parquet`, `--oracle-risk-column` (default `global_risk`), `--oracle-aggregate` (default `median`), `--output` → monitor summary JSON. Monitor-only readout; changes no config or actuation. Lightweight (pandas + numpy only).
11. `scripts/analysis/typed_targeting_confound.py` — Stage B B4.4 in-block confound diagnostic for the typed-targeting 50-protein gate (PLAN_RF_UNI_CTRL.md §B4.4). Stage B.0 changes only the active-block discovery source (→ `v_target`) while D2 keeps selecting in-block positions by the legacy per-window-subtraction `residue_excess`; a block discovered purely by `b_env`/`b_mem` can have legacy excess ≈ 0 everywhere, so D2 falls back to an entropy tie-break and a "typed ≤ static" gate result becomes ambiguous (targeting failure vs in-block-selection artifact). Joins `actionability_residues.parquet` (per-residue `v_target`, `legacy_residue_excess`, `active_block_id`, `active_target_flag`) with the D2 rows of `controller_events.parquet` (`position_i`, `block_id`, `realized_benefit_flag`) on `(protein_id, design_idx, seed, refresh_step, residue==position_i)` and computes: `d2_edit_vtarget_percentile` (per-block then averaged; ≈1 = edits track the typed signal, ≈0.5 = entropy-driven), `d2_edit_on_zero_legacy_excess_rate` (uses **legacy excess, not `b_cur`** — the two diverge at WT-high-risk windows), `realized_benefit_rate` stratified by {high-`v_target` / `legacy_excess=0`} (the gate-attribution test: high-stratum benefits + zero-legacy-stratum ≈ 0 + most edits in the latter ⇒ fix Stage B.1 in-block ranking, not reject typed targeting), and a `d2_edit_active_target_rate ≈ 1` sanity. Pure `compute_typed_targeting_confound(actionability_df, events_df, *, high_vtarget_quantile=0.75)` + thin CLI (`--run-dir`, `--high-vtarget-quantile`, `--output` → default `<run-dir>/analysis/typed_targeting_confound.json`). The `legacy_residue_excess` / `active_block_id` columns are written by `run_if_phase_c1.py` (controller telemetry) so the analysis is a drift-free two-artifact join.

---

## Notebooks

1. `notebooks/uricase_q00511_structural_profile.ipynb` — Head-free per-residue structural profile of *A. flavus* uricase (Q00511 / PDB 1R4U). Downloads 1R4U, aligns ATOM sequence to UniProt Q00511 (warns on numbering shift — observed `+1`), computes RSA via FreeSASA (Tien 2013 maxASA), DSSP 3-state SS (gracefully skips if mkdssp missing), Cα B-factor, contact number (8 Å cutoff), NetMHCIIpan 4.3 %Rank_EL for DRB1*07:01 + DRB1*04:01 (auto-detects bundled binary), and a stub loader for Kaiyi's `conservation_scores.tsv`. Annotates the 9 published catalytic residues as a positive control; surfaces residue-identity mismatches. Outputs `notebooks/outputs/q00511_profile.{parquet,csv}` plus two figures under `notebooks/outputs/figures/`. Reuses `inverse_folding.analysis.structural_features.compute_contact_number_from_coords` and `MAX_ASA_TIEN`.

---

## Atlas (Optional / Not Used)

1. `scripts/atlas/atlas_find_positions.py` — Substring search for Atlas peptide positions in UniProt sequences.
2. `scripts/atlas/analyze_atlas_stats.py` — Atlas-only stats for multi-hit positions.

---

## Legacy V1 (Deprecated)

1. `scripts/legacy_v1/build_atlas_clean_v1.py` — Atlas clean build (V1).
2. `scripts/legacy_v1/build_detailed_hla_table.py` — Atlas donor/allele table construction (V1).
3. `scripts/legacy_v1/build_combined_json.py` — Atlas+IEDB JSON export (V1).
4. `scripts/legacy_v1/process_hla_aggregated.py` — Atlas HLA aggregated processing (V1).
5. `scripts/legacy_v1/process_alleles_tissues.py` — Atlas allele/tissue cleanup (V1).
6. `scripts/legacy_v1/unify_clean_format.py` — Align Atlas clean format with IEDB (V1).
7. `scripts/legacy_v1/collect_atlas_accessions.py` — Atlas accessions collection (V1).
8. `scripts/legacy_v1/overlap_v1.py` — Legacy overlap analysis between Atlas and IEDB (V1).
9. `scripts/legacy_v1/analyze_dr_dp_dq_ratio.py` — DR/DP/DQ ratio analysis (V1).
10. `scripts/legacy_v1/inspect_headers.py` — Header inspection helper (V1/diagnostic).
11. `scripts/legacy_v1/add_flanks.py` — Legacy flanking extractor (superseded by `finalize_v2.py`).

---

## Standard Raw Input

- `data/iedb_raw.tsv` — Current standard filename for the IEDB raw export. Some scripts fall back to the legacy name if present.
