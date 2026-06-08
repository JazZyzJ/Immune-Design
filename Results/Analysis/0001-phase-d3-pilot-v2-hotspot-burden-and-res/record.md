# Phase D3 pilot_v2 hotspot burden and residue landscape

## Provenance

| field | value |
|---|---|
| dataset | `pilot_v2_HLA-DRB1_07_01`, 50 proteins, 400 designs per arm |
| NoD run | `phaseD_pilotv2_nod_n8_seed42__20260604T172853Z` |
| D3 runs | `phaseD_pilotv2_d3_t060_n8_seed42`, `phaseD_pilotv2_d3_t075_n8_seed42` |
| controller mode | D3 revisit only; `d2.enabled=false`; `t_start=0.60` and `t_start=0.75` |
| allele | `HLA-DRB1*07:01` |
| code version | `a503e0c` |
| local Python | `3.11.0` |
| libraries | `pandas 2.2.3`, `numpy 1.26.4`, `pyarrow 14.0.2` |
| testset path | `/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_07_01/pilot_v2_d3_t060_t075_n8__20260604T144231Z/meta/testset/pilot_v2_HLA-DRB1_07_01.parquet` |
| NoD aggregate inputs | `/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_07_01/phaseD_pilotv2_nod_n8_seed42__20260604T172853Z/{generation/generated.parquet,eval_immune/imm_head.parquet,eval_immune/imm_nmp.parquet,eval_structure/structural.parquet,eval_immune/imm_head_residues.parquet}` |
| D3 t060 aggregate inputs | `/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_07_01/pilot_v2_d3_t060_t075_n8__20260604T144231Z/{generation/phaseD_pilotv2_d3_t060_n8_seed42/generated.parquet,eval_immune/phaseD_pilotv2_d3_t060_n8_seed42_imm_full/imm_head.parquet,eval_immune/phaseD_pilotv2_d3_t060_n8_seed42_imm_full/imm_nmp.parquet,eval_structure/phaseD_pilotv2_d3_t060_n8_seed42_struct/structural.parquet,eval_immune/phaseD_pilotv2_d3_t060_n8_seed42_imm_full/imm_head_residues.parquet}` |
| D3 t075 aggregate inputs | `/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_07_01/pilot_v2_d3_t060_t075_n8__20260604T144231Z/{generation/phaseD_pilotv2_d3_t075_n8_seed42/generated.parquet,eval_immune/phaseD_pilotv2_d3_t075_n8_seed42_imm_full/imm_head.parquet,eval_immune/phaseD_pilotv2_d3_t075_n8_seed42_imm_full/imm_nmp.parquet,eval_structure/phaseD_pilotv2_d3_t075_n8_seed42_struct/structural.parquet,eval_immune/phaseD_pilotv2_d3_t075_n8_seed42_imm_full/imm_head_residues.parquet}` |
| static window inputs | `/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_07_01/pilot_v2_d3_t060_t075_n8__20260604T144231Z/generation/phaseD_pilotv2_d3_t060_n8_seed42/static_window_cache.parquet`; t075 cache has same schema and row count |
| head checkpoint digest in static cache | `97bf10c1378f2dde68e00f2740d636c742c10cee424c9811fed99384a36f1dce` |
| head config hash in static cache | `acb178be390f07c36c054b32e9c8108821523fc7881235b55ee0deed455dcf4c` |

### Measurement 1 Objective

Measure D3-vs-NoD paired hotspot outcomes by NoD burden group.

Pre-registered burden definitions:

| scope | threshold source | low_or_equal criterion | high criterion |
|---|---|---|---|
| `all50` | median NoD per-protein mean hotspot coverage | `<= 0.37566800531242095` | `> 0.37566800531242095` |
| `tier2` | median NoD per-protein mean hotspot coverage among Tier2 proteins | `<= 0.3768656716417911` | `> 0.3768656716417911` |

Pre-registered paired criteria:

| criterion | formula |
|---|---|
| hotspot coverage | `n_hotspot_positions / sequence_length` |
| D3 delta | `D3 metric - matched NoD metric`, same `protein_id + design_idx` |
| hotspot-coverage improved design | `delta_hotspot_coverage < 0` |
| hotspot-coverage over-intervention design | `delta_hotspot_coverage > 0` |
| scTM preserved design | `delta_scTM >= -0.05` |

### Measurement 1 Inputs

Inputs are the aggregate `generated.parquet`, `imm_head.parquet`, `imm_nmp.parquet`, `structural.parquet`, and testset parquet paths listed in Provenance.

Columns used:

| table | columns |
|---|---|
| generation | `protein_id`, `design_idx`, `sequence` |
| head aggregate | `protein_id`, `design_idx`, `mean_hotspot`, `max_hotspot`, `n_hotspot_positions`, `global_risk` |
| NMP aggregate | `protein_id`, `design_idx`, `n_strong_binders`, `mean_best_rank` |
| structure aggregate | `protein_id`, `design_idx`, `scTM`, `pLDDT`, `bb_RMSD`, `recovery` |
| testset | `protein_id`, `tier`, `sequence_length`, `head_n_hotspot`, `head_global_risk`, `netmhciipan_n_strong` |

### Measurement 1 Method

1. Load NoD, D3 t060, and D3 t075 aggregate tables.
2. Merge each arm by exact `protein_id + design_idx`.
3. Compute `hotspot_coverage = n_hotspot_positions / len(sequence)`.
4. Compute paired deltas as `D3 - NoD`.
5. Compute NoD per-protein mean hotspot coverage over 8 designs.
6. Assign burden groups using the thresholds in Measurement 1 Objective.
7. Aggregate over design rows and per-protein rows. No rows were dropped.

### Measurement 1 Results

Full results are in `data/burden_split_summary.csv` and `data/design_level_paired_deltas.csv`.

| scope | arm | burden_group | n_proteins | mean_delta_hotspot_coverage | protein_hotspot_coverage_improved_fraction | mean_delta_n_hotspot_positions | mean_delta_n_strong_binders | mean_delta_scTM | scTM_preserved_fraction_delta_ge_minus_0p05 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all50 | t060 | low_or_equal | 25 | 0.002814 | 0.520000 | 0.880000 | 2.315000 | 0.001625 | 0.930000 |
| all50 | t060 | high | 25 | -0.010133 | 0.560000 | -2.475000 | -2.825000 | 0.001120 | 0.960000 |
| all50 | t075 | low_or_equal | 25 | 0.011929 | 0.320000 | 1.995000 | 1.770000 | 0.002275 | 0.935000 |
| all50 | t075 | high | 25 | -0.010811 | 0.640000 | -2.680000 | -6.265000 | 0.000705 | 0.965000 |
| tier2 | t060 | low_or_equal | 18 | 0.003887 | 0.500000 | 1.152778 | -2.368056 | 0.002127 | 0.923611 |
| tier2 | t060 | high | 17 | -0.011710 | 0.529412 | -2.566176 | -1.948529 | 0.000296 | 0.948529 |
| tier2 | t075 | low_or_equal | 18 | 0.007532 | 0.388889 | 0.763889 | -1.555556 | 0.002162 | 0.930556 |
| tier2 | t075 | high | 17 | -0.007943 | 0.588235 | -0.845588 | -8.029412 | -0.000544 | 0.948529 |

### Measurement 2 Objective

Measure protein-level 8-design distribution summaries for representative rows selected by deterministic filters:

| case type | selection filter |
|---|---|
| distribution Pareto row | Tier2 row with `delta_hotspot_coverage_mean < 0`, `hotspot_coverage_improved_designs >= 5`, `delta_n_strong_binders_mean < 0`, `delta_scTM_mean >= -0.03`, `d3_scTM_mean >= 0.85` |
| distribution over-intervention row | Tier2 row with `delta_hotspot_coverage_mean > 0`, `hotspot_coverage_improved_designs <= 3` |
| distribution structure tradeoff row | `delta_hotspot_coverage_mean <= 0`, `delta_scTM_mean < -0.025` |

### Measurement 2 Inputs

Inputs are `data/per_protein_8design_distribution_summary.csv` derived from Measurement 1.

Columns used:

`protein_id`, `arm`, `tier`, `n_designs`, `delta_hotspot_coverage_mean`, `delta_hotspot_coverage_median`, `hotspot_coverage_improved_designs`, `delta_n_hotspot_positions_mean`, `delta_n_strong_binders_mean`, `n_strong_binders_improved_designs`, `delta_scTM_mean`, `scTM_preserved_designs_delta_ge_minus_0p05`, `scTM_drop_designs_delta_lt_minus_0p05`.

### Measurement 2 Method

1. Group design-level paired deltas by `protein_id + arm`.
2. For each metric, compute NoD mean, D3 mean, delta mean, delta median, and number of improved paired designs among 8.
3. Apply the deterministic filters in Measurement 2 Objective.
4. Persist selected rows to `data/distribution_case_selection.csv`.

### Measurement 2 Results

| case_label | protein_id | arm | tier | delta_hotspot_coverage_mean | hotspot_coverage_improved_designs | delta_n_strong_binders_mean | delta_scTM_mean | scTM_drop_designs_delta_lt_minus_0p05 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| distribution_pareto_strongest | 1USN_A | t075 | 2 | -0.065152 | 6 | -31.875000 | 0.002030 | 0 |
| distribution_pareto_high_burden_large_nmp_drop | 6K8P_C | t075 | 2 | -0.039116 | 7 | -50.375000 | 0.005122 | 0 |
| distribution_pareto_conservative | 7D66_J | t075 | 2 | -0.073958 | 6 | -10.500000 | -0.011951 | 1 |
| distribution_over_intervention | 7W62_A | t060 | 2 | 0.062500 | 3 | 20.250000 | -0.005780 | 0 |
| distribution_head_regression_nmp_near_neutral | 3Q80_B | t075 | 2 | 0.072821 | 2 | -1.500000 | 0.004811 | 1 |
| distribution_structure_tradeoff_weak_immune_shift | 3CYQ_D | t075 | 2 | -0.003788 | 4 | -4.625000 | -0.050261 | 3 |
| distribution_immune_coverage_improves_baseline_structure_weak | 3IEZ_B | t060 | 2 | -0.048684 | 5 | 1.000000 | -0.043954 | 4 |

### Measurement 3 Objective

Measure residue-level head landscape descriptors beyond hotspot count using the available `imm_head_residues.parquet` field `hotspot`.

Pre-registered residue landscape descriptors:

| descriptor | formula |
|---|---|
| residue hotspot quantile | empirical quantile of per-residue `hotspot` values within one design |
| positive mass | `sum(max(hotspot, 0))` |
| excess mass above 0.5 | `sum(max(hotspot - 0.5, 0))` |
| top10 positive-mass fraction | sum of largest `ceil(0.10 * L)` positive hotspot values divided by positive mass |
| positive-mass entropy norm | entropy of positive hotspot mass distribution divided by `log(L)` |
| positive-mass gini | Gini coefficient over non-negative hotspot values |
| residue z-score | `(hotspot - mean(hotspot)) / std(hotspot)`, per design, `ddof=0`; zero vector if std is zero |

### Measurement 3 Inputs

Inputs are `imm_head_residues.parquet` for NoD, D3 t060, and D3 t075.

Columns used: `protein_id`, `design_idx`, `residue_idx`, `residue_aa`, `hotspot`.

### Measurement 3 Method

1. Load per-residue hotspot rows for NoD, D3 t060, and D3 t075.
2. Aggregate residue rows into one landscape row per `protein_id + design_idx + arm`.
3. Compute descriptors listed in Measurement 3 Objective.
4. Pair D3 rows to NoD rows by `protein_id + design_idx`.
5. Compute descriptor deltas as `D3 - NoD`.
6. Aggregate descriptor deltas by scope, arm, and NoD burden group. No residue rows were dropped.

### Measurement 3 Results

Full design-level and per-protein results are in `data/residue_landscape_design_deltas.csv`, `data/residue_landscape_per_protein_summary.csv`, and `data/residue_landscape_burden_split_summary.csv`.

| scope | arm | burden_group | n_design_rows | mean_delta_hotspot_q95 | mean_delta_hotspot_q99 | mean_delta_hotspot_max | mean_delta_hotspot_gt0p5_count | mean_delta_hotspot_positive_mass | mean_delta_hotspot_excess_mass_gt0p5 | mean_delta_residue_z_q95 | mean_delta_residue_z_max | mean_delta_residue_z_gt2_count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all50 | t060 | low_or_equal | 200 | -0.010719 | -0.026344 | -0.024320 | 0.880000 | 11.967975 | 11.710407 | 0.010110 | 0.043266 | 0.025000 |
| all50 | t060 | high | 200 | -0.297057 | -0.161718 | -0.195793 | -2.475000 | -10.173482 | -9.320994 | -0.046463 | -0.006375 | -0.105000 |
| all50 | t075 | low_or_equal | 200 | 0.206198 | 0.214038 | 0.186204 | 1.995000 | 19.573062 | 18.966067 | -0.053474 | -0.063537 | -0.860000 |
| all50 | t075 | high | 200 | -0.344083 | -0.346339 | -0.334068 | -2.680000 | -17.381157 | -16.571510 | -0.031615 | 0.018874 | -0.105000 |
| tier2 | t060 | low_or_equal | 144 | -0.804620 | -0.887210 | -0.850694 | 1.152778 | -17.555776 | -18.166207 | -0.050390 | 0.014227 | -0.430556 |
| tier2 | t060 | high | 136 | 0.291767 | 0.453055 | 0.436187 | -2.566176 | 13.287984 | 14.210112 | 0.004694 | 0.031998 | 0.441176 |
| tier2 | t075 | low_or_equal | 144 | -0.328237 | -0.409588 | -0.415029 | 0.763889 | -4.588260 | -5.082486 | -0.092694 | -0.055762 | -0.597222 |
| tier2 | t075 | high | 136 | -0.083995 | -0.143730 | -0.121171 | -0.845588 | -3.674820 | -3.283612 | -0.004985 | 0.017719 | 0.492647 |

### Measurement 4 Objective

Record which head fields are available in transferred pilot_v2 artifacts for generated designs and which logits-level fields are static-only.

### Measurement 4 Inputs

Inputs are `imm_head.parquet`, `imm_head_residues.parquet`, `static_window_cache.parquet`, and the generated-design evaluation artifacts listed in Provenance.

### Measurement 4 Method

1. Read the schema of the available head artifacts.
2. Record row counts for the NoD aggregate/residue examples and t060 static cache example.
3. Summarize per-protein static window `z_static` values from the t060 static cache.
4. Persist field availability to `data/head_field_availability.csv`.

### Measurement 4 Results

| artifact_type | available columns | n_rows_example |
|---|---|---:|
| imm_head | `protein_id,design_id,design_idx,global_risk,mean_hotspot,max_hotspot,n_hotspot_positions` | 400 |
| imm_head_residues | `protein_id,design_id,design_idx,residue_idx,residue_aa,hotspot` | 92488 |
| static_window_cache | `protein_id,allele,sequence_md5,sequence_length,window_start_0b,window_end_0b,k,z_static,head_checkpoint_digest,head_config_hash,score_scale,window_k_min,window_k_max` | 149604 |
| dynamic_window_logits_generated_designs | no transferred artifact field | 0 |

Static `z_static` global summary is in `data/static_window_z_global_summary.csv`; per-protein static summary is in `data/static_window_z_summary.csv`.

| metric | mean_across_proteins | median_across_proteins | min_across_proteins | max_across_proteins |
|---|---:|---:|---:|---:|
| n_static_windows | 2992.080000 | 2758.000000 | 581.000000 | 6713.000000 |
| z_static_mean | -13.993286 | -14.016075 | -14.250168 | -13.646138 |
| z_static_median | -14.341282 | -14.356160 | -14.415430 | -14.203707 |
| z_static_std | 1.303856 | 1.249552 | 0.517921 | 2.411422 |
| z_static_q90 | -13.380252 | -13.454585 | -13.875473 | -12.476375 |

## Artifacts

| file | columns |
|---|---|
| `data/analysis_metadata.json` | JSON: thresholds, source paths, row counts, artifact file list |
| `data/design_level_paired_deltas.csv` | `protein_id`, `design_idx`, `tier`, `arm`, `seq_len`, burden labels, NoD/D3/delta columns for hotspot coverage, hotspot count, mean/max hotspot, global risk, NMP strong count, NMP rank, scTM, pLDDT, bb_RMSD, recovery |
| `data/per_protein_8design_distribution_summary.csv` | `protein_id`, `arm`, `tier`, `n_designs`, burden labels, per-protein NoD/D3/delta mean/median and improved-design counts for aggregate immune/NMP/structure metrics |
| `data/burden_split_summary.csv` | `scope`, `arm`, `burden_group`, thresholds, protein/design counts, mean/median deltas and improvement fractions for aggregate immune/NMP/structure metrics |
| `data/headline_burden_results.csv` | same columns as `burden_split_summary.csv` |
| `data/distribution_case_selection.csv` | selected per-protein distribution rows from `per_protein_8design_distribution_summary.csv` with `case_label` |
| `data/residue_landscape_design_deltas.csv` | `protein_id`, `design_idx`, `tier`, `arm`, burden labels, NoD/D3/delta columns for residue hotspot quantiles, counts, masses, concentration metrics, and per-design residue z-score metrics |
| `data/residue_landscape_per_protein_summary.csv` | per-protein 8-design means and improved-design counts for residue landscape metrics |
| `data/residue_landscape_burden_split_summary.csv` | scope/arm/burden summary of residue landscape deltas |
| `data/headline_residue_landscape_results.csv` | same columns as `residue_landscape_burden_split_summary.csv` |
| `data/head_field_availability.csv` | `artifact_type`, `source_path`, `columns`, `n_rows_example` |
| `data/static_window_z_summary.csv` | per-protein static window `z_static` count and distribution metrics |
| `data/static_window_z_global_summary.csv` | cross-protein summary of static window `z_static` metrics |

## Reproduction

Deterministic regeneration from the source parquets listed in Provenance with Python `3.11.0`, pandas `2.2.3`, numpy `1.26.4`, and pyarrow `14.0.2`; formulas and grouping rules are specified in the Method sections above.
