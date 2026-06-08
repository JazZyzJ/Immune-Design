# Phase-D pilot50 Diagnostics: Static-Prior Correlation & D3 Recommit Localization

Reusable analysis record. Other agents can consume the numbers and CSV artifacts here
directly instead of recomputing. All values are derived from the run artifacts listed
under **Provenance**; no model re-execution is required to reuse them.

## Provenance

| Field | Value |
|---|---|
| Allele | HLA-DRB1*07:01 |
| Dataset | `pilot_v2_HLA-DRB1_07_01` (pilot50, high epitope coverage), 50 proteins × 8 designs = 400 |
| D3 run dir | `Results/RF/HLA-DRB1_07_01/phaseD_pilot50r2_d3_t075_n8_seed42__20260607T185243Z` |
| noD run dir | `Results/RF/HLA-DRB1_07_01/phaseD_pilot50r2_nod_n8_seed42__20260607T185243Z` |
| git_sha | `a503e0ccb6754828c8f6fbd82208663fb147684f` |
| IF checkpoint digest | `b6ca4f7e…d50b3be5` |
| Head checkpoint digest | `97bf10c1…a36f1dce` |
| Base sampler | `amplification.form = constant_one` (g ≡ 1; position-dependent schedule disabled), `schedule.base_form = linear`, `n_steps = 100`, `remask.enabled = true` |
| Controller (D3 run) | `mode = d3_revisit`, `t_start = 0.75`, `refresh_interval = 5` (→ 5 refreshes at steps 75/80/85/90/95), `d2.enabled = false`, `d3.lambda_commit = 0.5`, window range k=12–25 |

Both arms share the same 50 proteins, seed (42), checkpoint, and base sampler; the only
difference is the D3 controller on/off.

---

## Diagnostic 1 — Static-prior persistence ρ = corr(H(x^WT), H(x^gen))

### Objective

Quantify whether residue positions that are high-risk in the WT hotspot
field remain high-risk in the generated-sequence hotspot field. This is the precondition
for a static (WT-derived) prior being usable to schedule generation.

**Pre-registered decision rule.**

- ρ high → high-risk positions are sequence-invariant → static prior usable.
- ρ low → high-risk landscape is sequence-specific → static prior not usable.

### Inputs

| Field | Source file | Column |
|---|---|---|
| H(x^WT) | `<run>/meta/h_maps_pilot50_r2_DRB1_07_01.parquet` | `h_processed` (per-residue list, length = `sequence_length`) |
| H(x^gen) | `<run>/eval_immune/imm_head_residues.parquet` | `hotspot` (one row per `protein_id × design_idx × residue_idx`) |

Validation: `spearman(h_processed, h_raw)` over the 50 proteins = 1.000 (min 0.9999), so
the WT field choice does not affect rank-based results.

### Method

1. For each protein, align the WT field and each design's gen field by `residue_idx`
   (equal length per protein; mismatched lengths skipped — none occurred).
2. Per `(protein, design)`: compute Spearman and Pearson correlation across residue
   positions.
3. Per protein: average the 8 design gen fields position-wise, then correlate the
   design-averaged gen field against the WT field (removes per-design sampling noise).

### Results

| Aggregation | N | Spearman mean | Spearman median | Spearman IQR | Spearman range | Pearson mean |
|---|---|---|---|---|---|---|
| per-(protein, design) | 400 | 0.383 | 0.405 | [0.232, 0.576] | [−0.617, 0.906] | 0.352 |
| per-protein (8-design avg) | 50 | 0.491 | 0.494 | [0.380, 0.652] | — | 0.512 |

Per-protein heterogeneity (design-averaged Spearman):

- Highest: `5L4J_B` 0.824, `8TG4_A` 0.813, `1V6X_A` 0.788, `1WZE_B` 0.774, `2EFO_A` 0.751.
- Lowest: `5ELD_B` −0.189, `1ZR3_A` −0.116, `5QKA_D` 0.179, `6T06_B` 0.182, `1PZD_A` 0.211.

### Artifacts

- `data/m1_rho_per_design.csv` — columns: `protein_id, design_idx, spearman, pearson` (400 rows).
- `data/m1_rho_per_protein.csv` — columns: `protein_id, seq_len, spearman_davg, pearson_davg` (50 rows, sorted ascending by ρ).
- `data/m1_rho_summary.json` — aggregate statistics above.

---

## Diagnostic 2 — D3 recommit localization enrichment

### Objective

Quantify whether D3 recommit (remask) events are preferentially placed on
residues with high online dynamic risk, versus distributed independently of the dynamic
signal.

**Pre-registered decision rule.**

$$
\mathrm{enrich} = \frac{P(\mathrm{remask}\mid \text{high dynamic-risk})}{P(\mathrm{remask}\mid \text{low dynamic-risk})}
$$

- enrich ≈ 1 → remasks independent of dynamic risk (generic remask).
- enrich > 1 → remasks enriched on dynamic hotspots (dynamic signal is used).

### Inputs

| Field | Source file | Column |
|---|---|---|
| Dynamic risk per residue per refresh | `<run>/generation/refresh_log.jsonl` | `m_i` (EMA risk-memory array, length L), keyed by `protein_id, design_idx, refresh_step` |
| Remask events | `<run>/generation/controller_events.parquet` | rows where `event_type == "D3"`; keys `protein_id, design_idx, refresh_step, position_i`; also `m_i`, `commit_score`, `reason` |

All 13,632 D3 rows have `remask_flag = True` (controller_events logs only realized
remasks; the eligible-residue denominator is taken from `refresh_log.m_i`).

### Method

1. Build the residue-cell pool: for every `(protein, design, refresh_step, residue_idx)`
   in `refresh_log`, record `m_i` and a boolean `remasked` (whether that cell appears in
   the D3 event set). Pool size = 449,640 cells; 13,632 remasked.
2. Conditioning variable is `m_i`, the same EMA risk memory D3 ranks on
   (`commit_score = z(ℓ_cur) − 0.5·z(m_i)`).
3. Enrichment computed under three splits:
   - **A**: `m_i > 0` vs `m_i == 0` (degenerate here — field is dense, only 279 zero cells).
   - **B**: top quartile (Q4) vs bottom quartile (Q1) of `m_i`, among `m_i > 0` cells (pooled).
   - **C**: per-refresh median split of `m_i > 0` cells, ratio of remask rates, averaged over refreshes.

**Methodological note (denominator validity).** The pool includes residues that may be
masked (ineligible to remask) at a given refresh. Because the base sampler uses g ≡ 1,
the unmask order is independent of immune risk, so committed-status is approximately
independent of `m_i`; the committed fraction cancels in the high/low ratio. Splits B and C
additionally restrict to `m_i > 0` and control for per-refresh scale.

### Results

| Split | P(remask \| high) | P(remask \| low) | enrich |
|---|---|---|---|
| A: m_i>0 vs ==0 | 0.0303 | 0.0358 | 0.85 (degenerate, ignore) |
| B: Q4 vs Q1 (m_i>0) | 0.0528 | 0.0194 | **2.73** |
| C: per-refresh median split (n=1813 refreshes) | — | — | median **2.00**, mean 2.41 |

Supporting statistics:

- `m_i` of remasked cells: mean 6.35, median 4.83. `m_i` of kept cells: mean 4.32, median 3.00.
- Remask `reason` attribution (controller's own label): `immune_risk` 5,004 / `low_confidence` 8,628 → immune-driven fraction **0.367**.

### Artifacts

- `data/m2_enrichment_summary.json` — all numbers above (`enrich_Q4Q1`, `enrich_perRefresh_median/mean`, `reason_immune_frac`, `mi_remasked_*`, `mi_kept_*`, counts).

---

## Cross-reference: final immune metrics (same two runs)

Context for the diagnostics above (paired, 50 proteins × 8 designs); lower head
`global_risk` and lower NMP `n_strong_binders` indicate lower immunogenicity.

| Metric | D3 | noD |
|---|---|---|
| head `global_risk` mean | −4.289 | −4.385 |
| head `global_risk` median | −4.185 | −4.064 |
| head `max_hotspot` mean | 11.856 | 11.642 |
| NMP `n_strong_binders` mean | 47.60 | 49.68 |
| NMP `mean_best_rank` mean | 0.523 | 0.569 |

D3 controller activity (`<run>/generation/per_protein_summary.json`, 400 designs):
`n_refreshes` = 5 (all), `total_D3_events` = `total_recommits` mean 34.08 (range 11–80),
`total_D2_events` = 0, `new_hotspot_rate` mean 0.0074, `productive_revisit_immune_only`
mean 0.552, `productive_revisit_joint` mean 0.543, `churn_rate` mean 0.096.

Sources: `<run>/eval_immune/imm_head.parquet`, `<run>/eval_immune/imm_nmp.parquet`,
`<run>/generation/per_protein_summary.json` for both D3 and noD run dirs.

---

## Reproduction

Diagnostics computed with `pandas` + `scipy.stats.spearmanr`. Inputs are the parquet/jsonl
files under the two run dirs in **Provenance**. Artifacts regenerate deterministically from
those files; no GPU or model load required.
