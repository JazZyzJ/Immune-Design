# Active-15 Uricase Evolution and Tetramer-Structure Audit

## 1. Scientific objective

Build a per-parent, per-position evidence map for the 15 experimentally active,
single-domain characterized uricases in RAR 0034. The map will support a later
near-WT redesign in which only positions inside WT MHC-II epitope cores are made
editable. This task does **not** choose the final parent/allele, generate RF
constraint manifests, or treat active-site membership as an automatic hard lock.

The primary question is:

> For each active WT parent and HLA allele, which residues inside each strong
> 9-mer core remain plausible to open when single-site conservation, pairwise
> compatibility, tetramer contacts, and interface energetics are considered?

The analysis is descriptive. Wet-lab activity remains the endpoint; evolutionary
and predicted-structure signals can prioritize or veto candidate edits, but do not
prove activity preservation.

## 2. Frozen cohort and source contracts

- Cohort: the 15 rows tagged `redesign_panel_r2` in
  `data/Uricase/08_redesign_panel/panel_manifest.tsv` from local RAR
  `0034-wt-active15-panel-epitope-cores-across-t` (git provenance `f80b134`).
- Sequence input: `redesign_panel15.fasta`; lengths 295--328 aa; canonical
  mature-sequence indexing is `index_0b = position_1b - 1`.
- Immune input: the RAR 0034 per-allele core tables/JSON for
  `HLA-DRB1_04_01`, `HLA-DRB1_07_01`, and `HLA-DRB1_15_01`, using
  NetMHCIIpan `rank_EL < 2` and the recorded core-start rule.
- Activity/expression are real wet-lab values from RAR 0034 and are annotations,
  not inferred placeholders.
- Q00511 controls: the existing deep EVcouplings model and 1R51 crystal tetramer
  results are retained as reproducibility controls, but the 15-parent MSA search
  is rerun uniformly rather than mixing old filtered/server MSAs.

Input acceptance requires 15 unique protein IDs, 15 unique AA20 sequences,
header-declared length equal to sequence length, and exact agreement between the
FASTA, panel manifest, and epitope JSON. Any mismatch fails before compute.

## 3. Reuse boundary and corrections

The workflow reuses the scientific definitions from the Q00511/ADA pipelines:

- query-centered A3M columns;
- exact EVcouplings weights at identity threshold `theta = 0.8`;
- PLMC through the EVcouplings wrapper, with `lambda_h = 0.01` and
  `lambda_J = 0.01 * 20 * (L - 1)`;
- gap-excluded entropy conservation, weighted WT-identity frequency (`pWT`), and
  top-`0.5L`/`L`/`2L` long-range coupling enrichment (`|i-j| >= 6`);
- full-sequence PLMC Hamiltonian decomposition;
- EC precision against heavy-atom contacts at 5 Angstrom.

Required corrections before reuse:

1. A single normalized focus MSA must be shared by PLMC and conservation.
   Query is first and unique, IDs are unique, and retained symbols are `AA20 + '-'`.
   A malformed query or focus-length mismatch fails; noncanonical natural rows are
   dropped with an explicit ledger. A source record is identified by its complete raw
   A3M header, not only its accession token: identical full-header/sequence records are
   collapsed, a repeated full header with conflicting focus sequences fails, and distinct
   alignment hits sharing one accession are retained under deterministic PLMC-safe IDs
   with an accession-collision ledger. No row is silently padded or truncated.
2. All target names, lengths, paths, structures, and cohort labels are inputs;
   no ADA/Q00511 or cluster path is hard-coded in Python.
3. A WT-identity lock requires both conservation and `pWT`; high conservation
   alone can describe a non-WT family consensus.
4. Raw Hamiltonians are never compared across parents. Every score is calibrated
   only against that parent's WT and natural-sequence reference.

## 4. Tasks and falsifiable gates

### T0 -- Freeze and validate inputs

Artifacts:

- checksummed copy of the RAR 0034 panel files in the Della `work/` analysis
  directory;
- `input_manifest.tsv/json` with sequence MD5/SHA-256, activity, expression,
  length, and per-allele core counts;
- an explicit numbering contract.

Acceptance: all cohort and numbering checks in Section 2 pass. Otherwise stop.

### T1 -- Rebuild one uniform query-centered MSA per parent

Run the registered local GPU ColabFold search against
`uniref30_2302 + colabfold_envdb_202108` with environment search enabled and
`--filter 0`, in one 15-query batch so database load is amortized. Normalize each
A3M and create focus alignments at minimum row coverage 0.6, 0.7, and 0.8.

Measurements per parent:

- raw and normalized row counts;
- invalid-row and duplicate-ID ledgers with explicit disposition;
- exact `N_eff` and `N_eff/L` at each coverage threshold;
- query MD5 and column count.

Gate:

- `N_eff/L >= 10`: covariance-qualified;
- `5 <= N_eff/L < 10`: covariance is exploratory and cannot create a hard veto;
- `N_eff/L < 5`: do not emit covariance-derived lock masks.

Conservation remains reportable when covariance is not qualified, provided the
coverage-stability measurements are valid.

### T2 -- Fit and validate one PLMC model per parent

Fit one descriptive 500-iteration model for every parent on the normalized
minimum-row-coverage 0.6 focus MSA,
with `theta=0.8`, `lambda_h=0.01`, and scaled
`lambda_J`. Record model/input hashes, runtime, effective samples, valid rows/sites,
and optimizer status.

Acceptance requires exact target-sequence identity, model indices `1..L`, and all
`L` sites valid. The T1 depth gate controls whether covariance may generate a
lock mask; it does not suppress the requested descriptive model, EC table, or
within-parent Potts output. Optimizer status is diagnostic: `MAXIMUMITERATION` or
`ROUNDING_ERROR` does not independently fail a model if the structural and
stability gates pass.

### T3 -- Compute evolution evidence and analytical WT-lock tiers

Per-position table fields include, at minimum:

- `index_0b`, `position_1b`, WT amino acid;
- weighted `C_nogap`, `pWT`, and gap fraction for coverage 0.6/0.7/0.8;
- minimum `C_nogap`, minimum `pWT`, and maximum gap fraction across filters;
- `sigma_0p5L`, `sigma_L`, `sigma_2L`, their percentiles, and
  `sigma_robust_pct = min(percentile_0p5L, percentile_L, percentile_2L)`;
- top pair memberships and partners from the complete EC table.

Evolution-only lock tiers (full position lists in 0-based and 1-based numbering):

- `C90_stable`: min conservation and min `pWT >= 0.90`, max gap `<=0.20`;
- `C80_stable`: the same at `0.80`;
- `C70_stable`: the same at `0.70`;
- `sigma90_L` and `sigma80_L`: raw top-10%/20% `sigma_L`, max gap `<=0.50`;
- `sigma90_robust` and `sigma80_robust`: the same thresholds on
  `sigma_robust_pct`;
- union/intersection grids are analytical artifacts only, not RF manifests.

Report rank correlations and top-decile/top-quintile Jaccard across the three
sigma cutoffs. A single top-L percentile is not presented as stable covariance.

### T4 -- Calibrate full-sequence Potts scores

For each model, score its WT and all normalized natural rows with coverage at
least 0.95 after WT-imputing gap positions. Report:

- `H_total`, `H_fields`, and `H_couplings`;
- WT percentile within that model's natural reference;
- natural-reference quantiles and unique-sequence count;
- both unweighted and MSA-weighted empirical summaries where feasible.

If the 0.95-coverage reference is too small for stable empirical quantiles, retain
the scores and mark calibration insufficient; do not invent a threshold. Future
candidates will use `Delta H = H(candidate) - H(own WT)` and within-model
percentiles only.

### T5 -- Build parent-specific tetramer contact and energy tiers

Use the Q00511 1R51 crystal as the experimental control. For other parents, use
the complete five-sample WT Protenix tetramer ensemble only when it passes the
four-chain topology/confidence checks; predict only missing WTs with the same
registered MSA/Protenix route and the same five-sample contract.

For each parent, derive from its own coordinates rather than transferring Q00511
indices:

- a BSA-ranked class and a separate mechanistic class for every D2 matching;
  `interface_1`/`interface_2` remain BSA labels and are not assumed to mean
  catalytic/assembly;
- `catalytic_interface_contact_5A` (the contact analogue of the old `mid` tier)
  and `catalytic_or_assembly_contact_5A` (the contact analogue of the old `full`
  tier), assigned from homolog-specific cross-subunit relay geometry;
- per-position contact frequency across five samples and two symmetry copies,
  with explicit `5of5`, `4of5`, and `1of5` consensus lists;
- diagonal-contact controls;
- per-position Rosetta alanine-scan `DeltaDeltaG_bind` on representative
  interface-1 and interface-2 chain pairs, with native Ala/Gly/Pro marked
  non-interpretable for side-chain hot-spot claims;
- descriptive energy tiers `>0`, `>=1`, and `>=2` REU, without declaring a
  universal biological cutoff.

The legacy active-site/pocket projection is reported as a separate annotation and
as a legacy-union comparison only. Functional analogs with a non-Q00511 residue
identity remain annotated rather than disappearing. The annotation is not
automatically included in the primary contact tiers or later lock recommendation.

Structure acceptance requires complete four-chain sequence mapping, three
resolvable D2 matchings, and exactly one matching that contains the two
symmetry-related composite catalytic relays. If both or neither biological
matching satisfies the relay geometry, the sample requires manual review rather
than inheriting the BSA label. Low-confidence or topology-failing predictions
yield `structure_unqualified`, not fabricated full/mid masks.

### T6 -- Validate covariance against monomer and tetramer structure

For each covariance-qualified parent, score top `L/2`, `L`, and `2L` long-range
EC precision against:

1. within-chain contacts;
2. all contacts in the biological tetramer.

Report the tetramer-minus-monomer precision gain and ECs explained only by the
tetramer. The inherited operational gate is top-L tetramer precision `>=0.60`;
below it, sigma remains descriptive and cannot act as a hard lock.

### T7 -- Join evolution/structure evidence to WT epitope cores

Produce one row per parent, allele, core, and core residue, including core register
position, binding rank, all evolution fields, contact/energy tiers, and active-site
annotation. Summaries must include:

- collisions for `all-core`, `P1/P4/P6/P9`, `P1+P4`, and `P1-only` policies;
- per-core count of positions free of each evidence tier;
- cores for which every plausible editable position is evolution- or
  structure-sensitive;
- implied recovery if only the remaining positions were opened.

This task ends with the complete trade-off surface. It explicitly does not choose
one allele/parent or generate a constraint config.

### T8 -- Reproducibility and handoff

Create a RAR containing the exact commands, hashes, job IDs/logs, software
versions, QC ledgers, full tables, compact JSON position lists, and a scientific
record. Sync the stable analysis archive to the local Results reference area.

## 5. Cost and feasibility model

Measured source costs:

- Q00511 unfiltered local ColabFold MSA: 1 query, 8 CPU + 1 GPU, 180 GB request,
  3m42s wall, 55.8 GB MaxRSS, 17,305 A3M rows.
- ADA two-query MSA batch: 8 CPU + 1 GPU, 240 GB request, 9m09s wall,
  59.0 GB MaxRSS.
- Q00511 PLMC 100 iterations: 8 CPU, 4m24s wall, 1.67 GB MaxRSS, 160.5 MB model.
- ADA PLMC 500 iterations: 48 CPU requested (about 10 effectively used), 3m20s
  wall, 2.35 GB MaxRSS, 231.9 MB model.

Bound for this panel:

- one 15-query MSA batch: one GPU allocation, expected minutes to tens of minutes;
  hard request bound is the registered 4-hour/240-GB launcher;
- 15 PLMC models: approximately 2.4--3.5 GB model storage total and, using the
  measured 100-iteration Q00511 time as a conservative linear bound, at most
  about 5.5 CPU-hours per model at 500 iterations on 8 cores, or 82.5 CPU-hours
  total; a bounded job array makes wall time approximately the slowest model;
- structure predictions are limited to missing WT parents; existing 12 WT
  Protenix tetramers and the Q00511 crystal are reused;
- Rosetta scans are CPU-only and are submitted separately so an energy failure
  cannot invalidate completed evolution/contact artifacts.

No GPU is allocated to PLMC or downstream tabular analysis.

## 6. Deliverables

The final analysis directory and RAR will contain:

- frozen input/provenance manifests;
- raw and normalized per-parent MSAs plus QC/coverage tables;
- PLMC models, complete EC tables, and model summaries;
- per-position conservation/covariance tables and exact WT-lock mask lists;
- per-parent Potts calibration tables;
- per-parent structure/contact/energy tables and QC;
- per-core joined evidence and policy-collision summaries;
- a cohort-level scientific summary that separates measurements, inferences,
  failed gates, and unresolved trade-offs.

## 7. Execution status (2026-08-02)

- T0--T7: complete for all 15 parents. Canonical joined evidence is under
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/therapeutic_enzymes/uricase_active15_ev_structure_v1/04_join_v1/evidence_join`.
- Production jobs were whole-cohort, non-array allocations: MSA `11901471`,
  normalization `11902558`, PLMC `11903824`, evolution `11905624`,
  structure/energy `11905871`, and join `11907087`.
- T8 RAR creation was explicitly deferred by the user. No RAR record or registry
  mutation was made. The non-RAR report and compact evidence bundle were returned
  to `/Users/jerry/Project/MHC-IF/Results/Reference/Uricase_Active15_EV_Structure_Audit/`.
- No parent/allele candidate or RF constraint config was selected or generated.
