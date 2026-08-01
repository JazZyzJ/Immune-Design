# PROTOCOL: Full-cohort refinement and tetramer-gate handoff

## Scope / when to use

Use this protocol for constrained RF redesign runs whose hard functional anchors are frozen and
whose final product must preserve a homo-tetrameric, cross-protomer active site.

This protocol supersedes the original two-list workflow. There is no longer a monomer-derived
“List 1” for tetramer prediction or “List 2” for refinement. **Every technically valid RF design is
given the same refinement search budget.** Every seed remains traceable through refinement and the
tetramer gate.

The historical Q00511 List-1/List-2 counts (225 and 42) are provenance only. They must not be used
as current selection targets.

## Core decision

The old workflow selected refinement seeds before measuring the structure that matters. That is
not defensible for uricase:

1. A monomer prediction cannot measure tetramer formation, either repeated interface, or
   cross-protomer catalytic-pocket registration.
2. Frozen WT anchor identities do not guarantee their three-dimensional geometry or the packing of
   the surrounding interface.
3. Refinement is not assumed to repair or degrade tetramer quality monotonically. Non-anchor edits
   can alter both monomer and quaternary structure.
4. Therefore monomer quality must not decide which valid seeds are allowed to enter refinement.
   The required comparison is:

   `same-backend predicted WT → original RF seed → canonical refined output`

5. The computational tetramer readout is a **plausibility gate**, not an activity classifier, until
   assembly/activity data calibrate its sensitivity and specificity.

## Input cohort and allowed exclusions

The refinement input is the complete generated cohort, represented by a self-contained parquet
with at least:

- `protein_id`
- `sequence`
- `design_idx` or a stable `design_id`
- generation seed/provenance when available

The only pre-refinement exclusions are technical:

- missing or invalid sequence;
- duplicate identity when the protocol calls for unique sequences;
- missing constraint-manifest entry;
- hard-anchor identity mismatch;
- missing backbone or other input that makes refinement impossible.

Do **not** exclude a valid seed using `scTM`, pLDDT, monomer active-site RMSD, immune score, or a
pre-tetramer structure tier. These metrics remain diagnostics and may explain later failure.

## Refinement contract

1. **Refine every listed seed.** `scripts/refine_rf_designs.py --seed-table` refines every row in
   the table. Job-array sharding changes scheduling only; it must not change per-seed search
   parameters.
2. **Use equal search controls.** Beam width, maximum rounds, proposal limits, refold cap,
   temperature, structure-gate profile, and immune objective must be identical across seeds in one
   comparison cohort.
3. **Freeze the declared anchors.** Anchor validation is fail-fast. This preserves residue identity,
   not tetramer geometry.
4. **Keep the live monomer structure gate.** During search it prevents accepting an obvious
   monomer-level degradation. It is an edit-acceptance guardrail, not a seed-eligibility filter and
   not evidence of tetramerization.
5. **Preserve every outcome.** Record seeds that reach immune zero, seeds that improve but remain
   non-zero, seeds with no accepted edit, and technical failures. Do not silently drop the latter
   categories.
6. **Choose a canonical refined output per seed for the primary tetramer cohort.**
   `best_count0_per_seed.parquet` is primary when immune zero is reached: minimum edit burden,
   then maximum monomer structural quality according to the merger contract. For a seed that does
   not reach zero, retain its lowest-residual-risk `top1` output as a diagnostic, clearly labeled
   non-product. Additional beam outputs may be tetramer-scored as a sensitivity panel.

## Required paired tetramer cohort

For every parent protein, the tetramer manifest must contain:

- a real WT sequence predicted with the same backend, MSA protocol, ligand stoichiometry, sampling
  settings, and evaluation code;
- every original RF seed;
- the canonical refined output from every seed;
- stable `parent`, `seed_id`, `refined_id`, and `stage={WT,seed,refined}` fields.

This paired design separates two questions:

- **Design effect:** how far is the RF seed from predicted WT?
- **Refinement effect:** did refinement improve or worsen that seed?

A refined-vs-WT comparison alone confounds these effects. Wet-lab selection may ultimately use only
the refined product, but protocol calibration requires all three stages.

Run coordinate interface metrics and Rosetta InterfaceAnalyzer for the complete paired cohort
during the present calibration phase. Once thresholds are experimentally validated, a later
production protocol may use coordinate metrics on all variants and reserve Rosetta for boundary
cases. Per-residue alanine scanning remains final-cohort only.

The tetramer prediction and interface metric definitions are specified in
[tetramer_prediction_gate.md](tetramer_prediction_gate.md).

## Metric hierarchy

Metrics are used in layers. A favorable lower layer cannot compensate for failure in an earlier
layer, and immune improvement cannot compensate for tetramer pathology.

### Layer 0 — technical validity (hard fail)

- prediction completed and the expected raw arrays/confidence files exist;
- exactly four complete protein chains and six protein–protein chain pairs;
- two copies of each biological interface class plus two diagonal pairs;
- coordinate and required Rosetta summaries are finite;
- same-backend predicted WT baseline is present;
- required catalytic and interface residues are mapped without numbering ambiguity;
- no gross coordinate clash or corrupted structure.

### Layer 1 — tetramer topology and gross interface preservation

Evaluate both biological interface classes separately and use the weaker copy of each class.

The following are **provisional computational-pathology bands**, not activity thresholds:

| Signal | Provisional red flag | Warning / tier | Rationale |
|---|---:|---:|---|
| `interface_topology_bsa_ratio` | `< 3` | `3–4` | The 14 active-parent WT predictions span 4.14–10.27. The ratio is used only as a lower-bound topology check; very large values are denominator-driven and must not be linearly ranked. |
| Within-class BSA copy balance, `min_copy/max_copy` | `< 0.80` | `0.80–0.90` | Active-parent WTs are all ≥0.976. Strong asymmetry means one of the two nominally equivalent copies has failed. |
| Per-interface WT-relative BSA and contact retention | both `< 0.80` for either interface | `0.80–0.85`; rank tiers at 0.85/0.90 | Marks gross interface loss. BSA and contacts are corroborating views, not independent votes. |
| `complex_TM_vs_wt` | `< 0.80` | inspect 0.80–0.90 | Gross global tetramer displacement only; it is not sufficiently discriminating near WT. |

For a strict operational screen, a red flag means “exclude from the current computationally
plausible set or repeat prediction,” not “experimentally inactive.” Borderline structures should be
re-predicted before exclusion.

Do not use a pooled absolute ipTM, pair-ipTM, gpde, pLDDT, BSA, contact, or raw Rosetta-energy
threshold across parents. Active WT predictions already span substantially different absolute
scales.

### Layer 2 — WT-relative interface physical quality

For each interface class, calculate conservative design-minus-WT or design/WT values:

- coordinate BSA retention;
- contact retention;
- Rosetta dSASA retention;
- `dG/dSASA` energy-density delta;
- packstat delta;
- shape-complementarity delta;
- buried-unsatisfied-polar delta;
- H-bond count plus H-bond energetic fraction.

Use `dG/dSASA` as the primary Rosetta energy signal. Raw dG is supporting information because it is
strongly area-dependent. Coordinate BSA and Rosetta dSASA are likewise alternate measurements of
interface area and must not be double-weighted.

Until activity/assembly labels exist, these physical metrics define Pareto tiers and orthogonal
red flags; they are not hard-AND requirements. In the current 3,208-design reference batch, requiring
all energy/packing metrics to be no worse than WT would remove essentially the entire cohort.

### Layer 3 — cross-protomer catalytic-pocket fidelity

After topology is plausible, prioritize the geometry that directly connects assembly to catalysis:

- Q00511 cross-protomer Lys11/Thr58/His257-to-Asn255 distances and deviations from predicted WT;
- local inter-chain PAE/confidence at the catalytic interface when available;
- catalytic-site copy symmetry across all four pockets;
- ligand pose only as advisory evidence until the backend is ligand-pose validated.

This layer is closer to the activity mechanism than global complex TM. It still requires wet-lab
calibration and cannot prove tetramer formation by itself.

### Layer 4 — immune objective

Immune score ranks only structures that have survived the preceding plausibility layers. Report
immune value, edit count, and structural tier separately; do not collapse them into one compensatory
weighted score.

## Monomer metrics: retained, but not a seed shortlist

From `eval_structure/structural.parquet` retain:

- `scTM`, pLDDT, and global CA RMSD;
- aggregate active-site side-chain RMSD;
- `predicted_active_site_min_pLDDT`;
- matched-anchor count.

From `structural_residues.parquet` retain:

- `cat_max_scRMSD`: maximum side-chain RMSD over direct-functional catalytic residues;
- `shell_max_scRMSD`: maximum over the remaining protected shell.

These remain useful for the live refinement accept gate and for explaining failures. They no longer
define T1/T2 seed classes or limit which seeds receive refinement.

## Evidence supporting the current bands

### 1R51 crystal reference

The persistent reference is already stored under:

`/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/tetramer_gate/reference_metrics/1R51/`

It anchors metric implementation and the two-interface mechanism. It is one standardized crystal
conformation, not a replicate ensemble, so it cannot provide statistical acceptance tolerances.
The two interfaces have different native energy/area scales and must never share one raw cutoff.

### Characterized-active parent batch

`char24_active_nmp0_all/tetramer_gate.parquet` contains 14 known-active parent WT predictions and
3,208 designs, all with complete Protenix coordinate and Rosetta metrics. It contains no Q00511 and
no design activity labels.

Useful descriptive findings:

- worst-interface WT-relative BSA/contact/dSASA retention has medians near 0.91;
- conservatively requiring both BSA and contact retention ≥0.80 leaves 89.8% of designs, while
  0.85 and 0.90 leave 70.3% and 40.6%, respectively; the red flag above is narrower and requires
  concordant loss of both signals at the same interface;
- only 8/3,208 designs have within-class BSA copy balance <0.80;
- coordinate BSA and Rosetta dSASA are nearly redundant (Spearman ρ≈0.98);
- raw dG and dG/dSASA are nearly redundant (Spearman ρ≈0.99);
- a single parent-matched WT prediction does not estimate predictor or Rosetta repeat variance.

This batch therefore supports gross-pathology thresholds and metric de-duplication. It does not
identify an activity boundary or estimate specificity.

## Reproduce: full-cohort refinement

Use a seed table containing the complete generated cohort. Do not call
`scripts/select_redesign_shortlists.py` for the current workflow.

```bash
SEED_TABLE=<full-generated-cohort.parquet> \
CONSTRAINT_MANIFEST=<run-dir>/meta/constraint_manifest.yaml \
GATE_SCTM_MIN=<run-calibrated-live-refinement-floor> \
GATE_CAT_MAX_SCRMSD_MAX=<run-calibrated-live-refinement-ceiling> \
GATE_ACTIVE_SITE_MIN_PLDDT_MIN=<run-calibrated-live-refinement-floor> \
ALLELE='<allele>' \
OUT_DIR=<run-dir>/refine_all \
sbatch --array=0-<n_shards_minus_1>%<max_concurrent> scripts/submit_refine.slurm
```

`--seed-table` bypasses best-of-N selection and refines every listed row. The input table must be
frozen before submission, and its row count/hash must be recorded with the run. Merge shards with
`scripts/merge_refine_shards.py`; use its per-seed canonical output for the primary tetramer cohort.

## Calibration still required

Before promoting provisional bands to a true experimental gate:

1. predict each parent WT with multiple seeds/samples to estimate backend and Rosetta repeat noise;
2. repeat borderline seed/refined structures rather than deciding from one sample;
3. add assembly measurements where possible;
4. join activity and expression/specific-activity readouts without converting sub-threshold values
   to undifferentiated zero;
5. estimate sensitivity first: an active design outside a proposed band falsifies a hard cutoff;
6. calibrate Q00511 separately before transferring thresholds across the wider uricase family.

Until then, output the full metric table, stage-paired deltas, plausibility flags, and Pareto tiers.

## Final wet-lab order instantiation

After the paired holo cohort and all required interface/ligand metrics are
complete, use
[`final_tetramer_wetlab_order.runner.md`](final_tetramer_wetlab_order.runner.md)
to turn the measured cohort into a traceable wet-lab order. The case file must
freeze all hard gates, axis components, normalization scope, cross-axis rule,
experimental-arm quotas, source-lineage deduplication, and sequence-diversity
requirements.

Final-order ranking must preserve this protocol's layer hierarchy. In
particular, candidate-specific Rosetta physical terms remain WT-relative,
interface-specific supporting evidence until activity calibration establishes
otherwise; they must not silently become a pooled activity score.
