# PROTOCOL: LuxSit parent/variant Protenix-holo analysis and refinement selection

## Scope / when to use

Use this protocol for LuxSit parent-background redesigns in which the 19 common safety anchors
are hard-fixed while the three published LuxSit-i optimization sites are free:

- parent genotype: `R60/A96/M110` (`RAM`)
- LuxSit-i genotype: `S60/L96/V110` (`SLV`)

The protocol supports exact-parent, exact-LuxSit-i, clean partial-recovery, and alternative
genotypes without applying cross-identity sidechain RMSD. It is intended for Protenix-DTZ holo
ensembles and produces a structure-first refinement shortlist. Immune metrics are reported and
may break close structural ties, but they do not rescue a structure failure.

First instantiation:
`Results/RF/HLA-DRB1_15_01/luxsit_parent_core117_drb1501_b1aopen_beta5p0_free6096110_n200_b1_seed10__20260716T224938Z`.

## Scientific interpretation boundary

Candidate apo refolds are not required. The analysis called `apo-reference sensitivity` compares
the same candidate holo coordinates with WT apo and WT holo references. It tests whether reference
choice changes the gate or ranking. It must not be described as ligand-induced conformational
change in the candidate.

The protein-only PDB extracted from a protein-ligand prediction is still a holo conformation. It is
not an apo prediction.

## Data contract

Required per run:

- `generation/generated.parquet`: one unique 117-aa sequence per design.
- `eval_structure/raw_holo/predictions/<cache_key>/seed_*/predictions/*sample_{0..4}.cif`.
- matching `summary_confidence_sample_{0..4}.json` files.
- WT parent and LuxSit-i Protenix apo and DTZ-holo references.
- a constraint manifest with exactly 19 hard anchors and monitored positions 60, 96, and 110.
- `eval_immune/imm_nmp.parquet` for refinement-cost context.

Fail fast if a design lacks five holo samples, any sample lacks the 117-aa protein, a DTZ ligand
does not contain 29 heavy atoms, or the sequence/cache-key mapping is ambiguous.

## Genotype labels

For every sequence, record `aa60`, `aa96`, `aa110`, the three-letter genotype, and:

- `exact_parent`: `RAM`.
- `exact_i`: `SLV`.
- `clean_partial_1`: exactly one LuxSit-i identity and two parent identities.
- `clean_partial_2`: exactly two LuxSit-i identities and one parent identity.
- `alternative`: at least one identity is neither the parent nor LuxSit-i identity at that site.
- `n_i_recovered`, `n_parent_retained`, and `n_alternative`.

Report per-site allele frequencies and compare the observed exact-`SLV` count with the independent
expectation from the three marginal LuxSit-i-allele frequencies. Enrichment above that expectation
is evidence of co-recovery, not proof that the structure model caused it.

Without token-level generation telemetry or an unguided matched arm, final-sequence recovery may
be called `pipeline-level spontaneous recovery`; do not call it a native structural prior.

## Alignment and reference rules

Use three coordinate frames:

1. **Global frame:** align all 117 matched C-alpha atoms. Use for global C-alpha RMSD and the
   globally aligned invariant-anchor sidechain RMSD.
2. **Pocket frame:** align C-alpha atoms of the invariant 19 anchors. Use for DTZ pose, catalytic
   geometry, free-3 packing, and pocket contact fingerprints.
3. **Identity-matched sidechain frame:** after pocket alignment, compare a free-site sidechain only
   when candidate and reference residue identities match.

Reference policy:

- exact `RAM`: parent reference is primary.
- exact `SLV`: LuxSit-i reference is primary.
- clean partial and alternative genotypes: report distance to both references and use the union of
  the two functional reference envelopes for gating.
- invariant-19 residues have the same identities in both references; use the better-matching
  reference for admissibility and report both for directionality.

Never compare sidechain RMSD across different residue identities.

## Metric layers

### G0 — integrity and confidence

- sequence length = 117; all 19 hard anchors preserved.
- five Protenix holo samples present; DTZ heavy-atom count = 29 in every sample.
- global `has_clash`, protein-ligand local clash, pTM, ipTM, ligand-chain pLDDT, and pair gPDE.
- at least four of five samples must be clash-free and structurally parseable.

### G1 — global fold

- global C-alpha RMSD to the reference envelope.
- canonical scTM and pLDDT for compatibility with other RF runs.
- design-level statistics are median and q80 across the five holo samples.

Default floor: median global C-alpha RMSD no greater than 1.20 A to at least one functional holo
reference. The q80 value is reported and used to identify unstable predictions.

### G2 — invariant-19 structure

After global alignment, calculate:

- aggregate symmetry-corrected invariant-19 sidechain RMSD.
- catalytic-4 sidechain RMSD for Y14/D18/R65/H98.
- worst invariant-anchor sidechain RMSD and its position.
- invariant-19 C-alpha pocket RMSD.

Default floors, evaluated against the better functional reference:

- invariant-19 aggregate sidechain RMSD <= 1.00 A.
- catalytic-4 aggregate sidechain RMSD <= 1.00 A.
- worst invariant-anchor sidechain RMSD <= 2.00 A.

Chemically equivalent terminal atoms must be symmetry-corrected for
ARG/ASP/GLU/LEU/PHE/TYR/VAL.

### G3 — free-3 identity-aware functional packing

For positions 60, 96, and 110, report in every holo sample:

- residue-ligand minimum heavy-atom distance.
- number of residue-ligand contacts at 4.0 A and 4.5 A.
- sidechain heavy-atom centroid to ligand-center distance.
- sidechain SASA with ligand and after removing ligand from the same coordinates.
- ligand-induced sidechain burial: `SASA_without_ligand - SASA_with_ligand`.
- local protein-ligand van der Waals overlap and severe-clash count.
- contact network with invariant pocket residues.
- identity-matched sidechain RMSD when a matching parent or LuxSit-i reference exists.

Site-specific interpretation:

- **Position 60:** second-shell/genotype site. Gate on local clash and gross network disruption;
  direct DTZ contact is diagnostic only.
- **Positions 96 and 110:** pocket-shape sites. Require clash-free packing and preservation of a
  parent/LuxSit-i-compatible DTZ contact envelope.
- **Alternative identities:** no reference sidechain RMSD. They remain admissible if packing,
  contact-network, and holo-functional gates pass.

Pocket-shell free volume and ligand-surface contact coverage are reported as diagnostics. They are
not hard gates in this protocol because grid spacing, surface sampling, and predicted rotamers can
move them materially.

### G4 — DTZ pose and catalytic geometry

After invariant-19 pocket alignment, calculate:

- DTZ center-of-mass displacement.
- DTZ heterocycle-core RMSD.
- full DTZ heavy-atom RMSD using consistent atom names; report it separately because phenyl-ring
  orientation is more variable than the functional core.
- Y14 OH - H98 ND1.
- H98 NE2 - DTZ O1.
- R65 CZ - DTZ N1 and nearest R65 terminal-N - DTZ N1.
- nearest D18 sidechain-O - R65 NE and nearest D18 sidechain-O - R65 backbone N.
- all-residue and pocket-residue ligand contact fingerprints.

Default floors against the parent/LuxSit-i functional reference envelope:

- six-distance geometry MAE <= 0.50 A.
- maximum single-distance deviation <= 1.00 A.
- DTZ center-of-mass displacement <= 1.00 A.
- DTZ heterocycle-core RMSD <= 1.00 A.
- full DTZ heavy-atom RMSD <= 2.50 A.

Ligand pose alone is never sufficient: a candidate must also pass G0-G3.

### G5 — five-sample consensus

Do not select sample 0 or the best-ranked sample alone. For each design:

- report median, q80, and range of every continuous metric.
- require at least four of five samples to pass integrity and clash checks.
- gate continuous structure/geometry metrics on the median; q80 instability is a manual-review
  flag when it exceeds the corresponding absolute floor.
- confidence floors are reference-relative: median ipTM must be at least
  `max(0.85, min(WT_parent, WT_i) - 0.05)`, and median ligand pLDDT must be at least
  `max(85, min(WT_parent, WT_i) - 5)`.

Use WT summary-confidence metadata when it exists. If the copied reference bundle contains only
coordinates, ligand pLDDT may be recovered from the Protenix DTZ-chain B factors, while the ipTM
floor must come from the validated WT calibration record or an explicit CLI value. Never estimate
a WT-relative floor from the candidate cohort itself. For the first instantiation, the validated
Protenix WT parent/i ipTM values are 0.968/0.965, giving an ipTM floor of 0.915; the DTZ-chain mean
pLDDT values are read directly from the two reference CIFs.

## Apo-reference sensitivity analysis

For each candidate holo sample, repeat global and invariant-19 comparisons against the matched WT
Protenix apo and holo references. Report:

- median metric shift caused by changing only the reference.
- Spearman rank correlation across designs.
- hard-gate concordance.

Interpret reference choice as negligible when rank correlation is at least 0.90, gate concordance
is at least 90%, and median RMSD shift is no greater than 0.15 A. Otherwise use the holo reference
for all final decisions and report the apo-reference bias. This analysis does not estimate induced
fit.

## Structure score and refinement shortlist

Apply gates before ranking. A genotype label never rescues a failure.

Within the gate-pass set, rank by a robust normalized structure score using design-level medians:

| Component | Weight |
|---|---:|
| invariant-19 sidechain RMSD | 0.20 |
| catalytic-4 sidechain RMSD | 0.15 |
| six-distance geometry MAE | 0.20 |
| DTZ heterocycle-core RMSD | 0.10 |
| DTZ COM displacement | 0.10 |
| invariant-19 pocket C-alpha RMSD | 0.10 |
| free-3 packing disruption | 0.10 |
| confidence/ensemble instability | 0.05 |

Normalize each component robustly over the gate-pass set using median and IQR; lower is better.
Keep the raw values beside the score.

Default output for a 200-design single-target run:

- **primary 8:** top eight structure-ranked candidates.
- **reserve 4:** next structure-competitive candidates, allowing genotype-mechanism coverage only
  from candidates within 0.5 robust-score units of the primary cutoff.
- exact `SLV` or clean partial recovery receives no score bonus, but wins a near-tie because it is
  experimentally interpretable and recovers a published high-activity genotype.
- `n_strong_binders` is a final tie-break and refinement-cost annotation, never a structure rescue.
- do not enforce seed or genotype diversity by admitting a materially worse structure.

The shortlist parquet must retain complete sequences, genotype labels, all gate columns, sample
pass counts, structure score, NMP burden, and explicit selection reason.

## Output contract

Write under `<run-dir>/analysis/parent_variant_holo/`:

- `sample_metrics.parquet`: one row per design and Protenix sample.
- `design_metrics.parquet` and `.csv`: five-sample aggregate and gates.
- `genotype_summary.csv`: counts and metric summaries by genotype class.
- `apo_holo_reference_sensitivity.csv`: reference-choice comparison.
- `refinement_shortlist_top12.{parquet,csv,fasta}`.
- `refinement_primary8.{parquet,csv,fasta}` and `refinement_reserve4.*`.
- `analysis_manifest.json`: paths, reference hashes, formulas, floors, counts, and software git SHA.
- `README.md`: concise verdict, caveats, shortlist, and refinement handoff.
