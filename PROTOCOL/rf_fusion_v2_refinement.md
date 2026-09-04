# PROTOCOL: RF Fusion V2 post-generation Head refinement

Protocol ID: `rf-fusion-v2-refinement/1`
Frozen: 2026-08-24

## Scope / when to use

This is the **second stage** of `rf_fusion_v2_test_set_design.md`. It consumes that protocol's
immutable, content-hashed generation surfaces and performs an NMP-free, Head-directed local
search around a bounded set of selected endpoints, under a seed-relative structure gate.

Use it when a closed four-root V2 grid exists and the question is *how much further a local
edit search can move an already-selected design under the same frozen Head*.

It does not cover, and must not silently absorb:

- any change to the RF/Fusion configuration. `D4/K12/r40`, the policy epsilons, the write cap and
  the generation structure floor are inputs here, never tunables;
- external NetMHCIIpan. NMP is absent from this stage by construction, exactly as in generation;
- constrained (hard-anchor) cohorts. Those need an active-site-aware structure gate — see
  `shortlist_and_refine_seed_selection.md`;
- cross-method comparison statistics. Those belong in an analysis record, not here.

## Three-layer record contract

| layer | authority |
|---|---|
| this PROTOCOL | which seeds, which retention, which gate, and how to read the output |
| `rf_fusion_v2_refinement.runner.md` | fixed execution order and `{{slot}}` command templates |
| `PROTOCOL/cases/<tag>.yaml` | the campaign, digests, resources, job IDs, realized counts |

Every scientific parameter must appear in the CASE. A value that exists only inside an sbatch
line is not recorded.

## The four questions the parent protocol requires this one to answer

`rf_fusion_v2_test_set_design.md` § *Refinement extension boundary* requires an explicit answer to
each of the following. They are stated here once and are binding.

### 1. Which generation surface supplies seeds

The **definitive-feasible pool**, not the already-materialized Pareto front. Eligibility is the
conjunction of four independent records, evaluated per endpoint and then deduplicated within
protein by exact `(protein_id, sequence_md5)`:

```
archive.feasibility_level == definitive
complete_endpoints.structure_evaluated AND structure_feasible     (cleared scTM >= 0.70)
terminal_validation.structure_definitive
terminal_validation.immune_passed                                  (frozen Head)
```

Seeding from the pool rather than the front is deliberate: the front is a *reporting* surface of
size 1–15 per protein, and seeding from it would make the refinement budget a function of front
width, which carries no scientific meaning.

Proteins with **zero** definitive-feasible endpoints across all four roots are handled by §3 below.

### 2. The per-protein / per-seed retention rule

**Input, per protein: up to 8 seeds.** Ordering, applied within each protein:

```
1. pareto_layer ascending
     iterated exact first fronts of the two frozen axes
     (Head global_risk, positive_mass_density = sum(max(residue_hotspot,0)) / sequence_length)
2. mean of the two axes' within-protein ranks, ascending
3. sequence_md5 ascending                       (deterministic tie-break)
take the first 8
```

A protein with fewer than 8 eligible endpoints contributes all of them. This is the same law the
parent protocol names for an `up-to-8` reporting panel, reused here as a *search budget*, and it
must be recorded in the CASE with its realized per-protein counts.

**Output, per seed: the complete Pareto front, untruncated.** The evidence-rich default keeps every
structure-admitted candidate and the merger reduces that history to each seed's exact first front
on the same two axes. `--official` is an output-retention switch: after the unchanged search has
finished, the driver writes that exact per-seed front directly instead of the dominated admitted
history. It does not prune the beam or proposal pool. Both modes therefore emit the identical
front; the resolved config records which retention surface was used.

**Product selection, when one representative per protein is required: one per seed.** Take each
seed's lowest `head_global_risk_after` row, tie-broken by refined-sequence md5. Measured on the
0701 cohort at an identical 8-row budget, this beats pooling across seeds:

| rule | rows | seed lineages / protein | within-protein Hamming | retains protein best |
|---|---:|---:|---:|---:|
| 8 per protein, pooled across seeds | 682 | median 2 | 31.0 | 0.92 |
| 1 per seed | 672 | median 8 | 68.0 | 1.00 |

Pooling collapses to ~2 lineages because one seed's front consumes the quota. Selecting by minimum
risk (rather than a two-axis mean rank) makes the protein's globally best row retained by
construction. The cost is explicit and must be stated wherever the panel is used: inside one seed's
front this is a single-axis choice, so `positive_mass_density` is not balanced.

For a requested final panel other than eight, use `--final-candidates-per-protein N`. It is
independent of `--official` and overrides its default `N=8`. Each seed contributes at most its
minimum-global-risk row; the merger applies exact Pareto layers, full-pool two-axis rank sum, then
refined-sequence MD5 across those seed representatives. This preserves lineage breadth while
producing up to N unique sequences. Selection and realized counts
are computed separately within each `seed_group`; diagnostic and product groups never pool.

### 3. The frozen Head-refinement profile and structure gate

**Proposal law.** Residue-local maxima of the Head hotspot landscape, above a fixed threshold and
capped; then all singles; then a small per-position Pareto AA set; then capped full doubles emitted
in position round-robin order.

| knob | frozen value | why this value |
|---|---|---|
| `HEAD_RESIDUE_THRESHOLD` | `0.15` | measured median editable fraction 0.209 of sequence length; the local-maxima filter then keeps ~10% of residues above it |
| `HEAD_MAX_TARGET_POSITIONS` | `12` | safety valve only — binds on 2.6% of seeds; the maxima filter is the real bound |
| `HEAD_AA_PER_POSITION` | `2` | exactly one representative per Head axis; introduces no scalar weight |
| `MAX_PAIRS` | `200` | cap on **full double mutants per parent per round** |
| `BEAM_WIDTH` | `8` | |
| `MAX_ROUNDS` | `12` | non-binding in practice (measured median 4, max 7, 0/654 hit the cap) |
| `PATIENCE` | `2` | the real stopping condition |
| `MAX_PATH_MUTATIONS` | `6` | edit-burden ceiling relative to the seed |
| `REFOLD_CAP` | `12` | dominant cost term: ~70% of wall time |
| `TOPB` | `64` | |

Threshold `0.15` localizes **proposals only**. `positive_mass` is defined from 0 regardless, so the
objective still counts every positive residue; `head_objective_dominates` requires both axes to not
worsen, which is what prevents mass being pushed below the threshold to game the score. This
decoupling is safe but must be stated in the CASE.

**Stopping has no minimum-improvement epsilon.** Patience compares the beam for exact equality, and
Pareto domination uses bare `<`. This differs from generation, which stops on
`epsilon_R = 0.005` raw logit. Any arbitrarily small improvement therefore resets patience; the
observed early convergence is an empirical property, not a guarantee.

**Structure gate: `legacy`, seed-relative, `SCTM_EPS = 0.05`.** A child must satisfy
`scTM >= seed.scTM - 0.05`. `CONSTRAINT_MANIFEST=none`.

No absolute floor is applied, for two independent reasons. First, the diagnostic seed groups of §3
start *below* the generation floor of 0.70, so an absolute floor rejects them by construction.
Second, absolute scTM is not a property of the design alone: on the DPLM-native baseline under an
identical protocol, 20/100 high-risk proteins and 4.1% of the full test set fold below 0.5. An
absolute cutoff rejects proteins, not designs. Where an absolute structural statement is needed,
use `delta_scTM` against that protein's own same-protocol baseline median.

**Immune objective: the Head the GENERATION used, byte-identical.** The CASE records its path and
SHA-256, and the runner asserts the digest before submitting. Refining against a different
checkpoint — a CV fold, another allele, or a full-data run's leaky `best.pt` — changes the
objective mid-pipeline and invalidates every downstream comparison.

### 4. How refined outputs map back to source provenance

Every seed row carries `protein_id`, `sequence_md5`, `seed_group`, `selection_rule_id`,
`source_campaign`, `source_depth`, and its source cell (root). Refined rows carry
`sequence_original`, which joins back to the seed table by exact sequence.

**Do not join on `design_idx`.** Sharding is at seed level and each shard assigns its own
per-protein sequential index starting at 0, so `(protein_id, design_idx)` collides across shards.
The merger's `design_uid` disambiguates merged tables; `sequence_original` is the key back to
provenance.

## Seed groups and their status

Three groups, always distinguishable by the `seed_group` column. **They are not poolable.**

| `seed_group` | status | content |
|---|---|---|
| `v2_head_pareto_top8` | **product** | up to 8 per protein from the definitive-feasible pool, by the §2 law |
| `structure_failed_best_sctm` | **diagnostic** | for a protein with zero feasible endpoints: its highest-scTM endpoint anyway |
| `wild_type_seed` | **diagnostic** | for that same protein: its wild type from the cohort table |

A protein with no feasible endpoint gets **both** diagnostic seeds. The reason is that the two
answer different questions and neither answers it alone: a below-floor design's low Head risk
describes a molecule that does not fold, so optimizing it further optimizes a fiction; the wild
type is a real, foldable starting point whose risk is real. Refining both makes the comparison
explicit rather than assumed. Measured on 0701, the WT-seeded arm moved risk `+2.06 -> -9.03` in 6
mutations with `delta_scTM +0.018`, i.e. into the same band as the product group.

Filter on `seed_group` before reporting any product-level number. Diagnostic rows must never be
pooled into a product count, a success rate, or a wet-lab shortlist.

## Technical exclusions

Only these exclusions are permitted before refinement, and each must be counted in the CASE:

- a protein whose reference structure does not resolve under `PDB_ROOT`
  (`resolve_structure_path` raises, which kills the whole shard rather than skipping one seed);
- a duplicate `(protein_id, sequence_md5)`;
- a missing or non-AA20 sequence.

Never exclude a seed on `scTM`, Head risk, pLDDT, or edit burden. Those are the measurements.

A protein excluded for an unresolvable reference is a **data-pipeline defect, not a result**. When
the reference is restored — including a `.cif` where the loader expected a `.pdb` — resubmit that
protein from the *same frozen selection table*, verify the loader end-to-end first
(`resolve_structure_path` then `prepare_reference_context`, checking that the parsed length matches
the test-set `sequence_length`), and record the restoration in the CASE. Do not re-select.

## Reading the output

`merge_refine_shards.py` in Head mode emits `master.parquet` (every structure-admitted candidate)
and `head_pareto_per_seed.{parquet,fasta}` (each seed's exact first front,
`selection_status=head_refinement_pareto`). NMP tables and count-based products are absent by
construction.

Report per `seed_group`, and report at minimum:

- coverage: seeds in, seeds with output, refined rows;
- `diverged` count — seeds where no candidate was ever admitted, whose emitted row is the
  unmodified seed. This is a real outcome, not a failure;
- front width per seed;
- best-per-seed delta on both axes;
- rounds actually used, against `MAX_ROUNDS`;
- `delta_scTM` against the protein's own same-protocol baseline, never absolute scTM alone.

**ESMFold2 is not deterministic at these settings.** Forty designs folded twice with identical
parameters and reference backbone gave `|dscTM|` median 0.0024, max 0.121, 0/40 exact, with random
sign. Compare distributions, not individual designs. A per-design scTM difference below ~0.05
against an arm scored in a different process is not evidence. This also bounds interpretation of
the refinement's own structural cost: a median `delta_scTM` of the same order as the fold noise
means *not distinguishable*, not *slightly degraded*.

## Cost model

Measured, 0701, 672 seeds: 8.1 GPU-hours, 34 minutes wall on 16 concurrent GPUs, per-seed median
0.7 minutes. Head scoring is ~1,080 sequences per round across the beam; refold is ~70% of round
time, so `REFOLD_CAP` — not Head parallelism — is the lever if cost must come down.

Scale linearly in seed count for planning, and size the walltime on the **slowest** shard. On
Della, an ailab request of `--time <= 01:00:00` is demoted to `gpu-test`
(`MaxJobsPU=3`, `MaxSubmitPU=25`), which serializes a 16-shard array; keep walltime above one hour.
