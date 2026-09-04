# PROTOCOL: IF benchmark test-set construction

Protocol ID: `if-benchmark-test-set/3`
Updated: 2026-09-02
Release state: **one working cohort built (DRB1\*15:01, 3,015 rows), two legacy cohorts retrofitted
to the C2 selection-unit rule (DRB1\*07:01, DRB1\*04:01); NO cohort has passed `release_validate`,
so none is a content-bound release.** Working cohorts may be used for experiments and are what the
current baselines and derived sets are built on; only a release may move a canonical alias or be
cited by digest. See **Realized builds**.

## Authority and scope

This is the operative specification for building or extending the natural-protein benchmark used
by inverse-folding methods. It supersedes `PLAN_DATA_SEL.md` as a decision authority. The PLAN is
historical design context; `LOG.md` records what was implemented or materialized; reproducible
measurements live in `Results/Analysis/`; neither can silently change this protocol.

The final product is a versioned, content-bound IF-ready cohort plus its cleaned `PDB_ROOT`, source
tables, selection ledgers, leakage tables, score annotations, and release manifest. A downstream RF
CASE cites the released cohort ID and digest; it does not redefine cohort membership.

This protocol does not cover:

- the uricase translational case study under `if_test_set/uricases/`;
- one-off therapeutic or binder targets, which use `new_target_data_pipeline.md`;
- RF generation, which uses `rf_fusion_v2_test_set_design.md`;
- design/refinement selection after generation.

`fast`, `pilot`, and `highrisk` are derived products. They may be built only from a released
cohort and must bind that release digest.

## Dataset entities

The protocol distinguishes five objects that the historical pipeline conflated:

| object | identity | role |
|---|---|---|
| Tier 2 selection unit | exact normalized entity-sequence group; ID is `t2-seq:<full-sequence-SHA256>` | receives one opportunity in overlap filtering and immune-density sampling |
| source entity candidate | one RCSB polymer entity in that exact-sequence group, e.g. `5K2J_1` | retains an independently deposited set of candidate structures |
| Tier 1 source unit | allele + UniProt accession; ID is `t1:<canonical-allele-tag>:<uniprot>` | receives one Tier 1 opportunity and carries allele-native experimental labels; SIFTS ranges are structure candidates |
| structure instance | one auth/asym chain mapped to a source entity or SIFTS range, e.g. `5K2J_A` | supplies coordinates after its selection unit survives the coarse screen |
| IF-ready row | the cleaned, DPLM-readable sequence/structure pair | exact substrate used by generation and every final WT readout |

Multiple crystallographic copies of one polymer entity are not multiple proteins. Distinct entities
with the same normalized sequence also receive only one Tier 2 sampling opportunity, but all of
their structures remain fallback candidates until materialization. A chain ID is never a selection
unit, and changing a chain suffix does not create a distinct benchmark example.

The primary benchmark and the Tier 1 diagnostic stratum are also distinct:

- `primary_generalization`: every row is isolated from CATH training under C4 and may enter
  headline inverse-folding generalization results;
- `tier1_overlap_diagnostic`: an allele-native, peptide-verified experimental Tier 1 row that
  overlaps CATH training. It may be generated and reported separately but never enters the primary
  aggregate or its nominal sample size.

## Core principles

1. **Tier 2 selection is exact-sequence-group-level with entity provenance.** Deposited chain
   multiplicity, repeat structures, and identical entities must not reweight the immune-density
   marginal.
2. **NetMHCIIpan is the only Tier 2 immune membership signal.** Tier 1 is the explicitly separate
   experimental-label stratum defined in C8. The epitope Head is the guidance model RF optimizes;
   its scores are computed only after membership is frozen and cannot prefilter, gate, rank, bin,
   or backfill either tier.
3. **Structure supplies eligibility, not a performance preference.** X-ray availability,
   resolution, readable backbone atoms, chain/entity mapping, and coverage are allowed eligibility
   gates. Fold tolerance, burial, pLDDT, scTM, CATH topology, or any structure-quality prediction is
   forbidden as a sampling or ranking signal.
4. **The final IF-ready sequence is the scoring authority.** A biological/entity-sequence screen
   may reduce cost, but CATH isolation, full NMP density, Head score, and Head-pool homology are all
   recomputed on the exact bytes shipped to generation.
5. **A flag records a measurement, not a policy exception.** `cath_overlap_flag=False` means a
   persisted search found no disqualifying hit. It may never mean “this tier was exempt” or “the
   source pool was believed to be clean.”
6. **Tier 1 labels must reach the final coordinate frame.** UniProt-absolute spans alone are not a
   consumable label. Every retained span must map UniProt -> entity/chain -> IF-ready indices and
   reproduce its peptide exactly.
7. **Release is versioned and atomic.** A candidate build never overwrites a canonical path in
   place. Only a fully passing release can move the canonical pointer.

## Frozen contracts

### C1 — source snapshot

The Tier 2 source is a persisted RCSB polymer-entity query snapshot, not a bare list reconstructed
from a changing web endpoint. The release stores:

- complete query JSON, query time, API/schema identity, and response digest;
- `rcsb_entity_id`, `pdb_id`, numeric `entity_id`, entity sequence, experimental method,
  resolution, and every mapped auth/asym chain;
- the source structure revision/content digest.

Entity eligibility is X-ray, resolution <= 2.5 A, canonical biological length 100-500, and an
AA20 sequence. There is no AFDB fallback for Tier 2: replacing an experimental entity with a
predicted UniProt structure changes the population and requires a different protocol.

### C2 — exact-sequence and entity uniqueness

Before any immune screen, normalize a sequence by removing whitespace, uppercasing it, and requiring
every residue to be in AA20. Then:

1. keep one source row per `rcsb_entity_id`;
2. group exact normalized entity sequences and assign `t2-seq:<full-sequence-SHA256>` as
   `selection_unit_id`;
3. retain every entity in the group as a structure candidate, ordered by better experimental
   resolution and then lexical `rcsb_entity_id`;
4. persist every entity -> sequence-group edge and the complete ordered fallback list.

Immune scoring and sampling occur once on the shared sequence, not once per candidate entity. C2
does not discard lower-ranked entities: C6 may fall through to them if a better-resolution entity
has no admissible IF-ready chain.

Within Tier 2, exact-deduplicate the final `sequence` again after C6. Representative choice uses
higher valid-backbone coverage, then better resolution, then lexical entity and chain IDs. Every
collapse remains in the ledger. Cross-tier collisions are resolved later by C8, whose Tier 1
precedence overrides this within-Tier-2 tie-break. The released primary table has unique
`selection_unit_id` and unique final sequence digest.

After primary assembly, compute family clusters over `primary_generalization` only with the
release-pinned MMseqs2 binary and the following explicit command contract:

```text
mmseqs easy-cluster final.fasta clu tmp \
  --min-seq-id 0.3 --cov-mode 0 -c 0.8 --cluster-mode 0 --threads 1
```

The input FASTA is sorted by `protein_id`; the manifest records the MMseqs version and all effective
defaults. The stable `family_cluster_id` is the SHA-256 of the sorted member `sequence_sha256`
values, not MMseqs' representative name. Homologs may remain, but downstream cohort-wide
uncertainty resamples whole clusters with replacement and uses shared draws for paired methods.
The primary independent-unit count is the number of resulting clusters, never the raw row count.
Diagnostic rows may be clustered separately or mapped to the nearest primary family after this
step, but they never enter the clustering input that defines a primary cluster.

### C3 — Tier 2 immune signal

`coverage_fraction` is the fraction of final IF-ready residues covered by at least one
NetMHCIIpan window with `%Rank_EL < 2`. Coverage is the union of half-open residue intervals across
peptide lengths 12-25, divided by final `sequence_length`. The release binds the exact
NetMHCIIpan binary/data identity, allele spelling, command parameters, and input sequence digest.

Every full-score row must have `nmp_status=complete`. Timeout/partial rows fail eligibility; zero
strong windows are valid and give `coverage_fraction=0`.

### C4 — CATH train isolation

Use the actual CATH 4.3 train split from `chain_set.jsonl` plus
`chain_set_splits.json`. Run MMseqs2 with:

- `--min-seq-id 0.3`
- `--cov-mode 0`
- `-c 0.8`

The implementation's frozen decision is **strict identity > 0.30 excludes**; exactly 0.30 passes.
Store every query, best target, identity, aligned coverage, decision, MMseqs version, parameters,
reference digest, and query-sequence digest.

An entity-sequence prefilter may be reused within the same content-bound build to save compute, but
the release gate always reruns C4 on every final IF-ready sequence. Tier 2 or Tier 1 rows that hit
CATH are excluded from `primary_generalization`. Allele-native Tier 1 hits may be retained only in
`tier1_overlap_diagnostic` with `cath_overlap_flag=True`.

### C5 — cost-controlled Tier 2 screen

The coarse stage scores length-15 NMP once per C2 sequence group. Its
`selection_source_coverage_fraction_15` is the union of 15-mer residue intervals with
`%Rank_EL < 2`, divided by the full entity-sequence length. It exists only to bound the full
multi-length cost; it never becomes a final WT readout.

Sort rows by `selection_unit_id`, form 20 equal-width bins over the realized minimum and maximum of
the Tier 2 source pool that passed the entity-sequence C4 prefilter, and allocate the declared
coarse target with equal bin weights, `min_per_bin=0`, no replacement, deterministic seed 42, and
the C7 allocation/draw rule with all $w_i=1$. A degenerate axis forms one populated bin. Persist
the complete scored pool, bin edges, availability, allocation, selected IDs, RNG/library identity,
and the recomputable ordered input.

Before scoring begins, the build manifest freezes the coarse target and an optional monotone
expansion ladder. If C6/final-C4 attrition leaves insufficient capacity for the requested final
Tier 2 target, the attempt fails or advances to the next preregistered ladder rung on the same
source snapshot. It may not opportunistically backfill individual rows after observing structure,
CATH, or full-NMP outcomes.

### C6 — entity-to-chain materialization

For each coarse-surviving sequence group, traverse its C2-ordered entities and enumerate only
chains mapped to each entity. Run the cleaning logic on every mapped chain. Rank viable
representatives only by:

1. higher fraction of entity residues represented exactly by complete N/CA/C/O backbone
   coordinates;
2. better experimental resolution;
3. lexical `rcsb_entity_id`, auth chain ID, and asym chain ID.

No immune or fold-prediction value enters this choice. The selected chain is recorded together with
all rejected chain attempts.

For Tier 2, `if_sequence_coverage` has one definition: the number of distinct entity-sequence
residues represented by an order-preserving, amino-acid-identical final residue with complete
N/CA/C/O coordinates, divided by full entity-sequence length. Missing source residues, including
internal unresolved gaps, are allowed subject to the coverage floor. Unmapped final insertions,
expression tags, and amino-acid substitutions are forbidden in the primary benchmark. The only
default coordinate-residue normalization is `MSE -> M`; any broader alias allowlist is a protocol
version change, not a release-time choice.

For Tier 1 the denominator is the frozen SIFTS-mapped UniProt range, and every retained final
residue must map order-preservingly and amino-acid-identically to that UniProt sequence under the
same residue-normalization rule. Each UniProt is one source unit: its viable SIFTS structure
candidates are ranked by higher exact UniProt coverage, better resolution, then lexical PDB/auth/
asym chain IDs, and only the winner can become a row.

The IF-ready gate requires:

- canonical AA20 final sequence;
- `if_sequence_coverage >= 0.8` under the denominator above;
- a complete per-residue mapping ledger with zero unmapped final insertions and zero amino-acid
  substitutions after the frozen residue normalization;
- exact `len(sequence) == sequence_length`;
- DPLM `load_coords()` returns that exact sequence and length;
- the cleaned structure exists under the released `PDB_ROOT` and is content-hashed.

The 100-aa source floor applies to the entity sequence; the final IF-ready sequence can therefore
be shorter. Realized final length range belongs in the release manifest.

### C7 — final Tier 2 marginal

After C6, repeat exact-sequence deduplication, run final C4, and compute full 12-25 NMP on the final
IF-ready sequence. Resolve Tier 1 collisions as specified in C8 before defining the Tier 2 eligible
pool. Gaussian sampling then operates only on the remaining CATH-clean, final-sequence Tier 2 rows.

Frozen Gaussian law:

- axis: final `coverage_fraction`;
- 20 equal-width bins over the eligible pool's realized range;
- center `mu`: eligible-pool median;
- minimum per populated bin: `min(40, available)`;
- deterministic seed: 42;
- no replacement.

Gaussian weights at bin centers $c_i$ are

$$
w_i = \exp\left[-\frac{1}{2}\left(\frac{c_i-\mu}{\sigma}\right)^2\right],
$$

where $d=\max(\mu-\min_i c_i,\max_i c_i-\mu)$ and

$$
\sigma = \frac{d}{\sqrt{2\log 3}}.
$$

Thus the theoretical weight at $\mu$ is three times that at the farthest bin center; if $d=0$,
weights are uniform. Allocation starts at the per-bin floor and requires both
`sum(floors) <= target <= sum(available)`; otherwise the build fails. Each subsequent unit goes to
the non-full bin with smallest `allocated_i / w_i`, with ties resolved by lower bin index. A single
NumPy `default_rng(42)` stream then visits bins 0 through 19 and draws without replacement from
lexically sorted `selection_unit_id` values; the NumPy version and bit-generator identity are
recorded.

Target size is release-specific. Every available/allocated/selected bin count and every cap is
persisted. The eligible table is sorted by `selection_unit_id` before seeded within-bin draws.
Final membership is reproducible from that table and histogram.

### C8 — Tier 1 construction and role

Tier 1 is allele-native. Its candidates come from that allele's epitope-head split IDs, allele-
matched IEDB EL spans, UniProt sequences, and PDBe SIFTS experimental mappings. The release stores
the complete SIFTS request set, response payloads, query time, API/schema identity, and content
digests; a changing `best_structures` endpoint is never queried without this snapshot. Candidate
filters are X-ray and resolution <=2.5 A. Initial length and epitope coverage use one frame: the
frozen SIFTS-mapped UniProt range. Its length is `unp_end - unp_start + 1` and must satisfy
`100 <= length <= 500`. Coverage is the number of distinct UniProt positions in the union of fully
contained, peptide-verified EL spans divided by that range length; it must satisfy
`0.10 <= coverage <= 0.50`, with at least two distinct verified spans. After IF projection, apply
the same inclusive span-count/coverage bounds using the union of verified IF-frame positions over
final `sequence_length`.

**Coordinate revision authority.** PDBe enriched mmCIF supplies residue mapping and entry
identity (`_entry.id`) only. It is **not** the revision authority and must never be required to
be one: production PDBe enriched files carry no `_pdbx_audit_revision_history` loop at all, and
requiring one rejected all 2,864 DRB1\*15:01 Tier 1 attempts and sealed an empty C8. The frozen
revision comes from the C1 RCSB entity snapshot (`entities_all.parquet`, the pre-eligibility
superset), normalized to one date per PDB entry. Consequences, all fail-closed:

- C8 depends on C1. A build cannot map Tier 1 residues before its revision authority is frozen.
- Two different revisions claimed for one entry is a build error, not a tie to break.
- A PDB entry absent from the frozen snapshot yields a typed per-entry failure, never a revision
  guessed from the coordinate file it is supposed to validate. (Measured on the DRB1\*15:01 pool:
  1,647 of 1,648 Tier 1 entries are covered.)
- The RCSB coordinate manifest's observed revision must equal the frozen revision; a mismatch is
  a typed failure for that entry, not a stage crash — one stale entry must not discard the
  hundreds of downloads a multi-hour login stage has already paid for.

**Database disagreements are typed, never fatal.** PDBe and RCSB can disagree about a chain: the
SIFTS residue map's entity, auth chain, author numbering, or insertion code may not match the
deposited coordinates. That is a disagreement between two sources about one chain instance, in the
same class as a SIFTS/UniProt amino-acid mismatch — it is recorded as a typed per-attempt failure
(`coordinate_identity_mismatch`) and the attempt is rejected, all-or-nothing across its chain
copies, because the failure ledger is keyed per attempt. Measured on DRB1\*15:01: **3 of 1,258**
candidates (0.24 %), touching one UniProt source unit. Letting that raise would discard a
completed multi-hour login stage — and the whole build — over three rows.

The distinction that governs both: a **fetch** failure (no bytes, or bytes that fail their content
digest) sets `release_blocked`; a **content** rejection (the bytes arrived and were judged
scientifically unusable) is a typed ledger row and does not. Only the first means the snapshot is
incomplete.

**C8 capacity gate.** C8 writes its full audit trail and then fails, before sealing, if the number
of viable Tier 1 **source units** (allele+UniProt, not candidate rows) is below the release's Tier 1
target. A zero-candidate C8 that seals is the failure mode this gate exists to prevent: every
downstream `afterok` dependency then becomes unsatisfiable while the DAG still reports the stage
complete, and the expensive C6 coordinate pool gets materialized for nothing.

Build and retain the complete ranked candidate pool; do not truncate to N before C4/C6. Then:

1. materialize and map all candidates;
2. rerun C4 on final IF-ready bytes;
3. project all spans into the final IF frame and reapply peptide identity, minimum-span, and
   coverage gates;
4. retain at most one C6-ranked structure winner per UniProt, then rank the surviving CATH-clean
   UniProt pool by final IF-frame peptide-verified epitope coverage descending, verified span count
   descending, verified covered-residue count descending, and lexical UniProt ID; take/backfill to
   the release's Tier 1 target (historically 15). Structure coverage and resolution are eligibility/
   within-UniProt representative signals and never rank distinct UniProt source units;
5. write CATH-overlapping but otherwise valid entries to the separate diagnostic artifact.

The target, filters, rank law, tie-breaks, and source snapshots are frozen before any candidate
materialization result is inspected. If the CATH-clean valid pool cannot fill the target, the build
fails; an overlap-diagnostic row never fills a primary vacancy.

Tier 1 source fields include `source_uniprot_id`, SIFTS range/mapping, raw UniProt-frame spans,
chain-frame spans, IF-ready-frame spans, and mapping/drop counts. All final spans use zero-based
half-open IF-ready coordinates.

Track `source_pool_memberships` separately from `selected_tier_memberships`. If a CATH-clean Tier 1
candidate collides with Tier 2 by entity or final sequence, remove that Tier 2 selection unit before
C7, retain one primary row with Tier 1 metadata, set source memberships to both pools, and set the
selected membership to Tier 1 only. A CATH-overlapping collision produces only one Tier 1
diagnostic row and cannot enter primary through its Tier 2 route. Same-entity/different-crop cases
are entity collisions, so Tier 1 still takes precedence. Never suffix the same structure or
sequence into a second benchmark row.

### C9 — Head annotation, never membership

After final cohort membership is frozen, score every primary and diagnostic final IF-ready sequence
with the allele's fixed-epoch full-data production Head:

| allele | production checkpoint |
|---|---|
| DRB1\*07:01 | `run/epitope_head_fulldata/drb0701/epoch_29.pt` |
| DRB1\*04:01 | `run/epitope_head_fulldata/drb0401/epoch_34.pt` |
| DRB1\*15:01 | `run/epitope_head_fulldata/drb1501/epoch_24.pt` |

`best.pt` from a full-data run is forbidden because its validation rows are inside the training
pool. The release binds the fixed-epoch checkpoint and config digests.

Head-pool homology is measured on final IF-ready bytes against the union of train, validation, and
test proteins used by that full-data Head, with MMseqs `--min-seq-id 0.3 --cov-mode 2 -c 0.8`.
Persist best hit/identity and `head_seen_overlap_flag`; annotate, do not exclude. Head-based claims
are reported by homology stratum, with the overlap-free stratum as the generalization readout.

Head scores and homology live in a versioned annotation table keyed by
`(dataset_release_id, protein_id, sequence_sha256, head_checkpoint_sha256)`. A new Head annotation
does not silently mutate base cohort membership.

### C10 — released schema

The historical 17-column assembled schema and 34-column IF-ready extension are not v3 schemas.
The v3 release must carry at least:

- release/entity identity: `dataset_release_id`, `allele`, `protein_id`, `selection_unit_id`,
  `pdb_id`, `entity_id`, `auth_chain_id`, `source_pool_memberships`,
  `selected_tier_memberships`, and `evaluation_role`;
- final sequence identity: `sequence`, `sequence_length`, `sequence_sha256`;
- source/mapping identity: entity sequence and digest, final-to-source mapping, mapping status,
  IF-ready coverage, selected/rejected chain provenance;
- structure identity: `pdb_path`, structure digest, method, resolution;
- selection evidence: final NMP metrics, `coverage_fraction`, bin, selection reason;
- leakage evidence: CATH decision/best hit/identity and family-cluster ID;
- Tier 1 evidence where applicable: allele, UniProt/SIFTS identity, all three span frames, and
  peptide-verification counts.

`cath_topology` and Tier 3 fields are not required. Extra annotation tables must bind the same
release and sequence digests. A schema-versioned validator checks both row and cross-artifact
contracts.

### C11 — versioned atomic release

Build first under an attempt-specific staging directory and publish only by atomic rename into an
immutable release directory:

```text
if_test_set/builds/<build_id>/<allele_tag>/
  ...same candidate payload and audit files...

if_test_set/releases/<release_id>/<allele_tag>/
  cohort_if_ready.parquet
  tier1_overlap_diagnostic_if_ready.parquet
  pdbs_if_ready/
  annotations/
  audit/
  dataset_manifest.json
```

Each attempt first receives a unique `build_id`. Failed IDs are never reused. After membership and
scientific gates pass, finalization assigns an immutable semantic `dataset_release_id`, serializes
that ID into every release table, reruns the byte-level gates, and then writes the manifest. The
manifest binds output hashes but no output embeds the manifest hash. Its own `manifest_sha256` is
computed from canonical JSON with the `manifest_sha256` field omitted. Downstream identity is the
pair `(dataset_release_id, manifest_sha256)`, eliminating a row/manifest hash cycle.

The audit directory retains the RCSB query/entity table, dedup/collapse ledgers, CATH searches,
coarse and full NMP tables, IF-ready chain-attempt/failure tables, Gaussian histogram, Tier 1
mapping table, Head annotation/homology, and load-coordinates results. The manifest binds every
input/output SHA-256, code revision, tool/model identity, stage count, and parameter.

Only after all release gates pass may an alias under `if_ready/main/` be atomically repointed.
Never append, replace, or partially publish rows at a path currently consumed by another run.

### C12 — derived products and h-maps

Every subset, WT baseline, generation facade, and score cache records the exact dataset release
digest. A historical descendant remains valid for its bound historical release, but it cannot be
reused under a new release identity. Unbound or digest-mismatched default products are rebuilt or
explicitly marked stale before the canonical alias moves.

h-maps are not a release product. Historical `if_ready/h_maps_v2/` files remain read-only for old
Phase C reproduction. Fusion V2 scores the Head directly and does not consume an h-map.

#### C12.1 — high-risk subsets: the selection axis may not be the reported axis

A high-risk subset exists to give a de-immunization method visible headroom. Selecting on an
extreme of a measured quantity and then reporting improvement on that same quantity inflates the
effect by regression to the mean: the selected extreme carries positive measurement/sampling noise
that does not persist. This is not hypothetical here — an earlier high-risk analysis measured a raw
Head improvement of −0.66/−1.37 that fell to a guidance residual of ≈ −0.07…−0.44 once the
selection bias was corrected.

The frozen rule follows from that:

1. **Selection axis is NetMHCIIpan.** NMP is external to the optimization loop, so its noise is
   independent of the Head's. A protein selected because an independent instrument calls it
   immunogenic gives an uninflated Head readout.
2. **Rank by agreement between two independent generators**, not by one. Per protein reduce each
   arm's 8 baseline designs to their median `strong_frac = n_strong_binders / n_windows_scored`,
   convert to within-cohort percentiles, and rank by `min` over arms. A protein is hard only if
   both samplers left it hard. Measured cross-arm agreement is Spearman +0.49/+0.62/+0.56
   (0701/0401/1501 on NMP), so this is a real constraint, not a formality.
3. **The Head enters only as a floor, never as a co-ranking axis.** Drop candidates whose
   min-over-arms Head percentile is below `head_floor_pct` (frozen at **0.20**). Rationale is
   figure hygiene, not selection: a protein the Head already scores as low-risk is a guaranteed
   non-mover on the reported axis. A bottom truncation at 0.20 enriches selection noise about 5×
   less than ranking by Head would (inverse Mills ratio 0.27σ vs 1.40σ at top-20%), and measurement
   confirms it is nearly inert — it replaced 5/7/2 of 250 rows and left the selected NMP median
   unchanged to four decimals.
4. **Floor exclusions are retained, not discarded.** They are written to
   `<output>.head_floor_excluded.parquet`. The NMP-high/Head-low rows are candidate Head failures
   and are the right sample to inspect when characterizing Head limitations.

Producer: `scripts/build_highrisk_demo.py --rank-mode nmp_min_arms --secondary-imm-dir ...
--head-floor-pct 0.20`.

#### C12.2 — dual-allele subsets

`protein_id` is not comparable across cohorts, so the join key is the IF-ready **sequence**.
Cohorts are sampled independently and overlap weakly; a dual set is therefore pool-limited, and the
pool — not the ranking rule — is the binding constraint. Consequences, all measured on the deduped
cohorts:

- shared sequences: 0701×0401 **358**, 0701×1501 195, 0401×1501 171, all three 37;
- cross-allele burden agreement: NMP Spearman **+0.358** / +0.216 / +0.114 and Head +0.220 / +0.257
  / +0.152. The immune landscape is strongly allele-specific, so "hard under both alleles" is only
  ~1.4–1.6× above chance;
- **0701×0401 is the only workable pair**: largest shared pool, the only NMP ρ in the workable
  middle band (too high ⇒ the dual objective is redundant, too low ⇒ no shared structure and no
  statistics), the two strongest Heads (CV Pearson 0.508/0.429 vs 0.354), and the pair that already
  carries the dual cohort, calibration overlays, RF campaign, and scored comparator panel.

Rank by `dual_nmp_min_pct` = min over the four (allele, arm) NMP percentiles. The Head floor is
min over the two **alleles** (arms averaged within an allele), not over all four columns: min of
four percentiles is far stricter than the single-allele min of two and was measured to exclude 52 %
of the shared pool instead of the intended few percent. Emit the whole ranked pool and let the
consumer slice depth; do not bake a top-N into the artifact.

A shared protein carries two **independent** baseline design draws — one made in each allele's
cohort and scored under that allele. Dual hardness is therefore a protein-level property measured
from two unbiased draws, not one design set scored under two alleles. Cross-scoring a single draw
under both alleles is ~3,900 designs (a few CPU shards) and is required before any claim that
compares the two alleles' *designs* rather than their *proteins*.

Producer: `scripts/build_dual_allele_hardset.py`.

## End-to-end build order

1. **Freeze build identity and sources.** Record `build_id`, allele, target sizes, code revision,
   RCSB/SIFTS snapshots, CATH content, NMP identity, full-data Head identity, and random seed before
   scoring. Reserve `dataset_release_id` only during successful finalization under C11.
2. **Build entity pools.** Produce the complete Tier 2 entity table and complete allele-native Tier
   1 ranked pool. Preserve entity-to-chain and UniProt/SIFTS mappings.
3. **Normalize and prefilter.** Apply AA20/length rules and C2 exact grouping. Apply entity-sequence
   C4 as a cost-saving exclusion only to Tier 2; for Tier 1 it is an annotation/cost-routing signal,
   because otherwise-valid hits must remain available to the diagnostic artifact. Persist all
   excluded/collapsed rows.
4. **Run Tier 2 coarse NMP.** Length-15, sequence-group-level, NMP-only; take the uniform
   cost-bounded pool.
5. **Materialize IF-ready representatives.** Enumerate only entity-mapped chains, apply C6, and
   persist every attempt/failure.
6. **Recompute final-sequence evidence.** Final exact deduplication; final CATH C4; full 12-25 NMP;
   and `coverage_fraction`.
7. **Finalize Tier 1.** Map/verify spans, separate CATH-overlap diagnostics, then rank/backfill the
   CATH-clean pool. Resolve Tier 1/Tier 2 collisions as one row.
8. **Sample Tier 2.** Resolve Tier 1 collisions first; then define the Tier 2 eligible pool, apply
   C7 Gaussian sampling, and persist the complete eligible table plus histogram.
9. **Assemble the base cohort.** Merge primary Tier 1 and Tier 2 under the v3 schema, retain the
   separate diagnostic artifact, validate unique entity/final sequence and every structure/mapping
   identity, and compute C2 primary family clusters without diagnostic rows.
10. **Annotate Head evidence.** Score the frozen primary and diagnostic artifacts and run C9
    homology; membership cannot change at this stage.
11. **Run release validation.** Execute every gate below, write `dataset_manifest.json`, then
    atomically publish the release pointer.

## Release gates

A cohort is released only if every condition is machine-verified:

1. source eligibility recomputes exactly from the RCSB/SIFTS snapshots; all source/query/tool/model
   files have content digests and complete funnel counts;
2. exact sequence groups, ordered entity fallbacks, the selected viable entity/chain, Tier 1
   collisions, and final deduplication all recompute from their ledgers; one row remains per
   selection unit/entity collision and per final exact sequence;
3. zero `cath_overlap_flag=True` rows in `primary_generalization` on final IF-ready bytes;
4. every final row passes AA20, coverage, mapping, structure existence/hash, and exact
   `load_coords()` sequence/length checks;
5. every Tier 1 primary and diagnostic row is allele-native, passes C6-equivalent mapping and
   `load_coords()` gates, and has only peptide-verified IF-frame spans; every diagnostic row has
   `evaluation_role=tier1_overlap_diagnostic` and a persisted `cath_overlap_flag=True`;
6. every Tier 2 row has complete final-sequence 12-25 NMP evidence and non-null
   `coverage_fraction`, resolution, entity ID, selection bin, and selection reason;
7. primary Tier 1 and Tier 2 row counts exactly equal their preregistered release targets after
   collision resolution; a short but otherwise valid cohort cannot release;
8. both the C5 coarse allocation/expansion decision and C7 Gaussian allocation/membership
   recompute exactly from their released ordered eligible tables;
9. all primary and diagnostic Head rows share one checkpoint/config identity and exact release-row
   sequence digests;
10. CATH and Head homology flags come from persisted searches, never constants;
11. the v3 row/cross-artifact validator passes, IDs and file paths are unique/resolvable, and all
    sidecar counts/digests equal the released artifacts;
12. every descendant binds its true source release digest; default consumers are explicitly moved
    to the new canonical alias or explicitly pinned to the prior release, and only unbound/mismatched
    descendants are marked stale;
13. publication is atomic and the prior release remains recoverable by release ID.

## Realized builds

Three cohorts exist. None has passed `release_validate`; all three are **working cohorts**.

| allele | rows | composition | basis | producer |
|---|---|---|---|---|
| DRB1\*15:01 | 3,015 | 15 Tier 1 + 3,000 Tier 2 | v3 stages, assembled from sealed stage artifacts | `build_if_benchmark_v3.py` (C1–C8) + direct assembly |
| DRB1\*07:01 | 2,429 → 2,015 | 15 Tier 1 + Tier 2 | legacy cohort, C2-retrofitted then C7-resampled | completed one-time migration; audited artifacts retained |
| DRB1\*04:01 | 2,435 → 2,029 | 29 Tier 1 + Tier 2 | same | same |

### DRB1\*15:01 — first v3 cohort

Built from `ifbench-v3-drb1501-r5000-20260831T015704Z`: 186 C7 NMP query shards plus the 15 sealed
Tier 1 rows, filtered to `status == complete` and `~cath_overlap_flag`, then C7 Gaussian-sampled to
3,000 Tier 2. Verified: unique `protein_id`, **0 CATH-overlapping rows**, `if_sequence_coverage`
min 0.8, lengths 81–497, Tier 2 `coverage_fraction` median 0.38 spanning 0→1, 0 missing structures,
Head scores complete under `drb1501/epoch_24.pt`.

It was assembled **directly from the stage artifacts** rather than through the sealed
`release_validate` chain. That is why it is a working cohort: the numbers are correct and
reproducible from the stages, but no release digest binds them. Completing the chain is a separate
task and is required before any external release or before a downstream CASE may cite a digest.

Four defects in the v3 implementation were found and fixed by executing it, and any re-execution
depends on the fixes:

1. C8 required a coordinate revision from the PDBe enriched mmCIF, which production PDBe files do
   not carry — this rejected all 2,864 Tier 1 attempts and sealed an empty stage. The revision
   authority is the C1 RCSB snapshot (see C8).
2. A PDBe/RCSB coordinate-identity disagreement raised, so 3 of 1,258 candidates (0.24 %) could
   discard a multi-hour stage. It is now a typed per-attempt failure.
3. `http.client.IncompleteRead` is not an `OSError` and escaped the network retry, killing a
   9,182-file download at 1,970. All five retry sites now catch `HTTPException`.
4. Shard partitioning used a ceiling divide and produced empty trailing shards, which fail-closed
   producers reject. Partitioning is now even.

### DRB1\*07:01 and DRB1\*04:01 — C2 retrofit of legacy cohorts

Both were assembled at CHAIN level, so **450/2,879 (15.6 %)** and **408/2,843 (14.4 %)** of their
rows were repeat copies of a sequence already present — about half of them several chains of one
PDB entry, worst multiplicities 15× and 26×. This is exactly what C2 exists to prevent, and the
redundancy is not neutral: it skews short and immunogenic. Collapsing moves the **WT** marginal by
NMP `n_strong` median 52→49 and Head `global_risk` −0.627→−0.788 (0701), 65→59 and −3.116→−3.626
(0401).

The retrofit is membership-only. Every surviving row is an unchanged row of the source cohort, so
existing generation and evaluation are reused by **subsetting**, never by re-running — measured
coverage of the collapsed sets is 2,428/2,429 and 2,421/2,435, the gaps being `6TN1_AAA` and the 14
native Tier 1 rows appended after those baselines ran.

Two artifacts per allele, and the distinction matters downstream:

- `*.c2collapse.parquet` — C2 collapse only (2,429 / 2,435). The **candidate basis for derived
  products**, because C7's Gaussian serves the primary benchmark's density marginal and imposing it
  on a derived set only shrinks an already-scarce pool: measured, it cut the 0701×0401 dual pool
  from 358 to 253 shared sequences for no corresponding benefit.
- `*.v3reorg.parquet` — C2 collapse + C7 Gaussian redraw at target 2,000 (2,015 / 2,029). The
  **primary-benchmark basis**.

`coverage_fraction`, the C7 axis, did not exist for either legacy cohort and was computed fresh on
all 5,722 WT sequences under the C3 definition. Both deduped pools are a broad **plateau**, not a
peak (median 0.377 / 0.435), so the Gaussian mostly trims the centre while the `min_per_bin` floor
pins the thin right tail; reshaping at target 2,000 is real but modest, and the design-level
baseline medians move by ≤0.08 Head and ≤0.003 scTM. What the retrofit actually fixes is the
**independent-unit count** and the correspondingly overstated paired-comparison precision, not the
reported design medians.

### Derived products currently built

All live under `if_ready/highrisk/`. Every one is reproducible from the commands below plus the
arm directories in `run/benchmark/_hardset_arms_20260901/`, which pair each generator arm's
full-data Head table with its NetMHCIIpan table and carry a `PROVENANCE.txt`. That pairing exists
because for ProteinMPNN on 0701/0401 the two tables come from **different runs**: the Head is the
full-data checkpoint (`epoch_29` / `epoch_34`), while the NMP is reused from the 2026-06-04 legacy
`if_phase_c` run. Reusing it is sound — NetMHCIIpan is head-independent and its keys are a superset
of the full-data Head's — but reading Head and NMP out of one legacy directory would silently pick
up a superseded checkpoint, which is what the staging directory prevents.

#### Single-allele high-risk sets

`highrisk_nmp2arm_v2_<allele>.parquet`, **250 proteins each**, built under C12.1: rank by
`min` over the two arms' NMP percentile, Head floor 0.20, `--min-coverage 0.8`, primary arm =
DPLM-native Gumbel, secondary = ProteinMPNN.

```
python scripts/build_highrisk_demo.py \
  --canonical-if-ready <cohort>.parquet \
  --primary-imm-dir   run/benchmark/_hardset_arms_20260901/gumbel_<tag> \
  --secondary-imm-dir run/benchmark/_hardset_arms_20260901/pmpnn_<tag> \
  --rank-mode nmp_min_arms --head-floor-pct 0.20 \
  --n-target 250 --min-coverage 0.8 \
  --output if_ready/highrisk/highrisk_nmp2arm_v2_<allele>.parquet
```

Realized. All medians are on the **primary arm** (DPLM-native Gumbel) so the selected and cohort
columns are the same measurement; the cohort column is that arm's median over the whole basis
cohort:

| allele | basis | pool after floor | NMP `strong_frac` cohort → set | Head `global_risk` cohort → set | Head pct range |
|---|---|---|---|---|---|
| 0701 | `.v3reorg` 2,015 | 1,375 | 0.0116 → 0.0270 (**2.32×**) | −8.118 → −0.069 (**Δ +8.05**) | [0.25, 1.00] |
| 0401 | `.v3reorg` 2,029 | 1,382 | 0.0157 → 0.0441 (**2.80×**) | −8.825 → −1.831 (**Δ +6.99**) | [0.20, 1.00] |
| 1501 | `main` 3,015 | 2,104 | 0.0145 → 0.0426 (**2.95×**) | −7.795 → −1.925 (**Δ +5.87**) | [0.21, 1.00] |

The Head column is the payoff: the set was chosen without the Head ranking it, yet the Head reads
6–8 units above the cohort median on it. That gap is the headroom a de-immunization method has to
work with, and because NMP did the selecting it is not inflated by regression on the Head axis.

Sizing evidence. The two arms' top-X% NMP sets intersect as follows, and **all three alleles cross
200 at top-20 %**, which is why `n-target 250` (≈ top-17–22 %) is the frozen size rather than a
tighter, more striking but under-powered tail:

| top-X% | 0701 | 0401 | 1501 |
|---|---|---|---|
| 10 % | 79 | 107 | 110 |
| 15 % | 147 | 199 | 181 |
| **20 %** | **219** | **282** | **281** |
| 25 % | 299 | 371 | 396 |
| 30 % | 393 | 464 | 525 |

The tails agree far above chance at the extreme (≈6× at top-2 %) and decay to ≈2× by top-20 %, so
the intersection is a real constraint at every depth used here.

The three sets share almost nothing — 0701∩0401 = 9, 0701∩1501 = 4, 0401∩1501 = 7 by sequence, from
a 730-protein union. That is a direct consequence of the allele-specific landscape in C12.2 and
means the three panels are three independent demonstrations, not three views of one set.

#### Dual-allele high-risk set

`dual_hardset_0701x0401_v1.parquet`, **236 ranked proteins** on the `.c2collapse` basis, built
under C12.2 (358 shared sequences, Head floor 0.20 over the two alleles removes 122).

```
python scripts/build_dual_allele_hardset.py \
  --a-label HLA-DRB1_07_01 --a-cohort <0701>.c2collapse.parquet \
  --a-gumbel-imm-dir .../gumbel_0701 --a-pmpnn-imm-dir .../pmpnn_0701 \
  --b-label HLA-DRB1_04_01 --b-cohort <0401>.c2collapse.parquet \
  --b-gumbel-imm-dir .../gumbel_0401 --b-pmpnn-imm-dir .../pmpnn_0401 \
  --head-floor-pct 0.20 --min-coverage 0.8 \
  --output if_ready/highrisk/dual_hardset_0701x0401_v1.parquet
```

Realized depth ladder (`dual_nmp_min_pct` threshold; NMP medians as 0701 gum/pm, 0401 gum/pm):

| depth | threshold | 0701 NMP | 0401 NMP | 0701 Head med | 0401 Head med |
|---|---|---|---|---|---|
| top-50 | 0.564 | 0.0244 / 0.0289 | 0.0347 / 0.0423 | −3.14 | −2.49 |
| top-100 | 0.380 | 0.0194 / 0.0233 | 0.0279 / 0.0325 | −4.19 | −4.74 |
| top-150 | 0.237 | 0.0180 / 0.0204 | 0.0249 / 0.0298 | −4.48 | −5.30 |

Depth is the consumer's choice; the artifact ships the full ranked pool with `dual_hardrank`.

#### Superseded

`highrisk_nod_v1_*` and `highrisk_pmpnn_demo_v1_*` predate both the C2 retrofit and C12.1. They
rank on a single primary arm using `min(nmp_pct, head_pct)` — i.e. the Head participates in
selection — and they sit on chain-level cohorts, so they carry duplicate sequences (0401 **15/100**,
0701 11/100). Retained for reproducing completed runs; not a basis for new figures.

Head↔NMP agreement, measured per allele on both arms, is stable and worth reporting rather than
assuming: gumbel/pmpnn Spearman **+0.698/+0.699** (0701), **+0.621/+0.645** (0401),
**+0.606/+0.595** (1501). The two arms agree with each other far less (+0.43/+0.44/+0.50 on Head),
so ρ(Head, NMP) is a property of the allele/model pair rather than of a sampler. The ordering
tracks Head CV quality (Pearson 0.508 / 0.429 / 0.354) exactly. High enough that the Head is
learning real immune signal; far enough from 1 that NMP is a genuine independent validator rather
than a restatement — and the sets the two axes select overlap only 18–33 % by Jaccard.

## Current warehouse audit

Objective measurements are frozen in RAR
`0052-if-benchmark-test-set-construction-contr` at git `0b353ac`. The audit establishes the
following properties of the historical/current warehouse artifacts without assigning release
authority to a transient path:

- 113,562 declared entity IDs expanded to 226,048 chain FASTA records and only 57,127 exact
  sequences; deposited chain copies therefore entered sampling as separate IDs;
- current Tier 2 contains exact-sequence and same-PDB chain-copy duplication in both alleles
  (since quantified at 450/2,879 and 408/2,843 rows and retrofitted; see **Realized builds**);
- biological-to-IF sequence bytes changed for 2,434/2,864 DRB1\*07:01 Tier 2 rows and
  2,391/2,814 DRB1\*04:01 rows;
- the biological-frame CATH screen has zero Tier 2 hits, but the required final IF-ready rerun finds
  69/2,864 and 49/2,814 hits, respectively;
- the two Tier 1 source JSONs have 11/15 and 6/15 CATH hits under the frozen strict `>0.30` rule;
- only 45/261 and 143/366 source Tier 1 spans directly index their chain sequence, while all spans
  match after applying the recorded UniProt-to-chain offset;
- every historical Tier 2 row fails the current per-row validator because the legacy schema still
  requires non-null CATH topology for a selection mode that intentionally omitted it;
- current IF-ready manifests/load-coordinate summaries still describe older 3,102/3,095-row,
  `min_coverage=0.0` generations rather than the current cohort artifacts;
- the historical Gaussian scored tables/histograms and per-row CATH evidence are absent from the
  warehouse, and `coverage_fraction`/Tier 2 resolution are absent from the canonical table;
- no DRB1\*15:01 cohort existed at audit time (one has since been built; see **Realized builds**).

These findings stand as of the audit. The duplication and the missing `coverage_fraction` have
since been addressed by the C2 retrofit, which is membership-only and changes no measurement; the
remaining findings — absent Gaussian evidence tables, absent per-row CATH evidence, stale manifests
— are provenance gaps that only a full v3 rebuild or a completed release chain can close.
Therefore, existing 0701/0401 artifacts may be used only to interpret an already completed run when
that run's exact input hash or snapshot can be recovered. The mutable `if_ready/main` alias is not a
valid input for a new run, and none of these artifacts is a v3 release. A new v3 cohort must be
rebuilt from source rather than described as an exact rematerialization of those files.

## Implementation status

The v3 surface is implemented separately from the legacy chain-sampled scripts above. Those legacy
entry points remain historical and are not adapters for this protocol.

| v3 surface | current status | remaining release gate |
|---|---|---|
| RCSB entity/instance producer | content-bound Search + official polymer-instance GraphQL, C1 revalidation, exact sequence groups, rejection ledger, and revision-addressed coordinate cache implemented | run the full fresh C1 snapshot for the production build |
| Tier 1 source/projector | strict-full canonical UniProt preflight passes 74 source units / 2,864 valid interval mappings; detailed PDBe enriched-mmCIF residue producer and three-frame projector replay implemented | freeze detailed residue maps for every production candidate and complete C6/C4/backfill |
| C4/C5/C7 immune/leakage | actual MMseqs cov-mode 0/2 producer/replay and sharded native NetMHCIIpan C5/C7 producer/replay implemented; 4.3i filter/context presence semantics proven on the real binary | score the realized full source/final eligible pools; no membership exists yet |
| C6 materialization | entity/label/auth-aware mmCIF parser, exact residue mapping, MSE->M normalization, coverage, clean mmCIF, coordinate registry, and DPLM `load_coords()` replay implemented | download the production coordinate snapshot and materialize every preregistered attempt |
| schema/assembly/release | v3 row/cross-artifact schema, Tier 1/Tier 2 collision/dedup, family clustering replay, typed evidence bundle, 13 machine gates, versioned candidate finalizer, and first-release publication mode implemented | assemble a complete 15 + 3,000 candidate and pass all 13 gates |
| execution | `build_if_benchmark_v3.py`, parameterized SLURM launcher, fixed 18-stage production DAG, bounded arrays, predecessor seals, and byte-exact resume implemented and documented in the runner | freeze the reviewed production contract/task tables, then execute stage-by-stage |

Real mini-integration has independently replayed two RCSB instances through clean mmCIF and
`load_coords()`, all 86 C5 and 1,155 C7 windows from the real NetMHCIIpan install, MMseqs v13.45111
under both frozen coverage modes, and the approved DRB1501 full-data epoch-24 Head. This proves the
interfaces, not cohort capacity after all attrition. A versioned full candidate and all 13 release
gates remain mandatory; `if_ready/main` must remain unchanged until a separate publication review.

## Governance boundaries

- Selection logic changes only by versioning this protocol; a realized build changes only through
  a new immutable release manifest.
- `PROTOCOL/cases/` remains the log for downstream screening/design decisions. Dataset releases use
  `dataset_manifest.json`; downstream CASEs cite its release ID and digest.
- Tier 3 cannot be reintroduced into the benchmark under this protocol.
- A new allele is a separate release, not an in-place edit to another allele's cohort.
- Historical cohort files and results are never relabeled as v3 after the fact.
