# Active-15 Uricase WT-Core Release Manifests

## 1. Objective and claim boundary

Build the RF input space for every Active-15 parent-by-allele cell whose WT
epitope cores contain at least one admissible MHC-II anchor. The product is a
set of near-WT constraint manifests: every non-admissible position remains
fixed to the parent WT and every admissible anchor in every openable WT core is
left editable.

This task does not claim that RF will mutate an editable position or eliminate
an epitope. `openable` is a pre-generation search-space property. `resolved`
is a post-generation property that requires rescoring all WT, shifted-register,
and newly created windows for the matching allele.

## 2. Frozen evidence and sequence inputs

- Position/core evidence:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/therapeutic_enzymes/uricase_active15_ev_structure_v1/04_join_v1/evidence_join`
  (production join completed 2026-08-02; 15 parents, 45 parent-by-allele
  cells, 273 WT cores, 2,457 core-position rows).
- Parent sequences:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/therapeutic_enzymes/uricase_active15_ev_structure_v1/00_inputs/query/query15_simple.fasta`.
- RF-ready parent rows:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/uricase_caseset_if_ready_unified.parquet`.
- RF backbone root:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/pdbs_if_ready`.
- Sampler config: the existing `c1_null.yaml`; no h-map, controller, sampler,
  seed, temperature, or step-count change is part of this task.
- DPLM checkpoint:
  `/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt`;
  its content SHA-256 is computed and frozen in the launch ledger.

All paths remain CLI inputs to code. The concrete paths above are the frozen
production invocation, not Python defaults.

## 3. Openable-core contract

Candidate anchor registers are `P1`, `P4`, `P6`, and `P9`.

A core-anchor position is blocked when any of the following is true:

1. `in_C80_stable`;
2. `in_sigma80_robust` and `sigma_hard_lock_status == eligible`;
3. `in_full_contact_5of5` (which contains the `mid_contact_5of5` set in the
   frozen evidence);
4. interpretable representative-sample Rosetta `in_energy_ddg_gt0_reu`;
5. `energy_position_status == scanned_noninterpretable_AGP`.

The energy rule is a conservative campaign veto, not a calibrated universal
activity threshold. Unscanned positions do not gain a safety claim from energy;
they remain governed by the other evidence gates.

Homolog active-site/relay annotations remain audit columns and are not an
independent hard lock. This follows the experiment definition: the final
manifest fixes every non-core position already, while a core anchor is admitted
or rejected by conservation, qualified covariance, structure, and energy rather
than by a transferred functional label. Any overlap remains explicit in the
decision ledger.

Definitions:

- `safe anchor`: a candidate anchor not in the frozen blocked union;
- `openable core`: a WT core with at least one safe anchor;
- `blocked core`: a WT core with no safe anchor;
- cell free set: the union of **all** safe anchors from **all** openable cores in
  that parent-by-allele cell;
- cell fixed set: every canonical parent position not in the free set.

No openable core is dropped, no minimum hitting set is used, and no mutation
identity is selected before RF. A cell is emitted when it has at least one
openable core and its fixed fraction is at least 0.90.

Measured acceptance target from the frozen join: 184/273 cores are openable;
44/45 cells are emitted; all 15 parents retain at least one allele; the largest
free set has 22 positions and the minimum fixed fraction is 0.9329. These are
recomputed outputs and fail-fast expectations for this production bundle, not
generic constants in the scientific definition.

## 4. Artifacts

Publish atomically under the caller-supplied output directory:

```text
decision/
  parent_allele_core_release_candidates.{tsv,parquet}
  core_anchor_decisions.{tsv,parquet}
constraint_manifests/
  <protein_id>__<allele_tag>__all_safe_anchors__strict_v1.yaml
cohorts/by_parent/
  <protein_id>.parquet
launch_manifest.tsv
metadata.json
```

Each constraint manifest has exactly one protein entry and expresses
`fixed = all positions - free` through the existing `hard_anchors` schema.
There is no runtime/schema extension. One-row parent cohorts prevent unrelated
rows from silently running unconstrained.

## 5. Validation and falsification gates

Before publishing:

1. Evidence rows must have the frozen cohort cardinalities and unique keys;
   parent positions must be exactly `0..L-1` and reconstruct the parent FASTA.
   The two consumed parquet SHA-256 values must equal the declarations in the
   production `join_metadata.json`; the bundle records both source hashes and
   the join-metadata hash.
2. FASTA, position evidence, IF-ready parquet, and resolved structure chain must
   agree on protein ID, full sequence, and length.
3. Every free position must be a P1/P4/P6/P9 position in an openable core and
   have zero collision with the frozen blocked union.
4. Every openable core must contribute all of its safe anchors to the free set;
   every blocked core must contribute none.
5. Fixed and free sets must be disjoint, exhaustive, non-empty on both sides,
   and `n_fixed / L >= 0.90`.
6. Every YAML must round-trip through `load_constraint_manifest`, pass
   `validate_against_sequence`, contain exactly `L - n_free` unique anchors,
   and separately match the full-sequence MD5 (the current runtime does not
   enforce `sequence_md5`).
7. Every one-row cohort must resolve exactly one existing structure and be
   paired only with manifests for the same parent.
8. The production bundle must recompute 44 emitted cells, 15 represented
   parents, 184 openable cores, and zero free/gate collisions. Any mismatch
   fails without replacing a prior output.

Post-RF acceptance is intentionally outside this builder. A later run must
verify all hard anchors, restrict observed mutations to the free set, and rescore
the complete allele-specific sequence landscape rather than only the original
WT core coordinates.
