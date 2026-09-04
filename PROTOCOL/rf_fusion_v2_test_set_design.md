# PROTOCOL: RF Fusion V2 single-allele test-set design

Protocol ID: `rf-fusion-v2-test-set-design/1`
Frozen: 2026-08-24

## Scope / when to use

Use this protocol to run the current single-allele RF Fusion V2 design method over an
**unconstrained IF-ready test-set cohort**. The input is one native/reference sequence and one
resolved backbone per protein. The output is a content-bound four-root archive plus two explicit
selection surfaces: a definitive-feasible Head Pareto front and, only for proteins with no
feasible endpoint, a labelled structure-rejected fallback front.

This protocol covers RF generation, recursive Fusion, evidence preservation, and the handoff of
generation products. It does not cover:

- active-site/interface-constrained target redesign;
- dual-allele steering;
- independent immune evaluation or method-comparison statistics;
- post-generation refinement. The refinement extension is deliberately reserved at the end of
  this document for the follow-up agent.

The scientific architecture is defined in `doc/FUSION_V2.md`; the historical experiments and
their evidence are in `doc/RF_Fusion_v2_Cluster_Runbook.md`. This protocol is the reusable
operational decision layer, not another scientific analysis record.

## Three-layer record contract

Keep stable logic, executable procedure, and one run's values separate:

| layer | authority |
|---|---|
| this PROTOCOL | scientific workflow, invariants, and interpretation |
| `rf_fusion_v2_test_set_design.runner.md` | fixed execution order and `{{slot}}` command templates |
| `PROTOCOL/cases/<tag>.yaml` | exact cohort, paths, digests, resources, job IDs, and outputs |

Do not copy a cluster path or job ID into this file. Do not hide a scientific parameter only in a
Slurm command; the CASE must record every realized value.

## Input eligibility

A cohort is eligible only when all of the following hold:

1. The test-set parquet has exactly one row per `protein_id`, a non-empty uppercase AA20
   `sequence`, and a matching `sequence_length`.
2. Every protein resolves to exactly one cleaned `.pdb` or `.cif` under one `PDB_ROOT`.
3. The sequence in the test set is the complete reference used by cumulative safety and D0 local
   attribution. Reference bytes are written once and signed per protein.
4. The selected Head is the full-data production checkpoint for the requested allele, at the
   pre-frozen CV-median epoch. A full-data model supplies steering but no held-out performance
   claim.
5. The policy artifact, permissive whole-landscape declaration, Head config, Head checkpoint,
   schedule band, sampler, DPLM checkpoint, structure stack, and code revision are all explicitly
   content-bound.
6. The cohort is unconstrained. A hard-anchor manifest changes the editable domain and therefore
   requires its own constraint-stratified schedule band and a different CASE/protocol.

Failure of any item blocks materialization; it is never repaired by dropping proteins silently.

## Frozen RF/Fusion method

| component | standard value |
|---|---|
| profile | `testset_d4_k12_r40` |
| substrate | 100-step DPLM reference flow, temperature 1.0, `constant_one`, controller off, background remask fraction 0 |
| recursion | progressive D4 at `r=40` |
| checkpoints | `50 -> 60 -> 70 -> 80 -> 90` |
| breadth | `K=12` exact lookaheads at every rung |
| population | four independent single-lineage roots; only `master_seed` differs within a protein |
| policy | Head-directed capped support, policy v2 |
| D0 reward bootstrap | best definitively admissible D0 endpoint |
| deeper donor gate | strict lineage-incumbent improvement by `epsilon_R=0.005` raw logit |
| local write floor | `0.017012596130371094` raw logit |
| write cap | `ceil(0.05 * N_editable)` |
| Head window domain | 12--25, raw-logit scale |
| runtime immune gate | whole-landscape gate effectively permissive; incremental gate off |
| structure admission | definitive ESMFold2 evaluation with `scTM >= 0.70` |
| archive | preserve every endpoint and every typed failure; archive elite is not the only retained result |
| external NMP | absent from generation and selection |

`D4` means four recursive transitions after the initial generated endpoint pool; artifacts may
therefore contain endpoints labelled D0 through D4. `K=12` is donor breadth per rung, not “12 final
designs.” Four roots are four independent seeded lineages, not an active population of width four
inside one ladder.

The counterfactual Head-call ceiling is an engineering bound derived from the cohort's largest
editable domain. It may increase to cover a longer eligible sequence; this does not alter the
search law. The policy's numerical epsilon, local tolerance, write fraction, D/K/r schedule, and
structure threshold do not change by protein or allele.

## Why K12 remains the standard

K12 is the smallest breadth with a completed multi-root ceiling and independent downstream
validation. Doubling it to K24 approximately doubled archive diversity and slightly increased
protein coverage, but did not move lineages deeper or produce a stable scalar-ceiling improvement.
K8 has no direct calibration evidence. Therefore K12 is retained as the generalization setting;
changing K is a new experiment, not a fairness adjustment.

## Head and calibration rules

Use the production full-data Head named by the CASE. The CASE records its checkpoint path and
SHA-256 digest. Never substitute a CV fold, a full-data run's `best.pt`, or another allele's Head.

The policy artifact must match the exact Head evaluator identity while carrying the globally
frozen operating values above. This identity binding is not protein-specific calibration. No
per-protein immune threshold is fitted. The schedule band is Head-free and may be reused only when
the substrate and unconstrained editable-domain law match; a constrained cohort must be
recalibrated by stratum.

## Evidence and failure semantics

Each `(protein, master_seed)` cell is a complete evidence unit. A usable campaign requires all four
cells to persist a manifest, archive, endpoint table, structure table, terminal table, and fragment
record, including cells with `n_ok=0`.

The serial launcher deliberately returns exit code `3` when a job contains both successful and
typed-failed cells. Slurm therefore labels that job `FAILED (3:0)` even when every requested cell
was processed and preserved. Campaign completeness is decided from manifests/fragments and clean
realized caps, not from the Slurm state string alone. Timeout, missing bundles, corrupt fragments,
foreign identities, and breached or unverifiable caps are genuine failures.

## Standard generation products

The authoritative product is the complete set of per-cell bundles. From the verified four-root
grid, materialize:

1. **Definitive-feasible immune proxy front** — deduplicate within protein by exact sequence, then
   retain the complete first front minimizing frozen Head `global_risk` and
   `positive_mass_density = sum(max(residue_hotspot, 0)) / sequence_length`.
2. **Structure-rejected fallback** — only for a protein with zero definitive-feasible endpoint
   across all four roots, retain the first front minimizing positive-mass density while maximizing
   scTM. Every row remains labelled `structure_rejected_fallback` and
   `terminal_validated=false`; it is not a standard final design.

Never use external NMP to retroactively change those Head-defined fronts. If NMP is later run, it
annotates the fronts or creates a separately labelled post-hoc diagnostic.

## Comparison and output-count discipline

The method searches many more than eight sequences per protein. Comparing its best archive member
against an eight-design baseline is a capability comparison, not matched-generation-budget
evidence.

When a paper table requires the same **reported output count**, use
`materialize_v2_archive_facade.py --mode feasible_immune_pareto --official`. The default official
panel is `up-to-8`; `--final-candidates-per-protein N` overrides eight without changing the search
or requiring `--official`. Selection always starts from the complete verified definitive-feasible
pool, peels exact Pareto layers of `(global_risk, positive_mass_density)`, orders within a layer by
the two full-pool average-rank sum and sequence MD5, and takes up to N. The raw per-cell bundles
remain authoritative and are never deleted by materialization. This equalizes the number
evaluated/reported, not the number searched. Always report together:

- the full archive/feasible-pool size and search configuration;
- the primary complete Pareto front;
- the exact requested N, selection law and realized count per protein;
- a statement that search compute remains unequal.

The unguided NoD `n=8` cohort is the correct RF-kernel baseline when available. It is a comparator,
not a parent or runtime input to Fusion V2.

## Refinement extension boundary

**Reserved for the follow-up refinement agent.** The RF portion ends only after four-root grid
closure and generation-surface materialization. The refinement extension must consume immutable,
content-hashed generation products and must state:

- which generation surface supplies seeds;
- the per-protein/per-seed retention rule;
- the frozen Head-refinement profile and structure gate;
- how refined Pareto outputs map back to source endpoint/root provenance.

It must not rewrite the RF configuration or reinterpret `K=12` as a final-output count.
