# PROTOCOL Index

Repeatable **manual decision procedures** (SOPs) agreed between user and agent, so that
recurring selection / gating / triage conventions are not re-litigated every run. Distinct
from `PLAN_*.md` (one-off task blueprints) and `LOG.md` (history): a PROTOCOL is a reusable
*how we decide* document with its scientific rationale.

Add one line per protocol below (`- [Title](file.md) — when to use`). Keep protocol bodies
in their own files; this index is only for locating them.

## Three layers

| layer | file | holds | changes |
|---|---|---|---|
| PROTOCOL | `<pipeline>.md` | **why** — decision logic + scientific rationale | rarely |
| RUNNER | `<pipeline>.runner.md` | **how** — the fixed agent procedure, variables left as `{{slots}}` | rarely |
| CASE | `PROTOCOL/cases/<tag>.yaml` | **this one decision** — inputs, criteria applied, outputs | one per decision |

A case points at configs, never copies them; `scripts/validate_case.py` is the runner's
mandatory Step 0. See `PROTOCOL/cases/SCHEMA.md`.

## Protocols

- [IF benchmark test-set construction](if_benchmark_test_set_construction.md) — v3 authority for a versioned natural-protein benchmark: exact entity-sequence groups (not deposited chain copies) as Tier 2 selection units while retaining all mapped structures as fallbacks, final-IF-sequence CATH isolation and NetMHCIIpan Gaussian sampling, allele-native Tier 1 coordinate projection, post-membership full-data Head annotation, and atomic content-bound release. Existing warehouse cohorts are historical and do not yet satisfy v3
- [New redesign target → RF-ready data](new_target_data_pipeline.md) — data-prep only: WT sequence + collaborator active sites → the IF-ready parquet + PDB_ROOT structures + active-site manifest a constrained RF/refine/inpainting run consumes; predicted structure, numbering verification, manifest validation; stops at RF-ready
- [De-immunizing a validated binder without losing binding](binder_interface_deimm_selection.md) — a binder that already works (de novo miniprotein, nanobody) must lose T-cell epitopes and keep its interface: the lock set built from a predicted complex (contact + 8 Å support shell) with a per-epitope anchor-pocket admissibility check, then the B0–B3 tiers. Constraint over selection; every binding decision computed on a complex, never a monomer
- [Holo catalytic-competence gating](holo_catalytic_gate_calibration.md) — cofactor-dependent enzymes (metal and/or ligand-enclosed pocket): the T0–T3 tiering for deciding which designs are catalytically plausible; exact-WT normalization, when a crystal absolute IS valid (metal site only), TS-analogue contact fingerprints, and the pre-registered monitor-only rule for substrate poses. Instantiated on ADA/P56658 (Zn + PRH + adenosine)
- [Constrained redesign shortlist & refinement](shortlist_and_refine_seed_selection.md) — constrained RF redesigns feeding refinement and, where applicable, assembly/holo evaluation; includes WT-relative structure calibration, shard sizing, and final wet-lab handoff
- [Monomer structure-rank refine seeds (no tetramer)](monomer_structure_rank_refine_seeds.md) — monomeric targets ranked by class-specific monomer structure evidence, with one refine-seed list per protein and allele
- [LuxSit parent/variant holo selection](luxsit_parent_variant_holo_selection.md) — LuxSit parent-background redesigns with free catalytic positions; multi-sample Protenix-DTZ gating, dual parent/variant references, and genotype-stratified refinement selection
- [Tetramer prediction, interface analysis & refold-gate](tetramer_prediction_gate.md) — run a paired WT/seed/refined cohort through ESMFold2/Protenix holo-tetramer prediction (local GPU MSA, 4 ligands, complete CIF+PAE outputs), then score both repeated uricase interfaces post-prediction with BSA/contact and Rosetta dG/SASA, packstat, H-bond, salt-bridge, and BUNS metrics; both backends reproduce the 1R51 crystal. Also holds the **frozen uricase frame contract**: generate full-length with the initiator Met hard-locked, predict/compare in the mature frame at `--crystal-offset 0`, order full-length with Met — plus the per-parent Met-excision table (A0A9P8P4R1 and Q9RV70 must NOT be Met-stripped)
- [RF Fusion V2 single-allele test-set design](rf_fusion_v2_test_set_design.md) — run the frozen unconstrained D4/K12/four-root V2 method with a full-data production Head, preserve the complete typed cell grid, materialize feasible and structure-rejected generation surfaces, and hand immutable products to later comparison/refinement
- [RF Fusion V2 post-generation Head refinement](rf_fusion_v2_refinement.md) — stage 2 of the V2 design protocol: seed a bounded NMP-free Head edit search from the definitive-feasible pool (up to 8 per protein by pareto-layer/rank-sum), refine under the generation's own full-data Head with a seed-relative scTM gate, emit an untruncated per-seed Pareto surface, and keep structure-failed and wild-type seeds labelled as diagnostic rather than product
