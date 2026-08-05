# PROTOCOL Index

Repeatable **manual decision procedures** (SOPs) agreed between user and agent, so that
recurring selection / gating / triage conventions are not re-litigated every run. Distinct
from `PLAN_*.md` (one-off task blueprints) and `LOG.md` (history): a PROTOCOL is a reusable
*how we decide* document with its scientific rationale.

Add one line per protocol below (`- [Title](file.md) — when to use`). Keep protocol bodies
in their own files; this index is only for locating them.

## Protocols

- [New redesign target → RF-ready data](new_target_data_pipeline.md) — data-prep only: a fresh target (WT sequence + collaborator functional residues, or a de novo design table) → the three artifacts a constrained RF run consumes (IF-ready parquet, `PDB_ROOT`, constraint manifest); predicted structure not the crystal, `if_sequence_offset == 0`, manifest round-trip validation. Stops at RF-ready
- [De-immunizing a validated binder without losing binding](binder_interface_deimm_selection.md) — a binder that already works (de novo miniprotein, nanobody) must lose T-cell epitopes and keep its interface: the lock set built from a predicted complex (contact + 8 Å support shell) with a per-epitope anchor-pocket admissibility check, then the B0–B3 tiers. Constraint over selection; every binding decision computed on a complex, never a monomer
- [Holo catalytic-competence gating](holo_catalytic_gate_calibration.md) — cofactor-dependent enzymes (metal and/or ligand-enclosed pocket): the T0–T3 tiering for deciding which designs are catalytically plausible; exact-WT normalization, when a crystal absolute IS valid (metal site only), TS-analogue contact fingerprints, and the pre-registered monitor-only rule for substrate poses. Instantiated on ADA/P56658 (Zn + PRH + adenosine)
- [Full-cohort refinement and tetramer-gate handoff](shortlist_and_refine_seed_selection.md) — refine every technically valid constrained RF design with equal search controls, retain the paired WT → seed → refined lineage, and hand every canonical refined output to the tetramer plausibility gate; no monomer-derived List 1/List 2 shortlist
- [Tetramer prediction, interface analysis & refold-gate](tetramer_prediction_gate.md) — run a paired WT/seed/refined cohort through ESMFold2/Protenix holo-tetramer prediction (local GPU MSA, 4 ligands, complete CIF+PAE outputs), then score both repeated uricase interfaces post-prediction with BSA/contact and Rosetta dG/SASA, packstat, H-bond, salt-bridge, and BUNS metrics; both backends reproduce the 1R51 crystal. Also holds the **frozen uricase frame contract**: generate full-length with the initiator Met hard-locked, predict/compare in the mature frame at `--crystal-offset 0`, order full-length with Met — plus the per-parent Met-excision table (A0A9P8P4R1 and Q9RV70 must NOT be Met-stripped)
