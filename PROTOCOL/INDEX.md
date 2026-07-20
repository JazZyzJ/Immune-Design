# PROTOCOL Index

Repeatable **manual decision procedures** (SOPs) agreed between user and agent, so that
recurring selection / gating / triage conventions are not re-litigated every run. Distinct
from `PLAN_*.md` (one-off task blueprints) and `LOG.md` (history): a PROTOCOL is a reusable
*how we decide* document with its scientific rationale.

Add one line per protocol below (`- [Title](file.md) — when to use`). Keep protocol bodies
in their own files; this index is only for locating them.

## Protocols

- [Holo catalytic-competence gating](holo_catalytic_gate_calibration.md) — cofactor-dependent enzymes (metal and/or ligand-enclosed pocket): the T0–T3 tiering for deciding which designs are catalytically plausible; exact-WT normalization, when a crystal absolute IS valid (metal site only), TS-analogue contact fingerprints, and the pre-registered monitor-only rule for substrate poses. Instantiated on ADA/P56658 (Zn + PRH + adenosine)
- [Monomer shortlist & refine-seed selection](shortlist_and_refine_seed_selection.md) — constrained single-protein RF redesign runs (hard active-site anchors) feeding AF3 tetramer + de-immunization refinement; how to pick List 1 (tetramer feeder) and List 2 (refine seeds)
- [Tetramer prediction, interface analysis & refold-gate](tetramer_prediction_gate.md) — run List 1 (or any designs) through ESMFold2/Protenix holo-tetramer prediction (local GPU MSA, 4 ligands, complete CIF+PAE outputs), then score both repeated uricase interfaces post-prediction with BSA/contact and optional Rosetta dG/SASA, packstat, H-bond, salt-bridge, and BUNS metrics; both backends reproduce the 1R51 crystal
