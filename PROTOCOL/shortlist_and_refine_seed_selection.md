# PROTOCOL: Monomer shortlist & refinement-seed selection (constrained redesign runs)

## Scope / when to use

Single-protein (or per-protein) **constrained** RF redesign runs — hard active-site anchors
frozen at exact WT — whose downstream is **AF3 tetramer prediction** and optional
**de-immunization refinement**. First instantiated on Q00511 (Aspergillus flavus uricase,
302 aa) @ HLA-DRB1\*04:01, run
`Results/RF/HLA-DRB1_04_01/q00511_safetyv1_drb0401_b1open_n600_seed42__20260713T061730Z`
(600 designs, 24 hard anchors, constraint `meta/constraint_manifest.yaml`).

Reuse this procedure whenever a constrained redesign run needs to be triaged into a
tetramer-feeder list and a refinement-seed list. Re-calibrate the numeric floors per run
(the *logic* is fixed; the *thresholds* are data-grounded per run — see §Calibration).

## Core principles (why the method is what it is)

1. **Global fold saturates under conservative active-site protection.** Typical run:
   scTM 0.91–0.98, foldability 100%. scTM / global pLDDT therefore have almost no
   discrimination — **do not gate or rank on them**. The discriminating signal is
   active-site reliability.
2. **The tetramer pocket is inter-protomer** (urate/O2 bind across protomers). A monomer
   refold can only report active-site *fidelity*, never tetramer *displacement*. So monomer
   active-site metrics are **GATES** (pass/fail reliability); the **AF3 tetramer is the real
   ranker**. Do not over-engineer monomer ranking as if it settled activity.
3. **Refinement is structure-one-sided.** Objective = immune (distinct epitope-core count);
   structure is a conjunctive **accept gate** over absolute `scTM`, direct-functional
   `cat_max_scRMSD`, and `predicted_active_site_min_pLDDT`. It preserves-or-degrades,
   **never repairs**. Expected structural change vs seed is ≤ 0. → **Seed refinement from the
   best structures.** A low-immune / bad-structure design is a bad seed: refinement cannot fix
   its fold. There is no "bad structure that refinement turns good" case to chase.
4. **Anchors are frozen during refinement** → active-site geometry is protected by
   construction; refinement structural risk is confined to global fold from *non-anchor* edits.
5. **Compute asymmetry drives list sizing.** AF3 tetramer + wet-lab are effectively uncapped;
   immune→0 via refinement is cheap; **good structure is the scarce resource** (empirically
   ~1/1000 yield for good-structure ∧ immune-0). → Cast a **wide but active-site-reliable** net
   for tetramer; **over-provision** refinement seeds/outputs.

## Metrics used (per-run schema)

From `eval_structure/structural.parquet`: `scTM`, `pLDDT`, `global_ca_RMSD`,
`active_site_sidechain_RMSD` (aggregate over active-site sidechain atoms),
`predicted_active_site_min_pLDDT` (worst active-site residue confidence),
`matched_anchor_count` (glycine anchors report `no_sidechain` → systematically one short; not a
defect, not a discriminator).

Derived from `eval_structure/structural_residues.parquet` (`is_anchor`, `residue_idx_1based`,
`sidechain_RMSD`):

- **`cat_max_scRMSD`** = max sidechain RMSD over the **direct-functional catalytic residues**
  (Q00511 1-based: 11, 58, 59, 160, 177, 228, 229, 255, 257). The activity firewall.
- **`shell_max_scRMSD`** = max over the safety-max shell / PROSITE anchors (looser tolerance).
- Glycine anchors (no sidechain, e.g. Q00511 Gly287) are excluded from these.

Immune from `eval_immune/imm_nmp.parquet`: `n_strong_binders` (NetMHCIIpan rank_EL < 2%).

## List 1 — monomer shortlist → AF3 tetramer (immune-agnostic)

**Floor** (active-site reliability, all conjunctive — this is what separates "plausible for
tetramer" from junk; global fold is *not* the cut):

```
cat_max_scRMSD ≤ 2.0 Å   AND   predicted_active_site_min_pLDDT ≥ 85   AND   scTM ≥ 0.94
```

**Rank** (AF3 run order; pure structure, immune never enters):
`cat_max_scRMSD ↑ → predicted_active_site_min_pLDDT ↓ → active_site_sidechain_RMSD ↑ → global_ca_RMSD ↑`.

**Tier label** for prioritization: T1 pristine
(`scTM≥0.95 ∧ pLDDT≥92 ∧ global_ca_RMSD≤2.0 ∧ min_pLDDT≥85 ∧ active_site_sidechain_RMSD≤1.1 ∧ cat_max≤1.5 ∧ shell_max≤2.5`);
the rest of the floor-passers are the extended net.

**No cap** (AF3 uncapped): deliver *all* floor-passers, ranked + tiered, best-first.

Q00511 instantiation: floor → **225** designs; T1 pristine subset → **42**.

## List 2 — refinement seeds → refine → re-eval → AF3 (de-immunized product)

**Seeds** = top structure slice (T1 pristine, Q00511 ~42). Deep-refine each (wide beam, keep
the full shortlist — one seed yields many outputs). Do **not** spread thin across
structure-mediocre seeds (principle 3).

**Structure-tie-break among seeds**: prefer lower starting `n_strong_binders` (fewer edits to
reach 0 → less non-anchor fold perturbation).

**Refinement config**: anchors frozen (run constraint manifest); objective = epitope-core
count → 0; live refold = ESMFold2; structure gate = `scTM ≥ X ∧ cat_max_scRMSD ≤ Y ∧
predicted_active_site_min_pLDDT ≥ Z`. The metric set is fixed, but `X/Y/Z` have no repository
defaults and must be calibrated per run. See `PLAN_RF_REFINE.md` / `scripts/refine_rf_designs.py`.

**Post-refine selection**: re-run structure eval on **all** refined outputs; keep
(good structure ∧ `n_strong==0`). Over-provision to ~1000+ outputs given the ~1/1000 yield.
Pilot first, then concentrate budget on the high-yield seeds/scaffolds.

## Pipeline

```
List 1  ──────────────────────────────────►  AF3 tetramer            (structure/activity ceiling;
        (225 ranked, immune-agnostic)                                  which scaffolds tetramerize;
                                                                       NOT de-immunized)
List 2  ──► refine → structure re-eval ─────►  AF3 tetramer            (de-immunized product:
        (T1 seeds, deep)   (keep good-struct ∧ n_strong==0)            immune 0 ∧ good structure)

cross-feed: List 1 tetramer winners prioritize which List 2 scaffolds to deep-refine.
The two lists overlap by design (both structure-first).
```

## Reproduce (script + commands)

Selection is implemented by **`scripts/select_redesign_shortlists.py`**
(registered in `doc/SCRIPTS.md` → Phase C · RF Refinement). It reads only the run's own eval
artifacts + constraint manifest, so anyone can regenerate both lists deterministically:

```bash
python scripts/select_redesign_shortlists.py \
    --run-dir Results/RF/HLA-DRB1_04_01/q00511_safetyv1_drb0401_b1open_n600_seed42__20260713T061730Z
# floors are CLI args; protocol defaults: --cat-max 2.0 --min-plddt 85 --sctm-min 0.94
```

Outputs land under `<run-dir>/selection/`:

- `list1_tetramer_shortlist.{parquet,csv}` — List 1, ranked + tier-labelled (Q00511: 225).
- `list2_refine_seeds.{parquet,csv}` — List 2 seeds, drop-in `--seed-table` (Q00511: 42).
- `selection_manifest.json` — git sha, thresholds, survivor counts, catalytic set (provenance).

Then List 2 → refinement (anchors frozen, wide beam, keep full shortlist), via
`scripts/refine_rf_designs.py` / `scripts/submit_refine.slurm`:

```bash
SEED_TABLE=<run-dir>/selection/list2_refine_seeds.parquet \
CONSTRAINT_MANIFEST=<run-dir>/meta/constraint_manifest.yaml \
GATE_SCTM_MIN=<run-calibrated-floor> \
GATE_CAT_MAX_SCRMSD_MAX=<run-calibrated-ceiling> \
GATE_ACTIVE_SITE_MIN_PLDDT_MIN=<run-calibrated-floor> \
ALLELE='HLA-DRB1*04:01' OUT_DIR=<refine-out> \
sbatch scripts/submit_refine.slurm
# deep search per seed; post-refine re-eval keeps protocol-gate-pass ∧ n_strong==0
```

## Calibration (re-run per new dataset)

The *logic* above is fixed; the numeric floors are per-run because global-fold saturation and
active-site RMSD scales shift with protein/allele. For a new run, recompute the
`cat_max ≤ X ∧ min_pLDDT ≥ Y` survivor table and pick the floor that lands a clean set
(target: well above a token 20–30, without admitting half-reliable active sites). Record the
chosen floor + survivor count in the run's returned artifacts, not by editing this protocol.
