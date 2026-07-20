# PROTOCOL: Holo catalytic-competence gating for cofactor-dependent enzymes

## Scope / when to use

A redesign target whose catalysis depends on a **metal cofactor and/or a ligand-enclosed pocket**, where
we must decide which candidates are *catalytically plausible* before spending wet-lab effort. Instantiated
on **ADA / P56658** (Zn²⁺ + adenosine substrate + PRH transition-state analogue, open↔closed cycle); the
tier logic generalizes to any metallo-enzyme with an experimental cofactor reference.

This is a **gating/decision** protocol. It assumes RF-ready data already exists (see
`new_target_data_pipeline.md`) and that designs have been generated.

## Core principles

1. **Normalize to exact-WT under the identical protocol.** Every geometric criterion is a *deviation from
   the exact-WT prediction produced by the same predictor, same ligand set, same sampling*. Absolute
   thresholds taken from a crystal are contaminated by predictor-vs-crystal systematic offsets and by any
   construct/sequence mismatch in the deposited structure.
2. **…except where a genuine physical absolute exists.** Metal coordination has real physical bounds
   (His Nε2–Zn ≈ 2.0–2.2 Å). Ligand *poses* do not. Use an absolute anchor **only** for the metal site.
   This is not redundant with (1): a relative-only gate has a blind spot — if the predictor
   **systematically** misplaces the cofactor, WT and candidates are wrong together and the relative gate
   passes everything. The absolute arm closes that hole.
3. **Never hold a candidate to a bar the exact-WT positive control cannot clear.** If WT's own prediction
   is unstable for a metric, that metric is a **monitor**, not a filter — decided *before* looking at candidates.
4. **A two-state enzyme must retain both states.** For an open↔closed cycle, "matches the closed state" is
   the wrong target; a candidate locked into one state can be catalytically dead. Two-state accessibility
   is a monitor, never a single-state gate.

## Tiers

| Tier | Content | Strength |
|---|---|---|
| **T0** | **apo structure** vs the exact-WT design backbone: scTM, active-site sidechain RMSD over the safety-max anchors, anchor pLDDT | **hard** |
| **T1** | **cofactor coordination** — two arms: (a) deviation vs exact-WT same-protocol, (b) absolute experimental band | **hard** |
| **T2** | **closed-pocket + TS analogue**: contact-set fingerprint vs exact-WT, cofactor–ligand geometry | rank / soft |
| **T3** | **physiological substrate**: multi-seed sampling; discriminator only if the WT control's dispersion clears a pre-declared bar | monitor (conditional) |
| — | **two-state accessibility** (open vs closed reference) | monitor |

### T0 — apo (hard)
Standard v2 struct eval against the exact-canonical design backbone. Calibrate the floor on the **WT
self-consistency** run (refold WT, evaluate against its own backbone) — a candidate should not be held to a
tighter bar than WT achieves. *ADA instantiation:* WT ESMFold2 self-consistency gives active-site sidechain
RMSD **0.27 Å** (P56658) / **0.82 Å** (P23793 ADI), anchors complete, anchor pLDDT 98.6 / 97.5.

### T1 — cofactor coordination (hard, two arms)
**(a) Relative arm:** coordination distances must not drift from the exact-WT same-protocol prediction
beyond the WT-calibrated tolerance.
**(b) Absolute arm:** coordinating atoms must fall inside the experimental band, measured here from both
references (they agree to ≤0.19 Å, confirming the site is **state-invariant** and therefore a valid absolute anchor):

| Coordination | 1VFL (open, 1.80 Å) | 1KRM (closed, 2.50 Å) | operational band |
|---|---:|---:|---|
| His15 **NE2**–Zn | 2.10 | 2.04 | 1.9–2.5 Å |
| His17 **NE2**–Zn | 2.00 | 2.13 | 1.9–2.5 Å |
| His214 **NE2**–Zn | 2.21 | 2.20 | 1.9–2.5 Å |
| Asp295 **OD1**–Zn | 2.42 | 2.23 | 1.9–2.6 Å |

**Operational details that silently break this gate if missed:** coordination is through His **NE2** (ND1 sits
at 4.1–4.4 Å) and Asp295 **OD1** (OD2 at 3.6–3.8 Å). Evaluating ND1/OD2 measures the wrong atom and passes/fails
everything. Ligand-bound evaluation must **explicitly retain the Zn**; an apo run cannot satisfy T1.

### T2 — closed pocket + TS analogue (rank / soft)
Use the **contact-set fingerprint** (which residues contact the ligand within a fixed cutoff; compare by set
overlap against exact-WT), **not** a tight pose RMSD — the closed reference is 2.50 Å, so ligand coordinates
carry real uncertainty and an RMSD threshold would be measuring crystal noise. Retaining the TS-analogue
pocket demonstrates the **closed conformation and pocket shape** are preserved; it is **necessary, not
sufficient** for catalysis (a transition-state analogue is not the substrate).

### T3 — physiological substrate (conditional monitor)
Predict exact-WT + cofactor + substrate with **multi-seed sampling** and measure the WT pose dispersion
(inter-sample ligand RMSD / contact-fingerprint consistency). **Pre-register** the admissibility bar before
scoring any candidate:
- WT dispersion **within** the bar → substrate pose may be used as a soft discriminator, threshold calibrated on WT spread.
- WT dispersion **outside** the bar → substrate pose is **monitor-only for the whole campaign**; it may never eliminate a candidate.

*Why:* there is no exact-sequence native-substrate Michaelis crystal for ADA, so an unstable WT pose is a
property of the prediction, not of the candidate. Protenix is the validated workhorse for this sampling
(calibrated against AF3 on LuxSit: apo TM 0.985, holo ligand COM 0.13 Å) and is fast enough for many seeds.

### Cross-cutting — two-state accessibility (monitor)
Compare against **both** the open and closed references and read them as one two-state check. Do not gate on
a single state: the design backbone may itself sit closer to one state (ADA: the exact-canonical AF model is
nearer **closed** — safety-anchor CA RMSD 0.475 Å vs 1KRM but 1.047 Å vs 1VFL), so a single-state gate would
pull candidates toward a state the backbone was never in.

## Calibrated instantiation — ADA / P56658 (Protenix, local ColabFold MSA, 2026-07-18)

Exact-WT holo baselines, run under the identical protocol candidates will use. **These numbers are the
normalization reference; re-derive them whenever the predictor, MSA source, or ligand set changes.**

**T1 — Zn coordination: the predictor PASSES its own absolute check.** Across 20 WT samples (3 configs),
every coordinating distance is **100% inside the 1.9–2.5 Å experimental band**, sd 0.01–0.03 Å:
His15 NE2 2.06–2.19 · His17 NE2 2.08–2.12 · His214 NE2 2.15–2.19 · Asp295 OD1 2.34–2.36
(experimental 2.04–2.10 / 2.00–2.13 / 2.20–2.21 / 2.23–2.42). The absolute arm is therefore **trustworthy
and armed**: a candidate that leaves the band is a real failure, not a predictor artifact.

**T2 — PRH fingerprint must be WT-normalized, NOT crystal-normalized.** Exact-WT recovers only
**13/15** experimental 1KRM contacts (Jaccard **0.72**) while being internally stable (intra-WT
Jaccard(∩,∪) = **0.89**, consensus 16 residues). Requiring candidates to match 1KRM above ~0.72 would fail
exact-WT itself. **Gate on deviation from the WT-predicted consensus; 0.89 is the achievable ceiling.**

**T3 — adenosine is PROMOTED from monitor to discriminator.** WT dispersion over 10 samples is tight:
pairwise ligand RMSD median **0.10 Å** (range 0.04–0.21), contact Jaccard **0.95**, consensus 15 residues.
This clears the pre-registered bar, so the substrate pose *may* discriminate, with the threshold scaled to
the WT spread (a candidate deviating ≫0.1 Å from the WT pose is signal, not noise).

**ADI / P23793 + L-arginine — primary state `L_ARG_PHYS_QP1` (+1); result is protonation-robust at WT level.**
The physiological zwitterion (`NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]`, net +1) is the **primary** state; the
neutral PubChem form is retained only as a sensitivity control (both defined with `state_id` in
`holo_baseline/ligand_states.yaml` — candidates must cite the same id to be comparable).

*Primary metric = group-aware interaction fingerprint* (ligand groups identified by connectivity, not atom
names). Under **+1**, all three groups are perfectly stable (intra-Jaccard **1.00**, 10 samples) and every
declared partner is present in **10/10** samples:

| group | declared partners | hit rate | salt bridge (mean ± sd) |
|---|---|---|---|
| guanidinium | D161, D271 | 10/10, 10/10 | 2.85 ± 0.07 / 2.87 ± 0.07 Å |
| carboxylate | R180, R232 | 10/10, 10/10 | 2.86 ± 0.04 / 2.71 ± 0.04 Å |
| α-ammonium | L44, N155, G392 | 10/10 each | — |

Catalytic monitor: **C398 SG → guanidinium CZ 2.98 ± 0.06 Å** (poised for the amidino-enzyme intermediate),
H269 ND1/NE2 → ligand 3.03 ± 0.04 Å.

**Protonation comparison:** all three group consensus sets are **identical** between +1 and neutral
(Jaccard 1.00 each), salt bridges agree within 0.06 Å, catalytic distances within 0.05 Å →
**protonation-robust** *at the WT level*. The ranking-stability clause of the decision rule cannot be
evaluated until candidates exist; re-check it on the first design cohort.

**Raw RMSD is an artifact detector here, not a metric.** Neutral run: raw pairwise RMSD median **0.94 Å**
but **symmetry-aware 0.11 Å**; charged run: raw 0.10 Å, symmetry-aware 0.10 Å. Permitting guanidinium
NH1/NH2 and carboxylate O/OXT exchange collapses the apparent 0.94 Å spread — it was atom-naming symmetry,
not a pose difference. **Always report the symmetry-aware value; keep raw RMSD as monitor only.**

**Independent validation of the anchor annotation:** every consensus contact residue for both ligands falls
inside the corresponding safety-max hard-freeze set (ADA 15/15, ADI 15/15) — the predicted substrate sits
exactly in the annotated active site.

## Gotchas

- **Non-canonical experimental constructs.** 1KRM/1VFL are 356-aa bovine preparations with 15–16 substitutions
  and **canonical = PDB author + 3**. They are geometry references, never design sequences. Verify the mapping
  by residue identity before measuring (author 12/14/211/292 → HIS/HIS/HIS/ASP).
  The substitutions do **not** touch the Zn ligands, which is why the absolute arm of T1 remains valid.
- **Assembly.** If the native state is an oligomer whose interface borders the substrate channel (ADI/P23793:
  A2 homodimer), monomer-only refolding is **exploratory** — add an assembly-return check and an experimental
  oligomeric-state assay before claiming retained activity.
- **Do not stack crystal-absolute thresholds outside the metal site.** Everything except T1(b) is exact-WT-normalized.

## Verification

- T1 absolute band reproduces on both references (≤0.2 Å spread) before any candidate is scored.
- WT positive control passes T0 and T1 under the identical protocol; if it does not, the protocol — not the
  candidate set — is at fault.
- T3 admissibility decided and written down before candidate scoring begins.
