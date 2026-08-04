# PROTOCOL: De-immunizing a validated binder without losing binding

## Scope / when to use

A **binder that already works** — a de novo miniprotein, nanobody, or any protein whose function is
a defined interface with a named partner — must be de-immunized while keeping that interface. Not
for de novo binder *generation* (use BindCraft/RFdiffusion) and not for enzymes (use
`holo_catalytic_gate_calibration.md`).

Assumes RF-ready data exists (`new_target_data_pipeline.md`), where for a binder "active site" is
replaced by the **interface consensus derived from a predicted complex** — the manifest cannot be
authored from the monomer or from a residue list.

First instantiated on PD1_b2 (114-aa BindCraft binder vs PD-1, `cases/pd1b2_binder_deimm_v2_*`).

## Why a binder is not just another enzyme

The architecture transfers — freeze the functional residues, redesign the rest, refine, gate — but
two things break, and both change the protocol rather than its parameters:

| | enzyme (uricase / ADA) | binder |
|---|---|---|
| frozen fraction | 8/300 ≈ **3%** | 24–57/114 = **21–50%** |
| geometry | small, buried, rigid pocket | large, solvent-exposed patch |
| **can a monomer refold testify to function?** | **yes** — catalytic geometry is buried and reproducible | **no** — the partner is absent |
| ligand for the holo check | Zn / substrate (small molecule) | **a whole protein chain** |

The second row is the one that matters. An enzyme's catalytic side-chain RMSD in an apo monomer
refold is real evidence. A binder's interface side chains in an apo monomer refold are **not
evidence about binding at all** — they are surface rotamers with no partner to order them. Every
gate that decides binding must therefore be computed on a **complex**.

## Core principles

1. **Constraint beats selection, and they are not substitutes.** A gate answers "which of these is
   least bad"; a lock set changes "what gets made". This is not a preference — it is the only thing
   the literature shows working. Every campaign that redesigned a protein and kept its function did
   it with a hard **positional** constraint: CoDAH forbade CDR mutations (25/25 retained binding);
   resurfacing touched only RSA > 30% surface; Salvat froze the active site (18/18 functional). And
   the scores did *not* work even where the method did — CoDAH reports **no correlation** between
   its own rotameric energy or human-string-content score and measured binding.
2. **Normalize to the exact parent under the identical protocol.** Same predictor, same chains, same
   sampling. A threshold borrowed from another target or another model is uncalibrated here.
3. **Never hold a candidate to a bar the parent cannot clear** — decided *before* looking at
   candidates. Corollary: measure the parent's own **sample-to-sample dispersion** first; a metric
   whose parent spread swamps the candidate differences is a monitor, not a filter.
4. **Epitope removal targets MHC-II anchor pockets, not covered spans.** A 15-mer strong window
   overlapping the interface is not itself a conflict. What decides whether an epitope is removable
   is whether its **P1/P4/P6/P9** pockets — the positions you must actually mutate — are free.
   Scoring "epitope overlaps interface" at span level will reject targets that are perfectly
   separable at pocket level.
5. **Immune load is the objective, not a feasibility gate.** Using epitope count as a hard filter
   selects for the most aggressively mutated designs — exactly what endangers binding. Maximize it
   *within* the set that passes the binding tiers, never the reverse.
6. **Mutation count is itself a risk variable.** Every experimentally validated de-immunization that
   kept function used **1–9** simultaneous mutations; the one campaign that removed all epitopes at
   once (Zhao 2015, 16 DR4 epitopes from lysostaphin) produced no folded, functional enzyme and
   needed four rounds of directed evolution to recover. Budget mutations like a resource.

## Stage 1 — the lock set (this is the protocol's real content)

Built from a predicted **complex** ensemble, in three shells:

| set | definition | v2 treatment |
|---|---|---|
| **contact** | binder residues with any heavy atom ≤ **4.0 Å** of the partner, in ≥ 60% of samples | **hard-frozen** |
| **support shell** | within **8.0 Å** but not in contact | **hard-frozen** |
| **free** | everything else | designable |

Freezing the support shell is not conservatism for its own sake. Roguska 1996 calibrates it
experimentally: replacing **all three** murine surface residues within **5 Å** of the CDRs lost
affinity, and retaining any **one** restored it — surface residues near the interface are not safe.
An 8 Å shell is the conservative side of that number.

### Lock-set admissibility — a pre-flight stop condition

After fixing the lock set and before generating anything, check **per epitope**:

> does at least one of its P1/P4/P6/P9 anchor pockets remain in the free set?

- **≥ 2 free pockets** → removable with room to choose.
- **exactly 1** → removable but brittle; that single position must carry the whole reduction.
- **0 free pockets** → **this epitope cannot be removed under this lock set.** Either relax the lock
  deliberately and record it, or accept the epitope and say so. Do not discover this after a run.

## Stage 2 — tiers

Mirrors the holo tier stack, with the complex playing the role the cofactor plays for an enzyme.

| Tier | Content | Strength |
|---|---|---|
| **B0** | **monomer foldability** vs the parent under the identical refolder: scTM, pLDDT | hard **floor only** |
| **B1** | **complex formation** — interface confidence (i_pTM / i_pAE) | hard |
| **B2** | **interface identity** — contact-set fingerprint vs the parent consensus | hard |
| **B3** | **ΔΔG_bind** by a point-mutation-sensitive physics method | optional / rank |
| — | anchor side-chain geometry in a **monomer** refold | **monitor only** |

### B0 — monomer foldability (floor)
Necessary, weak, and explicitly *not* evidence about binding. Anchor side-chain RMSD and
min-pLDDT-over-anchors belong here as **monitors**, because on a binder both statistics are maxima
or minima over a set that contains mobile termini and long surface side chains. Measure the parent
under the same protocol and read its own value as the floor of resolution.

### B1 — complex formation (hard)
Interface confidence on a re-predicted complex, which is the single most discriminating family
available. Two facts set the thresholds:

- **The community's ~10 Å interface-PAE convention is too loose.** Two independent optimizations say
  so: AlphaProteo's grid search over **640,000** experimentally characterized designs lands at
  `pae_interaction < 7.0` with `pLDDT > 90`, and RFdiffusion used per-target cutoffs of 5–8 Å. On
  BindCraft's own released data, tightening `i_pAE` from 0.35 (≡ 10.85 Å raw; the normalization is
  PAE/31) to 0.20 (≡ 6.2 Å) raises the hit rate 42.8% → 60.8% while keeping a third of designs.
- **Those thresholds are bound to the model they were calibrated on** — AF2-monomer, single
  sequence, templated target. The BindCraft authors report AF3-class models giving a large fraction
  of false positives in the same filtering role. Running the gate on an AF3-class predictor
  (Protenix) means the calibrated numbers do **not** transfer; either add an AF2-monomer
  reprediction to inherit them, or use parent-normalized thresholds and label them self-calibrated.

Caveat to check per target: if the predictor's i_pTM spread across parent samples is negligible,
parent-normalization gives a zero-width tolerance and an **absolute** floor must carry the gate.

### B2 — interface identity (hard)
The binder analogue of the ADA T2 transition-state fingerprint, and the same logic applies: compare
**contact sets**, not pose RMSD. Two things about *how* to compare them were got wrong once and are
recorded here so they are not got wrong again.

**Use retention of the parent's contacts, not symmetric Jaccard.** A Jaccard punishes a design for
*gaining* interface contacts exactly as hard as for *losing* the parent's — but only the loss is
damage. Measured on the PD1_b2 v2 cohort (400 designs): Spearman of Jaccard against contacts
**lost** was −0.381, against contacts **gained** −0.954. The symmetric statistic was 95% a measure
of how much *extra* surface a design buried. A `Jaccard ≥ 0.85` rule rejected 81/400 designs, nearly
all of them for gaining contacts, while **370/400 lost none of the parent's** at all. The gate that
answers "does this design still make the parent's interface" is

$$
\mathrm{retention} = \frac{|C_{\text{design}} \cap C_{\text{parent}}|}{|C_{\text{parent}}|}
$$

with the symmetric Jaccard kept as a monitor. Gains are only benign once you have checked they are
*additive* rather than a relocated interface — verify with complex TM vs the parent (0.983 median
here) and with the retention itself; a binder that rolled to a new epitope would lose parent
contacts, not keep all of them.

**Match the reference's noise level to the candidate's.** A design's number is a *consensus* (an
occupancy-filtered set) compared against the parent's *consensus*. The parent's sample-to-sample
pairwise Jaccard is therefore the wrong bar: it measures raw per-sample noise that the occupancy
filter has already removed, and it is systematically pessimistic. The apples-to-apples reference is
the parent's **leave-one-out consensus stability**. On PD1_b2 the two differ enormously — sample
pairwise median 0.808 / min 0.731, leave-one-out **1.000** — so a bar of 0.85 justified as "the
parent's own intra-ensemble minimum" was resting on a statistic from the wrong reference class.
(State the LOO estimate's limit too: with 5 samples and a 0.6 threshold, `ceil(0.6·4) = ceil(0.6·5) = 3`,
so some of that stability is threshold arithmetic rather than measured robustness.)

### B3 — ΔΔG_bind (optional)
The only family with positive evidence on *our* task. Note carefully that the evidence **inverts**
between two tasks that are easy to conflate:

| task | best family |
|---|---|
| is this design a binder at all? | AF confidence (i_pAE AUROC 0.769) — Rosetta static descriptors are at chance: ShapeComplementarity 0.532, n_InterfaceHbonds 0.523, n_InterfaceResidues 0.509, dSASA 0.508, Hotspot_RMSD 0.548, every CI spanning 0.5 |
| **does this mutant still bind?** | **physics ΔΔG** — SSIPe 0.78, FlexddG 0.77, FoldX 0.74; ΔAF2 mean_pae **0.54 (random)**, ΔAF3 ipTM 0.72 |

So: drop the static Rosetta interface descriptors as gates, and if an orthogonal arm is wanted, it
must be **ΔΔG_bind**, not shape complementarity. Even then, 0.77 is mediocre and FoldX's real 95%
prediction interval on binding ΔΔG is **±3.5 kcal/mol** — a "ΔΔG < 0.5 ⇒ still binds" rule is not
statistically defensible. The one defensible experimental anchor is Cao 2022's "|ΔΔG| < 1 kcal/mol
= no effect".

## Stage 3 — immune criteria

1. **Hard:** no strong-binder window may cover any **contact** anchor. (The parent may itself fail
   this; then it is an improvement target, not a non-inferiority bar — record which.)
2. **Objective:** reduce the strong-binder count relative to the parent, per allele.
3. **Guard:** no *new* epitope may be created whose anchor pockets fall in the contact or shell set.
4. **Staging:** remove epitopes allele-by-allele, or core-by-core, before attempting all at once.

Predictor caveat worth stating once per campaign: MHC-II predictors disagree badly — at matched
percentile thresholds MARIA and NetMHCIIpan accept only ~1/5 of each other's peptides. And epitope
*count* is not immunogenicity: human IL-2 carries ~560 predicted T-cell epitopes and is tolerated,
because central tolerance is not in any of these models.

## Gotchas

- **Do not read a monomer metric as binding evidence.** It is the single easiest mistake here,
  because the numbers look like the enzyme case's numbers.
- **AF is structurally blind to destroying mutations.** hHR23a L198A renders the domain
  experimentally disordered yet gives Cα RMSD 0.1 Å and pLDDT 84 for **both** WT and mutant. ΔpLDDT
  vs ΔΔG correlates at r ≈ −0.17. Never let a small ΔpLDDT argue that a mutant is fine.
- **"The identical protocol" includes the GPU architecture.** Measured on PD1_b2: the same parent
  JSON, same seed, same 5 samples gives Protenix i_pTM **0.9552** on RTX PRO 6000 Blackwell
  (torch 2.7.1+cu128) and **0.9507** on H200 (cu126) — a 0.0046 shift, **11× the 0.0004
  within-ensemble spread** the metric must resolve. Neither is more correct; they are the same
  weights under different kernels. So a parent baseline measured on one architecture cannot
  normalize designs predicted on another — re-predict the parent alongside the cohort, on the same
  hardware, and let it flow through the same `_ratio_to_wt` columns. pTM (+0.0012) and pLDDT
  (−0.086 on ~93.5) shift far less and are not at risk; interface confidence is.
- **Carry B-factors when extracting a chain from a predicted CIF into `PDB_ROOT`**, or the structural
  eval reports `reference_pLDDT = 0.00` and the reference confidence baseline is silently lost.
- **Gly anchors have no side chain** (`match_status = no_sidechain`), so `matched_anchor_count` reads
  n−1 for every design. Expected, not a defect.
- **No Δ-metric acceptance rule for "this mutant still binds" exists in the literature.** All AF
  thresholds in circulation were calibrated on design-vs-non-binder, a different discrimination.
  Either calibrate against experimental read-back or state plainly that this step is protected by
  the lock set rather than by a score.

## What a correctly constrained cohort looks like

Recorded from the first full instantiation (PD1_b2 v2, 400 designs, `cases/pd1b2_binder_deimm_v2_*`)
because it is the shape to expect, and because it changes what the tiers are *for*.

When the lock set does its job, **the interface tiers stop discriminating**:

| tier | pass |
|---|---|
| B0 monomer foldability (scTM ≥ 0.956) | 219/400 |
| B1 complex formation (i_pTM ≥ 0.90, no clash) | 397/400 |
| B2 parent-contact retention (lose ≤ 1 of 21) | 392/400 |
| all three | 217/400 |

B1 and B2 pass at 99% and 97% *even among the 181 designs that fail B0*, so the whole selection
reduces to B0 alone (219 → 217). That is not a failed gate — it is the constraint having already
done the work at generation time, leaving only the one failure mode the lock set does not control:
whether the redesigned 57 free positions still fold. Read it as confirmation, and do not report a
high B1/B2 pass rate as evidence that the designs bind — the tiers were near-vacuous by
construction. Their remaining job is to catch the individual outlier (here: 3 designs that lost ≥ 2
parent contacts, 3 that fell below the i_pTM floor).

The cost side has to be reported with it. Against the v1 cohort (24 locked positions) on the same
"folds **and** zero strong binders" criterion, the loosely constrained run produced **more** joint
winners — 48 vs 41 on DRB1*04:01, 43 vs 34 on DRB1*15:01. What the constrained run bought was a
median of **23–24 mutations instead of 40**, **0 of 33 support-shell positions mutated instead of
16**, and complex-level evidence where v1 had none. Whether that trade is worth ~20% fewer
candidates is a scientific judgement about mutation count as a risk variable (principle 6), not
something the numbers settle on their own — state it as a trade, never as a clean win.

## Verification (before any candidate is scored)

1. The **parent passes B0, B1 and B2** under the identical protocol. If it does not, the protocol is
   at fault, not the candidates.
2. The parent's **intra-ensemble dispersion** for every B1/B2 metric is measured and written down —
   this fixes each metric's role (filter vs monitor) and its achievable ceiling.
3. **Lock-set admissibility** is checked for every epitope; a zero-free-pocket epitope is declared
   before generation, never discovered after.
4. The **mutation budget** is stated as a target, and the realized count reported against it.
