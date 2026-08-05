# Uricase (Q00511) Tetramer Activity-Preservation Constraint — Structure + Coevolution

**Objective.** Define the *minimal set of monomer positions that must be held at wild-type identity* so that a
de-immunized redesign of *Aspergillus flavus* urate oxidase (uricase, UniProt **Q00511**, drug basis of
rasburicase) retains catalytic activity. Immunogenicity editing is handled separately; this document is only
about the activity/assembly constraint.

**TL;DR.** Uricase is an obligate homotetramer whose active site is *built across a subunit interface*
(catalytic residues donated by the neighbouring monomer). Locking the active-site pocket **identity** alone
(config `v1`) is insufficient at the current assay sensitivity: **29/29 assayed `v1` designs are below the
activity call threshold while WT is strong**. There is no experimental oligomeric-state measurement for these
designs, so the present data do not distinguish failed tetramerization, active-site mis-registration, and
folding/expression loss. Round 1 therefore uses a deliberately conservative **3×2 structure × sigma hard-lock
matrix plus a full-structure/sigma-off control**. Every cell also fixes the eight strongly conserved positions
that the full structural mask misses. Its purpose is activity rescue and threshold mapping, not a claim that
every retained WT identity is ultimately necessary.

---

## 0. Numbering convention (foundational — every position in this doc uses it)

Q00511 canonical = **302 aa**, `sequence_md5 = 37bdca69e4f1ddd590d6d54618ef9152`
[`data/Uricase/01_sequences/seeds/Q00511_Aspergillus_flavus.fasta`; matches the md5 declared in the v1 config].
Initiator Met1 is removed in the mature protein (N-acetyl-Ser2), and the 1R51 crystal resolves residues 2–295.

The following four indices are **numerically identical** and used interchangeably (verified this session, 0
mismatches over all cited positions):

```
index_0b  ==  (UniProt position − 1)  ==  PDB 1R51 author residue id  ==  Rosetta alanine-scan res_id  ==  EVcouplings pos_mature
```

Labels in prose use **UniProt** numbering (e.g. Thr58, Arg177). Tables use **index_0b** to match the design
config. Example cross-check: config `index_0b: 57 → Thr58`; 1R51 author Thr57; scan `res_id 57 = THR`; EV
`pos_mature 57, pos_uniprot 58, wt_aa T`. All agree.

---

## 1. Data provenance (all reproducible)

| Artifact | Path | What it is |
|---|---|---|
| 1R51 crystal (biological tetramer) | `Results/Reference/crystal/1R51_tetramer_ABCD.pdb` (`sha256 fd5575a5…`) | Experimental structure = Q00511. Used as `--crystal-ref`. |
| Interface metrics + Rosetta | `Results/Reference/1R51/` | BSA/SASA, InterfaceAnalyzer, per-residue Ala scan (AB, AD). |
| Rosetta run metadata | `Results/Reference/1R51/metadata.json` | Rosetta release-408 (`InterfaceAnalyzer`, `rosetta_scripts`), `ref2015`, contact 8 Å, interface-residue 5 Å, ddG sign = mutant−wildtype. Author: Kaiyi. |
| WT Protenix refold (holo tetramer) | `Results/Reference/protenix_wt_set0_AF/` | Q00511 WT prediction (`set0_AF`), 5 samples. Oracle calibration. |
| Design gate metrics (Q00511 rows) | `Results/Reference/gate/q00511_gate_metrics.csv` | WT + 4 order1 designs: activity/expression + iptm + `complex_TM_vs_*` + `xprot_*_dev`. |
| order2 designs (46 computed; 29 assayed) | `Results/Wetlab/order2/q00511_merged29_plus_refinement17_all_metrics.csv` | True-`v1` designs. The currently assayed 29/29 are below the activity call threshold; no oligomeric-state assay. 445 metric columns incl. sequence, xprot dev, active-site RMSD. |
| EVcouplings per-residue scores | `Results/Reference/Covariance/EVCoupling_covariance_per_residue_scores.csv` | Conservation `C_i` + coevolution `sigma` (+ percentiles/ranks), per position. Author: Kaiyi. |
| Round-1 sigma-v2 artifacts | `Results/Reference/Covariance/coupling/{02_couplings,03_scores,05_validation,06_round1_sigma_v2}/` | 500-iteration PLMC summary/ECs, per-residue scores, model-stability comparison, gap sensitivity, conservation add-on, exact matrix counts, and generation/preflight scripts. |
| v1 design constraint (24 anchors) | `inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml` | Current (failed) active-site lock. |
| Literature synthesis | `viz/1R51/uricase_oligomerization_literature.md` | 29 cited refs on uricase oligomerisation→activity. |
| Interface/active-site figures + BSA tiers | `viz/1R51/` (`analyze_interface.py`, `interface/design_priority.csv`) | Local per-residue BSA + tier assignment, composite-site views. |
| Proposed interface anchors (R1/R2) | `Results/Reference/proposed_v2_interface_anchors.yaml` | 25 Rosetta-hotspot additions, config-format. |

---

## 2. Structural basis: activity is a property of the tetramer, not the monomer

### 2.1 Interface architecture (D2 "dimer of dimers")

The tetramer has three symmetry-distinct interface *types* (chain labels are arbitrary; the types are not — they
are the three perpendicular 2-fold axes of D2). Numbers from
`Results/Reference/1R51/1R51_interface_summary.csv` (Rosetta ref2015):

| Interface type | Example pairs | BSA total (Å²) | ΔG_separated (REU) | salt bridges | shape compl. | role |
|---|---|---|---|---|---|---|
| **1 — catalytic dimer** | A:B, C:D | **5780** | **−155.4** | 2 | 0.717 | **active site sits here**; energetically dominant |
| 2 — tetramer | A:D, B:C | 5216 | −58.2 | 4 | 0.739 | holds the two dimers into the barrel; softer/more polar |
| 3 — diagonal | A:C, B:D | 739 | ~0 | 0 | — | minor |

Key asymmetry: **interface 1 is ~2.7× stronger in ΔG than interface 2** (−155 vs −58 REU) despite similar BSA —
i.e. the A:B catalytic dimer is the obligate, load-bearing unit. This is label-independent and reproduces the
literature BSA hierarchy (Girard 2010: AB ~6000 Å² > AC ~5200 > AD ~800) [lit synthesis §5]. Local heavy-atom
contact counts independently reproduce it (A:B/C:D 69 res ≫ A:D/B:C 60 ≫ A:C/B:D 9) [`viz/1R51/interface/interface_pairs.csv`].

### 2.2 Composite, cross-subunit active site

Each of the 4 active sites is assembled from **two** subunits [`Results/Reference/1R51/…` + `viz/1R51` geometry;
lit: Colloc'h 1997, Oksanen 2014, Kratzer 2014]:

- **Own subunit** contributes substrate anchoring: Phe160 (π-stack), Arg177 + Gln229 ("tweezers"), Asn255, His257.
- **Neighbour subunit (`*`)** contributes the catalytic chemistry: **Thr58\*** and **Lys11\*** (Thr–Lys general-base
  diad), **Asp59\***, plus pocket residues Tyr9\*, Ile55\*, Ala57\*.
- The **peroxo hole** (O₂/water site ~3.0–3.3 Å above the substrate plane) is built *across* the interface by
  **Thr58\*(neighbour) + Asn255(own)**; catalytic triad **Lys11\*–Thr58\*–His257** [Oksanen 2014].

Verified geometry on the crystal (this session, PyMOL on `1R51_tetramer_ABCD`): for site A (neighbour B),
min heavy-atom distance to the bound ligand — Arg177 2.96 Å, Gln229 2.85 Å, Asn255 3.41 Å, **Thr58\* 2.93 Å**;
the two catalytic residues sitting *above* the ring plane are just outside a 4 Å ligand shell — **His257 5.08 Å**
and **Lys11\* 5.51 Å** (and Lys11\* is 36.5 Å from its *own* subunit's ligand, confirming it is donated to the
neighbour's site). Consequence: a 4 Å ligand-contact cutoff *misses* Lys11\*/His257 — they must be added to any
"active-site" set by annotation, not geometry alone.

**Why this forces the constraint:** substrate *binding* (own-subunit tweezers) is physically separable from
*catalysis* (neighbour-subunit peroxo hole + relay). A tetramer that assembles but is mis-registered can retain
binding while losing turnover. Oligomerisation is therefore necessary for activity (textbook for this enzyme:
pH- or pressure-induced dissociation abolishes activity every time; interface point mutations F222S/S232L/Y240C
kill activity by degrading assembly, not by touching catalytic residues) [lit synthesis §2: Tian 2012, Girard
2010, Kratzer 2014 PNAS].

---

## 3. Forensics: why the de-immunized designs fail

Two batches of Q00511 redesigns, both experimentally inactive, isolate the cause.

### 3.1 order1 (n=4, pre-`v1`) — establishes the H2 signature

[`Results/Reference/gate/q00511_gate_metrics.csv`] WT `set0_AF`: activity **4**, iptm 0.970, min protein–protein
chain-pair iptm 0.944, `complex_TM_vs_crystal` 0.999, all `xprot_*_dev` = 0. The 4 designs:

| design | activity | complex_TM_vs_crystal | min_pp_iptm | xprot_Thr58_dev (Å) | v1 positions kept |
|---|---|---|---|---|---|
| design_0021 | 0 | 0.989 | 0.882 | 0.25 | 18/24 |
| design_0053 | 0 | 0.986 | 0.780 | 0.44 | 20/24 |
| design_0014 | 0 | 0.985 | 0.851 | 0.26 | 18/24 |
| design_0003 | 0 | 0.980 | 0.847 | **8.54** | 15/24 |

Interpretation: predicted complexes retain high global similarity (complex_TM 0.98–0.99), while one design has
large predicted cross-protomer catalytic drift. This motivates an H2 mis-registration hypothesis but does not
establish experimental assembly. These designs predate the v1 active-site lock (only 15–20/24 kept), so they do
not cleanly test v1.

### 3.2 order2 (46 computed; 29 assayed, true `v1`) — falsifies "lock the pocket only" at the current assay sensitivity

[`Results/Wetlab/order2/…csv`; 29/29 assayed below the activity call threshold per user]

- **v1 active-site lock enforced perfectly: 46/46 computed designs keep all 24 v1 positions** (this is the clean
  v1 constraint test order1 lacked); 29 currently have wet-lab activity measurements.
- Yet designs **mutate the interface heavily**: mean **69 of 130** interface positions changed (~53%), mean **17.2
  of 25** Rosetta interface hot spots, **0/46** preserve all hot spots; sequence identity to Q00511 ≈ 0.49.
- Expression collapses vs WT (design 5–14k vs WT 78,636; `expression` column) — a **stability**, not merely
  activity, signal.

**Conclusion (the central result): locking the active-site pocket identity is necessary but not sufficient.**
With the pocket fixed and the interface free, 29/29 assayed designs are below the activity call threshold.
The computational metrics are consistent with two candidate failure modes: (i) **registration/assembly** of the
composite active site and (ii) **stability/expression**. They do not establish oligomeric state. No positive
within-batch anchor exists yet; the usable contrast is WT (active) versus heavily redesigned `v1` sequences
(currently no called-active case).

---

## 4. Three constraint signals and how much they overlap

Three *independent* ways to ask "must this position stay WT", and their practical redundancy.

### 4.1 The signals

1. **Structure — burial/contacts** [`viz/1R51/interface/design_priority.csv`; local SASA]. Tiers (monomer,
   unique positions): tier1 cross-subunit active (5), tier2 own active (7), tier3 interface-core BSA≥20 Å² (87),
   tier4 interface-rim (33), tier5 non-interface (164).
2. **Energy — Rosetta alanine scan** [`Results/Reference/1R51/1R51_alanine_scan_by_position.csv`; ref2015, sign
   = mutant−wildtype, so **ΔΔG_bind > 0 = the WT residue carries interface binding energy (a hot spot)**].
   AB face 70 positions (sidechain max ΔΔG 5.69 REU), AD face 63 (max 5.78). "Hot spot" = a WT residue whose
   Ala-substitution destabilises the interface; the failed designs mutate these away.
3. **Evolution — EVcouplings** [`Results/Reference/Covariance/…csv`]. Two orthogonal columns:
   - `C_i` = **single-site conservation** (invariance). High C = catalytic/functional (e.g. Arg177 C_rank 3,
     Gln229 C_rank 2).
   - `sigma` = **coevolution / covariance** = a position's *marginal* coupling strength (a per-residue summary of
     the pairwise couplings J_ij). Top ECs correspond to **side-chain contacts / epistatic pairs** (Hopf/Marks;
     Marks 2011). High `sigma` does **not** prove that WT identity is intrinsically required: a Potts/DCA model
     encodes compatibility between residue states. **Round 1 nevertheless hard-locks the raw high-`sigma` node
     sets to WT as a deliberately blunt rescue intervention.** No conservation filter is applied to sigma: a
     residue supported by both signals is retained, because the current objective is rescue rather than signal
     attribution. This does not prove identity-level necessity or pairwise covariance causality; those require
     Potts ΔE scoring and the pair-swap test in §9.

### 4.2 Intersection (this session's key comparison)

| Comparison | Result | Reading |
|---|---|---|
| **C (conservation)** `C_i_nogap ≥ 0.8` (36) vs full structural scaffold (137) | 28/36 inside; **8 outside** | The eight full-structure blind spots become a universal round-1 add-on. |
| **C (conservation)** `C_i_nogap ≥ 0.5` (122) vs full structural scaffold (137) | 72/122 inside; **50 outside** | Adding all 50 would be a separate maximum-conservation intervention, not the present base. |
| **sigma (coevolution)** raw top-10% (31) vs full structural scaffold (137) | **4** overlap; Jaccard **0.024** | The raw sigma node set is nearly orthogonal to the structural scaffold. |
| **sigma (coevolution)** raw top-20% (61) vs full structural scaffold (137) | **10** overlap; Jaccard **0.053** | Lowering the sigma threshold adds mostly non-structural positions. |
| EV `active_site` flag (15) | 10 are high-C, **0 are high-sigma** | Catalytic residues are conserved but *not* coevolving (invariant ⇒ nothing to covary with). |

**Interpretation.** "Structural-contact" and raw `sigma` node sets are empirically complementary. The earlier
Jaccard 0.16 mixed conservation and sigma into one EV set and must not be attributed to sigma alone. Round 1 uses
raw sigma as an *operational rescue axis* and does not remove conserved sigma nodes. Separately, it fixes only the
eight `C_i_nogap ≥ 0.8` positions that lie outside the full structural scaffold. Structure-linked conserved
positions remain in their original structure tiers so that the structure ladder is not erased. Later rounds must
separate WT identity from pair compatibility (§9).

---

## 5. Unified constraint framework

### 5.1 Round-1 monomer lock ladder (301-residue mature monomer)

| Lock level | Contents | positions | % monomer | evidence source |
|---|---|---|---|---|
| loose base | v1 active-site/pocket (24) + eight high-C full-structure blind spots | **32** | **10.6%** | config + EV conservation |
| mid base | v1∪AB catalytic-dimer structure (79) + the same eight high-C positions | **87** | **28.9%** | structure + EV conservation |
| full base | v1∪AB∪AD full interface (137) + the same eight high-C positions | **145** | **48.2%** | structure + EV conservation |
| deferred maximum-conservation base | full structure ∪ all 50 outside-full `C_i_nogap ≥ 0.5` positions | **187** | **62.1%** | not used in round 1 |
| `+ sigma` | raw high-`sigma` positions with only a `gap_frac ≤ 0.5` majority-nongap guard | see §8 | — | EVcouplings |

Kaiyi's original grid masks use `C OR sigma` and therefore are not the round-1 sigma source: with `C0=0`, every
position is selected. Round 1 selects sigma directly from the per-residue score table with no `C_i` exclusion.
Conservation enters only through the predefined eight-position full-structure blind-spot add-on in §8; the 50
moderately conserved outside-full positions are not universal round-1 anchors.

### 5.2 From rescue mapping to a usable design threshold

The full structure mask is not a proposed final product: it deliberately compresses immune design space. Round 1
first asks whether any structure/sigma cell restores detectable activity. If more than one cell is active, select
the lowest-lock active cell as the starting bound. If only the most conservative cell is active, hold its sigma
level and lower the structure threshold in the next round. This establishes an empirical activity boundary
before optimizing immune score.

Refold metrics (`xprot_Thr58_dev`, `min_pp_chain_pair_iptm`, active-site RMSD) are retained as explanatory
telemetry. They do not replace wet-lab activity and should not be used to discard an entire matrix cell before the
first rescue experiment.

### 5.3 Deferred alternative: direct rescue edits

One can take an inactive design and restore WT at interface positions in decreasing-ΔΔG order. This directly
tests rescue on a fixed parent but is not the current round-1 route, which is intended to observe the RF inverse
folding process itself under controlled mask thresholds.

---

## 6. Caveats / open questions

- **No experimental oligomeric-state data yet** (SEC/MALS/native-MS). Current structural predictions and
  expression data generate registration/assembly and stability hypotheses, but cannot determine whether an
  inactive design formed a tetramer.
- **`sigma` is a compatibility, not identity, signal (central correction, §4.1/§9).** Round 1 intentionally uses
  WT hard-locking only as a rescue proxy. A positive result supports retaining the high-`sigma` network in this
  RF regime; it does not establish which identities or pairs are necessary.
- **Conservation add-on scope and numbering.** The universal add-on contains only the eight `C_i_nogap ≥ 0.8`
  positions outside the original full structural mask: `index_0b/pos_mature = 40,50,77,145,157,186,224,301`,
  corresponding to UniProt `G41,D51,E78,S146,S158,W187,S225,L302`. It is not a universal union of every high-C
  position. L302 has `gap_frac=0.911` and belongs to the terminal SKL/PTS1 context; it is retained as a
  rescue-first exact-WT anchor, not classified as activity-specific evidence.
- **EV MSA scope.** A same-sequence homomer family MSA mixes intra-chain and inter-chain constraints. Concatenating
  identical monomers does not create an independent paired-sequence signal. The appropriate QC is the observed
  monomer-versus-tetramer contact classification of EC pairs; pair-state causality remains a later experiment.
- **Design-level enforcement.** order1 shows the active-site lock was only partially honoured pre-v1; confirm any
  new (larger) lock is actually enforced 24/24 → N/N at generation time, not just intended.

---

## 7. Reproduction

```bash
# interface energetics + composite active-site composition (source: Results/Reference/1R51)
#   generated on Della by scripts/eval_tetramer_reference.py (Rosetta release-408, ref2015); see metadata.json
# local structural tiers + figures:
pymol -cq viz/1R51/analyze_interface.py            # -> viz/1R51/interface/{interface_pairs,active_site_composition,design_priority}.csv
# monomer tier ladder, EV intersection, order2 forensics: ad-hoc python over the CSVs cited in §1
#   (v1/R1/R2 index_0b sets and EV thresholds as defined in §5; numbering per §0)
# round-1 sigma/conservation sets and matrix counts:
python Results/Reference/Covariance/coupling/06_round1_sigma_v2/uricase_sigma_rescue_v2_audit.py --help
```

Key hard numbers and their sources are tagged inline above; all derived from files under `Results/Reference/`,
`Results/Wetlab/order2/`, `viz/1R51/`, and the v1 config. Coevolution definitions grounded in the EVcouplings
framework (Hopf et al. 2019, *Bioinformatics*; Marks et al. 2011, *PLoS ONE*).

---

## 8. Round-1 experiment plan (activity rescue)

**Design: one 3×2 structure × sigma matrix plus a full-structure/sigma-off control.** The two hypotheses are
partially testable in the same plate using the same RF seed IDs:

1. **Structure rescue:** at σ80, increasing loose → mid → full structure increases the probability or
   magnitude of activity.
2. **Sigma rescue:** at full structure, increasing OFF → σ90 → σ80 increases the probability or
   magnitude of activity.

These are operational intervention hypotheses. The matrix can locate an activity-rescue threshold and describe a
structure × sigma interaction. It cannot prove that the locked WT identities are biologically necessary or that
the effect is specifically pair compatibility. Full structure is a positive-rescue anchor, not the intended final
de-immunized setting; if it rescues, later rounds lower the structure threshold to recover immune design space.

**Signals + provenance.** Structure = Rosetta ref2015 interface + Ala scan
(`Results/Reference/1R51`); nested levels are V1 active-site (24) ⊂ V1∪AB catalytic dimer (79) ⊂
V1∪AB∪AD full interface (137). Every level receives the same eight-position `C_i_nogap ≥ 0.8` add-on outside
the full interface, giving effective bases 32 ⊂ 87 ⊂ 145. Sigma = EVcouplings `pairs.enrichment` from the deep
Q00511 family MSA (N_eff/L 10.7). The final `deep_iter500` PLMC run stopped with `ROUNDING_ERROR` rather than a
formal convergence flag. Its rankings are stable relative to the original 100-iteration run: mature-position
sigma Spearman = 0.997, top-L EC-pair Jaccard = 0.980, raw σ90 Jaccard = 1.000, and raw σ80 Jaccard = 0.968.
Tetramer-contact precision is 0.682 at top L. Exact source hashes and optimization status are recorded in each
config and the accompanying Reference artifacts listed in §1.

**Sigma definition.** Starting from the top-L long-range `sigma_pct`, select:

`sigma_pct >= t AND gap_frac <= 0.5`

There is **no conservation exclusion**. A site selected by both conservation and sigma stays fixed, which matches
the rescue-first objective. The 0.5 gap cap is only a majority-nongap sanity guard: all 31 σ90 and all 61 σ80
positions pass it, so the masks equal the raw top-10% and top-20% sets. Relative to the old 0.2 cap, σ80 retains
seven additional positions in the mature 190–202 segment (`gap_frac=0.216–0.483`). The σ90 set is a strict subset
of σ80. Alternative gap caps and 0.5L/2L pair definitions remain sensitivity telemetry and do not alter the mask.

**Universal conservation add-on.** Fix the eight `C_i_nogap ≥ 0.8` positions outside the original full
structure in every cell: `index_0b=[40,50,77,145,157,186,224,301]`, corresponding to UniProt
`G41,D51,E78,S146,S158,W187,S225,L302`. These eight do not overlap σ90 or σ80. This is deliberately the
full-structure blind-spot set, not a global union of every high-C position; otherwise conserved interface
positions would be removed from the structure ladder itself. The 50 outside-full positions at `C_i_nogap ≥ 0.5`
are deferred: adding them would make the full/σ80 cell 223/301 fixed and absorb 15 raw σ80 positions into the
common base.

**The matrix** (fixed positions; percent of 301-residue mature protein). Configs:
`inverse_folding/reference_flow/configs/uricase_q00511_r1_S{loose,mid,full}_sig{90,80}.yaml`.

| original structure ↓ + universal C8 \ σ → | **σ≥0.90** | **σ≥0.80** |
|---|---|---|
| **loose** = V1 (24) + C8 → base 32 | **63 (20.9%)** | **92 (30.6%)** |
| **mid** = V1∪AB (79) + C8 → base 87 | **115 (38.2%)** | **142 (47.2%)** |
| **full** = V1∪AB∪AD (137) + C8 → base 145 | **172 (57.1%)** | **196 (65.1%)** |

The σ90 and σ80 sets contain 31 and 61 positions, respectively; overlap with each base is counted once.

Control: `…_r1_Sfull_sigOFF.yaml` = full structure + C8, σ OFF = **145 (48.2%)**. Seven configs total.

**RF sampler contract.** Run all seven cells with the unchanged standard
`inverse_folding/reference_flow/configs/c1_null.yaml` (SHA256 `51563e79571f…`; default remask
`fraction_scale=1.0`). Checkpoint, backbone, temperature, number of steps, batch/scalar mode, seed IDs, and all
other runtime inputs remain identical; only the constraint manifest changes. Anchor count can change realized
commit/remask trajectories under this sampler. That mediated dynamics change is part of the production-RF
constraint intervention being measured, not normalized away as in a pure position-set causality experiment.
The provenance fields inside a constraint manifest do not select the sampler at runtime: submission must pass
`--config inverse_folding/reference_flow/configs/c1_null.yaml` and the cell-specific `--constraint-manifest`
explicitly.

**Plate allocation (75 unique sequences; seed-indexed replicate blocks).** Use the same 10 RF seed IDs in all
seven configs, giving a 70-sequence reproducible core. Add five extra RF seeds only to the most conservative
full/σ80 cell, so that cell has 15 total and the plate has 75 unique sequences. A shared seed ID defines a
deterministic whole-run outcome `Y_cell(seed)`, but it does **not** create a positionwise common-random-number
trajectory: changing the hard-anchor mask changes the number and order of categorical draws and desynchronizes
the random streams. The extra five estimate the positive-anchor hit rate and are not used in seed-blocked
contrasts.

**Primary wet-lab readout.** Activity rescue is primary. Preserve the continuous raw assay signal, background,
replicates, detection/quantification limits, and binary activity call; do not reduce sub-threshold wells to an
undifferentiated zero. Report the within-seed cross-cell association before claiming variance reduction from the
seed blocks; include seed ID as a blocking factor only when supported, and retain cell-level effect estimates and
active fraction as the primary robust summaries. Record expression/yield for every construct and, where the assay
permits, activity per expressed protein: an expression increase without a specific-activity increase is
folding/abundance rescue, not complete catalytic rescue. Without SEC/MALS/native-MS, no result should be labeled
tetramerization rescue.

**Computational telemetry (not a wet-lab gate).** Refold every design and retain
`xprot_Thr58/Lys11/His257_dev`, active-site RMSD, `min_pp_chain_pair_iptm`, global/tetramer confidence, WT
recovery, mutation count, and immune score. These metrics explain gradients after activity is known; they must not
replace the activity readout. If every cell remains below the assay detection limit, the result is "no rescue
detected under this assay," not evidence that structure or sigma has no effect.

---

## 9. Coevolution: causal validation and required artifacts

**Why a separate test.** The §8 coevolution ladder asks whether high-`sigma` WT locking helps in aggregate; it cannot say
**why** (identity vs compatibility): a design that frees a coupled position and fails may have failed by making an
*incompatible* combination, not by leaving WT. Distinguishing them needs a design that is **non-WT but
Potts-compatible** — which requires the pairwise
model, not the per-residue `sigma`.

**Required artifacts — NOW IN HAND** (`Results/Reference/Covariance/coupling/`, ported from Kaiyi's pipeline):
1. **Pairwise ECs** — `02_couplings/deep_iter500_ECs.txt` (45 451 pairs; `i aa_i j aa_j seg score`).
2. **Potts model** h_i(a), J_ij(a,b) — `deep_iter500.model` remains in the Della run directory; transfer it only
   when running ΔE because it is not needed by the round-1 hard-lock configs.
3. **N_eff — RESOLVED.** Deep-model N_eff/L = **10.7** and the pair model passes the structural QC gate
   (top-L tetramer-contact precision 0.682 ≥ 0.60, `05_validation/deep_iter500_validation.txt`). This establishes
   usable structural signal in the EC ranking; it does not by itself validate activity relevance, the sigma
   threshold, or WT hard-locking.
4. **Traceable run** — `02_couplings/deep_iter500_plmc_summary.json`, `03_scores/deep_iter500_scores.tsv`, and
   `06_round1_sigma_v2/` record theta 0.8, λ_h 0.01, λ_J 60.2, 12 223 valid sequences, gap sensitivity,
   conservation-base membership, final matrix counts, and exact scripts. The optimizer status is retained as `ROUNDING_ERROR` rather than
   being relabeled as convergence. Method = port of Skopintsev et al. *Science* 2026 (`science.aed6123.pdf`).

**Causal pair-swap experiment** (per strong-coupling pair (i,j); 6 conditions):

1. parent pair (both non-WT, as designed);
2. restore i alone;
3. restore j alone;
4. double-WT (restore both);
5. **Potts-compatible non-WT pair** (both non-WT, low ΔE by J_ij);
6. mutation-count-matched **incompatible** non-WT pair (control, high ΔE).

Decision: if **only WT-containing conditions rescue** → the constraint is *identity* (WT-lock justified). If the
**compatible non-WT pair (5) also rescues** while the incompatible control (6) does not → the constraint is *pair
compatibility* → do **not** hard-lock `sigma`; instead impose a **Potts ΔE ceiling** on the free positions and let
de-immunisation use any compatible substitution. This decision fixes how coevolution enters the design objective in
all later rounds.
