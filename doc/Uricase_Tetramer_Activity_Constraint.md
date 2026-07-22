# Uricase (Q00511) Tetramer Activity-Preservation Constraint — Structure + Coevolution

**Objective.** Define the *minimal set of monomer positions that must be held at wild-type identity* so that a
de-immunized redesign of *Aspergillus flavus* urate oxidase (uricase, UniProt **Q00511**, drug basis of
rasburicase) retains catalytic activity. Immunogenicity editing is handled separately; this document is only
about the activity/assembly constraint.

**TL;DR.** Uricase is an obligate homotetramer whose active site is *built across a subunit interface*
(catalytic residues donated by the neighbouring monomer). Locking the active-site pocket **identity** alone
(config `v1`) is provably insufficient: 46/46 true-`v1` designs are experimentally inactive while still
assembling into near-native tetramers, because the redesign is free to mutate the **interface** that holds the
two active-site halves in register, and the **intra-chain fold-core network** that maintains stability. Three
*independent* signals — structural burial, interface energetics, and evolutionary conservation/coevolution —
converge on a constraint scaffold of ~46–57% of the monomer. Recommended strategy: lock aggressively (~57%),
then run a **subtractive** (relax-from-max) experiment, because no active redesign exists yet to anchor an
additive search.

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
| order2 inactive designs (n=46) | `Results/Wetlab/order2/q00511_merged29_plus_refinement17_all_metrics.csv` | True-`v1` designs, all experimentally inactive (per user). 445 metric columns incl. sequence, xprot dev, active-site RMSD. |
| EVcouplings per-residue scores | `Results/Reference/Covariance/EVCoupling_covariance_per_residue_scores.csv` | Conservation `C_i` + coevolution `sigma` (+ percentiles/ranks), per position. Author: Kaiyi. |
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

Interpretation: designs **still assemble** (complex_TM 0.98–0.99) but the **cross-protomer catalytic residues
drift** (`xprot_*_dev` elevated vs WT 0.0) → **H2 (mis-registration)**, not gross H1 (dissociation). These
predate the v1 active-site lock (only 15–20/24 kept), so they do not cleanly test v1.

### 3.2 order2 (n=46, true `v1`) — falsifies "lock the pocket only"

[`Results/Wetlab/order2/…csv`; all inactive per user]

- **v1 active-site lock enforced perfectly: 46/46 keep all 24 v1 positions** (this is the clean v1 test order1 lacked).
- Yet designs **mutate the interface heavily**: mean **69 of 130** interface positions changed (~53%), mean **17.2
  of 25** Rosetta interface hot spots, **0/46** preserve all hot spots; sequence identity to Q00511 ≈ 0.49.
- Expression collapses vs WT (design 5–14k vs WT 78,636; `expression` column) — a **stability**, not merely
  activity, signal.

**Conclusion (the central result): locking the active-site pocket identity is necessary but not sufficient.**
With the pocket fixed and the interface free, 46/46 designs are inactive. Two failure modes operate together:
(i) **registration** — the composite active site drifts (H2); (ii) **stability/expression** — the fold is
destabilised. No within-batch gradient exists to exploit (all 46 broke the interface similarly); the signal is
the WT (0 interface mutations, active) vs all-46 (heavy interface mutation, all dead) contrast.

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
   - `sigma` = **coevolution / covariance** = a position's total evolutionary-coupling strength to the rest of
     the protein. Top ECs correspond to **side-chain contacts / epistatic pairs** (Hopf/Marks; Marks 2011;
     "co-evolving residues are in spatial proximity, mutations compensated by partners"). High sigma = the WT
     identity is constrained by its structural/functional partners; mutating it alone breaks a compensation network.

### 4.2 Intersection (this session's key comparison)

| Comparison | Result | Reading |
|---|---|---|
| **C (conservation)** top-10% (30) vs active site v1 (24) | overlap-coef **0.58**; 23/30 already inside the structural scaffold | **C is largely redundant** — it re-derives the active site we already lock. |
| **sigma (coevolution)** top-10% (31) vs inter-chain interface (130) | only **4** overlap; **27/31 lie outside the structural scaffold** | **sigma is nearly orthogonal** — it captures **intra-chain fold-core contacts** our inter-chain interface analysis structurally cannot see. |
| EV top-10% (61) vs structural scaffold (137) | **Jaccard 0.16** (84% of the union is non-overlapping) | The two schemes are mostly complementary. |
| EV `active_site` flag (15) | 10 are high-C, **0 are high-sigma** | Catalytic residues are conserved but *not* coevolving (invariant ⇒ nothing to covary with). |

**Interpretation.** "Structural-contact" and "covariance" are orthogonal *in principle* and, empirically, overlap
only through the conservation channel. The **distinctive** evolutionary contribution is `sigma`, and it is a
genuinely new layer: the intra-chain epistatic/fold-core network. Mechanistically this maps onto the *second*
failure mode — `sigma`-locking targets **stability/expression** (the collapsed `expression` in §3.2), whereas
interface-locking targets **registration/activity**. They should be **unioned**, not chosen between.

---

## 5. Unified constraint framework

### 5.1 Monomer lock ladder (cumulative; 301-residue mature monomer)

| Lock level | Contents | positions | % monomer | evidence source |
|---|---|---|---|---|
| `v1` | active-site pocket + 2nd shell (curated) | 24 | 8% | config |
| `v1 + R1+R2` | + Rosetta interface **hot spots** (AB registration + AD assembly) | 49 | 17% | Ala scan |
| `v1 + all interface` | + full AB∪AD interface (structural scaffold) | 137 | 46% | BSA/contacts + Ala scan |
| **`+ EV top-10%`** | ∪ conservation & coevolution (C∪sigma, pct≥0.90) | **171** | **57%** | EVcouplings |
| `+ EV top-15%` | ∪ pct≥0.85 | 189 | 63% | EVcouplings |

EV union sizes at other thresholds (∪ structural scaffold): top-25% → 218 (72%); top-5% → 155 (51%)
[reproducible from the EV csv]. The EV layer is *tunable*; "tight" spans 51% (top-5%) to 72% (top-25%).

### 5.2 Recommendation — subtractive experiment from a high lock

Because `v1` (lock 8%) is falsified by 46/46 designs, an *additive* search from v1 is unpromising. Start from a
**maximal** lock and **relax**:

- **L0 (start) = structural scaffold ∪ EV top-10% = 171 positions (57%)**, leaving ~130 (43%) free for
  de-immunisation. This matches the working hypothesis that >50% may need to be fixed.
- **Relax priority (peel first → last), by evidence strength:**
  1. EV-only positions with high `sigma`, low `C`, non-structural (indirect / possibly phylogenetic) — relax first.
  2. A:C diagonal / A:D non-hotspot interface.
  3. A:D hot spots / interface core.
  4. A:B catalytic face + R1 registration hot spots.
  5. **Never relax:** active site (`v1`) + high-conservation (`C`) + cross-subunit catalytic (Thr58\*, Lys11\*).
- **Fast screen before wet-lab:** at each rung, design → Protenix refold → read `xprot_Thr58_dev` /
  `min_pp_chain_pair_iptm` / active-site RMSD (all already computed by the gate). Wet-lab-test only the 2–3
  rungs bracketing where `xprot_*_dev` returns toward 0. The WT Protenix refold is a validated oracle (iptm 0.97,
  protein–protein chain-pair iptm 0.94–0.96, no clash) [`Results/Reference/protenix_wt_set0_AF/`].

### 5.3 A cheaper causal-isolation experiment

Take one order2 inactive design and **restore WT at interface positions in decreasing-ΔΔG order**, refolding at
each step; watch `xprot_Thr58_dev` collapse toward 0. This directly measures the *minimal interface WT set*
needed to recover native active-site geometry, without a full ladder.

---

## 6. Caveats / open questions

- **No experimental oligomeric-state data yet** (SEC/MALS/native-MS). In-silico metrics indicate H2
  (assembled-but-mis-registered) + stability loss; a "correct-MW tetramer that binds but is catalytically dead"
  has no clean precedent in the literature [synthesis §3] — this project could be the first to demonstrate it
  (retained inhibitor binding by ITC/Ki on an intact-SEC, inactive design ⇒ H2).
- **`sigma` hard-lock is conservative.** Coevolution reflects a *network*; a high-sigma position may tolerate
  substitution to a partner-compatible residue rather than strictly WT. Treated here as WT-lock for the maximal
  rung, and flagged as the first to relax.
- **EV MSA scope.** `sigma` here appears to derive from a monomer uricase-family MSA (intra-chain couplings),
  which is why it is disjoint from the inter-chain interface. A *complex* (concatenated) EVcouplings run would
  additionally recover inter-chain couplings and is worth checking against the Rosetta interface set.
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
```

Key hard numbers and their sources are tagged inline above; all derived from files under `Results/Reference/`,
`Results/Wetlab/order2/`, `viz/1R51/`, and the v1 config. Coevolution definitions grounded in the EVcouplings
framework (Hopf et al. 2019, *Bioinformatics*; Marks et al. 2011, *PLoS ONE*).

---

## 8. Round-1 experiment plan (activity rescue)

**Rationale.** No active redesign exists yet, so we start from a **heavy lock** and treat activity recovery as a
positive-control problem, then subtract in later rounds. Three signals are combined; **structure is held at its
maximum (full AB∪AD interface) in every union cell** because the evidence that interface breakage kills activity
is the strongest (§3.2: 46/46 true-`v1` designs assemble but drift and die). The **coevolution axis is titrated**
because its marginal value on top of full structure is the least-known quantity.

**Common core (all arms):** `v1` active site = 24 positions (index_0b listed in §2.2 / config). Every design is
de-immunised only on its *free* positions.

| Arm | Lock set | positions | % of 301 | free positions | n_seq | tests |
|---|---|---|---|---|---|---|
| **S** structure-only | `v1 ∪ interface(AB∪AD)` | 137 | 46% | coevolution-distinct + surface | 30 | registration/assembly hypothesis; also the **EV=0 anchor** of the union titration |
| **E** coevolution-only | `v1 ∪ EV(pct≥0.80)` | 124 | 41% | interface-distinct + surface | 30 | fold-network/stability hypothesis; **predicted-negative control** (frees the interface → expect order2-like drift) |
| **U57** union | `structure ∪ EV top-10% (pct≥0.90)` | 171 | 57% | surface only | 10 | union (min coevolution) |
| **U63** union | `structure ∪ EV top-15% (pct≥0.85)` | 189 | 63% | surface only | 10 | union (mid) |
| **U68** union | `structure ∪ EV top-20% (pct≥0.80)` | 205 | 68% | surface only | 10 | union (max) `= S ∪ E` |

Total = **90 designs**. Position sets are disjoint enough (§4.2, EV∩structure Jaccard 0.16) that S and E share
only the active-site core; their *distinctive* sets (81 interface-only vs 68 coevolution-only positions) are what
each arm actually probes.

**Titration structure.** S + U57 + U63 + U68 form a 4-point coevolution dose-response — **structure pinned at 137,
coevolution incremented** (EV = 0 → top-10% → top-15% → top-20%, i.e. +0/+34/+52/+68 positions over structure
alone: 46→57→63→68%). Comparing per-arm activity rate (out of 10–30) reads off how much coevolution locking is
needed once the full interface is fixed.

**Readout (round-1 decision table).**

| Outcome | Conclusion |
|---|---|
| some U active | heavy lock CAN rescue activity (positive control); **lowest active U level = coevolution requirement** |
| S active | full interface alone suffices → coevolution locking unnecessary → relax EV in round 2 |
| E active | coevolution substitutes for interface (unexpected) → overturns the registration hypothesis |
| all U dead | structure+coevolution insufficient → a critical position lies in the free surface, or cause is non-scaffold → rethink |

**Pre-wet-lab gate.** Refold every design (Protenix, same setup as the WT `set0_AF` oracle: iptm 0.97, protein–
protein chain-pair iptm 0.94–0.96, `xprot_*_dev` 0). Advance only designs with near-WT `xprot_Thr58/Lys11/His257_dev`,
active-site RMSD, and `min_pp_chain_pair_iptm`; this catches designs where a position outside the scaffold drifted,
independent of lock fraction.

**Open parameter.** The two EVcouplings selection configs chosen by Kaiyi map onto the EV-threshold column above
(top-10/15/20% = `sigma_pct/C_pct ≥ 0.90/0.85/0.80`); record their exact percentile/rank cutoffs here when confirmed
so the E arm and union titration use identical definitions.
