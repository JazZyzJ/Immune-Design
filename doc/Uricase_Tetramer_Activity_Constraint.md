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
   - `sigma` = **coevolution / covariance** = a position's *marginal* coupling strength (a per-residue summary of
     the pairwise couplings J_ij). Top ECs correspond to **side-chain contacts / epistatic pairs** (Hopf/Marks;
     Marks 2011). **Correction (per Codex, adopted):** high `sigma` does **not** mean "must be WT". A Potts/DCA
     model expresses J_ij(a,b) = whether two residue *states* are compatible in the current background; a
     high-coupling position usually means "**can vary, but a partner must co-vary to compensate**" — a
     *compatibility* constraint, not an *identity* constraint. Decomposing the Potts statistical energy makes this
     precise: the **independent field** h_i(a) is the identity/conservation term (legitimate hard-lock), the
     **coupling** Σ_j J_ij(a,b_j) is the compatibility term (must be *scored*, not frozen). Only positions that are
     high-`sigma` **and** high-`C` are effectively must-WT (via the conservation channel). Therefore `sigma` alone
     can only **propose candidate coupled pairs**; it must **not** enter a hard-lock union. Proper use: (i) a Potts
     ΔE filter on candidate designs, (ii) the causal pair-swap test in §9.

### 4.2 Intersection (this session's key comparison)

| Comparison | Result | Reading |
|---|---|---|
| **C (conservation)** top-10% (30) vs active site v1 (24) | overlap-coef **0.58**; 23/30 already inside the structural scaffold | **C is largely redundant** — it re-derives the active site we already lock. |
| **sigma (coevolution)** top-10% (31) vs inter-chain interface (130) | only **4** overlap; **27/31 lie outside the structural scaffold** | **sigma is nearly orthogonal** — it captures **intra-chain fold-core contacts** our inter-chain interface analysis structurally cannot see. |
| EV top-10% (61) vs structural scaffold (137) | **Jaccard 0.16** (84% of the union is non-overlapping) | The two schemes are mostly complementary. |
| EV `active_site` flag (15) | 10 are high-C, **0 are high-sigma** | Catalytic residues are conserved but *not* coevolving (invariant ⇒ nothing to covary with). |

**Interpretation.** "Structural-contact" and "covariance" are orthogonal *in principle* and, empirically, overlap
only through the conservation channel. The **distinctive** evolutionary contribution is `sigma` (the intra-chain
epistatic/fold-core network), and it does map onto the *second* failure mode (stability/expression, §3.2). **But
`sigma` is a compatibility signal, not an identity signal (§4.1 correction), so it enters as a Potts ΔE score plus
the §9 causal test — not as a hard-lock.** The legitimate hard-lock signals are structure (geometry) and
conservation / field h_i (identity); see §5.1. (Note: this intersection table used the percentile column `C_pct`;
the constraint masks in §5/§8 use Kaiyi's absolute `C_i_nogap ≥ 0.5` conservation net — the orthogonality
conclusion for `sigma` is unchanged, and stronger, since the broader conservation net overlaps structure more.)

---

## 5. Unified constraint framework

### 5.1 Monomer lock ladder (cumulative; 301-residue mature monomer)

| Lock level | Contents | positions | % monomer | evidence source |
|---|---|---|---|---|
| `v1` | active-site pocket + 2nd shell (curated) | 24 | 8% | config |
| `v1 + R1+R2` | + Rosetta interface **hot spots** (AB registration + AD assembly) | 49 | 17% | Ala scan |
| `v1 + all interface` | + full AB∪AD interface (structural scaffold) | 137 | 46% | BSA/contacts + Ala scan |
| **`+ conservation`** | ∪ field-h_i net (`C_i_nogap ≥ 0.5`, 122 pos; +50 beyond structure) | **187** | **62%** | EVcouplings conservation |
| `(+ coevolution)` | **not a hard-lock** — `sigma` is a compatibility signal (§4.1); via Potts ΔE + §9 test | — | — | — |

**Corrected mask numbers (definition fixed).** Kaiyi's masks are `(C_i_nogap ≥ 0.5) OR (sigma_pct ≥ t)`:
**S0.90 = 142, S0.85 = 154, S0.80 = 166** (an earlier draft mis-used the percentile column `C_pct` and reported
61/118 — wrong). Conservation net alone = 122; **structure + conservation = 187 (62%) is the defensible
hard-lock**. Folding in the high-`sigma`/low-`C` positions would push structure∪S0.90 to 203 (67%); those marginal
**16 high-`sigma` low-`C` non-structural positions** [UniProt 34,37,54,77,83,102,128,132,148,165,175,190,206,215,
224,243] are **candidate coupled positions for §9, not hard-lock members**.

### 5.2 Recommendation — subtractive experiment from a high lock

Because `v1` (lock 8%) is falsified by 46/46 designs, an *additive* search from v1 is unpromising. Start from a
**maximal** lock and **relax**:

- **L0 (start) = structure + conservation hard-lock = 187 positions (62%)** (geometry + identity signals only;
  `sigma` excluded per §4.1), leaving 114 (38%) free for de-immunisation. This matches the working hypothesis
  that >50% may need to be fixed.
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
- **`sigma` is a compatibility, not identity, signal (central correction, §4.1/§9).** Coevolution reflects
  pairwise compatibility J_ij(a,b); a high-`sigma` position may vary if a partner co-varies. It is therefore
  **excluded from the hard-lock** and handled as a Potts ΔE score + the §9 causal pair-swap test. Hard-locking it
  to WT is a blunt over-constraint that conflates compatibility with identity and cannot explain *why* a design
  worked.
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

**Decision (round-1): pragmatic blunt lock.** Prioritise a fast positive control over mechanistic resolution.
Lock **all three signals together** — structure + conservation + coevolution — accepting that `sigma`-locking is a
*blunt over-constraint* (§4.1): it **cannot hurt activity** (WT is active), it only spends de-immunisation freedom
and cannot explain *why* a design worked. The principled field-vs-coupling decomposition (§4.1) and its causal test
(§9) are **deliberately deferred** to later rounds, gated on obtaining the Potts model / N_eff (§9). This is an
explicit "dump-and-lock" first shot to establish that *any* heavy lock rescues activity.

**Cells** (masks in `Results/Reference/Covariance/round1_lock_masks.csv`; EV = Kaiyi's `C_i_nogap ≥ 0.5 OR
sigma_pct ≥ t`; numbering §0; `wt_aa` verified == Q00511 at all 301 positions):

| Cell | Hard-lock set | positions | % of 301 | free | role |
|---|---|---|---|---|---|
| **S** | structure only (`v1 ∪ interface`) | 137 | 46% | conservation+coevolution+surface | control / lower anchor |
| **U67** | `structure ∪ EV(S0.90)` | 203 | 67% | 98 | union, min |
| **U71** | `structure ∪ EV(S0.85)` | 213 | 71% | 88 | union, mid |
| **U74** | `structure ∪ EV(S0.80)` | 222 | 74% | 79 | union, max (dump) |

Titration S→U67→U71→U74 spans 46→74% — structure pinned at max, conservation net (122) fixed, `sigma` threshold
loosened 0.90→0.80. Suggested split: **30 at the strict union U74** (the guarantee), **S = 15** (lower anchor), and
one looser union (**10–15**) to read how far the lock can drop before activity is lost. (The earlier corrected
draft's 187/203 "A/B" split — separating conservation from `sigma` — is retained in §9 as the *later* mechanistic
follow-up, not run in round-1.)

**Readout / gate.** Refold every design (Protenix, WT `set0_AF` oracle: iptm 0.97, chain-pair 0.94–0.96,
`xprot_*_dev` 0); advance only near-WT `xprot_Thr58/Lys11/His257_dev`, active-site RMSD, `min_pp_chain_pair_iptm`.
The **lowest active cell** sets the round-2 relax target and feeds the §5.2 subtractive ladder.

---

## 9. Coevolution: causal validation and required artifacts

**Why a separate test.** §8 A-vs-B asks whether `sigma`-locking helps in aggregate; it cannot say **why** (identity
vs compatibility): a `sigma`-free design that fails may have failed by making an *incompatible* combination, not by
leaving WT. Distinguishing them needs a design that is **non-WT but Potts-compatible** — which requires the pairwise
model, not the per-residue `sigma`.

**Required artifacts (currently missing — request from Kaiyi).** The per-residue score CSV is insufficient:
1. **Pairwise ECs / coupling scores** — the `CouplingScores.csv`-style table (CN/FN score per residue pair i,j).
2. **Potts model parameters** h_i(a) and J_ij(a,b) (e.g. plmc `.params`) — to compute sequence statistical energy
   E and per-substitution ΔE, and to build compatible / incompatible pairs.
3. **N_eff (effective sequence count) + MSA depth** — **gating check first**: if N_eff is low relative to length L,
   the couplings are unreliable and the whole coevolution premise (including `sigma`) is suspect.
4. **Traceable generation config** — the EVcouplings/plmc run YAML (MSA source, filters, λ regularisation, scoring),
   so masks and energies are reproducible. (Also missing from the current drop: S0.85 file, pairwise EC, Potts model.)

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
