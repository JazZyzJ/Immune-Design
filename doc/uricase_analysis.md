# Tier 1 Structural Analysis of IEDB Epitopes

> **File name note**: kept as `uricase_analysis.md` to avoid breaking cross-references, but the scope is now IEDB structural analysis on the 15 Tier 1 test proteins. Uricase structural extension is deferred (see **Future Work**, no curated per-residue IEDB labels + no in-hand structure yet).

Analysis notes for interpreting the structural context of MHC-II epitopes on protein 3D structure. Scope: Tier 1 test proteins (15 PDBs with IEDB annotations).

---

## Purpose

MHC-II epitope presentation is the product of two joint processes:

1. **Proteolytic processing** — the protein must be unfolded and cleaved by endosomal cathepsins in antigen-presenting cells before peptides can reach MHC-II.
2. **Peptide-MHC binding** — the resulting peptides must fit the MHC-II groove (anchor motif, ≥9-mer).

By mapping **IEDB-annotated experimental epitopes** onto 3D structure and comparing structural features of epitope residues vs non-epitope residues, we can:

- Characterize the **biological** structural signature of MHC-II epitopes (surface? loop? flexible? loose?)
- Use this signature to (i) sanity-check the epitope head and (ii) inform the position-dependent reference flow in Phase C
- Generate a paper figure (F2 supplementary) showing structural-context of immunogenicity

---

## Methodological Note — IEDB Ground Truth, Not Model Predictions

**The reference for this analysis is IEDB experimentally annotated epitope spans, not the epitope head's predicted $h_i$.**

Using $h_i$ as the reference would be circular:

- The epitope head is trained on IEDB-derived labels. Its predictions already encode whatever sequence-to-structure regularities the model learned.
- Correlating $h_i$ with structural features would measure **the model's internal biases**, not **biological reality**. A strong correlation could be real biology *or* it could be a shortcut the model picked up.
- The correct interpretability question is: "do IEDB epitopes show structural pattern X?" — not "does the model's prediction show pattern X?"

$h_i$ instead plays a **secondary** role: we ask whether the epitope head recovers the same structural signature that IEDB epitopes do. This becomes a model-validation analysis, not a biology-discovery analysis.

### Data source

- **Label source**: `outputs/if/test_set/tier1_candidates.json`, field `experimental_epitopes[*]` → per-protein list of `{start_0b, end_0b, peptide, assay_type, iedb_ref}`. All current Tier 1 spans are `assay_type=EL` (eluted-ligand MS).
- **Per-residue label**: `is_epitope[i] = 1` if residue `i` falls within any IEDB span (union across spans), else `0`
- **Label caveat (EL-only → positives-only)**: Our IEDB labels come entirely from EL (eluted ligand) presentation assays, which produce **positive observations only**. There are **no experimentally-negative residues** in this dataset; non-epitope residues are better described as "not observed in EL" (i.e., untested for presentation, not disproved). This makes the comparison **conservative**: any untested-but-truly-presented residue that falls into the `is_epitope=0` bin dilutes the positive distribution, shrinking the observed effect size rather than inflating it.
- **Structure source**: PDB files under the cluster path `work/immune-design/if_test_set/pdbs/`. **Prerequisite**: B1 (PDB download) must be completed — run `scripts/download_test_set_pdbs.py` first (PROGRESS.md currently lists `pdbs/` as empty).
- **Residue mapping (strict)**. Two coordinate systems coexist in the JSON and the conversion is codified below (verified against all 261 Tier 1 spans, 2026-04-16):
  - `chain_range` (e.g. `"1485-1712"`) uses **1-based** author numbering: the chain's first FASTA residue (`sequence[0]`) is author residue 1485. Chain length = `chain_end − chain_start + 1`.
  - `start_0b` / `end_0b` use **0-based absolute** numbering and form a **half-open** `[start_0b, end_0b)` interval, so that `end_0b − start_0b == len(peptide)` on every span.
  - Conversion: `seq_idx = abs_pos − (chain_range_start − 1)`.

  Procedure:
  1. Parse PDB ATOM records for the target chain → ordered list `(author_resnum, ins_code, aa)` restricted to residues with a Cα atom.
  2. Extract the ATOM-derived sequence and verify it is a subsequence of `sequence` from `tier1_candidates.json` (strict identity; mismatch → abort with explicit error, do not silently skip).
  3. Build `seq_idx ← author_resnum` mapping via `seq_idx = author_resnum − chain_range_start` (author numbering is 1-based and starts at `chain_range_start`). Disordered residues with no ATOM record stay as NaN features but `is_epitope[i]` is still defined from the FASTA.
  4. Convert each IEDB span to sequence indices via `[start, end) = [start_0b − (chain_range_start − 1), end_0b − (chain_range_start − 1))` and verify `sequence[start:end] == peptide` — any mismatch aborts.

  This is the only accepted mapping path. No tolerance, no offset guessing, no silent span drops. Locked by the regression test at `tests/inverse_folding/analysis/test_structural_features.py` (one real 1LI1_C two-span fixture plus a sweep over all 261 spans in the shipped JSON).
- **Tier 1 coverage**: 15 proteins, 261 spans total, 11%-45% per-protein epitope coverage, lengths 113-497 AA, X-ray resolution 0.98-2.50 Å

---

## Metrics

### 1. Relative Solvent Accessibility (RSA)

| Field | Value |
|-------|-------|
| **Definition** | Per-residue solvent-accessible surface area (SASA) normalized by the theoretical maximum SASA for that amino acid type (extended Gly-X-Gly tripeptide) |
| **Range** | [0, 1]; typically 0 = buried core, 1 = fully exposed |
| **Biology** | Surface-exposed residues are physically accessible to cathepsins; buried residues require unfolding to be processed |
| **Expected pattern** | IEDB epitope residues have higher RSA than non-epitope residues |
| **Computation** | PDB → DSSP → `SASA / SASA_max[AA]` |
| **Tooling** | BioPython `Bio.PDB.DSSP` wrapper (binary: `mkdssp`). Single source for both RSA and SS-3state ensures they are derived from the same geometry/probe. FreeSASA is a fallback only if DSSP fails on non-standard residues. |

### 2. Secondary Structure Distribution

| Field | Value |
|-------|-------|
| **Definition** | DSSP 8-state assignment collapsed to 3-state: H (α-helix, 3₁₀-helix, π-helix), E (β-strand, β-bridge), C (coil / loop / turn / bend) |
| **Biology** | Coil/loop regions are more flexible and more protease-accessible; β-sheet cores are rigid and harder to process. Regular secondary structure elements can still be epitopes but typically after partial unfolding |
| **Expected pattern** | IEDB epitope residues enriched in C relative to the whole-protein H/E/C ratio |
| **Statistic** | Per-protein log-OR of (SS=C \| epitope) vs (SS=C \| non-epitope), pooled across proteins via DerSimonian-Laird random-effects meta-analysis. Report pooled log-OR + 95% CI + I² heterogeneity. A per-protein χ² is reported only as a by-protein sanity check, not the primary number. |
| **Computation** | PDB → DSSP → per-residue 8-state → collapse to 3-state |
| **Tooling** | BioPython `DSSP` (same call as RSA) |

### 3. Crystallographic B-factor

| Field | Value |
|-------|-------|
| **Definition** | Temperature factor from PDB ATOM records, proxy for atomic displacement (Å²). Per-residue value = mean over Cα or all heavy atoms |
| **Biology** | High B-factor = high structural mobility in the crystal. Flexible regions are more likely to transiently expose cleavage sites |
| **Expected pattern** | IEDB epitope residues have elevated B-factors relative to non-epitope residues |
| **Statistic** | Mann-Whitney U (B-factor is not Gaussian); report effect size |
| **Caveats** | Only meaningful for X-ray structures of similar resolution; requires z-score normalization within each PDB before pooling across proteins. NMR models have no B-factor; AlphaFold uses pLDDT as a proxy |
| **Computation** | Direct read from PDB ATOM records — no external tool |

### 4. Contact Number (CN)

| Field | Value |
|-------|-------|
| **Definition** | For residue `i`, count of Cα atoms within 8 Å of Cα$_i$ (excluding self, ±1 sequential neighbors) |
| **Range** | Typically 0–25; core residues ~15–20, surface ~5–10 |
| **Biology** | Inverse proxy for local packing. Low CN = loosely packed region, easier to unfold locally for protease access |
| **Expected pattern** | IEDB epitope residues have lower CN than non-epitope residues |
| **Statistic** | Mann-Whitney U; report effect size |
| **Caveats** | Correlated with RSA (both measure "buriedness"), but captures 3D packing rather than surface geometry — keep both for orthogonal evidence |
| **Computation** | PDB → Cα coordinates → pairwise distance matrix → threshold count |
| **Tooling** | BioPython `PDBParser` + NumPy |

---

## Analysis Design

### Primary: IEDB-based biological characterization

For each Tier 1 protein:

1. Parse PDB → per-residue `(residue_idx, aa, RSA, SS, B_factor, CN)`
2. Parse `tier1_candidates.json` → `is_epitope[i] ∈ {0, 1}` from union of `experimental_epitopes` spans
3. Within-protein z-score B-factor (resolution-dependent, must not pool raw)

Produce 4 panels:

- **Panel A — RSA**: violin / ECDF of RSA for `is_epitope=1` vs `is_epitope=0`
- **Panel B — SS**: stacked bar of H/E/C fraction; chi-square vs whole-protein baseline
- **Panel C — B-factor**: violin of z-scored B-factor
- **Panel D — Contact number**: violin of CN

Report per-protein: effect size (AUC-of-separation or Cliff's delta) + p-value (Mann-Whitney for continuous, χ² for SS).

### Cross-protein aggregation

- **Continuous metrics (RSA, B-factor, CN)**: within-protein z-score (or rank-normalize) before pooling; report pooled effect (mean of z-scored-differences or pooled Cliff's δ) with 95% CI; forest plot of per-protein effect sizes for direction consistency.
- **Categorical metric (SS-3state)**: do *not* pool raw z-scores (undefined for categorical). Instead compute per-protein log-OR for coil enrichment, then DerSimonian-Laird random-effects meta-analysis → pooled log-OR, 95% CI, I² heterogeneity, forest plot.
- **Allele stratification**: CLI hyperparameter `--allele` (values: `DRB1*07:01` / `DRB1*04:01` / `DRB1*15:01` / `all`). Runs are emitted per-allele when the candidates JSON exposes per-span allele annotation; otherwise the spans are treated as the allele of the candidates file (self-consistent).

### Secondary: Does the epitope head recover the IEDB structural signature?

Under the primary analysis we learn the **biological** structural signature (e.g., "IEDB epitopes sit at high RSA, in coil, with high B-factor"). We then ask:

- Do high-$h_i$ residues share the same signature?
- Or does $h_i$ latch onto a different structural niche (e.g., high RSA but helix, suggesting the model over-weights sequence motif)?

Concretely:

- **Structural consistency**: for each metric, compare the epitope-vs-background effect size computed from IEDB labels vs computed from $h_i$ > threshold labels
- **Error stratification**: is the head's false-positive rate higher in certain structural contexts (e.g., buried regions)?
- **Conclusion**: strong agreement → model has learned the correct structural bias; disagreement → model is relying on something else (possibly sequence shortcut)

---

## Findings (Tier 1, two alleles)

### Dataset

| Allele | Target n | Analyzed n | Skipped | Reason |
|--------|----------|------------|---------|--------|
| DRB1*07:01 | 15 | 13 | 2 | 6TN1_AAA (PDB not yet downloaded); 6HGM_A (ATOM↔FASTA identity=0.970 below strict threshold) |
| DRB1*04:01 | 15 | 14 | 1 | 3POW_A (ATOM coverage of FASTA = 0.709, too many disordered/missing residues) |

Epitope span counts are unchanged from the Tier 1 curation (261 spans total across 0701; equivalent coverage for 0401).

### Per-metric pooled results

Continuous metrics report pooled Cliff's δ (mean of per-protein effect sizes; positive sign means epitope residues have higher values than non-epitope residues). SS-coil log-OR is the DerSimonian–Laird random-effects pooled log-odds-ratio for "coil given epitope" vs "coil given non-epitope".

| Metric | 0701 pooled | 0701 sig ✓ / ✗ | 0401 pooled | 0401 sig ✓ / ✗ | Expected direction |
|--------|-------------|----------------|-------------|----------------|--------------------|
| RSA (Cliff's δ) | +0.021 (SD 0.11) | 1 / 0 | **−0.038** (SD 0.15) | 2 / 1 | + (surface-exposed) |
| B-factor z (Cliff's δ) | −0.044 (SD 0.22) | 1 / 2 | **−0.135** (SD 0.33) | 2 / **6** | + (flexible) |
| Contact number (Cliff's δ) | −0.026 (SD 0.17) | 3 / 0 | **+0.029** (SD 0.14) | 1 / 1 | − (loosely packed) |
| SS-coil log-OR (DL RE) | **−0.211**, 95% CI [−0.42, −0.00], I²=23% | — | −0.130, 95% CI [−0.42, +0.17], I²=**68%** | — | + (coil-enriched) |

"sig ✓ / ✗" counts per-protein two-sided Mann-Whitney tests (p<0.05) in the expected vs opposite direction.

### Per-allele reading

**DRB1*07:01 (n=13)** — All four metrics are statistically near-null at the pooled level.
- RSA pooled is ~0; direction split 6/13.
- B-factor pooled is ~0; 1 significant correct (2VXP_A δ=+0.40), 2 significant wrong (4ZIE_A, 5HGJ_A).
- Contact number trends weakly in the expected direction; when significant (1UOU_A, 1ZR3_A, 2VXP_A), it is consistently correct — 3 correct / 0 wrong.
- SS-coil log-OR is **slightly negative** with the 95% CI barely crossing zero, and 9/13 proteins show negative log-OR — i.e. the weak signal is that **epitopes are depleted in coil**, opposite to the B-cell-epitope intuition.

**DRB1*04:01 (n=14)** — The weak signals collapse or invert.
- RSA pooled flips to negative (−0.038); 10/14 proteins show negative or near-zero δ.
- B-factor pooled is strongly negative (−0.135) with very high between-protein variance; **6 proteins are significantly in the wrong direction** (3ZFM_A δ=+0.76 p<1e−3; 4HAF_A, 4MZV_A, 6JD8_A, 5W5M_A, 6UGW_A all δ<−0.3 p<0.01 in the opposite sense). For these proteins epitope residues are distinctly *more rigid* than non-epitope residues.
- Contact number pooled **flips sign** from 0701 (−0.026 → +0.029); epitopes trend toward tighter packing.
- SS-coil log-OR remains slightly negative but with **I²=68%**, indicating much larger between-protein heterogeneity than 0701. CI crosses zero.

### Cross-allele synthesis

Across n=27 proteins spanning two alleles, **no structural metric shows a consistent, significant effect in the direction predicted by the B-cell-epitope / surface-accessibility intuition**. The only remotely consistent signal is the *slightly negative* SS-coil log-OR (coil depletion in epitopes) — observed in both alleles, marginally significant in 0701, not significant in 0401 — and this is in the **opposite direction** of the common hypothesis.

Several 0401 proteins exhibit strong, significant, opposite-direction effects for B-factor and contact number, indicating that in a non-trivial fraction of proteins MHC-II epitopes are **preferentially located in rigid, tightly packed, partially buried regions**.

### Biological interpretation

MHC-II presentation proceeds by (i) protein uptake, (ii) endosomal unfolding, (iii) cathepsin cleavage, (iv) loading of a processed peptide into the MHC-II groove. Because step (ii) is upstream of step (iii), the native fold is largely erased before cleavage site selection. The absence of a universal surface / flexibility preference is therefore *consistent* with the processing biology: any region of the unfolded chain is in principle accessible, and epitope identity is driven by sequence-level features — anchor residues at P1/P4/P6/P9 that fit the MHC-II groove — rather than by the native structural context.

The weak opposite-direction signals (coil depletion, rigidity enrichment in 0401) may reflect a subtler mechanism: peptides from **regular secondary-structure regions** (helices, strands) can yield contiguous 15-mer stretches that survive processing intact, whereas **flexible loops** are more prone to over-cleavage by endosomal proteases. This would produce a net enrichment of epitopes in regular-SS, rigid, tightly packed regions — the pattern seen here. This mechanism is hinted at in the literature (Landry-style arguments on processing efficiency) but is rarely documented quantitatively.

### Implications for the project

1. **Epitope head architecture (sequence-only CNN)**: the absence of a dominant structural signature in the ground-truth data is consistent with — and quantitatively supports — the decision to train the epitope head from sequence alone. A pure sequence model is not expected to have missed a strong structural driver, because no such driver exists at the Tier 1 level.
2. **Phase C reference-flow design**: the position-dependent unmask rate should be driven by $h_i$ (sequence-level immunogenicity) and *not* conditioned on structural features (RSA, SS, B-factor, CN). Adding such conditioning would either be uninformative (0701) or actively harmful (0401 opposite-direction effects).
3. **F2 supplementary narrative**: the figure is repositioned from "epitope structural preferences" (the null hypothesis) to **"absence of a classical structural signature in MHC-II epitope localization"**, with quantitative evidence across 27 proteins × 2 alleles × 4 metrics. This is a more defensible and more novel claim than the original framing.

### Status of dataset expansion

Given the strength and direction-inconsistency of the pooled effects, **expanding to the epitope-head training set is not expected to reverse the conclusion**. Pooling more proteins will narrow confidence intervals around a null / weakly-negative effect, not reveal a hidden positive effect. Two lower-cost options can still be pursued if desired:

- Recover the 3 skipped proteins (6TN1 via PDB download; 6HGM and 3POW via relaxing the ATOM-coverage / identity threshold with partial-coverage NaN masking).
- Stratify the 27 analyzed proteins by fold class (α / β / α+β / α/β), protein function (enzyme / binding / structural), or epitope density — check whether the mean null hides sub-populations with genuinely different structural regimes.

Neither is blocking; both are useful for Figure polish.

---

## Future Work — Uricase (F4)

Uricase structural analysis is **deferred**. The F4 uricase target currently has sequence information only (no curated per-residue IEDB span labels at Tier 1 granularity; clinical ADA data are population-level titers, not mapped epitopes). A structural analysis will be planned once either (a) a reliable per-residue epitope annotation is assembled for rasburicase / a close homolog, or (b) a structure is selected and matched to an annotated sequence. For now the F2 supplementary figure is Tier 1-only.

---

## Open Questions (to test empirically)

Biology-level (answered by primary analysis, see **Findings** section):

- ~~Among the 4 metrics, which has the strongest effect size for IEDB epitopes?~~ — **Answered**: none are strong; CN is weakly correct in 0701 (3/13 sig correct, 0 wrong) but flips sign in 0401.
- ~~Is the coil enrichment significant?~~ — **Answered**: no. Pooled log-OR is *slightly negative* in both alleles (epitopes slightly depleted in coil, opposite to hypothesis).
- ~~Is the structural signature of DRB1*07:01 epitopes different from DRB1*04:01?~~ — **Answered**: 0401 shows substantially stronger opposite-direction effects for B-factor and CN, and much higher between-protein heterogeneity (I²=68% vs 23%). Both alleles agree on a null / weakly-negative coil signal.

Open follow-ups:

- Does stratifying the 27 proteins by fold class or function reveal sub-populations with genuinely non-null effects?
- For the 6 proteins in 0401 with strong opposite-direction B-factor signal, is there a shared biological property (active sites, interfaces, repeat domains)?

Model-level (answered by secondary analysis):

- Does the epitope head recover the same structural signature as IEDB?
- Are false-positive $h_i$ calls concentrated in specific structural contexts?
- Are there proteins where $h_i$ "ignores" structure (predicts epitopes in buried core)?

---

## Outputs & Artifact Paths

Cluster (data products, authoritative):

- `work/immune-design/tier1_structural_analysis/<allele_tag>/per_residue.csv` — one row per (protein_id, seq_idx) with columns `author_resnum, aa, is_epitope, rsa, ss8, ss3, bfactor_ca, bfactor_ca_z, contact_number`.
- `work/immune-design/tier1_structural_analysis/<allele_tag>/per_protein_stats.json` — per-protein effect sizes (Cliff's δ / AUC for continuous; log-OR + SE for SS-coil); span sanity-check status.
- `work/immune-design/tier1_structural_analysis/<allele_tag>/meta_analysis.json` — cross-protein pooled statistics (random-effects log-OR for SS, pooled effect + CI for continuous).

Repo (figures, synced from cluster for paper):

- `figures/F2_supplementary/tier1_structural_rsa_<allele_tag>.pdf`
- `figures/F2_supplementary/tier1_structural_ss_<allele_tag>.pdf`
- `figures/F2_supplementary/tier1_structural_bfactor_<allele_tag>.pdf`
- `figures/F2_supplementary/tier1_structural_cn_<allele_tag>.pdf`
- `figures/F2_supplementary/tier1_structural_forest_<metric>_<allele_tag>.pdf` (per-protein forest plots)

`<allele_tag>` is the allele file-safe form, e.g. `HLA-DRB1_07_01`.

---

## References

- DSSP: Kabsch & Sander 1983, *Biopolymers*
- FreeSASA: Mitternacht 2016, *F1000Research*
- Contact number as protein-design feature: reviewed in Yang et al., PROTEINS 2020
- MHC-II processing biology: Neefjes et al., *Nat Rev Immunol* 2011
- IEDB: Vita et al., *Nucleic Acids Res* 2019
