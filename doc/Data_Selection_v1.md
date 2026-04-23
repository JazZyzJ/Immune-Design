# Data Selection v1: Evaluation & Benchmark Protein Curation

> Proposal for curating the protein test sets used across the Immune-Design project.
> Covers IF evaluation (Module L), guidance validation (Module M), comparison baselines
> (Module N), and future Phase 2-3 experiments.

---

## 0. Motivation

CATH test split is designed for fold-type diversity, not immunogenicity evaluation.
Directly using it creates two problems:

1. Many CATH proteins may have no meaningful DRB1\*07:01 epitopes — guidance has
   nothing to steer away from, making immunogenicity evaluation uninformative.
2. CATH entries are often domain fragments, not full proteins — they don't represent
   the therapeutic protein scenarios that anchor the paper's translational story.

A good test set must simultaneously answer:
- **Q1 (structural):** Does the IF model generate sequences that fold correctly?
- **Q2 (immunogenicity):** Does conditioning/guidance actually reduce epitope risk?
- **Q3 (independence):** Is the risk reduction confirmed by an external predictor, not
  just our own head?

These three questions require different properties from the test proteins, which is
why we use a tiered strategy rather than a single homogeneous set.

---

## 1. Three-Tier Test Set Design

### Tier 1: Gold Standard — Experimentally Validated Epitopes (5-10 proteins)

Proteins with **experimentally confirmed DRB1\*07:01-restricted epitopes** (T-cell
assay or eluted ligand data from IEDB). Epitope positions are wet-lab validated,
independent of any predictor.

**Purpose in the paper:**
- Strongest possible evidence — "our method reduces presentation risk at
  experimentally confirmed epitope positions"
- Ground truth for validating both the epitope head and the guidance effect

**Selection criteria:**
- Source: IEDB query for DRB1\*07:01 + positive T-cell assay or MAPPs/EL data
- Must have a high-resolution PDB structure (X-ray or cryo-EM, < 2.5 Å)
- Length 100-500 AA, single chain preferred
- At least 2-3 experimentally confirmed epitope regions per protein

**Overlap handling:**
- CATH training overlap: **exclude** (strict, based on sequence identity > 30%)
- Epitope head training overlap: **annotate but do not exclude**. The head is the
  guidance signal, not the evaluator. NetMHCIIpan provides independent validation.
  Report results split by overlap status to assess head generalization.

### Tier 2: Computationally Pre-screened, Structurally Diverse (30-50 proteins)

Proteins without experimental epitope data, but where **both the epitope head and
NetMHCIIpan independently agree** on significant DRB1\*07:01 presentation risk.

**Purpose in the paper:**
- Generalization evidence — "our method works across diverse fold types"
- Statistical power for Pareto front analysis and ablation studies

**Selection workflow:**
1. Start from a broad PDB candidate pool:
   - Single chain, 100-500 AA, X-ray < 2.5 Å, deposited before 2023
   - Exclude membrane proteins, disordered proteins (pLDDT < 70 if AF2-predicted)
2. Remove CATH training overlaps (MMseqs2 clustering, sequence identity > 30%)
3. NetMHCIIpan pre-screen (DRB1\*07:01):
   - Run sliding window k ∈ [12, 25] on each protein
   - Keep proteins with ≥ 5 windows at %Rank\_EL < 2% (meaningful epitope load)
4. Epitope head pre-screen:
   - Run `predict_protein()` on each surviving candidate
   - Keep proteins where global risk R is in the top 50% of candidates
5. Structural diversity sampling:
   - Assign CATH topology codes to surviving candidates
   - Sample 1-2 proteins per topology to maximize fold diversity
   - Ensure coverage of: all-alpha, all-beta, alpha/beta, alpha+beta, few-SS
6. Manual review of the final selection

### Tier 3: Therapeutic-Relevant Proteins (5-10 proteins, optional)

Real therapeutic proteins with known or suspected immunogenicity issues.

**Purpose in the paper:**
- Translational story — "our method is directly applicable to therapeutic protein
  de-immunization"
- Even without DRB1\*07:01-specific epitope data, NetMHCIIpan screening provides
  allele-specific predictions

**Candidate examples:**
- Antibody scaffolds (e.g., adalimumab framework — known immunogenicity)
- Enzyme therapeutics (e.g., asparaginase, streptokinase)
- Cytokines (e.g., interferon-beta, IL-2)
- Any biologic with published clinical immunogenicity data and available PDB structure

**Selection criteria:**
- Published PDB structure (or high-confidence AF2 prediction)
- At least some NetMHCIIpan DRB1\*07:01 signal (≥ 3 windows at %Rank < 5%)
- Literature evidence of clinical immunogenicity (for narrative value)

---

## 2. Overlap Policy

| Overlap type | Policy | Rationale |
|---|---|---|
| Test protein ∩ CATH training (IF model) | **Strict exclude** (seq id > 30%) | Fair IF evaluation |
| Test protein ∩ Epitope head training | **Annotate, split-report** | Head is guidance signal, not evaluator. NetMHCIIpan validates independently. Split reporting reveals head generalization quality |
| Tier 1 epitope data ∩ head training positives | **Annotate** | If the head was trained on these exact EL spans, its h_i is "seen" data. Flag for transparency |

---

## 3. Pre-computed Artifacts Per Test Protein

For each accepted test protein, precompute and store:

| Artifact | Source | Purpose |
|---|---|---|
| `wt_sequence` | PDB extraction | Baseline for mutation count, recovery |
| `backbone_structure` | PDB coordinates (N, Cα, C, O) | IF input, structural comparison reference |
| `wt_hotspot_map` | Epitope head `predict_protein()` | h_i for guidance conditioning |
| `wt_window_logits` | Epitope head (all windows) | Detailed per-window baseline |
| `wt_netmhciipan` | NetMHCIIpan 4.3 (DRB1\*07:01, k ∈ [12,25]) | Independent baseline scores |
| `cath_overlap_flag` | MMseqs2 vs CATH train | Overlap status |
| `head_train_overlap_flag` | Check against epitope head training manifest | Overlap status |
| `tier` | Manual assignment | 1 / 2 / 3 |
| `experimental_epitopes` | IEDB (Tier 1 only) | Ground truth positions |

Stored as: `outputs/if/test_set/test_proteins.parquet` + per-protein PDB/FASTA files.

### Tier 1 span extraction: authoritative source

The authoritative per-peptide EL span table used by `scripts/build_tier1_candidates.py` and
`inverse_folding.evaluation.tier1_builder.load_iedb_spans` is **`data/mhc_if_v2.tsv`** —
**not** `data/iedb_clean_v1.tsv`.

Verified on 2026-04-17: all 15 entries / 261 spans of the shipped `tier1_candidates.json`
(0701 set) reproduce bit-for-bit from `mhc_if_v2.tsv` via the filter below.
`iedb_clean_v1.tsv` is a strict subset and misses many Atlas-sourced rows
(e.g. only 1 of 42 spans for Q93099 / 1EYB_A).

Relevant columns in `mhc_if_v2.tsv`:

- `source` — `"iedb"` or `"atlas"`
- `alleles` — JSON list of `HLA-DRB1*XX:XX`; filter with exact string match
- `protein_accessions` — JSON list of UniProt IDs
- `peptide_position_info` — JSON list of `{protein_id, start, end}`; `start`/`end`
  are 1-based inclusive UniProt positions with `end - start + 1 == len(peptide_seq)`

Conversion to the Tier 1 JSON schema (0-based half-open absolute UniProt):

```python
entry = {
    "start_0b": int(it["start"]) - 1,
    "end_0b":   int(it["end"]),
    "peptide":  row["peptide_seq"],
}
# dedup per uniprot by (start_0b, end_0b, peptide)
```

The Tier 1 JSON uses `"iedb_ref": "IEDB"` for all spans regardless of whether the
original row is `source=iedb` or `source=atlas` — by convention (downstream treats
this as an opaque free-text label).

Other files for orientation, usually not the right source:

- `iedb_clean_v1.tsv` — IEDB-only subset; missing Atlas rows.
- `iedb_low_reso_clean.tsv` — low-resolution allele rows (`HLA-DRB1` without subtype).

---

## 4. Execution Plan

### Step 1: Tier 1 Curation (manual, ~1 day)

1. Query IEDB for DRB1\*07:01-restricted positive assays with source protein accessions
2. Cross-reference source proteins against PDB for available structures
3. Filter by length, resolution, chain count
4. Check CATH training overlap
5. Manual inspection: are the experimentally confirmed epitopes in structurally
   interesting (non-trivial) regions?
6. Target: 5-10 proteins with strong experimental epitope evidence

### Step 2: Tier 2 Automated Pre-screening (scripted, ~1 day compute)

1. Download PDB candidate pool (single-chain, 100-500 AA, < 2.5 Å)
2. Run MMseqs2 overlap filter against CATH training set
3. Batch NetMHCIIpan scoring (can reuse the SLURM batching pattern from Module J)
4. Batch epitope head scoring
5. CATH topology assignment and diversity sampling
6. Manual review of top candidates per topology
7. Target: 30-50 proteins covering ≥ 10 CATH topologies

### Step 3: Tier 3 Selection (manual, ~half day)

1. Literature search for therapeutic proteins with clinical immunogenicity data
2. Check PDB structure availability
3. Run NetMHCIIpan screen
4. Target: 5-10 therapeutically relevant proteins

### Step 4: Artifact Generation (scripted, ~1 day compute)

1. Extract WT sequences and backbone coordinates from PDB
2. Run epitope head on all test proteins → store hotspot maps
3. Run NetMHCIIpan on all test proteins → store window scores
4. Generate overlap flags
5. Assemble `test_proteins.parquet`

---

## 5. Reuse Across Project Phases

This test set is designed to persist across the entire project lifecycle:

| Phase | How the test set is used |
|---|---|
| **Module L** (eval pipeline) | Initial pipeline validation |
| **Module M** (classifier guidance) | Level 2 guidance sweep evaluation |
| **Module N** (comparison baselines) | Same proteins, same metrics, fair comparison |
| **Phase 2** (conditioned IF) | Same test set, compare against Phase 1 baselines |
| **Phase 3** (property-aware flow) | Same test set, compare against Phase 1-2 |
| **Paper figures** | Tier 1 for case studies, Tier 2 for statistics, Tier 3 for translational story |

The test set should be **frozen after curation**. If expansion is needed later
(e.g., multi-allele evaluation), new proteins are added as a separate cohort, not
mixed into the original set.

---

## 6. Scaling to Multi-Allele (Future)

When the epitope head supports multiple alleles, the test set strategy extends:

- Tier 1 expands: IEDB has epitope data for many DR alleles beyond DRB1\*07:01
- Tier 2 re-screening: NetMHCIIpan pre-screen runs across a panel of DR alleles
  with population weighting
- New metric: population-weighted risk reduction (not just single-allele)
- The current DRB1\*07:01-only test set becomes one slice of a broader panel

This is out of scope for v1 but the tiered structure is designed to accommodate it.

---

## 7. What This Document Does NOT Cover

- Evaluation metric definitions and pipeline implementation (see `Inverse_Folding_v1.md` Section 2)
- CATH data preparation for IF training (see `PLAN_IF.md` Module K)
- Epitope head training data curation (see `Epitope_Head_v1.md`)
