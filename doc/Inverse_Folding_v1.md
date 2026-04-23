# Inverse Folding v1: Implementation

> Concrete implementation plan for building the IF pipeline, evaluation infrastructure,
> and initial conditioning validation. For high-level architecture and scientific
> motivation, see `Immune_Design_Architecture_v2.md`.

---

## 0. v1 Objectives and Boundaries

### v1 Objectives

1. **Unconditional IF baseline**: Train a working structure → sequence model based on
   DPLM v1 + GVP adapter. Verify it achieves reasonable structural fidelity (scTM > 0.8).
2. **Evaluation pipeline**: Build end-to-end evaluation infrastructure for scoring
   generated sequences on structural quality and immunogenicity.
3. **Classifier guidance validation (Level 2)**: Implement inference-time steering using
   the epitope head as classifier. Demonstrate that guidance reduces immunogenicity scores
   as measured by an **independent** external predictor (NetMHCIIpan).
4. ~~Level 3 comparison arm~~ — **DROPPED (2026-04-04)**. Focus shifted to core
   reference flow. Post-hoc filter (Level 1) is the only comparison baseline.

### v1 Boundaries (what is NOT in scope)

- Multi-allele epitope head (v1 uses single allele: HLA-DRB1\*07:01)
- Property-aware reference flow / position-dependent masking (Phase 2-3; see Architecture doc)
- Biased masking or FiLM conditioning training (Phase 2)
- Wet lab validation
- Pan-DR population weighting

### Success Criteria for v1

- [ ] Unconditional IF model generates sequences with scTM > 0.8 on ≥ 80% of test structures
- [ ] Evaluation pipeline runs end-to-end: PDB → generate → score (head + NetMHCIIpan + ESMFold)
- [ ] Classifier guidance produces measurable Δ risk (negative) on NetMHCIIpan, while scTM remains > 0.5
- [ ] Results are reproducible (seeded generation, deterministic evaluation)

---

## 1. Base Model: DPLM v1 + GVP Adapter

### 1.1 Why DPLM v1

Chosen for three reasons (see Architecture doc Section 9 for full rationale):

1. **Evolutionary prior**: ESM-2 backbone pre-trained on UniRef50 (~45M sequences).
   Strongest available sequence prior for exploring alternative amino acids at each position.
2. **Absorbing-state formulation**: Forward process masks tokens → reverse unmasks.
   Directly extensible to position-dependent masking in Phase 2-3.
3. **Proven conditioning ceiling**: CFP-Gen demonstrated 76% AAR on DPLM v1 via
   FiLM conditioning, confirming the architecture supports rich conditioning.

### 1.2 Architecture Summary

```
Input:  backbone structure S (N, Cα, C, O coordinates)
        noisy sequence x_t (partially masked)
        timestep t

GVP-Transformer Encoder (frozen):
  S → per-residue geometric embeddings [L, d_struct]

DPLM ESM-2 Backbone (frozen, 650M):
  x_t → sequence embeddings through 33 Transformer layers
  Last layer: cross-attention adapter injects structure embeddings
  Output: per-position logits over amino acid vocabulary [L, 20]

Training: only the cross-attention adapter is trainable.
```

### 1.3 Setup Steps

1. Clone `github.com/bytedance/dplm`, install `byprot` package
2. Download pre-trained checkpoint: `airkingbd/dplm_650m` from HuggingFace
3. Prepare CATH 4.3 dataset (their datamodule config exists: `cond_dplm_650m.yaml`)
4. Train GVP adapter on CATH using their config
5. Validate: check recovery rate and scTM on CATH test set

### 1.4 Known Limitations

- **No published IF checkpoint**: Must train the adapter ourselves (~1-2 days on 4× A100)
- **Classifier guidance not implemented in repo**: Must implement ourselves (Section 3)
- **GVP is transitional**: Adequate for v1 but will be upgraded in later phases to richer
  structure features (RBF distances, dihedrals, deeper integration). The GVP + last-layer
  adapter is the weakest part of the architecture — the pre-trained Transformer is what
  matters.

---

## 2. Evaluation Pipeline

The evaluation pipeline is the foundational infrastructure that persists across all phases.
Every future experiment (Phase 1-3, ablations, comparisons) runs through this pipeline.

### 2.1 Evaluation Dimensions

#### A. Structural Fidelity

For each generated sequence, predict its structure and compare to the input backbone:

| Metric | Tool | Threshold | Purpose |
|--------|------|-----------|---------|
| **scTM** | ESMFold → TM-align to input | > 0.5 (viable), > 0.8 (good) | Primary structural quality metric |
| **pLDDT** | ESMFold confidence | > 70 | Structure prediction confidence |
| **bb-RMSD** | TM-align | < 2.0 Å | Backbone deviation |
| **Foldability** | scTM > 0.5 | > 80% of designs | Pass rate |

#### B. Sequence Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Recovery** | % positions matching WT | IF benchmark comparison (not our target) |
| **Diversity** | Pairwise seq identity within N=50 samples | Generation diversity |
| **Mutation count** | # positions ≠ WT | Edit efficiency numerator |

#### C. Immunogenicity Assessment

**Two independent scorers** (breaking circular evaluation):

**Scorer 1: Our epitope head** (same allele used for conditioning)
- Per-residue hotspot $h_i$ on generated sequence
- Global risk $R$ on generated sequence
- Window logits for all (start, k) spans

**Scorer 2: NetMHCIIpan 4.3** (independent external predictor)
- Allele: HLA-DRB1\*07:01 (matching our head for consistency in v1)
- Sliding window: all k-mers for k ∈ [12, 25] over generated sequence
- Outputs per window: EL score, %Rank_EL
- Aggregated metrics:
  - **n_strong_binders**: count of windows with %Rank_EL < 2%
  - **n_weak_binders**: count of windows with %Rank_EL < 10%
  - **mean_best_rank**: average of top-5 %Rank_EL windows (captures worst-case risk)

**Comparison metrics** (conditioned vs unconditioned, or generated vs WT):

| Metric | Definition |
|--------|-----------|
| **Δ global risk (head)** | R(generated) − R(WT) |
| **Δ global risk (NetMHCIIpan)** | mean\_best\_rank(generated) − mean\_best\_rank(WT) |
| **Δ n_strong_binders** | n\_strong\_binders(generated) − n\_strong\_binders(WT) |
| **Hotspot reduction** | mean h_i reduction at originally top-10% hotspot positions |

#### D. Edit Efficiency (for targeted editing mode)

| Metric | Definition |
|--------|-----------|
| **Risk per mutation** | Δ risk / mutation_count |
| **Pareto dominance** | Does our method dominate baselines on the mutation-risk-scTM front? |

### 2.2 Test Protein Selection

Select ~50-100 test proteins from PDB (not in CATH training set) satisfying:
- Length 100-500 AA (practical for ESMFold and epitope head)
- X-ray resolution < 2.5 Å
- Single chain
- Have clear epitope hotspots (epitope head identifies ≥ 3 positions with h_i > median + 1σ)

Store as a fixed test set with pre-computed WT hotspot maps.

### 2.3 Evaluation Script Structure

```
scripts/evaluate_if.py

Inputs:
  --pdb_dir        : directory of test PDB files
  --generated_dir  : directory of generated FASTA files (one per PDB, N seqs each)
  --wt_fasta       : WT sequences for comparison
  --epitope_ckpt   : epitope head checkpoint
  --netmhciipan    : path to NetMHCIIpan binary
  --allele         : HLA-DRB1*07:01
  --output         : output directory for results

Outputs:
  structural_metrics.csv   : per-design scTM, pLDDT, RMSD, recovery
  immunogenicity_head.csv  : per-design h_i, R from epitope head
  immunogenicity_nmp.csv   : per-design NetMHCIIpan scores
  comparison.csv           : per-protein Δ metrics (generated vs WT)
  summary.json             : aggregate statistics
```

### 2.4 Existing Infrastructure to Reuse

| Component | Location | Adaptation needed |
|-----------|----------|-------------------|
| `InferencePredictor` | `epitope_head/inference/predictor.py` | None — use `predict_protein()` directly |
| `StandaloneRunner` | `epitope_head/data/netmhciipan_runner.py` | Minor: currently returns per-peptide scores; need aggregation wrapper |
| NetMHCIIpan 4.3 binary | `netMHCIIpan-4.3/` | Already installed |
| ESMFold | External dependency | Install `esm` package, write structure prediction wrapper |

---

## 3. Phase 1: Classifier Guidance (Level 2 Validation)

### 3.1 Goal

Demonstrate that steering IF generation with epitope hotspot scores reduces
immunogenicity as measured by an independent predictor, without any retraining.

This is the fastest path to validating: "Does epitope-aware steering of IF models work at all?"

### 3.2 Classifier Guidance Implementation

At each denoising step $t$ in DPLM's iterative unmasking:

1. **Predict clean sequence distribution**: $p_\theta(x_0 \mid x_t, S, t)$
   — standard DPLM forward pass, gives per-position logits over amino acids

2. **Sample candidate clean sequences**: Draw K candidates $\hat{x}_0^{(1)}, \dots, \hat{x}_0^{(K)}$
   from the predicted distribution

3. **Score each candidate**: Run epitope head on each candidate to get global risk $R(\hat{x}_0^{(k)})$

4. **Reweight**: Compute guided distribution by reweighting candidates:
   $$w_k \propto \exp(-\eta \cdot R(\hat{x}_0^{(k)}))$$
   where $\eta > 0$ is guidance strength (higher = more aggressive de-immunization)

5. **Select**: Sample next step from the reweighted candidates, or use the weighted
   average of their per-position logits to continue denoising

This follows ADFLIP's "training-free classifier guidance" pattern: the classifier
(epitope head) operates on predicted clean sequences, never on noisy intermediates.

### 3.3 Guidance Strength Sweep

Run the evaluation pipeline at multiple $\eta$ values (e.g., 0, 0.5, 1, 2, 5, 10) to
characterize the Pareto front:
- x-axis: scTM (structural quality)
- y-axis: Δ risk (immunogenicity reduction)
- Each point: mean over test proteins at a given $\eta$

$\eta = 0$ is the unconditioned baseline. Increasing $\eta$ should reduce risk at the
cost of structural quality. The slope and shape of this curve is the key Phase 1 result.

### 3.4 Computational Considerations

- Epitope head inference is fast (~50ms per protein on GPU)
- K candidates per step × N denoising steps = K × N epitope head calls per design
- With K=8, N=10: ~80 forward passes per design, ~4 seconds per protein
- For 100 test proteins × 50 designs each: ~5.5 hours total (parallelizable)

### 3.5 Expected Phase 1 Outcomes

**If guidance works (expected)**:

- NetMHCIIpan risk decreases with $\eta$ while scTM degrades gracefully
- Clear Pareto-dominant region vs unguided generation
- Validates the concept; provides baseline for Phase 2-3 comparison

**If guidance doesn't work well**:
- Possible cause: DPLM's logits at hotspot positions are too peaked (model is too
  confident in the WT-like amino acid, leaving no room for guidance to steer)
- This would actually strengthen the case for Phase 2-3: "inference-time guidance is
  insufficient because the model was never trained to explore alternatives at hotspots
  — we need to modify the generative process itself"

---

## 4. Level 3 Comparison Baseline — DROPPED

> **Decision (2026-04-04)**: Level 3 (DRAKES/DPO) dropped from v1 scope. Core focus
> shifted to position-dependent reference flow. DRAKES code available in `DRAKES/`
> for potential reviewer response. Post-hoc filter (Level 1, implemented as N1)
> is the only comparison baseline.

---

## 5. Phase 2: Property-Aware Conditioning (Stub)

> To be detailed after Phase 1 validation. See Architecture doc Sections 4-6.

High-level plan:
- Fine-tune DPLM with biased masking ($p_i^{\text{mask}}(t) = \rho(t) \cdot [\epsilon + (1-\epsilon) \tilde{h}_i]$)
- Add $h_i$ as explicit conditioning signal (Fourier features + FiLM / additive)
- Classifier-free guidance dropout (10-20% of steps, drop $h$)
- Training data: PDB structures + pre-computed $h_i$ from frozen epitope head
- Evaluate: same pipeline as Phase 1, compare against Level 2 baseline

---

## 6. Phase 3: Position-Dependent Reference Flow (Stub)

> To be detailed after Phase 2. See Architecture doc Sections 3, 5.

High-level plan:
- Replace uniform masking schedule with position-dependent survival probability:
  $\gamma_i(t) = \gamma_{\text{base}}(t)^{g(h_i)}$
- Potentially upgrade structure encoder (richer features, deeper integration)
- Risk-aware attention biases: $b_{ij}^{\text{risk}} = f(h_i, h_j)$
- The emergent ordering principle: validate that generation order correlates with
  hotspot scores (non-hotspot first, hotspot last)

---

## 7. Execution Order

### Phase v1.0: Infrastructure (current priority)

1. Set up DPLM v1 + GVP adapter training environment
2. Train unconditional IF model on CATH
3. Validate IF quality (recovery, scTM on CATH test set)
4. Build evaluation pipeline (Section 2)
5. Select and pre-process test protein set
6. Pre-compute WT hotspot maps on test proteins

### Phase v1.1: Classifier Guidance

7. Implement classifier guidance in DPLM sampling loop (Section 3)
8. Run guidance strength sweep
9. Analyze Pareto front (scTM vs Δ risk)
10. Compare guided vs unguided generation

### Phase v1.2: Core Contribution — Position-Dependent Reference Flow

11. Tier 0: modify sampling loop for position-dependent unmasking rates (no retraining)
12. Tier 1: retrain DPLM with position-dependent forward process (κ_i(t) = κ_base(t)^g_i)
13. Tier 2: add explicit h_i conditioning (FiLM + CFG dropout)
14. Evaluate all tiers on same test set with same pipeline

### Phase v1.3: Analysis & Paper Assembly

15. Compare: Level 1 post-hoc filter vs Tier 0 (sampling-only) vs Tier 1/2 (full method)
16. Pareto front analysis: scTM vs Δrisk across methods
17. Write up results as internal report → paper draft

---

## 8. Dependencies and Environment

### Software

| Package | Version | Purpose |
|---------|---------|---------|
| `byprot` (DPLM) | from source | IF base model |
| `fair-esm` | ≥ 2.0 | ESM-2 backbone + ESMFold |
| `torch` | ≥ 2.2 | GPU compute |
| `pytorch_lightning` | 2.2.x | DPLM training |
| `hydra-core` | 1.2.x | DPLM config |
| NetMHCIIpan | 4.3 | External evaluation (already installed) |

### Compute

- IF adapter training: ~1-2 days on 4× A100 (CATH is small)
- Evaluation (ESMFold): ~30s per protein, parallelizable
- Classifier guidance sampling: ~4s per protein per design (K=8, N=10)
- Full evaluation run (100 proteins × 50 designs): ~6 hours on single GPU

### Data

| Dataset | Purpose | Status |
|---------|---------|--------|
| CATH 4.3 | IF adapter training | Download via DPLM scripts |
| PDB test set (~100 structures) | Evaluation | To be curated |
| Epitope head checkpoint | $h_i$ generation | Trained (HLA-DRB1\*07:01) |
| WT sequences for test proteins | Baseline comparison | Extract from PDB |
