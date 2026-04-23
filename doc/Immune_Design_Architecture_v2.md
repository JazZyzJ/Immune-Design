# Immune-Design: Architecture & Formulation

> A structure-conditioned discrete flow model whose generative dynamics are shaped by the immunogenic risk landscape of the protein.

---

## 1. Core Thesis

Existing approaches to de-immunization treat immunogenicity as an **external constraint**
— generate first, filter/optimize later. This creates a fundamental mismatch: the generative model has no awareness of immune risk, so the "good" solutions must be found by brute-force search in the output space.

We propose the opposite: **embed the immunogenic risk landscape into the generative process itself.** The reference flow that defines how sequences are generated is position-dependent, with dynamics determined by per-residue epitope hotspot scores. The result is a model whose generative trajectories are intrinsically biased toward low-immunogenicity solutions,
without any post-hoc optimization.

---

## 2. Hierarchy of Immunogenicity Awareness

Our approach sits at the deepest level of integration:

| Level | Method | Where awareness lives |
|-------|--------|-----------------------|
| 1 | Generate + filter (ProteinMPNN → NetMHCIIpan) | Post-hoc, outside the model |
| 2 | Classifier guidance at inference | Sampling procedure, not the model |
| 3 | RL reward fine-tuning (DPO/PPO/DRAKES) | Model's output distribution (behavioral) |
| **4** | **Property-aware reference flow** | **Generative dynamics themselves (mechanistic)** |

Level 4 is the deepest **mechanistic integration** of immunogenicity awareness: the
property enters the generative dynamics themselves rather than only the output
distribution or inference-time sampler. In that sense it is more principled as a core
modeling story than Level 3. However, RL or guidance can still serve as useful baselines,
diagnostic tools, or hybrid extensions in practice. If such add-ons are needed for best
performance, that weakens the claim that reference flow alone is sufficient, but it does
not invalidate the reference-flow idea itself.

---

## 3. The Emergent Ordering Principle

This is the central scientific insight of the work.

### Position-Dependent Reference Flow

In standard discrete flow matching, every position evolves from mask to prediction at the
same rate. We replace this with a **position-dependent rate** determined by the
immunogenicity landscape:

$$
\lambda_i(t) = \lambda_{\text{base}}(t) \cdot g(h_i)
$$

where $h_i$ is the per-residue hotspot score from the epitope head, and $g: \mathbb{R}
\to [1, 1+c]$ is a monotonic mapping (e.g., $g(h) = 1 + c \cdot \sigma(\kappa(h - \mu))$).

High-risk positions ($h_i$ large) have **faster corruption rates** in the forward
process — their sequence information is destroyed earlier and more completely.

### What Emerges in the Reverse (Generative) Direction

In generation, the process runs backward. This creates a natural ordering:

1. **Non-hotspot positions are resolved first.** Their information was preserved longest
   in the forward direction, so the model has strong signal to recover them early.
   These are the structural/functional anchor residues.

2. **Hotspot positions are resolved last.** By this point, the model has already committed
   to the non-hotspot context — backbone-compatible, evolutionarily informed residue
   choices are locked in. The hotspot positions are then generated **conditioned on this
   full structural and sequence context**, with maximum freedom to explore
   immunogenicity-optimized alternatives.

This ordering is not hard-coded — it **emerges from the flow dynamics.** Unlike explicit
two-stage pipelines (fix anchors → inpaint hotspots), the ordering is soft, continuous,
and learned. The model discovers the optimal generation order through training on the
position-dependent reference flow.

### Why This Works Better Than Guidance or RL

- **Guidance** (Level 2) reweights a model that was never trained to explore alternatives at hotspot positions. The model's logits at hotspots may be too peaked around the WT sequence for guidance to meaningfully steer.
  
- **RL** (Level 3) trains the model to produce low-risk outputs, but the generative
  dynamics are unchanged — the model must "fight against" its own uniform-rate process to produce position-specific behavior.
  
- **Our approach** (Level 4) ensures that during training, the model has already learned
  that hotspot positions are "freely explorable" (they were almost always masked), while non-hotspot positions should be faithfully reconstructed. The asymmetry is baked into the training distribution, not imposed at inference.

> **$h_i$ is not a supervision target or a reward signal — it is a budget allocator for the generative dynamics.** It determines how much exploratory freedom each position receives during generation.

---

## 4. Architecture

Every architectural decision is motivated by the biology:

```
                     Immune-Design

  ┌──────────────────────────────────────────────┐
  │            Position-Dependent                 │
  │            Reference Flow                     │
  │   λ_i(t) = λ_base(t) · g(h_i)              │
  │                                              │
  │  ┌────────────┐   ┌───────────────────────┐  │
  │  │  Structure  │   │  Sequence Transformer  │  │
  │  │  Encoder    │──▶│  (evolutionary prior)  │  │
  │  │  (GNN)      │   │                       │  │
  │  │             │   │  Structure-aware attn  │  │
  │  │  local      │   │  + Risk-aware attn     │  │
  │  │  geometry   │   │  bias                  │  │
  │  └────────────┘   └───────────────────────┘  │
  │                                              │
  │  ┌────────────┐                              │
  │  │  Epitope   │   h_i → conditioning         │
  │  │  Hotspot   │   (Fourier + FiLM / add)     │
  │  └────────────┘                              │
  └──────────────────────────────────────────────┘
```

### 4.1 Structure Encoder: GNN

**Biological motivation:** A residue's amino acid identity is determined by its local
spatial environment — hydrogen bond networks (~3 Å), hydrophobic packing (~5 Å),
electrostatic interactions (~10 Å). This is inherently a graph problem.

- k-NN graph on Cα atoms
- Edge features: RBF distance encodings, backbone dihedrals, relative orientations
- SE(3)-invariant by construction (distance/angle-based features)
- Outputs per-residue structural features $v_i$ and pairwise features $e_{ij}$

### 4.2 Sequence Generator: Pre-trained Transformer

**Biological motivation:** De-immunization requires the model to explore diverse,
biologically plausible alternatives at hotspot positions. This exploration capacity
comes from understanding the evolutionary landscape of protein sequences — which amino acid combinations co-occur, how substitutions co-vary, what patterns are viable across protein families. Only large-scale pre-training on evolutionary data (tens of millions
of natural sequences) can provide this.

- Transformer backbone pre-trained on evolutionary sequence data (e.g., UniRef50)
- Global self-attention enables long-range coordination of compensatory mutations
  (e.g., if a hydrophobic hotspot residue is changed, distant core-packing residues
  can adjust in the same generation step)

### 4.3 Structure-Aware + Risk-Aware Attention Bias

**Biological motivation:** Two types of inter-residue relationships matter for
immunogenicity-aware design:

1. **Structural proximity** — spatially close residues have coupled fitness effects
2. **Co-membership in epitope regions** — residues within the same MHC-II binding
   core (9-mer) or extended epitope (12–25 mer) must be jointly optimized

Both are encoded as attention biases:

$$
\text{attn}_{ij} = \text{softmax}\Big(\frac{q_i \cdot k_j}{\sqrt{d}} + b_{ij}^{\text{struct}} + b_{ij}^{\text{risk}}\Big)
$$

- $b_{ij}^{\text{struct}}$: from GNN pairwise features (spatial relationship)
- $b_{ij}^{\text{risk}}$: from hotspot interaction (e.g., $f(h_i, h_j)$ — when both
  positions are high-risk and within the same epitope window, increase their mutual
  attention to coordinate mutations)

### 4.4 Hotspot Conditioning

The per-residue hotspot score $h_i$ enters the model through two pathways:

1. **Reference flow** (Section 3): determines the position-dependent corruption/generation
   rate — this is the mechanistic pathway
2. **Explicit conditioning signal**: $h_i$ is encoded (e.g., Fourier features) and
   injected into the Transformer via FiLM-style modulation or additive features —
   this allows the model to explicitly "read" the risk landscape when making predictions

Both pathways are necessary: the reference flow creates the emergent ordering, while
explicit conditioning allows the model to make risk-informed choices at each position.

---

## 5. Formulation

### 5.1 Forward Process (Reference Flow)

Given a clean protein sequence $x_0 = (x_{0,1}, \dots, x_{0,L})$ over amino acid
vocabulary $\mathcal{V} \cup \{m\}$ (with mask token $m$), and pre-computed hotspot
scores $h = (h_1, \dots, h_L)$:

The conditional forward flow interpolates from data to mask with position-dependent speed:

$$
p_{t|0}(x_t^i \mid x_0^i, h_i) = \gamma_i(t) \cdot \delta(x_0^i) + (1 - \gamma_i(t)) \cdot \delta(m)
$$

where $\gamma_i(t)$ is the position-dependent survival probability:

$$
\gamma_i(t) = \gamma_{\text{base}}(t)^{g(h_i)}
$$

- $\gamma_{\text{base}}(t)$: base schedule, monotonically decreasing from 1 to 0 over
  $t \in [0, 1]$
- $g(h_i) \geq 1$: amplification factor for high-risk positions

**Effect:** At any intermediate time $t$, high-risk positions have lower survival
probability (more likely to be masked) than low-risk positions. The entire sequence
converges to all-mask as $t \to 1$, but high-risk positions arrive there first.

### 5.2 Reverse Process (Generative Model)

The model parameterizes the reverse flow via a rate matrix:

$$
p_\theta(x_0^i \mid x_t, S, h, t) \quad \text{for each position } i
$$

Given the absorbing-state structure, the conditional rate has the form:

$$
R_t^i(x_t^i, j) = \mathbb{E}_{p_\theta(x_0^i \mid x_t, S, h, t)}
\Big[ \frac{\delta(x_0^i, j) \cdot \delta(x_t^i, m)}{1 - \gamma_i(t)} \Big]
$$

Only masked positions can transition, and they transition to the model's predicted clean
token. The rate denominator $1 - \gamma_i(t)$ is position-dependent, reflecting the
fact that different positions are at different "effective times" in their generative
trajectory.

### 5.3 Training Objective

Standard discrete flow matching cross-entropy:

$$
\mathcal{L} = \mathbb{E}_{x_0 \sim p_{\text{data}},\; t \sim U(0,1),\; x_t \sim p_{t|0}(\cdot \mid x_0, h)}
\Big[ - \sum_{i:\, x_t^i = m} \log p_\theta(x_0^i \mid x_t, S, h, t) \Big]
$$

Loss is computed only over masked positions. Because high-risk positions are masked more
often (by the position-dependent forward process), the model naturally receives more
training signal at these positions — learning richer conditional distributions over
alternatives.

**No auxiliary losses needed.** The position-dependent masking implicitly achieves:
- The effect of an "anchor loss" — low-risk positions are rarely masked, so the model
  learns to preserve them
- The effect of an "epitope-aware loss" — high-risk positions are frequently masked, so
  the model learns diverse reconstructions conditioned on structure

### 5.4 Mathematical Validity

The position-dependent forward process remains a well-defined probability flow:

- Each position's marginal $p_{t|0}(x_t^i \mid x_0^i, h_i)$ is a valid categorical
  distribution for all $t$
- Positions are conditionally independent given $x_0$ in the forward process
- The model $p_\theta$ predicts the joint distribution over $x_0$ given $x_t$, which
  can capture inter-position dependencies
- The cross-entropy loss provides an unbiased gradient estimator
- The framework is a direct instance of conditional discrete flow matching (Campbell et
  al., 2024) with a data-dependent reference flow — no additional theoretical machinery
  is required

**What we give up:** Closed-form $q(x_t \mid x_0)$ → ELBO on log-likelihood. This path
was already abandoned by modern discrete flow matching frameworks in favor of direct CE.

**What we gain:** A generative process whose dynamics are biologically meaningful — the
flow's geometry encodes immunogenic risk.

---

## 6. Training Pipeline

| Stage | What | Data | Trainable Components |
|-------|------|------|----------------------|
| **1. Evolutionary prior** | Learn protein sequence fitness landscape | UniRef50 (~45M seqs) | Transformer backbone |
| **2. Structure conditioning** | Learn structure → sequence mapping | PDB (~150K structures) | GNN encoder + structure adapter |
| **3. Property-aware flow** | Learn immunogenicity-aware generation | PDB + EL-derived $h_i$ | Reference flow rates, FiLM conditioning, risk-aware attention bias |

Stage 1-2 leverage existing pre-trained models (no need to train from scratch).
**Stage 3 is our contribution** — the transition from standard IF model to
property-aware generative model.

### Stage 3 Details

- Pre-compute $h_i$ for all training proteins using frozen epitope head
- Train with the position-dependent forward process (Section 5.1)
- Classifier-free guidance dropout: drop $h$ (set to null) 10-20% of training steps,
  enabling inference-time control over conditioning strength
- The GNN encoder and Transformer backbone can be frozen or fine-tuned with LoRA,
  depending on compute budget

---

## 7. Inference Modes

### Full Generation
Start from all-mask, generate via iterative unmasking with the learned reverse flow.
The emergent ordering naturally produces non-hotspot residues first, hotspot residues last.

### Targeted Editing
Fix known-good positions (e.g., active site, binding interface), mask only hotspot
regions, generate replacements. The position-dependent flow ensures the model has been
trained to handle exactly this asymmetric masking pattern.

### Controllable Strength
Using classifier-free guidance at inference:
$$
\log \tilde{p}(x_0^i) = (1 + w) \cdot \log p_\theta(x_0^i \mid S, h, t) - w \cdot \log p_\theta(x_0^i \mid S, t)
$$
$w$ controls conditioning strength: higher $w$ produces more aggressive de-immunization
at the cost of potentially reduced structural fidelity. This provides a continuous knob
for navigating the Pareto front.

---

## 8. The Teacher Corruption Concern

A valid concern (from v1 doc): if the forward process depends on $h(x_0)$ from our
epitope predictor, is the model learning to "denoise where the teacher says" rather
than learning genuine structural alternatives?

**Why this is manageable:**

1. $h_i$ only controls the **rate of corruption**, not the **target of reconstruction**.
   The reconstruction target is always the ground-truth sequence $x_0$. The model learns
   structure-compatible sequences, not teacher-compatible sequences.

2. The epitope head is **frozen** during Stage 3 training. There is no gradient flowing
   back through $h_i$ — no feedback loop between the predictor and the generator.

3. **Evaluation uses independent external predictors** (NetMHCIIpan, MixMHC2pred, etc.),
   not the epitope head. If risk reduction is confirmed by independent tools, the
   teacher-corruption concern is empirically resolved.

4. The classifier-free guidance mechanism means the model also learns the unconditional
   distribution $p(x \mid S)$. We can directly measure the **delta** between conditioned
   and unconditioned outputs to verify that conditioning is doing meaningful work.

---

## 9. Key Technical References

| Component | Reference | What we take |
|-----------|-----------|--------------|
| Discrete flow matching | Campbell et al., 2024 | Mathematical framework for rate-matrix parameterized flows |
| Pre-trained sequence Transformer | DPLM (Zheng et al., ICML 2024) | ESM2-architecture backbone + absorbing-state training |
| GNN structure encoder | ProteinMPNN (Dauparas et al., Science 2022) | k-NN graph, geometric edge features |
| Multi-signal conditioning | CFP-Gen (Yin et al., ICML 2025) | FiLM-style modulation on frozen DPLM backbone |
| Training-free classifier guidance | ADFLIP (Yi et al., ICML 2025) | Sample → score → reweight implementation |
| Epitope hotspot signal | Our epitope head (EL-supervised) | Per-residue $h_i$ via sliding-window aggregation |

---

## 10. What This Document Does NOT Cover

- **Implementation plan and evaluation pipeline**: See `Inverse_Folding_v1.md` for
  concrete phased execution, evaluation metrics, base model setup, and classifier
  guidance implementation
- **Epitope head architecture and training**: See `Epitope_Head_v1.md` and `epitope_head/`
- **Paper structure and wet lab plan**: See `MapOut.md` (Figures 5-5')
- **Comparison baselines**: ProteinMPNN + DRAKES/DPO as Level 3 comparison — see
  `Inverse_Folding_v1.md` Section 4
