# Position-Dependent Reference Flow: Mathematical Foundation

Working document. Derives the math behind our core contribution: position-dependent
reference flow for immunogenicity-aware inverse folding.

**Scope note**. This document now distinguishes three layers:

1. **Core mechanism**: static-prior reference flow (Tasks 0/A/B/C)  
   A fixed hotspot field defines the position-dependent schedule.
2. **Static conditioning axis**: the same fixed hotspot field can also enter the
   denoiser as an explicit condition.
3. **Optional adaptive controller**: online hotspot refresh, logit steering, and
   revisit/corrector policies. This is a higher-level extension, not required for the
   base reference-flow claim.

This split is deliberate. Following the lesson from canonicalized flow/diffusion
models, the main method should first stand on a **fixed external prior field**. Online
updates are useful, but they are an optional controller/stabilizer rather than the
foundational theorem object.

**Dependencies**: DFM (Campbell 2024), DFM (Gat 2024), MDLM (Sahoo 2024), DPLM (Wang 2024).

---

## 0. Notation & Convention

**Time**: follows DFM convention. $t \in [0,1]$, generation goes $0 \to 1$.

- $t=0$: all masked (source)
- $t=1$: clean data (target)
- Mapping to Architecture v2: $t_{\text{here}} = 1 - t_{\text{Arch}}$

**Symbols**:

| Symbol | Meaning |
|--------|---------|
| $x_1 = (x_1^1, \ldots, x_1^L)$ | clean protein sequence, $x_1^i \in \mathcal{V}$ |
| $m$ | mask token, $m \notin \mathcal{V}$ |
| $h_i$ | generic fixed per-residue hotspot score used by the base schedule |
| $h_i^{(0)}$ | static prior hotspot field, typically pre-computed from WT: $H(x^{WT})$ |
| $h_i^{\mathrm{dyn}}(t)$ | optional online hotspot estimate from the current partially generated state |
| $g: \mathbb{R} \to [1, \infty)$ | amplification function, monotonically increasing |
| $\kappa_{\text{base}}(t)$ | base schedule: $\kappa_{\text{base}}(0)=0$, $\kappa_{\text{base}}(1)=1$, increasing |
| $\kappa_i(t)$ | position-dependent schedule: $\kappa_i(t) = \kappa_{\text{base}}(t)^{g(h_i)}$ |
| $g_i$ | shorthand for $g(h_i)$ |
| $u_i^{\text{sched}}(t)$ | control field used by the scheduler (WHEN axis) |
| $u_i^{\text{cond}}(t)$ | control field used by the denoiser condition (WHAT axis) |
| $q_t^i(a)$ | denoiser token distribution $p_\theta(x_1^i=a \mid x_t, S, t, u_t^{\text{cond}})$ |
| $C(x_t, q_t)$ | completion operator that builds a temporary full sequence from a partial state |
| $r_i^-(t)$ | optional risk-aware remask/revisit rate for already-decided positions |

**Reference tags**: [Gat.Eq.X], [Campbell.Prop.X], [MDLM.Eq.X].

Unless stated otherwise, the completion operator $C(x_t, q_t)$ denotes any deterministic
or stochastic rule that fills masked positions using the current denoiser output to
produce a temporary full sequence $\bar{x}_t \in \mathcal{V}^L$ for external scoring.
Examples include argmax fill, ancestral sampling, or top-k/temperature sampling.

### 0.1 Layered method view

The current project should be read as four progressively stronger objects:

- **C1 (sampling-only)**: fixed prior schedule.  
  Use a frozen hotspot field $h^{(0)}$ only to define the reverse schedule. This is the
  cleanest ablation of the ordering mechanism.
- **C2 (retrain)**: fixed prior schedule written into the training distribution.  
  Same frozen field, but now it changes masking frequency during training and activates
  implicit weighting.
- **C3 (static conditioning)**: fixed prior field enters the denoiser explicitly.  
  This changes WHAT the model predicts, while the reference flow changes WHEN positions
  are decided.
- **Task D (optional adaptive controller)**: online hotspot refresh, logit steering,
  and/or revisit of already-decided positions.  
  This is not required for the core claim. It is an extension for handling newly emerged
  hotspots during generation.

Two guiding principles follow from this layering:

1. The **main theorem object** is the static-prior reference flow. We should not mix
   optional online control into the base mechanism.
2. The most natural adaptive controller for MHC-IF is **not** a learned rank head, but
   an external hotspot field controller built around the existing epitope head
   $H(\cdot)$.

---

## 1. Task 0: Emergent Ordering Guarantee

**Goal**: Prove that positions with higher $g_i$ (higher immunogenicity risk) are
unmasked later in the generative process.

### 1.1 Setup

Position-dependent conditional flow:

$$
p_{t|1}(x_t^i \mid x_1^i) = \kappa_i(t) \cdot \delta_{x_1^i}(x_t^i) + (1 - \kappa_i(t)) \cdot \delta_m(x_t^i)
$$

At generation time $t$, position $i$ is unmasked with probability $\kappa_i(t) = \kappa_{\text{base}}(t)^{g_i}$.

Since $\kappa_{\text{base}}(t) \in [0,1]$ and $g_i \geq 1$:
$$
g_i > g_j \implies \kappa_i(t) < \kappa_j(t) \quad \forall t \in (0,1)
$$

Higher-risk positions have lower probability of being unmasked at any intermediate time.

### 1.2 Expected unmasking time

**Definition**. Let $T_i$ be the first time position $i$ becomes unmasked in the generative CTMC. Its CDF is $P(T_i \leq t) = \kappa_i(t)$.

$$
\mathbb{E}[T_i] = \int_0^1 P(T_i > t) \, dt = \int_0^1 \big(1 - \kappa_{\text{base}}(t)^{g_i}\big) \, dt
$$

### 1.3 Theorem (Ordering guarantee)

**Theorem 1**. $\mathbb{E}[T_i]$ is strictly increasing in $g_i$ for any base schedule $\kappa_{\text{base}}$ with $\kappa_{\text{base}}(t) \in (0,1)$ on a set of positive measure.

**Proof**.
$$
\frac{\partial}{\partial g_i} \mathbb{E}[T_i] = -\int_0^1 \kappa_{\text{base}}(t)^{g_i} \ln \kappa_{\text{base}}(t) \, dt
$$

For $\kappa_{\text{base}}(t) \in (0,1)$: $\ln \kappa_{\text{base}}(t) < 0$, so the integrand $-\kappa_{\text{base}}(t)^{g_i} \ln \kappa_{\text{base}}(t) > 0$.

Therefore $\frac{\partial}{\partial g_i} \mathbb{E}[T_i] > 0$. $\square$

### 1.4 Why this holds for the unconditional process too

The total unmasking rate at a masked position $i$ is:

$$
r_i(t) = \sum_{j \in \mathcal{V}} R_t^i(m, j) = \frac{\dot{\kappa}_i(t)}{1 - \kappa_i(t)}
$$

**Corollary 1**. This rate depends only on $\kappa_i(t)$, not on the model $p_\theta$.

**Proof**. From [Gat.Eq.24] / [Campbell.Eq.32], the unconditional rate for masked $x_t^i = m$:

$$
R_t^i(m, j) = \frac{\dot{\kappa}_i(t)}{1 - \kappa_i(t)} \cdot p_\theta(x_1^i = j \mid x_t)
$$

Sum over $j \in \mathcal{V}$: $\sum_j p_\theta(x_1^i = j \mid x_t) = 1$, giving $r_i(t) = \frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)}$.

CTMC survival probability from $0$ to $t$:

$$
P(T_i > t) = \exp\!\Big({-\int_0^t \frac{\dot{\kappa}_i(s)}{1-\kappa_i(s)} ds}\Big) = \exp\!\big(\ln(1-\kappa_i(t))\big) = 1 - \kappa_i(t)
$$

Same as the conditional process. $\square$

**Implication**: the model's quality affects WHICH token is chosen at unmasking, but NOT WHEN unmasking happens. Ordering is a property of the schedule, not the model.

### 1.5 Quantitative example

Linear base schedule $\kappa_{\text{base}}(t) = t$:

$$
\mathbb{E}[T_i] = 1 - \frac{1}{g_i + 1} = \frac{g_i}{g_i + 1}
$$

| $g_i$ | $\mathbb{E}[T_i]$ | Biological interpretation |
|--------|--------------------|----|
| 1.0 | 0.500 | Non-hotspot: resolved at midpoint |
| 1.5 | 0.600 | Moderate risk |
| 2.0 | 0.667 | High risk: resolved in last third |
| 3.0 | 0.750 | Very high risk |
| 5.0 | 0.833 | Extreme hotspot: resolved in last ~17% |

### 1.6 Context Enrichment Lemma

**Goal**: Strengthen "hotspots decided late" $\rightarrow$ "hotspots decided in a more anchored context".

**Lemma 2 (Context Enrichment)**. At the random decision time $T_i$ of position $i$:

1. $\kappa_i(T_i) = U \sim \text{Uniform}(0,1)$ (probability integral transform)
2. For any other position $j$: $\kappa_j(T_i) = U^{g_j/g_i}$
3. $\mathbb{E}[\kappa_j(T_i)] = \dfrac{g_i}{g_i + g_j}$

**Proof**.

(1) Standard: if $X$ has CDF $F$, then $F(X) \sim \text{Uniform}(0,1)$.

(2) From $\kappa_j(t) = \kappa_{\text{base}}(t)^{g_j} = \big(\kappa_{\text{base}}(t)^{g_i}\big)^{g_j/g_i} = \kappa_i(t)^{g_j/g_i}$. Evaluate at $t = T_i$.

(3) $\mathbb{E}[U^a] = \int_0^1 u^a \, du = \frac{1}{a+1}$ for $a > 0$. Take $a = g_j/g_i$. $\square$

**Interpretation**: $\mathbb{E}[\kappa_j(T_i)]$ is the expected probability that position $j$ is already unmasked at the moment $i$'s identity is being decided.

Concrete values with $g_j = 1$ (non-hotspot):

| $g_i$ | $\mathbb{E}[\kappa_j(T_i)]$ | Fraction of non-hotspots unmasked at hotspot decision |
|--------|----|---|
| 2 | 2/3 ≈ 0.667 | majority |
| 3 | 3/4 = 0.750 | three-quarters |
| 5 | 5/6 ≈ 0.833 | five-sixths |

So at the moment a typical hotspot ($g_i = 2$–$3$) is decided, **2/3 to 3/4 of non-hotspot positions are already resolved**. Hotspot decisions are made conditioned on rich context.

### 1.7 Context quality (caveat)

Lemma 2 says hotspots see many resolved positions in their context, but not that these positions are *correctly* resolved. For the mechanism to help, unmasked non-hotspots must be close to structurally-correct amino acids.

This is plausible because non-hotspot positions have narrower conditional distributions under strong structural constraints, but it is an empirical assumption not captured by the lemma. Should be verified by measuring per-position entropy of $p_\theta(x_1^i | x_t)$ (connects to H5, §6).

**Status: COMPLETE**

---

## 2. Task A: Forward Process & Training Objective

**Goal**: Show that the position-dependent flow is a valid generative model with a well-defined training objective.

### 2.1 Position-dependent conditional flow

**Definition 2**. For position $i$ with hotspot score $h_i$:

$$
p_{t|1}(x_t^i \mid x_1^i) = \kappa_i(t) \cdot \delta_{x_1^i}(x_t^i) + (1-\kappa_i(t)) \cdot \delta_m(x_t^i) \qquad \kappa_i(t) = \kappa_{\text{base}}(t)^{g_i}
$$

Joint (positions independent given $x_1$):

$$
p_{t|1}(x_t \mid x_1, h) = \prod_{i=1}^L p_{t|1}(x_t^i \mid x_1^i)
$$

**Validity**: instance of [Gat.Eq.9] with per-position scheduler [Gat.Eq.8]. Boundary conditions: $p_{0|1} = \delta_m^{\otimes L}$, $p_{1|1} = \delta_{x_1}$. ✓

### 2.2 Conditional rate matrix

By [Campbell.Eq.31] / [Gat.Thm.3], the minimal-jump rate matrix for masking:

$$
R_t^{*,i}(m, x_1^i \mid x_1^i) = \frac{\dot{\kappa}_i(t)}{1 - \kappa_i(t)}
$$

All other entries zero. Substituting $\kappa_i(t) = \kappa_{\text{base}}(t)^{g_i}$:

$$
\dot{\kappa}_i(t) = g_i \cdot \kappa_{\text{base}}(t)^{g_i - 1} \cdot \dot{\kappa}_{\text{base}}(t)
$$

$$
R_t^{*,i} = \frac{g_i \cdot \kappa_{\text{base}}(t)^{g_i - 1} \cdot \dot{\kappa}_{\text{base}}(t)}{1 - \kappa_{\text{base}}(t)^{g_i}}
$$

**Regularity**: non-negative and finite for $t \in (0,1)$ when $\kappa_{\text{base}}(t) \in (0,1)$. Diverges as $t \to 1$ (standard; guarantees complete unmasking).

### 2.3 Unconditional rate matrix (model-parameterized)

By [Campbell.Prop.3.1] / [Gat.Thm.2]:

$$
R_t^{\theta,i}(m, j) = \frac{\dot{\kappa}_i(t)}{1 - \kappa_i(t)} \cdot p_\theta(x_1^i = j \mid x_t) \qquad j \in \mathcal{V}
$$

$$
R_t^{\theta,i}(v, j) = 0 \qquad v \neq m
$$

Only masked positions transition. Unmasked positions are absorbing.

### 2.4 NELBO

Following [Campbell.App.C.2.1] generalized to per-position $\kappa_i$:

$$
\mathcal{L}_{\text{NELBO}} = \mathbb{E}_{t, x_1, x_t}\!\left[\sum_{i=1}^L \mathbb{1}[x_t^i = m] \cdot \frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)} \cdot \big(-\log p_\theta(x_1^i \mid x_t)\big)\right]
$$

The weight $w_i(t) = \frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)}$ is per-position and per-time.

### 2.5 Unweighted CE as training objective

$$
\mathcal{L}_{\text{CE}} = \mathbb{E}_{x_1, t, x_t}\!\left[-\sum_{i: x_t^i = m} \log p_\theta(x_1^i \mid x_t)\right]
$$

**Proposition 2**. $\mathcal{L}_{\text{CE}}$ and $\mathcal{L}_{\text{NELBO}}$ have the same minimizer: $p_\theta^*(x_1^i \mid x_t) = p_{\text{true}}(x_1^i \mid x_t)$ for all $i, t, x_t$.

**Proof**. Both are weighted sums of per-position CE terms. Each term is independently minimized by the true conditional. Weights affect convergence speed but not the optimum. $\square$

Following [MDLM] and [Campbell]: use unweighted CE in practice.

### 2.6 Implicit weighting from position-dependent masking

The expected training weight for position $i$ (fraction of time it is masked):

$$
\bar{w}_i = \int_0^1 (1 - \kappa_{\text{base}}(t)^{g_i}) \, dt
$$

**Proposition 3**. $\bar{w}_i$ is strictly increasing in $g_i$.

Proof: same sign argument as Theorem 1.

For linear schedule: $\bar{w}_i = \frac{g_i}{g_i+1}$.

| $g_i$ | $\bar{w}_i$ | Relative to $g=1$ |
|--------|-------------|-----|
| 1.0 | 0.500 | 1.00× |
| 2.0 | 0.667 | 1.33× |
| 5.0 | 0.833 | 1.67× |

Hotspot positions automatically receive more training signal. No auxiliary loss needed.

**Status: COMPLETE** ✓

---

## 3. Task B: Train-Uniform / Sample-Dependent Gap

**Goal**: Determine if we can train with uniform schedule and deploy position-dependent schedule only at sampling time.

### 3.1 Problem statement

- **Training**: $\kappa_i^{\text{train}}(t) = \kappa_{\text{base}}(t)$ for all $i$ (uniform)
- **Sampling**: $\kappa_i^{\text{sample}}(t) = \kappa_{\text{base}}(t)^{g_i}$ (position-dependent)

Question: does the model trained under uniform masking produce correct outputs when sampling uses non-uniform masking patterns?

### 3.2 Why it should work (theoretical argument)

**[Gat.Prop.6]**: For masking source, the optimal denoiser $p_{1|t}^*(x^i \mid z)$ is independent of $t$. It depends only on the observed pattern $z$, not on which schedule produced $z$.

Therefore:
1. The optimal denoiser trained under ANY schedule gives the same function
2. At sampling time, the model sees a masking pattern $x_t$ and predicts $p_\theta(x_1^i \mid x_t)$
3. The prediction quality depends on whether the model has seen similar patterns during training

### 3.3 Where the gap arises (practical concern)

During training (uniform): at time $t$, each position is masked i.i.d. with probability $1 - \kappa_{\text{base}}(t)$.

During sampling (position-dependent): at time $t$, hotspot positions are more likely to be masked than non-hotspots. Under the current forward law, positions are still independent Bernoullis conditional on $t$; the shift is not a new correlation structure, but a **heterogeneous per-position marginal masking rate**.

However, the uniform training covers masking fractions from 0% to 100% (as $t$ varies). The position-dependent sampling at any $t$ produces patterns whose marginal masking fractions fall within this range. The difference is that different positions now occupy different points on that masking curve at the same global time.

### 3.4 Bound on distribution shift

Let $\mathcal{P}_{\text{train}}$ = distribution over masking patterns seen during training. Let $\mathcal{P}_{\text{sample}}$ = distribution at sampling time.

For a specific masking fraction $\rho$ (fraction masked):
- Training: each position masked independently with probability $\rho$
- Sampling: position $i$ masked with probability $1 - \kappa_{\text{base}}(t_\rho)^{g_i}$ where $t_\rho$ chosen to match overall fraction

Because both distributions are products of Bernoulli marginals, the KL divergence is:

$$
D_{\text{KL}}(\mathcal{P}_{\text{sample}} \| \mathcal{P}_{\text{train}}) = \sum_i \Big[(1-\kappa_i) \ln \frac{1-\kappa_i}{1-\kappa_{\text{base}}} + \kappa_i \ln \frac{\kappa_i}{\kappa_{\text{base}}}\Big]
$$

This is the exact sum of per-position binary KL divergences. For moderate $g_i$ (1-3) and typical $\kappa_{\text{base}} \in [0.3, 0.7]$, this is small.

### 3.5 Practical strategy

**Two-phase approach**:

1. **Phase 1 (cheap)**: Use existing uniform-trained model. Apply position-dependent schedule only at sampling time. Evaluate quality (scTM, immunogenicity metrics).
   - If quality is acceptable → done, no retraining needed.
   - Expected: works reasonably due to Prop. 6 (time-independent denoiser).
   
2. **Phase 2 (if needed)**: Retrain with position-dependent forward process. Benefits: model sees the exact distribution it will face at sampling time. Extra cost: one training run (~1-2 days on 4× A100).

### 3.6 Open question

Does position-dependent training (Phase 2) provide gains BEYOND the improved masking distribution? Specifically, does it improve the model's ability to generate diverse alternatives at hotspot positions (because it sees them masked more often)?

This is Task A's "implicit weighting" (Section 2.6) — only realized with retraining.

**Status: FRAMEWORK COMPLETE, awaiting experimental validation**

---

## 4. Task C: Sampling Algorithm

**Goal**: Concrete sampling procedure with position-dependent rates.

### 4.1 Per-position unmasking probability

At generation step $t$ with step size $\Delta t$, for masked position $i$:

$$
P(\text{unmask } i) = \min\!\Big(1,\; \frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)} \cdot \Delta t\Big)
$$

If unmasking: sample $x^i \sim p_\theta(x_1^i \mid x_t)$.

### 4.2 Step size constraint

For the Euler step to define a valid PMF [Gat.Eq.30]:

$$
\Delta t \leq \min_i \frac{1-\kappa_i(t)}{\dot{\kappa}_i(t)}
$$

For $\kappa_i(t) = \kappa_{\text{base}}(t)^{g_i}$, the binding constraint comes from the position with the LARGEST $\frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)}$ — which is the
$g_i = 1$ positions (non-hotspots) at most times.

**Why**: at early $t$, hotspot positions ($g_i > 1$) have near-zero rate, so they don't constrain the step. Near $t = 1$, L'Hôpital gives
$\frac{1-\kappa_i(t)}{\dot{\kappa}_i(t)} \to \frac{1-t}{1} = 1-t$ for all $g_i$ (with linear base).

**Conclusion**: position-dependent scheduling does NOT require smaller step sizes than the uniform case. Standard step count suffices.

### 4.3 Algorithm

```
ALGORITHM: Position-Dependent DFM Sampling

Input:  p_theta       -- trained denoiser
        kappa_base    -- base schedule function
        g             -- amplification per position [L]
        N             -- number of steps

Output: x             -- generated sequence [L]

1.  x <- [m, m, ..., m]                        // all masked
2.  FOR k = 0, 1, ..., N-1:
3.      t  = k / N
4.      dt = 1 / N
5.      logits = p_theta(x_1 | x, t)           // [L, |V|] model forward
6.      FOR each position i WHERE x[i] == m:
7.          rate_i = kappa_i'(t) / (1 - kappa_i(t))
8.          p_unmask = min(1, rate_i * dt)
9.          IF Bernoulli(p_unmask):
10.             x[i] ~ Categorical(logits[i])
11.     // unmasked positions unchanged
12. RETURN x
```

Lines 5-10 are the only difference from standard DFM sampling: `rate_i` varies by position instead of being uniform.

#### 4.3.1 Inference-time reparameterized refinement (non-DFM)

Our reference implementation (`inverse_folding/reference_flow/sampler.py`) supports an optional refinement layer modeled on DPLM's `reparam-uncond-deterministic-linear` decoding. After step $k$ (excluding the final step), the sampler optionally re-masks the bottom $\lceil n_{\text{committed}} \cdot (1 - (k+1)/N) \rceil$ committed positions ranked by chosen-token log-probability, allowing them to be resampled in subsequent steps.

This refinement **is not part of the DFM derivation**:

- The DFM forward chain is *absorbing*: residue → mask transitions never occur in the prescribed forward process, so the reverse chain prescribed by the rate matrix in §2.2 has no `vocab → mask` transitions either. Reparameterized re-masking introduces such transitions, which violates the strict Markov chain interpretation.
- However, the refinement is layered *on top of* the position-dependent unmasking schedule and **does not read $h_i$ or modify $g_i$**. The position-dependent schedule still governs *first*-unmask timing and remains the only place the hotspot signal enters the reverse process. This means the inductive bias derived in §1 (Emergent Ordering) is preserved.
- At $t=1$ the re-mask rate $1 - (k+1)/N$ goes to $0$, so the terminal distribution over fully unmasked sequences is unaffected.

**Recommended framing for evaluation**: Treat reparam refinement as inference-only, analogous to MaskGIT iterative MAP decoding, and report it as a sampling hyperparameter alongside step count and temperature. Pair main results with an ablation that disables refinement (`sampler.remask.enabled: false`) to isolate the contribution of position-dependent scheduling from the contribution of inference-time refinement.

### 4.4 With corrector steps (optional)

Following [Gat.Thm.4], corrector steps with per-position stochasticity $\eta_i$:

- Forward: re-mask position $i$ at rate $\eta_i$
- Backward: unmask at rate $\frac{\eta_i \kappa_i(t)}{1-\kappa_i(t)}$

Design choice: use higher $\eta_i$ at hotspot positions for more exploration. This is a secondary knob, not part of the core mechanism.

### 4.5 Compatibility with classifier-free guidance

CFG modifies $p_\theta$ but not the sampling schedule. The position-dependent rates $\frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)}$ are orthogonal to CFG. Both can be applied simultaneously:

$$
\tilde{p}(x_1^i \mid x_t) \propto p_\theta(x_1^i \mid x_t, h)^{1+w} \cdot p_\theta(x_1^i \mid x_t)^{-w}
$$

CFG controls HOW MUCH the model biases toward low-risk tokens. Position-dependent scheduling controls WHEN each position is resolved. These are independent axes.

**Status: COMPLETE** ✓

### 4.6 Static conditioning axis (C3 without adaptive control)

The same frozen hotspot prior that defines the schedule can also enter the denoiser as
an explicit conditioning signal:

$$
q_t^i(a) = p_\theta(x_1^i = a \mid x_t, S, t, h^{(0)})
$$

where $h^{(0)} = H(x^{WT})$ is a pre-computed per-residue hotspot field for the input
protein. In practice this can be injected via FiLM, additive features, Fourier features,
or attention bias. The exact architecture is implementation-dependent; mathematically it
is simply an additional conditioning variable.

This gives a clean separation of roles:

- **Reference flow / schedule**: controls **WHEN** positions are resolved
- **Static conditioning**: controls **WHAT** token distribution is predicted

Crucially, no auxiliary hotspot/rank prediction head is required for this core setup.
Unlike canonical rank in symmetry-based generation, hotspot scores already come from an
external function $H(\cdot)$ with a clear semantic meaning. A learned auxiliary head may
later be introduced as an amortized estimator for adaptive control, but it is **not**
part of the minimal formulation.

### 4.7 Optional adaptive controlled reverse process (Task D)

The fixed-prior method above is the core mechanism. However, during generation, newly
emerged hotspot regions may appear that were not salient in the WT prior. To handle this
we define an **optional controller** on top of the base reverse process.

#### D1. Online hotspot refresh

Let the current denoiser output be

$$
q_t^i(a) = p_\theta(x_1^i = a \mid x_t, S, t, u_t^{\text{cond}})
$$

Construct a temporary full sequence using a completion operator:

$$
\bar{x}_t = C(x_t, q_t)
$$

and evaluate the online hotspot field

$$
h_i^{\mathrm{dyn}}(t) = H_i(\bar{x}_t)
$$

This refresh need not happen at every step. A natural gated policy is:

$$
h^{\mathrm{dyn}}(t)\ \text{updates only if}\ t \ge t_{\mathrm{start}}
\ \text{and}\ k \equiv 0 \pmod K
$$

where $k$ is the reverse-step index. This treats online refresh as a stabilizer rather
than a mandatory part of the core sampler.

The role of $t_{\mathrm{start}}$ is reliability control. Early in reverse time,
$\bar{x}_t$ is dominated by local completion heuristics and need not lie close to the
protein manifold seen by the epitope head. In that regime $H(\bar{x}_t)$ may be a very
noisy proxy for the eventual sequence-level risk landscape. Later in reverse time,
$\bar{x}_t$ is typically more self-consistent and closer to the target manifold, so
$H(\bar{x}_t)$ becomes more trustworthy. Delayed activation at $t \ge t_{\mathrm{start}}$
is therefore justified as a confidence gate, not merely a heuristic.

In practice one may further strengthen this gate using a confidence term based on
remaining mask fraction, predictive entropy, or agreement across multiple completions.

#### D2. Split the controller into two axes

Define two control fields:

$$
u_i^{\text{cond}}(t) = (1-\beta_t)\, h_i^{(0)} + \beta_t\, h_i^{\mathrm{dyn}}(t)
$$

$$
u_i^{\text{sched}}(t) = (1-\alpha_t)\, h_i^{(0)} + \alpha_t\, h_i^{\mathrm{dyn}}(t)
$$

These play different roles:

- $u^{\text{cond}}$: WHAT-to-predict controller
- $u^{\text{sched}}$: WHEN-to-decide controller

This distinction should be preserved throughout the project. It prevents us from
overloading one mechanism to do two logically different jobs.

#### D3. Preferred first extension: dynamic commit/revisit control

The most natural adaptive extension is to keep the static schedule fixed and use the
online hotspot field to decide whether an already proposed residue should be considered
committed. In this view, the epitope head does not directly choose amino acids. It
controls the **editable state** of the reverse process.

Let $\ell_i(t)$ denote the denoiser confidence in the currently committed token at
position $i$, for example its chosen-token log-probability. Define an excess dynamic
risk term

$$
e_i(t) =
\mathrm{ReLU}\!\big(h_i^{\mathrm{dyn}}(t) - h_i^{(0)} - \tau\big)
$$

and a controller-adjusted commit score

$$
s_i^{\mathrm{commit}}(t) = \ell_i(t) - \lambda_t e_i(t).
$$

Generic reparameterized decoding revisits low-confidence residues. The risk-aware
controller revisits residues whose current identity is both insufficiently trusted by
the structural denoiser and associated with newly elevated immunogenic risk. Operationally,
the remask/revisit rule ranks committed residues by $s_i^{\mathrm{commit}}(t)$ rather
than by $\ell_i(t)$ alone.

This is different from external classifier guidance. The head is not used as a direct
token-level energy that tells the model which amino acid to sample. Instead, it decides
which residues should remain editable and lets the structure-conditioned denoiser
re-propose amino acids. The controller therefore acts on **commitment**, not directly on
the amino-acid distribution.

If the online risk estimate is noisy, a persistent risk memory can be used:

$$
m_i(t) = \gamma m_i(t-\Delta t) + (1-\gamma)e_i(t)
$$

and $m_i(t)$ replaces $e_i(t)$ in the commit score. This EMA-style memory prevents a
single noisy hotspot estimate from causing repeated remasking, while allowing persistent
newly emerged risk to keep a residue editable.

The residue-level field $h_i^{\mathrm{dyn}}(t)$ is already derived from window-level
epitope evidence by the head, so the first formulation should operate directly at the
residue level. Additional region/window smoothing can be treated as a later engineering
variant rather than part of the base controller.

#### D3b. Optional logit steering baseline

An external logit-steering controller can still be useful as a baseline:

$$
\tilde{q}_t^i(a) \propto q_t^i(a)\,
\exp\!\big(-\lambda_t \, \Delta \hat{R}_i(a; \bar{x}_t)\big),
$$

where $\Delta \hat{R}_i(a; \bar{x}_t)$ estimates the risk change if token $a$ were chosen
at position $i$. However, this is closer to conventional classifier guidance. It should
not be the primary generative mechanism unless the commit/revisit controller fails.

#### D4. Optional schedule modulation

If desired, the scheduler can also be updated online:

$$
\kappa_i(t; u^{\text{sched}}) =
\kappa_{\text{base}}(t)^{g(u_i^{\text{sched}}(t))}
$$

with corresponding hazard

$$
r_i^+(t) =
\frac{\partial_t \kappa_i(t; u^{\text{sched}})}{1-\kappa_i(t; u^{\text{sched}})}
$$

Important caveat: once $\kappa_i$ depends on the evolving state through$u_i^{\text{sched}}(t)$, Theorem 1 no longer applies directly. The original ordering guarantee relied on $\kappa_i(t)$ being a function of time alone, so that the first unmasking time had CDF $\kappa_i(t)$. Under dynamic schedule modulation, the process is
state-dependent and only a **conditional/local ordering interpretation** remains:
holding the current controller state fixed, larger $g(u_i^{\text{sched}}(t))$ still
delays local unmasking pressure. Recovering a global ordering statement would require
additional assumptions, for example eventual stabilization or monotonicity of the
controller state.

However, this should **not** be the default first controller. Dynamic scheduler updates
are more invasive, create additional train/sample mismatch concerns, and only affect
positions that are still masked.

#### D5. Why revisit/corrector matters more than dynamic scheduler

A dynamic scheduler alone cannot repair already-decided positions that become risky
later. Once a site has unmasked and absorbed a token, changing future hazards elsewhere
does not revisit that decision.

Therefore the principled mechanism for handling newly emerged hotspots is a risk-aware
commit/revisit process:

$$
r_i^-(t) = \eta_t \cdot s\!\big(h_i^{\mathrm{dyn}}(t), c_i(t)\big)
$$


where $c_i(t)$ may encode uncertainty, local risk increase, or another confidence score.
The exact choice of $s(\cdot)$ is a design decision, but conceptually:

- dynamic schedule controls **future masked sites**
- revisit/corrector controls **already-decided risky sites**

For MHC-IF, revisit is the more meaningful aggressive extension.

There is no theorem guaranteeing that arbitrary revisit policies converge to a stable
sequence. Convergence must be enforced by the sampling protocol. Sufficient practical
conditions are:

- finite reverse horizon,
- a revisit budget or revisit probability that decays to zero near $t=1$,
- a final hard-commit step with no further remasking,
- optional EMA/hysteresis so transient risk estimates do not cause oscillatory remasking.

Under these conditions, the process has operational convergence: every position is
eventually committed, although the final sequence is still a sample from the controlled
reverse process rather than the minimizer of an explicit energy.

One concrete high-level instantiation is to trigger revisit only when the online risk
rises above the static prior by a meaningful margin:

$$
\eta_t^i = \eta_0 \cdot
\mathrm{ReLU}\!\big(h_i^{\mathrm{dyn}}(t) - h_i^{(0)} - \tau\big)
$$

and then use $\eta_t^i$ inside the corrector/revisit mechanism. This makes revisit fire
only when the online controller detects a materially worse hotspot than expected from the
prior. In practice this risk term should be combined with a structural confidence term
such as denoiser log-probability or entropy. This makes structure and immunogenicity
interact through a transparent commit score rather than through an opaque fusion of
incompatible objectives.

If structure and immunogenicity conflict, the controller should expose the conflict
rather than hide it. High structural confidence plus high immune risk means the residue
is likely a real tradeoff and should only be revisited under a strong risk threshold.
Low structural confidence plus high immune risk is the preferred target: the model is
already uncertain, and the head indicates that keeping the residue committed is risky.

### 4.8 Recommended progression

The above discussion suggests a stable development order:

1. **Base method**: static prior schedule only (C1/C2)
2. **Second axis**: static hotspot conditioning (C3)
3. **First adaptive controller**: dynamic risk-aware commit/revisit with fixed schedule
4. **Optional comparator**: external logit steering
5. **Most invasive extension**: dynamic scheduler updates

This order matches the scientific need to keep the main mechanism interpretable:
reference flow first, adaptive controller second.

### 4.9 Training-side support for adaptive control

Adaptive control is not free: if the denoiser is trained only with static
$h^{(0)}$ but receives dynamic $u_t^{\text{cond}}$ at inference, then the controller is
feeding the model an out-of-distribution conditioning signal.

This issue is strongest when the dynamic field is fed **into the denoiser**:

$$
q_t^i(a) = p_\theta(x_1^i = a \mid x_t, S, t, u_t^{\text{cond}})
$$

In that case the training distribution should include at least mild corruption,
dropout, or mixing of the hotspot field, for example

$$
h^{\mathrm{corrupt}}(t) = h^{(0)} + \sigma(t)\,\epsilon,
\qquad \sigma(t) \propto (1-t)
$$

or blended conditions that occasionally replace the clean prior with a noisier proxy.
The purpose is not to simulate the exact online controller, but to expose the model to a
family of imperfect conditions whose reliability varies with time.

By contrast, if the adaptive controller acts only **outside** the denoiser, e.g. through
commit/revisit scoring or external logit reweighting

$$
\tilde{q}_t^i(a) \propto q_t^i(a)\,
\exp\!\big(-\lambda_t \, \Delta \hat{R}_i(a; \bar{x}_t)\big),
$$

then there is no conditioning OOD for the model itself; the challenge shifts from model
generalization to controller approximation error.

The recommended interpretation is therefore:

- **Static-prior schedule/conditioning**: main method, no extra training support needed
- **Dynamic condition fed into denoiser**: requires training-side robustness support
- **Dynamic commit/revisit**: no conditioning mismatch, but requires convergence gates and
  transparent structure-risk tradeoff reporting
- **External logit steering**: no conditioning mismatch, but closer to conventional guidance
- **Dynamic scheduler modulation**: additionally weakens the clean static-prior theorems
  and should be treated as a later-stage extension

---

## 5. Robustness to Hotspot Error

**Goal**: Characterize how errors in the predicted hotspot score $\hat{h}_i$ degrade the ordering mechanism. We use $\hat{g}_i = g(\hat{h}_i)$ from a frozen epitope head, not the true risk.

### 5.1 Magnitude sensitivity (R1)

Let $\hat{g}_i = g_i + \delta_i$. The effect on the schedule and expected unmasking time:

$$
\frac{\partial \kappa_i(t)}{\partial g_i} = \kappa_{\text{base}}(t)^{g_i} \cdot \ln \kappa_{\text{base}}(t) < 0
$$

$$
\frac{\partial \mathbb{E}[T_i]}{\partial g_i} = -\int_0^1 \kappa_{\text{base}}(t)^{g_i} \ln \kappa_{\text{base}}(t) \, dt > 0
$$

For linear base schedule $\kappa_{\text{base}}(t) = t$:

$$
\frac{\partial \mathbb{E}[T_i]}{\partial g_i} = \frac{1}{(g_i+1)^2}
$$

**Quantitative check**: at $g_i = 2$, perturbation $\delta = 0.5$ shifts $\mathbb{E}[T_i]$ by $\approx 0.056$ (from 0.667 to 0.723). Response is smooth in $g_i$, not step-like.

### 5.2 Ranking stability (R2)

Define the **ranking margin** $m_{ij} = |g_i - g_j|$.

**Proposition 4 (Ranking preservation)**. If prediction errors satisfy

$$
|\hat{g}_i - g_i| < \frac{m_{ij}}{2} \quad \text{and} \quad |\hat{g}_j - g_j| < \frac{m_{ij}}{2}
$$

then $\text{sign}(\hat{g}_i - \hat{g}_j) = \text{sign}(g_i - g_j)$, and the relative ordering of positions $i$ and $j$ in the generative process is preserved.

**Proof**: direct triangle inequality. $\square$

**Implication**: the mechanism tolerates prediction error up to half the ranking margin. For well-separated hotspots (large $m_{ij}$), the mechanism is robust. For positions of similar true risk, errors more readily flip their order, but the functional impact is
small (both positions already had similar $\mathbb{E}[T]$).

### 5.3 Shuffle ablation (prediction)

Replace $\hat{h}$ with a uniform random permutation $\hat{h}_{\pi(i)}$:
- $\hat{g}_{\pi(i)}$ is decoupled from true risk
- Averaged over $\pi$, there is no longer a systematic alignment between late-decided
  positions and truly high-risk positions
- The schedule remains heterogeneous within each run, but its heterogeneity is now
  unrelated to the biological target of interest

  **Prediction**: under shuffled hotspot map, the **systematic targeted advantage** of position-dependent flow should collapse toward the permutation baseline. Residual effects may remain due to generic heterogeneity or global entropy changes, but any
  advantage specifically attributable to aligning the schedule with true risk should substantially weaken.

### 5.4 Summary statement

The ordering mechanism is stable under the *ranking errors* of the hotspot score, degrading gracefully with prediction quality. Completely shuffling the hotspot map removes the alignment between schedule and true risk. The resulting system should lose
its targeted mechanistic advantage, providing a negative control even if some residual effects remain from generic schedule heterogeneity (see H3 in §6).

---

## 6. Testable Hypotheses (not theorems)

Tasks 0, A, B, C and §§1.6, 5 establish what can be PROVED from the scheduling math alone. The remaining scientific claims require EMPIRICAL validation. This section states each claim as a falsifiable hypothesis with theoretical basis, measurement,
and failure mode.

### H1: Edit localization

**Claim**: Position-dependent flow concentrates sequence edits at hotspot regions more than uniform-schedule baselines.

**Theoretical basis**: Context Enrichment Lemma (§1.6) — hotspots decided late in rich context allows alternative amino acids; non-hotspots decided early with tighter structural constraints favors WT-like recovery.

**Measurement**: edit distance from WT, stratified by hotspot status. Compute ratio

$$
\rho = \frac{\text{edits at hotspots}/N_{\text{hotspot}}}{\text{edits at non-hotspots}/N_{\text{non-hotspot}}}
$$

Predict $\rho > 1$ and increasing with $c$ (amplification strength).

**Failure mode**: if $\rho$ does not scale with $c$, the ordering mechanism is not driving edits preferentially — either the model ignores the position-dependent signal, or hotspot positions are structurally rigid and cannot accommodate alternatives.

### H2: Pareto dominance over baselines

**Claim**: At matched overall risk reduction, position-dependent flow disrupts non-hotspot regions less than uniform flow + post-hoc guidance.

**Theoretical basis**: the core practical value. Guidance applied to uniform flow must trade off against a one-size-fits-all distribution; position-dependent flow allocates exploration budget to positions that need it.

**Measurement**: (scTM, Δrisk) Pareto front comparison, position-dependent flow vs guidance-on-uniform baseline. Predict dominance in a non-trivial region of the front.

**Failure mode**: if no dominance, position-dependent flow gives no practical benefit over simpler methods. Significant negative result — would redirect the project toward guidance improvements rather than flow modifications.

### H3: Shuffle control collapses targeted advantage

**Claim**: Shuffling $\hat{h}$ to random positions collapses the targeted ordering-based advantage attributable to aligning the schedule with true risk.

**Theoretical basis**: §5.3 predicts loss of alignment between schedule and true risk. Therefore any targeted advantage attributable to the hotspot map should collapse toward the permutation baseline, though residual effects from generic heterogeneity may remain.

**Measurement**: compare (scTM, Δrisk) with real $\hat{h}$ vs permuted $\hat{h}_{\pi(i)}$, averaged over multiple permutations.

**Failure mode**: if shuffled version retains advantage, the benefit is NOT from position-specific targeting — likely from global entropy effects or schedule-induced regularization. This would invalidate the mechanism story even if H1/H2 pass.

### H4a / H4b: Retrain vs sample-only (two-sided)

**H4a**: Retraining with position-dependent forward process outperforms sample-only deployment.

**H4b**: Sample-only deployment achieves comparable performance to retrained model.

**Theoretical basis**: Prop. 6 in [Gat.2024] says optimal denoiser is schedule-independent → supports H4b. But practical denoisers are trained on uniform masking; implicit weighting (§2.6) only activates under retraining → supports H4a. Outcome depends on whether model capacity / calibration is the bottleneck.

**Measurement**: same evaluation pipeline, compare Phase 1 (sample-only) vs Phase 2 (retrained) on (scTM, Δrisk).

**Interpretation**:
- H4a confirmed → implicit weighting enters training distribution matters;
  motivates full Phase 2 training run
- H4b confirmed → uniform-trained model already has enough hotspot entropy;
  Phase 1 is sufficient (stronger simplicity claim, different paper narrative)

  Both outcomes publishable. H4b would be a cleaner story ("sampling-time scheduling alone is enough").

### H5: Hotspot entropy opens up after retraining

**Claim**: At hotspot positions, the retrained model's predictive entropy $H(p_\theta(x_1^i \mid x_t))$ exceeds the uniform-trained model's entropy at matched $x_t$ context.

**Theoretical basis**: implicit weighting (§2.6) — hotspots are masked more often in retraining, so the model encounters more alternatives at these positions, reducing over-confidence toward WT amino acids.

**Measurement**: for each test protein, compute per-position $H(p_\theta(\cdot \mid x_t))$ at hotspot vs non-hotspot positions, for both models, at matched $t$ and $x_t$. Predict the entropy gap $H_{\text{hotspot}} - H_{\text{non-hotspot}}$ widens after retraining.

**Role**: H5 is a **mechanistic upstream test** for H4a. If H4a confirmed but H5 fails, the retraining benefit comes from something other than the proposed mechanism (possibly numerical regularization or optimization dynamics). If H5 fails, H4a's theoretical explanation is undermined even if the empirical result stands.

**Failure mode**: entropy unchanged → retraining does not activate the "explore-at-hotspots" mechanism as intended.

### Summary

| Hypothesis | Tests | Failure implies |
|---|---|---|
| H1 | edit localization | ordering not driving generation |
| H2 | Pareto dominance | no advantage over guidance-on-uniform |
| H3 | shuffle control collapses targeted advantage | advantage is not position-specific |
| H4a/H4b | retrain vs sample-only | two-sided; both publishable |
| H5 | hotspot entropy gap | retraining gain not from implicit weighting |

H3 and H5 are the most mechanistically informative: they discriminate between "the method works for the reason we claim" and "the method works but for a different reason".

---

## 7. Design Choices (to be determined)

### 7.1 Amplification function $g(h_i)$

Requirements:
- $g: \mathbb{R} \to [1, \infty)$
- Monotonically increasing
- $g(h_i) = 1$ when $h_i$ indicates zero risk (non-hotspot positions evolve at base rate)

Candidates:

| Form | Expression | Properties |
|------|-----------|------------|
| **Linear clamp** | $g(h) = 1 + c \cdot \max(0, h - \mu)$ | Simple, interpretable $c$ |
| **Sigmoid** | $g(h) = 1 + c \cdot \sigma(\kappa(h-\mu))$ | Smooth, bounded $\in [1, 1+c]$ |
| **Power** | $g(h) = 1 + c \cdot (h/h_{\max})^p$ | Controllable nonlinearity |

Hyperparameters: $c$ (max amplification), $\mu$ (threshold), $\kappa$ or $p$ (sharpness).

Need to sweep $c$ experimentally. Architecture v2 used $c$ in the sigmoid form.

### 7.2 Base schedule $\kappa_{\text{base}}(t)$

Options from literature:
- Linear: $\kappa(t) = t$ [Campbell default]
- Cosine: $\kappa(t) = 1 - \cos(\frac{\pi}{2}t)$ [MDLM finding: often better]
- Cubic: $\kappa(t) = 3t^2 - 2t^3$ [Gat.Eq.33]

The choice interacts with $g_i$ through $\kappa_{\text{base}}(t)^{g_i}$.
Need to verify that the chosen schedule doesn't create numerical issues when raised to large powers.

### 7.3 Open design questions

1. What range of $g_i$ is biologically meaningful? (Need to map $h_i$ distribution to $g_i$ range)
2. Should $g(h_i)$ be learned or fixed? (Architecture doc assumes fixed/frozen)
3. Multi-allele: how to aggregate $h_i$ across alleles for population-level $g_i$?

---

## Appendix: Corrections to Architecture v2 Section 5

| Item | v2 statement | Correction |
|------|-------------|------------|
| Eq. for $\gamma_i(t)$ | Stated as ansatz | Now derived as instance of [Gat.Eq.8] per-position scheduler |
| Section 5.2 reverse rate | Informal formula | Corrected in Task A §2.3: rate is $\frac{\dot{\kappa}_i(t)}{1-\kappa_i(t)} \cdot p_\theta$ |
| Section 5.4 "validity" | Assertion-based | Replaced by Theorem 1 + Propositions 2-3 |
| "No auxiliary losses needed" | Correct but unjustified | Now justified by Proposition 3 (implicit weighting) |
| Time convention | Inconsistent with DFM literature | Mapping stated in §0 |
