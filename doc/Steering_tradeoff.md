# D2 Trust-Region Logit Pipeline

**Status**: design contract (not implementation spec)
**Scope**: Phase D D2 logit correction, structure–head balance
**Related**: `doc/Reference_Flow_Derivation.md` §4.7 D2, `PLAN_RF.md` Task D2/D3, `report/Experimentresults0522.md` §14

---

## 1. Problem statement

D2 is already a structure-prior-to-immune-posterior projection (`counterfactual.py:177-296`). The scientific question is *how strongly* and *when* to project, not *whether*. The current v1 failure is end-to-end: 0% final-sequence persistence under both `d2_logits` and `d2_d3_full` modes, despite the projection being mathematically present at the sampling moment.

The failure decomposes into a cascading three-filter problem (numbers from `Experimentresults0522.md` §14):

| Filter | Rate | Bottleneck |
|---|---|---|
| `selected_after_d2_rate` | **7.0%** (174 / 2,490) | Bernoulli unmask schedule is decoupled from D2 refresh — D2 writes to masked positions, 93% don't unmask at the refresh step |
| `paired_disagreement_rate` | **6.9%** conditional (12 / 174) | Effective logit shift after $\rho_B$ multiplication is too small to flip the categorical sample (median actual shift 0.07 nat) |
| `final_persistence_rate` | **0.0%** (0 / 12) | Post-sampling rank face uses structural-only `ℓ_i` (full mode bug) or loses relative competition against high-confidence commits (d2_logits mode) |

Marginal product: 0% — D2 currently has no effect on final sequences.

The fix is a coherent pipeline, not isolated patches. Calling each piece "an alternative design" obscures the fact that the bottleneck stack requires all four faces.

---

## 2. The D2 Trust-Region Logit Pipeline

**One-line definition**: D2 constructs an immune-favored posterior within a structurally-acceptable candidate space, delivers the resulting logit shift as a sparse sticky field over the refresh interval, samples through the existing categorical machinery, and lets a multi-objective commit score decide whether the resulting token is worth keeping.

Four faces, one mechanism. Each face has its own temperature(s).

```
Proposal Face   →   Delivery Face   →   Sampling Face   →   Commit Face
(δ_struct, β,           (sticky TTL)         (unchanged)        (α, ν, λ)
 η, T_struct)
```

### 2.1 Proposal Face: structure-constrained posterior

Construct $Q_B^{\mathrm{safe}}$ as the structural softmax restricted to a structural trust region, then tilt by immune energy:

$$
\mathcal{A}_B^{\mathrm{safe}} = \{ a_B : \ell_i^{\mathrm{struct}}(a_i) > \ell_i^{\mathrm{struct,top-1}} - \delta_{\mathrm{struct}} \quad \forall i \in A_B \}
$$

$$
Q_B^{\mathrm{safe}}(a_B) \propto Q_B(a_B; T_{\mathrm{struct}}) \cdot \mathbf{1}[a_B \in \mathcal{A}_B^{\mathrm{safe}}]
$$

$$
\pi_B(a_B) \propto Q_B^{\mathrm{safe}}(a_B) \cdot \exp(-\beta \, \Delta R_B(a_B))
$$

$$
\delta \ell_i(a) = \eta \, \rho_B(t) \, \big[\log \pi_{B,i}(a) - \log Q_{B,i}^{\mathrm{safe}}(a)\big]
$$

**Required**: renormalize $Q_B^{\mathrm{safe}}$ and $\pi_B$ over $\mathcal{A}_B^{\mathrm{safe}}$ before the log-ratio. Do **not** sample from the unfiltered $Q_B$ and post-filter — this breaks the $\beta = 0$ no-op invariant. In `sampled` candidate mode, candidates must be drawn from $Q_B^{\mathrm{safe}}$ directly.

**Knobs**:

| Knob | Role | v1 default | Calibration anchor |
|---|---|---|---|
| $\delta_{\mathrm{struct}}$ | hard trust region; max structural cost per token | **1.5 nat** | Gap B median 0.9 nat (median D2 candidate passes; aggressive ones cut) |
| $\beta$ | immune energy strength | 1.0 (scalar) | Gap A median 0.55 nat at $\beta = 0.5$ in v1; doubling $\beta$ scales tilt accordingly |
| $\eta$ | logit projection trust | 0.7 (scalar) | actual shift after $\rho_B$ is 0.07 nat in v1; $\eta < 1$ keeps projection sub-full |
| $T_{\mathrm{struct}}$ | structural proposal sharpness | 1.0 (no sweep v1) | structural top-1 typical log-prob ≈ −0.7 nat |

$\delta_{\mathrm{struct}}$ is **required**, not optional. Without it, D2 is free to push into structurally catastrophic territory and any observed immune gain becomes inseparable from structural cost.

### 2.2 Delivery Face: sticky pending logit field

D2 computes $\delta \ell_i$ at refresh step $t_r$. Instead of writing only at $t_r$ and discarding, cache the delta in a per-position pending field with TTL = `refresh_interval`. At each subsequent step $t \in (t_r, t_r + K]$ the controller's pre-step hook adds the cached $\delta \ell_i$ to that step's freshly-computed structural logits for any position still masked.

**Cache lifecycle** (three independent termination conditions, all required):

| Condition | Action | Reason |
|---|---|---|
| Position $i$ unmasks (Bernoulli passes) at step $t$ | Cache for $i$ realized in the categorical draw, then cleared | The shift accomplished its purpose |
| Position $i$ is remasked by `reparam` after being unmasked | Cache for $i$ cleared | D2's offer was implicitly rejected by the rank face; don't keep re-pushing |
| Refresh step $t_{r+1}$ arrives | Cache for $i$ is overwritten by a fresh $\delta \ell_i$ if D2 still cares about $i$, or cleared if D2 no longer targets it | New posterior supersedes the old |

**What this fixes**: `selected_after_d2_rate` ceiling. Under v1 (no sticky), D2 only has the refresh-step Bernoulli draw to land — 7% per refresh. Under sticky, D2 has the cumulative unmask probability across $K = 5$ consecutive steps; for a mid-trajectory masked position with $g_i = 1$ this is typically $\sim 25 \text{–} 50\%$, multiplied by `paired_disagreement_rate` for actual token change.

**What this does not change**: Bernoulli rate schedule. Theorem 1 (Ordering Guarantee) holds because $\dot \kappa_i$ is untouched — D2 only influences *which token* is sampled at an unmask draw, not *when* the unmask draw passes.

**Knob**:

| Knob | Role | v1 default |
|---|---|---|
| `sticky_ttl_steps` | how many steps after refresh the cache stays live | 5 (= `refresh_interval`) |

### 2.3 Sampling Face: unchanged

`sampler.py:131` samples from the (now-stickied-when-applicable) corrected logits via the existing categorical machinery. `sampler.py:135` writes `scores[selected] = sampled_logp` — the chosen-token log-prob under the actually-used logits. No API change.

The Sampling Face is the *no-op face* by design. Both Metropolis-style accept/reject and direct block write-back are deferred (§6.B); they introduce sampler API changes whose cost is unwarranted given that the existing categorical sampler already does the right thing once Proposal and Delivery are correct.

### 2.4 Commit Face: multi-objective rank

The post-sampling rank face (used by `_apply_reparam_remask` to decide which committed positions to remask) becomes a linear combination of four sources:

$$
\mathrm{rank}_i = \alpha \, z(\ell_i^{\mathrm{struct}}) \;+\; (1 - \alpha) \, z(\ell_i^{\mathrm{sample}}) \;-\; \lambda \, z(m_i) \;+\; \nu \, z(d_i^{\mathrm{D2}})
$$

Sources:

- $\ell_i^{\mathrm{struct}}$: chosen-token log-prob under structural logits (= the v1 commit.py term)
- $\ell_i^{\mathrm{sample}}$: chosen-token log-prob under the logits that actually decided the sample (= `scores[i]`, already maintained by sampler)
- $m_i$: D3 EMA of excess immune risk (D3 mechanism, unchanged)
- $d_i^{\mathrm{D2}}$: D2 evidence per position, populated for positions that passed both `selected_after_d2` and `paired_disagreement` filters at any refresh in the trajectory; carries weight $\propto \mathrm{Gap}_A^{(i)}$ with TTL decay

**$\alpha$ is the trust-region knob on the rank face** (parallel to $\delta_{\mathrm{struct}}$ on the proposal face). $\alpha = 0$ → D2 tokens enjoy full credit for the corrected log-prob; $\alpha = 1$ → D2 tokens are judged purely on structural log-prob (v1 broken state); $\alpha = 0.3$ → 30% structural sanity retained.

**Mode applicability**:

- `d2_d3_full` mode: the formula above replaces `commit.py`'s current `compute_commit_score`. All four terms active.
- `d2_logits` mode (D3 off): $\lambda \, z(m_i)$ vanishes (no D3 state); the formula reduces to $\alpha \, z(\ell_i^{\mathrm{struct}}) + (1 - \alpha) \, z(\ell_i^{\mathrm{sample}}) + \nu \, z(d_i^{\mathrm{D2}})$. This is the rank face used by `_apply_reparam_remask` when `rank_scores` is supplied; in d2_logits mode the controller's `post_step()` must therefore now return non-`None` `rank_scores` (previously it returned `None` and reparam fell back to bare `scores[]`).

**Knobs**:

| Knob | Role | v1 default |
|---|---|---|
| $\alpha$ | structural sanity retention on rank face | 0.3 |
| $\nu$ | D2 evidence weight | 1.0 |
| `d2_evidence_ttl_steps` | how many steps D2 evidence persists | 5 |
| $\lambda$ | D3 immune-EMA weight (unchanged from v1) | 1.0 |

---

## 3. Implementation priority

Single-mechanism pipeline, but three priority tiers because faces have different dependency relationships.

### P1: persistence pipeline (must come first)

Delivery Face (sticky) + Sampling Face (existing) + Commit Face (multi-objective rank, $\nu > 0$). Without P1 the D2 signal never reaches the final sequence regardless of any other parameter sweep.

Why all three together: sticky alone increases `selected_after_d2_rate` but the landed tokens still get remasked by the broken commit ranking. The multi-objective rank alone protects D2 tokens that land, but with same-step rate at 7% there are barely any to protect.

### P2: structure trust region (part of main method)

Proposal Face with $\delta_{\mathrm{struct}}$. This is *not* an ablation — without it, D2 is structurally unconstrained and any immune improvement is inseparable from potential structural cost.

### P3: time schedule (deferred)

$\beta(t)$, $\eta(t)$, $\alpha(t)$ as t-curves. Only consider after P1+P2 yields measurable persistence and a clear immune signal. The three-phase ramp (burn-in → edit window → freeze) is the natural target shape, but v1 keeps scalars to limit the hyperparameter surface.

---

## 4. Recommended v1 config

```yaml
d2:
  beta: 1.0
  eta: 0.7
  struct_temperature: 1.0      # T_struct, no sweep in v1
  delta_struct: 1.5            # nat; required, not ablation
  sticky_ttl_steps: 5          # = refresh_interval
  sticky_clear_on_selected: true
  sticky_clear_on_remask: true
  sticky_overwrite_on_refresh: true

d3:
  alpha_struct: 0.3            # structural sanity retention on rank face
  d2_evidence_nu: 1.0
  d2_evidence_ttl_steps: 5
  lambda_commit: 1.0           # unchanged from v1
```

Calibration anchors (from `Experimentresults0522.md` §14):

| Quantity | Value | Drives |
|---|---|---|
| Gap A median (logit shift on chosen token) | 0.55 nat | $\beta$, $\nu$ |
| Gap B median (structural punishment vs top-1) | 0.9 nat | $\delta_{\mathrm{struct}}$, $\alpha$ |
| $\sigma$(scores) across committed positions | 3 nat | $\alpha$, $\nu$ z-space conversion |

---

## 5. Process metrics and falsification

### 5.1 Canonical three-filter terminology

Use these terms consistently in code, telemetry, and reports:

| Term | Definition | Numerator / denominator | v1 baseline | v1 target (after pipeline) |
|---|---|---|---|---|
| `selected_after_d2_rate` | D2 event whose target position is unmasked within the cache TTL window | $N_{\mathrm{selected}}$ / $N_{\mathrm{D2\ events}}$ | 7.0% | **markedly higher than 7%** (sticky raises ceiling by factor ~K) |
| `paired_disagreement_rate` | Among selected, sampled token differs from paired structural counterfactual | $N_{\mathrm{disagree}}$ / $N_{\mathrm{selected}}$ | 6.9% (conditional) | similar (should not collapse) |
| `final_persistence_rate` | Among disagreements, D2-written token survives to final sequence | $N_{\mathrm{survived}}$ / $N_{\mathrm{disagree}}$ | 0% | **10–30%** |

Marginal product is the "D2 actually changed the final sequence" rate: under the pipeline, target ≥ 1% (vs 0% in v1). This sets the floor for evaluating whether immune/structure metrics are even worth interpreting.

### 5.2 Decision tree

After running the v1 config on the same 50-protein pilot:

1. If `selected_after_d2_rate` does not rise markedly above 7%: Delivery Face is broken; sticky cache lifecycle has a bug. Stop and inspect.
2. If `selected_after_d2_rate` rises but `final_persistence_rate` stays ≈ 0%: Commit Face is broken; $\alpha$ or $\nu$ is not effective. Verify rank_scores are being consumed in d2_logits mode and that Z-scoring uses the correct base population.
3. If all three filter rates rise and `final_persistence_rate` ≥ 10% but structure preservation drops below D3-only baseline: $\delta_{\mathrm{struct}}$ too loose. Tighten to 1.0 nat and retry.
4. If all three filter rates rise, `final_persistence_rate` ≥ 10%, structure preserved, but immune Pareto rate does not improve over D3-only: D2's posterior is firing but its directional information is not useful at this candidate-space size. Consider expanding M / top_K (separate concern from this pipeline).
5. If filter rates, structure, and immune all improve: pipeline succeeds. Proceed to multi-seed scale-up.

### 5.3 Falsification

The unified pipeline framework is falsified if **(1) and (4) both hold**: the engineering works (filter rates rise, structure preserved) but D2 adds < 5 percentage points to D3-only Pareto rate. In that case, D2's logit-layer mechanism does not carry net information beyond what D3's revisit captures, and D2 should be deprioritized to "revisit-only" form (§6.D).

---

## 6. Deferred ideas (appendix)

The unified pipeline above is the v1 main method. The following ideas are not part of it but are recorded here for future iteration.

### 6.A Time schedule (P3 of §3)

Three-phase $\beta(t)$, $\eta(t)$, $\alpha(t)$ ramp:
- Burn-in $t \in [0, t_{\mathrm{start}}]$: $\beta = 0$, structure-only
- Edit window $t \in [t_{\mathrm{start}}, t_{\mathrm{ramp\_off}}]$: $\beta$ at peak, D2 actively shapes posterior
- Freeze $t \in [t_{\mathrm{ramp\_off}}, 1]$: $\beta$ decays, D3 finalizes

The implicit denoising temperature (head reliability is low at both ends of $t$) motivates the shape. Considered after v1 pipeline shows measurable persistence.

### 6.B Post-sampling structural accept/reject

After D2 categorical sample, evaluate structural penalty and probabilistically revert to a paired structural sample (not argmax — preserves stochastic semantics):

$$
P(\mathrm{accept}) = \min\!\left(1, \exp(\beta_{\mathrm{AR}} \cdot \Delta \ell^{\mathrm{struct}}_i)\right)
$$

This is a structural veto filter, not strict Metropolis-Hastings (no proposal-ratio correction). Cleaner than the sticky+rank composition in principle, but requires `controller.post_step()` API extension to mutate `x_t`. Deferred unless the unified pipeline empirically falls short.

### 6.C Bernoulli rate coupling

At refresh step, raise $\dot \kappa_i$ for positions in active blocks so they unmask at higher per-step probability. This invades Theorem 1's protected schedule and changes the ordering interpretation. Lower priority than sticky delivery (§2.2) because sticky achieves similar footprint expansion without touching the rate schedule.

### 6.D Revisit-only D2 (fallback)

D2 doesn't push logits at all; only flags positions for D3 commit/revisit re-ranking. Loses the directional "which token" information from D2; keeps only the "where to revisit" signal. This is the natural fallback if the unified pipeline cannot demonstrate that the logit-layer directional information adds value beyond D3's revisit selection.

---

## 7. Out of scope

- C+B engineering implementation details (telemetry schema, dataclass layouts, decay function shapes): belongs in `PLAN_RF.md` Task D2 implementation contract once this design is approved.
- Multi-allele extension of $\beta$ and $\delta_{\mathrm{struct}}$: separate concern; depends on whether allele-specific risk fields are aggregated before or after the trust region filter.
- Sensitivity analysis of all five proposal-face knobs ($\delta_{\mathrm{struct}}$, $\beta$, $\eta$, $T_{\mathrm{struct}}$, $\rho_B$ gating): v1 fixes four to defaults and treats $\delta_{\mathrm{struct}}$ as the primary axis; full sweep deferred to v2.
- Theoretical analysis of whether sticky delivery preserves the time-marginal of the DFM reverse process: empirically the change is small (D2 only acts on positions that would unmask anyway under the natural Bernoulli draws), but a formal statement is deferred.
