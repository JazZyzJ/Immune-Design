# Reference Flow Adaptive Controller: Actuation and Unified Actionability

**Status**: design contract. Staged rollout in §2.9 — Stage A (actuation, §2.1–2.7) **done** (local gates passed, RAR 0003); Stage B (targeting field, §2.8) **demoted to a mechanism check**; Stage C (global-risk temperature, §2.8) **promoted to a co-primary first-order lever, run together with B**. See the **Empirical update** at the end of §2.9 (RAR 0005 / 0001).
**Scope**: Phase D adaptive controller — D2 actuation pipeline + unified GR × Phase C/D actionability field
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

## 2. Controller design

The controller has two layers, and they fix two different bottlenecks. §2.9 makes the ordering between them load-bearing:

- **Part I — Actuation pipeline (§2.1–2.7)**: whether an immune-steered token survives to the final sequence.
- **Part II — Unified controller (§2.8–2.9)**: where the controller should act, and how global pressure and the Phase C schedule consume one shared actionability field.

---

**Part I — Actuation pipeline (§2.1–2.7).**

**One-line definition**: D2 constructs an immune-favored posterior within a structurally-acceptable candidate space, delivers the resulting logit shift as a sparse sticky field over the refresh interval, samples through the existing categorical machinery, and lets a multi-objective commit score decide whether the resulting token is worth keeping.

Four faces, one mechanism. Each face has its own temperature(s). v1 intentionally does not add a separate hand-tuned "maneuverability" gate; structural freedom is expressed by the safe posterior itself.

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

Interpretation: $\delta_{\mathrm{struct}}$ is a **safety gate**, not a manually-defined "is this position editable?" gate. v1 should not add a separate `min_safe_support` / maneuverability threshold. If the structural context is already decisive, $Q_B^{\mathrm{safe}}$ collapses toward one token and the posterior log-ratio naturally becomes small; if the context leaves multiple structurally plausible paths, $Q_B^{\mathrm{safe}}$ exposes that freedom to the immune tilt. Degenerate candidate concentration should be handled through the existing ESS / $\rho_B$ reliability machinery, not through another binary gate.

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

**Required telemetry**: record `sticky_age_steps` for every selected D2 target, where age is the number of sampler steps between the refresh that created the cached $\delta \ell_i$ and the step where the position actually unmasked. Report `selected_after_d2_rate`, `paired_disagreement_rate`, realized benefit, and final persistence stratified by `sticky_age_steps`. v1 does **not** decay sticky deltas inside the TTL; the age-stratified telemetry decides whether a future decay term is needed.

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
- $d_i^{\mathrm{D2}}$: D2 evidence per position, populated only for positions that pass `selected_after_d2`, `paired_disagreement`, and realized-benefit checks at any refresh in the trajectory; carries weight $\propto \mathrm{Gap}_A^{(i)}$ with TTL decay

**Realized-benefit requirement.** A D2-written token is eligible for positive evidence only if the corrected branch improves the paired local head risk:

$$
\Delta R_{\mathrm{corrected}}^{\Omega(B)} < \Delta R_{\mathrm{uncorrected}}^{\Omega(B)}
$$

`paired_disagreement` alone is not enough: it proves that D2 changed the categorical sample, not that the change was useful. Events that disagree but fail the realized-benefit check remain in telemetry but do not receive $d_i^{\mathrm{D2}}$ rank credit.

**$\alpha$ is the trust-region knob on the rank face** (parallel to $\delta_{\mathrm{struct}}$ on the proposal face). $\alpha = 0$ → D2 tokens enjoy full credit for the corrected log-prob; $\alpha = 1$ → D2 tokens are judged purely on structural log-prob (v1 broken state); $\alpha = 0.3$ → 30% structural sanity retained.

**Mode applicability**:

- `d2_d3_full` mode: the formula above replaces `commit.py`'s current `compute_commit_score`. All four terms active.
- `d2_logits` mode (D3 off): $\lambda \, z(m_i)$ vanishes (no D3 state); the formula reduces to $\alpha \, z(\ell_i^{\mathrm{struct}}) + (1 - \alpha) \, z(\ell_i^{\mathrm{sample}}) + \nu \, z(d_i^{\mathrm{D2}})$. This is the rank face used by `_apply_reparam_remask` when `rank_scores` is supplied; in d2_logits mode the controller's `post_step()` must therefore now return non-`None` `rank_scores` (previously it returned `None` and reparam fell back to bare `scores[]`).

**Z-score population.** In v1, every $z(\cdot)$ term is standardized over all currently committed residues in the protein (`x_t != mask_token_id`), matching the global candidate pool used by `_apply_reparam_remask`. Positions with no D2 evidence carry $d_i^{\mathrm{D2}} = 0$; if a term has fewer than two eligible values or near-zero standard deviation, that term contributes zero, consistent with the existing D3 z-score fallback.

This choice is intentionally conservative: reparam makes one global remask decision, so the rank coordinates should also be global. The tradeoff is that sparse D2 evidence can be diluted when only a few active-block residues carry nonzero evidence. Record rank percentiles for D2-written positions under each component term and under the final combined score. Upgrade to a hybrid z-score only if all three conditions hold:

1. `selected_after_d2_rate` and `realized_benefit_rate` increase, proving D2 is landing useful tokens.
2. `final_persistence_rate` remains low because D2-written positions still fall below the remask cutoff.
3. The component telemetry shows $z(d_i^{\mathrm{D2}})$ is diluted globally while block-local evidence would rank those positions above the cutoff.

The hybrid fallback would keep $z(\ell_i^{\mathrm{struct}})$ and $z(\ell_i^{\mathrm{sample}})$ global, but compute $z(m_i)$ / $z(d_i^{\mathrm{D2}})$ on the active-block or nonzero-evidence population before embedding the result back into the full-length rank vector. Do not implement this fallback in v1.

**Knobs**:

| Knob | Role | v1 default |
|---|---|---|
| $\alpha$ | structural sanity retention on rank face | 0.3 |
| $\nu$ | D2 evidence weight | 1.0 |
| `d2_evidence_ttl_steps` | how many steps D2 evidence persists | 5 |
| `d2_evidence_requires_benefit` | require corrected branch to beat paired structural branch before assigning D2 evidence | true |
| $\lambda$ | D3 immune-EMA weight (unchanged from v1) | 1.0 |

### 2.5 Pareto-level temperature view

The knobs above should be interpreted as one Pareto controller, not as unrelated small parameters. D2 is balancing two objectives:

```text
stay on the structural prior  vs  move toward lower immune risk
```

At the proposal face, the main structure-immune temperature ratio is:

```text
immune pressure / structural pressure  ~=  beta relative to T_struct
```

- $T_{\mathrm{struct}}$ controls how soft the structural proposal is. Lower $T_{\mathrm{struct}}$ makes structure harder to move away from; higher $T_{\mathrm{struct}}$ exposes more plausible alternatives.
- $\beta$ controls how strongly the immune head tilts the safe structural proposal.
- $\delta_{\mathrm{struct}}$ is not a temperature; it is the hard Pareto safety boundary. Candidates outside this boundary are not part of the tradeoff.
- $\eta \cdot \rho_B$ is the projection / reliability throttle: it decides how much of the posterior preference is actually written into logits.
- $\alpha$ and $\nu$ are the post-sampling Pareto retention terms: they decide whether an immune-steered token gets enough rank credit to survive reparam while still retaining structural sanity.
- $\lambda$ is the D3 immune-risk pressure on already committed residues.

For v1, these remain separate fields because each one diagnoses a different failure mode in the three-filter bottleneck. The longer-term goal is to collapse part of this surface into a smaller controller-level temperature after Stage A proves that D2 can land and persist. That controller-level temperature is the global-risk scalar `g_GR` defined in §2.8, which scales both D2's $\beta$ and D3's $\lambda$ so low-risk proteins are not over-steered while high-risk proteins still receive strong intervention. In Stage C.1, the actuator uses a stable per-trajectory burden aggregate `B_GR`, not the raw per-refresh `G(t)`.

### 2.6 Context reliability for $\rho_B$

The current D1/D2 reliability gate uses whole-block structural entropy (`g_ent = exp(-mean_entropy / h0)`). That is too coarse for the D2 trust-region objective. Whole-block entropy mixes two different signals:

- unreliable context, where the hard-completed sequence is not structurally self-consistent and head scoring should be down-weighted;
- useful local freedom, where the structure model still admits multiple plausible amino-acid paths and D2 should be allowed to choose among them.

v1 should replace whole-block entropy as the main context-reliability signal with two self-aware structural-logit diagnostics:

```text
g_context = g_pnll * g_stability
rho_B     = g_time * g_context * g_ESS
```

`g_comp` may remain as telemetry, but it should not be the primary hard reliability gate once pseudo-NLL is available.

**Committed-token pseudo-NLL.** For committed residues in the active block context / receptive field, compute the current structural log-probability of the token already written in `x_t`:

$$
\mathrm{pnll}_B = -\frac{1}{|\mathcal{C}_B|}\sum_{i \in \mathcal{C}_B}\log p_{\mathrm{struct}}(x_i \mid x_t, \mathrm{backbone})
$$

where $\mathcal{C}_B$ is the set of committed residues used as context for block $B$. A low pseudo-NLL means the current sequence context is self-consistent under the structure model; a high pseudo-NLL means the hard completion is structurally suspicious and D2 should be weak.

A simple bounded reliability factor is:

$$
g_{\mathrm{pnll}} = \exp(-\mathrm{pnll}_B / h_{\mathrm{pnll}})
$$

**Temporal logit stability.** Between refreshes, compare the structural distributions for the same active-block positions. If logits are still drifting strongly, head scoring on the current hard completion is less reliable. A simple proxy is mean Jensen-Shannon divergence:

$$
g_{\mathrm{stability}} = \exp(-\mathrm{JSD}_B(p_t, p_{t-K}) / h_{\mathrm{jsd}})
$$

This factor is optional in the first implementation if previous-refresh logits are not yet cached for the same block geometry; it should still be recorded as the preferred second reliability signal because it directly measures context drift.

**Not used as a v1 reliability gate:** head variance over alternative completions. The local-window ensemble (§2.7) records variance and sign consistency as telemetry, but variance alone is not identifiable enough to drive $\rho_B$: high variance can mean either unreliable early context or a real immune opportunity.

### 2.7 Local-window completion ensemble for D2 head scoring

Argmax hard completion is too brittle as the only sequence context for D2 candidate scoring, especially early in denoising. It turns uncertain masked context into one over-confident pseudo-sequence, and the head can then score candidates against an input that the structure model has not really committed to.

v1 should keep the global controller completion method as `argmax`, but D2 candidate scoring should use a two-stage local-window ensemble:

1. Build the existing argmax hard completion and enumerate D2 candidates exactly as today.
2. Score all candidates once under the argmax completion to obtain a cheap preliminary $\Delta R_B^{\mathrm{argmax}}$.
3. Shortlist the best `completion_ensemble_rescore_top_m` candidates by preliminary $\Delta R_B^{\mathrm{argmax}}$; if candidate count is smaller than the shortlist size, rescore all candidates.
4. For each shortlisted candidate, generate `completion_ensemble_size` local completions over the scoring windows $\Omega(B)$ covering the active block. Residues outside $\Omega(B)$ stay at the argmax completion; residues inside $\Omega(B)$ that are still masked are sampled from the structural logits with deterministic per-block seeds.
5. Score each candidate under each local completion and use the ensemble mean in the D2 posterior:

$$
\overline{\Delta R}_B(a) = \frac{1}{K}\sum_{k=1}^{K}\Delta R_B^{(k)}(a)
$$

6. Use $\overline{\Delta R}_B(a)$ for feasibility, weights, ESS, and logit projection. Record ensemble variance and sign consistency as telemetry only:

$$
\mathrm{sign\_consistency}(a) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}[\Delta R_B^{(k)}(a) < 0]
$$

Do not use variance as a v1 gate. High ensemble variance can mean unreliable context, but it can also mark a real immune-sensitive peptide/anchor opportunity. The immediate goal is to avoid argmax-only false confidence while preserving a simple posterior objective.

Default v1 values:

| Field | Default | Role |
|---|---|---|
| `completion_ensemble_enabled` | true | enable local ensemble scoring for D2 |
| `completion_ensemble_size` | 3 | number of local completions per shortlisted candidate |
| `completion_ensemble_scope` | `local_windows` | restrict completion variability to $\Omega(B)$ |
| `completion_ensemble_rescore_top_m` | 8 | ensemble-rescore only the best preliminary candidates |
| `completion_ensemble_use_variance_gate` | false | log variance; do not gate on it in v1 |

Upgrade conditions:

- If `argmax_to_ensemble_rank_flip_rate` is high and ensemble sign consistency predicts realized benefit better than argmax $\Delta R$, increase `completion_ensemble_rescore_top_m` or `completion_ensemble_size`.
- If ensemble variance is high but sign consistency remains high for beneficial candidates, do **not** suppress those candidates; variance is then likely identifying a real immune opportunity rather than bad context.
- If ensemble variance is high and sign consistency is near random, delay D2 onset or strengthen `g_pnll` / `g_stability`; the problem is context reliability, not candidate weighting.

---

**Part II — Unified controller (§2.8–2.9).**

Part I makes a steered token survive. Part II decides *where* to steer, *how hard* globally, and *when* the Phase C schedule should expose or lock a region — all from one shared actionability field. The rollout (§2.9) keeps these layered: the field is only worth building once actuation (Part I) is proven.

### 2.8 Typed actionability state: GR × Phase C/D

**Rollout note — read before implementing this section.** Everything in §2.8 is *targeting* infrastructure: it defines *where* the controller acts. It does **not** address *actuation* — whether a steered token survives to the final sequence — which is the job of §2.1–2.4. The D3 pilot makes the ordering non-negotiable: targeting was faithful (recommit enrichment ≈ 2.7×) while actuation was empty (productive revisit ≈ coin flip, final immune ≈ NoD). Replacing D3's scalar `m_i` with the richer typed `A_i(t)` below does **not** change that outcome unless actuation is fixed first. The value of this entire field is therefore gated on D2 actuation being proven (**Stage A in §2.9**); do not implement `A_i(t)` before final-sequence persistence is demonstrated.

The unified architecture should not force GR, Phase C, D1, D2, and D3 to consume one universal scalar. The common object is a **typed local actionability state** that preserves evidence provenance:

```text
A_i(t) = source-aware evidence about whether residue i should stay editable,
         receive immune steering, be revisited, or be frozen.
```

The high-level narrative is:

```text
local actionability
  -> global pressure
  -> Phase C editability allocation
  -> D2 sticky logits execution
  -> D3 corrected-logit-aware commit protection
```

This keeps the controller unified without turning local actionability into a black-box scalar.

#### Source channels

The minimal theoretical state is:

```text
A_i(t) = {
  b_cur_i(t),
  b_env_i(t),
  b_mem_i(t),
  r_ctx_i(t),
  c_ready_i(t),
  f_rank_i(t)
}
```

The burden channels are three different time aspects, not redundant measurements:

| Channel | Question answered | Time aspect | Reliability behavior |
|---|---|---|---|
| `b_cur_i(t)` | Is this region risky now? | instantaneous current landscape | weak early; gated by `r_ctx` |
| `b_env_i(t)` | Is this region easy to become risky? | prospective proposal / completion envelope | meaningful even before commitment |
| `b_mem_i(t)` | Has this region been repeatedly risky? | persistent trajectory memory | becomes more trustworthy as observations accumulate |

Additional control channels:

| Channel | Meaning | Main use |
|---|---|---|
| `r_ctx_i(t)` | context reliability from pseudo-NLL, logit stability, completion reliability, ESS | controls trust in current hard-completion evidence |
| `c_ready_i(t)` | live correction readiness from D2 sticky state, trust-region pass, ESS, sign consistency, realized benefit | drives Phase C execute opportunity |
| `f_rank_i(t)` | structure/sample/commit stability | supports D3 freeze/protection |

`b_env_i(t)` may reuse the local-completion machinery from §2.7, but it must not be computed from the already-selected D2 active blocks it is meant to justify. In Stage B, compute proposal-envelope evidence from a pre-D2 current-landscape seed set or coarse residue/window sweep, then use that evidence to refine active-block discovery.

The current-burden channel is the residue-level projection of the head on the current hard completion:

$$
b_i^{\mathrm{cur}}(t) = \mathrm{Proj}_i\!\left(H(C(x_t,q_t))\right),
$$

which is exactly the online actionability signal that replaces the static-subtractive excess `z_dyn − z_static` (see *Static prior status* below) in active-block discovery.

`b_mem_i(t)` is the runtime replacement for the old static-prior role. The static prior tried to remember where a protein is intrinsically risky, but it used the WT sequence. `b_mem_i(t)` is the sequence-conditioned version: it records where this generation trajectory repeatedly becomes risky. It must cold-start from a neutral state and update only from fresh runtime evidence (`b_cur` / `b_env`), not from WT or static-prior initialization. Static prior is therefore not conceptually deleted; its intended role is upgraded into dynamic trajectory memory.

#### Runtime terminology

Use these names:

```text
GR(x):    completed-sequence reporting burden score
G(t):     runtime aggregate of the local actionability field
g_GR(t):  bounded controller pressure scalar derived from G(t)
```

Avoid `GR(x_t)` for runtime control. Mid-generation hard completions are not stable completed sequences; runtime control should consume the local state `A_i(t)` and its aggregate pressure `g_GR(t)`.

WT and NoD remain attribution references. They answer whether the final design resolved original WT burden, avoided new hotspots, and improved over unguided generation. They do not define runtime actionability. If a WT epitope is a true structure-conditioned attractor, it will reappear in `b_cur` / `b_env` / `b_mem` without WT subtraction. If it does not reappear, runtime steering should not chase it.

#### Risk fusion

The local risk/actionability field should be a soft union of evidence:

$$
u_i^{\mathrm{risk}}(t)
=
\mathrm{ClusterSupport}_i
\left[
\mathrm{SoftOR}\left(
w_{\mathrm{pos}}\!\left(r_i^{\mathrm{ctx}}(t)\right)b_i^{\mathrm{cur}}(t),
b_i^{\mathrm{env}}(t),
b_i^{\mathrm{mem}}(t)
\right)
\right]
$$

`SoftOR` is preferred over a weighted sum because any of the three burden channels can make a region actionable:

- `b_cur`: the hotspot is visible now;
- `b_env`: the hotspot is not committed yet but is easy to enter under the structure-conditioned generator;
- `b_mem`: the hotspot has repeatedly appeared across refreshes.

`w_pos(r_ctx)` is a positive-evidence reliability weight with a nonzero floor, not a hard gate. Low context reliability may downweight a current hotspot, but it should not erase it: early positive burden can still be a useful warning signal. In contrast, negative evidence is strict: absence of current burden is only allowed to support freeze when `r_ctx` is high.

`r_ctx` therefore only affects `b_cur`, and it affects positive and negative evidence asymmetrically. It should not gate `b_env` or `b_mem`: proposal risk and trajectory recurrence are precisely the evidence that prevents one quiet hard completion from being mistaken for safety.

Negative evidence is stricter than positive evidence. Low current burden plus low context reliability does **not** imply low actionability. Freeze requires current burden, envelope burden, and memory burden to all be low under reliable context.

#### Derived actuator fields

The controller should derive action-specific fields instead of sending `u_i^{risk}` everywhere:

```text
u_risk_i(t)    = where the controller should pay attention
u_hold_i(t)    = risky but not correction-ready
u_exec_i(t)    = live correction is ready and should be sampled soon
u_freeze_i(t)  = low-actionability plus structurally/commit stable
G(t)           = global aggregate from u_risk
g_GR(t)        = controller pressure mapped from G(t)
```

A useful semantic split is:

```text
u_hold   = u_risk and not c_ready
u_exec   = live c_ready within sticky TTL
u_freeze = low b_cur, low b_env, low b_mem, high r_ctx, high f_rank
```

`u_freeze` belongs mainly to the committed-residue / D3 surface, not to masked-position unmask allocation.

#### Phase C schedule coupling (Stage C)

The derived `u_hold` / `u_exec` fields drive the Phase C scheduler. This coupling **acts only in Stage C** (§2.9); Stage B computes the fields but does not change unmask allocation. A pure "high risk means later unmask" rule conflicts with D2: if the controller has already found a useful correction, keeping the target position masked too long lowers `selected_after_d2_rate` and prevents the correction from reaching the final sequence. The scheduler is therefore an **editability controller**, not a static ordering device, with three modes:

1. **Hold**: when context is unreliable or no structurally safe candidate exists, active risky regions remain editable and are delayed.
2. **Execute**: when a D2 posterior correction exists inside the sticky TTL (§2.2) and passes the trust-region / reliability logic (§2.1, §2.6, §2.7), the scheduler should increase the chance that still-masked target positions actually unmask during that TTL.
3. **Freeze**: when online risk drops and the commit face (§2.4) assigns high multi-objective rank, the region should stop churning and move toward final commitment.

A minimal schedule-coupling form is:

$$
\tilde{p}_i^{\mathrm{unmask}}(t)
\propto
p_i^{\mathrm{base}}(t)
\exp\!\left(-\alpha_{\mathrm{hold}}(t)u_i^{\mathrm{hold}}(t)\right)
\left(1+\alpha_{\mathrm{exec}}(t)u_i^{\mathrm{exec}}(t)\right),
$$

where $u_i^{\mathrm{exec}}(t)$ is nonzero only for positions with a live sticky D2 correction in $\mathcal{P}_{\mathrm{D2}}(t)$. The hold term preserves late editable space for risky regions; the execute term prevents immune-favored logits from being computed but never sampled. The total unmask budget should be normalized per step so this coupling changes **which** positions receive edit opportunity, not the overall denoising rate. The deferred Bernoulli-rate idea (§6.C) is the Phase C scheduler counterpart of the logit pipeline; it should be introduced only after the active-block definition is fixed and the sticky-logit persistence metrics are measurable.

#### GR aggregation: focal versus broad burden

`G(t)` should be length-normalized thresholded mass, not a max score. In the implemented typed field, the thresholding already happens upstream: `u_pressure_i(t)` is cluster-supported positive excess above `tau_ref_B`. Therefore runtime Stage C uses:

$$
G(t)
=
\frac{1}{L}
\sum_i
u_i^{\mathrm{pressure}}(t)
$$

Do **not** subtract `tau_ref` a second time at the `G(t)` layer. `tau_ref` is a background calibration reference, not a biological safety threshold, and it is consumed when building `b_cur` / `b_env` / `e_fresh`. This formula gives the desired focal-vs-broad behavior:

```text
sharp focal hotspot:
  only a few residues exceed tau_ref
  local u_risk high
  active block can appear
  G(t) remains moderate
  g_GR(t) medium / capped

broad prominent burden:
  many residues exceed tau_ref
  G(t) high
  g_GR(t) can approach full pressure
```

The division by $L$ is required; otherwise global pressure would scale with protein length.

**GR is residue-landscape based, not average-window based.** The local D2 proposal face stays window-level because the immune head scores peptide windows. Controller-level GR is instead computed from a residue-level landscape projected from window scores, matching the scientific claim that the method operates on a risk landscape, not an unordered pile of overlapping windows. The dataflow is:

```text
window scores z_w
  -> residue landscape h_i
  -> landscape burden / prominence / concentration descriptors
  -> G(t)
  -> g_GR(t)
```

Residue-level GR can express coherent hotspot clusters rather than isolated window outliers, smooth burden over a region rather than duplicated risk from many overlapping window lengths, landscape concentration (whether burden is localized enough to be actionable), and downstream compatibility with D3's residue revisit mechanism.

**No calibrated safe threshold in v1.** The head was not trained with an absolute calibrated "safe" threshold, so any threshold inside GR is a **background reference / calibration point**, not a biological safety threshold. Avoid language like `z_w < tau_safe means safe`; use `tau_ref defines the background level used to measure relative landscape burden`. The exact `tau_ref` calibration is an open design item — it may be corpus-based, per-protein robust, WT-distribution based for reporting, or NMP-aligned after calibration.

The pressure mapping remains a separate calibration layer. Stage B may log a per-refresh diagnostic scalar, but the Stage C.1 actuator must use a stable per-trajectory aggregate:

```text
G_step(t) = mean_i u_pressure_i(t)
B_GR(t)  = median({G_step(r): r <= t, r reliable})
g_GR(t)  = smoothstep(B_GR(t); B_low, B_high, g_min=0, g_max=1)
```

For the first actuator, `g_GR(t)` should primarily scale global immune pressure:

```text
beta_eff   = beta   * g_GR(t)
lambda_eff = lambda * g_GR(t)
```

Do not immediately let `g_GR` also change active-block thresholds, `max_windows`, $\rho_B$, or the Bernoulli schedule. Those couplings belong to the full architecture, but the first actuator should keep attribution readable. The per-step `G_step(t)` remains telemetry; it is not the pressure actuator.

#### Module read/write contract

| Component | Reads | Controls / writes |
|---|---|---|
| GR | distribution of `u_risk_i(t)` | `G(t)`, `g_GR(t)` global pressure |
| D1 active-block discovery | `u_risk_i(t)` and cluster support | active windows / active blocks |
| D2 proposal face | active blocks; local window $\Delta R_B$; structural trust region | token-direction posterior tilt |
| D2 sticky delivery | `c_ready_i(t)` / live correction state $\mathcal{P}_{D2}(t)$ | finite execution window for corrected logits |
| D3 commit/revisit | fresh burden evidence plus corrected-logit-aware rank | revisit, protection, freeze |
| Phase C schedule | `u_hold`, `u_exec`, deadline / catch-up | edit opportunity allocation |
| Attribution | WT / NoD references and final designs | resolved/new hotspot metrics |

The closed-loop control story becomes:

1. **Observe** current and proposal-envelope immune landscape.
2. **Remember** recurrent fresh evidence as `b_mem`, the dynamic replacement for static prior.
3. **Discover** active regions from `u_risk`.
4. **Budget** global pressure through `g_GR`.
5. **Steer** token direction through D2 local $\Delta R_B$ and trust region.
6. **Execute** corrections through sticky logits and Phase C `u_exec`.
7. **Commit, revisit, or freeze** through D3 corrected-logit-aware rank.
8. **Attribute** final changes against WT and NoD without feeding those references back as runtime subtraction.

#### Static prior status

The current `PLAN_RF.md` static-subtractive excess:

```text
z_dyn - z_static
```

should be demoted from runtime controller signal to attribution / negative-control telemetry. Static prior remains useful for explaining which WT hotspots were resolved and for demonstrating why static-prior Phase C was insufficient, but it should not define runtime actionability.

WT- and NoD-relative quantities are evaluation deltas, not runtime objects. Keep them named distinctly so they are never confused with the runtime field:

```text
delta_GR_vs_WT
delta_GR_vs_NoD
new_hotspot_rate
resolved_hotspot_rate
```

Goal: high-burden proteins receive strong D2/D3 pressure, while already-low-burden proteins avoid over-steering and unnecessary sequence churn. This is the low-burden over-intervention problem recorded in the D3 pilot, and it is solved as a controller-level temperature (`g_GR`) in Stage C, not as part of the actuation fix.

#### V1 theory boundary

This whole section is **Stage B** (§2.9). Its first version should be tightly scoped to the core pieces required to test whether the unified controller is doing the right kind of work:

```text
b_cur, b_env, b_mem, r_ctx, c_ready
  -> u_risk, u_hold, u_exec, u_freeze
  -> G(t), g_GR(t)
```

Expected Stage B outcomes (`G(t)` / `g_GR(t)` and Phase C `u_exec` are computed and logged here but only *act* in Stage C):

- the typed channels are populated and telemetry can separate failures across current observation, proposal envelope, memory, pressure mapping, and actuator execution;
- focal hotspots remain locally visible in `u_risk` without inflating `G(t)`;
- recurrent and proposal-envelope hotspots are not erased by one quiet hard completion (`b_mem` / `b_env` carry them);
- low-burden proteins become *identifiable* through smaller `G(t)` / `g_GR(t)`, so Stage C can later suppress over-intervention — Stage B itself does **not** yet scale β/λ;
- dynamic `A_i(t)` targeting is testable head-to-head against static `z_dyn − z_static` (the B→C gate).

### 2.9 Rollout: actuation before targeting

§2.1–2.7 are the D2 **actuation** pipeline — whether a steered token survives to the final sequence. §2.8 is the unified **targeting + pressure** field — where the controller acts and how hard it acts. These are different bottlenecks, but RAR 0005 changes the rollout interpretation: after actuation is proven, Stage B is a mechanism check and Stage C is the outcome lever. B and C are developed in the same campaign, with attribution separated by telemetry and matched configs rather than by a serial Pareto gate.

| Stage | Content | Decision point (failure → action) | Iteration |
|---|---|---|---|
| **A — actuation** | D2 sticky delivery + multi-objective commit + trust region + ensemble (§2.1–2.7). Uses the **existing** active-block definition; does **not** touch the §2.8 field. | Local actuation gate: steered tokens survive, preserve structure, and lower local risk where D2 fired. This gate has passed (RAR 0003); A_open shows a larger non-empty action space at low structural cost (RAR 0005). | done |
| **B — targeting mechanism** | Replace the active-block signal `z_dyn − z_static` with the typed `A_i(t)` field (§2.8). `b_cur + b_env + b_mem + r_ctx` land **together**. `G(t)` / `g_GR(t)` are computed and logged but do **not** scale β/λ in the B-only mechanism arm. | Does typed actionability populate, obey the firewall, and redistribute D2 fire relative to the matched static A_open comparator? Fail → fix typed-field wiring or targeting diagnostics; do **not** treat this as a D2 directional falsification. | one typed A_open arm |
| **C — global + schedule** | `g_GR` global-pressure temperature + Phase C editability (hold/exec/freeze, §2.8 Phase C coupling). | Does trajectory-level pressure suppress low-burden over-intervention while preserving high-burden D2 gain? This is the aggregate outcome/Pareto layer. | same campaign after B field sanity |

Three rules fix these boundaries.

**Stage B collapses the per-channel rollout (compression).** `b_cur`, `b_env`, `b_mem`, `r_ctx` all derive from one online computation (current head, the completion machinery specified in §2.7, a cross-refresh EMA). Introducing them one channel per cluster run buys little attribution: channel contribution is diagnosable from provenance telemetry, while causal channel ablations can be added later only if those diagnostics warrant them. The "v1 theory boundary" inside §2.8 is therefore the scope of **Stage B**, landed in a single stage — not a fourth thing done alongside Stage A.

**Stage A and Stage B/C cannot merge (actuation attribution floor).** This is a correctness constraint, not conservatism. Stage A changes actuation; Stage B changes the targeting signal source; Stage C changes global pressure. Run together, a null result cannot be separated into "the token did not persist" versus "the targeting signal was wrong" versus "global pressure was miscalibrated". Stage A is already separated by RAR 0003/0005.

**B/C attribution is telemetry-separated, not serial-Pareto-gated.** RAR 0005 showed that a flat aggregate Pareto under fixed β and no GR is expected because low-burden over-intervention is controller-wide. Therefore Stage B is judged on typed-field mechanism and redistribution, while Stage C owns the aggregate Pareto and D2-vs-D3-only verdict once `g_GR` is active.

**This is still not slow.** The next campaign needs one typed A_open mechanism arm plus the Stage C pressure arm(s). Stage B's readout is per-refresh/per-residue telemetry; Stage C's readout is burden-stratified aggregate outcome.

Gate conditions:

- **A → B/C**: three local actuation checks passed in RAR 0003, and A_open in RAR 0005 provides the operating point for B/C by opening a larger disagreement space without structural collapse.
- **B mechanism check**: typed `A_i(t)` populates, obeys the `b_mem`/D3 firewall, and measurably redistributes active blocks or editable positions against the matched static A_open comparator. Aggregate immune/Pareto is not a B gate.
- **C outcome check**: after `g_GR` is active, low-burden over-intervention must decrease without erasing high-burden D2 gains. The D2-vs-D3-only Pareto verdict and revisit-only fallback are evaluated here, not in B.

**Empirical update (A_open feasibility, 2026-06-11; RAR 0005 / 0001 / 0003).** Stage A passed its local gates (0003). A candidate-space-opening run (A_open: δ_struct=3, β=3, top_k=8, max_pos=4, min_ess=0) then revised the B/C boundary above on three counts:

1. Opening the action space gave 5× more persisted beneficial edits at ~0 structural cost, but did **not** improve the aggregate immune outcome. Candidate-space size is therefore not the binding constraint on outcome — it is the *test substrate* that makes the static-vs-typed targeting contrast measurable (A0's ~64 disagreements are too few; A_open's ~405 are enough).
2. The aggregate is masked by a low-burden over-intervention that is **controller-wide** (present in D3-only too, 0001) and **first-order, not second-order**: the GR-idealized counterfactual (zero low-burden steering) recovers a ~5× larger aggregate improvement, while the real D2 gain is concentrated and significant on high-burden proteins (internal-head median −2.57, 69% of designs) and masked elsewhere. Opening the space did **not** raise the high-burden ceiling (A_open ≈ A0 there), so β/space aggression is not the lever.
3. Typed targeting alone cannot fix the masking (it still fires on each protein's top positions regardless of absolute burden); only `g_GR` pressure driven by a stable per-trajectory burden aggregate does.

Consequences: **Stage B is demoted to a mechanism check** (does the typed field run and redistribute D2's fire on a non-empty operating point), and **Stage B and Stage C are developed and run together**, not B-gated-then-C. `g_GR` (Stage C) is promoted from deferred second-order optimization to the **co-primary, first-order outcome lever**. The B→C "D2 adds ≥5 pp Pareto over D3-only" verdict and the §5.3 / §6.D revisit-only fallback are **deferred to after `g_GR` is active** — a flat aggregate Pareto under fixed β with no GR is expected, not a falsification of D2. The attribution floor is preserved by **telemetry** (burden-stratified, typed-vs-static; implemented in Stage C), not by serial staging. NetMHCIIpan confirmation stays low-priority: the success criterion is steering visible on the internal head first. The implementation contract for this update lives in `PLAN_RF_UNI_CTRL.md`.

---

## 3. Implementation stages

The rollout in §2.9 defines stages by attribution boundary. Stage A is complete as a local actuation gate (RAR 0003) and A_open provides the current operating point for B/C (RAR 0005). This section records implementation content; the active implementation contract for B/C lives in `PLAN_RF_UNI_CTRL.md`.

### Stage A — actuation (completed)

Stage A proves a steered token can survive without structural cost. It is a single-mechanism pipeline with two sub-steps because the faces have different dependency relationships.

**A.1 — persistence core.** Delivery Face (sticky, §2.2) + Sampling Face (§2.3) + Commit Face (multi-objective rank, $\nu > 0$, §2.4). Without A.1 the D2 signal never reaches the final sequence regardless of any other parameter sweep. All three together: sticky alone increases `selected_after_d2_rate` but the landed tokens still get remasked by the broken commit ranking; the multi-objective rank alone protects D2 tokens that land, but with same-step rate at 7% there are barely any to protect.

**A.2 — structural safety + head-input reliability.** Proposal Face trust region ($\delta_{\mathrm{struct}}$, §2.1) + local-window completion ensemble (§2.7) + pseudo-NLL context reliability (§2.6). $\delta_{\mathrm{struct}}$ is *required*, not an ablation — without it D2 is structurally unconstrained and any immune improvement is inseparable from potential structural cost. The ensemble is the head-input reliability fix; $\delta_{\mathrm{struct}}$ is the structural-output safety fix. Both are needed before a D2 immune gain counts as a meaningful structure-preserving steering signal. Do not add a separate manual maneuverability condition: the only structural support control in v1 is the safety filter that builds $Q_B^{\mathrm{safe}}$.

### Stage B — targeting field

Replace static active-block selection with the typed `A_i(t)` field (§2.8). Channels (`b_cur + b_env + b_mem + r_ctx`) land together. In the B-only mechanism arm, `G(t)` / `g_GR(t)` are computed and logged but do not scale β/λ. The B readout is mechanism and redistribution, not aggregate Pareto.

### Stage C — global pressure + schedule

`g_GR` scaling β/λ + Phase C editability (§2.8 Phase C coupling). This is the co-primary outcome lever after RAR 0005: it addresses controller-wide low-burden over-intervention and owns the aggregate Pareto / D2-vs-D3-only verdict once active.

### Deferred: time schedule

$\beta(t)$, $\eta(t)$, $\alpha(t)$ as t-curves (§6.A) are orthogonal to the stage sequence. Consider only after Stage A yields measurable persistence and a clear immune signal. The three-phase ramp (burn-in → edit window → freeze) is the natural target shape, but v1 keeps scalars to limit the hyperparameter surface.

---

## 4. Recommended v1 config

```yaml
d2:
  beta: 1.0
  eta: 0.7
  struct_temperature: 1.0      # T_struct, no sweep in v1
  delta_struct: 1.5            # nat; required, not ablation
  context_pnll_h0: 2.0         # pseudo-NLL reliability scale
  context_jsd_h0: 0.5          # optional temporal-stability scale
  completion_ensemble_enabled: true
  completion_ensemble_size: 3
  completion_ensemble_scope: local_windows
  completion_ensemble_rescore_top_m: 8
  completion_ensemble_use_variance_gate: false
  sticky_ttl_steps: 5          # = refresh_interval
  sticky_clear_on_selected: true
  sticky_clear_on_remask: true
  sticky_overwrite_on_refresh: true

d3:
  alpha_struct: 0.3            # structural sanity retention on rank face
  d2_evidence_nu: 1.0
  d2_evidence_ttl_steps: 5
  d2_evidence_requires_benefit: true
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
| `realized_benefit_rate` | Among paired disagreements, corrected branch has lower local head risk than paired structural branch | $N_{\Delta R_{\mathrm{corr}} < \Delta R_{\mathrm{uncorr}}}$ / $N_{\mathrm{disagree}}$ | not used in v1 rank | **positive and used as D2-evidence eligibility** |
| `final_persistence_rate` | Among disagreements, D2-written token survives to final sequence | $N_{\mathrm{survived}}$ / $N_{\mathrm{disagree}}$ | 0% | **10–30%** |

Also report every rate above stratified by `sticky_age_steps`. If older sticky deltas show lower realized benefit or lower final persistence, add a decay term in a later revision; do not add decay before this telemetry is observed.

Additional ensemble diagnostics:

| Term | Meaning | Use |
|---|---|---|
| `argmax_to_ensemble_rank_flip_rate` | fraction of blocks whose best candidate changes after ensemble rescore | detects argmax-completion brittleness |
| `ensemble_delta_R_std` | standard deviation of $\Delta R_B^{(k)}$ across local completions | context uncertainty telemetry |
| `ensemble_sign_consistency` | fraction of ensemble completions where candidate improves local risk | directional robustness telemetry |
| `rank_percentile_d2_written` | final rank percentile of D2-written residues before remask | diagnoses whether commit ranking still erases useful D2 tokens |

Marginal product is the "D2 actually changed the final sequence" rate: under the pipeline, target ≥ 1% (vs 0% in v1). This sets the floor for evaluating whether immune/structure metrics are even worth interpreting.

### 5.2 Decision tree

After running the Stage A/A_open and B/C configs on matched 50-protein pilots:

1. If `selected_after_d2_rate` does not rise markedly above 7%: Delivery Face is broken; sticky cache lifecycle has a bug. Stop and inspect.
2. If `selected_after_d2_rate` rises but `final_persistence_rate` stays ≈ 0%: Commit Face is broken; $\alpha$ or $\nu$ is not effective. Verify rank_scores are being consumed in d2_logits mode and that Z-scoring uses the correct base population.
3. If all three filter rates rise and `final_persistence_rate` ≥ 10% but structure preservation drops below D3-only baseline: $\delta_{\mathrm{struct}}$ too loose. Tighten to 1.0 nat and retry.
4. If all three filter rates rise, `final_persistence_rate` ≥ 10%, and structure is preserved, but pre-GR aggregate Pareto is flat: this is not a D2 falsification. Check burden-stratified deltas; if high-burden designs improve while low-burden designs regress, proceed to Stage C `g_GR`.
5. If typed Stage B does not populate the field or does not redistribute active blocks / editable positions against the static A_open comparator: fix `actionability.py` / controller integration before judging outcome.
6. If `g_GR` suppresses low-burden over-intervention but the high-burden D2 marginal disappears or aggregate Pareto remains flat, then the directional value of D2 beyond D3 revisit is questionable.

### 5.3 Falsification

The old pre-GR "B → C" Pareto gate is superseded by the empirical update in §2.9. A flat aggregate Pareto under fixed β and no `g_GR` is expected and does not falsify D2. The unified pipeline is falsified only after:

1. Stage A/Aopen actuation is working and structure is preserved;
2. Stage B typed fields populate and redistribute relative to the static A_open comparator;
3. Stage C `g_GR` is active and demonstrably suppresses low-burden steering;
4. D2 still adds no high-burden marginal value beyond D3 revisit, or the aggregate Pareto remains flat despite low-burden suppression.

Only then should D2 be deprioritized toward the revisit-only fallback (§6.D).

---

## 6. Deferred ideas (appendix)

The unified pipeline above is the v1 main method. The following ideas are not part of it but are recorded here for future iteration.

### 6.A Time schedule (deferred per §3)

Three-phase $\beta(t)$, $\eta(t)$, $\alpha(t)$ ramp:
- Burn-in $t \in [0, t_{\mathrm{start}}]$: $\beta = 0$, structure-only
- Edit window $t \in [t_{\mathrm{start}}, t_{\mathrm{ramp\_off}}]$: $\beta$ at peak, D2 actively shapes posterior
- Freeze $t \in [t_{\mathrm{ramp\_off}}, 1]$: $\beta$ decays, D3 finalizes

The implicit denoising temperature (head reliability is low at both ends of $t$) motivates the shape. Considered after the Stage A pipeline shows measurable persistence.

### 6.B Post-sampling structural accept/reject

After D2 categorical sample, evaluate structural penalty and probabilistically revert to a paired structural sample (not argmax — preserves stochastic semantics):

$$
P(\mathrm{accept}) = \min\!\left(1, \exp(\beta_{\mathrm{AR}} \cdot \Delta \ell^{\mathrm{struct}}_i)\right)
$$

This is a structural veto filter, not strict Metropolis-Hastings (no proposal-ratio correction). Cleaner than the sticky+rank composition in principle, but requires `controller.post_step()` API extension to mutate `x_t`. Deferred unless the unified pipeline empirically falls short.

### 6.C Bernoulli rate coupling

At refresh step, raise $\dot \kappa_i$ for positions in active blocks so they unmask at higher per-step probability. This invades Theorem 1's protected schedule and changes the ordering interpretation. Lower priority than sticky delivery (§2.2) because sticky achieves similar footprint expansion without touching the rate schedule. It is the Phase C scheduler counterpart referenced in §2.8.

### 6.D Revisit-only D2 (fallback)

D2 doesn't push logits at all; only flags positions for D3 commit/revisit re-ranking. Loses the directional "which token" information from D2; keeps only the "where to revisit" signal. This is the natural fallback if the unified pipeline cannot demonstrate that the logit-layer directional information adds value beyond D3's revisit selection (the §5.3 / Stage B→C falsification outcome).

### 6.E Controller-level GR temperature

**Superseded by §2.8.** The GR controller temperature, the GR three-way split (`GR(x)` / `G(t)` / `g_GR(t)`), residue-landscape-based aggregation, the focal-vs-broad `G(t)` form, the `tau_ref` background-reference caveat, the "no calibrated safe threshold" warning, the `beta_eff` / `lambda_eff` scaling, the staging rule (do not couple other knobs first), and the evaluation deltas (`delta_GR_vs_WT`, `delta_GR_vs_NoD`, `new_hotspot_rate`, `resolved_hotspot_rate`) now live in §2.8's *GR aggregation* and *Static prior status* subsections. This entry is retained as a pointer only; do not duplicate the GR spec here. Runtime implementation is Stage C (§2.9).

### 6.F Head-variance as a controller gate

Head variance over alternative completions is recorded by the local-window ensemble (§2.7), but it is not part of the v1 controller as a gate. High variance can mean two different things:

- unstable early context, where the head is unreliable;
- a real immune opportunity, where one structurally safe mutation changes a peptide/anchor strongly.

Because these sources are not identifiable from variance alone, head-variance gating is deferred. It can later be tested as an analysis or ablation axis against the simpler structural-logit confidence proxies and the ensemble sign-consistency metric.

---

## 7. Out of scope

- C+B engineering implementation details (telemetry schema, dataclass layouts, decay function shapes): belongs in `PLAN_RF.md` Task D2 implementation contract once this design is approved.
- Multi-allele extension of $\beta$ and $\delta_{\mathrm{struct}}$: separate concern; depends on whether allele-specific risk fields are aggregated before or after the trust region filter.
- Sensitivity analysis of all five proposal-face knobs ($\delta_{\mathrm{struct}}$, $\beta$, $\eta$, $T_{\mathrm{struct}}$, $\rho_B$ gating): v1 fixes four to defaults and treats $\delta_{\mathrm{struct}}$ as the primary axis; full sweep deferred to v2.
- Theoretical analysis of whether sticky delivery preserves the time-marginal of the DFM reverse process: empirically the change is small (D2 only acts on positions that would unmask anyway under the natural Bernoulli draws), but a formal statement is deferred.

---

## 8. PLAN readiness check

Stage A pieces are implemented and locally validated. Stage B/C now live in the dedicated unified-controller plan; `PLAN_RF.md` remains the Stage A / baseline plan surface.

The following pieces are ready to move into `PLAN_RF.md` as implementation tasks:

| Piece | PLAN status | Notes |
|---|---|---|
| Sticky D2 pending-logit cache | ready (A.1) | required for delivery footprint |
| Benefit-gated D2 evidence | ready (A.1) | required to prove D2 effort is useful, not just different |
| Multi-objective commit rank | ready (A.1) | use all-committed global z-score in v1 |
| Structure trust region | ready (A.2) | $\delta_{\mathrm{struct}}$ is required, not an ablation |
| Pseudo-NLL context reliability | ready (A.2) | replaces whole-block entropy as primary reliability signal |
| Local-window completion ensemble | ready (A.2) | use two-stage shortlist + K=3 local completions |
| Process metrics / falsification tree | ready | required before interpreting immune/structure aggregate metrics |

Owned by `PLAN_RF_UNI_CTRL.md`:

| Piece | Reason |
|---|---|
| Typed actionability field (`A_i(t)`) | Stage B mechanism check on the A_open operating point |
| Controller-level global-risk temperature | Stage C outcome lever; evaluates low-burden suppression and high-burden D2 marginal |
| Time-varying $\beta(t), \eta(t), \alpha(t)$ | deferred v2 shape after scalar `g_GR` is understood |
| Head-variance gate | variance is logged by ensemble, but not identifiable enough to gate in v1 |

Conclusion: keep `PLAN_RF.md` scoped to Stage A and use `PLAN_RF_UNI_CTRL.md` for Stage B/C. Do not move B/C back under the Stage A task heading.
