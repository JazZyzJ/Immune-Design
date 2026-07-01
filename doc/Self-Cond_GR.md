# Self-Conditioned Proposal-Envelope GR Probe (SC-GR)

**Status**: live scientific branch document. Created 2026-06-18, forked from `doc/RF_Controller_Architecture.md` §6.G. Exploratory; **monitor-gated**. We will keep revising this file. The code-level monitor implementation is specified in `PLAN_RF_SC_GR.md`.

**Scope of the branch.** SC-GR replaces *only* the burden **estimator** behind the global-risk gain `g_GR` — the single argmax hard-completion `B_GR(t)` specified in `RF_Controller_Architecture.md` §2.8 "GR aggregation" — with a recycled, ensembled, pre-D2 terminal-risk probe. It does **not** change D2/D3 actuation and does **not** wholesale replace GR / Phase C/D. The typed targeting channels (`b_cur` / `b_env` / `v_target`, §2.8 of the architecture doc) are validated by Stage B.1 and are **not** frozen; only the `g_GR`-suppress-low mechanism is.

**Why this document is separate.** In `RF_Controller_Architecture.md`, the `g_GR` "suppress low-burden over-intervention" content (§2.5 / §2.8 GR-aggregation / §2.9 Empirical update / §3 Stage C / §8) is **frozen** — it rested on a premise that RAR 0008 showed was largely a regression-to-mean artifact (see that doc's §2.9 *Empirical update v2*). This file is where the GR redesign continues.

---

## 1. Motivation — corrected empirical context

The Stage C.1 actuator drove `g_GR` from a `B_GR(t)` computed on a **single argmax hard completion** of the current trajectory. Two corrected facts (RAR 0008, NoD-baseline analysis, 2026-06-18) reframe the GR effort:

**(a) The burden estimator was too noisy.** Against the unconditional (NoD) per-protein burden, the single-hard-completion `B_GR` correlates only **ρ ≈ 0.41**. A single hard completion is the *mode*, not the *expectation*, of the terminal-risk distribution; it misclassifies low-vs-high burden, which is what made C.1's pressure mis-fire (it neither protected low-burden proteins nor preserved high-burden gain cleanly — C1b). What the probe should estimate is the central tendency of the *unguided* terminal-risk functional $B_{\text{true}}(x_t) = \mathrm{RiskAgg}\big(p_{\text{struct}}(x_1 \mid x_t, S)\big)$. The argmax is one mode of $p_{\text{struct}}$, and for MHC-II this is especially fragile: binding burden is driven by a few anchor/register positions, so terminal risk is a sharply non-linear, multi-modal function of those sites — a single argmax completion can land in or skip a hotspot register almost arbitrarily. A stochastic-completion ensemble approximates the central tendency instead of one over-confident mode. [Monitor update, RAR 0010: on the 50-protein pilot the **aggregator** turned out to be the larger lever — switching the window→trajectory reduction from `mean_excess` to tail-sensitive `topm_lse` lifted Spearman vs NoD by **+0.29** (0.31→0.60 on a single argmax), and the K-ensemble added a further **+0.15** (→0.74). The ensemble's *distinctive* contribution is closing the true-high-as-low tail: `P(probe low | NoD high)` 0.12→0.00. Both levers are needed; the aggregator is primary, the ensemble secures the tail.]

**(b) The premise pointed at the wrong end.** The original "low-burden over-intervention is first-order" claim (RAR 0005) was largely a **regression-to-mean binning artifact**: binning designs by a *steered* baseline's risk and subtracting that same baseline inflates the low-burden bin (slope of `gr_arm ~ gr_d3` ≈ 0.27). Re-measured against the **correct unconditional (NoD) baseline** with a regression-free split-half:

| NoD-burden tercile | B1 Δ vs NoD | reading |
|---|---|---|
| low (already-safe) | **+0.25** (53% harmed) | real do-no-harm direction, but **small / within noise** |
| mid | −0.88 | helps |
| high (immunogenic) | **−1.03** | steering works; **the real lever** |

Aggregate B1 vs NoD = **−0.58** (40% of proteins made worse); the soft baseline D3 only reaches +0.10 vs NoD, so comparing B1 to D3 had badly understated it. Burden is a real but moderate protein property: **ICC ≈ 0.40** (between-protein sd 3.43, within-protein sd 4.17), so single-trajectory burden estimation is intrinsically noisy.

**Consequence.** A good burden estimator's primary value is deciding **where to push harder (amplify-high, the −1.0 nat lever)**, with do-no-harm protection of low-burden as the *secondary* constraint (the +0.25 within-noise effect). SC-GR is the redesigned estimator; the amplify direction is the redesigned objective.

---

## 2. Distinction from trained self-conditioning

Trained self-conditioning (Chen et al. *Analog Bits* 2022; HarmonicFlow / FlowSite §3) feeds the network its own previous detached terminal estimate (x̂₁) as an extra **input**, learned at training time (e.g. a 50%/detach recipe), with the standard flow/diffusion loss and **no special loss term**. The estimate **re-enters the denoiser**.

SC-GR is **inference-only and training-free**: the recycled terminal estimate seeds *diagnostic* completions whose frozen-head scores produce an **external scalar gain**; it never re-enters the denoiser and needs **no loss / no training**. The only shared primitive is "a running terminal estimate exists."

Engineering boundary (not a novelty claim): the probe's only job is to **sense** a trajectory-level burden scalar; it never selects completions, reweights particles, or emits per-token direction — token steering stays entirely in D2. Sensing and steering live in separate modules so the gain can be calibrated without touching D2's actuation.

---

## 3. Mechanism (v1)

Detached self-conditioned state carried across refreshes (all detached from autograd). Definitions are **pinned** so the estimator is reproducible — there is **no** cross-refresh completion cache in the controller today (`_build_hard_completion` recomputes fresh each refresh; grep for `prev_completion`/`x_hat` is empty), so this is genuinely new persisted state:

```
prev_x1_hat     := previous refresh's structural-argmax terminal estimate (tokens).
                   NOT head-scored, NOT a K-consensus — just the deterministic structural argmax.
prev_confidence := previous refresh's per-position canonical-softmax max-probability (over canonical
                   AA tokens). "High confidence" := prev_confidence >= confidence_threshold.
```

Both arms draw their recycled seed from this **deterministic structural** state — not from either arm's own stochastic samples — so the Fresh/SC contrast is not confounded by arm-specific sampling noise. The early-frozen per-design `B_sc` used for actuation is separate (§4); the monitor stage carries only these two (`PLAN_RF_SC_GR.md` §1).

At each refresh, **pre-D2**:

1. Build `K` one-shot pseudo-terminal completions (`K ≈ 3–4`) from the current `x_t` + structural logits. Committed positions are fixed at their current token.
2. Masked positions are seeded from `prev_x1_hat` at high-`prev_confidence` positions; low-confidence / uncovered positions are sampled from the structural softmax (or argmax). This recycling is what makes the probe *self-conditioned* rather than a fresh independent ensemble.
3. **Firewall (hard).** The probe must **not** read D2 active blocks, sticky state, corrected logits, or low-risk D2 candidates. It estimates terminal risk under the **pre-D2 structural posterior conditioned on the current trajectory `x_t`** — *not* the marginal NoD distribution: `x_t` may already carry earlier steering, and §4's early-freeze is what keeps the actuated estimate close to unguided. Reading D2 state would additionally make `g_GR` self-reinforcing.
4. Score each completion with the frozen head over the **full sequence** → a residue/window burden map (reduced to the scalar in step 5). **Scope is the load-bearing detail:** use the full-sequence projection — every window the head returns, the `b_cur` scope (`controller.py:1698-1704`) — **not** the seed-window-union `b_env` path (`_compute_envelope_burden`, `controller.py:1889`), which zeroes every non-seed residue and would miss burden distributed thinly across the sequence. Two separate products (do not collapse, and do **not** route either into the typed `b_env` channel):
   - a **full-sequence prospective risk map** `r_i(t)` — telemetry / a later targeting study only;
   - `B_sc(t)` — trajectory-level scalar burden proxy → `g_GR` pressure.
5. **Aggregate in three explicit stages, keeping tail-sensitivity and noise-robustness at the stages where each belongs:**
   - *residue → window* (within one completion): the frozen head's `local_risk_aggregation: LME` (log-mean-exp) is already tail-sensitive inside a window — keep it.
   - *window → trajectory scalar* `G(x̂₁,k)` (within one completion): the Stage-C estimator used `mean_i ReLU(u_i − τ_prom)`, a cross-window **mean** that dilutes a sharp few-window hotspot. Because the corrected objective is **amplify-high**, make this stage a monitored choice — `mean-ReLU` vs a tail-sensitive top-m-window LSE / supra-threshold mass — selected by amplify-high recall (§6), not by default.
   - *over the K completions (and refreshes)*: combine with a **robust** statistic (`median` / slow EMA). Do **not** put tail-sensitivity here — with `K ≈ 3–4` a high quantile (p90) is just the max of 3–4 noisy samples, high-variance and not estimable (it would contradict the small-effective-K caveat in §6). `B_sc(t)` is thus the robust central estimate of `G`: the expectation-like signal that sets the gain.
   - *separately*, log an **over-K instability** diagnostic (max / variance of `G`). "Occasionally spikes into a hotspot mode" is a **different** protein condition from "reliably high-burden" and warrants a different response, so do **not** fold the spike statistic into `B_sc`; it is a candidate *secondary* amplify trigger evaluated later, not part of the v1 gain.

Actuation:

```
g_GR = smoothstep(B_sc; ...)     # v1: g in [g_min, 1] (protect-low); v2: allow g > 1 (amplify-high)
beta_eff = beta * g_GR           # scale_lambda = false — locked for v1 (RAR 0008: scaling lambda reverts
                                 #   suppressed designs to base and loses D3's commit, harmful). Re-trial
                                 #   deferred, not abandoned.
```

D3 `b_mem` / `m_i` do **not** consume the probe (keep prospective signal out of persistence memory).

---

## 4. Closed-loop guard (required)

Because `g_GR` changes the trajectory, a per-refresh `B_sc(t)` that drifts with the steered state self-reinforces (lower steering → lower observed risk → lower pressure, or the reverse self-excitation). v1 therefore computes an **early-frozen per-design `B_sc`** from `t_start` / the first 1–2 reliable refreshes and holds it (or a very slow EMA) for actuation; later `B_sc(t)` is telemetry only.

Early-freeze at `t_start` is doubly motivated: the controller does **no** steering before `t_start` (`controller.py:485` returns identity logits; no D2, no D3 commit), so the first post-start refresh sees a trajectory shaped only by unguided DPLM denoising. The early-frozen `B_sc` is therefore both the **least self-contaminated** estimate **and** the one **closest to the NoD oracle** the monitor validates against — the two reasons to freeze early coincide.

Tension to track: this means in v1 the cross-refresh recycling mainly *stabilizes the early frozen estimate* and feeds telemetry — full recycling-into-live-actuation is a **v2** item, deferred until the loop is characterized.

---

## 5. Open design questions (answered by the monitor, not assumed)

- **When to probe.** Terminal burden is barely determined early (mostly-masked) yet steering can no longer respond late. Measure `Spearman(B_sc(t), NoD-burden)` as a function of refresh index to find the estimable-and-actionable sweet spot; do not assume `t_start` is optimal.
- **Does recycling help? (2-arm, Stage-1.)** Ablate exactly two arms — a **fresh** independent ensemble (`Fresh-K`) vs the **self-conditioned** recycled-seed ensemble (`SC-K`) — and judge each on *both* `Spearman(B_sc, NoD-burden)` *and* `Recall@High` (§6): recycling may stabilize the estimate yet go conservative and miss the high tail, which only the recall metric exposes. A continuous confidence-weighted hybrid (`p_reuse ∝ confidence`) is a v2 tuning knob, not part of the gate — the v1 high-confidence-threshold seeding (§3 step 2) is already its hard-threshold special case. **Outcome (RAR 0010): within-noise — `fresh` and `self_conditioned` differ by +0.009 Spearman against a 0.078 reseed noise floor, with identical `Recall@High`; SC1 uses `fresh`.**
- **Amplify scope.** v1 is protect-low (`g ≤ 1`); the do-no-harm finding says amplify-high (`g > 1`) is the larger lever. v2 candidate actuation forms: a two-sided `g = clip(1 + a·z(B_sc), g_min, g_max)` or a two-sided smoothstep with `g_max > 1`. Promote once the sensor is trusted; do not let v1 conservatism re-anchor the method on the small (within-noise) low-burden end.

---

## 6. Validation — monitor first, then behavior

### Stage 1 — monitor-only (no actuation change)

Log `B_sc`; compare to the NoD (and D3) reference burden. The gain must be **measured, not assumed**: the K-reliability upper bound `between / (between + within/K)` (K=3 → 0.82, K=8 → 0.92 from the NoD variance components) overstates the real number, because completions share committed context, structural logits and head bias (effective K < nominal) and are mid-trajectory, not final.

Per protein × refresh × trajectory, log: old `B_GR_argmax`; the `Fresh-K` and `SC-K` ensemble `B_sc`; the candidate window→trajectory aggregators (`mean-ReLU`, top-m-LSE, supra-threshold mass); the over-K instability (max / variance of `G`); within-completion variance and cross-refresh stability; old vs new pressure bin; and the oracle burdens (NoD final, D3-only final, guided final — post-hoc only).

Gate:

- `Spearman(B_sc, NoD per-protein burden) ≥ 0.6` (clearly above the current `B_GR` ≈ 0.41–0.45);
- burden-bin × pressure-bin crosswalk diagonal clearly above C.1;
- **true-high-misclassified-as-low rate** is low — operationalize as `P(B_sc ∈ low bin | NoD ∈ high bin)`, reported with `Recall@High = P(B_sc ∈ top bin | NoD ∈ top bin)`, both as functions of refresh index and per recycling arm. This *tail*, not the mean correlation, is what cost C.1 its high-burden gain.

**Stage 1 outcome — PASSED (2026-06-20; run `scgr_monitor_pilot50r2_aopen_n4_steps20_b4_seed42`; numbers + decomposition in RAR 0010).**

- Gate: best `Spearman(B_sc, NoD)` = 0.742 (fresh) / 0.752 (self_conditioned) ≥ 0.60; `Recall@High` 0.765 vs old `B_GR` literal 0.471; `P(probe low | NoD high)` 0.000 vs old 0.235 (n=50, high tercile=17).
- **Aggregator pinned = `topm_lse`** (decisive: 0.742 vs `supra_mass` 0.630 vs `mean_excess` 0.344 at refresh_step=1; `mean_excess` is gating-bad — the ensemble cannot rescue it, 0.306→0.344 — so the aggregator is the limiting factor).
- **Arm = `fresh` for the v1 actuator** (the `self_conditioned` +0.009 Spearman edge is an order of magnitude below the K=3 reseed noise floor of 0.078, measured fresh-vs-bootstrap at refresh_step=0; `Recall@High` identical). This does **not** reject self-conditioning as a mechanism; it only means the current monitor data do not justify promoting it into the first behavior run. SC1 uses the simpler memoryless arm, while self-conditioned recycling stays monitored as a v2 candidate.
- **Freeze horizon.** At the B.1 cadence only 2 refreshes fall after `t_start` (refresh_step 0,1), with ρ 0.62 (step 0) → 0.74 (step 1). SC1 freezes at the **first** reliable refresh (`freeze_after_reliable_refreshes=1`): refresh_step=0 is the genuinely *unguided* trajectory (no steering before `t_start`, `controller.py:485`), so it is the closed-loop-cleanest estimate, and one frozen value gates both steering refreshes. Calibration bands (`B_low`/`B_high`) come from this run's refresh_step=0 fresh `topm_lse` `B_sc` distribution. (Fallback if step-0's small tail leak `P(low|high)=0.06` surfaces in SC1: freeze at refresh_step=1 instead, gating only the later refresh.)

### Stage 2 — behavior (only if monitor passes)

`B1 + beta-only SC pressure`. Expected — **not** an immediate aggregate-mean jump, but:

- true-low / NoD-low over-intervention drops vs B1;
- true-high / NoD-high gain preserved vs B1;
- structure no worse than B1;
- pressure-bin vs burden-bin diagonal markedly higher than old C.1.

**SC1 outcome (RAR 0011).** Actuation + pressure diagonal validated (`Spearman(g_GR, NoD)=0.61`, `scale_lambda=false`, scTM 0.946); high-burden gain preserved (Δ≈0); protect-low immune effect directionally correct (low-tercile Δmean −0.41, recovering B1's tail-driven over-intervention) but **within-noise** (Wilcoxon p=0.38) — the small lever §1 predicted. The decisive test is the amplify-high lever (SC2 / `PLAN_RF_SC_GR.md` §6.2).

Prerequisites Stage 1 must hand off (none exist yet, all out of scope for the monitor — `PLAN_RF_SC_GR.md` §6): a **single pinned aggregator** (the §3 monitored candidate maximizing `Recall@High`); its **own calibration JSON** with `tau_prom` / `B_low` / `B_high` recomputed on the `B_sc` distribution (the `B_GR` bands do **not** transfer); and a **new `pressure_source` value + actuator dispatch seam** — the current `pressure_source` is only a config-load validator, not a runtime branch, so the estimator code (`_update_pressure_state`) needs a real seam before a probe-fed `B_sc` changes behavior (D2/D3 actuation stays untouched).

---

## 7. Evaluation references (oracles only — never runtime)

NoD (DPLM unconditional, `phaseD_pilot50r2_nod_n8`) and ProteinMPNN-unconditional are **evaluation oracles**: they define ground-truth burden and measure do-no-harm / amplification. They are never a runtime prior — the runtime substitute is the SC-GR probe. WT stays attribution only (`resolved` / `new` hotspot), never a runtime classifier.

---

## 8. From scalar sensing to a three-resolution steering field — status & goals

This section sets the forward direction past the **v1 §2 engineering boundary** ("the probe's only job is to sense a trajectory-level burden scalar; it never emits per-token direction — token steering stays in D2") and the §3-step-4 note that `r_i(t)` is "telemetry / a later targeting study only." The estimator is unchanged; what changes is how many of its readouts we consume. Method is deliberately left open here (under active design).

### 8.1 Recent results — one estimator, three resolutions

The same SC terminal estimator (§3: one frozen-head ensemble over stochastic completions) is read at three decreasing granularities, each validated against the NoD oracle; accuracy falls as granularity rises (~0.74 / ~0.6 / ~0.44):

- **`B_sc` — trajectory/protein scalar** — [RAR 0010](../Results/Analysis/0010-sc-gr-monitor-estimator-x-aggregator-dec/record.md): the `topm_lse` ensemble `B_sc` tracks NoD per-protein burden at ρ≈0.74 (old single-argmax `B_GR` ≈0.41), closing the true-high-as-low tail. *(= the §6 Stage-1 gate, restated as the coarsest rung.)*
- **`r_i` — per-residue / region map** — [RAR 0020](../Results/Analysis/0020-sc-gr-per-residue-r-i-accuracy-vs-nod-pe/record.md): per-residue `residue_excess` ranks residues within a protein like the NoD per-residue oracle (region ρ median 0.578, `Recall@HighResidue` 0.645 at the frozen horizon) — the "where to push" gate PASSES.
- **`P_cheap` — block-candidate tuple score** — [RAR 0019](../Results/Analysis/0019-candidate-level-signal-direction-v2-pair/record.md): a paired-seed, omega-LME-aligned cheap probe ranks D2 candidate tuples by terminal immune at ρ≈0.44 vs 0.28 for local `L` (~0.62 of the oracle's noise-capped ceiling), usable as a **soft** ranker but not a hard selector (`probe_capture` 0.09–0.16); the conditioned variant `P_cond` is weaker and not promoted.

`r_i` is not a separate object — it is the per-residue field that `B_sc` reduces to a scalar, and `P_cheap` is the same machinery read on a candidate tuple. One forward pass, three projections.

### 8.2 Status — estimator validated; first direct-allocation actuator failed

**Sensing is settled.** The branch question — *can the recycled probe sense terminal immune risk accurately enough to steer?* — is answered at all three resolutions (RAR 0010 `B_sc` ρ0.74, RAR 0020 `r_i` region ρ0.58, RAR 0019 `P_cheap` ρ0.44). None is a per-token causal oracle, but all sense.

**The first *consumption* of that sensing failed.** The `planner_pilot47` behavior run (RAR 0019 M4) tested direct position allocation — `r_i → Φ → v_target` reweight — and **H1 was not supported at any `g_max`**: the allocation arm did **not** beat the `v_target` baseline (alloc−v_target median +0.0004/+0.0013/+0.0008; p 0.29/0.12/0.18), and it gave back the baseline's reduction-vs-NoD (`v_target` −7% → alloc ~0%) at matched scTM. This was **not** a floor effect (cohort NoD immune spans 0.0007–0.042; 28% > 0.02).

Two findings reframe the open problem:

1. **Consumption semantics (RAR 0019 M4).** `r_i` marks *prospective terminal burden*; `v_target` already encodes *current editability* (`b_cur`/`b_env`/`b_mem`/`r_ctx`). The multiplicative reweight `v_target·Φ` treated prospective burden as edit allocation, redirecting ~45% of active residues (Jaccard 0.38) toward higher-`Φ`/**lower-`v_target`** sites (alloc-only median `v_target` 10.24 vs v_target-only 11.09; corr(Φ,v_target)=0.29) — i.e. it dragged fire onto positions `v_target`'s actionability terms had deliberately demoted, shifting budget without reliable token-level improvement.

2. **Per-edit authority ceiling (RAR 0019 M5 + RAR 0013).** Independently of *where* the budget is spent, each edit buys only ~0.2 nats: `delta_R_B` median −0.206 is **flat across `g_max` 1.5→2.5**, `ESS_candidates` ≈ 3.3, applied logit shift pegs at the cap; RAR 0013 showed candidate-space opening (delta_struct/top_k/max_cand/beta 5→8) is null on high-risk proteins. More budget/pressure does not buy more reduction regardless of allocation.

So the open problem is no longer estimation **or** simple allocation. It is (a) the **consumption semantics** of `r_i`, and (b) the **per-edit actuator authority** — the latter bounds the upside of *any* `r_i` consumption pathway. Forward options are §8.4 (the §8.3 direct-allocation stack below is retained as the recorded, falsified v1).

### 8.3 Direct-allocation v1 (RECORDED — FALSIFIED 2026-06-28; see §8.4)

> **Outcome banner.** This stack was implemented (`PLAN_PLANNER_SC_GR.md`) and run (`planner_pilot47`). The **Allocation** layer below — `r_i → Φ → v_target` reweight — is the actuator H1 falsified (RAR 0019 M4, §8.2). The **Budget** layer (`B_sc → g`) and the sensing remain valid. Retained verbatim as the experimental record; the live forward direction is §8.4. Do not implement this stack as-is.

The next deliverable is the first **closed-loop** run that converts the field into terminal-immune reduction at preserved scTM. The design is **task-driven** — lower MHC-II terminal risk at held structure — and reads each resolution where it is strongest, as a three-layer stack (*how much / where & how long / which token*). The where-vs-what split echoes training-free path-planning masked-diffusion samplers (cf. P2, arXiv 2502.03540, validated on DPLM), but the organizing logic is the task, not their sampler: we borrow the idea, not the scheme.

- **Budget — `B_sc` — *how much*.** The coarsest, most reliable readout sets one per-trajectory scalar gain `G`: *how much total immune-edit budget this trajectory deserves* — not where, not what. v1's protect-low gain (`g ≤ 1`, §6) extends to amplify-high (`g_max > 1`, the −1.0-nat lever of §1). The budget must **not** be spent as uniform per-position strength — that is just stronger uniform guidance, the thing the thesis must beat.
- **Allocation — `r_i` — *where & how long* (the position-dependent thesis).** `r_i` distributes the budget across positions as a **budget-preserving editability mass** `Φ_i` (the allocation mass — **distinct from the typed-actionability field `A_i(t)`**; `Φ_i` *reweights* that field, it never is it): the raw signal smoothed to window/region grain (MHC-II risk is register-level; RAR 0020's region signal, ρ 0.578, is the validated resolution) and normalized **at the protein level to mean 1** (so budget flows toward high-risk regions). The controller consumes `Φ_i` by **reweighting the existing typed selection field `v_target`** — `s_alloc = v_target·(ε + c·Φ_i)` — **not** by replacing it (so `v_target`'s reliability/memory/envelope structure is preserved) and **not** as a per-residue pressure multiplier `β_i = β·r_i`. This where-editable lever, not a larger logit correction at a fixed schedule, is what makes the reference flow position-dependent and is the chapter's core claim. Reweighting (vs replacing) insulates the controller from `r_i`'s scale drift and makes the uniform control exact: `Φ_i ≡ 1` rescales `v_target` by a constant → identical selection at the **same** `G`. High `Φ_i` ⇒ selected/edited sooner; low ⇒ left to the baseline. *(v1 actuates the **where** via this selection reweight; **how long editable** needs the schedule/remask seam and is deferred — `PLAN_PLANNER_SC_GR.md`.)*
- **Value correction — `P_cheap` — *which token*.** At the sites Allocation opens, bias *which token* lands toward low terminal immune. `P_cheap` is **not** a replacement actuator and does not pick tokens alone: it is a reliability-gated **correction added to local `L`** (soft, not argmin; RAR 0019), active only where reliable/actionable. Token choice stays the denoiser's job, and the §3 firewall holds: `P_cheap` and the external head never enter the budget/allocation (sensing) decision; they only nudge values at already-selected sites, so the *where* can never be contaminated by the *what*.

**Thesis spine — pre-registered, falsifiable (H1). → OUTCOME: NOT SUPPORTED (RAR 0019 M4, 2026-06-28).** The allocation arm did not Pareto-dominate the uniform (`v_target`) arm at any `g_max`; see §8.2 for the mechanism. The pre-registration below stands as written (it is why the result is interpretable), but the hypothesis is now falsified, not open. v1 had to show first **not** "immune dropped by X" but that the allocation *itself* helps. With budget `G` and structure held, the `r_i`-driven allocation was predicted to Pareto-dominate the uniform one:

$$
Y_{\text{imm}}(G,\,\Phi^{r}) \;<\; Y_{\text{imm}}(G,\,\Phi^{\text{unif}}) \quad \text{at matched scTM}
$$

where `Y_imm` is residual terminal immune (lower = better), `Φ^r` the `r_i` allocation mass, `Φ^unif ≡ 1` the uniform control (which, under reweighting, *is* the baseline `v_target` selection arm), `G` the shared budget. Because both arms spend the same `G` and differ only in `A`, any immune/scTM-Pareto gain is attributable to **position-dependence**, not to more pressure. If H1 holds, the claim generalizes past "adaptive β" to a methods statement: *a hard-completion prospective-risk field can allocate discrete-diffusion editability more efficiently than uniform guidance.* Maximizing the absolute immune drop is a **later** objective, gated on H1.

**Time axis — fixed for v1, not dropped.** The *when* is the other half of "where/when-editable," so it stays in the conceptual controller; v1 **fixes** it rather than tuning it. v1 freezes **both** the budget and the allocation mass `Φ_i` early (§4 early-freeze, firewall-clean) and steers over the controller's **existing schedule + early-freeze window** — it does **not** add a new time-varying ramp-up knob. The current read of the evidence — suppress early (early guidance over-accelerates unmasking → premature commit), strengthen through middle/late where steering is both more reliable (ρ 0.62→0.74, §6) and most effective (cf. discrete-masked-diffusion guidance, arXiv 2507.08965), *not* a quiet-late stabilization phase — is the **shape a tuned ramp-up should take**, deferred to a later variant. Freezing `Φ_i` early also sidesteps, for v1, the self-reinforcement that late `r_i` re-reading would invite; **updating `Φ_i` timing online** under the firewall is that first deferred variant.

**Dropped from v1.** Ensemble instability `σ_i` (no accuracy gate yet, unlike `r_i`/RAR 0020) and content-dependent noise schedules (hyperschedules are training-bound, not an inference-time field).

Method — the budget→mass coupling, the editability mapping of `A_i`, and the `P_cheap` correction form — remains under design; this section fixes the task decomposition, the budget-preserving control, and the H1 falsification, not the implementation.

### 8.4 Revised direction (under discussion, 2026-06-28)

Given §8.2's two findings — wrong *consumption semantics* for `r_i`, and a *per-edit authority ceiling* that bounds any `r_i` consumption — three candidate paths. Sensing (§1–7) and the Budget layer are kept; only how `r_i`/`P_cheap` are consumed changes.

- **Path A — cheap re-consumption of `r_i` (leading, low-risk).** `r_i` stops being an edit *allocator* and becomes a **triage / tie-break *within* the `v_target`-eligible set**, additive and rank-based, not a multiplier: `eligible = {v_target > τ_v}` (or top-q), `priority(block) = rank(v_target) + λ·capped_rank(r_i)` over eligible blocks only. This removes the failure mechanism (§8.2.1): restricted to eligible blocks, `r_i` can no longer drag fire onto low-`v_target` sites; rank-space kills scale domination.
  - **Strict eligibility is load-bearing — do *not* widen `τ_v` to manufacture room for `r_i`.** RAR 0019 M4 showed `r_i`'s signal *beyond* `v_target` lives on **low-`v_target` (low-actionability) sites**; lowering the eligibility bar re-admits exactly the sites whose edits did not convert, re-creating the `v_target·Φ` harm. `r_i`'s legitimate room is the gap between the **eligible set and the edit budget / active-window cap**: when `|eligible| > cap`, `r_i` chooses which genuinely-actionable blocks get the scarce budget first. If that room is too small to move the outcome, **that is the finding** (`r_i ⊥ v_target` is unactionable) → it points to Path C, not to widening.
  - `P_cheap` enters only as a **value-layer soft shortlist**, a **separate arm** from the where-triage (for attribution; never bundled into the same arm), and is bounded by the same authority ceiling (RAR 0019: ρ0.44 ranker, `probe_capture` 0.09–0.16 — not a selector). Lower priority than getting the where-triage + C right.
  - Low downside; **upside bounded by the authority ceiling** — expect "harmful → neutral / slightly positive," not a large win.
- **Path B — expensive local-oracle D2 (demoted from actuator to evaluator / teacher for C).** Rolling out candidates to rank them is what RAR 0019's K=8 oracle already did: candidates are near-equivalent at the terminal (self-capture 0.20, M3) over a ~3-effective structurally-safe set (M5), so an expensive ranker on the **current** support is ceiling-bounded — **not** a production actuator. Its remaining value is as the **evaluator for Path C**: an oracle that audits whether a new C-style proposal family actually expands terminal headroom (do the coordinated-register candidates contain terminally-better, structure-acceptable options the cheap signals miss?). Keep it as measurement / teacher / training-label source, not as the next actuator.
- **Path C — coordinated-proposal sampler (NO-GO 2026-06-30; parked). The lever was mis-identified as *coordination*; it is *breadth*.** The binding constraint (RAR 0019 M5 / RAR 0013) is that each edit buys only ~0.2 nats within a ~3-candidate structurally-safe set, and naive widening (delta_struct/top_k/beta) is null. Path C hypothesised the fix was *coordinated* multi-position edits (anchor + compensating neighbours) needing a reconditioned sampler. **C0b (RAR 0022/0023) falsified that specific hypothesis:** coordination (reconditioned vs independent) adds ≈0; the real headroom is captured by *independent* wide-block resampling + head selection (**breadth**), and the winning edit fits D2's existing 4-position cap. So the coordinated-sampler rewrite is not built. The remaining sub-bullets (channel, atomic move, C-what) are retained as the *record of the falsified design*; the live direction is the breadth lever in the Recommendation below.
  - **Interface — a new channel, not a wider `v_target`.** C **does not widen `v_target`**; it adds a **separate coordinated-proposal feasibility channel** so high-`r_i` regions become *legitimately* eligible (real feasible proposals exist there) rather than *pretendingly* eligible (a lowered `τ_v`). Keep the validated field as `v_target_single` (today's single/block-local actionability, untouched) and add `v_target_coord` (feasibility of a coordinated-register proposal). `r_i` decides **which high-risk registers are worth attempting** a coordinated proposal on. Eligibility for the coordinated path is a conjunction: `eligible_C = high r_i ∧ coordinated-proposal feasible ∧ oracle/probe shows terminal headroom ∧ structure guard passes`. This closes the A/C relationship: A reorders `r_i` *within* `v_target_single`; C *creates* new actionable space (`v_target_coord`) where `v_target_single` is low but `r_i` is high — the principled inverse of widening `τ_v`.
  - **Cost reality (code-verified 2026-06-28).** C splits into two very unequal halves. **C-where** (the `v_target_coord` channel) is cheap: the three `v_target` consumers read through named swappable seams (`active_window_source` dict `controller.py:668` + `_selection_field` :1818; `within_block_source`/`typed_field` :767 → `counterfactual.py:68`; pressure path :2175/2182), so a parallel `(L,)` channel can be registered without touching `v_target`. **C-what** (actually making a coordinated edit) is large: D2's actuation core is **per-position-factorized end-to-end** — candidate generation is independent-per-position (`enumerate_candidates`: cartesian / i.i.d.), the structural gate is per-position *before* tuples form (`build_safe_candidate_support`: a lone anchor change is filtered out individually, no joint gate), and the correction collapses to per-(position,token) marginals → independent per-position logit shifts (`project_marginals` → `compute_delta_logit` → `apply_logit_correction`). A correlated anchor+compensation edit **cannot be represented** in that pipeline. So C requires reworking D2's generation + structural gate + marginal→logit-shift representation, not just adding a channel — no RAR/precedent exists for joint proposals (new ground). The register↔block unit is favourable (a ~9-residue register ≈ one head window ≈ a small block), but D2 today edits only 2–4 masked residues/block, so coordinated register edits also need the editable-subset cap lifted jointly.
  - **Staged plan (C0–C3, agreed 2026-06-29) — measure before the rewrite.**
    - **C0 — offline GO/NO-GO (precedes all C-what; no D2/sampler rewrite).** RAR 0019/0013 proved the *factorized* support is ceiling-bound; they did **not** prove a *coordinated* proposal has headroom. Test **H-coord** (joint anchor+compensation exists, factorized D2 can't express it → build C) vs **H-struct** (hotspots are fold-core, joint also only trades structure for immune → don't rewrite) first, in two cheap steps:
      - **C0a — correlational screen (existing data, no new generation).** Join per-residue `r_i` (`actionability_residues` `legacy_residue_excess` / residue telemetry) with per-position structural constraint (`controller_events` `struct_top1_logprob` / `struct_top2_gap`, or burial/SASA from the PDB) on the `planner_pilot47` outputs. High-`r_i` sites systematically structurally *locked* → leans H-struct (early NO-GO); structurally *tolerant* yet single-edit buys only ~0.2 nats → leans H-coord (proceed). **Outcome (2026-06-30, RAR `0021-c0a-…`): not a NO-GO.** High-`r_i` sites are not overwhelmingly locked (tolerant-fraction 0.40–0.59 vs the <0.2 NO-GO bar; lock-coupling weak — Spearman +0.035 direct / −0.18 `r_ctx`), and the C-target (high-`r_i` ∧ low-`v_target`) cells are *less* locked than the D2-reachable ones. **New finding (M4): the demotion is a sensing gap, not structure** — `v_target` is low at C-target sites because *local* head evidence (`b_env` most, C/D2 ratio 0.31) is low, while `r_ctx`/structure is flat (ratio 1.07). Qualified: the direct structural signal covers only 0.4% of C-target cells (leans on `r_ctx`), and "tolerant" ≠ "a coordinated immune-reducing move exists" — that is still C0b.
      - **C0b — constructive test (new *offline* compute, still no D2 rewrite).** On high-`r_i` registers, draw register-level **multi-step reconditioned** DPLM resamples — *true* joint: a single forward over a remasked register is still a per-position marginal *product* (the factorization trap one level up), so only reconditioning across partial commits yields anchor+compensation correlation — score each for terminal immune (head/oracle) + a structure proxy, and ask: **does any joint resample beat the single-position best at acceptable structure, for a material fraction of registers?** GO iff yes. **Outcome (RAR 0022 + decomposition RAR 0023): NO-GO for coordination, but the multi-position headroom is real — it is *breadth*, not coordination.** Coordination (true reconditioned − fake independent) adds ≈0 at the median (win fraction 0.300 < 0.334; `median(joint)−median(fake)` +0.05; a ~30% tail only), and the depth sweep 18→36 is flat (`depth_gap` −0.07). The gate compared joint vs single, which mostly measures *degrees of freedom* (9 edits > 1): the fake (independent 9-position) arm beats single by a large margin. Decomposing that (RAR 0023): the design sits on an **immune peak** — untargeted 9-position resampling regresses −2.6 nats off it (single −0.04), so ~2/3 of fake's advantage over single is regression-to-mean, not targeting; and the best-of-K is **sampling-budget-unsaturated** (K 8/16/32 → −8.4/−9.7/−10.5), so the magnitude is a search number, not a fixed lever. But it is **not** regenerate-away: the best-immune structure-acceptable fake keeps recovery (0.566 vs design 0.559) and changes only **4 of 9** register residues — a genuine low-immune, structure-safe, recovery-preserving multi-position edit within D2's existing 4-position cap.
    - **C1 — channel (only if C0 = GO).** Add `v_target_coord` as a *where / feasibility* channel only (the cheap C-where; registers into the named seams). It never carries the edit itself.
    - **C2 — C-what as an atomic register resample-accept move (sampler / commit layer, NOT a D2 logit correction).** Propose a whole-register tuple (C0b's generator) → evaluate → accept/reject → if accepted, **commit/protect the tuple as a unit**. Closer to a D3 / local-MCMC move than a D2 marginal nudge; it **deliberately avoids** the per-position-logit-shift representation, whose independent sampling **cannot guarantee** tuple co-occurrence (the broken "project joint → per-position shifts" path — telemetry can *observe* the co-occurrence failure but not *prevent* it; it reproduces the current ceiling). This relocation is the point: do not fight the factorized D2 pipeline, step outside it at the sampler.
    - **C3 — Path B oracle = evaluator / teacher, never production actuator.**
  - **JointProposal object.** The execution unit is not the `(L,)` channel but a `JointProposal = (positions, tokens, gen_logprob, seed)`; `struct_feasible` and `immune_delta` are stamped *downstream* by the guard and the oracle (do not fold evaluation outputs into the generated object, or generation ends up needing the oracle). The structural guard must be **joint and post-tuple**: a per-position / factorized pseudo-likelihood is exactly the existing `Q_B` and **cannot** judge anchor+compensation; the cheap-enough joint floor is a one-step reconditioned denoiser, and the expensive tier *is* the Path B oracle — run it in the **same** rollout that yields the immune delta (one pass, not two).

**A and C are coupled.** A is bounded precisely because today's high-`r_i` sites are low-`v_target` (unactionable under single-token safe edits); C makes those sites actionable via coordinated proposals — the only principled way to give `r_i` real, *safe* room (vs widening `τ_v`, which gives unsafe room). So the shape is **A (止血, cheap) + C (unlock, main line), with B as C's evaluator** — not three independent options.

**A3 — terminal-aware actionability (M4-motivated intermediate; coder change, not a config tweak).** C0a M4 found `v_target` misses high-`r_i` sites because *local* head evidence (`b_env`) is low there, not because of structure — a terminal-vs-local sensing gap (the original SC-GR thesis, one rung down). A3 would let `r_i` / a terminal probe inform the actionability field so D2 *sees* those sites. Today the §3 firewall forbids the probe feeding typed `b_env`/`v_target`, so A3 is a **new actionability channel** (coder work), not a config flip. Upside is **authority-bounded**: per-position edits at those sites still buy only ~0.2 nats — A3 spreads capped edits to more terminally-relevant sites, it does **not** break the ceiling (that is still C). Lower priority than C0b, and gated behind whether A1/A2 already recover to neutral.

**Recommendation (updated 2026-06-30, post-C0b).** The experimental arc resolved which axis carries the headroom:

- **A (where + value): neutral** — A1 triage and A2 terminal-probe both ≈0 at matched scTM (`PLANNER_README.md` §A). A tested *where* and *which-token*; neither is the lever.
- **Coordination (Path C): NO-GO, parked** — reconditioned joint adds ≈0 over independent joint (RAR 0022/0023). Do not build the coordinated sampler.
- **Breadth is the live lever** — head-selected *independent* multi-position resampling finds structure-safe, recovery-preserving, low-immune ~4-residue edits (RAR 0023). It is real but **soft**: partly regression-to-mean (the design sits on an immune peak) and best-of-K-unsaturated, so the offline magnitude is an upper bound. Because the winning edit fits D2's existing 4-position cap, the mechanism is likely **A3 (terminal-aware actionability, so D2 *selects* the high-`r_i` register) + wider-register candidate *search* under D2's real budget**, not a new sampler.
- **Next decisive test — closed-loop breadth (not offline), staged:**
  - **Stage 1 (config-only, no new code):** lift D2's `max_positions_per_block` toward the register width + raise candidate budget/K, on the blocks D2 *already* selects; measure how much of the offline breadth headroom survives D2's real per-block budget (no oracle fold-filter, bounded K) and its **per-position marginal** application (D2 biases marginals, it does not commit best-of-K — realizing wide breadth through marginal biasing is part of what this tests). Agent-executable (`PLANNER_README.md`).
  - **Stage 2 (A3, coder + short PLAN, only if Stage 1 undershoots):** terminal-aware actionability so D2 *selects* the high-`r_i` **low-`v_target`** registers it currently skips (C0a: skipped for low `b_env`, not structure). Crosses the §3 firewall (prospective signal into targeting) — self-reinforcement guarded by early-freeze, like `B_sc`; needs a reviewed spec, not a config tweak.
- **Evaluation axes = immune + scTM.** scTM is the structural safety gate; **recovery is a *movement/diversity diagnostic*, not a gate** — a recovery drop at held scTM is acceptable (RAR 0023's recovery-preserved / 4-of-9-changed is read as "the winning edit is *local*, not a regeneration," not as a safety criterion).
- **Path B** (oracle) is retired with Path C; the coordinated-sampler / atomic-joint-commit rewrite is **not** built (breadth uses independent marginal application). **`P_cheap`/A2** stay authority-bounded neutral.
