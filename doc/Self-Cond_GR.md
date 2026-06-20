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
- **Arm = `fresh`** (the `self_conditioned` +0.009 Spearman edge is an order of magnitude below the K=3 reseed noise floor of 0.078, measured fresh-vs-bootstrap at refresh_step=0; `Recall@High` identical). Recycling is *not harmful* but *not measurably helpful* → use the simpler memoryless arm. SC1 therefore overrides the analysis script's raw `best.arm=self_conditioned`.
- **Freeze horizon.** At the B.1 cadence only 2 refreshes fall after `t_start` (refresh_step 0,1), with ρ 0.62 (step 0) → 0.74 (step 1). SC1 freezes at the **first** reliable refresh (`freeze_after_reliable_refreshes=1`): refresh_step=0 is the genuinely *unguided* trajectory (no steering before `t_start`, `controller.py:485`), so it is the closed-loop-cleanest estimate, and one frozen value gates both steering refreshes. Calibration bands (`B_low`/`B_high`) come from this run's refresh_step=0 fresh `topm_lse` `B_sc` distribution. (Fallback if step-0's small tail leak `P(low|high)=0.06` surfaces in SC1: freeze at refresh_step=1 instead, gating only the later refresh.)

### Stage 2 — behavior (only if monitor passes)

`B1 + beta-only SC pressure`. Expected — **not** an immediate aggregate-mean jump, but:

- true-low / NoD-low over-intervention drops vs B1;
- true-high / NoD-high gain preserved vs B1;
- structure no worse than B1;
- pressure-bin vs burden-bin diagonal markedly higher than old C.1.

Prerequisites Stage 1 must hand off (none exist yet, all out of scope for the monitor — `PLAN_RF_SC_GR.md` §6): a **single pinned aggregator** (the §3 monitored candidate maximizing `Recall@High`); its **own calibration JSON** with `tau_prom` / `B_low` / `B_high` recomputed on the `B_sc` distribution (the `B_GR` bands do **not** transfer); and a **new `pressure_source` value + actuator dispatch seam** — the current `pressure_source` is only a config-load validator, not a runtime branch, so the estimator code (`_update_pressure_state`) needs a real seam before a probe-fed `B_sc` changes behavior (D2/D3 actuation stays untouched).

---

## 7. Evaluation references (oracles only — never runtime)

NoD (DPLM unconditional, `phaseD_pilot50r2_nod_n8`) and ProteinMPNN-unconditional are **evaluation oracles**: they define ground-truth burden and measure do-no-harm / amplification. They are never a runtime prior — the runtime substitute is the SC-GR probe. WT stays attribution only (`resolved` / `new` hotspot), never a runtime classifier.
