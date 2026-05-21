## Phase C: Core Contribution — Position-Dependent Reference Flow (expanded 2026-04-24)

**Objective**: Implement and validate position-dependent discrete flow matching as an immunogenicity-aware generative mechanism, with a clear separation between the base reference-flow mechanism, the static conditioning axis, and any optional adaptive controller.

> **Scope of this plan section**: Phase C is now treated as a layered method family.
> - **Base mechanism**: static-prior reference flow (`C1`, `C2`)
> - **Second axis**: static hotspot conditioning (`C3`)
> - **Optional extension**: adaptive controller (`Task D`)
>
> Concrete design choices for `C2/C3/D` are intentionally left open until offline design is frozen. Only `C1` has a detailed implementation contract in this file today.

**Planned experiment layers**:
- **C0**: DPLM native sampler (baseline) — Task C0
- **C1 (Tier 0)**: static-prior DFM sampler + position-dependent schedule (includes `g ≡ 1` uniform-schedule control via config) — Task C1
- **C2 (Tier 1)**: retrain with static-prior position-dependent forward process — STUB
- **C3 (Tier 2)**: static hotspot conditioning axis (FiLM/additive/attention-bias family; CFG optional) — STUB
- **Task D (optional extension)**: adaptive controller layer (online hotspot refresh / logit steering / revisit-corrector / schedule modulation) — STUB

**Mathematical reference**: `doc/Reference_Flow_Derivation.md` §4 (sampling + conditioning + adaptive controller), §7 (design choices).

### Task C0: DPLM Native-Sampler Baseline

**Goal**: Drive Module K's trained adapter under DPLM's native sampler on the full test set. No algorithmic novelty; a reproducible driver + SLURM.

**Inputs**
- Module K checkpoint: `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt`
- Test set parquet (per allele) + B1 backbone PDBs
- Allele selection

**Outputs** (per run)
- `work/immune-design/if_phase_c/c0/<allele_tag>/<run_id>/generated.parquet` (columns: `protein_id`, `design_idx`, `sequence`, `seed`, `wall_seconds`)
- `generated.fasta`
- `run_config.yaml` (resolved)
- `manifest.json` (git SHA, checkpoint digest, timestamp)

**Planned File Touchpoints**
- Create: `scripts/run_if_phase_c0.py`
- Create: `scripts/submit_if_phase_c.slurm` (`MODE=native`)
- Modify: `doc/SCRIPTS.md` (register under Phase C)

**CLI contract**
- Required: `--checkpoint`, `--test-set-parquet`, `--pdb-root`, `--allele`, `--output-root`.
- Optional: `--n-designs-per-protein` (default 1), `--seed` (default 42), `--max-iter` (default 10, DPLM native), `--temperature` (default 1.0), `--device cuda|cpu` (default cuda).
- Echo all resolved hyperparameters to stdout at start (per `feedback_print_hyperparams`).

**TDD Gate**
1. RED: driver writes designs without linking to source `protein_id`, or silently swallows DPLM sampler errors.
2. GREEN: 3-protein fixture produces `generated.parquet` where (a) every row's `protein_id` matches a row in the input test set, (b) `len(sequence) == input sequence_length`, (c) manifest contains non-empty checkpoint digest + git SHA.

**Acceptance**
- `generated.parquet` contains `test_rows × n_designs_per_protein` rows minus logged failures.

---

### Task C1: Sampling-Only Position-Dependent DFM

**Goal**: Implement the §4.3 DFM sampling algorithm as a standalone module that consumes a frozen Module K checkpoint + B2 h_maps and generates sequences under a **fully config-driven** per-position unmasking schedule. No retraining. No model weight modification.

**Frozen design principles (2026-04-23)**
1. **Every experimental knob is a YAML field.** Same code path runs uniform-schedule control (`g ≡ 1`), linear-clamp / sigmoid / power amplification, three base schedule forms, variable step count, H3 h-shuffle — all by config.
2. **Sampler is denoiser-agnostic.** Takes a callable `denoiser(x_t, t, struct) → logits[L, V]` and applies §4.3. DPLM-specific wiring lives in the CLI driver, not in the sampler module (so C2/C3 reuse the module with a different denoiser).
3. **Resolved config is the arm identifier.** Every run serializes the fully-materialized config alongside its output; no arm label is hard-coded in the code.

**Hyperparameter schema** (YAML, validated by `inverse_folding/reference_flow/config.py`; all fields required unless marked optional)

```
sampler:
  n_steps: int                    # DFM step count N
  seed: int                       # sampling RNG seed
  temperature: float              # logits scaling before Categorical (optional, default 1.0)
  n_designs_per_protein: int      # independent samples per input

schedule:
  base_form: str                  # "linear" | "cosine" | "cubic"
                                  #   linear: κ_base(t) = t
                                  #   cosine: κ_base(t) = 1 − cos(π/2 · t)
                                  #   cubic : κ_base(t) = 3t² − 2t³

amplification:
  form: str                       # "constant_one" | "linear_clamp" | "sigmoid" | "power"
  c: float                        # max amplification strength
                                  #   required unless form == "constant_one"
                                  #   linear_clamp: g(h) = 1 + c · max(0, h − mu)
                                  #   sigmoid:      g(h) = 1 + c · σ(kappa · (h − mu))
                                  #   power:        g(h) = 1 + c · max(0, (h − mu))^p
  mu: float                       # threshold h₀ (optional, default 0.0 — assumes centered h)
  kappa: float                    # sigmoid sharpness (required iff form == "sigmoid")
  p: float                        # power exponent (required iff form == "power")
  h_source: str                   # "h_raw" | "h_processed" | "h_normalized_corpus"
                                  #   h_normalized_corpus uses (h_raw − μ_corpus) / σ_corpus
                                  #   from B3 sidecar JSON (must be passed via --h-corpus-stats)
  g_max_cap: float                # upper clamp on g(h) to guard tail outliers
                                  # (optional, default 20.0)

h_shuffle:                        # H3 control mode; permutes h per-protein before sampling
  enabled: bool                   # optional, default false
  seed: int                       # required iff enabled; acts as a base seed,
                                  # effective permutation is derived per
                                  # (protein_id, design_idx)
```

**Preset config scaffolds** (starting points only; the experimenter sweeps values, switches forms, and mixes modes — nothing in code depends on these file names):
- `inverse_folding/reference_flow/configs/c1_null.yaml`     — `form=constant_one` (DFM + uniform schedule)
- `inverse_folding/reference_flow/configs/c1_linclamp.yaml` — `form=linear_clamp, mu=0.0, h_source=h_processed`
- `inverse_folding/reference_flow/configs/c1_sigmoid.yaml`  — `form=sigmoid, mu=0.0`
- `inverse_folding/reference_flow/configs/c1_power.yaml`    — `form=power`
- `inverse_folding/reference_flow/configs/c1_shuffle.yaml`  — `form=linear_clamp` + `h_shuffle.enabled=true`

**Algorithm** (direct transcription of `doc/Reference_Flow_Derivation.md §4.3`)

```
Input:  denoiser p_θ(· | x_t, t, struct)
        h[1..L]                      (from B2)
        g(·)                         (from amplification config)
        κ_base(·), κ_base'(·)        (from schedule config)
        N, seed, temperature

Derive: g_i    = g(h[i])             (optionally clamped to g_max_cap)
        κ_i(t) = κ_base(t)^{g_i}
        κ_i'(t)= g_i · κ_base(t)^{g_i − 1} · κ_base'(t)

rng ← seed
x   ← [m, m, ..., m]                 (length L)
for k = 0, 1, ..., N − 1:
    t  ← k / N
    dt ← 1 / N
    logits ← p_θ(x, t, struct) / temperature    # [L, V]
    for i where x[i] == m:
        rate_i   ← κ_i'(t) / (1 − κ_i(t))
        p_unmask ← min(1, rate_i · dt)
        if rng.Bernoulli(p_unmask):
            x[i] ← rng.Categorical(softmax(logits[i]))
# final guard: force-unmask any residual mask tokens with the last logits
for i where x[i] == m:
    x[i] ← rng.Categorical(softmax(logits[i]))
return x
```

**Boundary / numerical notes** (implementation must honor)
- `t = 0`: `κ_i(0) = 0`, `1 − κ_i = 1`; `κ_i'(0) = g_i · 0^{g_i−1} · κ_base'(0)` — zero for `g_i > 1`, `κ_base'(0)` for `g_i = 1`. No division-by-zero.
- `t → 1`: rate diverges by construction (§2.2). `min(1, rate·dt)` clamps it.
- Compute `κ_i(t)` via `exp(g_i · log κ_base(t))` with a floor on `log κ_base(t)` (suggest `−50`) to prevent underflow.
- `g_max_cap` guards against extreme `h` outliers pushing `g_i` into numerically fragile ranges. Log a warning when the cap triggers.

**Inputs**
- Module K checkpoint (same path as C0)
- Test set parquet + backbone PDBs (B1)
- B2 h_maps parquet matching the allele
- B3 sidecar JSON (required iff `amplification.h_source == h_normalized_corpus`)
- YAML config matching the schema above

**Outputs** (per run)
- `work/immune-design/if_phase_c/c1/<allele_tag>/<run_id>/generated.parquet` (columns: `protein_id`, `design_idx`, `sequence`, `seed`, `wall_seconds`)
- `generated.fasta`
- `run_config.yaml` (resolved — every default materialized)
- `manifest.json` (git SHA, checkpoint digest, h_maps source path, h_source choice, schedule form, amplification params, timestamp)
- `trajectories/<protein_id>.parquet` (optional; written iff `--save-trajectories`; columns: `step`, `t`, `unmasked_mask` [bool, L], `token_argmax` [int, L])

**Planned File Touchpoints**
- Create: `inverse_folding/reference_flow/__init__.py`
- Create: `inverse_folding/reference_flow/schedule.py` (base-schedule functions + derivatives)
- Create: `inverse_folding/reference_flow/amplification.py` (g(h) functions)
- Create: `inverse_folding/reference_flow/sampler.py` (sampler class consuming a `denoiser` callable)
- Create: `inverse_folding/reference_flow/config.py` (dataclass + YAML validator)
- Create: `inverse_folding/reference_flow/configs/{c1_null,c1_linclamp,c1_sigmoid,c1_power,c1_shuffle}.yaml`
- Create: `scripts/run_if_phase_c1.py`
- Create: `scripts/submit_if_phase_c.slurm` (`MODE=reference_flow`)
- Create: `tests/inverse_folding/test_reference_flow_schedule.py`
- Create: `tests/inverse_folding/test_reference_flow_amplification.py`
- Create: `tests/inverse_folding/test_reference_flow_sampler.py`
- Create: `tests/inverse_folding/test_reference_flow_config.py`
- Modify: `doc/SCRIPTS.md` (register under Phase C)

**CLI contract**
- Required: `--checkpoint`, `--test-set-parquet`, `--pdb-root`, `--h-maps-parquet`, `--allele`, `--config <yaml>`, `--output-root`.
- Optional: `--h-corpus-stats <B3 sidecar json>` (required iff `h_source == h_normalized_corpus`), `--n-designs-per-protein` / `--seed` / `--n-steps` (override matching config fields if present), `--device cuda|cpu` (default cuda), `--save-trajectories` (default false), `--fail-pct-threshold` (default 0.05), `--resume-from <run_id>`.
- Echo every resolved hyperparameter to stdout at start (per `feedback_print_hyperparams`).

**SLURM**
- `scripts/submit_if_phase_c.slurm` follows `scripts/submit_cnn_enhance.slurm` template; default QOS `gpu-medium`; `--output`/`--error` route to `logs/`.
- `MODE=native` launches Task C0; `MODE=reference_flow` launches Task C1.
- Incremental parquet writes required (granularity is coder's call) so preemption does not lose a full test-set pass.

**Failure handling**
1. h_maps row missing for a test protein → skip protein, log `{protein_id, reason: "missing_h_map"}`.
2. h array length ≠ sequence length → hard abort with exit code 2 (data-consistency bug, not a skip).
3. NaN in logits at any step → abort that design, log `(protein_id, design_idx, step, t)`, continue with next design.
4. GPU OOM on one protein → retry single design on CPU once; on second failure, log and continue.
5. Cumulative skip rate > `--fail-pct-threshold` → abort before overwriting final parquet; partial parquet retained under `.partial/`.

**TDD Gate**
1. RED:
   - `schedule` returns `κ(0) ≠ 0` or `κ(1) ≠ 1` for any form.
   - `amplification` returns `g(h) < 1` for any valid `(form, params, h)` triple.
   - `sampler` emits sequences with mask tokens remaining, or with length ≠ input L.
   - `config` validator accepts a config with `form ≠ constant_one` but missing `c`, or `form == sigmoid` missing `kappa`, or `h_shuffle.enabled == true` missing `seed`.
2. GREEN:
   - `schedule`: `κ(0)=0, κ(1)=1, κ'(t)>0` on `(0,1)` over a dense grid for all three base forms; analytical `κ'` vs finite-difference approximation agree within 1e-4.
   - `amplification`: `g(h) ≥ 1` over `h ∈ [−10, 10]` for all four forms; `form == constant_one` returns exactly 1 for every h; sigmoid/power satisfy documented asymptotic sentinels.
   - `sampler`: on a stub denoiser returning uniform logits, a small fixture run (a) produces mask-free outputs of correct length, (b) with fixed seed + `form == constant_one` is deterministic (two calls → identical outputs), (c) with `form == linear_clamp, c > 0` and a stratified h input, top-`g_i`-quartile positions have a higher mean unmask-step index than bottom-`g_i`-quartile positions (directional sign required; exact threshold left to the implementer).
   - `config`: accepts all 5 preset YAMLs; rejects each missing-required-field case in RED.

**Acceptance**
- All TDD gates green.
- `scripts/run_if_phase_c1.py --config inverse_folding/reference_flow/configs/c1_null.yaml` completes one allele end-to-end under `--fail-pct-threshold=0.05` and emits the full artifact bundle.
- `inverse_folding/reference_flow/sampler.py` consumes `denoiser(x_t, t, struct)` as a plain callable — no import of DPLM-specific internals inside the sampler module.

---

### Task C2: Retrain with Position-Dependent Forward Process — STUB

Deferred. This task is the **training-distribution version of the static-prior reference
flow**: same hotspot prior family as C1, but written into the forward process during
training so implicit weighting is realized. Detailed spec will be added after C1
produces baseline + ablation runs and H4a/b signal is inspected.

### Task C3: Static Hotspot Conditioning Axis — STUB

Deferred. This task adds the **WHAT-to-predict** axis on top of the base reference flow:
the frozen hotspot prior enters the denoiser explicitly (FiLM / additive features /
attention bias family). CFG may be used as an optional enhancement, but adaptive online
updates do **not** belong to C3's core definition. Detailed spec to be added after C2.

### Task D: Optional Adaptive Controller — STUB

Deferred. This task is explicitly outside the base theorem object. It covers optional
inference-time controller variants layered on top of C1/C2/C3, including:
- online hotspot refresh from the current partially generated state
- logit steering using an external hotspot field
- risk-aware revisit/corrector of already-decided positions
- dynamic schedule modulation (most invasive option; not the default first controller)

#### Task D0: Dynamic Controller Monitor, Metric, and Ablation Contract

**Goal**
- Freeze the diagnostic contract for the dynamic controller before implementation so that D2 logits guidance, D3 recommit, and their interaction are attributable, budget-aware, and externally validated.

**Scope boundary**
1. Static reference flow remains the primary theorem object and baseline contribution.
2. NetMHCIIpan remains an external evaluator only. It cannot be used by the controller.
3. D-phase success must be diagnosed from process telemetry, not only final sequence-level metrics.

**Controller surfaces**
1. D2 acts on logits through active-block hard counterfactual risk estimates.
2. D3 acts on commitment through residue-level EMA risk memory and reparameterized remask ranking.
3. The monitor must keep these surfaces separate: D2 supplies token direction; D3 supplies reversibility.

**Layer A: Opportunity / Precondition**

Purpose: decide whether the controller had anything meaningful to do.

Required metrics:
1. Static-dynamic drift: per-protein and per-window distribution of `h_dyn - h_static`. Report two scalar summaries alongside the full distribution — mean $|h_i^{\mathrm{dyn}} - h_i^{(0)}|$ (drift magnitude) and Spearman $\rho(h^{\mathrm{dyn}}, h^{(0)})$ (shape preservation). KL is inappropriate here because $h$ is a continuous risk field, not a probability distribution.
2. New hotspot rate: fraction of windows or residues that were low under static prior but active under online refresh.
3. Resolved hotspot rate: active windows that later drop below the activation threshold.
4. Risk volatility: step-to-step change in online risk field. Persistence $\int e_i(t)\,dt$ is reported only as a $\gamma$-free reference during EMA $\gamma_t$ calibration; not a headline metric.
5. Candidate feasibility rate in conservative mode. A block is feasible only if at least one candidate is both immune-improving and structure-acceptable:

$$
\Delta R_B(a_B) < -\epsilon_R
\quad\text{and}\quad
-\log Q_B(a_B) < c_B.
$$

Failure interpretation:
1. If new hotspot rate is near zero, the dynamic head is not adding targets beyond static prior.
2. Resolved hotspot rate near 0 + new hotspot rate is high: continuously finding new hotspot while not fix it.
3. If candidate feasibility is low, conservative top-$K_{\mathrm{AA}}$ support is too restrictive and bounded exploratory support should be considered before tuning controller strength.

**Layer B: Mechanism**

Purpose: decide whether D2 and D3 perform the intended operations.

D2 required metrics:
1. Directional alignment between logit correction and an independent per-AA counterfactual risk estimate:

$$
\mathrm{corr}_{a\in\mathcal{K}_i}
\left(
\Delta \ell_i(a),
-\widehat{\Delta R_i(a)}
\right).
$$

Here $\widehat{\Delta R_i(a)}$ is computed by changing only residue $i$ to amino acid $a$ in the current completed sequence, while keeping other active-block positions fixed. It must not be derived from $\pi_B$; otherwise the metric only verifies projection arithmetic.

2. Beneficial mass shift toward the lower-risk half of candidate tokens:

$$
\sum_{a\in\mathrm{bottom}_{50\%}\widehat{\Delta R_i}}
\left(
\tilde{p}_i(a) - p_i^{\mathrm{struct}}(a)
\right).
$$

3. Expected block risk improvement:

$$
\mathbb{E}_{Q_B}[\Delta R_B] - \mathbb{E}_{\pi_B}[\Delta R_B].
$$

4. KL intervention budget:

$$
D_{\mathrm{KL}}\!\left(\tilde{p}_i \,\|\, p_i^{\mathrm{struct}}\right).
$$

5. Max per-token logit shift as a safety/debug metric, not a headline metric.
6. ESS of the candidate reweighting distribution.
7. Structure cost of selected tokens: original structural rank, chosen-token log-prob drop, and top-1 flip rate.

D3 required metrics:
1. Recommit count by protein, residue, step, and active-block status.
2. Recommit localization enrichment:

$$
\frac{
P(\mathrm{remask}\mid\mathrm{active})
}{
P(\mathrm{remask}\mid\mathrm{inactive})
}.
$$

3. Productive revisit rate with immune-only, structure-only, and joint criteria. These three rates must be reported separately so the analysis can distinguish immune gain from structure sacrifice:

$$
\mathrm{immune\text{-}only}_i
=
\mathbf{1}\!\left[\Delta R_H^{\Omega\ni i}<0\right].
$$

$$
\mathrm{structure\text{-}only}_i
=
\mathbf{1}\!\left[
\ell_i^{\mathrm{post}}(a_i^{\mathrm{post}})
-
\ell_i^{\mathrm{pre}}(a_i^{\mathrm{pre}})
>
-\delta_\ell
\right].
$$

$$
\mathrm{joint}_i
=
\mathrm{immune\text{-}only}_i
\cdot
\mathrm{structure\text{-}only}_i.
$$

Default $\delta_\ell = 0.5$ nat unless later calibration suggests a different value.

4. Churn rate: repeated remask cycles at the same residue, especially late in sampling.
5. EMA half-life after a risk drop: number of refreshes required for $m_i$ to fall by 50%. If this regularly exceeds three refreshes, $\gamma_t$ is likely too sticky.
6. Same-refresh grace usage: how often D2-corrected residues are protected from immediate D3 immune penalty, and how often they are remasked at the next refresh.

Failure interpretation:
1. D2 fails if alignment is non-positive, beneficial mass does not increase, or expected risk improvement is absent despite feasible candidates.
2. D3 fails if recommit is not enriched in active regions, productive revisit is low, or churn remains high near the final steps.

**Layer C: Attribution and Budget Normalization**

Purpose: decide whether improvements are caused by the dynamic controller rather than extra sampling noise or larger intervention budget.

Required controls:
1. Monitor-only D: run online head refresh, active-block discovery, candidate scoring, and all controller scores without modifying logits or recommit ranking.
2. Paired seed lists across No-D, monitor-only, D2-only, D3-only, and full D. Pairing reduces early-step variance but trajectories diverge after the first controller-induced token difference, so final estimates still require multiple seeds per protein.
3. D2 paired counterfactual sampling: for each D2 event, record the actual corrected sample and a same-seed uncorrected sample from structural logits. Report token disagreement and local risk difference.
4. Static vs emergent edit contribution: final beneficial edits split by static hotspot, dynamic new hotspot, and non-hotspot.
5. Same-refresh D2/D3 conflict frequency: D2-corrected positions that D3 tries to remask immediately or at the next refresh. Equivalent to the same-refresh grace usage in Layer B (D3 metric 6); compute once and cross-reference rather than reporting twice.

Budget reporting is mandatory for every ablation row:
1. D2 event count.
2. Total corrected positions.
3. Total KL intervention budget.
4. Total recommit count.
5. Active block coverage.
6. Final mutation count inside active blocks.

Budget-normalized effect rates are reported alongside raw deltas:

$$
\mathrm{per\text{-}D2\ effect}
=
\frac{\Delta R_H^{\mathrm{total}}}{\#\mathrm{D2\ events}},
\qquad
\mathrm{per\text{-}recommit\ effect}
=
\frac{\Delta R_H^{\mathrm{total}}}{\#\mathrm{recommits}}.
$$

Initial experiments use soft budget reporting plus per-intervention effect rates. Hard budget matching by tuning $\eta_t$ or $\lambda_t$ is deferred because it can move each arm away from its natural operating point.

**Layer D: External Validity / Anti-Gaming**

Purpose: decide whether the controller truly reduces immunogenicity rather than exploiting its own head.

Required metrics:
1. Head drop vs NetMHCIIpan drop, globally and on active windows.
2. Local agreement between $\Delta R_H$ and $\Delta R_{\mathrm{NMP}}$ on active windows.
3. Wrong-allele head control: the wrong-allele controller should not improve the target allele's NetMHCIIpan metric.
4. Shuffled-head control: position-shuffled risk should destroy systematic targeted advantage while preserving generic heterogeneity.
5. Pareto metric: immune reduction per unit structure cost.
6. Regression rate for immune and structure outcomes:

$$
\mathrm{regression\ rate}
=
\frac{
\left|\{p: R^{\mathrm{full}}(p)>R^{\mathrm{base}}(p)\}\right|
}{
|\mathrm{test\ set}|
}.
$$

7. Right-tail Pareto quantiles, not only mean improvement.
8. Active-block amino-acid distribution shift: KL between sampled AA distribution in active blocks and the corresponding WT-restored or baseline distribution.

**Process trajectory diagnostics**

Per reverse-time decile, evaluate the completed sequence $\bar{x}_t$ and record:
1. Head risk trajectory.
2. Active window count.
3. Candidate feasibility trajectory.
4. Cumulative KL intervention budget.
5. Cumulative recommit count.
6. Structure-confidence trajectory where available.

Controller-side metrics (active window count, cumulative KL budget, cumulative recommit count) are by construction zero for $t < t_{\mathrm{start}}$ and are informative only in post-$t_{\mathrm{start}}$ deciles; the head-risk and structure-confidence trajectories remain informative across the full $t$ range.

Expected behavior is not strictly monotone at every step, but a useful controller should not show a large mid-stage immune drop followed by late-stage rebound. A late rebound points to refresh timing, final commit timing, or D3 grace/revisit settings.

**Telemetry artifact contract**

D-phase generation must emit process telemetry with stable schemas. Extra debug fields are allowed, but the following fields are the minimum required contract.

`controller_events.parquet` records one row per intervention or monitor-only candidate event:

| Field | Semantics |
|---|---|
| `protein_id` | Source protein identifier |
| `design_idx` | Design index within protein |
| `seed` | Sampling seed |
| `refresh_step` | Refresh counter |
| `step` | Sampler step index |
| `t` | Reverse-time scalar |
| `event_type` | `D2`, `D3`, or `monitor` |
| `block_id` | Active block identifier, nullable for residue-only events |
| `position_i` | Residue index for residue-level events, nullable for block-only rows |
| `window_start`, `window_end` | Local head window span when applicable |
| `a_before` | Token before event, nullable if unresolved |
| `a_after` | Token after event or proposed token (D2 events: sampled under corrected logits); nullable if no write occurred |
| `a_uncorrected` | D2 paired-attribution counterfactual: same-seed sample under structural logits; nullable for non-D2 events |
| `delta_R_corrected` | Realized local head $\Delta R^{\Omega(B)}$ after applying `a_after` (D2 actual branch); nullable for non-D2 events |
| `delta_R_uncorrected` | Realized local head $\Delta R^{\Omega(B)}$ after applying `a_uncorrected` (D2 counterfactual branch); nullable for non-D2 events |
| `paired_disagreement_flag` | Boolean: `a_after != a_uncorrected` for D2 events; nullable for non-D2 events |
| `logit_struct` | Structural chosen-token logit or log-prob before correction |
| `logit_corrected` | Corrected chosen-token logit or log-prob after D2 |
| `delta_logit_max` | Maximum absolute logit shift at the affected position |
| `kl_struct_corrected` | KL between corrected and structural token distributions |
| `delta_R_B` | Block-level counterfactual risk change when available |
| `delta_R_i` | Independent per-AA or residue-level risk change when available |
| `ESS` | Candidate effective sample size for D2 |
| `rho_B` | Block reliability gate value for D2 |
| `m_i` | EMA risk memory for D3 |
| `commit_score` | Final remask ranking score for D3 |
| `remask_flag` | Whether the event caused remasking |
| `grace_flag` | Whether same-refresh D2 grace suppressed immune penalty |
| `reason` | Controlled string reason, e.g. `low_confidence`, `immune_risk`, `monitor_only` |

`refresh_log.jsonl` records one JSON object per protein/design/refresh:

| Field | Semantics |
|---|---|
| `protein_id`, `design_idx`, `seed` | Run identity |
| `refresh_step`, `step`, `t` | Refresh location |
| `active_blocks` | List of block records with `block_id`, residue span, windows, gate values, and candidate counts |
| `h_static` | Static residue-level prior array |
| `h_dyn` | Online residue-level risk array from current completion |
| `r_windows_static` | Static WT window-level risk records matched by `(start_0b, end_0b, k)` |
| `r_windows_dyn` | Online window-level risk records from the head, matched to `r_windows_static` |
| `window_excess` | Window-level excess risk records `max(0, z_dyn - z_static - tau_W)` used for active-block discovery |
| `e_i` | Nullable residue-level excess risk array; D1 leaves this null because D3 will define the window-to-residue projection used by EMA |
| `m_i` | Nullable EMA risk memory array; populated from D3 onward |
| `candidate_feasibility` | Per-block feasibility booleans and best feasible `delta_R_B` |
| `rho_B` | Per-block reliability gate values and factor breakdowns |
| `ESS_B` | Per-block candidate ESS |
| `completion_fraction` | Fraction of resolved residues globally and per active block |
| `mean_struct_entropy` | Structural entropy globally and per active block |
| `head_risk_LME`, `head_risk_max` | Completed-sequence head risk under both aggregators |

`per_protein_summary.json` records one aggregate object per protein/design:

| Field | Semantics |
|---|---|
| `protein_id`, `design_idx`, `seed`, `allele`, `arm` | Run identity |
| `n_refreshes` | Number of online head refreshes |
| `n_active_blocks_total` | Total active blocks observed across refreshes |
| `new_hotspot_rate` | Fraction of dynamic active windows whose static risk is below `new_hotspot_static_threshold` |
| `new_hotspot_static_threshold` | Threshold used to define static absence; D1 sets this equal to `active_windows.excess_threshold` unless a later plan changes the rule |
| `candidate_feasibility_rate` | Fraction of active blocks with a feasible candidate |
| `total_D2_events` | Count of D2 logit-correction events |
| `total_D3_events` | Count of D3 commit/revisit events |
| `total_corrected_positions` | Number of position-level D2 corrections |
| `total_recommits` | Number of remasked committed residues |
| `total_KL_budget` | Sum of D2 KL intervention budget |
| `productive_revisit_immune_only` | D3 immune-only productive rate |
| `productive_revisit_structure_only` | D3 structure-only acceptable rate |
| `productive_revisit_joint` | D3 joint productive rate |
| `churn_rate` | Repeated-remask rate |
| `same_refresh_conflict_rate` | D2 positions remasked immediately or at next refresh |
| `head_regression_flag` | Whether final internal head risk worsened vs base or WT reference |
| `nmp_regression_flag` | Whether final NetMHCIIpan metric worsened vs base or WT reference |
| `structure_regression_flag` | Whether final structure metric crossed the degradation threshold |
| `active_block_mutation_count` | Final mutation count inside any active block |

**Priority metrics**

The first D-phase implementation should make these metrics available before large-scale sweeps:
1. Candidate feasibility rate.
2. Directional alignment plus beneficial mass shift.
3. Productive revisit rate plus churn rate.
4. New hotspot contribution to final edits.
5. Head-NetMHCIIpan agreement plus wrong-allele and shuffled-head controls.

**Core ablation matrix**

| Run | Arm | Purpose | v1 |
|---|---|---|---|
| 1 | No-D base | Static reference-flow or best available Phase C baseline | core |
| 2 | Monitor-only D | Same online scoring compute, no intervention | core |
| 3 | D2-only | Isolate active-block logit correction | conditional |
| 4 | D3-only | Isolate immune-aware recommit | conditional |
| 5 | D2 + D3 full | Main adaptive controller | core |
| 6 | Full + shuffled head | Negative control for position specificity | conditional |
| 7 | Full + wrong-allele head | Negative control for allele specificity | conditional |
| 8 | Full LME vs max | Aggregation sensitivity | deferred |
| 9 | Full conservative vs bounded exploratory support (opt. freezing) | Candidate support sensitivity | deferred |

`core` = first version runs; `conditional` = run if core shows signal worth decomposing/validating; `deferred` = sensitivity / paper figures only.

**Sensitivity sweeps**
1. $\beta_t$: 0.5 / 1 / 2 as the primary controller-strength sweep. Calibrate the range on a 1-protein dry-run so the smallest value yields per-event KL intervention budget $\lesssim 0.1$ nat (effectively no correction) and the largest yields KL $\gtrsim 1$ nat (near-deterministic candidate selection); adjust endpoints if 0.5–2 does not span this range under the chosen $r_W$ scale.
2. Refresh interval $K$: 3 / 5 / 10 if core runs indicate meaningful but unstable dynamic signal.
3. Static-vs-dynamic-vs-excess risk is not a primary ablation. It is a diagnostic fallback if the default excess-risk controller fails.
4. Refiner and reparameterized decoding toggles are not part of the D-phase matrix. Freeze the structure-generation substrate first; otherwise the D matrix becomes confounded with IF_IMP ablations.

**Acceptance**
1. The generation artifact for D-phase runs contains enough telemetry to compute every priority metric above.
2. The core ablation matrix can distinguish opportunity failure, mechanism failure, attribution failure, and external-validation failure.
3. Full D is not considered successful unless improvement survives monitor-only, shuffled-head, wrong-allele, and NetMHCIIpan checks.

#### Task D1: Reliability-Gated Online Refresh and Active Blocks

**Goal**
- Add the monitor-only online controller scaffold that computes hard completions, refreshes the epitope-head risk field, discovers active blocks, evaluates reliability gates, and writes D0-compatible telemetry without changing C1 logits, schedule, or remask ranking.

**Code reality from preread**
1. `inverse_folding/reference_flow/sampler.py` is currently denoiser-agnostic. `PositionDependentDFMSampler.sample()` calls `denoiser(x_t, t, struct)` once per step, receives `[L, V]` logits, samples newly unmasked positions, then optionally applies DPLM-style confidence remask.
2. `inverse_folding/reference_flow/runtime.py` wraps DPLM through `build_dplm_denoiser_context()` and `make_dplm_denoiser()`. The wrapped denoiser returns CPU logits with special tokens masked out.
3. L0090's GeoEGNN-IPA encoder replacement preserves the RF-relevant `forward_encoder()` contract: `encoder_out["feats"]` remains `[B, L_token, 512]`, `coord_mask` is present in both draft and non-draft branches, and decoder use still flows through `context.task.model.decoder(..., encoder_out=context.encoder_out)`. D1 therefore stays downstream of the encoder and must reason over decoder logits, not over the concrete encoder implementation.
4. The epitope head production API is `scripts.infer_v1.build_predictor()` plus `InferencePredictor.predict_protein(seq, allele_idx=0)`. Window scoring is batched inside one sequence through `enumerate_and_score(..., window_batch_size)`, but existing multi-protein or multi-candidate paths still loop over sequences.
5. B2 h-map artifacts store residue-level `h_raw`, `h_processed`, `global_risk`, and `n_windows`; they do not store static window logits. D1 active blocks require window-level static-vs-dynamic excess risk, so D1 must build a static WT window-score cache instead of inferring active blocks from residue thresholds alone.
6. C0/C1 `generated.parquet` schema is stable: `protein_id`, `design_idx`, `sequence`, `seed`, `wall_seconds`. D1 telemetry must be separate from generated sequences.

**Scope boundary**
1. D1 is monitor-only. It may compute refresh state, active blocks, reliability factors, and telemetry. It must return unchanged logits and must not modify `_apply_reparam_remask()`.
2. The static reference-flow schedule remains the theorem object. D1 must not update `h_values`, `g_values`, `positionwise_unmask_probabilities()`, or the C1 YAML schedule/amplification semantics.
3. The sampler must not import epitope-head modules. Head scoring lives behind a controller/scorer object passed from the C1 runner.
4. NetMHCIIpan remains external evaluation only and is not part of D1 refresh or active-block discovery.
5. D2 counterfactual logit correction and D3 EMA recommit are not implemented in D1. D1 reserves the telemetry fields that D2/D3 will fill later.
6. The GeoEGNN-IPA encoder path from `PLAN_IF_ENCODER.md` is not part of D1's implementation surface. D1 must not import GeoEGNN modules or add mandatory `--encoder-checkpoint` flags. Default RF runs continue to use `load_if_task()` with the Module K/DPLM checkpoint exactly as C1; if a future experiment needs a GeoEGNN-IPA RF arm, add an explicit optional encoder-swap helper and manifest provenance in a separate task.

**Architecture**
- Add an optional controller hook to `PositionDependentDFMSampler.sample()`. The hook is called after structural logits are produced and validated, before unmask sampling. With `controller=None`, current C1 behavior and deterministic tests must remain unchanged.
- Add a Reference Flow head-scoring adapter with a stable batch interface. The first implementation can internally loop over `predict_protein()`, but it must construct the predictor once per run and expose ordered batch results so D2 can later score block candidates without changing sampler code.
- Add a D1 controller module that owns hard completion, static window cache, active-window selection, connected active-block merging, reliability gate calculation, and telemetry buffers.

**Planned file touchpoints**
- Modify: `epitope_head/inference/predictor.py`
  - Add optional `window_batch_size` to `predict_protein()`.
  - Add `predict_proteins(records, allele_idx=0, window_batch_size=None)` as an ordered batch facade that reuses one loaded model.
- Create: `inverse_folding/reference_flow/head_scoring.py`
  - D1-specific typed schema for `WindowRiskRecord`, `HeadScore`, `BatchHeadScores`, and `OnlineHeadScorer`.
  - Static WT window-score cache keyed by `(protein_id, sequence_md5, allele, head_checkpoint_digest, head_config_hash, score_scale, window_k_min, window_k_max)`.
- Create: `inverse_folding/reference_flow/controller_config.py`
  - YAML schema for D-controller config, loaded separately through `--controller-config`.
- Create: `inverse_folding/reference_flow/controller.py`
  - `SamplerStepContext`, `ControllerStepResult`, `D1RefreshRecord`, `ActiveBlock`, and `D1MonitorController`.
- Modify: `inverse_folding/reference_flow/sampler.py`
  - Add optional `controller` argument to `sample()`.
  - Call the hook after logits validation and before sampling. D1 returns identity logits.
- Modify: `scripts/run_if_phase_c1.py`
  - Add `--controller-config`, `--head-checkpoint`, `--head-config-dir`, `--head-variant-id`, `--head-device`, and `--head-window-batch-size`.
  - Build the head predictor once per process when controller config is enabled.
  - Pass a fresh per `(protein_id, design_idx)` D1 controller instance into `sampler.sample()`.
  - Write D telemetry alongside existing generated artifacts.
- Modify: `scripts/submit_if_phase_c.slurm`
  - Expose `CONTROLLER_CONFIG`, `HEAD_CHECKPOINT`, `HEAD_CONFIG_DIR`, `HEAD_VARIANT_ID`, `HEAD_DEVICE`, and `HEAD_WINDOW_BATCH_SIZE` for `MODE=reference_flow`.
- Modify: `doc/SCRIPTS.md`
  - Register the D1 extension of `run_if_phase_c1.py` and `submit_if_phase_c.slurm`; no new script is required for v1.

**Controller config contract**

```yaml
controller:
  enabled: true
  mode: monitor_only
  t_start: 0.50
  refresh_interval: 5
  completion:
    method: argmax
  head:
    score_scale: raw_logit
    static_cache_policy: lazy_write
  active_windows:
    excess_threshold: 0.0
    max_windows: 16
    selection: threshold_then_top_n
    merge_overlapping_scoring_windows: true
  reliability:
    time_k: 20.0
    entropy_h0: 1.5
    min_completion_fraction: 0.50
    min_rho_to_emit_event: 0.0
  telemetry:
    write_refresh_log: true
    write_controller_events: true
    write_per_protein_summary: true
```

Validation rules:
1. `controller.enabled=false` must materialize a no-op config and require no head flags.
2. `mode` must be `monitor_only` for D1.
3. `t_start` must be in `[0, 1]`; `refresh_interval` must be positive.
4. `completion.method` must be `argmax` for D1 so refresh scoring is deterministic.
5. `head.score_scale` must be `raw_logit`; z-score or probability scales are not mixed into D1.
6. `active_windows.selection` must be `threshold_then_top_n`; D1 does not support residue-threshold block cutting.
7. `max_windows` must be positive and caps the number of active windows before block merging.
8. Reliability factors are clipped into `[0, 1]`, but clipping must be logged in refresh telemetry.

**D1 step behavior**
1. On every sampler step, the hook receives `x_t`, structural `logits`, `scores`, `step`, `t`, `mask_token_id`, and protein/design identity.
2. If `t < t_start` or `step % refresh_interval != 0`, the hook returns `ControllerStepResult(logits=logits, refresh_record=None)`.
3. At a refresh step, build the hard completion `bar_x_t`:
   - committed residues keep their current sampled token;
   - masked residues are filled with `argmax(logits[i])`;
   - special-token logits are already masked by the DPLM denoiser wrapper.
4. Score `bar_x_t` through `OnlineHeadScorer.score_batch_same_protein()` with exactly one sequence in D1. The interface still accepts a list because D2 will score block candidates through the same path.
5. Load or compute static WT window scores for the original test-set sequence using the same head checkpoint, config, allele, score scale, and resolved window range `(window_k_min, window_k_max)`.
6. Match dynamic and static window records by `(start_0b, end_0b, k)`. A mismatch is a hard error because it means the dynamic sequence length or head config differs from the static cache.
7. Compute per-window excess risk as `max(0, z_dyn - z_static - excess_threshold)`.
8. Select active windows by threshold first, then keep the top `max_windows` by excess risk if the threshold selects too many.
9. Convert active windows into residue spans by the window receptive field, then merge blocks whose scoring-window sets overlap.
10. Compute reliability factors for each block:
    - `g_time`: sigmoid ramp centered at `t_start`;
    - `g_comp`: fraction of block residues already committed in `x_t`, set to `0.0` when the block completion fraction is below `min_completion_fraction`;
    - `g_ent`: `exp(-mean_struct_entropy / entropy_h0)`, where entropy is computed from structural logits over canonical residue tokens;
    - `g_ESS`: `1.0` in D1 because no candidate reweighting exists yet.
11. Set `rho_B = g_time * g_comp * g_ent * g_ESS`.
12. Append a refresh record. Append `monitor` rows to `controller_events.parquet` only for active blocks with `rho_B >= min_rho_to_emit_event`.
13. Compute `new_hotspot_rate` from active windows using `new_hotspot_static_threshold = active_windows.excess_threshold`; an active window is counted as new when `z_static < new_hotspot_static_threshold`.
14. Return unchanged logits. D1 must not sample, overwrite, or remask any token.

**Telemetry artifacts**
- Keep `generated.parquet`, `generated.fasta`, `run_config.yaml`, and `manifest.json` schema-compatible with C1.
- Add under each D1 run directory:
  - `refresh_log.jsonl`: one JSON object per refresh.
  - `controller_events.parquet`: one row per emitted monitor block event.
  - `per_protein_summary.json`: one aggregate object per `(protein_id, design_idx)`.
  - `static_window_cache.parquet`: one row per `(protein_id, window_start_0b, window_end_0b, k)` with static window scores.
  - `static_window_cache.meta.json`: sidecar metadata following the B2 h-map pattern.
- Manifest additions:
  - `controller_mode`
  - `controller_config`
  - `head_checkpoint_path`
  - `head_checkpoint_digest`
  - `head_variant_id`
  - `head_config_hash`
  - `window_k_min`
  - `window_k_max`
  - `head_device`
  - `static_window_cache_path`
  - `static_window_cache_meta_path`

`static_window_cache.parquet` schema:

| Field | Semantics |
|---|---|
| `protein_id` | Source protein identifier |
| `allele` | Controller head allele label |
| `sequence_md5` | MD5 of the exact WT/static sequence scored by the head |
| `sequence_length` | Static sequence length |
| `window_start_0b`, `window_end_0b`, `k` | Window coordinates and peptide length |
| `z_static` | Raw head logit for the static WT window |
| `head_checkpoint_digest` | Digest of the head checkpoint used for the score |
| `head_config_hash` | Hash of model/inference config used to build the predictor |
| `score_scale` | Must be `raw_logit` for D1 |
| `window_k_min`, `window_k_max` | Resolved enumerated peptide-length range |

`static_window_cache.meta.json` required keys:

| Field | Semantics |
|---|---|
| `allele` | Allele label requested by the D1 run |
| `head_checkpoint_path`, `head_checkpoint_digest` | Head checkpoint provenance |
| `head_variant_id`, `head_config_hash` | Predictor construction provenance |
| `score_scale` | Must match every parquet row |
| `window_k_min`, `window_k_max` | Must match every parquet row and the dynamic head call |
| `n_proteins_total`, `n_windows_total` | Cache size accounting |
| `source_dataset`, `source_dataset_rowcount` | Test-set source provenance |
| `git_commit`, `timestamp` | Code provenance |

**Test plan**
1. `tests/epitope_head/inference/test_batch_predictor.py`
   - `predict_protein(window_batch_size=2)` matches the old output ordering and scores.
   - `predict_proteins([(id1, seq1), (id2, seq2)])` returns results in input order.
   - A fake predictor counter proves the model is constructed once and reused across the batch facade.
2. `tests/inverse_folding/test_reference_flow_head_scoring.py`
   - `OnlineHeadScorer` returns stable `WindowRiskRecord` keys and ordered batch results.
   - Static cache hit does not call the head twice for the same `(protein_id, sequence_md5, allele, digest, config_hash, score_scale)`.
   - Sequence-md5, allele, checkpoint digest, or score-scale mismatch fails fast.
3. `tests/inverse_folding/test_reference_flow_controller_config.py`
   - The D1 YAML above validates.
   - `mode != monitor_only`, invalid `t_start`, zero `refresh_interval`, non-argmax completion, and non-raw-logit score scale are rejected.
   - `enabled=false` accepts an empty controller block and does not require head flags.
4. `tests/inverse_folding/test_reference_flow_d1_controller.py`
   - No refresh is performed before `t_start`.
   - Refresh count equals the expected cadence for a fixed `n_steps`, `t_start`, and interval.
   - Hard completion keeps committed tokens and fills only masked tokens from argmax logits.
   - Single active window produces one block whose residue span equals that window's receptive-field span.
   - Two overlapping active windows merge into one block whose residue span is the union of both windows.
   - Three windows `A`, `B`, and `C` merge transitively when `A` overlaps `B` and `B` overlaps `C`, even if `A` and `C` do not directly overlap.
   - `max_windows` caps the active-window list after thresholding by keeping the highest-excess windows.
   - Reliability gate factors are in `[0, 1]`; low completion or high entropy reduces `rho_B`.
   - `completion_fraction < min_completion_fraction` forces `g_comp = 0.0`.
   - With no active windows, refresh telemetry is written and no controller event rows are emitted.
5. `tests/inverse_folding/test_reference_flow_sampler_controller.py`
   - `controller=None` reproduces current sampler outputs for the same seed.
   - A monitor-only controller that returns identity logits produces the same final tokens as no controller.
   - The hook is called after logits validation and before sampling by checking observed `x_t` and selected positions in a stub denoiser run.
6. `tests/scripts/test_run_if_phase_c1_d1.py`
   - A fake head factory is called once per process, not once per refresh.
   - A 2-protein fixture writes existing C1 generated artifacts plus D1 telemetry.
   - `generated.parquet` remains limited to the C1 row schema.
   - `run_config.yaml` and `manifest.json` include controller and head provenance.

**Acceptance**
1. `pytest tests/inverse_folding/test_reference_flow_sampler.py tests/inverse_folding/test_reference_flow_sampler_controller.py -q` passes and confirms no-controller C1 behavior is unchanged.
2. `pytest tests/inverse_folding/test_reference_flow_d1_controller.py tests/inverse_folding/test_reference_flow_head_scoring.py tests/inverse_folding/test_reference_flow_controller_config.py -q` passes.
3. `pytest tests/scripts/test_run_if_phase_c1_script.py tests/scripts/test_run_if_phase_c1_d1.py -q` passes.
4. A local fake-model fixture run of `scripts/run_if_phase_c1.py --controller-config <d1_monitor.yaml>` emits all four D1 telemetry artifacts and leaves generated sequences identical to the same fixture run without `--controller-config`.
5. The plan-level D0 priority metrics `static-dynamic drift`, `new hotspot rate`, `risk volatility`, `active block coverage`, and `head risk trajectory` can be computed from D1 artifacts without rerunning generation.

#### Task D3: Residue-Level EMA Commit/Revisit Control — STUB

Deferred. D3 is the commitment/revisit surface and must not be implemented until D1 monitor-only telemetry is stable and D2's active-block logit correction has a working candidate-risk interface.

**Deferred data-plumbing decision**
1. Current `inverse_folding/reference_flow/sampler.py` keeps `scores[i]` as the log-probability of the token sampled at the step where position `i` was last unmasked. This is the value used by `_apply_reparam_remask()`.
2. D3's commit score needs the chosen-token log-probability under the latest structural logits at a refresh step, not necessarily the old unmask-step score.
3. D1 must keep `scores[]` semantics unchanged and must not pre-reserve a `l_i_cur_at_refresh` telemetry array.
4. D3 implementation must choose one route explicitly:
   - Route A: leave `scores[]` as unmask-step confidence and compute latest committed-token log-probability from the refresh-step structural logits inside the D3 controller;
   - Route B: redefine or add a separate array for refresh-step committed-token log-probabilities, then update `_apply_reparam_remask()` tests because the remask ranking semantics change.
5. Default recommendation for D3 planning is Route A because it preserves current C1/D1 remask behavior and isolates the immune-aware commit score inside the controller.

---
