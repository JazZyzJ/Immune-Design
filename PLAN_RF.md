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

Detailed spec will be added only after the static-prior method family (C1/C2/C3) is
frozen well enough to serve as a clean baseline.

---
