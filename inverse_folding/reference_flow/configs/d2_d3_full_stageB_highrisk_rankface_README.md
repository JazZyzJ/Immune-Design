# Rank-face destructive control — config family

Single question (Stage 1): **does the reparam-remask rank face / proposal-write
strength have real control over structure & persistence** on the 100 high-risk
HLA-DRB1\*07:01 proteins — or is the frozen-backbone + remask dynamics a strong
attractor that no controller-side knob can move?

The reparam-remask retention rank (`commit.py:306-311`) is

```text
rank = alpha_struct * z_struct
     + (1 - alpha_struct) * z_sample
     - lambda_commit * z_m
     + d2_evidence_nu * z_d2
```

B1 / SC1 / cap20 / L1 / L2 all ran this face at the fixed conservative point
`alpha_struct=0.3, lambda_commit=1.0, d2_evidence_nu=1.0, max_abs_logit_shift=10.0`.
This family is the first systematic sweep of that face + the proposal-write valve.

## Base & cohort

- Base: `d2_d3_full_stageB_highrisk_commit_gate_L2_ttl12_nobenefit` (cap20 + L2).
  - cap20 = `d2.top_k_tokens=20` + `d2.delta_struct=1e6` + `d2.max_candidates_per_block=20`
    (full top-20 token gate; structural prefilter off).
  - L2 = `d3.d2_evidence_requires_benefit=false` + `d3.d2_evidence_ttl_steps=12`
    (most-open landing).
- Cohort: 100 high-risk (NoD-worst) proteins,
  `highrisk_dplm_iter_v1_HLA-DRB1_07_01.parquet`; n_designs_per_protein=2,
  seed=42, n_steps=100 (same as the existing L2 run).
- Paired-test baselines: the existing **L2_ttl12_nobenefit** run (= the
  alpha=0.3 / shift=10 / nu=1 / lambda=1 anchor) and **B1** (high-risk beta5/A_open).

## Arms

The anchor point (alpha_struct=0.3, max_abs_logit_shift=10, d2_evidence_nu=1,
lambda_commit=1) **is the existing L2 run — reuse it, do not re-run.**

### Stage 1 — run first (rank-structure-trust α + proposal-write valve)

| file | edit vs base | cell |
|---|---|---|
| `..._rankface_a0p0_shift10.yaml` | `d3.alpha_struct=0.0` | retention ignores structure |
| `..._rankface_a0p7_shift10.yaml` | `d3.alpha_struct=0.7` | strong structure trust |
| `..._rankface_a1p0_shift10.yaml` | `d3.alpha_struct=1.0` | structure-only retention |
| `..._rankface_a0p0_shift30.yaml` | `d3.alpha_struct=0.0` + `d2.max_abs_logit_shift=30.0` | **decoupling cell** |

The `a0p0_shift30` cell is mandatory: at shift=10 D2 can only write near-structural
tokens, so an `alpha=0` null is uninterpretable without it (proposal-write confound —
`alpha` only judges *retention* of already-committed tokens; `max_abs_logit_shift`
governs whether an off-native token gets *written* in the first place).

### Stage 2 — run only if Stage 1 shows structure/persistence move (decision A/D)

| file | edit vs base | axis |
|---|---|---|
| `..._rankface_nu0.yaml` | `d3.d2_evidence_nu=0.0` | D2 evidence credit off |
| `..._rankface_nu2.yaml` | `d3.d2_evidence_nu=2.0` | |
| `..._rankface_nu4.yaml` | `d3.d2_evidence_nu=4.0` | |
| `..._rankface_lambda0.yaml` | `d3.lambda_commit=0.0` | D3 immune-EMA revisit pressure off |
| `..._rankface_lambda2.yaml` | `d3.lambda_commit=2.0` | |

(`nu=1` and `lambda=1` anchors = the L2 run, reuse.)

## Launch

One SLURM job per file (cluster convention: stamp to
`experiment_configs/<ts>_rankface_dc/`, submit per-arm):

```text
CONTROLLER_CONFIG=<arm>.yaml CONFIG_PATH=<sampler yaml> MODE=reference_flow \
  ... sbatch scripts/submit_if_phase_c.slurm
```

(reuse the commit_gate sweep's `CONFIG_PATH` / head / cohort env). Then eval per arm
(`MODE=phase_c` → head_immune + structure).

## Pre-registered decision rules (Stage 1)

- **A** — alpha sweep moves persistence / scTM tail ⇒ rank face is live ⇒ proceed to Stage 2.
- **B** — `alpha=0 + shift30` still does NOT move scTM tail / sequence divergence ⇒
  frozen-backbone + remask is a strong attractor; the retention line cannot demonstrate
  a Pareto axis — stop small-sweeping alpha/lambda.
- **C** — structure/persistence move but immune does NOT ⇒ not a rank-face failure; the
  local-head signal (`z_m` / `z_d2`) is wrong ⇒ gate to terminal-probe rerank
  (probe as rank evidence). (L1/L2 already hinted this on the `nu` channel.)
- **D** — structure drops AND immune drops ⇒ Pareto axis found ⇒ GR may be wired to
  rank/proposal strength.

## Pre-registered λ readings (Stage 2 lambda arms)

- `lambda↑` immune down, churn not up much ⇒ D3 revisit pressure useful.
- `lambda↑` persistence down / churn up / immune flat ⇒ EMA lag or local evidence wrong;
  not "lambda unimportant", but "current `m_i` unsuitable to upweight".
- `lambda=0` more stable ⇒ D3 immune-EMA penalty is harming D2 landing.

## Required metrics (every arm)

scTM p5/p10/frac<0.5; commit proxy (persistence); later-remask fraction; rank
percentile; selected rate / disagreement; applied-shift pegged fraction; terminal
global_risk paired vs L2 and B1 (Wilcoxon).

`lambda` arms additionally: revisit_churn_rate, same-position remask count,
D2-fixed-position reopened rate, productive-revisit (immune-only / joint).

## Notes

- Diagnostic destructive-control family — NOT baselines or estimators; do not `cp` as presets.
- `shift` is tested only at `alpha=0` (decoupling cell); isolating shift-alone
  (`alpha=0.3 × shift30`) is a possible follow-up.
- `nu` sweep is predicted immune-inert (it upweights the same local-head evidence L1/L2
  moved); it confirms signal-not-weight.
- `GR→lambda_commit` and `probe→z_m` are deferred, gated on decision A/C/D — not assumed here.
