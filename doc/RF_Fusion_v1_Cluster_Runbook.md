# RF-Refine Fusion v1 — Cluster Agent Runbook

AGENT TASK SPEC. Execute top-to-bottom. Stop and report on any FAIL. Do not tune a
failed gate into a positive result.

> **Current status (2026-07-25): PRE-IMPLEMENTATION — DO NOT LAUNCH.** The current
> sampler can capture a single-trajectory mid-course snapshot and fork complete rollouts,
> but it cannot yet propagate a selected partial-root population with replayable RNG
> lineage. Sections 1–5 are blocking gates. Cluster execution begins only after the
> implementation, tests, script registration, cohort manifests, and frozen config
> block all exist.

Authority order:

1. `doc/FUSION_V1.md` — scientific question and claim boundary;
2. `doc/RF-Refine-Fusion.md` — frozen v0 method and evidence;
3. this file — v1 operational experiment contract.

## 0. Scope

This runbook specifies only **V1-A: pre-terminal continuation-value allocation**.

- All editable residues begin masked. No complete parent supplies their identities.
- Entry generation is frozen to `c1_null`, `controller=None`, and no
  `H_MAPS_PARQUET`. The frozen Head scores complete continuations but never supplies a
  position-dependent generation signal.
- Deterministic beam is the selector. The frozen v0 FK result remains a negative
  ablation; do not tune or rerun FK.
- Do not activate non-null static C1/position-dependent h-map steering, D2, D3,
  SC-GR, dynamic schedules, a fuller RERD re-noising process, or explicit planning.
- The prior non-null h-map actuator did not establish value and is an independent
  retraining/extension question, not part of this coder or experiment scope.
- The terminal package is the frozen v0 repair+beam configuration. Do not reopen the
  repair study inside V1-A.
- NetMHCIIpan is post-hoc only. It is never imported by the trajectory driver and is
  never used for selection.

**Dual allele is PAPER-CRITICAL / HIGH PRIORITY / SPEC RESERVED.** It has a separate
user-directed design. Do not infer an objective law, create arms, or launch a dual
experiment from this runbook. Section 13 is intentionally only a placeholder.

## 1. Frozen scientific contract

Both P1 entry arms must share:

- target backbone and source-row identity;
- hard-anchor policy;
- DPLM checkpoint and tokenizer;
- frozen runtime Head checkpoint and inference configuration;
- sequence-length and coordinate masks;
- structure backend, absolute gate, independent-repeat evaluation, and refold cap;
- terminal repair+beam config, population size, rounds, and seed policy;
- requested protein IDs and failure accounting.

Only the following P1 contrast is authorized:

| Arm | Generation and entry law | Scientific role |
|---|---|---|
| Terminal | independent `c1_null` complete trajectories, then complete-sequence Head selection | matched terminal reference |
| Pre-terminal | the same `c1_null` trajectory, continuation-value allocation at frozen $\rho_{\mathrm{edit}}<1$, then fresh completion of selected roots | test of reward-facing allocation before completion |

Pre-terminal versus Terminal is a matched-compute system contrast, not a
clock-time-only intervention. It changes candidate correlation, value precision, whether
scored endpoints are inherited, and which state receives future compute. It does not
measure a position-dependent schedule contribution.

Head receives only canonical, complete AA20 sequences. A masked or unknown token at
the Head boundary is a hard error, even if the current Head implementation would return
a finite value. Head outputs are continuation/final scores only; they are never converted
to an h-map or fed back into denoising.

## 2. Inputs to freeze before implementation

Base paths:

```bash
BASE=/scratch/gpfs/KAIYIJIANG/zijie
PROJECT_ROOT=/home/zc1519/src/Immune-Design
WORK_DIR=${BASE}/work/immune-design
RUN_ROOT=${BASE}/run/inverse_folding/fusion_v1
LOG_ROOT=${BASE}/logs/fusion_v1
```

Required frozen inputs:

| Variable | Required source | Current status |
|---|---|---|
| `CODE_BRANCH` / `CODE_COMMIT` | a frozen descendant of `fusion_rf_refine` containing v0 Fusion and the reviewed V1 implementation | **TBD; blocking** |
| `DPLM_CHECKPOINT` | Module-K production checkpoint used by RF/Fusion | verify path and SHA256 |
| `HEAD_CHECKPOINT` | exact RAR 0031 runtime Head, including variant and allele index; complete-sequence scoring only | verify path and SHA256 |
| `ENTRY_RF_CONFIG` | both entry arms: `inverse_folding/reference_flow/configs/c1_null.yaml` | existing; verify hash and `amplification.form=constant_one` |
| `ENTRY_CONTROLLER` | both entry arms | must resolve to `None` / disabled |
| `TERMINAL_REPAIR_RF_CONFIG` | both arms' repair kernel: `inverse_folding/reference_flow/configs/c1_null.yaml` | existing; verify separately as the terminal-stage role |
| `H_MAPS_PARQUET` | position-dependent generation input | **must resolve empty/absent after launcher defaults; any non-empty value is a preflight failure** |
| `FUSION_CONFIG` | `inverse_folding/reference_flow/configs/rf_refine_fusion_final_repair_beam.yaml` on the Fusion branch | existing; verify hash |
| `TEST_SET_PARQUET` | frozen cohort manifest | **TBD; blocking** |
| `PDB_ROOT` | canonical IF-ready structure root matching the manifest | verify 100% resolution |
| `DEV_IDS` / `HOLDOUT_IDS` | disjoint protein-ID manifests | **TBD; blocking** |
| `SEED_MANIFEST` | disjoint root/est/eval/final/Terminal/full namespaces plus random-membership seed; selected/random views share root/est/eval | **TBD; blocking** |

The filename `c1_null.yaml` is retained for compatibility with the existing RF sampler,
but this is the null, non-position-dependent path: reward amplification is
`constant_one`. The entry manifest must record `controller_enabled=false` and
`h_maps_present=false`. The driver/launcher must not pass `--h-maps-parquet`; if an
existing launcher would inject its allele-default h-map when the variable is merely
unset, the V1 mode must explicitly clear it and verify the resolved command. A non-null
entry config, controller state, or h-map input is a hard preflight failure.

## 3. Cohort contract

### 3.1 Load-bearing generic cohort

Draw development and holdout IDs from the canonical DRB1*07:01 IF-ready main set.
Freeze them before trajectory outcomes are opened.

- Exclude every protein used in high-risk B1/P2 development, P3 `pilot_v3`, and RAR
  0031 `fast_v2` final integration.
- Keep development and holdout protein IDs disjoint.
- Balance on pre-existing, outcome-independent strata such as sequence length and
  dataset coverage. If generated-design burden is used for stratification, use a
  disjoint seed split and never reuse those designs for evaluation.
- Use no hard anchors in the load-bearing generic cohort.
- Determine cohort size from the T0 variance and measured cost, then freeze it before
  the P1 holdout launch. Do not invent a target $n$ in this runbook.

Persist the two manifests, source hashes, exclusion lists, and derivation command under
`WORK_DIR`. A protein absent from either the test-set parquet or `PDB_ROOT` is a
preflight failure, not a runtime omission.

### 3.2 Constrained application transfer

Q00511 or another anchored uricase may be used only after the generic P1 verdict. It is
a constrained conditional-redesign application, not a substitute for generic trajectory
evidence. The current Q00511 safety-max policy fixes 24/302 positions and leaves 278
editable.

## 4. Editable maturity and anchor telemetry

For protein $p$, define the editable set:

$$
E_p = \{i: i \text{ is editable and not a hard anchor}\}.
$$

At a snapshot, define:

$$
\rho_{\mathrm{edit}}
=
\frac{\#\{i\in E_p: x_i\in\mathrm{AA20}\}}{|E_p|}.
$$

Hard anchors and fixed motifs are excluded from both numerator and denominator. Also
report the known sequence-identity fraction:

$$
\rho_{\mathrm{known}}
=
\frac{n_{\mathrm{fixed}}+n_{\mathrm{resolved,editable}}}{L_{\mathrm{total}}}.
$$

Because reparameterized remask can make maturity non-monotone, capture the first
**post-remask upward crossing** of the requested $\rho_{\mathrm{edit}}$. Freeze the state
boundary as `x_t_phase=pre_denoiser_after_previous_remask`; the current implementation may
compute and persist structural logits afterwards, but those logits must not change the
captured `x_t`. Persist `snapshot_phase`, target and realized maturity, step, and crossing
direction.

Every root row must contain at least:

- `protein_id`, `root_id`, `root_seed`, `snapshot_step`, `rho_target`, `rho_actual`;
- `length_total`, `n_fixed`, `fixed_fraction`, `n_editable`,
  `n_resolved_editable`, `n_unresolved_editable`,
  `rho_known_sequence_identity`, `coordinate_valid_fraction`;
- fixed-position checksum and 100% anchor-preservation flag;
- unresolved editable-position mask and editable-segment length summary.

Fixed positions do not turn the run into parent seeding, but they reduce conditional
entropy. A high-constraint result must therefore be stratified by absolute unresolved
mass and descendant diversity; $\rho_{\mathrm{edit}}$ alone is insufficient.

Fail fast when `n_editable == 0`, any anchor changes, or the snapshot sequence/coordinate
length disagrees with the source row.

## 5. V1-0 — implementation readiness gate

The current `SamplerSnapshot` / `ResumeState` surface is diagnostic-only. Implement the
smallest separate partial-entry layer that satisfies all of the following. Do not weaken
the v0 complete-state `ParticleState` contract.

1. A typed partial-root state persists exact `x_t`, committed-token scores,
   unmask history, step/maturity, conditioning identity, fixed tokens, root RNG state,
   and content hashes.
2. An identity continuation from a saved root reproduces the original uninterrupted
   trajectory byte-for-byte.
3. Independent continuation seeds fork from the same partial bytes without collision.
   Equal descendants from distinct seeds are recorded as true low-entropy convergence,
   not silently deduplicated.
4. Selected partial roots, not their estimator rollouts, are propagated. Final parents
   are generated with seeds unused by estimation or T0 evaluation.
5. The partial-entry layer materializes complete AA20 rows before calling the unchanged
   v0 Fusion engine.
6. Every T0/P1 entry trajectory resolves to `c1_null`, `controller=None`, and no
   h-map; the resume payload contains no controller memory.
7. All new scripts and SLURM launchers are registered in `doc/SCRIPTS.md`; launchers
   follow `scripts/CLAUDE.md` and write stdout/stderr only under `LOG_ROOT`.

Required tests before cluster use:

- unit tests for maturity accounting with and without anchors and with a remask reversal;
- snapshot serialization and corruption/hash failure;
- identity replay and disjoint-seed fork determinism;
- strict complete-AA20 Head firewall;
- fail-fast rejection of a non-null entry config, controller state, or non-empty
  `H_MAPS_PARQUET`;
- an audit showing that complete-sequence Head scoring never changes the denoiser inputs;
- distinct-root selection and one-root-one-parent materialization;
- facade ordering and `design_idx` semantics at the v0 handoff;
- one ordinary-protein smoke and one anchor-heavy Q00511 smoke.

Run existing regression tests as well. Do not launch until all are green and the exact
new driver/SLURM names replace the placeholders in §11.

## 6. T0 — one compact maturity calibration

Question: does any predeclared late maturity contain a continuation-value signal that
generalizes to independent continuations while leaving real basin choice?

Freeze in `resolved_t0.yaml` before launch:

- a small late-maturity grid (`RHO_GRID`; values and count TBD before data inspection);
- frozen root-prefix attempt count per protein `B` and minimum valid unique-root
  capacity after equivalence collapse;
- estimator continuations per root `K_EST`;
- independent evaluation continuations per root `K_EVAL`;
- population size `N=4`, inherited from v0;
- all seed sets and the estimator definition;
- refold shortlist size and all decision margins.

Use the arithmetic mean terminal Head `global_risk` over `K_EST` complete rollouts as
the root-value estimate; lower is better. Report alternative summaries descriptively but
do not switch the selector among mean/min/median after observing results.

This is a frozen surrogate for **pre-Fusion parent quality**, not an estimate of the final
elite after eight repair+beam rounds. T0 can unlock P1, but it cannot establish that the
entry advantage will survive the terminal package.

For every maturity and protein:

1. execute exactly `B` pre-seeded root-prefix attempts from all-editable-mask
   `c1_null` with `controller=None` and no h-map; retain every failure/duplicate in
   coverage and cost, collapse valid equivalent states to the canonical unique-root
   set `U`, and never backfill after outcomes are observed;
2. require the frozen minimum `|U|`, then run `K_EST` complete rollouts per unique
   root and estimate its value;
3. select `N` distinct roots by deterministic beam;
4. evaluate every unique root through one common, disjoint `K_EVAL` continuation table;
5. compare continuation-value-selected roots with random distinct roots drawn from the
   same root pool; selected and random policies reuse the same per-root `K_EVAL` rows;
6. compare both root-allocation policies with independent complete `c1_null`
   trajectories matched on denoiser-forward equivalents and ranked by exact
   complete-sequence Head under the Terminal law;
7. definitively refold equal-size, predeclared terminal subsets from selected-root,
   random-root, and independent-full policies.

Estimator and evaluation rollouts are evidence only. None may become a P1 parent.
Branch-and-materialize is outside this V1-A runbook and coder scope.
In the seed manifest, `selected_partial` and `random_partial` are membership views over
`entry_arm=preterminal`; their policy labels never enter root/`est`/`eval` continuation
seeds. Only random subset membership has its own selection seed.

### 6.1 T0 GO/KILL rule

All four conditions must hold at one maturity:

1. root identity has reproducible out-of-sample value: estimator rank predicts the
   disjoint `K_EVAL` mean beyond within-root rollout noise;
2. selected roots beat random roots on independent continuations under the frozen
   paired decision rule;
3. selected roots are not materially worse than the compute-matched full-trajectory
   control on the structure-feasible terminal frontier;
4. meaningful action remains: a frozen minimum unresolved editable count/fraction,
   multiple unique descendants, and more than one terminal basin are observed under
   the frozen non-degeneracy rule.

Freeze statistical tests, confidence level, and practical margins from null/replicate
variation before opening arm labels. Do not choose them from the observed treatment
difference.

If several maturities pass, freeze the earliest (lowest $\rho_{\mathrm{edit}}$) passing
one. If none pass, stop V1-A and report the negative. Do not rescue it by sweeping more
maturities, increasing `K`, changing the estimator, tuning beam, or moving the gate
toward $\rho=1$.

## 7. P1 — frozen two-arm validation

Run development first. Freeze the complete P1 config and analysis code, then execute
once on the untouched holdout.

P1 must report both the parent frontier **before** the terminal Fusion loop and the final
elite frontier **after** it. This separates a failed continuation-value estimator from a
real entry signal that the terminal package later erases or saturates.

### 7.1 Entry allocation

For Terminal, spend the matched generation budget on independent complete `c1_null`
trajectories, evaluate complete Head, and rank them by exact terminal Head. Refold in
that frozen order until `N=4` feasible parents are found or the common initial-refold cap
is exhausted.

For Pre-terminal, estimate partial roots using `K_EST`, rank distinct roots by estimated
value, and materialize them in that frozen root order with fresh seeds. Refold materialized
children in root-rank order until `N=4` feasible parents are found or the same cap is
exhausted. Do not replace this with the best estimator rollout or silently reorder by
the materialized terminal Head.

Encode the intended rank as `design_idx`, because the current v0
`build_initial_population()` sorts source rows by `(design_idx, seed)` and accepts the
first `N` feasible rows; it does not perform Head top-N itself. Persist the pre-facade
rank so this behavior is auditable.

Both complete parent facades then enter the same frozen
`rf_refine_fusion_final_repair_beam.yaml` terminal loop with the same
`TERMINAL_REPAIR_RF_CONFIG=c1_null.yaml`. Entry and repair are separate provenance roles,
but both must resolve to the null config with no controller or h-map leakage.

### 7.2 Budget contract

Use denoiser-forward equivalents (DFE) as the primary entry-generation budget. Let $S$
be the total `c1_null` denoising steps, $A$ the frozen set of `B` prefix attempts,
$d_a^{\mathrm{prefix}}$ the charged prefix DFE for attempt $a$, $U$ the canonical valid
unique-root set after collapse, $s_u$ the captured next-step index of root $u$, and $F$
the frozen set of fresh final materializations. Actual Pre-terminal entry work is:

$$
C_{\mathrm{DFE,actual}}^{\mathrm{Pre}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ K_{\mathrm{EST}}\sum_{u\in U}(S-s_u)
+ \sum_{j\in F}(S-s_j).
$$

Controls must be matched to a cap frozen before continuation Head values, final-materialization
success, or structure outcomes are opened. Let $F_{\mathrm{cap}}$ be the common
facade/refold-attempt cap and

$$
r_{\max}=\max_{u\in U}(S-s_u).
$$

The reserved P1 Pre-terminal entry cap is:

$$
C_{\mathrm{DFE,reserved}}^{\mathrm{Pre}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ K_{\mathrm{EST}}\sum_{u\in U}(S-s_u)
+ F_{\mathrm{cap}}r_{\max}.
$$

Terminal receives the largest integer number of independent $S$-step trajectories whose
DFE does not exceed that reserved cap. Do not rematch it to the number of final parents that
later succeed.

T0 has no final materialization. Its reserved root-allocation work is:

$$
C_{\mathrm{DFE,reserved}}^{\mathrm{T0}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ (K_{\mathrm{EST}}+K_{\mathrm{EVAL}})
  \sum_{u\in U}(S-s_u).
$$

The `K_EVAL` term is charged once for the shared per-root evaluation table; selected/random
policy views do not duplicate it. The independent-full policy receives the largest integer
number of full trajectories not exceeding this reserved cap. Record every integer remainder,
failure, early-stop saving, and unused reservation separately; never backfill root attempts or
give Pre-terminal hidden extra trajectories.

Separately report:

- Head calls and scored residues/windows;
- initial and terminal definitive refolds, cache hits, and configured caps;
- DPLM GPU-seconds, Head GPU-seconds, ESMFold2 GPU-seconds, total GPU-hours, and walltime;
- requested, entry-feasible, terminal-success, and common-feasible protein counts.

The v0 P1 probe measured approximately 2.9 s per ESMFold2 refold at scale, and RAR
0031 measured a 1.30-fold refold and 1.66-fold logged-round-time cost for repair+beam
over its cheap baseline. These are feasibility references only. Replace projections with
T0-measured unit costs before the P1 submission.

### 7.3 P1 readouts

Primary endpoint: paired terminal-elite Head `global_risk` on common-feasible holdout
proteins, with all requested-protein coverage and failures reported alongside it.

Required secondary endpoints:

- pre-Fusion parent Head/structure/cost frontier and its retention through terminal Fusion;
- independent-repeat scTM to the same target backbone, not only cached in-loop scTM;
- structure-feasible Head frontier and structural failure tail;
- terminal NMP transfer, evaluated only after generation;
- DFE, refold, GPU-hour, and walltime frontiers;
- selected-root lineage, unique terminal sequences, editable Hamming diversity, and
  terminal-basin allocation;
- unresolved editable mass at handoff and anchor preservation where applicable.

Pre-register one primary P1 reading:

**Pre-terminal vs Terminal** asks whether reward-facing allocation before completion
improves the matched immune--structure--cost frontier on the same controller-free
`c1_null` substrate and after the same terminal v0 package. T0 selected-versus-random
is the mechanism check; it is not a third P1 arm.

Pre-terminal supports the trajectory claim only if it improves the matched immune--structure--cost
frontier, or meets a predeclared practical non-inferiority margin while demonstrating
non-degenerate pre-terminal basin allocation. A material immune, structure, coverage, or
cost regression is a negative; do not tune toward $\rho=1$ after holdout.

## 8. Optional constrained transfer — only after P1

Use the frozen P1 method and a paired terminal control under exactly the same backbone,
anchors, and budget. Do not recalibrate maturity on Q00511.

Report `n_fixed`, `fixed_fraction`, `rho_edit`, `rho_known_sequence_identity`,
`coordinate_valid_fraction`, absolute unresolved mass,
anchor preservation, descendant diversity, Head/NMP, and independent-repeat structure.
This experiment can support constrained
conditional-redesign transfer; it cannot rescue a negative generic P1.

## 9. Required artifacts

Every T0/P1 run must emit:

- `root_attempts.parquet` with all `B` prefix attempts, paid DFE,
  crossing/payload/equivalence status, and explicit failure/duplicate reason;
- `partial_roots.parquet` plus content-addressed snapshot payloads;
- `continuations.parquet` with `root_id`, seed, and `set_tag=est|eval|final`;
- `root_values.parquet`;
- `root_selection.parquet`;
- `maturity_telemetry.parquet`;
- `complete_entry_pool.parquet` for Terminal or T0 independent-full candidates;
- `t0_control_membership.parquet` with
  `policy=selected_partial|random_partial|independent_full`, shared root/evaluation-pool
  provenance, and no branch-and-materialize rows;
- T0-only `t0_structure_subset.parquet` and `t0_structure_results.parquet` for the
  equal-size definitive evaluation of those three policies;
- `terminal_parent_facade.parquet` and its ordering audit;
- `terminal_parent_admission.parquet` linking facade rank, definitive initial structure
  verdict, admitted slot, and failure;
- the standard v0 Fusion artifacts (`fusion_candidates`, `fusion_particles`,
  `fusion_lineage`, `fusion_elite`, rounds, failures, generated, manifest);
- `cost_ledger.jsonl` with DFE, Head, refold, cache, GPU, and walltime fields;
- `cohort_coverage.parquet` over every requested protein/arm/phase;
- `manifest.json`, resolved configs, input hashes, git commit, environment lock, and exact
  launch command.

Arm-inapplicable tables are emitted with their stable empty schema rather than omitted.

Resolved manifests use `arm=terminal|preterminal` and record
`entry_rf_config=c1_null`, `controller_enabled=false`, and `h_maps_present=false`.
Do not create a placeholder h-map hash.

The artifacts must replay:

```text
root bytes -> selected partial root -> fresh completion -> ordered complete facade
           -> v0 initial population -> terminal lineage and elite
```

## 10. Failure handling

- Stop after 2–3 failures of the same step. Report command, run directory, log path,
  stderr tail, and the invariant that failed.
- Never silently relax structure, anchor, maturity, refold, or statistical gates.
- A Head call containing masks/unknown residues is a code failure.
- A seed collision, snapshot hash mismatch, or replay mismatch invalidates the run.
- `insufficient_feasible_initial_population` remains a scientific coverage outcome after
  inputs and caps are verified; do not hide it by duplicating parents.
- Do not reuse partial artifacts after any checkpoint/config/tokenizer/anchor or entry-mode
  identity changes.

## 11. Environment and launch gate

Before any new launcher is used:

```bash
module purge
module load anaconda3/2025.12
conda activate immune-design
cd /home/zc1519/src/Immune-Design
git branch --show-current
git log --oneline -1
git status --short
python -m py_compile inverse_folding/reference_flow/sampler.py
python scripts/run_rf_refine_fusion.py --help >/dev/null
bash -n scripts/submit_if_phase_c.slurm
bash -n scripts/submit_refine.slurm
```

Then run the exact targeted and regression tests named by the implementation PLAN.

`V1_DRIVER` and `V1_SLURM` are intentionally unset. Once implementation lands, register
their real paths in `doc/SCRIPTS.md`, replace this paragraph with exact `--help`, dry-run,
canary, T0, and P1 commands, and record the expected git commit. Do not fabricate a
script name or submit a generic Python driver directly. Every resolved launch command must
show `ENTRY_RF_CONFIG=c1_null`, `controller=None`, an empty/absent h-map argument, and the
`terminal|preterminal` arm label.

## 12. Return

Return through `mhc-if-local` to:

```text
/Users/jerry/Project/MHC-IF/Results/RF/HLA-DRB1_0701/fusion_v1_<phase>__<timestamp>/
```

Use the standard layout:

- `generation/` — parent facades and replayable root/continuation artifacts;
- `eval_immune/` — Head and post-hoc NMP outputs;
- `eval_structure/` — in-loop and independent-repeat structure outputs;
- `analysis/` — frozen summaries and decision artifacts;
- `meta/` — manifests, configs, hashes, commands, environment;
- `logs/` — SLURM stdout/stderr copied from `${LOG_ROOT}`.

Returning experiment artifacts does not require a `LOG.md` entry.

## 13. Dual-allele placeholder — HIGH PRIORITY, DO NOT SPECIFY HERE

Dual-allele capability remains paper-critical and should proceed on its separate track.
Its objective, implementation, controls, metrics, and launch order are reserved for the
user-provided authority. No agent should fill those details, infer them from older notes,
or make them contingent on the V1-A verdict.

## 14. Final interpretation table

| Result | Allowed conclusion |
|---|---|
| T0 fails all maturities or selected roots do not beat random | no usable pre-terminal continuation-value allocation signal under `c1_null`; retain Terminal |
| T0 selected beats random but is materially worse than independent full trajectories | root value contains signal, but allocation has no matched-compute system value; do not enter P1 |
| P1 Pre-terminal improves Terminal | reward-facing population allocation before completion is useful under the frozen null-generator design |
| P1 Pre-terminal is practically non-inferior with verified non-degenerate action | pre-terminal allocation is feasible, but no terminal improvement was measured |
| P1 Pre-terminal loses Terminal or violates structure/cost/coverage margin | V1-A is negative; do not tune it toward terminal entry |
| constrained transfer succeeds after generic P1 | method transfers under hard safety constraints; do not generalize anchor-heavy behavior to unconstrained generation |
