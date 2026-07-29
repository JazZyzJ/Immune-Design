# RF-Refine Fusion v1 — Cluster Agent Runbook

AGENT TASK SPEC. Execute top-to-bottom. Stop and report on any FAIL. Do not tune a
failed gate into a positive result.

> **Current status (2026-07-29): Canaries A, B, the terminal-arm smoke, AND Canary C (T0, B*+structure-handoff) EXECUTED and verified on the cluster; the scientific T0 CONFIG is frozen in §6.0A, while launch remains BLOCKED on the cohort/exclusion manifests and the preregistered analysis gate.** Both P1 arms are
> implemented and wired end to end (`preterminal` and `terminal`), the null-runtime firewall
> verifies BOTH reference-flow roles, the launch gate is fail-closed on inputs/cohort/anchors,
> S and N/R_parent/n_rounds are resolved from the frozen YAMLs with CLI cross-check, structure
> is honestly deferred to v0, and the per-protein matched budget `C_reserved` is persisted.
> §11 below carries the real dry-run and canary commands.
>
> **Cluster canary execution — 2026-07-29 (PLI H100, `--account=pli_x --qos=pli-low --partition=pli`).**
> Every canary ENTRY job and its §11.7 v0 stage ran and was verified. This proves
> deployment/identity/ledger/anchor/replay wiring ONLY — no HT1/HT2 and no arm effect (the two entry
> cohorts overlap; the numbers below are for sizing, not comparison).
>
> - **Platform constraint (load-bearing for every future run).** The idle `rtx6000` partition is
>   RTX PRO 6000 Blackwell (sm_120) and is INCOMPATIBLE with the `immune-design` torch 2.5.1+cu121
>   build (sm_50..sm_90): DPLM restores on CPU but the first CUDA kernel aborts with `sm_120 not
>   compatible` → exit 2, 0 proteins. Use a compatible GPU — PLI H100, ailab H200
>   (`--constraint=h200`), or A100 (sm_80) — and request a SHORT `--time` (entries ~2–3 min, canary
>   v0 ~8–9 min). The launcher default `--time=12:00:00` did NOT backfill (estimated 11–14 h wait);
>   `--time=00:20:00` (entry) / `02:00:00` (v0) started immediately. `--partition=gpu` is rejected by
>   the Della submit filter; the mig default is a 10 GB A100 slice (avoid for generation).
> - **Entry→v0 chain + §2.11 crosswalk verified.** Canary A (preterminal) and the terminal-arm smoke
>   each produced a 6-row facade per protein; the §11.7 v0 stage closed the crosswalk end to end —
>   `fusion_elite.entry_source_id` (e.g. `5ZHV_B:preterminal:rho0.850:r14`) →
>   `fusion_initial_admission_verdicts` → entry `terminal_parent_facade.source_id`, with NO sequence
>   join anywhere.
> - **Anchor path verified (§11.6).** Canary B (Q00511 + 24-anchor safety manifest): `maturity_telemetry`
>   `n_fixed=24`, `n_editable+n_fixed==length(302)`, and all 24 hard-anchor residues held at their WT
>   identity across every `terminal_parent_facade` (6) and `continuations` (62) row.
>   `anchor_preservation` stays NULL (the known §11.6 gap — not a measurement).
> - **Sizing telemetry (feeds §3/§6/§7 caps; NOT arm effect).** Entry crossing 30/32 and 14/16
>   (~10 % `no_crossing` — over-provision `prefix_attempts` accordingly); |U|/B ≈ 0.88–1.00; facade
>   filled `F_cap=6`/6 every run; entry DFE ≈ 1.6 k/protein (root_prefix-dominated), 0 refolds
>   (structure deferred); persisted `C_reserved` ≈ 1.6 k DFE/protein. v0: ~120–145 refolds per
>   successful protein over 8 rounds, ~8–9 min walltime for a 2-protein canary.
> - **Feasibility / coverage finding (sizing).** At `F_cap=6`, `N=4`: 5ZHV_B produced an elite in
>   BOTH arms; 9L2Q_A hit `insufficient_feasible_initial_population 0<4` in BOTH arms (a per-protein
>   foldability outcome, NOT an arm effect); Q00511 (anchored) reached `2<4`. These are legitimate
>   §10 coverage outcomes — the null pre-terminal feasibility rate at ρ_edit≈0.85 is low enough that
>   the scientific run needs a larger `F_cap`. **Known gap:** because Q00511 never reached `N=4`, the
>   anchored v0 REPAIR loop was not exercised, so v0-side anchor preservation THROUGH repair is not
>   yet demonstrated on the cluster.
> - **Canary C (T0 B* + integrated structure handoff) — 2026-07-29, PLI H100, 16m18s, exit 0.**
>   Both proteins reached `t0_complete`. All `72` structure requests (2 proteins × 3 rho × 3
>   policies × Q_T0=4) were REALLY evaluated (real scTM 0.33–0.92, per-policy gate-pass 0.46–0.50),
>   with exactly Q_T0=4 verdicts per `(protein, rho, policy)` and the `independent_full` pool = the
>   F_cap=6 frontier per rho. The wide smoke grid resolved separated resume points (.50/.70 crossed
>   16/16, .85 14–15/16). This is a WIRING canary — scTM/gate-pass are sizing evidence, NOT a
>   GO/KILL number; the scientific grid/`K_EVAL`/`Q_T0`/cohort remain frozen from §6 before a real T0.
> - **T0 sizing inputs from Canary C (2 proteins, 100–105 aa; feed §3.1/§6/§7, do NOT over-read).**
>   - *Cost model.* Per protein per rho: reserved partial-root allocation ≈ 1.7k–2.3k DFE
>     (root prefix ≈1.5k plus estimator/evaluation tails); the matched `independent_full` control
>     spends approximately the same reservation, giving **≈3.9k total logical DFE per
>     protein-rho**. Each point also uses `3*Q_T0=12` structure requests, of
>     which ~11 are real folds (a few deterministic-completion cache hits). Whole run: 23214 DFE +
>     66 real ESMFold2 folds for 2×3 points, **16m18s gross** (incl. one DPLM+ESMFold2 load).
>     **Unit costs are still ANCHORS, not this-round measurements**: the oracle ledger recorded
>     `walltime_s=0` per event (a pre-existing gap; the structure evaluator now times itself, so the
>     real dev-cohort T0 will measure `seconds_per_refold`/`seconds_per_dfe` cleanly). Use
>     `seconds_per_refold≈2.9` (RAR 0031, at scale) and `seconds_per_dfe≈0.02` (Canary A actual /
>     c1_null b4 gen) as the current projection anchors — length-dependent, so a longer cohort costs
>     more per protein.
>   - *Outcome-independent maturity telemetry (the §11.5 basis for freezing the scientific rho grid,
>     NOT a treatment effect).* The reparam-remask sampler makes ρ_edit rise steeply at the end:
>     rho_target 0.50/0.70/0.85 are captured at ρ_actual **0.52/0.73/0.89**, mean step **94/97/99**
>     of 100, mean **unresolved editable ≈ 49/27/11** positions. So the three points ARE distinct in
>     remaining action but all land in the last ~6 steps; a maturity with materially more action
>     left needs ρ_target well below 0.50. (Only 2 proteins — a hint, not the frozen basis.)
> - **What Canary C does and does NOT determine for the scientific run.** It confirms the pipeline
>   runs end to end and gives the cost/feasibility MODEL + the maturity-vs-step relationship above,
>   so the Thinker can freeze the **T0** config (rho grid, `K_EVAL`, `Q_T0`) from §6. It does NOT
>   give the **P1 cohort size** — §3.1 sizes that from the T0 *variance*, which needs a real
>   dev-cohort T0 (2 wiring proteins carry no variance) — nor the frozen statistical tests/margins
>   (§6.1). Sequence: freeze T0 config → run the dev-cohort T0 (measures unit costs + variance) →
>   size and freeze P1.
>
> Both P1 arms and T0 are implemented, wired and canary-configured; null isolation is verified at
> the sampler (not only in the config). T0 now emits the standard §9 evidence tables, keeps each
> `rho_grid` point in its own matched root-attempt group, Head-ranks its independent-full control,
> and reports `t0_structure_deferred` rather than success when nothing was structurally evaluated.
>
> **Known gaps. Every one of these is a reason a canary may NOT be read as evidence:**
> 1. ~~T0 structure is deferred, not evaluated~~ — **CLOSED (2026-07-29).** `build_entry_oracles`
>    now wires an INTEGRATED evaluator for `phase=='t0'` (the same v0 `esmfold2_live` refold +
>    TMalign + absolute scTM floor, same on-disk cache identity); P1 arms still defer to v0 (§2.11).
>    Canary C evaluated every `3*Q_T0` request (72/72 `evaluated`, real scTM) and reached
>    `t0_complete`, so GO/KILL condition 3 is now computable from a T0 run.
> 2. ~~The `independent_full` eligible pool implementation still uses all survivors~~ — **CLOSED
>    (2026-07-29).** `run_t0_protein` now uses the policy-faithful B\* law (§6.0): `independent_full`
>    = exact-Head-ranked frontier `full_pool[:F_cap]`; `selected/random` root-balanced (one held-out
>    endpoint per held root, content-hash, shared-cache reuse); config gates `Q_T0==N`, `Q_T0<=F_cap`,
>    `Q_T0<=N*K_EVAL` + runtime `survivors>=F_cap`. Canary C exercised it (independent_full pool =
>    F_cap=6 per rho; exactly Q_T0=4 verdicts per policy per rho).
> 3. **The v0 terminal stage is a SECOND job that the runbook now specifies (§11.7).** The entry
>    driver stops at the facade; without §11.7 no arm reaches v0, and `submit_refine.slurm`'s
>    default `FUSION_CONFIG` is the SMOKE package, which runs to completion and emits a plausible
>    elite from the wrong method.
> 4. **`maturity_telemetry.anchor_preservation` is null and is not a measurement** (§11.6).
> 5. ~~v0 admission records no rejection rows~~ — **CLOSED.** `build_initial_population` now
>    records every examined row (admitted or rejected, with the definitive gate reason and its
>    scTM/pLDDT) and counts the round-0 folds; both reach
>    `fusion_initial_admission_verdicts.parquet` and `manifest.initial_refolds_by_protein`. This
>    closes the `facade design_idx -> admission attempt and verdict -> initial slot / particle_id`
>    edge of §2.11 — the ENTRY stage could never close it, because its own gate defers.
> 6. **A crash inside an oracle loses that call's already-burned cost.** The ledger event is
>    appended after the oracle returns, so an exception mid-call leaves no row. Loud (the protein
>    fails, and it is excluded from `n_proteins_ok`), and the LOGICAL DFE is re-emitted by the
>    retry — only physical forwards/walltime are lost. But it violates §3.3's "never silently
>    dropped", and the bound is **not one call**: `admit_facade` runs the whole structure scan
>    before any append, so a gate raising on candidate *k* loses *k* real structure attempts —
>    bounded by `F_cap`, not by 1. That path costs nothing today only because the production gate
>    defers with `walltime_s=0`; it becomes real the moment gap 5 is wired. The arms are also
>    asymmetric here: the terminal trajectory generator catches and charges full `S`, the
>    pre-terminal completer catches nothing. The real fix is at the oracle contract — a failing
>    oracle must report the cost it burned — and is NOT done. Booking a zero instead would be a
>    fabricated measurement, so nothing is booked.
>
> **Consequence.** Canary A + §11.7, Canary B + §11.7, the terminal-arm smoke, AND Canary C (T0,
> 2026-07-29) have all executed. A passing canary proves deployment, identity, ledger, anchor,
> replay and — for Canary C — the B\*/structure-handoff wiring only; **never HT1/HT2**. Three
> launcher/env integration gaps surfaced only when T0 first ran on GPU and are fixed: §11.5 now
> passes `REFOLD_CACHE_DIR`/`ESMFOLD2_SITE_PACKAGES`; the `v1_entry` launcher branch points
> `HF_HOME`/`TORCH_HOME` at the shared `model_cache/hf` so the offline ESMFold2 worker resolves
> `biohub/ESMFold2`; and the entry oracle takes the per-request maturity from `rho_id` (a T0 grid
> varies rho while the oracle is built once — the P1 single-`rho_target` path is unchanged).
>
> **Still blocking a scientific run** (not a canary): the 24-protein T0 development cohort,
> prior-cohort exclusion manifest, four fixed six-protein shard manifests, and the preregistered
> T0 analysis command/output do not yet exist. The scientific T0 knobs are frozen in §6.0A and
> `rf_fusion_v1_entry_t0_dev.yaml`; `SECONDS_PER_REFOLD=2.9` and `SECONDS_PER_DFE=0.02` remain
> conservative launch anchors and must be replaced by measured T0 values before P1. No SCIENTIFIC run has
> launched — the canaries executed 2026-07-29 (see the execution block above), but a canary is a
> contract smoke only; do not read any treatment effect out of it.

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

- the frozen maturity grid `RHO_GRID=[.30,.50,.70]`;
- frozen root-prefix attempt count per protein `B` and minimum valid unique-root
  capacity after equivalence collapse;
- estimator continuations per root `K_EST`;
- independent evaluation continuations per root `K_EVAL`;
- population size `N=4`, inherited from v0;
- all seed sets and the estimator definition;
- refold shortlist size and all decision margins.

Canary A entry sizing provides one important operational constraint but does not freeze the
scientific grid. On `5ZHV_B` and `9L2Q_A`, `rho_target=.85` produced 32/32 crossings and 30/32
unique roots, but used `3168/32=99` prefix DFE with `S=100`; the estimator continuations used
`128/(30*4)=1.07` tail DFE on average. Thus `.85` was a one-step root under this sampler schedule.
The last forward may still resolve many editable residues, so this does not prove that the state
is degenerate. It does prove that nominally different late rho targets may collapse onto the same
final maturity jump.

The scientific `RHO_GRID` is frozen from a dedicated OUTCOME-INDEPENDENT maturity scan
(`scripts/rho_maturity_scan.py`; frozen machine-readable evidence at
`run/inverse_folding/fusion_v1/rho_maturity_scan/rho_maturity_scan.parquet`), NOT from the
2-protein canary — the canary only proved the pipeline runs, never where rho should be (§11.5). The scan swept `[.20,.30,.40,.50,.70,.85]` on four
length-stratified generic proteins (90/150/210/280 aa), reading ONLY crossing/unique-root coverage,
snapshot step, ρ_actual, unresolved editable mass, and Head-free fork descendant diversity — never
Head or structure. Findings are length-invariant: every target crosses 8/8 with full unique-root
coverage (even `.20`), so low rho IS usable; the reparam schedule compresses all crossings into the
last ~16 steps, but unresolved mass and fork branching separate the targets cleanly —
unresolved ≈ 79/69/58/47/25/11 % and fork-diversity ≈ 0.38/0.37/0.34/0.28/0.16/0.08 at
ρ=.20/.30/.40/.50/.70/.85. `.85` (11 % unresolved, fork 0.08) is effectively best-of-K, not
pre-terminal allocation, and `.70` sits in the low-branching regime. The frozen grid `[.30,.50,.70]`
therefore spans the fidelity×influence tradeoff with three genuinely-separated states: `.30`
(69 % unresolved, fork 0.37, genuinely pre-terminal), `.50` (47 %, 0.28, balanced), `.70` (25 %,
0.16, near-terminal boundary). The scientific T0 must still verify non-trivial unique-descendant
branching and multiple terminal basins under §6.1, and lets condition 4 reject a target whose
separation does not generalize to the 24-protein cohort. Do not add a lower rho after opening T0
outcomes.

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

### 6.0 FROZEN DECISION — policy-faithful B*

Step 7 refolds equal-size, predeclared terminal subsets from three different allocation
policies. The common requirement is the **same definitive sample size and Head-independent
within-pool sampling rule**. It is not a requirement that all three methods manufacture their
eligible pool with the same selector.

Freeze the eligibility law as:

| Policy | Eligible structure pool |
|---|---|
| `selected_partial` | all common `K_EVAL` endpoints of the `N` value-selected roots |
| `random_partial` | all common `K_EVAL` endpoints of the `N` random roots |
| `independent_full` | valid matched full trajectories ordered by exact complete-sequence Head, restricted to `full_pool[:F_cap]` |

This is **B\*** rather than either earlier reading:

- all-survivor A removes the Terminal method's defining endpoint Head allocation and weakens the
  control, biasing condition 3 toward a false GO;
- applying top-`F_cap` terminal Head truncation to the partial pools would give Pre-terminal an
  extra endpoint-selection stage after its root-value selection. That is not the P1 method and
  would confound the question by testing a double-selected partial policy.

After these policy-faithful pools exist, freeze `Q_T0=N=4`. Partial-policy structure selection is
root-balanced: one held-out endpoint per held root, chosen by a stable Head-independent content
hash. If selected/random share a root, they reuse the same endpoint. For `independent_full`, use
the same Head-independent content ordering to choose `Q_T0` rows from the top-`F_cap` frontier.
The all-survivor full-pool Head/coverage distribution may be reported descriptively, but it is
not the condition-3 comparator.

Hard gates, evaluated before spending the corresponding generation/structure budget:

- `Q_T0 == N`;
- `Q_T0 <= F_cap`;
- `Q_T0 <= N*K_EVAL`;
- valid `independent_full` survivors `>= F_cap`;
- every policy produces exactly `Q_T0` definitive structure verdicts, with no shrink or backfill.

**Implementation/Canary closure (2026-07-29).** All five former coder items are implemented and
covered by contract tests. Canary C exercised the final B* law and produced exactly four
definitive verdicts per policy and grid point. It remains a wiring canary and publishes no GO/KILL
number.

Estimator and evaluation rollouts are evidence only. None may become a P1 parent.
Branch-and-materialize is outside this V1-A runbook and coder scope.
In the seed manifest, `selected_partial` and `random_partial` are membership views over
`entry_arm=preterminal`; their policy labels never enter root/`est`/`eval` continuation
seeds. Only random subset membership has its own selection seed.

### 6.0A FROZEN SCIENTIFIC T0 CONFIG — 2026-07-29

The scientific T0 config is
`inverse_folding/reference_flow/configs/rf_fusion_v1_entry_t0_dev.yaml`. Do not edit it after the
first scientific shard starts. Freeze the following values:

| Quantity | Frozen value | Basis |
|---|---:|---|
| development proteins | 24, generic anchor-free DRB1*07:01 | gives 24 paired protein units and 96 structure verdicts per policy/rho |
| execution layout | 4 fixed shards × 6 proteins | bounds one failed job without changing the frozen cohort |
| `rho_grid` | `[.30, .50, .70]` | maturity scan (`scripts/rho_maturity_scan.py`, 4 lengths): separated in unresolved mass (~69/47/25 %) AND fork branching (~0.37/0.28/0.16); `.85` dropped as near-terminal best-of-K (fork ~0.08) |
| `B=prefix_attempts` | 16 | Canary A/C crossing and unique-root yield |
| `unique_root_capacity` | 12 | coverage gate; also supports the common `F_cap` |
| `K_EST` | 4 | frozen estimator budget used by the canaries |
| `K_EVAL` | 8 | stronger held-out estimate than the smoke; cheap at the measured late tails |
| `N` | 4 | frozen v0 population |
| `F_cap` | 12 | common future P1 cap; at Canary-C pass rates 0.46–0.50, an iid sizing model gives ~0.88–0.93 probability of at least four feasible rows |
| `Q_T0` | 4 | equals `N`; policy-faithful B* structure denominator |
| estimator | arithmetic mean Head `global_risk` | frozen scientific definition |
| shard caps | 800000 DFE, 240 refolds, 21600 s | worst-case six-protein preflight at `S=100`, plus startup margin |
| launch unit-cost anchors | 0.02 s/DFE, 2.9 s/refold | the values the T0 shards launched under; **superseded by the measured T0 costs in §6.1A — use those for P1** |

The 24 IDs must be drawn and frozen under §3.1 before any scientific shard runs. Stratify on
pre-existing length/coverage only; persist the exclusion list for P2/P3/RAR0031 proteins and four
non-overlapping six-ID shard manifests. All four shards use the byte-identical config, model
digests and analysis commit. Opening one shard's treatment results before the remaining shards
are irrevocably submitted is prohibited.

This freezes only T0 and the shared method constants needed to define its Terminal comparator.
It does **not** freeze the P1 cohort size, dev/holdout split size, primary practical margin, or
winning maturity. Those are frozen after T0 from measured variance/cost; the winning P1 maturity
is the lowest rho that passes every §6.1 condition.

### 6.0B FROZEN T0 DEVELOPMENT COHORT — 2026-07-29

Drawn and frozen by `scripts/freeze_t0_dev_cohort.py` under §3.1. **This section is the human
record; the machine-readable manifests are the single source of truth and live only at**
`WORK_DIR/fusion_v1/` (`t0_dev_cohort.{parquet,csv}`, `t0_dev_exclusions.{parquet,csv}`,
`t0_dev_shard_0{0..3}.ids`, `t0_dev_provenance.json`). Do not re-draw after the first shard starts.

**Provenance (frozen).** `master_seed=20260729`; deterministic order = `sha256("<tag>|seed|id")[:8]`
big-endian (never Python's salted `hash()`); source table
`if_ready/main/test_proteins_if_ready_HLA-DRB1_07_01.parquet` `sha256=366767db…87488` (2879 rows);
`cohort_sha256=9a2bb600…9ab4f38`; derivation git `6b80757`.

**Selection accounting.** pool 2879 → excluded-in-main 673 → eligible 2206 → exact-sequence dedup
collapses 231 → 1975 representatives → 6 per WT-length quartile (edges 155/229/314.5; eligible reps
per stratum 489/498/494/494) → **24**, round-robined into four length-mixed six-ID shards. Integrity
drops 0 (every pool protein is AA20-complete, length-consistent, structure-resolvable). Homology:
this set carries no CATH/mmseqs cluster labels and mmseqs is not installed, so **exact WT-sequence
identity is the frozen one-per-cluster proxy**; `head_train_overlap_flag` is recorded but is NOT an
exclusion criterion (§3.1 excludes dev/pilot/integration membership + homology, not Head-training
membership).

**Exclusions (per §3.1, deliberately more conservative).** dataset IDs removed from the pool:
`highrisk_nod` 100, `highrisk_pmpnn` 100 (B1/P2 high-risk dev), `pilot_v2` 47, `pilot_v3` 50 (P3),
`fast_v2` 286 (RAR0031 integration); Canary A/C `{5ZHV_B, 9L2Q_A}`; maturity-scan
`{2O4T_A, 5YAA_B, 3O1Q_C, 7V2T_A}`; plus 183 main-pool proteins sharing an exact WT sequence with
any excluded protein (near-duplicate homologs). Union = 673.

**Frozen shards** (byte-identical config; each spans all four length strata; length in aa):

| Shard | Protein IDs (WT length) |
|---|---|
| `shard_00` | 6EXP_F(103) 1NQ3_F(133) 6G6Q_H(160) 7JM0_B(274) 6QT8_A(289) 9BZ4_D(442) |
| `shard_01` | 6QWV_K(148) 1KHI_A(147) 7B4B_D(186) 2DXT_B(235) 4P3I_D(289) 3KV3_R(334) |
| `shard_02` | 3SZ6_A(116) 8I18_B(156) 7FF9_A(197) 6D2V_B(306) 8WVR_A(323) 8AGR_B(325) |
| `shard_03` | 3BD4_A(112) 2ABL_A(163) 7NDP_D(201) 5UA0_C(274) 8H0C_A(387) 2BWN_A(396) |

All four shards pass login-node `--print-config` + `--dry-run` (`n_valid=6/6`, `anchors=0`, DFE
748800<800000, refold 216<240, projected walltime 15602 s<21600 s, null-kernel firewall clean). The
frozen HEAD is `run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt`
(`LC1`) and the DPLM base `run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/…/best.ckpt`.

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

The preregistered analysis command must be implemented and committed before launch as
`scripts/analysis/rf_fusion_v1_t0_gate.py`; it writes one immutable JSON verdict plus the
protein-level table. It must apply the following frozen rules independently at each rho:

1. **Coverage first:** at least 20/24 requested proteins must have complete root/eval/structure
   evidence. Report all 24 and every failure; never analyze only successful rows.
2. **Value reliability:** compute per-protein Spearman correlation between `K_EST` root value and
   the disjoint `K_EVAL` mean. Pass only when the median correlation is positive and a 10,000-draw
   within-protein root-label permutation test remains significant after Holm correction across
   the three rho values at family-wise `alpha=0.05`.
3. **Selected over random:** for each protein compute mean held-out Head risk of selected minus
   random roots from the shared `K_EVAL` table. Pass only when the median delta is negative and a
   10,000-draw protein-level sign-flip test remains significant after Holm correction across the
   three rho values at family-wise `alpha=0.05`.
4. **Structure non-inferiority:** compare selected-partial with independent-full using the frozen
   definitive gate-pass denominator. The one-sided 98.33% protein-cluster bootstrap lower bound
   (Bonferroni family-wise 0.05 across three rho values) for
   `pass_rate(selected)-pass_rate(independent_full)` must exceed `-0.10`. Report scTM deltas
   descriptively; do not substitute them for the gate.
5. **Meaningful action:** median unresolved editable count is at least 10 **and** median unresolved
   editable fraction is at least 0.10; at least 75% of selected roots have two or more distinct
   `K_EVAL` terminal sequences; and, on at least 75% of complete-evidence proteins, median
   between-root editable Hamming distance exceeds median within-root distance.

These tests, family-wise confidence levels and the `-0.10` structure margin are frozen before
scientific arm labels are opened. Do not choose an alternative test from the observed difference.

If several maturities pass, freeze the earliest (lowest $\rho_{\mathrm{edit}}$) passing
one. If none pass, stop V1-A and report the negative. Do not rescue it by sweeping more
maturities, increasing `K`, changing the estimator, tuning beam, or moving the gate
toward $\rho=1$.

### 6.1A T0 EXECUTED — measured costs and the pre-registered verdict (2026-07-29)

The four frozen shards ran to completion on `rtx6000` (RTX PRO 6000 Blackwell, `della-h23g1`) under
the `immune-design-blackwell` env (torch 2.7.1+cu128); jobs `11771059-62`, 50–57 min each. Every
shard exited 0.

**Execution evidence (structural, not treatment).** 24/24 proteins `t0_complete`, matching the
frozen cohort exactly; **864/864** definitive structure requests `evaluated` (= 24 × 3 rho × 3
policies × `Q_T0`=4) with zero `failure_reason`; all 216 `(protein, rho, policy)` cells hold exactly
`Q_T0`=4 verdicts — the B\* law held with no shrink or backfill; the `config_digest`
`570a422b…d589be` is **identical across all four shards**, which is what makes them one experiment.

**Measured unit costs (these supersede the §6.0A launch anchors).**

| Quantity | Measured | Launch anchor |
|---|---:|---:|
| seconds per definitive refold | **2.42** (1893 s of self-timed structure walltime / 782 real folds; 82/864 served from cache) | 2.9 |
| seconds per DFE | **≤0.029** (aggregate 12915 s job walltime − 1893 s structure, over 381686 logical DFE; **includes model startup**, so it is an upper bound) | 0.02 |

The DFE anchor is the optimistic one: P1 walltime projections should use ~0.03 s/DFE, not 0.02.
Refolds are ~17 % cheaper than assumed. Actual per-shard walltime (≈3.2 ks) came in far under the
21.6 ks cap because the cap is a worst-case projection at `max_dfe`, while realized DFE was 381686
for the whole cohort.

**Pre-registered gate output.** `scripts/analysis/rf_fusion_v1_t0_gate.py` was run **once** on the
four-shard aggregate (`--n-resample 10000 --seed 20260729`); the immutable verdict and the
24-protein × 3-rho table are at `<run>/t0_dev_v1/analysis/`. Per-rho results:

| rho | coverage | value reliability (median $\rho_s$, Holm $p$) | selected−random (median $\Delta$, Holm $p$) | structure LB vs −0.10 | meaningful action | GO |
|---|---|---|---|---|---|---|
| 0.30 | 24/24 | 0.149, 0.0135 ✅ | −0.043, 0.311 ❌ | **−0.156** ❌ | med $|U|$=144, frac 0.683 ✅ | ❌ |
| 0.50 | 24/24 | 0.287, 0.0003 ✅ | −0.643, 0.0020 ✅ | −0.042 ✅ | med $|U|$=101.5, frac 0.471 ✅ | ✅ |
| 0.70 | 24/24 | 0.724, 0.0003 ✅ | −1.838, 0.0006 ✅ | 0.000 ✅ | med $|U|$=57.5, frac 0.247 ✅ | ✅ |

Two maturities pass, so §6.1's "freeze the earliest passing" rule selects **`rho=0.50`**. Rho 0.30
fails on two independent counts (no separable selected-vs-random signal, and its structure lower
bound breaches the frozen −0.10 margin), i.e. the pre-terminal state is too immature there to carry
continuation value. Rho 0.70 shows the largest effects but the least remaining action.

**Gate verification (why these numbers are trustworthy).** Before the gate was committed, all five
rules were re-derived from the raw parquet by an independent implementation and matched exactly:
coverage 24/24; median Spearman 0.148529/0.286765/0.723529 (cross-checked against
`scipy.stats.spearmanr`, which also agrees with the script's own tie-averaged ranker to ~1e-16 on
tie-heavy inputs); median delta −0.042775/−0.643198/−1.837647; observed structure diff
−0.06250/0.00000/0.03125; median unresolved 144.0/101.5/57.5. Re-running with the same seed
reproduces a **byte-identical** verdict. Independently, the gate's measured unresolved fractions
(0.683/0.471/0.247) reproduce the frozen maturity scan's (0.689/0.471/0.251) through a completely
different code path — the scan read the live sampler, the gate reads the persisted tables.

This freezes T0 only. The P1 cohort size, dev/holdout split sizes and primary practical margin are
still open and must be set from the measured variance and costs above.

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

`V1_DRIVER` = `scripts/run_rf_fusion_v1_entry.py`, `V1_SLURM` = `scripts/submit_if_phase_c.slurm`
with `MODE=v1_entry` (both registered in `doc/SCRIPTS.md`). Never submit the Python driver
directly on a compute node — the launcher owns the environment and the null-runtime firewall.

### 11.1 Model-free checks (login node, no GPU)

```bash
python scripts/run_rf_fusion_v1_entry.py --help >/dev/null
bash -n scripts/submit_if_phase_c.slurm
python -m pytest tests/scripts/test_rf_fusion_v1_null_firewall.py \
                 tests/scripts/test_rf_fusion_v1_launch_gate.py \
                 tests/scripts/test_rf_fusion_v1_terminal_arm.py -q
```

### 11.1b OUTPUT ISOLATION — set OUTPUT_ROOT on EVERY V1 command

The launcher defaults to `OUTPUT_ROOT=${RUN_DIR}/if_phase_c/${MODE}`, and every V1 entry run shares
`MODE=v1_entry`. Two runs that do not override it write into one directory. Per-protein checkpoints
are now namespaced by phase and arm (`{protein}__{phase}__{arm}.json`), so they no longer collide —
but the AGGREGATED artifact family (`*.parquet`, `cost_ledger.jsonl`, `manifest.json`) is written
per run and WOULD be overwritten. That is not merely untidy: the Terminal arm reads its
`C_reserved` out of the pre-terminal run's `manifest.json`, so a Terminal run pointed at the same
`OUTPUT_ROOT` overwrites the very manifest it consumed, and the run becomes unreproducible.

**Give every command below its own `OUTPUT_ROOT`**, e.g.

```bash
OUTPUT_ROOT=${RUN_ROOT}/canary_a_preterminal      # §11.2
OUTPUT_ROOT=${RUN_ROOT}/canary_terminal_smoke     # §11.3  (reads canary_a's manifest.json)
OUTPUT_ROOT=${RUN_ROOT}/canary_c_t0               # §11.5
OUTPUT_ROOT=${RUN_ROOT}/canary_b_q00511_preterminal  # §11.6
```

### 11.2 Canary A — PRE-TERMINAL arm

`V1_SUBMODE=dry_run` first; it loads no model and exits non-zero on any gate failure.

```bash
MODE=v1_entry V1_SUBMODE=dry_run \
OUTPUT_ROOT=${RUN_ROOT}/canary_a_preterminal \
ENTRY_CONFIG=inverse_folding/reference_flow/configs/rf_fusion_v1_entry_canary.yaml \
TERMINAL_REPAIR_CONFIG=inverse_folding/reference_flow/configs/c1_null.yaml \
RF_SAMPLER_CONFIG=inverse_folding/reference_flow/configs/c1_null.yaml \
FUSION_CONFIG=inverse_folding/reference_flow/configs/rf_refine_fusion_final_repair_beam.yaml \
V1_PROTEINS="<dev ids>" S_STEPS=100 R_PARENT=8 N_ROUNDS=8 \
SECONDS_PER_REFOLD=<measured> SECONDS_PER_DFE=<measured> \
HEAD_CHECKPOINT=<path> HEAD_VARIANT_ID=<id> \
sbatch scripts/submit_if_phase_c.slurm
```

Re-run with `V1_SUBMODE=run` once the dry run reports `ok: true`. The job's exit code is the
cohort outcome: `0` = every requested protein produced a terminal parent, `2` = nothing usable,
`3` = partial cohort. Do NOT set `H_MAPS_PARQUET` or `CONTROLLER_CONFIG`; the branch refuses a
caller-supplied value with exit 2, and leaving them unset is what clears the allele default.

### 11.3 TERMINAL-arm coverage smoke

**This is NOT the PLAN's Canary B.** PLAN:1137-1145 defines Canary B as an anchor-heavy Q00511
**pre-terminal** run that verifies anchors are excluded from `rho_edit`, appear in known-identity
telemetry, and survive "root, every continuation, facade, v0 initial population, and terminal
elite". The Terminal arm has none of those: `rho_target`/`rho_grid` are refused by its config
schema, and its `partial_roots` / `maturity_telemetry` / `continuations` tables are empty by
construction. It is run here only to exercise terminal generation → facade → v0. The real Canary B
is §11.6.

`V1_PROTEINS` must be a SUBSET of the Canary A cohort: the loader refuses any protein the
pre-terminal run never reserved a budget for, and it also refuses a manifest whose `campaign_id` /
`phase` / `split_role` / `f_cap` disagrees with this run's config.

```bash
MODE=v1_entry V1_SUBMODE=run \
OUTPUT_ROOT=${RUN_ROOT}/canary_terminal_smoke \
ENTRY_CONFIG=inverse_folding/reference_flow/configs/rf_fusion_v1_entry_canary_terminal.yaml \
PRETERMINAL_RESERVATION=<canary_A_out_dir>/manifest.json \
V1_PROTEINS="<subset of Canary A's dev ids>" \
... (same TERMINAL_REPAIR_CONFIG / RF_SAMPLER_CONFIG / FUSION_CONFIG / HEAD_* as above) \
sbatch scripts/submit_if_phase_c.slurm
```

Do NOT pass `CONSTRAINT_MANIFEST` here. §3.1 keeps the load-bearing generic cohort anchor-free, so
Q00511 is not in Canary A's reservation: asking for it fails at the loader, and pointing an anchor
manifest at a cohort that does not contain its protein fails at the launch gate. Both refusals are
correct — do not work around either.

### 11.4 What every resolved command must show

`--print-config` output must contain `entry_rf_config: c1_null`, `controller_enabled: false`,
`h_maps_present: false`, no `h_map_parquet` key at all, the `terminal|preterminal` arm label, and
a `resolved_runtime_params` block whose `s_steps` matches the sampler YAML. A disagreement between
any CLI flag and the frozen YAMLs is a hard failure, not a warning.

### 11.5 Canary C — T0 calibration

T0 needs no `PRETERMINAL_RESERVATION`: it has no Terminal counterpart to match, so it computes its
own budget. It writes no `generated.parquet` (it builds no parent); its deliverables are
`t0_control_membership` / `t0_structure_subset` / `t0_structure_results`.

**Implemented and verified by Canary C (2026-07-29):** T0 no longer stops at
`t0_structure_deferred` when the definitive evaluator is available. The integrated evaluator:

1. reads every frozen request from `t0_structure_subset.parquet` without changing membership;
2. preserves `(protein_id, rho_id, policy, request_id, sequence, sequence_md5)` exactly;
3. calls the same v0 target-backbone structure oracle, cache identity and absolute gates for all
   three policies;
4. writes one definitive `t0_structure_results` row plus physical/cache/walltime ledger evidence
   per request;
5. requires exactly `Q_T0` definitive verdicts for every `(protein_id, rho_id, policy)` and fails
   closed on any missing, duplicate or mismatched row;
6. promotes the grid point to `t0_complete` only after those rows are reconciled.

Do **not** run the eight-round Fusion loop and do not create a parent for T0. This handoff reuses
only the definitive structure evaluator required by GO/KILL condition 3.

Because T0 has no v0 stage, its integrated evaluator folds the `3*Q_T0` subset itself, so — unlike
Canary A/B — it REQUIRES the esmfold2_live inputs the v0 stage uses (`REFOLD_CACHE_DIR` and
`ESMFOLD2_SITE_PACKAGES`). The T0 branch of `build_entry_oracles` fails closed without them when the
fusion config backend is `esmfold2_live`. T0 is the anchor-free generic cohort (§3.1), so do NOT
pass `CONSTRAINT_MANIFEST` (the scTM-only T0 gate refuses one).

```bash
MODE=v1_entry V1_SUBMODE=run \
OUTPUT_ROOT=${RUN_ROOT}/canary_c_t0 \
ENTRY_CONFIG=inverse_folding/reference_flow/configs/rf_fusion_v1_entry_canary_t0.yaml \
REFOLD_CACHE_DIR=${RUN_ROOT}/canary_c_t0/refold_cache \
ESMFOLD2_SITE_PACKAGES=/home/zc1519/.conda/envs/esmfold2/lib/python3.12/site-packages \
... (same TERMINAL_REPAIR_CONFIG / RF_SAMPLER_CONFIG / FUSION_CONFIG / HEAD_* as §11.2) \
sbatch scripts/submit_if_phase_c.slurm
```

The `rho_grid` in the canary config was a deliberately wide SMOKE grid. Canary A showed that
`.85` was captured near step 99 on two proteins, so the Canary C grid spans `.50/.70/.85` to
exercise meaningfully different resume points. Canary C subsequently showed distinct unresolved
mass at all three points, and §6.0A freezes the scientific grid and other T0 knobs. The canary
still answers only whether the pipeline runs; it contributes no HT1/HT2 effect evidence.

### 11.5A Scientific T0 development launch

Do not submit this block until the 24-ID cohort/exclusion manifest, four six-ID shard manifests,
and `scripts/analysis/rf_fusion_v1_t0_gate.py` are committed and hashed. Run `--print-config` and
`dry_run` for every shard first. Each shard uses the same config:

```bash
MODE=v1_entry V1_SUBMODE=dry_run \
OUTPUT_ROOT=${RUN_ROOT}/t0_dev_v1/shard_00 \
ENTRY_CONFIG=inverse_folding/reference_flow/configs/rf_fusion_v1_entry_t0_dev.yaml \
V1_PROTEINS="$(xargs < ${WORK_DIR}/fusion_v1/t0_dev_shard_00.ids)" \
REFOLD_CACHE_DIR=${RUN_ROOT}/t0_dev_v1/refold_cache \
ESMFOLD2_SITE_PACKAGES=/home/zc1519/.conda/envs/esmfold2/lib/python3.12/site-packages \
SECONDS_PER_DFE=0.02 SECONDS_PER_REFOLD=2.9 \
... (same frozen c1_null / Fusion / Head / test-set / PDB inputs as Canary C) \
sbatch scripts/submit_if_phase_c.slurm
```

Repeat for `shard_01..03`; only after all four dry-runs pass, submit all four with
`V1_SUBMODE=run`. The first scientific shard must not be used to revise IDs, rho, `K_EST`,
`K_EVAL`, `F_cap`, `Q_T0`, seeds, tests or margins. Aggregate all four shards, run the frozen gate
script once, and archive config/input/code digests with its JSON verdict.

### 11.6 Canary B — anchor-heavy Q00511 (PRE-TERMINAL)

The PLAN's Canary B (PLAN:1137-1145). It is a **pre-terminal** run over a one-protein cohort, with
no `PRETERMINAL_RESERVATION` (the pre-terminal arm PRODUCES the reservation) and the real frozen
constraint manifest:

**Use the 24-anchor file.** §3.2 and `doc/FUSION_V1.md:113` fix the Q00511 safety-max policy at
**24 of 302** residues. `uricase_q00511_active_site_safety_v1.yaml` is now present on this branch
(restored from `c2d8bcd`, byte-identical to `main`) and loads through the real
`load_constraint_manifest` with 24 anchors at 0-based indices 8..288. The other file in the same
directory, `uricase_q00511_active_site_v0.yaml`, declares **8** — it is the v0 direct-catalytic
family-projection template, a legitimate but DIFFERENT spec. Substituting it would silently leave
16 authority-required positions editable and still report a clean anchored run; nothing downstream
knows how many anchors there should have been. `tests/inverse_folding/test_reference_flow_fusion_v1_shipped_configs.py`
pins both counts and pins this section to the safety file.

```bash
MODE=v1_entry V1_SUBMODE=dry_run \
OUTPUT_ROOT=${RUN_ROOT}/canary_b_q00511_preterminal \
ENTRY_CONFIG=inverse_folding/reference_flow/configs/rf_fusion_v1_entry_canary.yaml \
CONSTRAINT_MANIFEST=inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml \
V1_PROTEINS="Q00511" \
... (same TERMINAL_REPAIR_CONFIG / RF_SAMPLER_CONFIG / FUSION_CONFIG / HEAD_* as §11.2) \
sbatch scripts/submit_if_phase_c.slurm
```

Then `V1_SUBMODE=run` and assert on the artifacts:

- `maturity_telemetry` is non-empty and its `n_fixed` equals the manifest's hard-anchor count;
- those anchor indices are absent from the editable set (`n_editable + n_fixed == length`);
- every `partial_roots` payload, every `continuations` row and the `terminal_parent_facade`
  sequence carry the manifest's expected residue at each 0-based anchor index.

**Known gap — do not report this canary as satisfying PLAN:1142 in full.**
`maturity_telemetry.anchor_preservation` is NULL and is not a measurement. Root-level preservation
is enforced structurally (`v1_records` refuses a payload whose `x_t` disagrees with a declared
fixed token), so a root-level value would be 1.0 by construction. The completed-sequence checksum
PLAN:1142 asks for needs the DPLM alphabet to decode `fixed_tokens` and is NOT computed — the third
bullet above is a manual assertion the agent performs, not a gate the pipeline enforces. Report it
as such.

Scheduling: §3.2 gates Q00511 to AFTER the generic P1 verdict, so this canary is a contract/
application smoke only. It is not evidence for the generic trajectory claim (PLAN:1145).

### 11.7 The v0 terminal stage — REQUIRED, and every override must be pinned

The v1 entry driver stops at the facade. It writes `generated.parquet` and returns; it never calls
`run_fusion`. PLAN:874 says to "reuse `scripts/submit_refine.slurm MODE=fusion` for the terminal
complete-state stage", and PLAN's Canary A acceptance requires an "ordered facade accepted by v0",
so **the entry job alone does not satisfy V1F7** — this second job does.

`submit_refine.slurm`'s defaults are wrong for V1 in a way that does NOT fail loudly. Pin all of
them:

```bash
MODE=fusion \
RUN_DIR=${RUN_ROOT}/canary_a_preterminal \
OUT_DIR=${RUN_ROOT}/canary_a_terminal_v0 \
FUSION_CONFIG=inverse_folding/reference_flow/configs/rf_refine_fusion_final_repair_beam.yaml \
BASE_IF_CHECKPOINT=<DPLM_CHECKPOINT> \
RF_SAMPLER_CONFIG=inverse_folding/reference_flow/configs/c1_null.yaml \
TEST_SET_PARQUET=<same as the entry job> PDB_ROOT=<same as the entry job> \
PROTEINS="<same dev ids as the entry job>" \
HEAD_CHECKPOINT=<path> HEAD_VARIANT_ID=<id> \
sbatch scripts/submit_refine.slurm
```

**Why each pin matters:**

| Variable | Default | What the default does |
|---|---|---|
| `FUSION_CONFIG` | `rf_refine_fusion_smoke.yaml` | **SILENTLY WRONG.** Smoke is `n_rounds: 3`, `mode: greedy`, `repair.enabled: false`; the frozen V1 package (§3.1, §8) is `final_repair_beam` = `n_rounds: 8`, `mode: beam`, `repair.enabled: true`. It runs to completion and emits a plausible terminal elite from the WRONG method. |
| `BASE_IF_CHECKPOINT` / `RF_SAMPLER_CONFIG` | empty | guarded — but ONLY when repair is enabled. Under the smoke default repair is off, so the guard never fires and the two silent legs compose. |
| `TEST_SET_PARQUET` / `PDB_ROOT` | uricase char24 | wrong cohort and backbones for a DRB1 run. |
| `RUN_DIR` | empty | loud: `no generated.parquet under .` |

Run it for BOTH arms (each pointed at its own entry `OUTPUT_ROOT`), or the two arms are not
compared through the same terminal package. Then assert the crosswalk closes end to end:
`fusion_elite.entry_source_id` → `fusion_initial_admission_verdicts` → `terminal_parent_facade` →
`continuations` / `partial_roots`, **with no join on a sequence anywhere**.

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
