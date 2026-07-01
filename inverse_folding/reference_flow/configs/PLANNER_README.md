# Planner v1 allocation experiment grid (PLAN_PLANNER_SC_GR.md Task 7)

3 selection-field arms × 3 amplify `g_max` = 9 configs, all on the
`d2_d3_full_stageC_scgr_betaonly_aopen.yaml` Budget layer + the new Allocation
layer (frozen per-residue `Φ_i` reweighting the typed selection field).

| config | `selection_field_mode` | role | `g_max` | calibration JSON |
|---|---|---|---|---|
| `planner_flat_gmax15_aopen.yaml`    | `flat`             | C2 content-blind control | 1.5 | `amplify_calib_gmax15.json` |
| `planner_vtarget_gmax15_aopen.yaml` | `v_target`         | C1 uniform-allocation control (Φ≡1) | 1.5 | `amplify_calib_gmax15.json` |
| `planner_alloc_gmax15_aopen.yaml`   | `v_target_x_alloc` | treatment (immune-tilted) | 1.5 | `amplify_calib_gmax15.json` |
| `planner_flat_gmax20_aopen.yaml`    | `flat`             | C2 control | 2.0 | `amplify_calib_gmax20.json` |
| `planner_vtarget_gmax20_aopen.yaml` | `v_target`         | C1 control | 2.0 | `amplify_calib_gmax20.json` |
| `planner_alloc_gmax20_aopen.yaml`   | `v_target_x_alloc` | treatment | 2.0 | `amplify_calib_gmax20.json` |
| `planner_flat_gmax25_aopen.yaml`    | `flat`             | C2 control | 2.5 | `amplify_calib_gmax25.json` |
| `planner_vtarget_gmax25_aopen.yaml` | `v_target`         | C1 control | 2.5 | `amplify_calib_gmax25.json` |
| `planner_alloc_gmax25_aopen.yaml`   | `v_target_x_alloc` | treatment | 2.5 | `amplify_calib_gmax25.json` |

`g_min=0.5` is pinned for all (SC2 amplify); the band (`B_low`/`B_high`) is **not**
hardcoded — it is stamped at run setup from the per-`g_max` amplify calibration
JSON (`--global-pressure-calibration-json`), which the loader cross-checks against
each YAML's `amplify` / `g_min` / `g_max` (`run_if_phase_c1.py` calibration
contract, LOG L0119).

## 1. Emit the 3 amplify calibration JSONs (one per `g_max`, cluster step)

Built on the SC1 pilot run's early-frozen `refresh_step=0 / fresh / topm_lse` `B_sc`
distribution (matches the configs' `actuation_arm=fresh`,
`actuation_aggregator=topm_lse`, `freeze_after_reliable_refreshes=1` ⇒
`refresh_step=0`). The SC2 emitter solves the band so
`smoothstep_pressure(B_median; band, g_min, g_max) == 1.0` at the per-design
median for the asymmetric gain (LOG L0119). All paths via CLI — no hardcoding.

```bash
# repeat for GMAX in {1.5, 2.0, 2.5} with GTAG in {15, 20, 25}
python scripts/analysis/sc_gr_monitor_probe.py \
  --run-dir       <SC1_PILOT_RUN_DIR> \
  --oracle-parquet <ORACLE_PARQUET> \
  --output        <MONITOR_REPORT_gmax${GTAG}.json> \
  --emit-calibration inverse_folding/reference_flow/configs/amplify_calib_gmax${GTAG}.json \
  --calibration-aggregator topm_lse \
  --calibration-arm        fresh \
  --calibration-refresh-step 0 \
  --low-quantile 0.25 --high-quantile 0.75 \
  --amplify --g-min 0.5 --g-max ${GMAX}
```

The 3 arm configs at a given `g_max` SHARE that `g_max`'s calibration JSON.

## 2. Run a config (cluster step)

```bash
python scripts/run_if_phase_c1.py ... \
  --controller-config inverse_folding/reference_flow/configs/planner_alloc_gmax20_aopen.yaml \
  --global-pressure-calibration-json inverse_folding/reference_flow/configs/amplify_calib_gmax20.json
```

Cohort: 50 pilot50r2 DRB1\*07:01 proteins, n=8 designs/protein, seed=42 (RAR 0020).
Evaluate with the existing immune (NetMHCIIpan) + scTM path, then gate with
`scripts/analysis/planner_h1_pareto.py` (PLAN §5; the H1 CLI derives
`immune_nmp := n_strong_binders / n_windows_scored`, renames `scTM→sctm`, and
injects `g_max` / `arm_mode` from each run's stamped controller config).

Calibration JSONs are large generated artifacts — keep them out of git; emit them
on the cluster next to the configs (or pass an absolute path).

---

## §A — Revised v1.1 ablations (LIVE; the 3×3 grid above is SUPERSEDED)

The direct-allocation H1 (`v_target_x_alloc`) was **NOT supported** (RAR 0019 M4):
the multiplicative `v_target·(ε+c·Φ)` reweight dragged fire onto low-`v_target`
sites. Two independent, default-off ablations replace it (doc §8.4 Path A;
`PLAN_PLANNER_SC_GR.md` §A). Both run at `N_STEPS=100`. Pre-registered success is
**neutral-or-better at matched scTM** (authority ceiling, RAR 0019 M5), not a large drop.

### §A1 — `v_target_triage` (WHERE layer)

`r_i` (via the frozen `Φ_i`) becomes an eligibility-gated additive rank tie-break
**within** the `v_target`-eligible set (no `τ_v` widening). **Strict eligibility:**
eligible = top-(1−`eligible_quantile`) **among strictly-positive** actionability
(`v_target > floor`, `floor = active_window_min_excess`); the quantile is over the
positive values only, so a sparse `v_target` whose median is 0 can never admit
zero-actionability sites (the §8.2.1 failure). 3 arms at `g_max=2.0`
only (authority ceiling ⇒ no g_max sweep), cohort **pilot47**:
`planner_flat_gmax20_aopen.yaml` + `planner_vtarget_gmax20_aopen.yaml` (exist) +
`planner_triage_gmax20_aopen.yaml` (`selection_field_mode: v_target_triage`,
`allocation.eligible_quantile: 0.5`, `triage_lambda: 0.3`). Reuse
`amplify_calib_gmax20.json` (re-emit at the 100-step cadence). Gate:

```bash
python scripts/analysis/planner_h1_pareto.py \
  --runs-json <flat+vtarget+triage runs json> --output <h1_triage.json> \
  --arm-treatment v_target_triage \
  --expected-n-proteins 47 --expected-n-designs 8 --expected-g-max 2.0
```

### §A2 — terminal candidate-probe `P_cheap` (VALUE layer) on B1

D2 ranks shortlisted candidates by a full-sequence terminal completion instead of
the local block-span ensemble. 2 arms on the **B1** base
(`d2_d3_full_stageB_aopen.yaml`, `global_pressure` off, β=3.0), **high-risk cohort**
(RAR 0013 NoD-worst-100; cohort path via CLI): `b1_local` (= the unchanged B1
config) + `b1_terminal_aopen.yaml` (`d2.candidate_score_source: terminal`,
`candidate_terminal_K_P: 4`; `terminal` requires `completion_ensemble_enabled=true`,
which the B1 base already sets). `b1_local` and `b1_terminal` share
`targeting.selection_field_mode`/`g_max`, so the `--pairwise` runs-json entries
**must carry explicit `arm_mode`** (`b1_local` / `b1_terminal`) — config
auto-derive cannot distinguish them. Direct two-arm gate (no Pareto):

```bash
python scripts/analysis/planner_h1_pareto.py \
  --runs-json <b1_local+b1_terminal runs json> --output <a2_terminal.json> \
  --pairwise --arm-treatment b1_terminal --arm-baseline b1_local \
  --expected-n-proteins <cohort_size> --expected-n-designs 8
```

### §A3 — `v_target_terminal_union` (WHERE layer)

A1 triage is a WITHIN-eligible tie-break: a low-`v_target` register never
activates. **A3 is the principled inverse of A1** — Fork A ("terminal-OR-local
gated union", doc §8.4). Union eligibility PROMOTES high-`r_i` (high-`Φ`)
low-`v_target` registers INTO the active set: `eligible = local | terminal`, where
`local` = strictly-positive actionability (`v_target > floor`,
`floor = active_window_min_excess`) and `terminal` = the register-smoothed `Φ_i`
in the top-(1−`terminal_eligible_quantile`) **taken over the positive `Φ` subset
only** (`quantile(Φ[Φ>0], q)`, mirroring A1 triage). Over the union the score is
`norm_rank(v_target) + terminal_lambda · norm_rank(Φ_i)` (`v_target` keeps
coefficient 1). C0a M4 found those registers are skipped for low *local* head
evidence (`b_env`), not structure, so the terminal channel supplies the
actionability evidence the local term misses. This is a **separate terminal gate,
not a lowered `v_target` bar** (never widens `τ_v`). The terminal quantile and the
window-max selection unit are both **register-grain** (`Φ` is the frozen,
register-smoothed allocation mass). **Two `Φ`-degeneracy guards keep the terminal
set from widening onto dead sites** (RAR 0019 M4 / §8.2.1): (i) the positive-subset
quantile — the realistic `allocation_mass` output is **sparse** (mostly-zero), so a
full-array quantile would collapse to 0 and admit every dead site (the review P1);
(ii) the `max>min` flat guard — an all-zero `r_i` yields a **uniform** `Φ` (ones)
with no discriminative signal. In both, the terminal set is empty ⇒ admit NOTHING
beyond the local set.
A3 **deliberately crosses the §3 firewall**, which is guarded by the early-freeze
of `Φ_i`. 3 arms at `g_max=2.0` only (authority ceiling ⇒ no g_max sweep):
`planner_flat_gmax20_aopen.yaml` + `planner_vtarget_gmax20_aopen.yaml` (exist) +
`planner_terminal_union_gmax20_aopen.yaml` (`selection_field_mode:
v_target_terminal_union`, `allocation.terminal_eligible_quantile: 0.9`,
`terminal_lambda: 0.3`). Reuses the scgr_betaonly Budget layer +
`amplify_calib_gmax20.json` (re-emit at the 100-step cadence). Gate:

```bash
python scripts/analysis/planner_h1_pareto.py \
  --runs-json <flat+vtarget+terminal_union runs json> --output <a3_terminal_union.json> \
  --arm-treatment v_target_terminal_union \
  --expected-n-proteins 47 --expected-n-designs 8 --expected-g-max 2.0
```

A3 is a **3-arm WHERE-layer** experiment (like §A1, not the §A2 value layer): use the
3-arm `compute_h1` gate (**no `--pairwise`**) so `flat` and the macro leg
(`v_target < flat` AND `terminal_union < flat`) are consumed — `terminal_union` is
paired against `v_target` (primary) and the macro control against `flat`.

On A1/A2/A3 outcomes: register a RAR (objective measurements only). A1≈neutral and
A2≈neutral ⇒ confirms the authority ceiling binds → green-lights Path C (the main
line; out of scope here).

---

## §C0b — Path C headroom existence test (offline; gates the C rewrite)

**Why (doc `Self-Cond_GR.md` §8.4 Path C, step C0b):** before any D2/sampler rewrite
for coordinated edits, prove the headroom exists. C0a (RAR 0021) ruled out the
structural-lock NO-GO; C0b is the decisive constructive test. **Run now** —
offline, independent of the running A1/A2, **no controller/firewall change** — but
it needs a **new offline script** (coder), not a config tweak.

**Question:** on a high-`r_i` register, does a **coordinated** resample reduce
terminal immune *more* than the best **single-position** safe edit, at acceptable
structure? GO ⇒ Path C worth building; NO-GO ⇒ drop the C rewrite.

**Method (new script, ~existence probe, not a full sweep):**
1. Sample ~30–50 high-`r_i` registers (top-tercile `r_i` ≈9-residue windows) across a
   few proteins, from existing designs (planner_pilot47 or B1 high-risk) + their
   per-residue `r_i`.
2. Per register, **three arms** holding the rest of the sequence fixed — each
   **folded (ESMFold)** so all comparisons are at acceptable structure:
   - **single-position best (DOF=1):** enumerate each position's structurally-safe
     support, score terminal immune (frozen head), keep the best
     structure-acceptable single-token edit.
   - **fake joint (DOF=9, factorized):** remask the register, draw `K` resamples by
     a **single forward** = independent per-position marginal product (no
     reconditioning). Control for "more editable positions" — achievable in the
     *existing* factorized D2 by lifting the per-block edit cap.
   - **true joint (DOF=9, coordinated):** remask the register, draw `K`
     **multi-step reconditioned** resamples (recondition across partial commits →
     anchor+compensation correlation). This is the thing only a Path-C coordinated
     sampler gives.
   Per arm: keep the best **structure-acceptable** candidate (scTM ≥ baseline − tol).
3. **Structure proxy:** cheap screen = DPLM structural log-prob of the resampled
   register; confirm the shortlist with ESMFold scTM (cluster). "Acceptable" =
   scTM not worse than the single-position arm by a fixed tol.
4. Compare per register: `joint_best_immune` vs `single_best_immune` at matched/
   acceptable structure.

**Pre-registered decision (fix before results) — isolates *coordination*, not
degrees-of-freedom.** All margins are at acceptable structure (folded). Define per
register: `coordination_margin = true_best − fake_best`; `dof_margin = fake_best −
single_best` (immune reduction, nat).
- **Path-C GO (build the coordinated sampler):** `coordination_margin ≥ ~0.2 nat`
  (one single-edit ceiling-unit) for **≥ 1/3** of registers.
- **Secondary readout (the cheaper alternative):** if `dof_margin` is large but
  `coordination_margin ≈ 0`, the conclusion is **do NOT build Path C — instead lift
  D2's per-block editable cap** (more positions in the existing factorized D2).
- **NO-GO for Path C** if folded `true ≤ fake` (coordination buys nothing at
  acceptable structure). Note: the smoke's `fake ≳ true` was **pre-structure**; the
  decisive test is **post-fold** — fake's immune wins are expected to be
  structurally infeasible and flip. If they do not flip, Path C is not justified.

Report per register `(single_best_immune, fake_best_immune, true_best_immune,
single_scTM, fake_scTM, true_scTM, coordination_margin, dof_margin)` + the
GO-fraction. Register a RAR. The old `joint_wins` (joint-vs-single) is **dropped**
as a gate — it measures DOF (9 edits > 1), not coordination.

**Reuse:** DPLM sampler (the generator) + frozen head scorer + ESMFold (all exist);
new code is only the register-remask + multi-step-resample driver + the scorer
loop. Cluster compute (ESMFold on the shortlist). Paths via CLI, no hardcoding.

**Outcome (RAR 0022 + 0023): coordination NO-GO; the lever is BREADTH, not
coordination.** Reconditioned (true) vs independent (fake) joint adds ≈0 (win
fraction 0.300, depth sweep flat). The multi-position headroom is real but is
carried by *independent* wide resampling + head selection, and the winning edit
changed only 4 of 9 register residues (fits D2's existing cap). It is **soft**:
partly regression-to-mean (design on an immune peak) and best-of-K-unsaturated, so
the offline −10 nat is an upper bound. → do NOT build the coordinated sampler;
test breadth in the closed loop (§Breadth).

---

## §Breadth-S1 — WITHDRAWN (config-only cap-widening is blocked AND moot)

**Do not run the `max_positions_per_block` sweep.** Two reasons (2026-07-01):

1. **Blocked.** With `candidate_mode=structure_topk`, `top_k_tokens=8`, a block's
   candidate count is `8^P` (P = `max_positions_per_block`). `P=4` → `8^4=4096` =
   `max_candidates_per_block` (cartesian, OK); `P=6/8` → `262k / 16.7M` ≫ 4096 →
   the code falls to the `sampled` path, which **`raise NotImplementedError`** when
   `completion_ensemble` is on (uniform-over-shortlist `Q` marginal is biased; it
   fails-fast rather than emit a biased correction — `counterfactual.py:1114`). The
   earlier "`max_candidates 4096 → sampled path is used`" note was **wrong**; pos4
   ran only because `8^4` is exactly on the cartesian ceiling. No config (holding
   `top_k=8`) runs `P>4`.
2. **Moot.** **RAR 0023 M6**: the 4-cap is essentially free — 54% of winning edits
   change ≤4 residues, and where a ≤4 candidate exists (24/35 registers) restricting
   to ≤4 costs **median +0.000 nat** (~8% of registers pay >1 nat). The binding
   constraint is **which register/positions D2 selects**, not positions-per-block.

→ **The live Stage 1 is §A3** (`v_target_terminal_union`): make D2 *select* the
high-`r_i` low-`v_target` registers it skips, and `Φ`-rank the right ≤4 positions
within them. Evaluate immune (NetMHCIIpan) + scTM; recovery is a movement diagnostic
only. **Widening the cap** (implement full-candidate-set projection for the `sampled`
path — a localised factorized-D2 fix) is a **deferred fallback**, only if A3 shows
the reached registers genuinely need >4 changes (the ~8% >1-nat tail).
