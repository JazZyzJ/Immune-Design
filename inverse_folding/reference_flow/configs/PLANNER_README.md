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

On A1/A2 outcomes: register a RAR (objective measurements only). A1≈neutral and
A2≈neutral ⇒ confirms the authority ceiling binds → green-lights Path C (the main
line; out of scope here).
