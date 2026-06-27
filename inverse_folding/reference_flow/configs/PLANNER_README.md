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
