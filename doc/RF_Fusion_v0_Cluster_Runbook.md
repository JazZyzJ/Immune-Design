# RF-Refine Fusion v0 — Cluster Agent Runbook

AGENT TASK SPEC. Terse, imperative, machine-actionable. Execute top-to-bottom. Stop and report on any FAIL.

## 0. What this is
- v0 = **terminal-population handoff** (t⋆ = T): Phase 1 runs base DPLM native inverse folding to completion → complete designs; Phase 2 (this task) reads them as the N initial parents and runs the Head-guided edit-and-repair loop.
- This run is an **engineering smoke + calibration probe**, NOT a target-hitting run. Goal: pipeline executes end-to-end, produces artifacts, yields gate/telemetry to calibrate `scTM_min / eps_H / active_site_RMSD_max / N / n_rounds / beta`.
- Head is the ONLY runtime immune signal. NetMHCIIpan is NOT used (not imported). Do not pass any `--nmp*` / `--netmhciipan*` / `--seed-table`.

## 1. Environment
```bash
module purge && module load anaconda3/2025.12 && conda activate immune-design
cd /home/zc1519/src/Immune-Design
git fetch origin && git checkout fusion_rf_refine && git pull
git log --oneline -1        # EXPECT head commit 8566233 (RF-Refine Fusion landing) or later
```
FAIL if the branch/commit is absent.

## 2. Login-node dry checks (no GPU)
```bash
python -m py_compile scripts/run_rf_refine_fusion.py inverse_folding/reference_flow/fusion/*.py
python -c "from inverse_folding.reference_flow.fusion.config import load_fusion_config as L; L('inverse_folding/reference_flow/configs/rf_refine_fusion_smoke.yaml'); print('cfg OK')"
python scripts/run_rf_refine_fusion.py --help >/dev/null && echo "argparse OK"
bash -n scripts/submit_refine.slurm && echo "slurm OK"
```
All must print OK / exit 0.

## 3. Resolve inputs (VERIFY every path exists; do NOT fabricate)
Base = `/scratch/gpfs/KAIYIJIANG/zijie`. These are the `submit_refine.slurm` MODE=fusion defaults — confirm each file is present, else report which is missing.

| Var | Default (verify exists) | Meaning |
|---|---|---|
| `ALLELE` | `HLA-DRB1*07:01` | Head allele |
| `HEAD_CHECKPOINT` | `${BASE}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt` | frozen Head |
| `HEAD_CONFIG_DIR` | `${PROJECT_ROOT}/epitope_head/configs` | Head config |
| `HEAD_VARIANT_ID` / `HEAD_ALLELE_IDX` | `LC1` / `0` | Head variant / allele index |
| `TEST_SET_PARQUET` | `${C24_INPUTS}/uricase_characterized24_if_ready.parquet` | target backbones (reference seq + length) |
| `PDB_ROOT` | `${BASE}/work/immune-design/if_test_set/uricases/pdbs_if_ready` | reference PDBs for scTM/shell RMSD |
| `CONSTRAINT_MANIFEST` | `${C24_INPUTS}/uricase_characterized24_active_site_manifest.yaml` | active-site hard anchors |

`C24_INPUTS = ${BASE}/work/RF/Uricases_RF/characterized24_inputs/20260703T201539Z`.
NOTE: smoke config sets `active_site_RMSD_max=2.0`, so anchored proteins REQUIRE a valid manifest; the driver fail-fasts if a manifest anchor disagrees with the reference sequence. If you intend a non-enzyme run, set `CONSTRAINT_MANIFEST=none`.

## 4. Phase 1 — generation surface (RUN_DIR)
Fusion reads complete native-IF designs from `RUN_DIR` (`generated.parquet` clean OR `generation/*shard*/generated.parquet`), columns `protein_id, design_idx, sequence`.

1. Locate an EXISTING DPLM-native (C0) generation for the SAME proteins + backbone as §3:
   ```bash
   find ${BASE}/run/inverse_folding -path '*native*/*generated.parquet' 2>/dev/null | head
   ```
   Reuse only if it covers the char24 uricase `protein_id`s and used the same `TEST_SET_PARQUET`/`PDB_ROOT`.
2. If none exists, PRODUCE one (native, no Head, no controller). `CHECKPOINT` = base DPLM IF checkpoint — verify the `submit_if_phase_c.slurm` default resolves, else set it explicitly. Use `gumbel_argmax` (native `argmax` is deterministic → identical designs, which would collapse the N-parent population):
   ```bash
   MODE=native ALLELE="HLA-DRB1*07:01" \
   CHECKPOINT=<base IF ckpt; verify submit_if_phase_c.slurm default> \
   TEST_SET_PARQUET=<§3> PDB_ROOT=<§3> \
   N_DESIGNS_PER_PROTEIN=8 NATIVE_SAMPLING_STRATEGY=gumbel_argmax \
   OUTPUT_ROOT=${BASE}/run/inverse_folding/phase_c \
   sbatch scripts/submit_if_phase_c.slurm
   ```
   Result dir = `${OUTPUT_ROOT}/<allele_tag>/<run_id>/` containing `generated.parquet`. That dir is `RUN_DIR`.
   Requirement: designs-per-protein ≥ fusion `N` (=4), else Phase 2 fails `insufficient_feasible_initial_population`.

Set `RUN_DIR=<the run dir from step 1 or 2>`.

## 5. Phase 2 — run Fusion (S0: explicit-only, repair OFF)
Smoke config `rf_refine_fusion_smoke.yaml`: N=4, n_rounds=3, greedy, explicit-only, scTM_min=0.85, active_site_RMSD_max=2.0, eps_H=0.05, beta 1→4. `moves.repair.enabled=false` ⇒ leave `BASE_IF_CHECKPOINT`/`RF_SAMPLER_CONFIG` UNSET (no DPLM sampler this run).

```bash
MODE=fusion \
RUN_DIR=<§4> \
OUT_DIR=${BASE}/run/inverse_folding/fusion/uricase_c24_s0_$(date +%Y%m%dT%H%M%S) \
PROTEINS=all \
ALLELE="HLA-DRB1*07:01" \
sbatch --cpus-per-task=2 scripts/submit_refine.slurm
```
GPU required (Head + ESMFold). All other inputs default to §3; override only if §3 verification pointed elsewhere.

## 6. Success criteria
Under `OUT_DIR`, EXPECT: `fusion_candidates.parquet`, `fusion_particles.parquet`, `fusion_lineage.parquet`, `fusion_elite.parquet`, `fusion_rounds.jsonl`, `fusion_failures.parquet`, `generated.parquet`, `manifest.json`.
- Job exits 0. Driver exits non-zero iff 0 proteins succeeded — treat that as FAIL and report `fusion_failures.parquet` reasons.
- `manifest.json`: `nmp_absent == true`, `n_proteins_ok > 0`, `config_hash` + `provenance` present.

## 7. Analysis (extract, do NOT tune gates yourself — report numbers)
From the artifacts compute and report:
1. **Coverage**: `n_proteins_ok / n_proteins_failed`; per failed protein the `reason`/`message` (esp. `insufficient_feasible_initial_population` → scTM_min too strict OR designs-per-protein < N).
2. **Elite improvement**: from `fusion_elite.parquet`, `initial_head_global_risk → final_head_global_risk` delta and `improvement` distribution (median, IQR); `n_rounds` actually used.
3. **Gate behavior**: from `fusion_candidates.parquet`, counts by `reason` (`ok / offtarget_hotspot / not_shortlisted / structure fail`), `feasible` fraction, `scTM` and `active_site_RMSD` distributions on evaluated candidates.
4. **Round dynamics**: from `fusion_rounds.jsonl`, per-round `n_feasible_children`, `n_selected_children`, `elite_risk`, `n_refolds`, `cache_hits`, `wall_time_s`.
5. **Population health**: initial unique-sequence fraction (population collapse check).
Optional cross-check: re-score the elites (`fusion_elite.parquet` → 3-col facade) with `scripts/evaluate_phase_c.py --mode all` for an independent immune/structure read.

Write findings to `${OUT_DIR}/analysis/fusion_v0_s0_summary.md` (tables of the 5 blocks above + one paragraph: does the pipeline behave, are the placeholder gates roughly right, which gate to move first).

## 8. Return
Via `mhc-if-local` to `/Users/jerry/Project/MHC-IF/Results`, bucket `RF/HLA-DRB1_0701/fusion_v0_s0__<timestamp>/`:
- `generation/` ← `RUN_DIR/generated.parquet` (the Phase-1 seeds actually used)
- `analysis/` ← the fusion artifact set + `analysis/fusion_v0_s0_summary.md`
- `meta/` ← `manifest.json`, resolved config, the exact sbatch command
- `logs/` ← SLURM `.out`/`.err`
Return does not require LOG.md edits.

## 9. Failure handling
- Stop after 2–3 failed attempts of the same step; report what failed (command + stderr tail) and the blocking input. Do not loop.
- `insufficient_feasible_initial_population` on many proteins → report; likely fix is more Phase-1 designs-per-protein or a looser `scTM_min` (a calibration decision, not yours to make silently).
- Missing `--base-if-checkpoint`/`--rf-sampler-config` errors ⇒ you accidentally enabled repair; S0 must keep `moves.repair.enabled=false` and leave those unset.

## 10. NOT in scope this run (deferred, do not attempt)
- S1 repair/H3 arm (needs `BASE_IF_CHECKPOINT` + `RF_SAMPLER_CONFIG` + a repair-enabled config) — separate task after S0 looks sane.
- Gate calibration edits, round-boundary resume, partial-step t⋆<T handoff.
