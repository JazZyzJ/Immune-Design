# PLAN — SC-GR per-residue `r_i` accuracy gate

**Thread:** SC-GR. Follow-up to RAR 0010 (protein-scalar `B_sc`, ρ0.74) and RAR
0018 (candidate-tuple cheap-probe failed, ρ0.17). Linked from
`doc/Self-Cond_GR.md` §3 step 4 / §5 (the "later targeting study" the `r_i` map
was reserved for). Home of the position-dependent ("local SC1") gate.

## 0. Question & hypothesis

RAR 0010 validated the SC-GR ensemble at the **protein scalar** (a LEVEL,
between-protein sd 3.43, ρ0.74). RAR 0018 showed the **candidate-tuple terminal
effect** (a within-block MARGINAL, joint over ≤4 positions) is NOT cheaply
predictable. The untested middle granularity is the ensemble's **per-residue map
`r_i`** — `residue_excess` = `max(0, Proj_i(head windows) − tau_ref_B)` over the
`K` completions. `r_i` is a **LEVEL** (per-residue terminal risk), and MHC-II
burden concentrates at a few anchor/register sites, so the within-protein
per-residue level has large dynamic range. Question:

> Does the SC-GR ensemble's per-residue `r_i` rank residues within a protein the
> way the unconditional (NoD) per-residue burden does — at the residue scale and
> at the coarser region scale that targeting actually uses?

This is the gate for the position-dependent path: if `r_i` localizes, an
`r_i`-driven position-dependent pressure (local SC1, the project thesis) is on
the table; if it does not, only the global protein-scalar SC1 survives.

**Caveat (Method fact, not a verdict):** `r_i` is from the **guided** trajectory
at a refresh; the oracle is the **NoD unconditional** per-residue burden — the
exact protein-scalar pairing RAR 0010 validated, one granularity down. Spearman
is rank-based, so probe `residue_excess` and oracle `hotspot` need only be
monotone in residue immunogenicity, not equal.

## 1. Pre-registered decision rules (fixed before results)

Per (arm × refresh_step), over the 50 proteins; the actuation-relevant cell is
`arm=fresh`, `refresh_step=0` (RAR 0010's frozen horizon).

- **PASS (position-dependent path viable):** median **region** Spearman ≥ **0.40**
  AND `Recall@HighResidue` ≥ **0.50** (chance = 0.33). The ensemble localizes
  burden well enough at the targeting granularity → build the `r_i` → position
  -dependent pressure prototype (a later plan; NOT in scope here).
- **FAIL (signal dead at sub-protein scale):** median region Spearman < **0.20**
  AND median residue Spearman < **0.20** → `r_i` does not localize → the
  position-dependent lever is not on this signal; only global protein-scalar SC1
  (RAR 0010 / SC1) survives. Close the local-SC1 direction.
- **PARTIAL:** residue Spearman weak but region Spearman ≥ 0.40 → only **coarse
  regional** targeting is viable (not residue-resolution); record and decide.

Report `P(probe low | oracle high)` (the per-residue true-high-as-low tail, the
RAR-0010 metric) alongside, but it is diagnostic, not a gate threshold.

## 2. What is already done (local, this commit)

- **Code:** `self_conditioned_gr.SCGRRiskAggregates` now surfaces `residue_excess`;
  the probe appends it per completion to `sc_gr_probe_samples.parquet` **only**
  when `self_conditioned_gr.write_residue_telemetry=true` (default False keeps
  existing monitor telemetry byte-identical). Tested:
  `tests/inverse_folding/test_reference_flow_self_conditioned_gr.py`,
  `tests/inverse_folding/test_reference_flow_d1_controller.py`.
- **Config (one per experiment):**
  `inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor_residue.yaml`
  — identical to the monitor preset except `write_residue_telemetry: true`.
- **Analysis:** `scripts/analysis/scgr_residue_accuracy.py` (registered in
  `doc/SCRIPTS.md` #14), tested on synthetic data
  (`tests/scripts/test_scgr_residue_accuracy.py`).

## 3. Cluster run-spec (hand to the cluster coder)

Two GPU jobs, then one local CPU analysis. Cohort = the **same pilot50r2 50
proteins** as RAR 0010 (`scgr_monitor_pilot50r2_aopen_n4_steps20_b4_seed42`);
the monitor test-set parquet must match that run.

**(a) Probe `r_i` — re-run the monitor with residue telemetry on** (mirror RAR
0010: `N_STEPS=20`, `n4`, `seed=42`):

```bash
MODE=reference_flow \
CONTROLLER_CONFIG=/home/zc1519/src/Immune-Design/inverse_folding/reference_flow/configs/d2_d3_full_stageB_aopen_scgr_monitor_residue.yaml \
N_STEPS=20 \
N_DESIGNS_PER_PROTEIN=4 \
SEED=42 \
GLOBAL_PRESSURE_CALIBRATION_JSON="" \
sbatch scripts/submit_if_phase_c.slurm
```

Expected: `generation/sc_gr_probe_samples.parquet` now carries a `residue_excess`
list-column (length L per row); everything else byte-identical to the monitor
preset up to deterministic head-call side effects.

**(b) NoD per-residue oracle — re-eval the EXISTING NoD designs with `--imm-full`**
(no new code; reuses `evaluate_phase_c.py --imm-full`, which emits
`imm_head_residues.parquet` with per-residue `hotspot`). Score the same NoD
generation RAR 0010 used (`phaseD_pilot50r2_nod_n8_seed42`):

```bash
MODE=phase_c \
GENERATED_PARQUET=<.../phaseD_pilot50r2_nod_n8_seed42__*/generation/generated.parquet> \
TEST_SET_PARQUET=<pilot50r2 IF-ready parquet> \
ALLELE="HLA-DRB1*07:01" \
EVAL_MODE=imm \
IMM_FULL=1 \
RUN_ID=phaseD_pilot50r2_nod_n8_seed42_immfull \
sbatch scripts/submit_benchmark.slurm
```

Expected: `eval_immune/.../imm_head_residues.parquet`
(`protein_id, design_idx, residue_idx, residue_aa, hotspot`).

**(c) Local analysis** (CPU; both bundles returned via `mhc-if-local`):

```bash
python scripts/analysis/scgr_residue_accuracy.py \
  --probe-samples <monitor_residue_run>/generation/sc_gr_probe_samples.parquet \
  --oracle-residues <nod_immfull_run>/eval_immune/.../imm_head_residues.parquet \
  --oracle-aggregate median \
  --region-size 9 \
  --output <out>/scgr_residue_accuracy.json
```

Then apply §1 rules to the `arm=fresh, refresh_step=0` cell and register a RAR
(supersede/extend RAR 0010 is NOT required — this is a new run with new
telemetry; commit a fresh RAR citing 0010/0018).

## 4. Out of scope

- Building the `r_i` → position-dependent pressure actuator (gated on §1 PASS;
  a later plan). This gate only **measures** whether `r_i` localizes.
- Touching D2/D3, remask, `global_pressure`, or the protein-scalar SC1/SC2 path.
- The learned per-tuple amortization (the RAR-0018 follow-up) — separate thread.
