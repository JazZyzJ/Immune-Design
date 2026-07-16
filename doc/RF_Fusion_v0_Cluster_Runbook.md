# RF-Refine Fusion v0 — Cluster Agent Runbook

AGENT TASK SPEC. Terse, imperative, machine-actionable. Execute top-to-bottom. Stop and report on any FAIL.

## 0. What this is
- v0 = **terminal-population handoff** (t⋆ = T): Phase 1 runs base DPLM native inverse folding to completion → complete designs; Phase 2 (this task) reads them as the N initial parents and runs the Head-guided edit-and-repair loop.
- This run is an **engineering smoke + calibration probe**, NOT a target-hitting run. Goal: pipeline executes end-to-end, produces artifacts, yields gate/telemetry to calibrate `scTM_min / eps_H / active_site_RMSD_max / N / n_rounds / beta`.
- Head is the ONLY runtime immune signal. NetMHCIIpan is NOT used (not imported). Do not pass any `--nmp*` / `--netmhciipan*` / `--seed-table`.
- **Cohort scope — read before interpreting anything.** The smoke cohort is `uricase_characterized24`: a **single enzyme family** with prior inpainting references and AFDB-predicted backbones. It is chosen because it is the **only** cohort that exercises the constrained path (active-site anchors present). It is a **mechanism + plumbing** cohort, NOT an immune-effect cohort. Do **not** read a final immune result from this run: (a) a single Pfam family has near-zero cross-protein immune-burden variance, so no generalizable effect (or CI) is estimable; (b) under AFDB references, `scTM` is ESMFold↔AlphaFold **prediction self-consistency**, not experimental structural fidelity — high baseline, low discriminative power. Neither structure threshold (`scTM_min`, `active_site_RMSD_max`) is scientifically *calibratable* on this degenerate distribution; the smoke only exercises the gate mechanics. **The formal falsifier gate (P1–P3: immune effect + scTM generalization) runs on the diverse test-set high-risk cohort** — where proteins are unconstrained (mostly non-enzyme → active-site gate is N/A there), so the constrained active-site/anchor mechanism must be validated *here*, on uricase. This deliberately refines PLAN §6 (which pre-registered the P1–P3 cohort as this uricase run extended to its low-risk proteins); reconcile PLAN §6 before running the formal gate.

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

### 6.1 Mechanism acceptance — what this smoke proves (and what it does NOT)

The smoke is judged on **mechanism**, not immune effect. The smoke series (S0 now; S1/S2 deferred, §10) answers exactly these — each evidenced by an artifact/assertion, nothing about final immune burden:

| # | Mechanism question | Evidence (artifact / assertion) | Exercised by |
|---|---|---|---|
| 1 | Is the exact accepted child truly carried into the next round? | `fusion_particles.parquet` / `fusion_lineage.parquet`: selected child sequence (md5) at round `r` == parent sequence at round `r+1`, byte-identical | **S0 (now)** |
| 2 | Are edit-core, active-site anchors, and outside-halo positions preserved? | `fusion_candidates.parquet` edited/halo columns; manifest anchors never mutate; only the target register (S0) / halo (S1) differs from parent | **S0** (anchors + target-only) / S1 (edit-core + halo) |
| 3 | Does RF repair change the halo neighborhood WITHOUT reverting the immune edit? | repaired child differs from parent inside the halo but keeps the protected edited position | S1 (deferred) |
| 4 | Are structure-failed children ABSOLUTELY barred from ancestry? | `fusion_candidates.parquet`: every ancestry-eligible/selected child has `structure_evaluated=true` AND passes the absolute gate; no `feasible=false` child is a parent next round | **S0 (now)** |
| 5 | Do greedy / beam / FK consume the SAME evaluated candidate pool? | identical candidate evaluations replayed through the three selectors; only descendant multiplicity differs | S2 (deferred) |

**This S0 run exercises #1, #2 (anchors + target-only), #4.** #2 (edit-core/halo), #3, #5 arrive with S1/S2.

**Do NOT conclude from this smoke:** any immune-reduction claim, any scTM-based structural-quality claim, or any calibrated threshold — for the §0 cohort-scope reasons (single family; AFDB `scTM` = self-consistency). Elite Head-risk movement is reported as **mechanism telemetry** (does the loop keep/enrich lower-Head feasible states), never as an effect size.

## 7. Analysis (extract, do NOT tune gates yourself — report numbers)
These are **mechanism telemetry** (see §0 cohort-scope and §6.1): report numbers, do **not** state an immune-reduction or scTM structural-quality conclusion — that verdict belongs to the test-set high-risk formal gate.
From the artifacts compute and report:
1. **Coverage**: `n_proteins_ok / n_proteins_failed`; per failed protein the `reason`/`message` (esp. `insufficient_feasible_initial_population` → scTM_min too strict OR designs-per-protein < N).
2. **Elite improvement**: from `fusion_elite.parquet`, `initial_head_global_risk → final_head_global_risk` delta and `improvement` distribution (median, IQR); `n_rounds` actually used.
3. **Gate behavior**: from `fusion_candidates.parquet`, counts by `reason` (`ok / offtarget_hotspot / not_shortlisted / structure fail`), `feasible` fraction, `scTM` and `active_site_RMSD` distributions on evaluated candidates.
4. **Round dynamics**: from `fusion_rounds.jsonl`, per-round `n_feasible_children`, `n_selected_children`, `elite_risk`, `n_refolds`, `cache_hits`, `wall_time_s`.
5. **Population health**: initial unique-sequence fraction (population collapse check). NOTE — `fusion_rounds.jsonl` does not yet emit `unique_fraction` (known §1.7/§F9 telemetry gap, deferred to a follow-up patch); **derive it post-hoc** = `n_distinct(sequence) / N` over the round-0 slots of `fusion_particles.parquet`.
Optional cross-check: re-score the elites (`fusion_elite.parquet` → 3-col facade) with `scripts/evaluate_phase_c.py --mode all` for an independent read — treat as **telemetry only**, not a gate or an effect claim (per §0 cohort-scope, immune/scTM here are not conclusive).

Write findings to `${OUT_DIR}/analysis/fusion_v0_s0_summary.md` (tables of the 5 blocks above + a §6.1 mechanism-acceptance verdict per row + one paragraph: does the **pipeline behave** and do the **gate mechanics** fire correctly, and which placeholder gate to move first when calibration later runs on the high-risk cohort. Do **not** assert an immune-reduction or structural-quality result).

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

## 11. S0 result (2026-07-13) — MECHANISM GATE: **PASS**

Run `uricase_c24_s0_20260711T222303` (job 11032649, ailab H200, 01:24:18; `config_hash 624d3585…`; git `f6eb126`). Parent pool = NoD native gumbel_argmax gen (24×8, all-unique, no collapse; min pairwise Hamming 84).

- **Coverage 18/24 ok.** 6 handoff `insufficient_feasible_initial_population` (never reached the round loop): **4 anchor-firewall** (native gumbel changed a catalytic residue → correctly rejected) + **2 structure-gate attrition**. Not a bug — the handoff anchor firewall working as designed.
- **All §6.1 mechanism invariants PASS**, independently re-verified (0 violations across 64,239 candidates / 216 inherited particles): #1 exact byte-identical carryover, #2 anchor + target-only edits, #4 structure firewall (no infeasible child in ancestry); elite monotone (18/18 Head-risk ↓ — **mechanism telemetry, NOT an immune claim**); init population no collapse (unique-frac 1.0 through all rounds).
- **Binding-gate signals (mechanism, not immune):** `scTM_min=0.85` is **non-binding** here (all 0.91–0.98 — AFDB self-consistency, as §0 predicts); the **active-site RMSD gate (2.0)** and the **N_H off-target gate (0.10, rejects 80% of edits)** are the operative constraints. Real thresholds belong to the B1 high-risk formal gate.
- **Full analysis returned:** `mhc-if-local:…/Results/RF/HLA-DRB1_0701/fusion_v0_s0__20260712T005832Z/` (`analysis/fusion_v0_s0_summary.md` + artifacts + logs + meta).
- **Two Coder caveats (guarded, not in the invariant set):** #1 full closure needs S2 selector-replay; the elite bypasses the #4 firewall by construction and relies on the `state.py` feasibility guard.

**Next (deferred, §10):** S1 repair/H3 arm · S2 selector replay · formal **P1–P3 immune gate on the B1 high-risk test set** (not this uricase family). Structure metric for future runs → **ESMFold2 + sidechain** (this S0 used ESMFold v1 scTM + Cα-shell RMSD). PLAN §6 cohort still to be reconciled to B1 (currently entangled with the uncommitted sidechain-structure PLAN edit — reconcile when that lands).

## 12. High-risk P1 (H1) — calibration IN PROGRESS (2026-07-13)

Per the local Thinker: S0 mechanism cleared → run **P1 (H1) on the high-risk cohort, explicit-only greedy path** (the exact S0-verified mechanism). **P2 (H3 repair) / P3 (H6 FK-beam) stay on hold** until their mechanism smokes (S1/S2, §10) pass. The three structure/search gates are **calibrated from high-risk telemetry, not transferred from the enzyme/AFDB S0** (where scTM_min was non-binding).

**Resolved facts (verified this session):**
- **Cohort** = `highrisk_nod_v1_HLA-DRB1_07_01` (100 proteins, len 102–454). **Crystal premise VERIFIED: 100/100 experimental X-ray** (PDB headers EXPDTA X-RAY, REMARK-2 res 0.84–2.5 Å, crystallographic B-factors) — the deliberate opposite of the uricase AFDB cohort, so scTM(design→crystal) genuinely binds (29-overlap sample already shows scTM bimodal: dense 0.90+ mode + real 0.2–0.5 tail).
- **Parents** = reuse the S0-consistent **DPLM-native gumbel** full-pool (`dplm_native_full_v2_gumbel`, covers 100/100 × 8 unique designs) — NOT the RF-DFM c1_null NoD pool (would swap the parent kernel). No regeneration.
- **Structure = ESMFold2-live** (`backend: esmfold2_live`, ~7 s/refold vs v1's 2.4 s; new metrics `global_ca_RMSD` / sidechain-anchor / pLDDT are additive; scTM still = TMalign vs crystal ref). Cohort is **non-enzyme → `constraint_manifest=none` → scTM is the SOLE structural gate** (active-site path N/A, no fail-fast). pdb_root = `pdbs_if_ready/HLA-DRB1_07_01` (resolves 100/100). Head = a1res03 cv5 fold0.
- **Config** = `configs/rf_refine_fusion_highrisk_probe.yaml`.

**Calibration probe DONE (job 11194105, 9 ok / 1 fail, 81.8 min; 10 length-stratified proteins, n_rounds=8, scTM_min=0.5 loose):**

| Gate | Telemetry | Calibrated value |
|---|---|---|
| **scTM_min** | candidate scTM (edited seqs vs crystal, esmfold2+TMalign): p05=0.742 / p10=0.864 / p25=0.915 / p50=0.951. Rejection: 0.85→9.7%, 0.88→11%, 0.90→16%. Real lower tail (unlike S0's compressed 0.91–0.98). | **0.85** (binding; drops ~10% structure-breaking edits at candidate-p10) |
| **n_rounds** | median elite_risk r0=−3.3→r1=−8.1(Δ4.83)→r2=−9.08(Δ0.94)→r3=−9.26(Δ0.18)→r4=−9.39(Δ0.13)→r5=−9.43(Δ0.04)→r6=−9.45(Δ0.017)→r8=−9.53. Elbow r3–4; r6 captures ~99%. Not fully plateaued by 8 (5/9 still microdescending). | **6** (S0's 3 was too short by ~0.19) |
| **N_H off-target** | rejects 75.7% of candidates (≈S0's 80%) BUT n_feasible_children stays ~28 every round (min 13, never →0); n_selected decline = convergence not starvation. | **keep 0.10** (not starving on the high-burden regime) |

**Cost (ESMFold2, MEASURED):** actual **~2.9 s/refold** (1618 refolds / 78.9 min) — NOT the 7 s the single-fold codex probe suggested (that was model-load-dominated); at scale + cache it matches ESMFold v1. → **full P1 effect run (100 proteins, n_rounds=6) ≈ 12–15 GPU-h**, shard into ~5–6 explicit-PROTEINS sbatch jobs (fusion branch has no array sharding).

**Coverage note:** 1 probe protein (5G3X_A) failed `insufficient_feasible_initial_population` even at the loose 0.5 gate — its DPLM-native parents refold to scTM<0.5 vs crystal. At the effect gate (0.85) more proteins will drop at handoff; this is the scTM gate correctly rejecting proteins whose redesigns do not recover the native crystal fold (report coverage; do not silently loosen).

**Native-parent 0.85 coverage check (pre-launch condition, from probe data, no refold):** recovered all folded parents' scTM from the probe esmfold2 cache via TMalign vs crystal → **≥8/10 probe proteins keep ≥4 native parents at 0.85**; 1 genuine structural fail (5G3X_A: all 8 parents scTM 0.23–0.50 — native IF cannot recover its fold); 1 uncertain (9J7X_A, 2/4 folded ≥0.85). The dropped set is a small, legitimate "design does not fold" tail — **not a cherry-picked subset**. The effect run's own handoff failures give the exact full-100 denominator.

**Effect run (P1 measurement) — LAUNCHED (config `rf_refine_fusion_highrisk_p1.yaml`; local-Thinker GO):** scTM_min 0.85, n_rounds 6, N_H 0.10, N=4, esmfold2_live, greedy explicit-only, constraint_manifest=none, all 100 cohort proteins, **5 shards × 20** (jobs 11199298–11199302, ailab H200). Coverage denominator = the handoff `insufficient_feasible_initial_population` set. Immune-effect readout is elite `global_risk` initial→final at scTM≥0.85, reported vs the NoD baseline (with the same high-risk-is-a-NoD-extreme-set caveat as PROGRESS Module L). **P2 (H3 repair) / P3 (H6 FK-beam) remain on hold pending S1/S2** — do NOT reuse n_rounds=6 for them (their convergence elbow must be re-read after their own mechanism smokes).

**P1 RESULT (jobs 11199298–11199302, 5 shards, ~2 h each; 2026-07-15) — H1 SUPPORTED:**
- **Coverage 76/100** (24 dropped at handoff = native DPLM-gumbel parents cannot reach ≥4 designs at scTM≥0.85 vs crystal). Dropped set is **not** length-biased (dropped median len 172 < entered 227) → a genuine "design does not fold" tail, not a systematic cherry-pick.
- **H1 effect (paired elite init→final, @scTM≥0.85):** Head `global_risk` Δ median **−7.78** (IQR [−10.36, −2.15]); **75/76 improved, 1 flat, 0 worse; 87% meaningful (Δ<−1.0)**; scTM held (init 0.966 → final 0.962, all ≥0.85). This is the pre-registered doc §11 H1: Head-guided hard-state feedback lowers Head burden under the structure constraint.
- **n_rounds=6 validated:** real-run median elite_risk plateaus at r6 (Δ_r6=0.025, near-flat); calibration held.
- **vs NoD (context, confounded):** P1 final beats the *best* NoD design on 75/76 (median Δ −4.57) and the median NoD design on 76/76 (−11.8). Confounded by (a) selection (high-risk = NoD-extreme → regression) and (b) parent kernel (P1=DPLM-native vs NoD=RF-DFM). **The clean H1 claim is the paired init→final −7.78** (frozen-Head-verified before/after on the same DPLM-native design); effect *size* is specific to this high-headroom cohort, direction/mechanism generalizes.
- Deferred external check: NetMHCIIpan on the 76 elites as an independent immune reference (Head is the runtime signal; NMP was never in the loop).

## 13. S1 → P2 (repair, **H3**) — framing + pre-registration (2026-07-15, local-Thinker GO)

**Framing (corrected):** RF repair (`K_repair`) is a **core move family**, NOT a §8.2 proposal-efficiency ablation (that is D2). Its value **= H3**, which doc §13 marks **load-bearing**: it tests the *central novelty* of the whole Fusion vs a standalone post-hoc refiner — **structural co-adaptation / "in-RF beats post-hoc."** P2 is a core mechanism test, not an efficiency knob.

**Raised bar (honest, pre-registered):** P1's explicit-greedy already achieved median −7.78 at scTM≥0.85 with only ~10% of edits rejected by the structure gate. Therefore:
- repair's **"rescue" channel** (fold a rejected explicit edit back to feasible) has a **small upper bound** (~the 10% rejected mass);
- what repair must actually prove is the **"halo-reopen + re-denoise" channel** — reaching synergistic **multi-residue** immune-structure states that explicit single/double-point edits **cannot** reach. This channel is not bounded by the 10%; it is genuinely open.
- **Both H3 outcomes are valid and publishable:** support → co-adaptation novelty confirmed, keep repair; refute (repair reverts the immune edit / no structural gain / dominated by explicit + the state-level filter, doc §11 H3 wording) → Fusion simplifies to **Head-guided explicit-greedy** (still a strong, cheap, NMP-free method) and the co-adaptation claim is honestly dropped.

**S1 (repair MECHANISM smoke — repair has NEVER run with the real DPLM sampler; `build_repair_fn_factory` docstring):** `moves.repair.enabled=true`, `--base-if-checkpoint` (dplm_v1_adapter best.ckpt) + `--rf-sampler-config c1_null.yaml`, esmfold2_live, small cohort/rounds. **Pass iff (PLAN §6 S1):** repair changes ≥1 allowed halo position in a non-degenerate case; the protected immune edit + hard anchors + outside-halo are byte-identical; output complete and Head/structure re-evaluated after repair; no sampler-level accepted-state remask after the child enters ancestry.

**S1 RESULT (job 11226837, 2 proteins, 2026-07-15) — PASS.** First real DPLM-sampler repair run. Two env bugs surfaced+fixed on the way (both explicit-only-invisible, repair-first-only): (1) `submit_refine.slurm` did not `export PROJECT_ROOT` → DPLM `.hydra/config.yaml` `${oc.env:PROJECT_ROOT}` KeyError (added the export); (2) split HF cache — DPLM `airkingbd/dplm_650m` lives in `model_cache/huggingface` but `HF_HOME=model_cache/hf` (where ESMFold2 lives) → symlinked the DPLM model into `hf/hub` (keeps the P1-proven `HF_HOME=hf`). Mechanism verification (126/126 edit_repair candidates): outside-halo byte-identical (diff ⊆ halo) 126/126; repair changed ≥1 halo position beyond the edit 126/126; protected edit applied and **not reverted** (== source explicit edit) 126/126; canonical AA20, no mask 126/126; per-child seed diversity 47/47 edit-groups have distinct children (0 identical); repair children Head+structure re-evaluated. **Repair kernel is mechanism-correct — P2 unlocked.**

**P2 (H3 effect) — PRE-REGISTERED NOW:**
- **Two arms differing ONLY in `moves.repair.enabled`** (explicit-only vs edit+repair), **matched refold budget** (same `N × max_refolds_per_parent × n_rounds` → the binding structure-eval cost is equal; repair is not allowed to win by spending more refolds).
- **Metric:** paired per-protein terminal-elite Head `global_risk` at scTM≥0.85 (same 76-protein high-risk cohort + DPLM-native gumbel parents as P1). **Decompose** any edit+repair gain into (a) *rescue* (repaired versions of otherwise-rejected explicit edits) vs (b) *novel* (repair-only states no explicit edit reached) — the H3 claim rests on (b).
- **Acceptance (direction pre-registered; margin fixed from S1 telemetry BEFORE P2):** edit+repair median terminal Head < explicit-only by a margin exceeding the S1-fixed noise floor, at matched scTM and matched budget. Falsifier = no margin, or all gain is rescue-channel (a).
- **n_rounds:** do NOT reuse P1's 6 — repair does more work per round and converges deeper/slower; its elbow is re-read from S1/P2's own `fusion_rounds` telemetry.

**Repair calibration probe (job 11227328, 10 P1-ok proteins, n_rounds=8, scTM_min=0.85; 2026-07-15):** repair-arm median elite_risk elbow ≈ **r7** (Δ_r5=0.077, Δ_r6=0.079, Δ_r7=0.007, Δ_r8=0.013; 2/10 still micro-descending at r8) — only **marginally later** than explicit's ~r6, NOT the deep-slow convergence the caveat feared. Cost 3.5 s/refold (vs explicit 2.9 — DPLM-sampler overhead). First hint (not conclusive — different proteins, needs the paired run): repair converges to nearly the same median depth as explicit, so the H3 test rests on whether it reaches *distinct better states* on matched proteins, decomposed rescue-vs-novel.

**P2 matched-budget launch plan (HELD for Thinker GO — the H3 core deliverable, ~35 GPU-h):** both arms FRESH at **n_rounds=8** (≥ both elbows; matched budget — reusing P1's explicit@6 would violate matched-budget), scTM_min 0.85, N_H 0.10, N=4, esmfold2_live, same **76 P1-ok proteins**, sharded ~5–6/arm (10–12 jobs). Explicit arm config = `rf_refine_fusion_highrisk_p1.yaml` with n_rounds 6→8; repair arm = the same with `moves.repair.enabled=true` + base-if/rf-sampler. Acceptance (pre-registered): paired Wilcoxon p<0.05 on (edit+repair − explicit) terminal Head at scTM≥0.85, with the advantage attributable to the **novel** channel (not just the ≤10% rescue).
