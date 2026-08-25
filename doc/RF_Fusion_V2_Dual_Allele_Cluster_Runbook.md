# RF Fusion V2 Dual-Allele — Cluster Runbook

**Authority:** `PLAN_RF_FUSION_V2_DUAL_ALLELE.md` and
`doc/Dual_Allele_Steering.md`.

**Purpose:** after local code completion, answer only three questions:

1. Can two frozen Heads and one joint objective be bound reproducibly to the same V2 endpoints?
2. Does the second Head change one feedback decision and move fresh descendants toward lower joint
   risk?
3. Does recursive joint steering outperform either single-allele steering law under the same V2
   generation authority?

This runbook has one preparation stage and two GPU experiments. It deliberately contains no tiny
GPU smoke, D/K sweep, temperature sweep, protected sequential pass, FK/Pareto extension, NMP run,
or new structure experiment.

---

## 0. Code-complete gate

Do not submit a GPU job until the coder has closed all of the following:

- Dual-off single-Head V2 regression equivalence;
- paired Head endpoint binding and content-signed calibration;
- objective-aware family/archive/incumbent/donor ordering;
- joint LOO write evidence and symmetric union reopen evidence;
- two-allele cumulative safety with one inherited structure verdict;
- arm-independent seed derivation, so runs that differ only in `--dual-arm` share their
  root capture and depth-zero pool exactly and the comparison is paired across runs;
- Dual artifacts, cost ledger, resume refusal, reader, dry-run, and assembly preflight;
- optional Dual modes in the existing driver/materializer/launcher; and
- updated `doc/SCRIPTS.md`.

> **STATUS (2026-08-25).** Every item above is CLOSED except the reader, with two exceptions the
> operator must not wait on:
>
> * **the Dual reader and the facade's common-$J$ ranking are DEFERRED, deliberately** — see §4.3.
>   `scripts/analysis/read_v2_mechanism.py` and `scripts/materialize_v2_archive_facade.py` contain
>   no Dual path and are not expected to. The §4.3 read is DEFINED (formula + GO conditions) and has
>   no tool; it is built against real bundles once the three arms have run, because join logic
>   written against a schema no run has emitted acquires errors nobody can see;
> * `scripts/calibrate_v2_head_policy.py` was NOT extended. The Dual calibration is a different
>   population (a natural panel, not V2 run bundles) measured on two Heads, so it is
>   `scripts/calibrate_v2_dual_objective.py` — a separate producer, registered in `doc/SCRIPTS.md`.
>
> The three scientific decisions D1/D2/D3 are frozen in `doc/DUAL_ALLELE_DUALF0_AUDIT.md` §J.7 and
> **executed** in §J.8. The §2.1a sensitivity artifact **exists**
> (`dual_overlap_sensitivity_v1.json`, $\lvert\Delta\theta\rvert/c_u = 0.0057$, `within_credit`)
> and both campaign overlays are **signed**. P0's calibration work is therefore complete: your first
> real action is §2.2, materialization and preflight.
>
> **S6's 80/100 coverage floor is still open (AUDIT §4.2, quantified in §J.8) — but it is a READ-TIME
> input, not a launch input.** It appears only in the §4 GO conditions. Do not block submission on
> it; do not declare GO or NO-GO on the coverage criterion until the user has settled it. A healthy
> single-root campaign falls below 80 about **5 % of the time** by chance alone, so a result of 78
> or 79 is not by itself evidence against Dual.

The implementation must extend these existing surfaces:

- `scripts/calibrate_v2_head_policy.py`
- `scripts/materialize_v2_canary_config.py`
- `scripts/run_rf_fusion_v2.py`
- `scripts/preflight_v2_canary_assembly.py`
- `scripts/analysis/read_v2_mechanism.py`
- `scripts/materialize_v2_archive_facade.py`
- `scripts/submit_rf_fusion_v2_canary.slurm`

If a real one-prefix GPU run is still needed to discover that the Dual policy cannot be assembled,
the code is not complete. Fix assembly/preflight instead of adding a smoke stage.

---

## 1. Frozen campaign identity

The first Dual pair is:

- role A: `HLA-DRB1*07:01`, the current Fusion V2 objective;
- role B: `HLA-DRB1*04:01`, a separately frozen a1-res03 Head;
- primary objective: frozen-normalized smooth maximum;
- strict hard maximum: offline sensitivity/control only;
- runtime substrate: current V2 D4/K12/r40 for recursive capability;
- structure, anchors, source/replay, schedule, and intervention budget: unchanged V2 behavior.

Set the common paths once on Della:

```bash
set -euo pipefail
: "${PS1:=}"
export PS1

export PROJECT_ROOT=/home/zc1519/src/Immune-Design
export SCRATCH_BASE=/scratch/gpfs/KAIYIJIANG/zijie
export PYTHONPATH=${PROJECT_ROOT}
cd "${PROJECT_ROOT}"

GIT_SHA=$(git rev-parse HEAD)
CAMPAIGN=dual_0701_0401_smoothmax_d4k12_r40_v1
WORK=${SCRATCH_BASE}/work/immune-design/v2_dual/${CAMPAIGN}__${GIT_SHA:0:8}
RUN=${SCRATCH_BASE}/run/inverse_folding/v2_dual/${CAMPAIGN}__${GIT_SHA:0:8}
mkdir -p "${WORK}"/{calibration,resolved_configs,cell_lists,analysis} "${RUN}"

HEAD_CONFIG=${PROJECT_ROOT}/epitope_head/configs
HEAD_A_CHECKPOINT=${SCRATCH_BASE}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt
HEAD_B_CHECKPOINT=${SCRATCH_BASE}/run/epitope_head/cnn_himp_a1_res03_drb0401_seed42_cv5_fold0/runs/LC1/seed_42/best.pt
HEAD_VARIANT_ID=LC1
HEAD_A_ALLELE=DRB1_0701
HEAD_B_ALLELE=DRB1_0401
HEAD_A_ALLELE_IDX=0
HEAD_B_ALLELE_IDX=0
HEAD_WINDOW_BATCH_SIZE=64
SCORE_SCALE=raw_logit
HEAD_WINDOW_K_MIN=12
HEAD_WINDOW_K_MAX=25
# The per-cycle CANDIDATE domain C, frozen per campaign by AUDIT J.7 D1. It is unique
# counterfactual sequences / legal candidate positions per cycle; the number of Heads observing
# each sequence is resource accounting and may NOT shrink this domain. Both values were already
# signed in POSITION units, so nothing is converted here.
M1_COUNTERFACTUAL_CEILING=278   # 302 - 24 editable, Q00511 mechanism basis (V2F5A signed ceiling)
C1_COUNTERFACTUAL_CEILING=454   # measured maximum legal editable domain, basis 6T88_A
                                # (artifacts/head_policy_v2_eps005_c454.json, unit
                                #  editable_positions_per_cycle)

DUAL_POLICY_SPEC=${PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json
COHORT_TABLE=${SCRATCH_BASE}/work/immune-design/if_test_set/if_ready/highrisk/highrisk_nod_v1_HLA-DRB1_07_01.parquet
NOD_NORMALIZATION_ENDPOINTS=${SCRATCH_BASE}/run/inverse_folding/fusion/highrisk_nod_v1_0701/parents/generated.parquet
BASE_K12_RUN=${SCRATCH_BASE}/run/inverse_folding/v2_highrisk/highrisk_gumbel_d4k12_r40_v2_eps005_4root_0701__21cd73d
M1_SAFETY_REFERENCE_MANIFEST=${SCRATCH_BASE}/work/immune-design/v2_canary/references.json
# The ONE frozen calibration directory. Deliberately NOT ${WORK}/calibration: the panel and
# the signed overlay are campaign-independent artifacts and every campaign reads the same
# ones. An earlier revision of this runbook pointed the calibration commands at the
# per-campaign tree, where none of these files exist.
CALIBRATION_DIR=${SCRATCH_BASE}/work/immune-design/calibration

test -r "${HEAD_A_CHECKPOINT}"
test -r "${HEAD_B_CHECKPOINT}"
test -r "${DUAL_POLICY_SPEC}"
test -r "${COHORT_TABLE}"
test -r "${NOD_NORMALIZATION_ENDPOINTS}"
test -r "${M1_SAFETY_REFERENCE_MANIFEST}"
test -d "${BASE_K12_RUN}"
test -r "${CALIBRATION_DIR}/dual_overlay_v1.json"
```

Do not substitute a legacy DRB1*04:01 Head or hand-edit a resolved YAML. If either checkpoint or
its content digest differs from the calibration artifact, stop.

---

## 2. P0 — One calibration and exact preflight

### 2.1 Produce the Dual calibration

Head inference only. Run inside an `ailab` GPU allocation (interactive, or the site's approved
`sbatch --wrap` pattern), never on the login node. No new SLURM file is required.

**Frozen v1 objective** — the single human-chosen number, already committed in
`inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json`:

$$
c_u = 0.10 \ \text{(normalized risk units)}, \qquad \tau = \frac{0.10}{\log 2} = 0.14426950408889636.
$$

The non-worst allele may buy at most 0.10 normalized risk units against the worse one. The producer
VALIDATES this and signs it; it may never select or sweep it from P0/M1/C1 outcomes.

**Panel** — already BUILT and frozen at `work/immune-design/calibration/tier2_natural_panel_v1.fasta` (13,836 sequences; report alongside it). Rebuild it only if an input changes:

```bash
python scripts/build_dual_calibration_panel.py \
  --candidate-fasta        "${B}/if_test_set/tier2_candidates_merged.fasta" \
  --head-a-manifest-dir    "${B}/manifests/drb0701" \
  --head-b-manifest-dir    "${B}/manifests/drb0401" \
  --deployment-protein-ids "${W}/deployment_protein_ids.txt" \
  --evaluation-protein-ids "${W}/eval_ids_DRB1_07_01.txt" \
  --evaluation-protein-ids "${W}/eval_ids_DRB1_04_01.txt" \
  --seen-splits train,val \
  --out-fasta  "${W}/tier2_natural_panel_v1.fasta" \
  --out-report "${CALIBRATION_DIR}/tier2_natural_panel_v1.report.json" \
  --mmseqs "${MMSEQS}"
```

1. deduplicate by exact sequence — 225,821 raw records to **57,127**;
2. keep 100–500 aa — **all 57,127 already are**;
3. keep canonical AA20 only — drops **652** entries carrying `B`, `O`, `U`, `X` or `Z`, leaving
   **56,475**. A Head request must describe a COMPLETE canonical design and the oracle refuses
   anything else; substituting a residue for an unknown one would invent the measurement the
   coordinate is built from;
4. drop the **union** of both alleles' training homology, `cov-mode 2` at 30 % identity / 80 %
   coverage. MEASURED: **14.12 % for Head A (DRB1\*07:01, 1,182 seen proteins) and 15.78 % for
   Head B (DRB1\*04:01, 1,676 seen proteins)**; union 11,149; **asymmetry 1.66 percentage points**.
   "Seen" is train ∪ val — train fits the weights and validation selects which checkpoint exists;
5. drop the 100 deployment proteins, so no protein's own natural sequence enters its own $u_a$;
6. drop the two per-allele evaluation sets (2,879 + 2,829 ids), so calibrating does not spend the
   test set; and
7. cluster by homology and keep one representative per cluster.

Executed by `scripts/build_dual_calibration_panel.py` (SLURM 12891590). Realized:
57,127 → 56,475 → 45,326 → 45,302 → 44,021 → **13,836 clusters = the panel**.

> **Two corrections to earlier revisions of this passage.** (a) It cited "4.16 % / 4.91 % published
> pool figures" for the homology step. Those numbers appear in no code, artifact or other document;
> they came from the DUALF0 subagent pass and were propagated without their provenance being
> checked. The measured values are ~3.4x larger. (b) It then carried the numbers of the FIRST panel
> build (SLURM 12890926: 14.06 % / 15.69 %, union 11,216, 13,872 clusters), which was superseded the
> same day by 12891590 after the canonical-AA20 filter was added — so this appendix contradicted
> Appendix J and the artifact on disk. The figures above are 12891590's, which is the panel that
> exists.

> **Order of operations.** §2.1a runs BEFORE this section's re-sign, because the re-sign records
> §2.1a's `abs_delta_theta`. The section order here is by subject, not by execution order.

**The calibration is MEASURED and frozen, and both campaign overlays are already SIGNED.**
`dual_overlay_v1.json` (SLURM 12891919, 1 h 26 m on one h200, digest `4bc1e6142e64`) is the primary
record and the coordinate authority. The two launchable artifacts derived from it are

| file | $C$ | logical Head calls / cycle | digest |
|---|---:|---:|---|
| `dual_overlay_m1_v1.json` | 278 | 556 | `f70b96ffe9ad` |
| `dual_overlay_c1_v1.json` | 454 | 908 | `be53f59fd493` |

Both already carry $f_A$, $f_B$ and the §2.1a $\lvert\Delta\theta\rvert$. **Do not re-measure and
do not re-sign.** Run the block below only to VERIFY that the shipped files reproduce — it takes
~0.6 s on a login node and touches no GPU — and compare the printed digests against the table.
Write to a scratch `--out-overlay` when verifying, so a mistake cannot overwrite a signed artifact.

```bash
# M1 and C1 share ONE measured calibration and differ only in the candidate ceiling (J.7 D1).
for CELL in "m1:${M1_COUNTERFACTUAL_CEILING}" "c1:${C1_COUNTERFACTUAL_CEILING}"; do
  NAME=${CELL%%:*}; C=${CELL##*:}
  python scripts/calibrate_v2_dual_objective.py \
    --resign-from "${CALIBRATION_DIR}/dual_overlay_v1.json" \
    --objective-spec "${DUAL_POLICY_SPEC}" \
    --panel-report "${CALIBRATION_DIR}/tier2_natural_panel_v1.report.json" \
    --overlap-inclusion-shift "${OVERLAP_INCLUSION_SHIFT}" \
    --out-rows    "${CALIBRATION_DIR}/dual_calibration_rows_v1.parquet" \
    --out-overlay "${CALIBRATION_DIR}/dual_overlay_${NAME}_v1.json" \
    --max-counterfactual-sequences-per-cycle "${C}"
done
```

`--resign-from` carries the Head-B runtime binding forward from the signed artifact; a
`--head-b-*` value that contradicts it is refused, so the only thing a re-sign chooses is C.
`--out-rows` must point at the SAME measured rows file, because its sha256 is the overlay's
`calibration_artifact_digest`. `${OVERLAP_INCLUSION_SHIFT}` is `abs_delta_theta` from §2.1a and
must be in hand before this runs — re-signing twice would leave two digests per campaign and an
agent could launch the superseded one.

<details><summary>How the primary calibration was measured (already executed — for the record)</summary>

```bash
python scripts/calibrate_v2_dual_objective.py \
  --panel-fasta "${CALIBRATION_DIR}/tier2_natural_panel_v1.fasta" \
  --panel-id tier2_natural_v1 \
  --deployment-protein-ids "${CALIBRATION_DIR}/deployment_protein_ids.txt" \
  --objective-spec inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json \
  --out-rows    "${CALIBRATION_DIR}/dual_calibration_rows_v1.parquet" \
  --out-overlay "${CALIBRATION_DIR}/dual_overlay_v1.json" \
  --max-counterfactual-sequences-per-cycle "${CANDIDATE_CEILING}" \
  --head-a-config-dir "${HEAD_CONFIG}" --head-a-checkpoint "${HEAD_A_CHECKPOINT}" \
  --head-a-variant-id "${HEAD_VARIANT_ID}" --head-a-allele "${HEAD_A_ALLELE}" \
  --head-a-allele-idx "${HEAD_A_ALLELE_IDX}" \
  --head-a-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --head-b-config-dir "${HEAD_CONFIG}" --head-b-checkpoint "${HEAD_B_CHECKPOINT}" \
  --head-b-variant-id "${HEAD_VARIANT_ID}" --head-b-allele "${HEAD_B_ALLELE}" \
  --head-b-allele-idx "${HEAD_B_ALLELE_IDX}" \
  --head-b-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --window-k-min "${HEAD_WINDOW_K_MIN}" --window-k-max "${HEAD_WINDOW_K_MAX}"
```

</details>

`--max-counterfactual-sequences-per-cycle` is **C**, the per-cycle CANDIDATE domain in editable
positions — the same domain a single-Head run has. The logical Head-call budget is `2C` and is
projected by the preflight; it is never enforced by the policy and the legacy single-Head field is
never reinterpreted. All three arms bill `2C`, because every arm scores both Heads and executes
joint safety.

**Read the producer's output before going further.** It prints, per quantity, `b_A s_A b_B s_B` and
the derived joint margin; the normalized intercept `b_A/s_A − b_B/s_B` with its bootstrap SE; and the
measured cross-allele Pearson r. That last number is worth pausing on: the existing r ≈ 0.19 is a
WITHIN-family (uricase) estimate, and the whole domain-transfer argument is calibrated against it.

The producer refuses, rather than reports, a coordinate whose scale is at or below that Head's own
measured repeat drift: the normalized coordinate would then be mostly instrument noise and no margin
derived from it could mean anything.

**Margins are derived, never declared.** With $d_a = e_a/s_a$ and $e_a$ the measured same-sequence
drift, $\epsilon_{\mathrm{donor}} = \epsilon_{\mathrm{write}} = 2\max(d_A, d_B)$ — twice, because
both gates compare a DIFFERENCE of two measurements. The frozen raw figure `0.017012596130371094` is
a raw-logit quantity and may not be copied into the normalized joint objective.

There is no $\epsilon_J$.

### 2.1a Overlap-inclusion sensitivity — MANDATORY before M1/C1 submission

AUDIT §J.7 D2. A second calibration whose panel build is identical to the primary in every filter,
Head identity, objective law and scoring protocol, and differs ONLY in that step 3 keeps the union
of both Heads' training homology instead of removing it.

**It is a label, not a candidate.** The overlap-EXCLUDED 13,836-cluster panel was selected by an
outcome-independent leakage rule and remains the coordinate authority whatever this measures. A
large drift limits the method claim to that frozen reference panel; it never promotes the
contaminated one. An integrity or provenance failure of the artifact DOES block launch.

```bash
# 1. the sensitivity panel (CPU, ~45 s)
python scripts/build_dual_calibration_panel.py \
  --candidate-fasta        "${SCRATCH_BASE}/work/immune-design/if_test_set/tier2_candidates_merged.fasta" \
  --head-a-manifest-dir    "${SCRATCH_BASE}/work/immune-design/manifests/drb0701" \
  --head-b-manifest-dir    "${SCRATCH_BASE}/work/immune-design/manifests/drb0401" \
  --deployment-protein-ids "${CALIBRATION_DIR}/deployment_protein_ids.txt" \
  --evaluation-protein-ids "${CALIBRATION_DIR}/eval_ids_DRB1_07_01.txt" \
  --evaluation-protein-ids "${CALIBRATION_DIR}/eval_ids_DRB1_04_01.txt" \
  --seen-splits train,val \
  --head-overlap include \
  --out-fasta  "${CALIBRATION_DIR}/tier2_overlap_included_panel_v1.fasta" \
  --out-report "${CALIBRATION_DIR}/tier2_overlap_included_panel_v1.report.json" \
  --work-dir   "${CALIBRATION_DIR}/_panel_work_incl" --mmseqs "${MMSEQS}"

# 2. the sensitivity calibration + comparison (ailab h200, ~1 h 35 m; request 02:30:00 -- a
#    longer wall clock does not get scheduled on this partition)
python scripts/calibrate_v2_dual_objective.py \
  --panel-fasta  "${CALIBRATION_DIR}/tier2_overlap_included_panel_v1.fasta" \
  --panel-id     tier2_overlap_included_v1 \
  --panel-report "${CALIBRATION_DIR}/tier2_overlap_included_panel_v1.report.json" \
  --deployment-protein-ids "${CALIBRATION_DIR}/deployment_protein_ids.txt" \
  --objective-spec "${DUAL_POLICY_SPEC}" \
  --out-rows       "${CALIBRATION_DIR}/dual_calibration_rows_overlap_included_v1.parquet" \
  --out-overlay    "${CALIBRATION_DIR}/dual_overlay_overlap_included_v1.json" \
  --compare-to     "${CALIBRATION_DIR}/dual_overlay_v1.json" \
  --out-comparison "${CALIBRATION_DIR}/dual_overlap_sensitivity_v1.json" \
  --max-counterfactual-sequences-per-cycle "${M1_COUNTERFACTUAL_CEILING}" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260821 --repeat-window-batch-size 32 \
  --head-a-config-dir "${HEAD_CONFIG}" --head-a-checkpoint "${HEAD_A_CHECKPOINT}" \
  --head-a-variant-id "${HEAD_VARIANT_ID}" --head-a-allele "${HEAD_A_ALLELE}" \
  --head-a-allele-idx "${HEAD_A_ALLELE_IDX}" \
  --head-a-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --head-b-config-dir "${HEAD_CONFIG}" --head-b-checkpoint "${HEAD_B_CHECKPOINT}" \
  --head-b-variant-id "${HEAD_VARIANT_ID}" --head-b-allele "${HEAD_B_ALLELE}" \
  --head-b-allele-idx "${HEAD_B_ALLELE_IDX}" \
  --head-b-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --window-k-min "${HEAD_WINDOW_K_MIN}" --window-k-max "${HEAD_WINDOW_K_MAX}"
```

The C signed into the sensitivity overlay is immaterial — that artifact is a MEASUREMENT and is
never launched. Only `dual_overlay_m1_v1.json` and `dual_overlay_c1_v1.json` are launchable.

`dual_overlap_sensitivity_v1.json` (`dual-overlap-sensitivity-1`) reports, for both panels, the
four coordinate constants `b_A s_A b_B s_B`, the scale ratio, `theta = b_A/s_A - b_B/s_B`,
cross-allele `r`, the bootstrap SE, panel counts and content digests; then `delta_theta`,
`abs_delta_theta`, `delta_log_scale_ratio`, `abs_delta_theta_over_credit` and the
`within_credit` / `exceeds_credit` label. The producer REFUSES two calibrations that share a panel
digest, or that differ in objective law or evaluator identity — a drift measured across two
instruments is not a panel sensitivity.

If the report is ever lost, recompute it from the two signed overlays instead of re-running the
GPU job:

```bash
python scripts/calibrate_v2_dual_objective.py \
  --compare-overlays "${CALIBRATION_DIR}/dual_overlay_v1.json" \
                     "${CALIBRATION_DIR}/dual_overlay_overlap_included_v1.json" \
  --out-comparison   "${CALIBRATION_DIR}/dual_overlap_sensitivity_v1.json"
```

**Then feed `abs_delta_theta` into §2.1's re-sign as `${OVERLAP_INCLUSION_SHIFT}`.** The value
lands on the signed key `leave_overlap_out_shift` (kept under its historical name so the schema
digest is stable) and the producer refuses to revise an already-measured one.

**EXECUTED 2026-08-25 — this artifact exists; do not re-run it.** The two panels agree on every
count through `after_canonical` = 56,475 and on `f_A` = 14.12 % / `f_B` = 15.78 %; the include
build (SLURM 12931496) keeps 56,475 and clusters to **14,937** against the primary's 13,836 —
11,149 extra pool entries add only 1,101 independent families. The calibration (SLURM 12931501,
1 h 32 m) measured

$$
\Delta\theta = -5.6885\times10^{-4},
\qquad \frac{\lvert\Delta\theta\rvert}{c_u} = 0.0057,
\qquad \Delta\log\frac{s_A}{s_B} = 3.75\times10^{-3},
$$

label **`within_credit`**. `OVERLAP_INCLUSION_SHIFT=0.0005688505037048097` is therefore already
baked into `dual_overlay_{m1,c1}_v1.json`, which are signed and ready; §2.1's re-sign is a
VERIFICATION step for you, not a production step. See AUDIT §J.8 for what this measurement does and
does not license.

### 2.2 Materialize and preflight

Use the ordinary V2 materialization protocol from the executed V2 runbook. Preserve every non-Dual
argument from its signed source cell and add only:

```text
--dual-overlay      ${DUAL_OVERLAY}
--head-b-config-dir ${HEAD_CONFIG}
--head-b-checkpoint ${HEAD_B_CHECKPOINT}
```

Paths only. The role B variant id, allele index and window batch size live in the SIGNED overlay and
are read from there; the materializer does not accept them, because a knob that reads as a control
and changes nothing is worse than a missing one.

**`${DUAL_OVERLAY}` is the CAMPAIGN's overlay, and the two campaigns do not share one** (J.7 D1):

```bash
# M1 mechanism cells
DUAL_OVERLAY=${CALIBRATION_DIR}/dual_overlay_m1_v1.json   # C = 278, logical Head calls 556
# C1 high-risk D4/K12 cells
DUAL_OVERLAY=${CALIBRATION_DIR}/dual_overlay_c1_v1.json   # C = 454, logical Head calls 908
```

Both carry the SAME measured calibration — identical panel, coordinates, law and objective digest —
so the two campaigns are steered by one objective and differ only in how much candidate domain a
cycle may open. They are distinct `content_digest`s and therefore distinct run signatures, which is
what stops an M1 fragment being reused by a C1 cell. Never launch
`dual_overlay_overlap_included_v1.json`: it is a MEASUREMENT of panel sensitivity, not a campaign.

**One config per protein.** The arm is a LAUNCH flag (`--dual-arm`), not a materialized bundle:
there are no per-arm configs, no arm bundles to resolve and no in-cell branching. The same resolved
config is launched once per arm.

Derive the per-cycle counterfactual sequence ceiling from the cohort's maximum legal editable
domain and sign that measured ceiling into every resolved config. A Dual run scores the same
counterfactual set on BOTH Heads, so the run's Head cap must count two scorers over that domain --
the launch gate already prices this (`project_v2_budget(..., n_heads=2)`), but the signed ceiling
itself is yours to set.

## 3. Comparing the arms

**Run it twice and compare the artifacts.** There is no in-process matched-pair machinery for this
and none is needed.

Seeds derive from `(campaign_id, split_role, master_seed, protein_id)`; the Dual arm enters none of
them, and the depth-zero root capture never sees the Dual runtime. So two runs that differ only in
`--dual-arm` produce **the same root, the same lookahead pool and the same role A raw risks**, and
diverge exactly where the design intends -- at selection. The comparison is therefore PAIRED by
construction, not by machinery. `tests/inverse_folding/test_fusion_v2_dual_arms.py::
test_separate_runs_of_two_arms_share_their_pool_by_determinism_alone` asserts this so it cannot
silently stop being true.

```bash
for ARM in joint a_only; do
  python scripts/run_rf_fusion_v2.py \
    --v2-config "${V2_CONFIG}" --cohort "${V2_COHORT}" \
    --out-dir "${RUN}/${ARM}" \
    --dual-overlay "${DUAL_OVERLAY}" --dual-arm "${ARM}" \
    --input-file "${INPUT_FILES[@]}" --shard-input "${SHARD_INPUTS[@]}"
done
```

**Verify the pairing held**, on the artifacts rather than on trust -- for each protein, the depth-0
rows of `complete_endpoints.parquet` must agree between the two runs on `endpoint_id`,
`sequence_md5` and `head_global_risk`. They will differ from that point on, and that difference IS
the result.

**What to read.** `dual_endpoint_evidence` carries both raw risks, both normalized coordinates, the
arm's value and the active-worst allele, so the two-axis return boundary is recoverable per arm.
`dual_feedback_evidence` carries the per-allele leave-one-out contrasts and the union reducer's
winner, which answers *which allele demanded a given write or reopen*. `dual_terminal_summary`
carries each arm's best joint value and the depth at which it last improved. Every row is stamped
with its arm, and the run manifest carries the arm plus the overlay, calibration, objective and
role-B checkpoint identities, so two bundles can never be confused.

**One thing this does NOT give you**: the two runs each pay for the shared source prefix and its
depth-0 lookahead pool. An engine that forked one realized pool would pay for that once. That is a
compute saving and nothing else -- it changes no scientific property of the comparison.

## 4. Recursive capability across the arms

Use the existing frozen high-risk 100-protein cohort, progressive D4/K12/r40, one source root per
protein. Launch the whole cohort once per arm:

```bash
for ARM in joint a_only b_only; do
  CELL_LIST="${PART}" WORK="${WORK}" RUNDIR="${RUN}/${ARM}" \
  DUAL=1 DUAL_ARM="${ARM}" DUAL_OVERLAY="${DUAL_OVERLAY}" \
  EXPLORATORY_DEPTH_OVERRIDE=1 \
  sbatch --job-name="dual_${ARM}_$(basename "${PART}")" \
    --account=kaiyijiang --partition=ailab --constraint=h200 --gres=gpu:1 \
    --time=08:00:00 --mem=64G \
    scripts/submit_rf_fusion_v2_canary.slurm
done
```

Every non-arm argument must be byte-identical across the three launches -- same config, same
`campaign_id`, same `split_role`, same `master_seed`, same cohort, same inputs. That is what makes
them one experiment rather than three: the Dual arm enters no seed, so the three runs share their
root capture and their depth-zero pool exactly, and diverge only at selection.

Do not tune or relaunch an arm after viewing another arm. Typed per-cell failures remain in the
requested denominator.

### 4.3 Common-objective read

Each single-allele runtime archive was optimized under a different law, so final fairness does not
come from comparing each arm's own native elite. Reconstruct the same smooth-max frontier from all
definitively feasible exact endpoints:

$$
F_{p,a,d}
=
\min_{\substack{
y\in\mathrm{arm}\ a,\;\mathrm{depth}(y)\leq d\\
I_{\mathrm{adm}}^{AB}(y\mid\bar y)=1
}}
J_\tau(y).
$$

> **STATUS (2026-08-25): this read has no tool yet.** An earlier revision printed a
> `read_v2_mechanism.py --dual-capability --run-root … --requested-cohort … --dual-calibration-json
> … --out-dir …` command. **None of those five flags exists** — the reader's whole surface is
> `--bundle` (required, nargs=+), `--delta`, `--floor`, `--cap`, `--confirmatory`,
> `--qualification`, `--config`, `--min-prefixes`, `--out` — and no producer anywhere writes
> `dual_policy_calibration.json`; that filename occurred in that one line and nowhere else in the
> repository. The Dual reader is the item deliberately deferred in
> `doc/DUAL_ALLELE_DUALF0_AUDIT.md` ("Not implemented, deliberately"): writing join logic against a
> schema no run had emitted is how an analysis path acquires an error nobody can see. Now that
> `dual_endpoint_evidence`, `dual_feedback_evidence` and `dual_terminal_summary` have a producer and
> a written schema, it can be built — against real bundles, once the three arms have run.
>
> The read itself is defined by the formula above and by the GO conditions below; what is missing is
> the tool, not the definition.

The reader must fail unless the three arms share the same depth-zero **molecules and identities**:
identical `sequence_md5`, identical `endpoint_id` and identical `source_state_id` per fork. It may
**not** compare endpoint content digests. A `CompleteEndpoint` carries one Head's binding and score,
and the `b_only` arm's endpoints necessarily carry role B's evidence for the same molecule, so their
content digests differ by construction; requiring byte-identical digests would fail every valid
three-arm run. `endpoint_id` is derived from `(source_state_id, fork_index, sequence_md5)` and is
therefore the identity that genuinely is shared.
It reports protein-equal-weight:

- joint final vs shared D0;
- joint final vs A-only final;
- joint final vs B-only final;
- raw and normalized A/B changes;
- per-depth common-$J$ frontier and last improving depth;
- donor/write identity divergence and typed stall composition;
- balanced elite and derived non-dominated admissible endpoint view;
- logical DFE/Head/refold cost, cache reuse, and full 100-protein coverage.

**GO / freeze the optional Dual module:**

The shared-D0 checks below are performed ACROSS the three runs' artifacts -- there is no in-cell
branching to verify. For each protein, the depth-0 rows of the three `complete_endpoints.parquet`
must agree on `endpoint_id`, `sequence_md5` and `source_state_id`. If they do not, some non-arm
argument differed between the launches and the comparison is not paired.

- at least 80 of 100 proteins have a common valid D0 and readable three-arm comparison;
- the joint D0-to-final paired bound satisfies $\operatorname{UCB}_{95\%}\left[\operatorname{mean}\left(\Delta J\right)\right]<0$;
- the joint-vs-A and joint-vs-B paired bounds each satisfy the same;
- both raw allele axes and all catastrophic guardrails are reported without hidden prefiltering;
- all 100 requested proteins, including failures, remain in coverage; and
- shared-D0 identity, common admissible selection domain, nominal DFE/refold budget, paired seed
  law, and per-Head call accounting all pass their exact parity checks.

The 80-protein coverage floor is inherited from the prior 84/100 RAR 0049 successful-protein
surface. It is a validity/coverage guard, not a new structure objective or a reason to relax the
inherited gate.

Compute paired protein differences over proteins with a valid shared D0 and all three readable
arms. A typed stall returns its carried incumbent and therefore remains a readable arm; only an
invalid/incomplete execution is missing. Retain every invalid/missing protein in the 100-protein
coverage table without imputing an effect. Use a one-sided 95% upper percentile-bootstrap bound with 10,000 protein resamples and seed
`20260821`. The joint-vs-A and joint-vs-B claim is conjunctive, so the two 95% tests form an
intersection-union decision; joint-vs-D0 is a required directionality check but not an additional
comparator claim.

**Interpretation:**

- joint beats only one single arm: partial result; simultaneous Dual advantage is not established;
- joint does not beat A-only: the optional Dual layer has not improved the current method;
- joint does not beat B-only: the method mostly switched which single allele it optimizes;
- a separately signed deployment-protection analysis may report no protected design without
  changing the balanced capability verdict;
- smooth max and hard max rank the realized endpoint surface identically: report the offline
  sensitivity only. Terminal rank agreement is insufficient to replace smooth max because donor,
  LOO write, and recursive identities may still differ. Any future replacement requires complete
  decision replay with objective-specific margins and, if identities differ, a matched recursive
  control.

---

## 5. Stop condition and returned evidence

After the arm comparison, stop. Do not automatically run K24/K32/D8, multiple roots, protected sequential,
preference branching, online Pareto/FK/MCTS, NMP, or a separate structure study. Those are
conditional studies only if the primary evidence identifies a specific bottleneck.

The cluster handoff is complete when these exist:

- frozen Dual calibration JSON and row table;
- every resolved config/args file and preflight report;
- one complete V2 bundle per arm, with Dual endpoint/feedback evidence;
- protein-equal cross-arm summaries;
- common-$J$ endpoint frontier and full requested-cohort coverage;
- Slurm job IDs, `sacct`, stdout/stderr, code/config/content digests, and cost ledgers; and
- one RAR evidence record returned through the standard stable archive path.

Experiment submission/result return does not modify `LOG.md`. A future independent application
cohort may use the frozen winning arm without retuning; it is not an automatic fourth stage of this
runbook.
