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
- shared-D0 matched-arm branching and arm-independent paired fork seeds;
- Dual artifacts, cost ledger, resume refusal, reader, dry-run, and assembly preflight;
- optional Dual modes in the existing driver/materializer/launcher; and
- updated `doc/SCRIPTS.md`.

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
M1_COUNTERFACTUAL_CEILING=278

DUAL_POLICY_SPEC=${PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json
COHORT_TABLE=${SCRATCH_BASE}/work/immune-design/if_test_set/if_ready/highrisk/highrisk_nod_v1_HLA-DRB1_07_01.parquet
NOD_NORMALIZATION_ENDPOINTS=${SCRATCH_BASE}/run/inverse_folding/fusion/highrisk_nod_v1_0701/parents/generated.parquet
BASE_K12_RUN=${SCRATCH_BASE}/run/inverse_folding/v2_highrisk/highrisk_gumbel_d4k12_r40_v2_eps005_4root_0701__21cd73d
M1_SAFETY_REFERENCE_MANIFEST=${SCRATCH_BASE}/work/immune-design/v2_canary/references.json

test -r "${HEAD_A_CHECKPOINT}"
test -r "${HEAD_B_CHECKPOINT}"
test -r "${DUAL_POLICY_SPEC}"
test -r "${COHORT_TABLE}"
test -r "${NOD_NORMALIZATION_ENDPOINTS}"
test -r "${M1_SAFETY_REFERENCE_MANIFEST}"
test -d "${BASE_K12_RUN}"
```

Do not substitute a legacy DRB1*04:01 Head or hand-edit a resolved YAML. If either checkpoint or
its content digest differs from the calibration artifact, stop.

---

## 2. P0 — One calibration and exact preflight

### 2.1 Produce the Dual calibration

Reuse the existing V2 exact sequence surfaces; do not regenerate or refold anything.

- Frozen NoD endpoints define $b_A,s_A,b_B,s_B$.
- Existing r40 mechanism endpoints define same-sequence repeatability and preserve the established
  one-cycle source-prefix substrate.
- The RAR 0049 endpoint surface is cross-scored only as an opportunity/launchability read.
- Per-allele cumulative hotspot references use the same exact reference sequences and window grid.

The normalization producer uses the PLAN-frozen protein-equal weighted median/IQR law after
`(protein_id, sequence_md5)` deduplication. This is target-cohort normalization on the A-selected
high-risk NoD panel, not an allele-neutral population claim.

Run the payload below inside an `ailab` GPU allocation (interactive allocation or the site's
approved `sbatch --wrap` pattern), not on the login node. It performs Head inference only and does
not need a new SLURM file.

The extended calibration command is:

```bash
python scripts/calibrate_v2_head_policy.py \
  --mode dual \
  --normalization-parquet "${NOD_NORMALIZATION_ENDPOINTS}" \
  --opportunity-run-root "${BASE_K12_RUN}" \
  --bundle "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism/5zhv_mech" \
  --bundle "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism_b2/5zhv_mech" \
  --bundle "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism/q00511_mech" \
  --bundle "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism_b2/q00511_mech" \
  --head-a-config-dir "${HEAD_CONFIG}" \
  --head-a-checkpoint "${HEAD_A_CHECKPOINT}" \
  --head-a-variant-id "${HEAD_VARIANT_ID}" \
  --head-a-allele "${HEAD_A_ALLELE}" \
  --head-a-allele-idx "${HEAD_A_ALLELE_IDX}" \
  --head-a-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --head-b-config-dir "${HEAD_CONFIG}" \
  --head-b-checkpoint "${HEAD_B_CHECKPOINT}" \
  --head-b-variant-id "${HEAD_VARIANT_ID}" \
  --head-b-allele "${HEAD_B_ALLELE}" \
  --head-b-allele-idx "${HEAD_B_ALLELE_IDX}" \
  --head-b-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --score-scale "${SCORE_SCALE}" \
  --window-k-min "${HEAD_WINDOW_K_MIN}" \
  --window-k-max "${HEAD_WINDOW_K_MAX}" \
  --max-counterfactual-sequences-per-cycle "${M1_COUNTERFACTUAL_CEILING}" \
  --safety-reference-manifest "${M1_SAFETY_REFERENCE_MANIFEST}" \
  --safety-reference-run-root "${BASE_K12_RUN}" \
  --objective-spec "${DUAL_POLICY_SPEC}" \
  --out-rows "${WORK}/calibration/dual_calibration_rows.parquet" \
  --out-json "${WORK}/calibration/dual_policy_calibration.json" \
  --out-opportunity "${WORK}/analysis/dual_opportunity_summary.json"
```

This one JSON must bind both Heads, the normalization population, objective and $\tau$, joint
repeatability/margins (including the analysis margin $\epsilon_J$), both cumulative safety
contracts, score scale, window grid, and content
digests. No value may be copied from a single-Head calibration merely because its name is similar.
The tracked objective spec must already contain an explicit frozen $\tau$; P0 validates it and does
not select or sweep it from these outcomes.

P0 stops before generation when any of these hold:

- either Head misses or mismatches an exact endpoint;
- normalization/objective values are non-finite or content identities disagree;
- the strict and smooth objective readers cannot replay the same endpoint table deterministically;
- the second Head never changes any beyond-margin donor or positive joint-write opportunity on the
  frozen opportunity surface; or
- either `5ZHV_B` or `Q00511` has fewer than 32 stored source prefixes with a common, replayable
  joint-vs-A donor/support contrast opportunity; or
- either allele safety reference cannot be constructed on the common grid.

This is the only opportunity probe. It is a launch gate on the frozen A-driven V2 opportunity
substrate, not a general verdict that a Dual objective can or cannot work on every proposal cloud.

### 2.2 Materialize and preflight the two experiment families

Use the ordinary V2 materialization protocol from the executed V2 runbook. Preserve every
non-Dual argument from its signed source cell and add only:

```text
--dual-policy-spec ${DUAL_POLICY_SPEC}
--dual-calibration-json ${WORK}/calibration/dual_policy_calibration.json
--head-b-config-dir ${HEAD_CONFIG}
--head-b-checkpoint ${HEAD_B_CHECKPOINT}
--head-b-variant-id ${HEAD_VARIANT_ID}
--head-b-allele ${HEAD_B_ALLELE}
```

Materialize:

1. `5zhv_dual_r40` and `q00511_dual_r40` with the matched arm bundle
   `joint,a_only_with_joint_safety`, D=1, K=4, 56 source prefixes, and the signed
   `M1_COUNTERFACTUAL_CEILING`; and
2. one config for every frozen high-risk protein with the matched recursive arm bundle
   `joint,a_only_with_joint_safety,b_only_with_joint_safety`, one shared D0/root, D4/K12/r40.

For C1, derive the per-cycle counterfactual sequence ceiling from the frozen cohort's maximum legal
editable domain and sign that measured ceiling into every resolved config. Do not reuse the M1
value `278` by convenience; the run Head cap must then count both allele scorers over that domain.

For every resolved config:

```bash
source "${WORK}/resolved_configs/${CELL}.args.sh"

MODE_ARGS=()
case "${DUAL_MODE}" in
  qualification)
    MODE_ARGS+=(--dual-qualification --mechanism-prefixes 56)
    ;;
  comparison)
    MODE_ARGS+=(--dual-comparison --exploratory-depth-override)
    ;;
  *)
    echo "unsupported DUAL_MODE=${DUAL_MODE}" >&2
    exit 4
    ;;
esac

python scripts/run_rf_fusion_v2.py \
  --v2-config "${V2_CONFIG}" \
  --out-dir "${RUN}/preflight/${CELL}" \
  --cohort "${V2_COHORT}" \
  "${MODE_ARGS[@]}" \
  --input-file "${INPUT_FILES[@]}" \
  --shard-input "${SHARD_INPUTS[@]}" \
  "journal_dir=${RUN}/preflight/${CELL}/journals" \
  "device=cuda" \
  --dry-run

python scripts/preflight_v2_canary_assembly.py \
  --cell "${V2_CONFIG}=${V2_PROTEIN_ID}"
```

Preflight must print and verify:

- both Head identities and one structure backend;
- objective/calibration/safety digests;
- exact arm bundle and shared-root identity;
- DFE, per-Head logical calls, counterfactual calls, shared refolds, and caps;
- D=1 for mechanism or D4/K12/r40 for recursive capability;
- one common D0 checkpoint for all recursive arms; and
- NMP runtime absent.

All configs must pass before the first GPU submission.

---

## 3. M1 — One-cycle Dual directionality

### 3.1 Frozen comparison

Use the two already established mechanism proteins and prefix substrate:

- `5ZHV_B`
- `Q00511`

For each protein:

- 56 independent source prefixes;
- `r=40`, `c_source=50`, `c_next=60`, D=1, K=4, matching the signed V2F5A mechanism
  substrate;
- `joint` treatment and `A_only_with_joint_safety` control share source, exact endpoint pool,
  descendant fork seeds, horizon, write count, reopen count, and both safety gates;
- both arms score and report both Heads; and
- the only treatment difference is donor/write/reopen evidence authority.

Do not rerun source-shuffle, Head-blind transmission, or a separate structure mechanism. Those
substrate questions are already closed by V2.

### 3.2 Submit

```bash
DUAL_QUALIFICATION=1 \
MECHANISM_PREFIXES=56 \
CELL=5zhv_dual_r40 \
WORK="${WORK}" \
RUNDIR="${RUN}/m1" \
sbatch --job-name=dual_m1_5zhv --time=06:00:00 --mem=64G \
  --account=kaiyijiang --partition=ailab --constraint=h200 --gres=gpu:1 \
  scripts/submit_rf_fusion_v2_canary.slurm

DUAL_QUALIFICATION=1 \
MECHANISM_PREFIXES=56 \
CELL=q00511_dual_r40 \
WORK="${WORK}" \
RUNDIR="${RUN}/m1" \
sbatch --job-name=dual_m1_q00511 --time=06:00:00 --mem=64G \
  --account=kaiyijiang --partition=ailab --constraint=h200 --gres=gpu:1 \
  scripts/submit_rf_fusion_v2_canary.slurm
```

Wait for both jobs and verify `sacct`, launcher logs, manifests, caps, and artifact joins before
opening the comparison.

### 3.3 Read once

```bash
python scripts/analysis/read_v2_mechanism.py \
  --dual-qualification \
  --run-dir "${RUN}/m1/5zhv_dual_r40" \
  --run-dir "${RUN}/m1/q00511_dual_r40" \
  --dual-calibration-json "${WORK}/calibration/dual_policy_calibration.json" \
  --out-dir "${WORK}/analysis/m1"
```

The unit of analysis is the source prefix:

$$
D_p
=
\operatorname{mean}_{\mathrm{forks}}
\left[
J(y_{\mathrm{joint}})-J(y_{\mathrm{A-only}})
\right].
$$

For each protein, average forks within prefix, then form paired prefix differences. Compute the
one-sided 95% upper percentile-bootstrap bound with 10,000 prefix resamples and analysis seed
`20260821`. The two-protein claim is conjunctive (intersection-union): both predeclared proteins
must pass; no extra protein multiplicity correction is applied.

**GO:**

- at least 32 contrastable prefixes per protein;
- exact source/pool/fork integrity and realized write/reopen parity;
- joint treatment changes a donor and/or selected support identity beyond trivial ties;
- one-sided 95% upper bound for paired descendant $D_p$ is below
  $-\epsilon_J$, the frozen joint matched-difference margin, on both proteins; and
- raw A/B outcomes, safety, cost, and typed stalls are complete.

**KILL / unresolved:**

- fewer than 32 contrastable prefixes: underpowered; do not replace proteins or tune objective;
- decision identities do not change: Head B remains an observer;
- projected states change but descendant $J$ is null/adverse: joint actuation fails;
- any source, seed, cardinality, digest, safety, or cost mismatch: invalid run, not a scientific
  negative.

Only M1 GO authorizes recursive Dual comparison.

---

## 4. C1 — Matched recursive capability

### 4.1 Frozen comparison

Use the existing frozen high-risk 100-protein cohort and one source root per protein:

- progressive D4/K12/r40;
- one common D0 capture and exact endpoint pool;
- three branches inside the **same protein cell**:
  - `joint`;
  - `a_only_with_joint_safety`;
  - `b_only_with_joint_safety`;
- all branches use both catastrophic safety gates and the same nominal generation/refold budget;
- all branches preserve the same structure, anchor, source, schedule, and projection contracts; and
- fork seed derivation is matched across objective labels.

Do not launch three separate campaigns with the same numeric seed. V2 seed identity includes
campaign/split content, so those would be independent draws rather than a matched comparison.

The materializer writes 100 cell stems. Split by protein, never by arm:

```bash
split -d -l 25 "${WORK}/cell_lists/all_100.txt" "${WORK}/cell_lists/part_"
```

Each 25-protein list contains three recursive branches sharing D0. This is approximately 75
single-arm cell-equivalents per job, below the executed 100-cell-per-job RAR 0049 K12 scale; the
real preflight budget remains authoritative.

### 4.2 Submit all four before reading outcomes

```bash
for PART in "${WORK}"/cell_lists/part_*; do
  TAG=$(basename "${PART}")
  DUAL_COMPARISON=1 \
  EXPLORATORY_DEPTH_OVERRIDE=1 \
  CELL_LIST="${PART}" \
  WORK="${WORK}" \
  RUNDIR="${RUN}/c1" \
  sbatch --job-name="dual_c1_${TAG}" \
    --account=kaiyijiang --partition=ailab --constraint=h200 --gres=gpu:1 \
    --time=08:00:00 --mem=64G \
    scripts/submit_rf_fusion_v2_canary.slurm
done
```

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

```bash
python scripts/analysis/read_v2_mechanism.py \
  --dual-capability \
  --run-root "${RUN}/c1" \
  --requested-cohort "${COHORT_TABLE}" \
  --dual-calibration-json "${WORK}/calibration/dual_policy_calibration.json" \
  --out-dir "${WORK}/analysis/c1"
```

The reader must fail unless shared-D0 endpoint digests are byte-identical across the three arms.
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

- at least 80 of 100 proteins have a common valid D0 and readable three-arm comparison;
- joint D0-to-final paired upper bound is below $-\epsilon_J$;
- joint-vs-A and joint-vs-B paired upper bounds are both below $-\epsilon_J$;
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

After C1, stop. Do not automatically run K24/K32/D8, multiple roots, protected sequential,
preference branching, online Pareto/FK/MCTS, NMP, or a separate structure study. Those are
conditional studies only if the primary evidence identifies a specific bottleneck.

The cluster handoff is complete when these exist:

- frozen Dual calibration JSON and row table;
- every resolved config/args file and preflight report;
- complete M1 and C1 V2 bundles with Dual endpoint/feedback evidence;
- protein/prefix-equal M1 and C1 summaries;
- common-$J$ endpoint frontier and full requested-cohort coverage;
- Slurm job IDs, `sacct`, stdout/stderr, code/config/content digests, and cost ledgers; and
- one RAR evidence record returned through the standard stable archive path.

Experiment submission/result return does not modify `LOG.md`. A future independent application
cohort may use the frozen C1 winner without retuning; it is not an automatic fourth stage of this
runbook.
