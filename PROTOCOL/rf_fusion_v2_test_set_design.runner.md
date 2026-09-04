# RUNNER: RF Fusion V2 single-allele test-set design

Fixed execution procedure for `PROTOCOL/rf_fusion_v2_test_set_design.md`. All `{{slots}}` come
from one CASE. Execute in order. A failed verification stops the run; do not loosen a threshold,
drop a protein, change a seed, or hand-edit a resolved YAML.

## Required CASE slots

| slot | CASE field |
|---|---|
| `{{tag}}` | `tag` |
| `{{code_revision}}` | `config.code_revision` |
| `{{campaign_id}}` | `config.campaign_id` |
| `{{test_set_parquet}}`, `{{pdb_root}}` | `inputs.*` |
| `{{allele}}`, `{{head_allele}}` | `inputs.allele`, `config.head_allele` |
| `{{head_checkpoint}}`, `{{head_config_dir}}` | `models.*` |
| `{{dplm_checkpoint}}`, `{{structure_backend}}` | `models.*` |
| `{{policy_calibration_json}}`, `{{hotspot_json}}`, `{{band_json}}` | `inputs.*` |
| `{{work_root}}`, `{{run_root}}`, `{{log_root}}` | `outputs.*` |
| Slurm partition/QoS/GPU/CPU/memory/time/cells-per-job | `cli_overrides.resources` |

The four standard root seeds are `20260811`, `20260812`, `20260813`, and `20260814` unless a new
protocol revision changes them. The same numeric seeds in a new `campaign_id` remain independent
because campaign identity enters seed derivation.

## Step 0 — validate the CASE

```bash
python scripts/validate_case.py PROTOCOL/cases/{{tag}}.yaml
```

Proceed only on `ALL-PASS`.

## Step 1 — freeze a clean code worktree

Never run from a dirty research worktree.

```bash
export RF_CODE_REVISION={{code_revision}}
export RF_PROJECT_ROOT={{frozen_project_root}}

if [ ! -e "${RF_PROJECT_ROOT}" ]; then
  git worktree add --detach "${RF_PROJECT_ROOT}" "${RF_CODE_REVISION}"
fi
test "$(git -C "${RF_PROJECT_ROOT}" rev-parse HEAD)" = "${RF_CODE_REVISION}"
test -z "$(git -C "${RF_PROJECT_ROOT}" status --porcelain)"
```

The chosen revision must contain the closed `testset_d4_k12_r40` profile and must be the revision
written into every resolved config. Do not copy current-tree Python files over this worktree.

## Step 2 — declare paths and immutable method inputs

```bash
export TEST_SET_PARQUET={{test_set_parquet}}
export PDB_ROOT={{pdb_root}}
export HEAD_ALLELE={{head_allele}}
export HEAD_CHECKPOINT={{head_checkpoint}}
export HEAD_CONFIG_DIR={{head_config_dir}}
export DPLM_CHECKPOINT={{dplm_checkpoint}}
export POLICY_CALIBRATION_JSON={{policy_calibration_json}}
export HOTSPOT_JSON={{hotspot_json}}
export BAND_JSON={{band_json}}
export STRUCTURE_BACKEND={{structure_backend}}
export ESMFOLD2_SITE_PACKAGES={{esmfold2_site_packages}}

export CAMPAIGN_ID={{campaign_id}}
export WORK_ROOT={{work_root}}
export RUN_ROOT={{run_root}}
export LOG_ROOT={{log_root}}

export TEMPLATE_SOURCE="${RF_PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_canary_state_transition.yaml"
export POLICY_SPEC="${RF_PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_head_directed_capped_policy_v2.json"
export RF_SAMPLER_CONFIG="${RF_PROJECT_ROOT}/inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml"
export STRUCTURE_CONFIG="${RF_PROJECT_ROOT}/inverse_folding/reference_flow/configs/rf_refine_fusion_highrisk_sctm070.yaml"
export MATERIALIZER="${RF_PROJECT_ROOT}/scripts/materialize_v2_canary_config.py"
export LAUNCHER="${RF_PROJECT_ROOT}/scripts/submit_rf_fusion_v2_canary.slurm"

test -f "${TEST_SET_PARQUET}"
test -d "${PDB_ROOT}"
test -f "${HEAD_CHECKPOINT}"
test -d "${HEAD_CONFIG_DIR}"
test -f "${DPLM_CHECKPOINT}"
test -f "${POLICY_CALIBRATION_JSON}"
test -f "${HOTSPOT_JSON}"
test -f "${BAND_JSON}"
test -f "${STRUCTURE_BACKEND}"
test -f "${TEMPLATE_SOURCE}"
test -f "${POLICY_SPEC}"
test -f "${RF_SAMPLER_CONFIG}"
test -f "${STRUCTURE_CONFIG}"
```

## Step 3 — build campaign-local reference, stratum, template, and job-list inputs

This step is cheap and model-free. It refuses duplicate IDs, non-AA20 sequences, length mismatch,
or ambiguous/missing backbones. It writes no scientific score.

```bash
export CELLS_PER_JOB={{cells_per_job}}
mkdir -p "${WORK_ROOT}"/{analysis,artifacts,references,resolved_configs,cell_lists,refold_cache} \
         "${RUN_ROOT}" "${LOG_ROOT}"

python - <<'PY'
import hashlib, json, math, os
from pathlib import Path
import pandas as pd
import yaml

aa20 = set("ACDEFGHIKLMNPQRSTVWY")
test_set = Path(os.environ["TEST_SET_PARQUET"])
pdb_root = Path(os.environ["PDB_ROOT"])
work = Path(os.environ["WORK_ROOT"])
head_allele = os.environ["HEAD_ALLELE"]
cells_per_job = int(os.environ["CELLS_PER_JOB"])

frame = pd.read_parquet(test_set).copy()
required = {"protein_id", "sequence", "sequence_length"}
missing = sorted(required - set(frame.columns))
assert not missing, missing
assert frame["protein_id"].nunique() == len(frame)

references, backbones, records = {}, {}, []
for row in frame.sort_values("protein_id").itertuples(index=False):
    protein_id, sequence = str(row.protein_id), str(row.sequence)
    assert sequence and not (set(sequence) - aa20), protein_id
    assert len(sequence) == int(row.sequence_length), protein_id
    found = [pdb_root / f"{protein_id}{suffix}" for suffix in (".pdb", ".cif")
             if (pdb_root / f"{protein_id}{suffix}").is_file()]
    assert len(found) == 1, (protein_id, found)
    sequence_path = work / "references" / f"{protein_id}.seq"
    sequence_path.write_text(sequence, encoding="utf-8")
    digest = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    references[protein_id] = {"path": str(sequence_path.resolve()), "sha256": digest}
    backbones[protein_id] = str(found[0].resolve())
    records.append((protein_id, int(row.sequence_length), backbones[protein_id],
                    str(sequence_path.resolve())))

(work / "references.json").write_text(json.dumps(references, indent=2, sort_keys=True))
(work / "strata.json").write_text(json.dumps(
    {protein_id: "highrisk_global_unconstrained" for protein_id in references},
    indent=2, sort_keys=True))
(work / "analysis" / "cells.tsv").write_text("".join(
    f"{protein_id}\t{backbone}\t{sequence_path}\n"
    for protein_id, _length, backbone, sequence_path in records))

n_parts = math.ceil(len(records) / cells_per_job)
capacity = math.ceil(len(records) / n_parts)
bins, loads = [[] for _ in range(n_parts)], [0] * n_parts
for protein_id, length, _backbone, _sequence_path in sorted(
        records, key=lambda row: (-row[1], row[0])):
    eligible = [index for index in range(n_parts) if len(bins[index]) < capacity]
    target = min(eligible, key=lambda index: (loads[index], len(bins[index]), index))
    bins[target].append(protein_id)
    loads[target] += length
for root_index in range(4):
    for part_index, protein_ids in enumerate(bins):
        (work / "cell_lists" / f"root{root_index}_part{part_index}.txt").write_text(
            "".join(f"root{root_index}_{protein_id}\n" for protein_id in sorted(protein_ids)))

template = yaml.safe_load(Path(os.environ["TEMPLATE_SOURCE"]).read_text())
template["head"]["allele"] = head_allele
template_path = work / "artifacts" / "v2_testset_template.yaml"
template_path.write_text(yaml.safe_dump(template, sort_keys=True), encoding="utf-8")

summary = {
    "n_proteins": len(records),
    "max_sequence_length": max(row[1] for row in records),
    "longest_protein": max(records, key=lambda row: (row[1], row[0]))[0],
    "n_parts_per_root": n_parts,
    "root_seeds": [20260811, 20260812, 20260813, 20260814],
    "head_allele": head_allele,
}
(work / "analysis" / "campaign_inputs.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

The CASE must record the resulting cohort count/hash and `max_sequence_length`. The
`POLICY_CALIBRATION_JSON` counterfactual ceiling must cover that maximum editable domain. Set the
whole-run Head cap to the larger of 6000 and `4 * policy_ceiling + 128`; this is an engineering
ceiling, not a search reward.

## Step 4 — materialize all four roots

```bash
export CAMPAIGN_TEMPLATE="${WORK_ROOT}/artifacts/v2_testset_template.yaml"
export POLICY_C=$(jq -r '.head_directed.max_counterfactual_head_calls_per_cycle' \
  "${POLICY_CALIBRATION_JSON}")
export MAX_SEQUENCE_LENGTH=$(jq -r '.max_sequence_length' \
  "${WORK_ROOT}/analysis/campaign_inputs.json")
if [ "${POLICY_C}" -lt "${MAX_SEQUENCE_LENGTH}" ]; then
  echo "policy counterfactual ceiling ${POLICY_C} is below max sequence length ${MAX_SEQUENCE_LENGTH}" >&2
  exit 4
fi
export RUN_MAX_HEAD_CALLS=$(( 4 * POLICY_C + 128 ))
if [ "${RUN_MAX_HEAD_CALLS}" -lt 6000 ]; then export RUN_MAX_HEAD_CALLS=6000; fi

ROOT_SEEDS=(20260811 20260812 20260813 20260814)
for ROOT_INDEX in 0 1 2 3; do
  MASTER_SEED=${ROOT_SEEDS[$ROOT_INDEX]}
  while IFS=$'\t' read -r PID BACKBONE REFERENCE_SEQUENCE; do
    python "${MATERIALIZER}" \
      --template "${CAMPAIGN_TEMPLATE}" \
      --out "${WORK_ROOT}/resolved_configs/root${ROOT_INDEX}_${PID}.yaml" \
      --protein-id "${PID}" --r-step 40 \
      --stratum-key highrisk_global_unconstrained \
      --code-revision "${RF_CODE_REVISION}" \
      --band-json "${BAND_JSON}" --hotspot-json "${HOTSPOT_JSON}" \
      --reference-manifest "${WORK_ROOT}/references.json" \
      --stratum-manifest "${WORK_ROOT}/strata.json" \
      --reference-sequence "${REFERENCE_SEQUENCE}" \
      --projection-policy-spec "${POLICY_SPEC}" \
      --head-config-dir "${HEAD_CONFIG_DIR}" \
      --head-checkpoint "${HEAD_CHECKPOINT}" --head-variant-id LC1 \
      --structure-backend "${STRUCTURE_BACKEND}" \
      --structure-config "${STRUCTURE_CONFIG}" \
      --v0-structure-gate-config "${STRUCTURE_CONFIG}" \
      --rf-sampler-config "${RF_SAMPLER_CONFIG}" \
      --dplm-checkpoint "${DPLM_CHECKPOINT}" \
      --cohort-table "${TEST_SET_PARQUET}" --backbone "${BACKBONE}" \
      --pdb-root "${PDB_ROOT}" --refold-cache-dir "${WORK_ROOT}/refold_cache" \
      --esmfold2-site-packages "${ESMFOLD2_SITE_PACKAGES}" \
      --campaign-id "${CAMPAIGN_ID}" --master-seed "${MASTER_SEED}" \
      --policy-calibration-json "${POLICY_CALIBRATION_JSON}" \
      --exploratory-profile testset_d4_k12_r40 \
      --run-max-head-calls "${RUN_MAX_HEAD_CALLS}" \
      --esmfold2-model biohub/ESMFold2 --esmfold2-num-loops 3 \
      --esmfold2-num-sampling-steps 50 --esmfold2-num-diffusion-samples 1 \
      --esmfold2-seed 0
  done < "${WORK_ROOT}/analysis/cells.tsv"
done
```

Verify exactly `4 * n_proteins` YAML, `.args.sh`, and `.structure_runtime.json` files. The
materializer loads every generated YAML before returning; no resolved YAML is edited afterward.

## Step 5 — real-driver model-free preflight

Run on the longest protein because it gives the largest Head-call projection.

```bash
LONGEST_PID=$(jq -r '.longest_protein' "${WORK_ROOT}/analysis/campaign_inputs.json")
source "${WORK_ROOT}/resolved_configs/root0_${LONGEST_PID}.args.sh"

python "${RF_PROJECT_ROOT}/scripts/run_rf_fusion_v2.py" \
  --v2-config "${V2_CONFIG}" \
  --out-dir "${RUN_ROOT}/__dry_run_probe__" \
  --cohort "${V2_COHORT}" --dry-run --exploratory-depth-override \
  --input-file "${INPUT_FILES[@]}" \
  --shard-input "${SHARD_INPUTS[@]}" \
  journal_dir="${RUN_ROOT}/__dry_run_probe__/journals" device=cuda
```

Require: `depth_cap=4`, `active_population_width=1`, `1990` logical DFE, `60` definitive refolds,
no breached cap, policy v2, epsilon `0.005`, local tolerance `0.017012596130371094`, and
`split_role=exploratory_testset_design_v1`.

## Step 6 — submit balanced serial cell lists

One Slurm job processes one list serially. Do not create one job per protein.

```bash
for CELL_LIST in "${WORK_ROOT}"/cell_lists/root*_part*.txt; do
  STEM=$(basename "${CELL_LIST}" .txt)
  sbatch --job-name="v2ts_${STEM}" \
    --partition={{partition}} --qos={{qos}} --gres={{gpu_request}} \
    --cpus-per-task={{cpus_per_task}} --mem={{memory}} --time={{walltime}} \
    --output="${LOG_ROOT}/%x_%j.out" --error="${LOG_ROOT}/%x_%j.err" \
    --export="ALL,PROJECT_ROOT=${RF_PROJECT_ROOT},SCRATCH_BASE={{scratch_base}},WORK=${WORK_ROOT},RUNDIR=${RUN_ROOT},CELL_LIST=${CELL_LIST},CONDA_ENV={{conda_env}},EXPLORATORY_DEPTH_OVERRIDE=1,MECHANISM_PREFIXES=0,QUALIFICATION=0,DEVICE=cuda" \
    "${LAUNCHER}"
done
```

Record every returned job ID in the CASE immediately.

## Step 7 — monitor without misclassifying typed partial completion

```bash
squeue -j {{comma_separated_job_ids}} -o '%.18i %.20j %.2t %.10M %.10l %R'
sacct -j {{comma_separated_job_ids}} \
  --format=JobIDRaw,JobName%24,State,ExitCode,Elapsed,Timelimit,MaxRSS,AllocTRES%40 -P
```

Exit `3:0` is expected when a serial list contains both successful and typed-failed cells. A
timeout, OOM, missing list tail, Python traceback, or absent bundle is not expected.

## Step 8 — close the complete cell grid

Require exactly `4 * n_proteins` readable manifests and all standard parquet tables. Require clean
realized caps in every manifest. Count `n_ok=1` and `n_ok=0` separately and report protein-level
coverage across four roots; never drop the latter.

```bash
python - <<'PY'
import collections, json, os
from pathlib import Path
import pyarrow.parquet as pq

run_root = Path(os.environ["RUN_ROOT"])
expected = 4 * json.loads(
    (Path(os.environ["WORK_ROOT"]) / "analysis" / "campaign_inputs.json").read_text()
)["n_proteins"]
bundles = sorted(path.parent for path in run_root.glob("root*/run_manifest.json"))
assert len(bundles) == expected, (len(bundles), expected)

by_protein, totals = collections.defaultdict(list), collections.Counter()
tables = ("archive.parquet", "complete_endpoints.parquet",
          "structure_evaluations.parquet", "terminal_validation.parquet")
for bundle in bundles:
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    protein_id = manifest["requested_cohort"][0]
    by_protein[protein_id].append(int(manifest["n_ok"]))
    assert manifest["realized_caps"]["within"] is True
    assert manifest["realized_caps"]["breached"] == []
    assert manifest["realized_caps"]["unverifiable"] == []
    for name in tables:
        path = bundle / name
        assert path.is_file(), path
        totals[name] += pq.ParquetFile(path).metadata.num_rows
assert all(len(values) == 4 for values in by_protein.values())
print(json.dumps({
    "bundles": len(bundles),
    "proteins": len(by_protein),
    "successful_cells": sum(sum(values) for values in by_protein.values()),
    "zero_ok_cells": sum(4 - sum(values) for values in by_protein.values()),
    "proteins_with_success": sum(sum(values) > 0 for values in by_protein.values()),
    "all_root_failed_proteins": sum(sum(values) == 0 for values in by_protein.values()),
    "rows": totals,
}, indent=2, sort_keys=True))
PY
```

## Step 9 — materialize the two standard generation surfaces

```bash
mkdir -p "${WORK_ROOT}/selection"
BUNDLE_ARGS=()
while IFS=$'\t' read -r PID _BACKBONE _REFERENCE_SEQUENCE; do
  for ROOT_INDEX in 0 1 2 3; do
    BUNDLE_ARGS+=(--bundle "${RUN_ROOT}/root${ROOT_INDEX}_${PID}")
  done
done < "${WORK_ROOT}/analysis/cells.tsv"

SEED_ARGS=(
  --expected-master-seed 20260811
  --expected-master-seed 20260812
  --expected-master-seed 20260813
  --expected-master-seed 20260814
)

python "${RF_PROJECT_ROOT}/scripts/materialize_v2_archive_facade.py" \
  "${BUNDLE_ARGS[@]}" "${SEED_ARGS[@]}" \
  --mode feasible_immune_pareto \
  --output "${WORK_ROOT}/selection/feasible_immune_pareto.parquet"

python "${RF_PROJECT_ROOT}/scripts/materialize_v2_archive_facade.py" \
  "${BUNDLE_ARGS[@]}" "${SEED_ARGS[@]}" \
  --mode structure_rejected_fallback \
  --output "${WORK_ROOT}/selection/structure_rejected_fallback.parquet"
```

The exporter validates the complete master-seed grid, campaign/config/cap identities, exact Head
payload identity, terminal evidence, structure backend/gate identity, and sequence convergence
before selecting. Do not replace it with a parquet concatenation.

## Step 10 — close the CASE and hand off

Update the CASE with actual cell/protein counts, row counts, job states, and both selection paths.
Set `status: selected` only after Step 9 succeeds. The full run tree remains the archive authority.

The refinement agent consumes the frozen selection paths and extends this RUNNER after the
reserved boundary; it does not change Steps 0--9 retroactively.
