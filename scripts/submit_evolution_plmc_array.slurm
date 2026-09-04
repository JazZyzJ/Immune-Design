#!/bin/bash
# CPU-only batched-array worker for query-centered PLMC models.
#
# Submit through launch_evolution_plmc_array.sh so stdout/stderr are always placed
# in the caller-supplied logs/ layer. The all-parent manifest is emitted by
# prepare_query_centered_msas.py; N_eff gates annotate covariance interpretation
# but never suppress Potts model fitting. A scheduler task can cover a strided
# subset of manifest rows; every parent still runs in its own fresh PLMC process.
#SBATCH --job-name=uric_plmc
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

MANIFEST="${1:?PLMC_MANIFEST.tsv}"
OUTPUT_ROOT="${2:?RUN_OUTPUT_ROOT}"
PLMC_BINARY="${3:?PLMC_BINARY}"
EVCOUPLINGS_PYTHON="${4:?EVCOUPLINGS_PYTHON}"
RUNNER="${5:?RUN_EVOLUTION_PLMC.py}"
ROW_OFFSET="${6:-0}"
ROW_STOP_EXCLUSIVE="${7:-}"
ROW_STRIDE="${8:-1}"
OVERWRITE_MODE="${9:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "${ROW_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: manifest row offset must be a non-negative integer" >&2
    exit 2
fi
GLOBAL_TASK_ID="$((ROW_OFFSET + TASK_ID))"
if [[ -z "${ROW_STOP_EXCLUSIVE}" ]]; then
    ROW_STOP_EXCLUSIVE="$((GLOBAL_TASK_ID + 1))"
fi
if [[ ! "${ROW_STOP_EXCLUSIVE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: manifest row stop must be a positive integer" >&2
    exit 2
fi
if [[ ! "${ROW_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: manifest row stride must be a positive integer" >&2
    exit 2
fi
if [[ "${OVERWRITE_MODE}" != "0" && "${OVERWRITE_MODE}" != "1" ]]; then
    echo "FATAL: overwrite mode must be 0 or 1" >&2
    exit 2
fi
if ((GLOBAL_TASK_ID >= ROW_STOP_EXCLUSIVE)); then
    echo "FATAL: task has no assigned manifest row: global=${GLOBAL_TASK_ID}, " \
         "stop=${ROW_STOP_EXCLUSIVE}" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "FATAL: PLMC manifest not found: ${MANIFEST}" >&2
    exit 2
fi
if [[ ! -x "${PLMC_BINARY}" ]]; then
    echo "FATAL: PLMC binary is not executable: ${PLMC_BINARY}" >&2
    exit 2
fi
if [[ ! -x "${EVCOUPLINGS_PYTHON}" ]]; then
    echo "FATAL: EVcouplings Python is not executable: ${EVCOUPLINGS_PYTHON}" >&2
    exit 2
fi

if [[ ! -f "${RUNNER}" ]]; then
    echo "FATAL: PLMC runner not found: ${RUNNER}" >&2
    exit 2
fi

expected_header=$'protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length'
actual_header="$(head -n 1 "${MANIFEST}" | tr -d '\r')"
if [[ "${actual_header}" != "${expected_header}" ]]; then
    echo "FATAL: unexpected PLMC manifest header: ${actual_header}" >&2
    exit 2
fi
export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"

echo "local_task_id=${TASK_ID} row_offset=${ROW_OFFSET} " \
     "row_stop_exclusive=${ROW_STOP_EXCLUSIVE} row_stride=${ROW_STRIDE} " \
     "overwrite_mode=${OVERWRITE_MODE}"
processed=0
for ((global_task_id = GLOBAL_TASK_ID; global_task_id < ROW_STOP_EXCLUSIVE; global_task_id += ROW_STRIDE)); do
    line="$(awk -F '\t' -v row="$((global_task_id + 2))" 'NR == row { print; exit }' "${MANIFEST}")"
    if [[ -z "${line}" ]]; then
        echo "FATAL: local array task ${TASK_ID} has no global manifest row " \
             "${global_task_id}" >&2
        exit 2
    fi
    IFS=$'\t' read -r PROTEIN_ID QUERY_ID ALIGNMENT_PATH COVARIANCE_GATE NEFF NEFF_PER_L <<< "${line}"
    if [[ ! "${PROTEIN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "FATAL: unsafe protein_id in manifest: ${PROTEIN_ID}" >&2
        exit 2
    fi
    if [[ "${QUERY_ID}" != "${PROTEIN_ID}" ]]; then
        echo "FATAL: query_id ${QUERY_ID} does not match protein_id ${PROTEIN_ID}" >&2
        exit 2
    fi
    if [[ ! -f "${ALIGNMENT_PATH}" ]]; then
        echo "FATAL: focus alignment not found: ${ALIGNMENT_PATH}" >&2
        exit 2
    fi

    OUTPUT_PREFIX="${OUTPUT_ROOT}/${PROTEIN_ID}/plmc"
    mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
    echo "local_task_id=${TASK_ID} global_task_id=${global_task_id} " \
         "protein_id=${PROTEIN_ID} covariance_gate=${COVARIANCE_GATE} " \
         "neff=${NEFF} neff_per_L=${NEFF_PER_L}"
    runner_args=(
        --alignment "${ALIGNMENT_PATH}"
        --query-id "${QUERY_ID}"
        --output-prefix "${OUTPUT_PREFIX}"
        --plmc-binary "${PLMC_BINARY}"
        --theta 0.8
        --lambda-h 0.01
        --lambda-j-base 0.01
        --iterations 500
        --cpu "${CPUS}"
    )
    if [[ "${OVERWRITE_MODE}" == "1" ]]; then
        runner_args+=(--overwrite)
    fi
    srun "${EVCOUPLINGS_PYTHON}" -u "${RUNNER}" "${runner_args[@]}"
    processed="$((processed + 1))"
done
echo "local_task_id=${TASK_ID} completed_models=${processed}"
