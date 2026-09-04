#!/usr/bin/env bash
# Submit all rows of a query-centered PLMC manifest as a bounded batched CPU array.
#
# Every path is caller-supplied. --output-root must live under a run/ layer and
# --log-dir under a logs/ layer. Explicit sbatch stdout/stderr paths keep runtime
# products and scheduler logs separated. Manifest rows are distributed by stride
# over a fixed scheduler-task count to remain below array/QOS submit limits.

set -euo pipefail
: "${PS1:=}"
export PS1

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}"
}

MANIFEST=""
OUTPUT_ROOT=""
LOG_DIR=""
PLMC_BINARY=""
EVCOUPLINGS_PYTHON=""
MAX_CONCURRENT=15
ARRAY_TASK_COUNT=512
DRY_RUN=0
SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="${2:?missing value for --manifest}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
        --log-dir) LOG_DIR="${2:?missing value for --log-dir}"; shift 2 ;;
        --plmc-binary) PLMC_BINARY="${2:?missing value for --plmc-binary}"; shift 2 ;;
        --evcouplings-python)
            EVCOUPLINGS_PYTHON="${2:?missing value for --evcouplings-python}"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="${2:?missing value for --max-concurrent}"
            shift 2
            ;;
        --array-task-count)
            ARRAY_TASK_COUNT="${2:?missing value for --array-task-count}"
            shift 2
            ;;
        --sbatch-arg)
            SBATCH_ARGS+=("${2:?missing value for --sbatch-arg}")
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for variable in MANIFEST OUTPUT_ROOT LOG_DIR PLMC_BINARY EVCOUPLINGS_PYTHON; do
    if [[ -z "${!variable}" ]]; then
        echo "FATAL: missing required option for ${variable}" >&2
        usage >&2
        exit 2
    fi
done
if [[ ! "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: --max-concurrent must be a positive integer" >&2
    exit 2
fi
if [[ ! "${ARRAY_TASK_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: --array-task-count must be a positive integer" >&2
    exit 2
fi
for argument in "${SBATCH_ARGS[@]}"; do
    case "${argument}" in
        --array=*|--output=*|--error=*|--dependency=*|--parsable)
            echo "FATAL: --sbatch-arg may not override launcher-owned option: ${argument}" >&2
            exit 2
            ;;
    esac
done
if [[ ! -f "${MANIFEST}" ]]; then
    echo "FATAL: manifest not found: ${MANIFEST}" >&2
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
case "/${OUTPUT_ROOT}/" in
    */run/*) ;;
    *) echo "FATAL: --output-root must be inside a run/ layer: ${OUTPUT_ROOT}" >&2; exit 2 ;;
esac
case "/${LOG_DIR}/" in
    */logs/*) ;;
    *) echo "FATAL: --log-dir must be inside a logs/ layer: ${LOG_DIR}" >&2; exit 2 ;;
esac

expected_header=$'protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length'
actual_header="$(head -n 1 "${MANIFEST}" | tr -d '\r')"
if [[ "${actual_header}" != "${expected_header}" ]]; then
    echo "FATAL: unexpected PLMC manifest header: ${actual_header}" >&2
    exit 2
fi
n_rows="$(awk 'NR > 1 && NF > 0 { n += 1 } END { print n + 0 }' "${MANIFEST}")"
if [[ "${n_rows}" -lt 1 ]]; then
    echo "FATAL: PLMC manifest has no model rows" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/submit_evolution_plmc_array.slurm"
if [[ ! -f "${WORKER}" ]]; then
    echo "FATAL: array worker not found: ${WORKER}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

scheduler_tasks="${ARRAY_TASK_COUNT}"
if ((n_rows < ARRAY_TASK_COUNT)); then
    scheduler_tasks="${n_rows}"
fi
array_last="$((scheduler_tasks - 1))"
printf 'PLMC array rows=%s scheduler_tasks=%s row_stride=%s max_concurrent=%s\n' \
    "${n_rows}" "${scheduler_tasks}" "${scheduler_tasks}" "${MAX_CONCURRENT}"

command=(
    sbatch
    --parsable
    "--array=0-${array_last}%${MAX_CONCURRENT}"
    "--output=${LOG_DIR}/plmc_%A_%a.out"
    "--error=${LOG_DIR}/plmc_%A_%a.err"
)
command+=("${SBATCH_ARGS[@]}")
command+=(
    "${WORKER}"
    "${MANIFEST}"
    "${OUTPUT_ROOT}"
    "${PLMC_BINARY}"
    "${EVCOUPLINGS_PYTHON}"
    "${SCRIPT_DIR}/run_evolution_plmc.py"
    0
    "${n_rows}"
    "${scheduler_tasks}"
    1
)

printf 'PLMC batched command='
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: no job submitted"
else
    submit_output="$("${command[@]}")"
    job_id="${submit_output%%;*}"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "FATAL: sbatch returned an invalid job id: ${submit_output}" >&2
        exit 2
    fi
    echo "PLMC submitted_job_id=${job_id}"
fi
