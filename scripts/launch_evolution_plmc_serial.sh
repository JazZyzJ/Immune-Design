#!/usr/bin/env bash
# Submit one allocation that reruns every PLMC manifest row sequentially.
#
# Usage: launch_evolution_plmc_serial.sh --manifest TSV --output-root RUN_DIR
#   --log-dir LOG_DIR --plmc-binary PLMC --evcouplings-python PYTHON
#   [--sbatch-arg ARG ...] [--dry-run]

set -euo pipefail
: "${PS1:=}"
export PS1

usage() {
    sed -n '2,6p' "${BASH_SOURCE[0]}"
}

MANIFEST=""
OUTPUT_ROOT=""
LOG_DIR=""
PLMC_BINARY=""
EVCOUPLINGS_PYTHON=""
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
        --sbatch-arg)
            case "${2:?missing value for --sbatch-arg}" in
                --array*|--output*|--error*)
                    echo "FATAL: serial launcher owns --array/--output/--error" >&2
                    exit 2
                    ;;
            esac
            SBATCH_ARGS+=("$2")
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
    *) echo "FATAL: --output-root must be inside a run/ layer" >&2; exit 2 ;;
esac
case "/${LOG_DIR}/" in
    */logs/*) ;;
    *) echo "FATAL: --log-dir must be inside a logs/ layer" >&2; exit 2 ;;
esac

expected_header=$'protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length'
actual_header="$(head -n 1 "${MANIFEST}" | tr -d '\r')"
if [[ "${actual_header}" != "${expected_header}" ]]; then
    echo "FATAL: unexpected PLMC manifest header: ${actual_header}" >&2
    exit 2
fi
n_rows="$(awk 'NR > 1 && NF > 0 { n++ } END { print n + 0 }' "${MANIFEST}")"
if [[ "${n_rows}" -lt 1 ]]; then
    echo "FATAL: PLMC manifest has no model rows" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/submit_evolution_plmc_serial.slurm"
RUNNER="${SCRIPT_DIR}/run_evolution_plmc.py"
if [[ ! -f "${WORKER}" || ! -f "${RUNNER}" ]]; then
    echo "FATAL: serial worker or PLMC runner is missing under ${SCRIPT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

command=(
    sbatch
    "--output=${LOG_DIR}/plmc_serial_%j.out"
    "--error=${LOG_DIR}/plmc_serial_%j.err"
)
command+=("${SBATCH_ARGS[@]}")
command+=(
    "${WORKER}" "${MANIFEST}" "${OUTPUT_ROOT}"
    "${PLMC_BINARY}" "${EVCOUPLINGS_PYTHON}" "${RUNNER}"
)

printf 'PLMC serial rows=%s command=' "${n_rows}"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: no job submitted"
else
    "${command[@]}"
fi
