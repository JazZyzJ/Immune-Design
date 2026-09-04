#!/bin/bash
# Packed CPU array worker for representative-sample uricase Rosetta energy.
#SBATCH --job-name=uric_energy
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

ENERGY_MANIFEST="${1:?ENERGY_ELIGIBLE_MANIFEST.parquet}"
OUTPUT_ROOT="${2:?ENERGY_OUTPUT_ROOT}"
RUN_ROOT="${3:?ROSETTA_RUN_ROOT}"
LOG_ROOT="${4:?ROSETTA_LOG_ROOT}"
PYTHON_BIN="${5:?PYTHON_BIN}"
RUNNER="${6:?RUN_URICASE_ENERGY_PARENT.py}"
EVALUATOR="${7:?EVAL_TETRAMER_REFERENCE.py}"
EXPECTED_PARENTS="${8:?EXPECTED_PARENTS}"
ARRAY_TASKS="${9:-${EXPECTED_PARENTS}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || \
   [[ ! "${EXPECTED_PARENTS}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${ARRAY_TASKS}" =~ ^[1-9][0-9]*$ ]] || ((TASK_ID >= ARRAY_TASKS)); then
    echo "FATAL: invalid task/count: task=${TASK_ID} tasks=${ARRAY_TASKS} parents=${EXPECTED_PARENTS}" >&2
    exit 2
fi
for required_file in "${ENERGY_MANIFEST}" "${PYTHON_BIN}" "${RUNNER}" "${EVALUATOR}"; do
    [[ -f "${required_file}" ]] || { echo "FATAL: missing file: ${required_file}" >&2; exit 2; }
done
case "/${OUTPUT_ROOT}/" in */work/*) ;; *) echo "FATAL: OUTPUT_ROOT must be in work/" >&2; exit 2 ;; esac
case "/${RUN_ROOT}/" in */run/*) ;; *) echo "FATAL: RUN_ROOT must be in run/" >&2; exit 2 ;; esac
case "/${LOG_ROOT}/" in */logs/*) ;; *) echo "FATAL: LOG_ROOT must be in logs/" >&2; exit 2 ;; esac

module purge
module load anaconda3/2025.12
module load rosetta/3.15
conda activate immune-design
ROSETTA_INTERFACE_ANALYZER="$(command -v InterfaceAnalyzer)"
ROSETTA_SCRIPTS="$(command -v rosetta_scripts)"

export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export PYTHONHASHSEED=0
export PYTHONPATH="/home/zc1519/src/Immune-Design:${PYTHONPATH:-}"
export LD_PRELOAD="/home/zc1519/src/Immune-Design/lib/ijit_stub.so${LD_PRELOAD:+:${LD_PRELOAD}}"

for ((ROW_INDEX=TASK_ID; ROW_INDEX<EXPECTED_PARENTS; ROW_INDEX+=ARRAY_TASKS)); do
    "${PYTHON_BIN}" -u "${RUNNER}" \
        --energy-manifest "${ENERGY_MANIFEST}" \
        --row-index "${ROW_INDEX}" \
        --expected-parents "${EXPECTED_PARENTS}" \
        --output-root "${OUTPUT_ROOT}" \
        --run-root "${RUN_ROOT}" \
        --log-root "${LOG_ROOT}" \
        --evaluator "${EVALUATOR}" \
        --rosetta-interface-analyzer "${ROSETTA_INTERFACE_ANALYZER}" \
        --rosetta-scripts "${ROSETTA_SCRIPTS}" \
        --rosetta-timeout-seconds 1800
done
