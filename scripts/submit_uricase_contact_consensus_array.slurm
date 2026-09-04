#!/bin/bash
# CPU array worker for one five-sample uricase contact-consensus parent.
# Scheduler stdout/stderr are caller-supplied under logs/.
#SBATCH --job-name=uric_contact
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

PILOT_MANIFEST="${1:?PILOT_MANIFEST.parquet}"
PRED_ROOT="${2:?PROTENIX_PRED_ROOT}"
CANONICAL_FASTA="${3:?PILOT_FASTA}"
POSITION_ANNOTATIONS="${4:?POSITION_ANNOTATIONS.parquet}"
OUTPUT_ROOT="${5:?CONTACT_OUTPUT_ROOT}"
PYTHON_BIN="${6:?PYTHON_BIN}"
RUNNER="${7:?RUN_URICASE_CONTACT_CONSENSUS_PARENT.py}"
EVALUATOR="${8:?EVAL_TETRAMER_REFERENCE.py}"
EXPECTED_PARENTS="${9:?EXPECTED_PARENTS}"
ARRAY_TASKS="${10:-${EXPECTED_PARENTS}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "${EXPECTED_PARENTS}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${ARRAY_TASKS}" =~ ^[1-9][0-9]*$ ]] || ((TASK_ID >= ARRAY_TASKS)); then
    echo "FATAL: invalid task/parent count: task=${TASK_ID} tasks=${ARRAY_TASKS} parents=${EXPECTED_PARENTS}" >&2
    exit 2
fi
for required_file in "${PILOT_MANIFEST}" "${CANONICAL_FASTA}" "${POSITION_ANNOTATIONS}" "${RUNNER}" "${EVALUATOR}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "FATAL: required file missing: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! -d "${PRED_ROOT}" ]]; then
    echo "FATAL: prediction root missing: ${PRED_ROOT}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "FATAL: Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
case "/${OUTPUT_ROOT}/" in
    */work/*) ;;
    *) echo "FATAL: OUTPUT_ROOT must be inside a work/ layer" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export PYTHONHASHSEED=0

for ((ROW_INDEX=TASK_ID; ROW_INDEX<EXPECTED_PARENTS; ROW_INDEX+=ARRAY_TASKS)); do
    "${PYTHON_BIN}" -u "${RUNNER}" \
        --pilot-manifest "${PILOT_MANIFEST}" \
        --row-index "${ROW_INDEX}" \
        --expected-parents "${EXPECTED_PARENTS}" \
        --pred-root "${PRED_ROOT}" \
        --canonical-fasta "${CANONICAL_FASTA}" \
        --position-annotations "${POSITION_ANNOTATIONS}" \
        --output-root "${OUTPUT_ROOT}" \
        --evaluator "${EVALUATOR}" \
        --sasa-point-number 1000
done
