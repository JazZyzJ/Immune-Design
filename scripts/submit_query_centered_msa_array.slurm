#!/bin/bash
# Normalize one deterministic ColabFold A3M shard into strict query-centered MSAs.
#
# Positional arguments:
#   1 frozen input root (contains shards/shard_NNN.fasta)
#   2 raw A3M shard root (contains NNN/<protein_id>.a3m)
#   3 normalized shard root (writes NNN/)
#   4 EVcouplings environment Python
#   5 prepare_query_centered_msas.py
#
# Submit as an array with stdout/stderr explicitly routed to logs/.
#SBATCH --job-name=uric_msa_norm
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

FROZEN_ROOT="${1:?FROZEN_INPUT_ROOT}"
RAW_SHARD_ROOT="${2:?RAW_A3M_SHARD_ROOT}"
NORMALIZED_SHARD_ROOT="${3:?NORMALIZED_SHARD_ROOT}"
EVCOUPLINGS_PYTHON="${4:?EVCOUPLINGS_PYTHON}"
RUNNER="${5:?PREPARE_QUERY_CENTERED_MSAS.py}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ ! -x "${EVCOUPLINGS_PYTHON}" ]]; then
    echo "FATAL: EVcouplings Python is not executable: ${EVCOUPLINGS_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${RUNNER}" ]]; then
    echo "FATAL: normalization runner not found: ${RUNNER}" >&2
    exit 2
fi

printf -v SHARD_TAG "%03d" "${TASK_ID}"
QUERY_FASTA="${FROZEN_ROOT}/shards/shard_${SHARD_TAG}.fasta"
RAW_DIR="${RAW_SHARD_ROOT}/${SHARD_TAG}"
OUT_DIR="${NORMALIZED_SHARD_ROOT}/${SHARD_TAG}"
if [[ ! -f "${QUERY_FASTA}" ]]; then
    echo "FATAL: frozen query shard not found: ${QUERY_FASTA}" >&2
    exit 2
fi
if [[ ! -d "${RAW_DIR}" ]]; then
    echo "FATAL: raw A3M shard directory not found: ${RAW_DIR}" >&2
    exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
    echo "FATAL: normalized shard output already exists: ${OUT_DIR}" >&2
    exit 2
fi
mkdir -p "${NORMALIZED_SHARD_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "shard=${SHARD_TAG} query=${QUERY_FASTA} raw=${RAW_DIR} out=${OUT_DIR}"
srun "${EVCOUPLINGS_PYTHON}" -u "${RUNNER}" \
    --query-fasta "${QUERY_FASTA}" \
    --a3m-dir "${RAW_DIR}" \
    --output-dir "${OUT_DIR}" \
    --coverage 0.6 0.7 0.8 \
    --plmc-coverage 0.6 \
    --theta 0.8 \
    --neff-engine evcouplings \
    --covariance-neff-per-l 10.0
