#!/bin/bash
#SBATCH --job-name=imm-merge
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/imm_merge_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/imm_merge_%j.err

# Thin SLURM wrapper around scripts/merge_eval_immune_shards.py — merges the
# per-shard evaluate_phase_c immune outputs (from build_wt_facade.py --n-shards +
# submit_benchmark.slurm N_SHARDS array) into one unified run dir. Submit as a
# --dependency=afterok job on the facade-shard array. Parameterized via env:
#   OUTPUT_ROOT  (e.g. /scratch/.../run/benchmark/wt_v2)
#   ALLELE_TAG   (e.g. HLA-DRB1_04_01 — safe_allele_tag form)
#   RUN_ID       (BASE run_id, no shard suffix)
#   N_SHARDS     (must match the array size)
#   EXPECTED_PROTEINS (optional; asserts merged unique protein count)

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
ALLELE_TAG="${ALLELE_TAG:?set ALLELE_TAG}"
RUN_ID="${RUN_ID:?set RUN_ID}"
N_SHARDS="${N_SHARDS:?set N_SHARDS}"
EXPECTED_PROTEINS="${EXPECTED_PROTEINS:-0}"

module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Immune shard merge"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  OUTPUT    : ${OUTPUT_ROOT}"
echo "  ALLELE    : ${ALLELE_TAG}"
echo "  RUN_ID    : ${RUN_ID}"
echo "  N_SHARDS  : ${N_SHARDS}"
echo "  EXPECTED  : ${EXPECTED_PROTEINS}"
echo "  Timestamp : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

python -u "${PROJECT_ROOT}/scripts/merge_eval_immune_shards.py" \
    --output-root "${OUTPUT_ROOT}" \
    --allele-tag "${ALLELE_TAG}" \
    --run-id "${RUN_ID}" \
    --n-shards "${N_SHARDS}" \
    --expected-proteins "${EXPECTED_PROTEINS}"

echo "Merge finished."
