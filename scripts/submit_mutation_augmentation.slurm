#!/bin/bash
#SBATCH --job-name=mut-aug
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/network/zc1519/run/augmentation/shard_%a_%j.out
#SBATCH --error=/scratch/network/zc1519/run/augmentation/shard_%a_%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
AUG_OUTPUT="/scratch/network/zc1519/work/immune-design/augmentation"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

N_SHARDS=20

mkdir -p "${AUG_OUTPUT}"
mkdir -p /scratch/network/zc1519/run/augmentation

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TMPDIR="/tmp/netmhciipan_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

echo "=========================================="
echo "Stage J: Mutation Registry — Shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
echo "=========================================="

# ── Build registry shard ─────────────────────────────────────────────────
python -m scripts.build_mutation_registry \
    --backend standalone \
    --binary "${NETMHCIIPAN_BIN}" \
    --data-root "${DATA_DIR}" \
    --output-dir "${AUG_OUTPUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --batch-size 30

echo "Shard ${SLURM_ARRAY_TASK_ID} complete."
rm -rf "${TMPDIR}"
