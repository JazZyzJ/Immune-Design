#!/bin/bash
#SBATCH --job-name=aug-drb0401
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/augmentation/drb0401_shard_%a_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/augmentation/drb0401_shard_%a_%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/manifests/drb0401"
AUG_OUTPUT="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/augmentation/drb0401"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/augmentation"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

N_SHARDS=20

mkdir -p "${AUG_OUTPUT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TMPDIR="/tmp/netmhciipan_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage J: Mutation Registry (DRB1*04:01) — Shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Output dir   : ${AUG_OUTPUT}"
echo "  NetMHCIIpan  : ${NETMHCIIPAN_BIN}"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Build registry shard ─────────────────────────────────────────────────
python -m scripts.build_mutation_registry \
    --config "${PROJECT_ROOT}/epitope_head/configs/augmentation_drb0401.yaml" \
    --backend standalone \
    --binary "${NETMHCIIPAN_BIN}" \
    --data-root "${DATA_DIR}" \
    --output-dir "${AUG_OUTPUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    --batch-size 30

echo "Shard ${SLURM_ARRAY_TASK_ID} complete."
rm -rf "${TMPDIR}"
