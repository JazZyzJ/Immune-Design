#!/bin/bash
#SBATCH --job-name=mut-aug
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/network/zc1519/run/augmentation-%j.out
#SBATCH --error=/scratch/network/zc1519/run/augmentation-%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
AUG_OUTPUT="/scratch/network/zc1519/work/immune-design/augmentation"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

mkdir -p "${AUG_OUTPUT}"
mkdir -p /scratch/network/zc1519/run

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# NetMHCIIpan uses TMPDIR for temp files
export TMPDIR="/tmp/netmhciipan_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# ── Verify NetMHCIIpan ───────────────────────────────────────────────────
if [ ! -x "${NETMHCIIPAN_BIN}" ]; then
    echo "ERROR: NetMHCIIpan not found at ${NETMHCIIPAN_BIN}"
    echo "Install NetMHCIIpan 4.3 and update NETMHCIIPAN_BIN path"
    exit 1
fi

echo "=========================================="
echo "Stage J: Mutation Augmentation Pipeline"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Data dir:     ${DATA_DIR}"
echo "Output dir:   ${AUG_OUTPUT}"
echo "NetMHCIIpan:  ${NETMHCIIPAN_BIN}"
echo "=========================================="

# ── Step 1: Build mutation registry (J2-J5) ──────────────────────────────
echo ""
echo "[Step 1/2] Building mutation registry..."
echo ""

python -m scripts.build_mutation_registry \
    --backend standalone \
    --binary "${NETMHCIIPAN_BIN}" \
    --data-root "${DATA_DIR}" \
    --output-dir "${AUG_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "ERROR: Registry build failed"
    exit 1
fi

echo ""
echo "[Step 1/2] Registry complete."
echo ""

# ── Step 2: Materialize augmented train samples (J6) ─────────────────────
echo "[Step 2/2] Materializing augmented train samples..."
echo ""

python -m scripts.materialize_augmented_train_set \
    --data-root "${DATA_DIR}" \
    --output-dir "${AUG_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "ERROR: Materialization failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline complete. Augmented train artifact:"
echo "  ${AUG_OUTPUT}/protein_samples_strict_aug_train.parquet"
echo ""
echo "To train with augmented data:"
echo "  python -m scripts.train_v2_ablation \\"
echo "      --variant-id E1 --seed 42 --device cuda \\"
echo "      --aug-train-parquet ${AUG_OUTPUT}/protein_samples_strict_aug_train.parquet"
echo "=========================================="

# Cleanup
rm -rf "${TMPDIR}"
