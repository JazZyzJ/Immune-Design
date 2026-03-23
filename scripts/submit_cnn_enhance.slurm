#!/bin/bash
#SBATCH --job-name=LC1-lite-aug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/logs/epitope_head/LC1_lite_aug_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/epitope_head/LC1_lite_aug_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
OUTPUT_ROOT="/scratch/network/zc1519/run/epitope_head/LC1_lite_aug"
LOG_DIR="/scratch/network/zc1519/logs/epitope_head"

# Optional: augmented train data from Stage J
AUG_PARQUET="/scratch/network/zc1519/work/immune-design/augmentation/protein_samples_strict_aug_train.parquet"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/network/zc1519/model_cache/torch"
export WANDB_MODE=offline

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Epitope Head: LC1_lite_aug Training"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Aug parquet  : ${AUG_PARQUET}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Build augmentation flag ──────────────────────────────────────────────
AUG_FLAG=()
if [[ -f "${AUG_PARQUET}" ]]; then
    AUG_FLAG=(--aug-train-parquet "${AUG_PARQUET}")
    echo "  Augmented data found — will train with base + augmented entries"
else
    echo "  No augmented data — training with base entries only"
fi

# ── Launch LC1_lite_aug training ─────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/train_v2_ablation.py" \
    --variant-id LC1 \
    --seed 42 \
    --device cuda \
    --profile strict \
    --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
    --override-config "${PROJECT_ROOT}/epitope_head/configs/cnn_full_aug.yaml" \
    --data-dir "${DATA_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    "${AUG_FLAG[@]+"${AUG_FLAG[@]}"}" \
    --wandb \
    --wandb-name "LC1_lite_aug_seed42" \
    --wandb-entity "chenzijie040528-zhejiang-university" \
    --wandb-project "Immune-Design"

echo "LC1_lite_aug training finished."
