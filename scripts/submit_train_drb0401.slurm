#!/bin/bash
#SBATCH --job-name=LC1-drb0401
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/logs/epitope_head/LC1_drb0401_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/epitope_head/LC1_drb0401_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests/drb0401"
OUTPUT_ROOT="/scratch/network/zc1519/run/epitope_head/LC1_drb0401_aug"
LOG_DIR="/scratch/network/zc1519/logs/epitope_head"

# Stage J: runtime augmentation via registry
REGISTRY="/scratch/network/zc1519/work/immune-design/augmentation/drb0401/mutation_registry_strict.parquet"
P_AUG=0.20

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
echo "Epitope Head: LC1 DRB1*04:01 Training (runtime p_aug)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Registry     : ${REGISTRY}"
echo "  p_aug        : ${P_AUG}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Build augmentation flag ──────────────────────────────────────────────
AUG_FLAG=()
if [[ -f "${REGISTRY}" ]]; then
    AUG_FLAG=(--registry-path "${REGISTRY}" --p-aug "${P_AUG}")
    echo "  Registry found — runtime augmentation enabled (p_aug=${P_AUG})"
else
    echo "  No registry found — training with base entries only"
fi

# ── Launch training ──────────────────────────────────────────────────────
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
    --wandb-name "LC1_drb0401_aug_p${P_AUG}_seed42" \
    --wandb-entity "chenzijie040528-zhejiang-university" \
    --wandb-project "Immune-Design"

echo "LC1 DRB1*04:01 training finished."
