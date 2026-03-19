#!/bin/bash
#SBATCH --job-name=if-v1-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/network/zc1519/logs/inverse_folding/if-train-%j.out
#SBATCH --error=/scratch/network/zc1519/logs/inverse_folding/if-train-%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DPLM_ROOT="${PROJECT_ROOT}/inverse_folding/dplm"
DATA_DIR="/scratch/network/zc1519/work/immune-design"
OUTPUT_ROOT="/scratch/network/zc1519/run/inverse_folding"
LOG_DIR="/scratch/network/zc1519/logs/inverse_folding"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/network/zc1519/model_cache/torch"
export HF_HOME="/scratch/network/zc1519/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Inverse Folding v1: DPLM Adapter Training"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-4}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  DPLM root    : ${DPLM_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch Training ───────────────────────────────────────────────────────
python -m scripts.train_if_v1 \
    --data-dir "${DATA_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --dplm-root "${DPLM_ROOT}" \
    --num-gpus 2 \
    --seed 42

echo "Training job finished."
