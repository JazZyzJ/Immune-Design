#!/bin/bash
#SBATCH --job-name=K3-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/network/zc1519/logs/inverse_folding/K3_validate_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/inverse_folding/K3_validate_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work"
DPLM_ROOT="${PROJECT_ROOT}/inverse_folding/dplm"
OUTPUT_ROOT="/scratch/network/zc1519/run/inverse_folding"
LOG_DIR="/scratch/network/zc1519/logs/inverse_folding"

# ── FILL IN: checkpoint and experiment paths ──────────────────────────────
# EXPERIMENT_PATH: the training run directory (contains .hydra/)
EXPERIMENT_PATH="${OUTPUT_ROOT}/dplm_v1_adapter/<FILL_IN_RUN_ID>"
# CKPT_PATH: the best checkpoint file
CKPT_PATH="${EXPERIMENT_PATH}/checkpoints/<FILL_IN_BEST_CKPT>"

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
echo "K3: IF Baseline Validation (CATH test split)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  DPLM root    : ${DPLM_ROOT}"
echo "  Experiment   : ${EXPERIMENT_PATH}"
echo "  Checkpoint   : ${CKPT_PATH}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/validate_if_baseline.py" \
    --dplm-root "${DPLM_ROOT}" \
    --experiment-path "${EXPERIMENT_PATH}" \
    --ckpt-path "${CKPT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${EXPERIMENT_PATH}" \
    --max-iter 10 \
    --temperature 1.0 \
    --num-gpus 1

echo "K3 baseline validation finished."
