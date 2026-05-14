#!/bin/bash
#SBATCH --job-name=K3-validate
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding/K3_validate_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding/K3_validate_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design"
DPLM_ROOT="${PROJECT_ROOT}/inverse_folding/dplm"
OUTPUT_ROOT="/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding"
GENERATED_PARQUET="${GENERATED_PARQUET:-}"

# ── FILL IN: checkpoint and experiment paths ──────────────────────────────
# EXPERIMENT_PATH: the training run directory (contains .hydra/)
EXPERIMENT_PATH="${OUTPUT_ROOT}/dplm_v1_adapter/seed42_20260319_094244"
# CKPT_PATH: the best checkpoint file
CKPT_PATH="${EXPERIMENT_PATH}/checkpoints/best.ckpt"        

if [[ -n "${GENERATED_PARQUET}" ]]; then
    GENERATED_INPUT="${GENERATED_PARQUET%/}"
    if [[ -d "${GENERATED_INPUT}" ]]; then
        GENERATED_RUN_DIR="${GENERATED_INPUT}"
    else
        GENERATED_RUN_DIR="$(dirname "${GENERATED_INPUT}")"
    fi
    GENERATED_RUN_NAME="$(basename "${GENERATED_RUN_DIR}")"
    VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${OUTPUT_ROOT}/if_imp/validation/${GENERATED_RUN_NAME}}"
else
    VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${EXPERIMENT_PATH}}"
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${VALIDATION_OUTPUT_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
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
echo "  Generated in : ${GENERATED_PARQUET:-<run DPLM generation>}"
echo "  Output dir   : ${VALIDATION_OUTPUT_DIR}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

if [[ -n "${GENERATED_PARQUET}" ]]; then
    python "${PROJECT_ROOT}/scripts/validate_if_baseline.py" \
        --generated-parquet "${GENERATED_PARQUET}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${VALIDATION_OUTPUT_DIR}" \
        --temperature 1.0 \
        --num-gpus "${NUM_GPUS}"
else
    python "${PROJECT_ROOT}/scripts/validate_if_baseline.py" \
        --dplm-root "${DPLM_ROOT}" \
        --experiment-path "${EXPERIMENT_PATH}" \
        --ckpt-path "${CKPT_PATH}" \
        --experiment "dplm/cond_dplm_650m" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${VALIDATION_OUTPUT_DIR}" \
        --max-iter 10 \
        --temperature 1.0 \
        --num-gpus "${NUM_GPUS}"
fi

echo "K3 baseline validation finished."
