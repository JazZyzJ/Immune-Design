#!/bin/bash
#SBATCH --job-name=B3-hmaps-cath
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/h_maps_cath_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/h_maps_cath_%j.err

set -euo pipefail

# Paths
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
CATH_DIR="${CATH_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/cath_4.3}"
OUTPUT_DIR="${OUTPUT_DIR:-${CATH_DIR}/h_maps}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval}"

ALLELE="${ALLELE:-HLA-DRB1*07:01}"
VARIANT_ID="${VARIANT_ID:-LC1}"
DEVICE="${DEVICE:-cuda}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-4096}"
FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.02}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
SPLIT="${SPLIT:-train}"

if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_TAG="DRB1_07_01"
    EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt}"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_TAG="DRB1_04_01"
    EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt}"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01 or HLA-DRB1*04:01" >&2
    exit 2
fi

CHAIN_SET="${CHAIN_SET:-${CATH_DIR}/chain_set.jsonl}"
SPLITS_JSON="${SPLITS_JSON:-${CATH_DIR}/chain_set_splits.json}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${OUTPUT_DIR}/h_maps_cath_${ALLELE_TAG}.parquet}"
OUTPUT_META="${OUTPUT_META:-${OUTPUT_DIR}/h_maps_cath_${ALLELE_TAG}.meta.json}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-${PROJECT_ROOT}/epitope_head/configs/inference.yaml}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Environment
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}/inverse_folding/dplm/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

echo "============================================================"
echo "B3: CATH train h-map precompute"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Node            : ${SLURM_NODELIST}"
echo "  GPUs            : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root    : ${PROJECT_ROOT}"
echo "  Allele          : ${ALLELE}"
echo "  CATH JSONL      : ${CHAIN_SET}"
echo "  Splits JSON     : ${SPLITS_JSON}"
echo "  Split           : ${SPLIT}"
echo "  Epitope ckpt    : ${EPITOPE_CKPT}"
echo "  Output parquet  : ${OUTPUT_PARQUET}"
echo "  Output meta     : ${OUTPUT_META}"
echo "  Device          : ${DEVICE}"
echo "  Window batch    : ${WINDOW_BATCH_SIZE}"
echo "  Fail threshold  : ${FAIL_PCT_THRESHOLD}"
echo "  Python          : $(python --version 2>&1)"
echo "  PyTorch         : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail      : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

RESUME_FLAG=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_FLAG=(--resume-from "${RESUME_FROM}")
fi

python -u "${PROJECT_ROOT}/scripts/precompute_h_maps.py" \
    --head-checkpoint "${EPITOPE_CKPT}" \
    --inference-config "${INFERENCE_CONFIG}" \
    --config-dir "${CONFIG_DIR}" \
    --variant-id "${VARIANT_ID}" \
    --source jsonl \
    --input "${CHAIN_SET}" \
    --splits-json "${SPLITS_JSON}" \
    --split "${SPLIT}" \
    --id-field CATH \
    --sequence-field seq \
    --allele "${ALLELE}" \
    --output-parquet "${OUTPUT_PARQUET}" \
    --output-meta "${OUTPUT_META}" \
    --device "${DEVICE}" \
    --window-batch-size "${WINDOW_BATCH_SIZE}" \
    --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo "B3 h-map precompute finished."
