#!/bin/bash
#SBATCH --job-name=B2-hmaps-test
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --account=kaiyijiang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/h_maps_test_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/h_maps_test_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

# Paths
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
DATA_DIR="${DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/if_ready}"
# fast-scoped output dir so the default never clobbers the canonical if_ready/h_maps_v2
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/h_maps_fast_v2}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval}"

ALLELE="${ALLELE:-HLA-DRB1*07:01}"
VARIANT_ID="${VARIANT_ID:-LC1}"
DEVICE="${DEVICE:-cuda}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-4096}"
FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.05}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"

if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_07_01"
    ALLELE_TAG="DRB1_07_01"
    EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_v1_npoff_drb0701_seed42/runs/LC1/seed_42/best.pt}"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_04_01"
    ALLELE_TAG="DRB1_04_01"
    EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_v1_npoff_drb0401_seed42/runs/LC1/seed_42/best.pt}"
elif [[ "${ALLELE}" == "HLA-DRB1*15:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_15_01"
    ALLELE_TAG="DRB1_15_01"
    EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_a1_res03_drb1501_seed42/runs/LC1/seed_42/best.pt}"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01, HLA-DRB1*04:01, or HLA-DRB1*15:01" >&2
    exit 2
fi
    
INPUT_PARQUET="${INPUT_PARQUET:-${DATA_DIR}/fast/fast_v2_${ALLELE_FILE_TAG}.parquet}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${OUTPUT_DIR}/h_maps_${ALLELE_TAG}.parquet}"
OUTPUT_META="${OUTPUT_META:-${OUTPUT_DIR}/h_maps_${ALLELE_TAG}.meta.json}"
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
echo "B2: Test-set h-map precompute"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Node            : ${SLURM_NODELIST}"
echo "  GPUs            : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root    : ${PROJECT_ROOT}"
echo "  Allele          : ${ALLELE}"
echo "  Input parquet   : ${INPUT_PARQUET}"
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
    --source parquet \
    --input "${INPUT_PARQUET}" \
    --id-column protein_id \
    --sequence-column sequence \
    --allele "${ALLELE}" \
    --output-parquet "${OUTPUT_PARQUET}" \
    --output-meta "${OUTPUT_META}" \
    --device "${DEVICE}" \
    --window-batch-size "${WINDOW_BATCH_SIZE}" \
    --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}"

echo "B2 h-map precompute finished."
