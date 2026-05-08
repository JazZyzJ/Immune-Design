#!/bin/bash
#SBATCH --job-name=if-imp-refiner
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.err

set -euo pipefail

# ── Mode selection ────────────────────────────────────────────────────────
MODE="${MODE:-generate_refiner}"   # train_refiner | train_sidecar | generate_refiner | compare_ablation

# ── Paths (env defaults; CLI flags receive these) ─────────────────────────
# CHECKPOINT is mode-dependent: train_refiner / train_sidecar / generate_refiner
# all need it; compare_ablation does not. Each mode branch below validates
# the checkpoint when required so a compare-only job does not need a
# dummy --checkpoint.
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
CHECKPOINT="${CHECKPOINT:-}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
CATH_ROOT="${CATH_ROOT:-}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-}"
PDB_ROOT="${PDB_ROOT:-}"
# OUTPUT_ROOT is required for every mode (train run dir, generate run dir,
# or compare output dir).
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp}"

# Refiner training overrides
SPLIT="${SPLIT:-train}"
MAX_LENGTH="${MAX_LENGTH:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT_BATCHES="${LIMIT_BATCHES:-}"
USE_DPLM_ENTROPY="${USE_DPLM_ENTROPY:-0}"

# Refiner generation overrides
N_DESIGNS_PER_PROTEIN="${N_DESIGNS_PER_PROTEIN:-1}"
MAX_ITER="${MAX_ITER:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MC_DROPOUT_PASSES="${MC_DROPOUT_PASSES:-1}"
MASK_RATIO_CENTER="${MASK_RATIO_CENTER:-0.4}"
MASK_RATIO_DEVIATION="${MASK_RATIO_DEVIATION:-0.2}"
FUSION_TEMPERATURE="${FUSION_TEMPERATURE:-1.0}"
LIMIT_PROTEINS="${LIMIT_PROTEINS:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/inverse_folding/dplm/src:${PYTHONPATH:-}"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "IF Improvement: ${MODE}"
echo "  Job ID            : ${SLURM_JOB_ID:-local}"
echo "  Node              : ${SLURM_NODELIST:-N/A}"
echo "  GPUs              : ${SLURM_GPUS_ON_NODE:-1}"
echo "  PROJECT_ROOT      : ${PROJECT_ROOT}"
echo "  CHECKPOINT        : ${CHECKPOINT}"
echo "  REFINER_CHECKPOINT: ${REFINER_CHECKPOINT}"
echo "  CATH_ROOT         : ${CATH_ROOT}"
echo "  TEST_SET_PARQUET  : ${TEST_SET_PARQUET}"
echo "  PDB_ROOT          : ${PDB_ROOT}"
echo "  OUTPUT_ROOT       : ${OUTPUT_ROOT}"
echo "  ALLELE            : ${ALLELE}"
echo "  DEVICE            : ${DEVICE}"
echo "  SEED              : ${SEED}"
echo "  Timestamp         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

if [ "${MODE}" = "train_refiner" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for train_refiner}"
    : "${CATH_ROOT:?set CATH_ROOT for train_refiner}"
    EXTRA_FLAGS=()
    if [ -n "${LIMIT_BATCHES}" ]; then
        EXTRA_FLAGS+=(--limit-batches "${LIMIT_BATCHES}")
    fi
    if [ "${USE_DPLM_ENTROPY}" = "1" ]; then
        EXTRA_FLAGS+=(--use-dplm-entropy)
    fi
    python "${PROJECT_ROOT}/scripts/train_if_imp_refiner.py" \
        --cath-root "${CATH_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --split "${SPLIT}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --mask-ratio-center "${MASK_RATIO_CENTER}" \
        --mask-ratio-deviation "${MASK_RATIO_DEVIATION}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --num-workers "${NUM_WORKERS}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
elif [ "${MODE}" = "train_sidecar" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for train_sidecar}"
    : "${CATH_ROOT:?set CATH_ROOT for train_sidecar}"
    EXTRA_FLAGS=()
    if [ -n "${LIMIT_BATCHES}" ]; then
        EXTRA_FLAGS+=(--limit-batches "${LIMIT_BATCHES}")
    fi
    python "${PROJECT_ROOT}/scripts/train_if_imp_sidecar.py" \
        --cath-root "${CATH_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --split "${SPLIT}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --num-workers "${NUM_WORKERS}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
elif [ "${MODE}" = "generate_refiner" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for generate_refiner}"
    : "${TEST_SET_PARQUET:?set TEST_SET_PARQUET for generate_refiner}"
    : "${PDB_ROOT:?set PDB_ROOT for generate_refiner}"
    EXTRA_FLAGS=()
    if [ -n "${REFINER_CHECKPOINT}" ]; then
        EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    fi
    if [ -n "${SIDECAR_CHECKPOINT:-}" ]; then
        EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    fi
    if [ -n "${ARM:-}" ]; then
        EXTRA_FLAGS+=(--arm "${ARM}")
    fi
    if [ "${ABLATION_MODE:-0}" = "1" ]; then
        EXTRA_FLAGS+=(--ablation-mode)
    fi
    if [ -n "${LIMIT_PROTEINS}" ]; then
        EXTRA_FLAGS+=(--limit-proteins "${LIMIT_PROTEINS}")
    fi
    python "${PROJECT_ROOT}/scripts/run_if_imp_refiner.py" \
        --checkpoint "${CHECKPOINT}" \
        --test-set-parquet "${TEST_SET_PARQUET}" \
        --pdb-root "${PDB_ROOT}" \
        --allele "${ALLELE}" \
        --output-root "${OUTPUT_ROOT}" \
        --n-designs-per-protein "${N_DESIGNS_PER_PROTEIN}" \
        --seed "${SEED}" \
        --max-iter "${MAX_ITER}" \
        --temperature "${TEMPERATURE}" \
        --device "${DEVICE}" \
        --mc-dropout-passes "${MC_DROPOUT_PASSES}" \
        --mask-ratio-center "${MASK_RATIO_CENTER}" \
        --mask-ratio-deviation "${MASK_RATIO_DEVIATION}" \
        --fusion-temperature "${FUSION_TEMPERATURE}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
elif [ "${MODE}" = "compare_ablation" ]; then
    : "${OUTPUT_ROOT:?set OUTPUT_ROOT for compare_ablation}"
    EXTRA_FLAGS=()
    if [ -n "${BASELINE_RUN:-}" ]; then EXTRA_FLAGS+=(--baseline-run "${BASELINE_RUN}"); fi
    if [ -n "${REFINER_RUN:-}" ];  then EXTRA_FLAGS+=(--refiner-run  "${REFINER_RUN}");  fi
    if [ -n "${SIDECAR_RUN:-}" ];  then EXTRA_FLAGS+=(--sidecar-run  "${SIDECAR_RUN}");  fi
    if [ -n "${COMBINED_RUN:-}" ]; then EXTRA_FLAGS+=(--combined-run "${COMBINED_RUN}"); fi
    python "${PROJECT_ROOT}/scripts/compare_if_imp_ablation.py" \
        --output-dir "${OUTPUT_ROOT}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        ${OVERWRITE:+--overwrite}
else
    echo "ERROR: unknown MODE='${MODE}' (expected train_refiner|train_sidecar|generate_refiner|compare_ablation)" >&2
    exit 2
fi

echo "if-imp ${MODE} finished."
