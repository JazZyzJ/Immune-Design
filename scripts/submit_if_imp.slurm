#!/bin/bash
#SBATCH --job-name=if-imp
#SBATCH --account=kaiyijiang
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

# ── Mode selection ────────────────────────────────────────────────────────
MODE="${MODE:-generate_refiner}"   # train_refiner | train_sidecar | generate_refiner | compare_ablation
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

# ── Paths (canonical Della layout from PROGRESS.md) ───────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
WORK_DIR="${WORK_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp}"
DPLM_ROOT="${DPLM_ROOT:-${PROJECT_ROOT}/inverse_folding/dplm}"
export PROJECT_ROOT WORK_DIR RUN_DIR LOG_DIR DPLM_ROOT

# Frozen Module K checkpoint (PROGRESS.md §IF-K)
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt}"

# Allele-dependent test-set + PDB root (PROGRESS.md §IF Test Set)
if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_07_01"
    ALLELE_SHORT="0701"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_04_01"
    ALLELE_SHORT="0401"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01 or HLA-DRB1*04:01" >&2
    exit 2
fi
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${WORK_DIR}/if_test_set/if_ready/fast/test_proteins_if_ready_fast_${ALLELE_FILE_TAG}.parquet}"
PDB_ROOT="${PDB_ROOT:-${WORK_DIR}/if_test_set/pdbs_if_ready/${ALLELE_SHORT}}"

# CATH 4.3 split (PROGRESS.md §Cluster — `work/immune-design/cath_4.3/`)
CATH_ROOT="${CATH_ROOT:-${WORK_DIR}/cath_4.3}"

# Refiner / sidecar artifacts. Set these as env when running an arm that needs them.
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
SIDECAR_CHECKPOINT="${SIDECAR_CHECKPOINT:-}"

# Mode-specific output roots
case "${MODE}" in
    train_refiner)    OUTPUT_ROOT_DEFAULT="${RUN_DIR}/if_imp/refiner_train" ;;
    train_sidecar)    OUTPUT_ROOT_DEFAULT="${RUN_DIR}/if_imp/sidecar_train" ;;
    generate_refiner) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/${ARM:-baseline}" ;;
    compare_ablation) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/compare" ;;
    *) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/${MODE}" ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_DEFAULT}}"

# Refiner training overrides
SPLIT="${SPLIT:-train}"
MAX_LENGTH="${MAX_LENGTH:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT_BATCHES="${LIMIT_BATCHES:-}"
USE_DPLM_ENTROPY="${USE_DPLM_ENTROPY:-0}"

# Sidecar training overrides (defaults trimmed for IPA + 512-dim projection)
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-2}"
SIDECAR_EPOCHS="${SIDECAR_EPOCHS:-5}"
SIDECAR_LR="${SIDECAR_LR:-1e-4}"
SIDECAR_HIDDEN_DIM="${SIDECAR_HIDDEN_DIM:-128}"
SIDECAR_IPA_DEPTH="${SIDECAR_IPA_DEPTH:-6}"
SIDECAR_OUTPUT_DIM="${SIDECAR_OUTPUT_DIM:-512}"

# Refiner generation overrides
ARM="${ARM:-}"
ABLATION_MODE="${ABLATION_MODE:-1}"   # default ON for the if_imp launcher
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}" # K3-style batched DPLM generate; >1 enables length-bucketed batching
N_DESIGNS_PER_PROTEIN="${N_DESIGNS_PER_PROTEIN:-1}"
MAX_ITER="${MAX_ITER:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MC_DROPOUT_PASSES="${MC_DROPOUT_PASSES:-1}"
MASK_RATIO_CENTER="${MASK_RATIO_CENTER:-0.4}"
MASK_RATIO_DEVIATION="${MASK_RATIO_DEVIATION:-0.2}"
FUSION_TEMPERATURE="${FUSION_TEMPERATURE:-1.0}"
LIMIT_PROTEINS="${LIMIT_PROTEINS:-}"

# Compare-mode arm dirs (callers point these at completed run dirs)
BASELINE_RUN="${BASELINE_RUN:-}"
REFINER_RUN="${REFINER_RUN:-}"
SIDECAR_RUN="${SIDECAR_RUN:-}"
COMBINED_RUN="${COMBINED_RUN:-}"
ALLOW_UNPAIRED="${ALLOW_UNPAIRED:-0}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design
module load proxy/default

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold:${PROJECT_ROOT}/inverse_folding/dplm/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# wandb storage roots — keep all run state on /scratch.
export WANDB_DIR="${WANDB_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CONFIG_DIR}"
WANDB_FLAG="${WANDB:-1}"
WANDB_MODE_FOR_RUN="${WANDB_MODE:-online}"
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-mhc-if-imp-refiner}"
WANDB_GROUP_FOR_RUN="${WANDB_GROUP:-}"
WANDB_TAGS_FOR_RUN="${WANDB_TAGS:-}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "IF Improvement: ${MODE}"
echo "  Job ID            : ${SLURM_JOB_ID:-local}"
echo "  Node              : ${SLURM_NODELIST:-N/A}"
echo "  GPUs              : ${SLURM_GPUS_ON_NODE:-1}"
echo "  PROJECT_ROOT      : ${PROJECT_ROOT}"
echo "  CHECKPOINT        : ${CHECKPOINT}"
echo "  REFINER_CHECKPOINT: ${REFINER_CHECKPOINT:-<none>}"
echo "  SIDECAR_CHECKPOINT: ${SIDECAR_CHECKPOINT:-<none>}"
echo "  CATH_ROOT         : ${CATH_ROOT}"
echo "  TEST_SET_PARQUET  : ${TEST_SET_PARQUET}"
echo "  PDB_ROOT          : ${PDB_ROOT}"
echo "  OUTPUT_ROOT       : ${OUTPUT_ROOT}"
echo "  ALLELE            : ${ALLELE}"
echo "  ARM               : ${ARM:-<auto>}"
echo "  ABLATION_MODE     : ${ABLATION_MODE}"
echo "  DEVICE            : ${DEVICE}"
echo "  SEED              : ${SEED}"
echo "  Python            : $(python --version 2>&1)"
echo "  PyTorch           : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail        : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name          : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  wandb             : ${WANDB_FLAG} (project=${WANDB_PROJECT_FOR_RUN} mode=${WANDB_MODE_FOR_RUN})"
echo "============================================================"

WANDB_ARGS=()
if [[ "${WANDB_FLAG}" == "1" ]]; then
    WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT_FOR_RUN}" --wandb-mode "${WANDB_MODE_FOR_RUN}")
    if [[ -n "${WANDB_GROUP_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-group "${WANDB_GROUP_FOR_RUN}")
    fi
    if [[ -n "${WANDB_TAGS_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-tags "${WANDB_TAGS_FOR_RUN}")
    fi
fi

if [ "${MODE}" = "train_refiner" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for train_refiner}"
    : "${CATH_ROOT:?set CATH_ROOT for train_refiner}"
    EXTRA_FLAGS=()
    [[ -n "${LIMIT_BATCHES}" ]] && EXTRA_FLAGS+=(--limit-batches "${LIMIT_BATCHES}")
    [[ "${USE_DPLM_ENTROPY}" = "1" ]] && EXTRA_FLAGS+=(--use-dplm-entropy)
    python -u "${PROJECT_ROOT}/scripts/train_if_imp_refiner.py" \
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
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "train_sidecar" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for train_sidecar}"
    : "${CATH_ROOT:?set CATH_ROOT for train_sidecar}"
    EXTRA_FLAGS=()
    [[ -n "${LIMIT_BATCHES}" ]] && EXTRA_FLAGS+=(--limit-batches "${LIMIT_BATCHES}")
    python -u "${PROJECT_ROOT}/scripts/train_if_imp_sidecar.py" \
        --cath-root "${CATH_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --split "${SPLIT}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${SIDECAR_BATCH_SIZE}" \
        --epochs "${SIDECAR_EPOCHS}" \
        --lr "${SIDECAR_LR}" \
        --hidden-dim "${SIDECAR_HIDDEN_DIM}" \
        --ipa-depth "${SIDECAR_IPA_DEPTH}" \
        --output-dim "${SIDECAR_OUTPUT_DIM}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --num-workers "${NUM_WORKERS}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "generate_refiner" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for generate_refiner}"
    : "${TEST_SET_PARQUET:?set TEST_SET_PARQUET for generate_refiner}"
    : "${PDB_ROOT:?set PDB_ROOT for generate_refiner}"
    EXTRA_FLAGS=()
    [[ -n "${REFINER_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    [[ -n "${SIDECAR_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    [[ -n "${ARM}" ]]                && EXTRA_FLAGS+=(--arm "${ARM}")
    [[ "${ABLATION_MODE}" = "1" ]]   && EXTRA_FLAGS+=(--ablation-mode)
    [[ -n "${LIMIT_PROTEINS}" ]]     && EXTRA_FLAGS+=(--limit-proteins "${LIMIT_PROTEINS}")
    python -u "${PROJECT_ROOT}/scripts/run_if_imp_refiner.py" \
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
        --batch-size "${GEN_BATCH_SIZE}" \
        --mc-dropout-passes "${MC_DROPOUT_PASSES}" \
        --mask-ratio-center "${MASK_RATIO_CENTER}" \
        --mask-ratio-deviation "${MASK_RATIO_DEVIATION}" \
        --fusion-temperature "${FUSION_TEMPERATURE}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "compare_ablation" ]; then
    EXTRA_FLAGS=()
    [[ -n "${BASELINE_RUN}" ]] && EXTRA_FLAGS+=(--baseline-run "${BASELINE_RUN}")
    [[ -n "${REFINER_RUN}"  ]] && EXTRA_FLAGS+=(--refiner-run  "${REFINER_RUN}")
    [[ -n "${SIDECAR_RUN}"  ]] && EXTRA_FLAGS+=(--sidecar-run  "${SIDECAR_RUN}")
    [[ -n "${COMBINED_RUN}" ]] && EXTRA_FLAGS+=(--combined-run "${COMBINED_RUN}")
    [[ "${ALLOW_UNPAIRED}" = "1" ]] && EXTRA_FLAGS+=(--allow-unpaired)
    [[ -n "${OVERWRITE}" ]] && EXTRA_FLAGS+=(--overwrite)
    python -u "${PROJECT_ROOT}/scripts/compare_if_imp_ablation.py" \
        --output-dir "${OUTPUT_ROOT}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
else
    echo "ERROR: unknown MODE='${MODE}' (expected train_refiner|train_sidecar|generate_refiner|compare_ablation)" >&2
    exit 2
fi

echo "if-imp ${MODE} finished."
