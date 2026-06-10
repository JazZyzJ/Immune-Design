#!/bin/bash
#SBATCH --job-name=if-imp
#SBATCH --account=kaiyijiang
#SBATCH --constraint="h200"
#SBATCH --partition=ailab
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_imp/if_imp_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

# ── Mode selection ────────────────────────────────────────────────────────
MODE="${MODE:-generate_refiner}"   # train_refiner | train_sidecar | generate_refiner | compare_ablation | diag_arms | train_encoder | generate_encoder | diag_encoder
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
INPUT_SOURCE="${INPUT_SOURCE:-if_test}" # if_test | cath

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
#REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/if_imp/refiner_train/train_if_imp_refiner_20260512T145317Z/refiner_last.pt}"
#SIDECAR_CHECKPOINT="${SIDECAR_CHECKPOINT:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/if_imp/sidecar_train/train_if_imp_sidecar_20260512T170058Z/sidecar_last.pt}"

# Mode-specific output roots
case "${MODE}" in
    train_refiner)    OUTPUT_ROOT_DEFAULT="${RUN_DIR}/if_imp/refiner_train" ;;
    train_sidecar)    OUTPUT_ROOT_DEFAULT="${RUN_DIR}/if_imp/sidecar_train" ;;
    train_encoder)    OUTPUT_ROOT_DEFAULT="${RUN_DIR}/if_imp/encoder_train" ;;
    generate_refiner) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/${INPUT_SOURCE}/${ARM:-baseline}" ;;
    generate_encoder) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/${INPUT_SOURCE}/encoder" ;;
    compare_ablation) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/compare" ;;
    diag_arms)        OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/diag" ;;
    diag_encoder)     OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/diag_encoder" ;;
    *) OUTPUT_ROOT_DEFAULT="${WORK_DIR}/if_imp/${MODE}" ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_DEFAULT}}"

# Encoder-mode overrides (MODE=train_encoder | generate_encoder | diag_encoder)
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-}"
ENCODER_KIND="${ENCODER_KIND:-geoegnn_ipa}"          # geoegnn_ipa | gvp (default for legacy)
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-2}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-10}"
ENCODER_LR="${ENCODER_LR:-1e-4}"
ENCODER_D_MODEL="${ENCODER_D_MODEL:-512}"
ENCODER_EGNN_DEPTH="${ENCODER_EGNN_DEPTH:-6}"
ENCODER_EGNN_HIDDEN_DIM="${ENCODER_EGNN_HIDDEN_DIM:-128}"
ENCODER_IPA_DEPTH="${ENCODER_IPA_DEPTH:-6}"
ENCODER_IPA_HIDDEN_DIM="${ENCODER_IPA_HIDDEN_DIM:-128}"
ENCODER_LAMBDA_AUX="${ENCODER_LAMBDA_AUX:-1.0}"      # PLAN_IF_ENCODER.md E7: bumped from 0.15
ENCODER_ADAPTER_NUM_LAYERS="${ENCODER_ADAPTER_NUM_LAYERS:-1}"
ENCODER_ADAPTER_GATED="${ENCODER_ADAPTER_GATED:-0}"
ENCODER_ADAPTER_GATE_INIT="${ENCODER_ADAPTER_GATE_INIT:-0.0}"
ENCODER_UPDATE_COORS="${ENCODER_UPDATE_COORS:-0}"     # 1 enables --update-coors
ENCODER_USE_UPDATED_BIAS="${ENCODER_USE_UPDATED_BIAS:-0}"
# Task E7 — validation + optional aux-only pre-stage.
ENCODER_VAL_BATCH_SIZE="${ENCODER_VAL_BATCH_SIZE:-8}"
ENCODER_VAL_EVAL_EVERY_EPOCHS="${ENCODER_VAL_EVAL_EVERY_EPOCHS:-1}"
ENCODER_PRETRAIN_AUX_ONLY_EPOCHS="${ENCODER_PRETRAIN_AUX_ONLY_EPOCHS:-0}"
ENCODER_PRETRAIN_TARGET="${ENCODER_PRETRAIN_TARGET:-0.30}"
# Resume + multi-iter val sweep — train_if_imp_encoder.py follow-ups.
ENCODER_RESUME_FROM_CKPT="${ENCODER_RESUME_FROM_CKPT:-}"
ENCODER_VAL_MAX_ITERS="${ENCODER_VAL_MAX_ITERS:-1 2 100}"
ENCODER_TRAIN_RECOVERY_ON_VAL="${ENCODER_TRAIN_RECOVERY_ON_VAL:-0}"

# Refiner training overrides
SPLIT="${SPLIT:-train}"
MAX_LENGTH="${MAX_LENGTH:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT_BATCHES="${LIMIT_BATCHES:-}"
USE_DPLM_ENTROPY="${USE_DPLM_ENTROPY:-1}"

# Sidecar training overrides (defaults trimmed for IPA + 512-dim projection)
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-2}"
SIDECAR_EPOCHS="${SIDECAR_EPOCHS:-5}"
SIDECAR_LR="${SIDECAR_LR:-1e-4}"
SIDECAR_HIDDEN_DIM="${SIDECAR_HIDDEN_DIM:-128}"
SIDECAR_IPA_DEPTH="${SIDECAR_IPA_DEPTH:-6}"
SIDECAR_OUTPUT_DIM="${SIDECAR_OUTPUT_DIM:-512}"
# Generation-time sidecar residual strength: feats + SIDECAR_SCALE * sidecar_feats
SIDECAR_SCALE="${SIDECAR_SCALE:-0.5}"

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
APPLY_STEPS="${APPLY_STEPS:-all}"               # all | last_2 | final_only
FUSION_MODE="${FUSION_MODE:-entropy}"           # entropy | residual
FUSION_ALPHA="${FUSION_ALPHA:-0.25}"             # only used when FUSION_MODE=residual
LIMIT_PROTEINS="${LIMIT_PROTEINS:-}"
CATH_SPLIT="${CATH_SPLIT:-test}"
CATH_MAX_LENGTH="${CATH_MAX_LENGTH:-500}"
CATH_MIN_RESOLVED_RATIO="${CATH_MIN_RESOLVED_RATIO:-0.90}"
TAG="${TAG:-}"
# Compare-mode arm dirs (callers point these at completed run dirs)
BASELINE_RUN="${BASELINE_RUN:-}"
REFINER_RUN="${REFINER_RUN:-}"
SIDECAR_RUN="${SIDECAR_RUN:-}"
COMBINED_RUN="${COMBINED_RUN:-}"
ALLOW_UNPAIRED="${ALLOW_UNPAIRED:-0}"
OVERWRITE="${OVERWRITE:-}"

# Diagnostic-arms mode overrides (MODE=diag_arms)
DIAG_N_PROTEINS="${DIAG_N_PROTEINS:-200}"
DIAG_ARMS="${DIAG_ARMS:-baseline sidecar refiner sidecar_refiner}"
DIAG_EMIT_PER_POSITION="${DIAG_EMIT_PER_POSITION:-0}"
DIAG_RUN_ID="${DIAG_RUN_ID:-}"
DIAG_PROGRESS_EVERY="${DIAG_PROGRESS_EVERY:-10}"
# Empty = respect checkpoint config; set to "true"/"false" to override generator.use_draft_seq.
USE_DRAFT_SEQ="${USE_DRAFT_SEQ:-}"

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
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-mhc-if-imp-encoder}"
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
echo "  SIDECAR_SCALE     : ${SIDECAR_SCALE}"
echo "  CATH_ROOT         : ${CATH_ROOT}"
echo "  TEST_SET_PARQUET  : ${TEST_SET_PARQUET}"
echo "  PDB_ROOT          : ${PDB_ROOT}"
echo "  INPUT_SOURCE      : ${INPUT_SOURCE}"
echo "  CATH_SPLIT        : ${CATH_SPLIT}"
echo "  CATH_MAX_LENGTH   : ${CATH_MAX_LENGTH}"
echo "  CATH_MIN_RES_RATIO: ${CATH_MIN_RESOLVED_RATIO}"
echo "  OUTPUT_ROOT       : ${OUTPUT_ROOT}"
echo "  ALLELE            : ${ALLELE}"
echo "  ARM               : ${ARM:-<auto>}"
echo "  ABLATION_MODE     : ${ABLATION_MODE}"
echo "  ENCODER_ADAPTER_N : ${ENCODER_ADAPTER_NUM_LAYERS}"
echo "  ENCODER_ADAPTER_GT: ${ENCODER_ADAPTER_GATED} init=${ENCODER_ADAPTER_GATE_INIT}"
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
    EXTRA_FLAGS=()
    EXTRA_FLAGS+=(--input-source "${INPUT_SOURCE}")
    if [ "${INPUT_SOURCE}" = "if_test" ]; then
        : "${TEST_SET_PARQUET:?set TEST_SET_PARQUET for INPUT_SOURCE=if_test}"
        : "${PDB_ROOT:?set PDB_ROOT for INPUT_SOURCE=if_test}"
        EXTRA_FLAGS+=(--test-set-parquet "${TEST_SET_PARQUET}")
        EXTRA_FLAGS+=(--pdb-root "${PDB_ROOT}")
    elif [ "${INPUT_SOURCE}" = "cath" ]; then
        : "${CATH_ROOT:?set CATH_ROOT for INPUT_SOURCE=cath}"
        EXTRA_FLAGS+=(--cath-root "${CATH_ROOT}")
        EXTRA_FLAGS+=(--cath-split "${CATH_SPLIT}")
        EXTRA_FLAGS+=(--cath-max-length "${CATH_MAX_LENGTH}")
        EXTRA_FLAGS+=(--cath-min-resolved-ratio "${CATH_MIN_RESOLVED_RATIO}")
    else
        echo "ERROR: INPUT_SOURCE='${INPUT_SOURCE}' (expected if_test|cath)" >&2
        exit 2
    fi
    [[ -n "${REFINER_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    [[ -n "${SIDECAR_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    EXTRA_FLAGS+=(--sidecar-scale "${SIDECAR_SCALE}")
    [[ -n "${ARM}" ]]                && EXTRA_FLAGS+=(--arm "${ARM}")
    [[ "${ABLATION_MODE}" = "1" ]]   && EXTRA_FLAGS+=(--ablation-mode)
    [[ -n "${LIMIT_PROTEINS}" ]]     && EXTRA_FLAGS+=(--limit-proteins "${LIMIT_PROTEINS}")
    python -u "${PROJECT_ROOT}/scripts/run_if_imp_refiner.py" \
        --checkpoint "${CHECKPOINT}" \
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
        --apply-steps "${APPLY_STEPS}" \
        --fusion-mode "${FUSION_MODE}" \
        --fusion-alpha "${FUSION_ALPHA}" \
        --tag "${TAG}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "diag_arms" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for diag_arms}"
    : "${CATH_ROOT:?set CATH_ROOT for diag_arms}"
    EXTRA_FLAGS=()
    [[ -n "${REFINER_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    [[ -n "${SIDECAR_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    [[ -n "${DIAG_RUN_ID}" ]]        && EXTRA_FLAGS+=(--run-id "${DIAG_RUN_ID}")
    [[ -n "${OVERWRITE}" ]]          && EXTRA_FLAGS+=(--overwrite)
    [[ "${DIAG_EMIT_PER_POSITION}" = "1" ]] && EXTRA_FLAGS+=(--emit-per-position)
    [[ -n "${USE_DRAFT_SEQ}" ]]      && EXTRA_FLAGS+=(--use-draft-seq "${USE_DRAFT_SEQ}")
    # shellcheck disable=SC2086  # we WANT DIAG_ARMS to word-split
    python -u "${PROJECT_ROOT}/scripts/diag_if_imp_arms.py" \
        --checkpoint "${CHECKPOINT}" \
        --cath-root "${CATH_ROOT}" \
        --cath-split "${CATH_SPLIT}" \
        --cath-max-length "${CATH_MAX_LENGTH}" \
        --cath-min-resolved-ratio "${CATH_MIN_RESOLVED_RATIO}" \
        --n-proteins "${DIAG_N_PROTEINS}" \
        --seed "${SEED}" \
        --max-iter "${MAX_ITER}" \
        --temperature "${TEMPERATURE}" \
        --mc-dropout-passes "${MC_DROPOUT_PASSES}" \
        --mask-ratio-center "${MASK_RATIO_CENTER}" \
        --mask-ratio-deviation "${MASK_RATIO_DEVIATION}" \
        --fusion-temperature "${FUSION_TEMPERATURE}" \
        --apply-steps "${APPLY_STEPS}" \
        --fusion-mode "${FUSION_MODE}" \
        --fusion-alpha "${FUSION_ALPHA}" \
        --device "${DEVICE}" \
        --arms ${DIAG_ARMS} \
        --progress-every "${DIAG_PROGRESS_EVERY}" \
        --output-dir "${OUTPUT_ROOT}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
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
elif [ "${MODE}" = "train_encoder" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for train_encoder}"
    : "${CATH_ROOT:?set CATH_ROOT for train_encoder}"
    EXTRA_FLAGS=()
    [[ -n "${LIMIT_BATCHES}" ]] && EXTRA_FLAGS+=(--limit-batches "${LIMIT_BATCHES}")
    [[ "${ENCODER_ADAPTER_GATED}" = "1" ]] && EXTRA_FLAGS+=(--adapter-gated)
    [[ "${ENCODER_UPDATE_COORS}" = "1" ]] && EXTRA_FLAGS+=(--update-coors)
    [[ "${ENCODER_USE_UPDATED_BIAS}" = "1" ]] && EXTRA_FLAGS+=(--use-updated-coord-bias)
    [[ -n "${ENCODER_RESUME_FROM_CKPT}" ]] && EXTRA_FLAGS+=(--resume-from-ckpt "${ENCODER_RESUME_FROM_CKPT}")
    [[ "${ENCODER_TRAIN_RECOVERY_ON_VAL}" = "1" ]] && EXTRA_FLAGS+=(--train-recovery-on-val)
    python -u "${PROJECT_ROOT}/scripts/train_if_imp_encoder.py" \
        --cath-root "${CATH_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --split "${SPLIT}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${ENCODER_BATCH_SIZE}" \
        --epochs "${ENCODER_EPOCHS}" \
        --lr "${ENCODER_LR}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --num-workers "${NUM_WORKERS}" \
        --lambda-aux "${ENCODER_LAMBDA_AUX}" \
        --adapter-num-layers "${ENCODER_ADAPTER_NUM_LAYERS}" \
        --adapter-gate-init "${ENCODER_ADAPTER_GATE_INIT}" \
        --d-model "${ENCODER_D_MODEL}" \
        --egnn-depth "${ENCODER_EGNN_DEPTH}" \
        --egnn-hidden-dim "${ENCODER_EGNN_HIDDEN_DIM}" \
        --ipa-depth "${ENCODER_IPA_DEPTH}" \
        --ipa-hidden-dim "${ENCODER_IPA_HIDDEN_DIM}" \
        --val-batch-size "${ENCODER_VAL_BATCH_SIZE}" \
        --val-eval-every-epochs "${ENCODER_VAL_EVAL_EVERY_EPOCHS}" \
        --val-max-iters ${ENCODER_VAL_MAX_ITERS} \
        --pretrain-aux-only-epochs "${ENCODER_PRETRAIN_AUX_ONLY_EPOCHS}" \
        --pretrain-target "${ENCODER_PRETRAIN_TARGET}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "generate_encoder" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for generate_encoder}"
    : "${ENCODER_CHECKPOINT:?set ENCODER_CHECKPOINT for generate_encoder}"
    # Mirror the generate_refiner branch's full parameter passthrough,
    # adding only the GeoEGNN-IPA encoder swap flags. Keeping these in
    # sync ensures generation config (temperature / batch-size / n-designs
    # / fusion / cath filters) isn't silently dropped.
    EXTRA_FLAGS=()
    EXTRA_FLAGS+=(--input-source "${INPUT_SOURCE}")
    if [ "${INPUT_SOURCE}" = "if_test" ]; then
        : "${TEST_SET_PARQUET:?set TEST_SET_PARQUET for INPUT_SOURCE=if_test}"
        : "${PDB_ROOT:?set PDB_ROOT for INPUT_SOURCE=if_test}"
        EXTRA_FLAGS+=(--test-set-parquet "${TEST_SET_PARQUET}")
        EXTRA_FLAGS+=(--pdb-root "${PDB_ROOT}")
    elif [ "${INPUT_SOURCE}" = "cath" ]; then
        : "${CATH_ROOT:?set CATH_ROOT for INPUT_SOURCE=cath}"
        EXTRA_FLAGS+=(--cath-root "${CATH_ROOT}")
        EXTRA_FLAGS+=(--cath-split "${CATH_SPLIT}")
        EXTRA_FLAGS+=(--cath-max-length "${CATH_MAX_LENGTH}")
        EXTRA_FLAGS+=(--cath-min-resolved-ratio "${CATH_MIN_RESOLVED_RATIO}")
    else
        echo "ERROR: INPUT_SOURCE='${INPUT_SOURCE}' (expected if_test|cath)" >&2
        exit 2
    fi
    [[ -n "${REFINER_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    [[ -n "${SIDECAR_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    EXTRA_FLAGS+=(--sidecar-scale "${SIDECAR_SCALE}")
    [[ -n "${ARM}" ]]                && EXTRA_FLAGS+=(--arm "${ARM}")
    [[ "${ABLATION_MODE}" = "1" ]]   && EXTRA_FLAGS+=(--ablation-mode)
    [[ -n "${LIMIT_PROTEINS}" ]]     && EXTRA_FLAGS+=(--limit-proteins "${LIMIT_PROTEINS}")
    python -u "${PROJECT_ROOT}/scripts/run_if_imp_refiner.py" \
        --checkpoint "${CHECKPOINT}" \
        --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
        --encoder-kind "${ENCODER_KIND}" \
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
        --apply-steps "${APPLY_STEPS}" \
        --fusion-mode "${FUSION_MODE}" \
        --fusion-alpha "${FUSION_ALPHA}" \
        --tag "${TAG}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
elif [ "${MODE}" = "diag_encoder" ]; then
    : "${CHECKPOINT:?set CHECKPOINT for diag_encoder}"
    : "${CATH_ROOT:?set CATH_ROOT for diag_encoder}"
    : "${ENCODER_CHECKPOINT:?set ENCODER_CHECKPOINT for diag_encoder}"
    # Mirror the diag_arms branch's full parameter passthrough, adding
    # only the encoder swap flags. In particular CATH_MAX_LENGTH /
    # CATH_MIN_RESOLVED_RATIO must be forwarded so the diagnostic
    # protein set matches the launcher's filters (the script's own
    # default resolved-ratio is 0.95, which would otherwise drift from
    # the launcher default 0.90).
    EXTRA_FLAGS=()
    [[ -n "${REFINER_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--refiner-checkpoint "${REFINER_CHECKPOINT}")
    [[ -n "${SIDECAR_CHECKPOINT}" ]] && EXTRA_FLAGS+=(--sidecar-checkpoint "${SIDECAR_CHECKPOINT}")
    [[ -n "${DIAG_RUN_ID}" ]]        && EXTRA_FLAGS+=(--run-id "${DIAG_RUN_ID}")
    [[ -n "${OVERWRITE}" ]]          && EXTRA_FLAGS+=(--overwrite)
    [[ "${DIAG_EMIT_PER_POSITION}" = "1" ]] && EXTRA_FLAGS+=(--emit-per-position)
    [[ -n "${USE_DRAFT_SEQ}" ]] && EXTRA_FLAGS+=(--use-draft-seq "${USE_DRAFT_SEQ}")
    # shellcheck disable=SC2086  # we WANT DIAG_ARMS to word-split
    python -u "${PROJECT_ROOT}/scripts/diag_if_imp_arms.py" \
        --checkpoint "${CHECKPOINT}" \
        --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
        --encoder-kind "${ENCODER_KIND}" \
        --cath-root "${CATH_ROOT}" \
        --cath-split "${CATH_SPLIT}" \
        --cath-max-length "${CATH_MAX_LENGTH}" \
        --cath-min-resolved-ratio "${CATH_MIN_RESOLVED_RATIO}" \
        --n-proteins "${DIAG_N_PROTEINS}" \
        --seed "${SEED}" \
        --max-iter "${MAX_ITER}" \
        --temperature "${TEMPERATURE}" \
        --mc-dropout-passes "${MC_DROPOUT_PASSES}" \
        --mask-ratio-center "${MASK_RATIO_CENTER}" \
        --mask-ratio-deviation "${MASK_RATIO_DEVIATION}" \
        --fusion-temperature "${FUSION_TEMPERATURE}" \
        --apply-steps "${APPLY_STEPS}" \
        --fusion-mode "${FUSION_MODE}" \
        --fusion-alpha "${FUSION_ALPHA}" \
        --device "${DEVICE}" \
        --arms ${DIAG_ARMS} \
        --progress-every "${DIAG_PROGRESS_EVERY}" \
        --output-dir "${OUTPUT_ROOT}" \
        "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
else
    echo "ERROR: unknown MODE='${MODE}' (expected train_refiner|train_sidecar|train_encoder|generate_refiner|generate_encoder|compare_ablation|diag_arms|diag_encoder)" >&2
    exit 2
fi

echo "if-imp ${MODE} finished."
