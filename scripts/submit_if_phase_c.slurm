#!/bin/bash
#SBATCH --job-name=if-phase-c
#SBATCH --account=kaiyijiang
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
WORK_DIR="${WORK_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c}"
export PROJECT_ROOT WORK_DIR RUN_DIR LOG_DIR

MODE="${MODE:-native}"   # native | reference_flow
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"

if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_07_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_07_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_07_01.parquet"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_04_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_04_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_04_01.parquet"
elif [[ "${ALLELE}" == "HLA-DRB1*15:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_15_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_15_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_15_01.parquet"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01, HLA-DRB1*04:01, or HLA-DRB1*15:01" >&2
    exit 2
fi

CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${WORK_DIR}/if_test_set/if_ready/fast/fast_v2_${ALLELE_FILE_TAG}.parquet}"
PDB_ROOT="${PDB_ROOT:-${PDB_ROOT_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/if_phase_c/${MODE}}"

N_DESIGNS_PER_PROTEIN="${N_DESIGNS_PER_PROTEIN:-1}"
SEED="${SEED:-42}"
# C0 (DPLM native) iterations. Aligned with the DPLM paper's inverse-folding
# evaluation default (README task.generator.max_iter=100). C1 iter count is
# read from the YAML config's sampler.n_steps (also 100) for fair comparison.
MAX_ITER="${MAX_ITER:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NATIVE_SAMPLING_STRATEGY="${NATIVE_SAMPLING_STRATEGY:-argmax}"
NATIVE_BATCH_SIZE="${NATIVE_BATCH_SIZE:-1}"

# h-map parquet is optional (run_if_phase_c1.py --h-maps-parquet is optional):
# leave H_MAPS_PARQUET unset to use the allele default; set it explicitly EMPTY
# (H_MAPS_PARQUET= sbatch ...) to run without an h-map (constant_one /
# controller-only arms, where g(h)=1 and h is inert). The `-` (not `:-`) keeps an
# explicit empty value instead of falling back to the default.
H_MAPS_PARQUET="${H_MAPS_PARQUET-${H_MAPS_DEFAULT}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/inverse_folding/reference_flow/configs/c1_null.yaml}"
FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.05}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-0}"
H_CORPUS_STATS="${H_CORPUS_STATS:-}"
FLAG="${FLAG:-}"
RUN_ID="${RUN_ID:-}"
RF_BATCH_SIZE="${RF_BATCH_SIZE:-1}"

# Phase D1 controller wiring (optional; only consumed by MODE=reference_flow).
# CONTROLLER_CONFIG empty → controller is disabled and the head flags are
# ignored.
CONTROLLER_CONFIG="${CONTROLLER_CONFIG:-}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-}"
HEAD_CONFIG_DIR="${HEAD_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"
HEAD_VARIANT_ID="${HEAD_VARIANT_ID:-}"
HEAD_DEVICE="${HEAD_DEVICE:-cuda}"
HEAD_WINDOW_BATCH_SIZE="${HEAD_WINDOW_BATCH_SIZE:-}"
HEAD_ALLELE_IDX="${HEAD_ALLELE_IDX:-0}"
# Stage C.1 global-pressure calibration JSON (PLAN_RF_UNI_CTRL.md C1.3). Required
# only when the controller config has global_pressure.enabled=true; empty → not
# forwarded (the controller fails fast at load if the band is needed).
GLOBAL_PRESSURE_CALIBRATION_JSON="${GLOBAL_PRESSURE_CALIBRATION_JSON:-}"

# Uricase enzyme mode v0 hard-anchor constraint manifest (optional; consumed only
# by MODE=reference_flow). Empty → unconstrained global RF (unchanged behavior).
CONSTRAINT_MANIFEST="${CONSTRAINT_MANIFEST:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

module purge
module load anaconda3/2025.12
conda activate immune-design
module load proxy/default

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}/inverse_folding/dplm/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# wandb storage roots — keep all run state on /scratch, never in $HOME or cwd.
export WANDB_DIR="${WANDB_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CONFIG_DIR}"
WANDB_FLAG="${WANDB:-1}"
WANDB_MODE_FOR_RUN="${WANDB_MODE:-online}"
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-mhc-if-phase-c-sampling}"
WANDB_GROUP_FOR_RUN="${WANDB_GROUP:-}"
WANDB_TAGS_FOR_RUN="${WANDB_TAGS:-}"
export WANDB_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb"
export WANDB_CACHE_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache"
export WANDB_DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/data"
export WANDB_CONFIG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/config"


echo "============================================================"
echo "Phase C generation"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Mode            : ${MODE}"
echo "  Node            : ${SLURM_NODELIST}"
echo "  GPUs            : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root    : ${PROJECT_ROOT}"
echo "  Allele          : ${ALLELE}"
echo "  Checkpoint      : ${CHECKPOINT}"
echo "  Test parquet    : ${TEST_SET_PARQUET}"
echo "  PDB root        : ${PDB_ROOT}"
echo "  Output root     : ${OUTPUT_ROOT}"
echo "  Device          : ${DEVICE}"
echo "  Python          : $(python --version 2>&1)"
echo "  PyTorch         : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail      : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

WANDB_ARGS=()
if [[ "${WANDB_FLAG}" == "1" ]]; then
    WANDB_ARGS+=(--wandb)
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT_FOR_RUN}")
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE_FOR_RUN}")
    if [[ -n "${WANDB_GROUP_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-group "${WANDB_GROUP_FOR_RUN}")
    fi
    if [[ -n "${WANDB_TAGS_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-tags "${WANDB_TAGS_FOR_RUN}")
    fi
fi
echo "  wandb           : ${WANDB_FLAG} (project=${WANDB_PROJECT_FOR_RUN} mode=${WANDB_MODE_FOR_RUN})"

case "${MODE}" in
    native)
        echo "  Designs/protein : ${N_DESIGNS_PER_PROTEIN}"
        echo "  Seed            : ${SEED}"
        echo "  Max iter        : ${MAX_ITER}"
        echo "  Temperature     : ${TEMPERATURE}"
        echo "  Sampling strat  : ${NATIVE_SAMPLING_STRATEGY}"
        echo "  Batch size      : ${NATIVE_BATCH_SIZE}"
        python -u "${PROJECT_ROOT}/scripts/run_if_phase_c0.py" \
            --checkpoint "${CHECKPOINT}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            --allele "${ALLELE}" \
            --output-root "${OUTPUT_ROOT}" \
            --n-designs-per-protein "${N_DESIGNS_PER_PROTEIN}" \
            --seed "${SEED}" \
            --max-iter "${MAX_ITER}" \
            --temperature "${TEMPERATURE}" \
            --sampling-strategy "${NATIVE_SAMPLING_STRATEGY}" \
            --device "${DEVICE}" \
            --batch-size "${NATIVE_BATCH_SIZE}" \
            --flag-name "${FLAG}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    reference_flow)
        echo "  h_maps parquet  : ${H_MAPS_PARQUET:-<none: h absent, g(h)=1>}"
        echo "  Config          : ${CONFIG_PATH}"
        echo "  Config stem     : $(basename "${CONFIG_PATH}" .yaml)"
        echo "  Fail threshold  : ${FAIL_PCT_THRESHOLD}"
        echo "  Save traj       : ${SAVE_TRAJECTORIES}"
        echo "  Designs/protein : ${N_DESIGNS_PER_PROTEIN:-<from config>}"
        echo "  Seed            : ${SEED:-<from config>}"
        echo "  N steps         : ${N_STEPS:-<from config>}"
        echo "  h corpus stats  : ${H_CORPUS_STATS:-<none>}"
        echo "  Resume from     : ${RESUME_FROM:-<fresh run>}"
        echo "  Run ID          : ${RUN_ID:-<auto>}"
        echo "  RF batch size   : ${RF_BATCH_SIZE}"
        echo "  Controller YAML : ${CONTROLLER_CONFIG:-<disabled>}"
        if [[ -n "${CONTROLLER_CONFIG}" ]]; then
            echo "  Head checkpoint : ${HEAD_CHECKPOINT:-<missing>}"
            echo "  Head config dir : ${HEAD_CONFIG_DIR}"
            echo "  Head variant    : ${HEAD_VARIANT_ID:-<missing>}"
            echo "  Head device     : ${HEAD_DEVICE}"
            echo "  Head WBS        : ${HEAD_WINDOW_BATCH_SIZE:-<predictor default>}"
            echo "  Head allele idx : ${HEAD_ALLELE_IDX}"
            echo "  GR calibration  : ${GLOBAL_PRESSURE_CALIBRATION_JSON:-<none / pressure off>}"
        fi
        EXTRA_FLAGS=()
        if [[ -n "${H_CORPUS_STATS}" ]]; then
            EXTRA_FLAGS+=(--h-corpus-stats "${H_CORPUS_STATS}")
        fi
        if [[ -n "${RESUME_FROM:-}" ]]; then
            EXTRA_FLAGS+=(--resume-from "${RESUME_FROM}")
        fi
        if [[ "${SAVE_TRAJECTORIES}" == "1" ]]; then
            EXTRA_FLAGS+=(--save-trajectories)
        fi
        if [[ -n "${N_DESIGNS_PER_PROTEIN:-}" ]]; then
            EXTRA_FLAGS+=(--n-designs-per-protein "${N_DESIGNS_PER_PROTEIN}")
        fi
        if [[ -n "${SEED:-}" ]]; then
            EXTRA_FLAGS+=(--seed "${SEED}")
        fi
        if [[ -n "${N_STEPS:-}" ]]; then
            EXTRA_FLAGS+=(--n-steps "${N_STEPS}")
        fi
        EXTRA_FLAGS+=(--batch-size "${RF_BATCH_SIZE}")
        if [[ -n "${RUN_ID}" ]]; then
            EXTRA_FLAGS+=(--run-id "${RUN_ID}")
        fi
        if [[ -n "${CONTROLLER_CONFIG}" ]]; then
            EXTRA_FLAGS+=(--controller-config "${CONTROLLER_CONFIG}")
            if [[ -n "${HEAD_CHECKPOINT}" ]]; then
                EXTRA_FLAGS+=(--head-checkpoint "${HEAD_CHECKPOINT}")
            fi
            EXTRA_FLAGS+=(--head-config-dir "${HEAD_CONFIG_DIR}")
            if [[ -n "${HEAD_VARIANT_ID}" ]]; then
                EXTRA_FLAGS+=(--head-variant-id "${HEAD_VARIANT_ID}")
            fi
            EXTRA_FLAGS+=(--head-device "${HEAD_DEVICE}")
            EXTRA_FLAGS+=(--head-allele-idx "${HEAD_ALLELE_IDX}")
            if [[ -n "${HEAD_WINDOW_BATCH_SIZE}" ]]; then
                EXTRA_FLAGS+=(--head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}")
            fi
            # Stage C.1 global-pressure calibration (PLAN_RF_UNI_CTRL.md C1.3).
            if [[ -n "${GLOBAL_PRESSURE_CALIBRATION_JSON}" ]]; then
                EXTRA_FLAGS+=(--global-pressure-calibration-json "${GLOBAL_PRESSURE_CALIBRATION_JSON}")
            fi
        fi
        # Uricase enzyme mode v0 hard-anchor constraint manifest (optional;
        # PLAN_URICASE_ENZYME_MODE). Forwarded only in MODE=reference_flow, with or
        # without a controller.
        if [[ -n "${CONSTRAINT_MANIFEST}" ]]; then
            EXTRA_FLAGS+=(--constraint-manifest "${CONSTRAINT_MANIFEST}")
        fi

        # Pass --h-maps-parquet only when H_MAPS_PARQUET is non-empty; an empty
        # value means "no h-map" and the flag is omitted entirely.
        H_MAPS_FLAG=()
        if [[ -n "${H_MAPS_PARQUET}" ]]; then
            H_MAPS_FLAG=(--h-maps-parquet "${H_MAPS_PARQUET}")
        fi

        python -u "${PROJECT_ROOT}/scripts/run_if_phase_c1.py" \
            --checkpoint "${CHECKPOINT}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            "${H_MAPS_FLAG[@]+"${H_MAPS_FLAG[@]}"}" \
            --allele "${ALLELE}" \
            --config "${CONFIG_PATH}" \
            --output-root "${OUTPUT_ROOT}" \
            --device "${DEVICE}" \
            --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
            --flag-name "${FLAG}" \
            "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    *)
        echo "Unsupported MODE=${MODE}; expected native or reference_flow" >&2
        exit 2
        ;;
esac

echo "Phase C ${MODE} finished."
