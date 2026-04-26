#!/bin/bash
#SBATCH --job-name=if-phase-c
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
WORK_DIR="${WORK_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c}"

MODE="${MODE:-native}"   # native | reference_flow
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"

if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_07_01"
    ALLELE_TAG="DRB1_07_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs/0701"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/h_maps/h_maps_DRB1_07_01.parquet"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_04_01"
    ALLELE_TAG="DRB1_04_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs/0401"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/h_maps/h_maps_DRB1_04_01.parquet"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01 or HLA-DRB1*04:01" >&2
    exit 2
fi

CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${WORK_DIR}/if_test_set/test_proteins_${ALLELE_FILE_TAG}.parquet}"
PDB_ROOT="${PDB_ROOT:-${PDB_ROOT_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/if_phase_c/${MODE}}"

N_DESIGNS_PER_PROTEIN="${N_DESIGNS_PER_PROTEIN:-1}"
SEED="${SEED:-42}"
MAX_ITER="${MAX_ITER:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"

H_MAPS_PARQUET="${H_MAPS_PARQUET:-${H_MAPS_DEFAULT}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/inverse_folding/reference_flow/configs/c1_null.yaml}"
FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.05}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-0}"
H_CORPUS_STATS="${H_CORPUS_STATS:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

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
# wandb is opt-in per run via WANDB=1; default mode honors env (env.sh sets online).
WANDB_FLAG="${WANDB:-0}"
WANDB_MODE_FOR_RUN="${WANDB_MODE:-online}"
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-mhc-if-phase-c-sampling}"
WANDB_GROUP_FOR_RUN="${WANDB_GROUP:-}"
WANDB_TAGS_FOR_RUN="${WANDB_TAGS:-}"

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
            --device "${DEVICE}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    reference_flow)
        echo "  h_maps parquet  : ${H_MAPS_PARQUET}"
        echo "  Config          : ${CONFIG_PATH}"
        echo "  Config stem     : $(basename "${CONFIG_PATH}" .yaml)"
        echo "  Fail threshold  : ${FAIL_PCT_THRESHOLD}"
        echo "  Save traj       : ${SAVE_TRAJECTORIES}"
        echo "  Designs/protein : ${N_DESIGNS_PER_PROTEIN:-<from config>}"
        echo "  Seed            : ${SEED:-<from config>}"
        echo "  N steps         : ${N_STEPS:-<from config>}"
        echo "  h corpus stats  : ${H_CORPUS_STATS:-<none>}"
        echo "  Resume from     : ${RESUME_FROM:-<fresh run>}"
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

        python -u "${PROJECT_ROOT}/scripts/run_if_phase_c1.py" \
            --checkpoint "${CHECKPOINT}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            --h-maps-parquet "${H_MAPS_PARQUET}" \
            --allele "${ALLELE}" \
            --config "${CONFIG_PATH}" \
            --output-root "${OUTPUT_ROOT}" \
            --device "${DEVICE}" \
            --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
            "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    *)
        echo "Unsupported MODE=${MODE}; expected native or reference_flow" >&2
        exit 2
        ;;
esac

echo "Phase C ${MODE} finished."
