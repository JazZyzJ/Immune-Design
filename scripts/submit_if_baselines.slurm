#!/bin/bash
#SBATCH --job-name=if-baselines
#SBATCH --account=kaiyijiang
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_baselines/baseline_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_baselines/baseline_%j.err
#
# Module N (IF Comparison Baselines) parameterized launcher.
#
# Usage:
#   MODE=level1     sbatch scripts/submit_if_baselines.slurm
#   MODE=proteinmpnn ALLELE="HLA-DRB1*07:01" APPLY_NMP_FILTER=1 \
#       sbatch scripts/submit_if_baselines.slurm
#
# MODE=level1 does not need a GPU; submit with override flags to save GPU hours:
#   MODE=level1 sbatch --partition=cpu --gres=none --cpus-per-task=4 \
#       --mem=16G --time=01:00:00 scripts/submit_if_baselines.slurm

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
WORK_DIR="${WORK_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_baselines}"
export PROJECT_ROOT WORK_DIR RUN_DIR LOG_DIR

mkdir -p "${LOG_DIR}"

MODE="${MODE:-level1}"   # level1 | proteinmpnn

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -f "${PROJECT_ROOT}/lib/ijit_stub.so" ]]; then
    export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
fi
export TORCH_HOME="${TORCH_HOME:-/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch}"
export HF_HOME="${HF_HOME:-/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "============================================================"
echo "Module N (IF Comparison Baselines)"
echo "  Job ID       : ${SLURM_JOB_ID:-<no-slurm>}"
echo "  Mode         : ${MODE}"
echo "  Node         : ${SLURM_NODELIST:-<local>}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

case "${MODE}" in
    level1)
        # ── N1: Level-1 post-hoc filter on M3 sweep candidates ─────────────
        M3_OUTPUT="${M3_OUTPUT:-${RUN_DIR}/guidance/sweep_seed42}"
        OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/baselines/level1_filter/sweep_seed42}"
        mkdir -p "${OUTPUT_DIR}"

        echo "  M3 source    : ${M3_OUTPUT}"
        echo "  Output dir   : ${OUTPUT_DIR}"
        echo "------------------------------------------------------------"

        python -u "${PROJECT_ROOT}/scripts/run_if_level1_filter.py" \
            --m3-output-dir "${M3_OUTPUT}" \
            --output-dir "${OUTPUT_DIR}"
        ;;

    proteinmpnn)
        # ── N2: ProteinMPNN baseline + optional NMP post-hoc filter ────────
        ALLELE="${ALLELE:-HLA-DRB1*07:01}"
        if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
            ALLELE_FILE_TAG="HLA-DRB1_07_01"
            ALLELE_TAG="DRB1_07_01"
            PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/0701"
        elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
            ALLELE_FILE_TAG="HLA-DRB1_04_01"
            ALLELE_TAG="DRB1_04_01"
            PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/0401"
        else
            echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01 or HLA-DRB1*04:01" >&2
            exit 2
        fi

        INPUT_PDB_FOLDER="${INPUT_PDB_FOLDER:-${PDB_ROOT_DEFAULT}}"
        OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/baselines/proteinmpnn/${ALLELE_TAG}}"
        mkdir -p "${OUTPUT_DIR}"

        # ProteinMPNN knobs
        MPNN_MODEL_NAME="${MPNN_MODEL_NAME:-v_48_020}"
        NUM_SEQ_PER_TARGET="${NUM_SEQ_PER_TARGET:-8}"
        SAMPLING_TEMP="${SAMPLING_TEMP:-0.1}"
        MPNN_BATCH_SIZE="${MPNN_BATCH_SIZE:-1}"
        MPNN_SEED="${MPNN_SEED:-42}"
        MPNN_DEVICE="${MPNN_DEVICE:-0}"
        DESIGN_CHAINS="${DESIGN_CHAINS:-}"
        CA_ONLY="${CA_ONLY:-0}"
        OVERWRITE="${OVERWRITE:-0}"

        # NMP filter knobs (only used when APPLY_NMP_FILTER=1)
        APPLY_NMP_FILTER="${APPLY_NMP_FILTER:-0}"
        NETMHCIIPAN_BIN="${NETMHCIIPAN_BIN:-${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan}"
        NMP_PEP_LENGTHS="${NMP_PEP_LENGTHS:-15}"
        NMP_RANK_THRESHOLD="${NMP_RANK_THRESHOLD:-2.0}"
        NMP_RULE="${NMP_RULE:-min_n_sb}"
        NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-30}"
        NMP_MAX_LENGTHS_PER_CALL="${NMP_MAX_LENGTHS_PER_CALL:-4}"
        NMP_WORKERS="${NMP_WORKERS:-1}"
        NMP_TIMEOUT="${NMP_TIMEOUT:-600}"

        echo "  Allele       : ${ALLELE}"
        echo "  Input PDBs   : ${INPUT_PDB_FOLDER}"
        echo "  Output dir   : ${OUTPUT_DIR}"
        echo "  MPNN model   : ${MPNN_MODEL_NAME}"
        echo "  N seq/target : ${NUM_SEQ_PER_TARGET}"
        echo "  Sampling T   : ${SAMPLING_TEMP}"
        echo "  MPNN batch   : ${MPNN_BATCH_SIZE}"
        echo "  MPNN seed    : ${MPNN_SEED}"
        echo "  MPNN device  : ${MPNN_DEVICE}"
        echo "  Design chains: ${DESIGN_CHAINS:-<all>}"
        echo "  CA-only      : ${CA_ONLY}"
        echo "  Overwrite    : ${OVERWRITE}"
        echo "  Apply NMP    : ${APPLY_NMP_FILTER}"
        if [[ "${APPLY_NMP_FILTER}" == "1" ]]; then
            echo "  NMP binary   : ${NETMHCIIPAN_BIN}"
            echo "  NMP lengths  : ${NMP_PEP_LENGTHS}"
            echo "  NMP threshold: ${NMP_RANK_THRESHOLD}%"
            echo "  NMP rule     : ${NMP_RULE}"
            echo "  NMP batch    : ${NMP_BATCH_SIZE}"
            echo "  NMP lens/call: ${NMP_MAX_LENGTHS_PER_CALL}"
            echo "  NMP workers  : ${NMP_WORKERS}"
            echo "  NMP timeout  : ${NMP_TIMEOUT}s"
        fi
        echo "------------------------------------------------------------"

        EXTRA_FLAGS=()
        if [[ "${CA_ONLY}" == "1" ]]; then
            EXTRA_FLAGS+=(--ca-only)
        fi
        if [[ "${OVERWRITE}" == "1" ]]; then
            EXTRA_FLAGS+=(--overwrite)
        fi
        if [[ -n "${DESIGN_CHAINS}" ]]; then
            EXTRA_FLAGS+=(--design-chains "${DESIGN_CHAINS}")
        fi
        if [[ "${APPLY_NMP_FILTER}" == "1" ]]; then
            if [[ ! -x "${NETMHCIIPAN_BIN}" ]]; then
                echo "ERROR: NetMHCIIpan binary not found or not executable: ${NETMHCIIPAN_BIN}" >&2
                exit 2
            fi
            EXTRA_FLAGS+=(
                --apply-nmp-filter
                --netmhciipan-bin "${NETMHCIIPAN_BIN}"
                --allele "${ALLELE}"
                --nmp-pep-lengths "${NMP_PEP_LENGTHS}"
                --nmp-rank-threshold "${NMP_RANK_THRESHOLD}"
                --nmp-rule "${NMP_RULE}"
                --nmp-batch-size "${NMP_BATCH_SIZE}"
                --nmp-max-lengths-per-call "${NMP_MAX_LENGTHS_PER_CALL}"
                --nmp-workers "${NMP_WORKERS}"
                --nmp-timeout "${NMP_TIMEOUT}"
            )
        fi

        python -u "${PROJECT_ROOT}/scripts/run_proteinmpnn_baseline.py" \
            --input-pdb-folder "${INPUT_PDB_FOLDER}" \
            --output-dir "${OUTPUT_DIR}" \
            --mpnn-model-name "${MPNN_MODEL_NAME}" \
            --num-seq-per-target "${NUM_SEQ_PER_TARGET}" \
            --sampling-temp "${SAMPLING_TEMP}" \
            --mpnn-batch-size "${MPNN_BATCH_SIZE}" \
            --mpnn-seed "${MPNN_SEED}" \
            --mpnn-device "${MPNN_DEVICE}" \
            "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
        ;;

    *)
        echo "Unsupported MODE=${MODE}; expected level1 or proteinmpnn" >&2
        exit 2
        ;;
esac

echo "Module N baseline (${MODE}) finished."
