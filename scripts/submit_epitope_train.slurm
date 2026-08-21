#!/bin/bash
#SBATCH --job-name=epi-train
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/epitope_head/epi_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/epitope_head/epi_%j.err

# Parameterized epitope-head training launcher (HIMP-aware).
# Replaces per-allele submit_train_drb*.slurm by reading allele / config /
# seed / p_aug from environment variables. Defaults reproduce the legacy
# DRB1*04:01 + cnn_himp_v1 + seed 42 behavior so a no-arg ``sbatch`` still
# does something sensible.
#
# Sweep example:
#   for ALLELE in "HLA-DRB1*07:01" "HLA-DRB1*04:01"; do
#     for CFG in cnn_himp_v1 cnn_himp_v1_npoff cnn_himp_v1_resoff \
#                cnn_himp_v1_ignore cnn_himp_v1_aggmax; do
#       ALLELE="$ALLELE" \
#       OVERRIDE_CFG="$PWD/epitope_head/configs/${CFG}.yaml" \
#       SEED=42 \
#       sbatch scripts/submit_epitope_train.slurm
#     done
#   done

: "${PS1:=}"
export PS1

set -euo pipefail

# ── Sweepable env-var inputs (with defaults) ──────────────────────────────
ALLELE="${ALLELE:-HLA-DRB1*04:01}"
SEED="${SEED:-42}"
P_AUG="${P_AUG:-0.20}"     # default 0.20 (LC1 baseline); set 0.00 for 1501
PROFILE="${PROFILE:-strict}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
OVERRIDE_CFG="${OVERRIDE_CFG:-${PROJECT_ROOT}/epitope_head/configs/cnn_himp_v1.yaml}"

# ── Derive allele tag (drb0701 / drb0401 / drb1501) from allele name ──────
# Accepts "HLA-DRB1*07:01" or "DRB1*07:01"; produces "drb0701".
strip="${ALLELE#HLA-}"
strip="${strip#DRB1\*}"
ALLELE_TAG="drb${strip/:/}"

CFG_NAME=$(basename "${OVERRIDE_CFG}" .yaml)
RUN_TAG="${CFG_NAME}_${ALLELE_TAG}_seed${SEED}"

# Wave-3 CV: SPLITS_SUBDIR (e.g. cv5/fold0) selects fold-specific id files and
# gives the run its own output dir so folds do not collide.
SPLITS_SUBDIR="${SPLITS_SUBDIR:-}"
if [ -n "${SPLITS_SUBDIR}" ]; then
    RUN_TAG="${RUN_TAG}_${SPLITS_SUBDIR//\//_}"
fi

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/manifests/${ALLELE_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/${RUN_TAG}}"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/epitope_head"
REGISTRY="${REGISTRY:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/augmentation/${ALLELE_TAG}/mutation_registry_strict.parquet}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
export PS1="${PS1:-}"
conda activate immune-design
module load proxy/default

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"


# wandb storage roots — keep all run state on /scratch, never in $HOME or cwd.
export WANDB_DIR="${WANDB_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CONFIG_DIR}"
# wandb is opt-in per run via WANDB=1.
# fasta/iedb modes still run without wandb; phase_c forwards the flags below.
WANDB_FLAG="${WANDB:-0}"
WANDB_MODE_FOR_RUN="${WANDB_MODE:-online}"
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-Immune-Design}"
WANDB_GROUP_FOR_RUN="${WANDB_GROUP:-}"
WANDB_TAGS_FOR_RUN="${WANDB_TAGS:-{CONFIG:${CFG_NAME},ALLELE:${ALLELE_TAG},SEED:${SEED}}}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Epitope Head HIMP Training"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Run tag        : ${RUN_TAG}"
echo "  Allele         : ${ALLELE}  (tag=${ALLELE_TAG})"
echo "  Override cfg   : ${OVERRIDE_CFG}"
echo "  Seed           : ${SEED}"
echo "  Profile        : ${PROFILE}"
echo "  p_aug          : ${P_AUG}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output root    : ${OUTPUT_ROOT}"
echo "  Registry       : ${REGISTRY}"
echo "  Python         : $(python --version 2>&1)"
echo "  PyTorch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail     : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name       : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Build augmentation flag (only if registry exists and p_aug > 0) ──────
AUG_FLAG=()
if [[ -f "${REGISTRY}" && "${P_AUG}" != "0.00" && "${P_AUG}" != "0" ]]; then
    AUG_FLAG=(--registry-path "${REGISTRY}" --p-aug "${P_AUG}")
    echo "  Registry found — runtime augmentation enabled (p_aug=${P_AUG})"
else
    echo "  Augmentation OFF (registry missing or p_aug=0)"
fi

# ── Optional smoke mode (SMOKE=1) ─────────────────────────────────────────
SMOKE_FLAG=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG=(--smoke)
    echo "  SMOKE MODE enabled — small data + few epochs"
fi

# ── Wave-3 CV fold passthrough ────────────────────────────────────────────
SPLITS_FLAG=()
if [ -n "${SPLITS_SUBDIR}" ]; then
    SPLITS_FLAG=(--splits-subdir "${SPLITS_SUBDIR}")
    echo "  CV fold split subdir: ${SPLITS_SUBDIR}"
fi

# ── Launch training ──────────────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/train_v2_ablation.py" \
    --variant-id LC1 \
    --seed "${SEED}" \
    --device cuda \
    --profile "${PROFILE}" \
    --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
    --override-config "${OVERRIDE_CFG}" \
    --data-dir "${DATA_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    "${AUG_FLAG[@]+"${AUG_FLAG[@]}"}" \
    "${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"}" \
    "${SPLITS_FLAG[@]+"${SPLITS_FLAG[@]}"}" \
    --wandb \
    --wandb-name "${RUN_TAG}" \
    --wandb-entity "chenzijie040528-zhejiang-university" \
    --wandb-project "Immune-Design"

echo "Epitope head training finished: ${RUN_TAG}"
