#!/bin/bash
#SBATCH --job-name=esmfold2_gt
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/esmfold2/gt_%A_%a.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/esmfold2/gt_%A_%a.err

# The ailab default selects H200. RTX PRO 6000 jobs override --partition and --gres at sbatch time.

set -euo pipefail
: "${PS1:=}"
export PS1

# Predict GT structures for the uricase missing-structure case set with ESMFold2.
# Array of NUM_SHARDS tasks; each folds a round-robin shard of the FASTA and
# writes its own per-shard manifest (mmcif/pdb filenames are per-protein, no
# cross-shard collision). Resume-safe: re-running skips ids already in a shard's
# manifest. NOTE: keep --array size in the header == NUM_SHARDS below.

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
URICASE_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases"
FASTA="${FASTA:-${URICASE_DIR}/esmfold2_inputs/to_fold.fasta}"
OUT_DIR="${OUT_DIR:-${URICASE_DIR}/esmfold2_structures}"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/esmfold2"

# ── ESMFold2 / gate knobs (overridable at submit time) ────────────────────
NUM_SHARDS="${NUM_SHARDS:-8}"               # must match #SBATCH --array=0-(N-1)
NUM_LOOPS="${NUM_LOOPS:-3}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-100}"   # high-quality (fast preset=32, example=50)
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-1}"
PLDDT_MIN="${PLDDT_MIN:-0.90}"               # ESMFold2 pLDDT is 0-1; 0.90 == AF "very high"
PTM_MIN="${PTM_MIN:-0.80}"                   # "strong" global topology

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ── Environment (dedicated esmfold2 env — NOT immune-design) ──────────────
module purge
module load anaconda3/2025.12
conda activate esmfold2

export HF_HOME=/scratch/gpfs/KAIYIJIANG/zijie/model_cache/hf
export HF_HUB_OFFLINE=1                       # weights pre-cached; no compute-node internet needed
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"   # for repo imports in --from-parquet/--cache-layout mode

# ── Refold-cache precompute mode (optional) ───────────────────────────────
# FROM_PARQUET: fold the UNIQUE (protein_id, sequence) pairs of a generated parquet
#   (header id = cache_key) instead of --fasta.
# CACHE_LAYOUT: also normalize each fold into the shared refold cache
#   (<CACHE_LAYOUT>/<cache_key>.pdb + .plddt, 0-100) consumed by
#   evaluate_phase_c.py --refold-model esmfold2.
FROM_PARQUET="${FROM_PARQUET:-}"
if [ -n "${FROM_PARQUET}" ]; then
    CACHE_LAYOUT="${CACHE_LAYOUT:-${OUT_DIR}/cache}"
else
    CACHE_LAYOUT="${CACHE_LAYOUT:-}"
fi

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "ESMFold2 GT prediction (uricase case set)"
echo "  Job / Array  : ${SLURM_ARRAY_JOB_ID:-NA} / task ${SLURM_ARRAY_TASK_ID:-NA}"
echo "  Node         : ${SLURM_NODELIST:-NA}"
echo "  FASTA        : ${FASTA}"
echo "  Out dir      : ${OUT_DIR}"
echo "  Shards       : ${NUM_SHARDS} (this = ${SLURM_ARRAY_TASK_ID:-0})"
echo "  loops/steps/samples : ${NUM_LOOPS}/${NUM_SAMPLING_STEPS}/${NUM_DIFFUSION_SAMPLES}"
echo "  gate         : pLDDT>=${PLDDT_MIN} & pTM>=${PTM_MIN}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo NA)"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo NA)"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Fold this shard ───────────────────────────────────────────────────────
# --from-parquet needs pyarrow (immune-design env), which the esmfold2 env lacks:
# emit the fold FASTA (header id = cache_key) in an immune-design subshell first,
# then fold that FASTA in the esmfold2 env (parent shell).
if [ -n "${FROM_PARQUET}" ]; then
    FASTA="${OUT_DIR}/_fold_input.fasta"
    (
        module purge
        module load anaconda3/2025.12
        conda activate immune-design
        export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
        python "${PROJECT_ROOT}/scripts/predict_esmfold2_gt.py" \
            --from-parquet "${FROM_PARQUET}" --emit-fold-fasta "${FASTA}" --out-dir "${OUT_DIR}"
    )
fi

INPUT_ARGS=(--fasta "${FASTA}")
if [ -n "${CACHE_LAYOUT}" ]; then
    INPUT_ARGS+=(--cache-layout "${CACHE_LAYOUT}")
fi

python "${PROJECT_ROOT}/scripts/predict_esmfold2_gt.py" \
    "${INPUT_ARGS[@]}" \
    --out-dir "${OUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-idx "${SLURM_ARRAY_TASK_ID}" \
    --num-loops "${NUM_LOOPS}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
    --num-diffusion-samples "${NUM_DIFFUSION_SAMPLES}" \
    --plddt-min "${PLDDT_MIN}" \
    --ptm-min "${PTM_MIN}" \
    --device cuda

echo "Shard ${SLURM_ARRAY_TASK_ID} finished."
