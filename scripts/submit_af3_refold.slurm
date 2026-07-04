#!/bin/bash
#SBATCH --job-name=af3_refold
#SBATCH --partition=ailab
#SBATCH --constraint="h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/af3/refold_%A_%a.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/af3/refold_%A_%a.err

# AlphaFold3 refold-cache precompute — GPU INFERENCE + normalize (one shard per task).
#
# AF3's MSA is a LOCAL database search (no external server), so full-MSA is two-stage but
# both stages stay on compute nodes (no login/egress needed):
#   USE_MSA=true : stage A (scripts/submit_af3_data.slurm, CPU) must have written the per-shard
#     data JSONs to <AF3_OUT>/_data/shard{k}of{N}/<name>_data.json; THIS GPU job runs
#     `af3-run --mode inference` on them (GPU busy immediately, no MSA on the GPU node).
#   USE_MSA=false: builds empty-MSA JSONs here and runs `--mode inference` single-sequence.
# Then normalizes each fold's best model into CACHE_DIR (<cache_key>.pdb + .plddt, 0-100).
# The group install (/scratch/gpfs/KAIYIJIANG/tools/alphafold3) is CALLED, never modified;
# af3-run is self-contained (its own interpreter). Match --array size to N_SHARDS.
# Required env: FROM_PARQUET, CACHE_DIR; overridable: AF3_OUT, N_SHARDS, USE_MSA, NUM_SAMPLES.

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
AF3_RUN="/scratch/gpfs/KAIYIJIANG/tools/alphafold3/bin/af3-run"

FROM_PARQUET="${FROM_PARQUET:?set FROM_PARQUET (generated parquet with protein_id,sequence)}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR (shared refold cache dir)}"
AF3_OUT="${AF3_OUT:-/scratch/gpfs/KAIYIJIANG/zijie/run/af3_refold/$(basename "${FROM_PARQUET%.parquet}")}"
N_SHARDS="${N_SHARDS:-8}"                 # must match #SBATCH --array=0-(N-1)
USE_MSA="${USE_MSA:-true}"               # true = fold stage-A data JSONs; false = no-MSA
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
JSON_DIR="${AF3_OUT}/_json/shard${SHARD_IDX}of${N_SHARDS}"
DATA_DIR="${AF3_OUT}/_data/shard${SHARD_IDX}of${N_SHARDS}"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/af3"

mkdir -p "${AF3_OUT}" "${JSON_DIR}" "${LOG_DIR}" "${CACHE_DIR}"

echo "============================================================"
echo "AlphaFold3 refold-cache precompute — GPU inference"
echo "  Job / task : ${SLURM_ARRAY_JOB_ID:-NA} / ${SHARD_IDX} of ${N_SHARDS}"
echo "  Node       : ${SLURM_NODELIST:-NA}"
echo "  From parquet: ${FROM_PARQUET}"
echo "  AF3 out    : ${AF3_OUT}"
echo "  Cache dir  : ${CACHE_DIR}"
echo "  use_msa    : ${USE_MSA} | num_samples: ${NUM_SAMPLES}"
echo "  Timestamp  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Select the inference input dir ─────────────────────────────────────────
if [ "${USE_MSA}" = "true" ]; then
    # Stage A (--mode data) writes the MSA'd input as <DATA_DIR>/<name>/<name>_data.json
    # (nested per fold). AF3 inference --input_dir globs top-level *.json, so gather the
    # nested data JSONs into a flat dir via symlinks.
    INFER_INPUT="${AF3_OUT}/_infer_input/shard${SHARD_IDX}of${N_SHARDS}"
    mkdir -p "${INFER_INPUT}"
    n_data=0
    for f in "${DATA_DIR}"/*/*_data.json; do
        [ -f "${f}" ] || continue
        ln -sf "${f}" "${INFER_INPUT}/$(basename "${f}")"
        n_data=$((n_data + 1))
    done
    if [ "${n_data}" -eq 0 ]; then
        echo "FATAL: full-MSA requested but no <name>/*_data.json under ${DATA_DIR}." >&2
        echo "       Run stage A first: sbatch scripts/submit_af3_data.slurm." >&2
        exit 2
    fi
    echo "  gathered ${n_data} data JSON(s) -> ${INFER_INPUT}"
else
    # no-MSA: build empty-MSA JSONs here (immune-design)
    module purge
    module load anaconda3/2025.12
    conda activate immune-design
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    python "${PROJECT_ROOT}/scripts/precompute_af3_refold.py" --mode build-json --no-msa \
        --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
        --json-dir "${JSON_DIR}"
    INFER_INPUT="${JSON_DIR}"
fi

# ── GPU inference (isolated subshell; af3-run is self-contained) ────────────
# AF3/JAX can SIGABRT during interpreter teardown AFTER all outputs are written;
# the normalize step below is the real gate (fail-fast on any missing fold), so a
# post-output non-zero exit here must NOT fail the job.
(
    module purge
    "${AF3_RUN}" --mode inference \
        --input_dir "${INFER_INPUT}" \
        --output_dir "${AF3_OUT}" \
        --num_diffusion_samples="${NUM_SAMPLES}"
) || echo "[warn] af3-run exited non-zero (JAX teardown?); normalize validates outputs"

# ── Normalize into the shared refold cache (immune-design) ─────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python "${PROJECT_ROOT}/scripts/precompute_af3_refold.py" --mode normalize \
    --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
    --af3-out "${AF3_OUT}" --cache-dir "${CACHE_DIR}"

echo "Shard ${SHARD_IDX} finished."
