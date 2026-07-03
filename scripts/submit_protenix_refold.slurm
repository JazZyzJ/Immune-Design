#!/bin/bash
#SBATCH --job-name=protenix_refold
#SBATCH --partition=ailab
#SBATCH --constraint="h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/protenix/refold_%A_%a.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/protenix/refold_%A_%a.err

# Protenix v2 refold-cache precompute — STAGE B (GPU fold + normalize).
#
# Full-MSA on Della is two-stage because the compute-node proxy does NOT whitelist
# the Protenix MSA server (403) and a GPU job idle-waiting on MSA gets auto-killed:
#   Stage A (login node, direct egress): scripts/protenix_msa_login.sh fetches MSA and
#     writes <PROTENIX_OUT>/_json/shard{k}of{N}-update-msa.json (a3m paths filled in).
#   Stage B (THIS GPU job): folds that pre-fetched-MSA JSON — `--use_msa true` reads the
#     LOCAL a3m (no server call → GPU busy immediately, never idle) — then normalizes
#     each fold's best sample into CACHE_DIR (<cache_key>.pdb + .plddt, 0-100).
# USE_MSA=false folds single-sequence (no-MSA, degraded protenix-v2) and needs no stage A.
# The group install (/scratch/gpfs/KAIYIJIANG/tools/protenix) is CALLED, never modified.
# Required env: FROM_PARQUET, CACHE_DIR; overridable: PROTENIX_OUT, N_SHARDS, USE_MSA,
# SEEDS, SAMPLE. Match --array size to N_SHARDS.

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
PROTENIX_ROOT="/scratch/gpfs/KAIYIJIANG/tools/protenix"

FROM_PARQUET="${FROM_PARQUET:?set FROM_PARQUET (generated parquet with protein_id,sequence)}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR (shared refold cache dir)}"
PROTENIX_OUT="${PROTENIX_OUT:-/scratch/gpfs/KAIYIJIANG/zijie/run/protenix_refold/$(basename "${FROM_PARQUET%.parquet}")}"
N_SHARDS="${N_SHARDS:-8}"                 # must match #SBATCH --array=0-(N-1)
USE_MSA="${USE_MSA:-true}"               # true = fold pre-fetched full-MSA (stage A first); false = no-MSA
SEEDS="${SEEDS:-101}"
SAMPLE="${SAMPLE:-5}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_JSON="${PROTENIX_OUT}/_json/shard${SHARD_IDX}of${N_SHARDS}.json"
UPDATED_JSON="${PROTENIX_OUT}/_json/shard${SHARD_IDX}of${N_SHARDS}-update-msa.json"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/protenix"

mkdir -p "${PROTENIX_OUT}/_json" "${LOG_DIR}" "${CACHE_DIR}"

echo "============================================================"
echo "Protenix refold-cache precompute — STAGE B (GPU fold)"
echo "  Job / task   : ${SLURM_ARRAY_JOB_ID:-NA} / ${SHARD_IDX} of ${N_SHARDS}"
echo "  Node         : ${SLURM_NODELIST:-NA}"
echo "  From parquet : ${FROM_PARQUET}"
echo "  Cache dir    : ${CACHE_DIR}"
echo "  Protenix out : ${PROTENIX_OUT}"
echo "  use_msa      : ${USE_MSA}"
echo "  seeds/sample : ${SEEDS}/${SAMPLE}"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Select the fold JSON ───────────────────────────────────────────────────
if [ "${USE_MSA}" = "true" ]; then
    FOLD_JSON="${UPDATED_JSON}"
    if [ ! -f "${FOLD_JSON}" ]; then
        echo "FATAL: full-MSA requested but ${FOLD_JSON} is missing." >&2
        echo "       Run stage A first: scripts/protenix_msa_login.sh (login node)." >&2
        exit 2
    fi
else
    # no-MSA: build this shard's plain JSON in immune-design (pyarrow) if absent
    FOLD_JSON="${SHARD_JSON}"
    if [ ! -f "${FOLD_JSON}" ]; then
        module purge
        module load anaconda3/2025.12
        conda activate immune-design
        export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
        python "${PROJECT_ROOT}/scripts/precompute_protenix_refold.py" --mode build-json \
            --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
            --out-json "${FOLD_JSON}"
    fi
fi

# ── Step 1: fold with Protenix (isolated subshell, GPU; a3m is local → no proxy) ──
(
    module purge
    source "${PROTENIX_ROOT}/bin/protenix-env.sh"
    protenix-run pred \
        -i "${FOLD_JSON}" \
        -o "${PROTENIX_OUT}" \
        --use_msa "${USE_MSA}" \
        --seeds "${SEEDS}" \
        --sample "${SAMPLE}"
)

# ── Step 2: normalize into the shared refold cache (immune-design) ─────────
module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python "${PROJECT_ROOT}/scripts/precompute_protenix_refold.py" --mode normalize \
    --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
    --protenix-out "${PROTENIX_OUT}" --cache-dir "${CACHE_DIR}"

echo "Shard ${SHARD_IDX} finished."
