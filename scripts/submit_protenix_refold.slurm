#!/bin/bash
#SBATCH --job-name=protenix_refold
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/protenix/refold_%A_%a.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/protenix/refold_%A_%a.err

# The ailab default selects H200. RTX jobs also set PROTENIX_ENV to v2.0.0-blackwell.

# Protenix v2 refold-cache precompute — GPU fold + normalize (single-stage, local MSA).
#
# MSA comes from our LOCAL ColabFold (the deprecated Protenix login-node server path is gone):
#   emit-fasta -> local MSA -> inject, all offline. Prep the a3m once:
#     precompute_protenix_refold.py --mode emit-fasta --fasta-out <f>   (cache_key-keyed FASTA)
#     submit_tetramer_msa_local.slurm <f> <MSA_A3M_DIR>                  (local GPU ColabFold)
#   USE_MSA=true (default): build-json injects <MSA_A3M_DIR>/<cache_key>.a3m as paired/unpaired
#     paths, then `protenix-run pred --use_msa true` reads the LOCAL a3m (no server, GPU busy
#     immediately) and normalizes each fold's best sample into CACHE_DIR (<key>.pdb + .plddt).
#   USE_MSA=false: single-sequence no-MSA (degraded), no a3m needed.
# The group install (/scratch/gpfs/KAIYIJIANG/tools/protenix) is CALLED, never modified.
# Required env: FROM_PARQUET, CACHE_DIR; USE_MSA=true also needs MSA_A3M_DIR. Overridable:
# PROTENIX_OUT, N_SHARDS, USE_MSA, MSA_A3M_DIR, SEEDS, SAMPLE. Match --array size to N_SHARDS.

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
PROTENIX_ROOT="/scratch/gpfs/KAIYIJIANG/tools/protenix"

FROM_PARQUET="${FROM_PARQUET:?set FROM_PARQUET (generated parquet with protein_id,sequence)}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR (shared refold cache dir)}"
PROTENIX_OUT="${PROTENIX_OUT:-/scratch/gpfs/KAIYIJIANG/zijie/run/protenix_refold/$(basename "${FROM_PARQUET%.parquet}")}"
N_SHARDS="${N_SHARDS:-8}"                 # must match #SBATCH --array=0-(N-1)
USE_MSA="${USE_MSA:-true}"               # true = local ColabFold MSA (needs MSA_A3M_DIR); false = no-MSA
MSA_A3M_DIR="${MSA_A3M_DIR:-}"           # USE_MSA=true: dir of <cache_key>.a3m from submit_tetramer_msa_local.slurm
SEEDS="${SEEDS:-101}"
SAMPLE="${SAMPLE:-5}"
LIGAND_SMILES="${LIGAND_SMILES:-}"       # non-empty = HOLO cofold; empty = apo
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_JSON="${PROTENIX_OUT}/_json/shard${SHARD_IDX}of${N_SHARDS}.json"
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

# ── Build the fold JSON (immune-design) ────────────────────────────────────
FOLD_JSON="${SHARD_JSON}"
module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [ "${USE_MSA}" = "true" ]; then
    # LOCAL ColabFold MSA: inject <MSA_A3M_DIR>/<cache_key>.a3m as paired/unpaired paths
    [ -n "${MSA_A3M_DIR}" ] || { echo "FATAL: USE_MSA=true needs MSA_A3M_DIR (local ColabFold a3m; run emit-fasta -> submit_tetramer_msa_local.slurm first)." >&2; exit 2; }
    python "${PROJECT_ROOT}/scripts/precompute_protenix_refold.py" --mode build-json \
        --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
        --out-json "${FOLD_JSON}" --msa-a3m-dir "${MSA_A3M_DIR}" --msa-px-dir "${PROTENIX_OUT}/_msa" \
        --ligand-smiles "${LIGAND_SMILES}"
else
    # no-MSA single-sequence
    python "${PROJECT_ROOT}/scripts/precompute_protenix_refold.py" --mode build-json \
        --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
        --out-json "${FOLD_JSON}" --ligand-smiles "${LIGAND_SMILES}"
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
    --protenix-out "${PROTENIX_OUT}" --cache-dir "${CACHE_DIR}" \
    --ligand-smiles "${LIGAND_SMILES}"

echo "Shard ${SHARD_IDX} finished."
