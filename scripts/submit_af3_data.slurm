#!/bin/bash
#SBATCH --job-name=af3_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/af3/data_%A_%a.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/af3/data_%A_%a.err

# AlphaFold3 refold-cache precompute — STAGE A (CPU MSA data pipeline; full-MSA only).
#
# AF3's MSA is a LOCAL database search (jackhmmer/nhmmer over the AF3 DBs at
# /scratch/gpfs/DATASETS/alphafold), NOT an external server — so it runs on a CPU compute
# node (no egress, no GPU idle). Per shard: build the AF3 JSONs with the MSA fields OMITTED
# (so the pipeline fills them), then `af3-run --mode data` -> <AF3_OUT>/_data/shard{k}of{N}/
# <name>_data.json for the GPU inference (submit_af3_refold.slurm USE_MSA=true) to fold.
# The MSA search is slow (hours/protein); size --array so each shard fits --time.
# Required env: FROM_PARQUET; overridable: AF3_OUT, N_SHARDS.

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
AF3_RUN="/scratch/gpfs/KAIYIJIANG/tools/alphafold3/bin/af3-run"

FROM_PARQUET="${FROM_PARQUET:?set FROM_PARQUET (generated parquet with protein_id,sequence)}"
AF3_OUT="${AF3_OUT:-/scratch/gpfs/KAIYIJIANG/zijie/run/af3_refold/$(basename "${FROM_PARQUET%.parquet}")}"
N_SHARDS="${N_SHARDS:-8}"                 # must match #SBATCH --array=0-(N-1)
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"
JSON_DIR="${AF3_OUT}/_json/shard${SHARD_IDX}of${N_SHARDS}"
DATA_DIR="${AF3_OUT}/_data/shard${SHARD_IDX}of${N_SHARDS}"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/af3"

mkdir -p "${JSON_DIR}" "${DATA_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "AlphaFold3 refold STAGE A (CPU MSA data pipeline)"
echo "  Job / task : ${SLURM_ARRAY_JOB_ID:-NA} / ${SHARD_IDX} of ${N_SHARDS}"
echo "  From parquet: ${FROM_PARQUET}"
echo "  JSON dir   : ${JSON_DIR}"
echo "  Data dir   : ${DATA_DIR}"
echo "  Timestamp  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── build this shard's AF3 JSONs with MSA fields OMITTED (immune-design) ───
module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python "${PROJECT_ROOT}/scripts/precompute_af3_refold.py" --mode build-json \
    --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${SHARD_IDX}" \
    --json-dir "${JSON_DIR}"

# ── CPU MSA data pipeline (isolated subshell; af3-run self-contained) ──────
# Match jackhmmer/nhmmer thread count to the allocated cores (avoid oversubscription
# when submitted with fewer --cpus-per-task). Tolerate a post-output teardown non-zero
# exit; stage B fail-fasts if any <name>_data.json is missing.
DATA_CPUS="${SLURM_CPUS_PER_TASK:-8}"
(
    module purge
    "${AF3_RUN}" --mode data \
        --input_dir "${JSON_DIR}" \
        --output_dir "${DATA_DIR}" \
        --jackhmmer_n_cpu="${DATA_CPUS}" \
        --nhmmer_n_cpu="${DATA_CPUS}"
) || echo "[warn] af3-run --mode data exited non-zero; stage B validates *_data.json presence"

echo "Stage A shard ${SHARD_IDX} done -> ${DATA_DIR}/<name>_data.json"
