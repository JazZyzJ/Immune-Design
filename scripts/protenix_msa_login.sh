#!/bin/bash
# Protenix refold-cache precompute — STAGE A (MSA fetch, run on a LOGIN node).
#
# Full-MSA on Della is two-stage: the compute-node proxy/default does NOT whitelist
# the Protenix MSA server (403 Forbidden) and a GPU job idle-waiting on the MSA server
# gets auto-killed, but a LOGIN node has direct egress. This script (login node) builds
# the Protenix JSONs from a generated parquet and fetches full MSA (a3m) per shard,
# writing <PROTENIX_OUT>/_json/shard{k}of{N}-update-msa.json (a3m paths filled in) for
# stage B (scripts/submit_protenix_refold.slurm) to fold on GPU with NO in-job MSA call.
#
# Usage (login node):
#   FROM_PARQUET=/path/generated.parquet PROTENIX_OUT=/scratch/.../protenix_out \
#     N_SHARDS=8 bash scripts/protenix_msa_login.sh
# Overridable: MSA_SERVER_MODE (protenix|colabfold).
#
# NOTE: this POSTs your sequences to the external MSA server (protenix-server.com) and
# is server-rate-limited; it is network I/O (no GPU), so it belongs on a login node.

set -euo pipefail
: "${PS1:=}"; export PS1

PROJECT_ROOT="/home/zc1519/src/Immune-Design"
PROTENIX_ROOT="/scratch/gpfs/KAIYIJIANG/tools/protenix"

FROM_PARQUET="${FROM_PARQUET:?set FROM_PARQUET (generated parquet with protein_id,sequence)}"
PROTENIX_OUT="${PROTENIX_OUT:?set PROTENIX_OUT (must match the stage-B job)}"
N_SHARDS="${N_SHARDS:-8}"
MSA_SERVER_MODE="${MSA_SERVER_MODE:-protenix}"

mkdir -p "${PROTENIX_OUT}/_json" "${PROTENIX_OUT}/_msa"

echo "[stage A] build-json for ${N_SHARDS} shard(s) (immune-design)"
module purge
module load anaconda3/2025.12
conda activate immune-design
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
for k in $(seq 0 $((N_SHARDS - 1))); do
    python "${PROJECT_ROOT}/scripts/precompute_protenix_refold.py" --mode build-json \
        --from-parquet "${FROM_PARQUET}" --n-shards "${N_SHARDS}" --shard-idx "${k}" \
        --out-json "${PROTENIX_OUT}/_json/shard${k}of${N_SHARDS}.json"
done

echo "[stage A] fetch MSA per shard (protenix env, login egress, server=${MSA_SERVER_MODE})"
module purge
source "${PROTENIX_ROOT}/bin/protenix-env.sh"
for k in $(seq 0 $((N_SHARDS - 1))); do
    IN="${PROTENIX_OUT}/_json/shard${k}of${N_SHARDS}.json"
    UPD="${PROTENIX_OUT}/_json/shard${k}of${N_SHARDS}-update-msa.json"
    echo "  msa shard ${k}: ${IN}"
    protenix-run msa -i "${IN}" -o "${PROTENIX_OUT}/_msa" --msa_server_mode "${MSA_SERVER_MODE}"
    if [ ! -f "${UPD}" ]; then
        echo "FATAL: MSA did not produce ${UPD}" >&2
        exit 1
    fi
done

echo "[stage A] done -> ${PROTENIX_OUT}/_json/shard{k}of${N_SHARDS}-update-msa.json"
echo "[stage A] next: FROM_PARQUET=${FROM_PARQUET} CACHE_DIR=<cache> PROTENIX_OUT=${PROTENIX_OUT} \\"
echo "          N_SHARDS=${N_SHARDS} sbatch --array=0-$((N_SHARDS - 1)) scripts/submit_protenix_refold.slurm"
