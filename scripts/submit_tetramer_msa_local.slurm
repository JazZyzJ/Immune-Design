#!/bin/bash
# Local GPU ColabFold MSA search for the tetramer pipeline (PLAN_TETRAMER_GATE.md).
# Replaces the throttled ColabFold server (api.colabfold.com, ~6.5 min/seq) with the local
# GPU-indexed DB at /scratch/gpfs/KAIYIJIANG/databases/colabfold (uniref30_2302 + envdb),
# via mmseqs GPU search — no throttle, hundreds of sequences in minutes. Produces one ColabFold
# a3m per query (feeds ESMFold2 MSA.from_a3m directly; convert for Protenix).
#
# Usage:
#   sbatch --output=<logs>/cfmsa_%j.out --error=<logs>/cfmsa_%j.err \
#     scripts/submit_tetramer_msa_local.slurm <QUERY.fasta> <RESULT_DIR> [extra colabfold_search flags]
#
#SBATCH --job-name=cf_msa_local
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# 240G is the historical request; MEASURED peak is far lower (74-query shards on ailab, 2026-08-07:
# MaxRSS 45.9-55.8 GB, 7m15s-7m47s each). --db-load-mode 2 sizes memory by the DB, not the query
# count, so a bigger batch does not scale this up much. On a contended ailab (1500G / 8 GPUs =
# ~187G fair share per GPU) a 240G ask exceeds one GPU's share and delays scheduling for nothing;
# override with `sbatch --mem=96G` for sharded batches. Left at 240G here only because no
# large-batch measurement exists yet.
#SBATCH --mem=240G
#SBATCH --time=04:00:00
set -eo pipefail   # NOT -u: colabfold conda activation references unset $PS1
export PS1="${PS1:-}"
module load colabfold/1.5.5

DB="${COLABFOLD_DB:-/scratch/gpfs/KAIYIJIANG/databases/colabfold}"
QUERY="${1:?QUERY.fasta}"; RESULT="${2:?RESULT_DIR}"; shift 2
mkdir -p "$RESULT"

# --gpu-server 0: one batch call loads the GPU DB once for all queries (no separate persistent
# gpuserver process needed; --gpu-server 1 requires a pre-started `mmseqs gpuserver`).
exec colabfold_search \
  --gpu 1 --gpu-server 0 --db-load-mode 2 --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --db1 uniref30_2302_db --db3 colabfold_envdb_202108_db --use-env 1 \
  "$QUERY" "$DB" "$RESULT" "$@"
