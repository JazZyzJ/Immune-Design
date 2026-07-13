#!/bin/bash
# Local GPU ColabFold MSA search in PROTENIX format (pairing.a3m + non_pairing.a3m per query),
# using Protenix's own scripts/colabfold_msa.py against the shared local DB. This is the Protenix
# counterpart of submit_tetramer_msa_local.slurm (which emits a single ColabFold a3m for ESMFold2).
# colabfold_msa.py runs colabfold_search locally (gpu_server=0, no persistent-server hang) then
# splits + adds taxonomy IDs so the output is directly usable as Protenix pairedMsaPath/unpairedMsaPath.
#
# Usage:
#   sbatch --output=<logs>/cfmsapx_%j.out --error=<logs>/cfmsapx_%j.err \
#     scripts/submit_tetramer_msa_local_protenix.slurm <QUERY.fasta> <RESULT_DIR> [extra colabfold_msa.py flags]
#
#SBATCH --job-name=cf_msa_px
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=04:00:00
set -eo pipefail
export PS1="${PS1:-}"
module load colabfold/1.5.5

DB="${COLABFOLD_DB:-/scratch/gpfs/KAIYIJIANG/databases/colabfold}"
CFBIN=/projects/MOLBIO/local/src/localcolabfold-1.5.5/localcolabfold/colabfold-conda/bin
PXROOT=/scratch/gpfs/KAIYIJIANG/tools/protenix
COLABFOLD_MSA_PY="${COLABFOLD_MSA_PY:-$PXROOT/releases/v2.0.0/scripts/colabfold_msa.py}"
PXPY="${PXPY:-$PXROOT/envs/v2.0.0/bin/python}"

QUERY="${1:?QUERY.fasta}"; RESULT="${2:?RESULT_DIR}"; shift 2
mkdir -p "$RESULT"

exec "$PXPY" "$COLABFOLD_MSA_PY" "$QUERY" "$DB" "$RESULT" \
  --db1 uniref30_2302_db --db3 colabfold_envdb_202108_db --gpu 1 \
  --colabsearch "$CFBIN/colabfold_search" --mmseqs "$CFBIN/mmseqs" "$@"
