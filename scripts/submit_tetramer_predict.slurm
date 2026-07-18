#!/bin/bash
# Compute-node Protenix prediction array for the tetramer refold gate (PLAN_TETRAMER_GATE.md §5).
# Each array task predicts ONE homo-oligomer job from a ready-list of MSA-updated JSONs produced
# by scripts/build_protenix_jsons.py (from a local ColabFold a3m). MSA is already attached — this
# runs `protenix pred --use_msa true`, which reads the pairedMsaPath/unpairedMsaPath in each JSON.
#
# Reuses the shared Protenix install (envs/checkpoint via bin/protenix-run), mirroring
# /scratch/gpfs/KAIYIJIANG/tools/protenix/slurm/protenix_pred.slurm but adding array indexing.
#
# Usage:
#   sbatch --array=0-N%K \
#     --output=<logs>/tetra_pred_%A_%a.out --error=<logs>/tetra_pred_%A_%a.err \
#     scripts/submit_tetramer_predict.slurm <READY_LIST> <PRED_ROOT> [extra protenix pred flags]
#
#SBATCH --job-name=tetra_pred
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint="h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail

LIST="${1:?READY_LIST}"; PRED_ROOT="${2:?PRED_ROOT}"; shift 2
PROTENIX="${PROTENIX_RUN:-/scratch/gpfs/KAIYIJIANG/tools/protenix/bin/protenix-run}"

JSON="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")"
[ -n "${JSON:-}" ] && [ -f "$JSON" ] || { echo "no JSON for array task ${SLURM_ARRAY_TASK_ID} in $LIST" >&2; exit 2; }

stem="$(basename "$JSON" .json)"; stem="${stem%-update-msa}"
OUT="$PRED_ROOT/$stem"
mkdir -p "$OUT"
echo "[pred] task=${SLURM_ARRAY_TASK_ID} json=$JSON -> $OUT"
exec "$PROTENIX" pred -i "$JSON" -o "$OUT" --use_msa true --seeds 101 "$@"
