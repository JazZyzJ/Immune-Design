#!/bin/bash
# Packed Protenix pred for submit-capped partitions (ailab): ONE small array (<= submit cap),
# each task strides through a shared ready-list and runs its slice serially on its 1 GPU.
# T tasks x 1 GPU = T-way parallelism from a single array job. Idempotent (skips finished),
# fault-tolerant (a failed design doesn't kill the task).
# Usage: sbatch --array=0-$((T-1)) submit_packed_pred.slurm <READY_LIST> <PRED_ROOT> <T> [protenix flags]
#SBATCH --job-name=packed_pred
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -uo pipefail
LIST="${1:?READY_LIST}"; PRED_ROOT="${2:?PRED_ROOT}"; NTASKS="${3:?NTASKS}"; shift 3
PROTENIX="${PROTENIX_RUN:-/scratch/gpfs/KAIYIJIANG/tools/protenix/bin/protenix-run}"
mkdir -p "$PRED_ROOT"
mapfile -t JSONS < "$LIST"
i="${SLURM_ARRAY_TASK_ID}"; n="${#JSONS[@]}"
echo "[packed t$i/$NTASKS] $(date +%H:%M:%S) striding over $n designs"
done_ct=0; skip_ct=0; fail_ct=0
for ((k=i; k<n; k+=NTASKS)); do
  JSON="${JSONS[$k]}"
  [ -n "${JSON:-}" ] && [ -f "$JSON" ] || continue
  stem="$(basename "$JSON" .json)"; stem="${stem%-update-msa}"
  OUT="$PRED_ROOT/$stem"
  if compgen -G "$OUT/*/seed_*/predictions/${stem}_sample_0.cif" > /dev/null 2>&1; then
    skip_ct=$((skip_ct+1)); continue
  fi
  mkdir -p "$OUT"
  if "$PROTENIX" pred -i "$JSON" -o "$OUT" --use_msa true --seeds 101 "$@"; then
    done_ct=$((done_ct+1))
  else
    fail_ct=$((fail_ct+1)); echo "[packed t$i] FAILED $stem"
  fi
done
echo "[packed t$i] $(date +%H:%M:%S) DONE  done=$done_ct skip=$skip_ct fail=$fail_ct"
