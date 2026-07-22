#!/bin/bash
# Protenix tetramer prediction array over the MSA-ready list from build_protenix_jsons.py.
#
# This compatibility entry point delegates to submit_packed_pred.slurm. A full-size array retains
# one design per task; a smaller array automatically strides over the list and keeps one persistent
# Protenix model per GPU task. For large runs, prefer a small array (for example 0-21) so checkpoint
# loading is amortized across many designs.
#
# H200 default:
#   sbatch --array=0-21 --output=<logs>/tetra_%A_%a.out --error=<logs>/tetra_%A_%a.err \
#     scripts/submit_tetramer_predict.slurm <READY_LIST> <PRED_ROOT> [protenix flags]
# PLI H100:
#   sbatch --account=pli_x --qos=pli-low --partition=pli --gres=gpu:h100:1 --array=0-21 \
#     --output=<logs>/tetra_%A_%a.out --error=<logs>/tetra_%A_%a.err \
#     scripts/submit_tetramer_predict.slurm <READY_LIST> <PRED_ROOT> [protenix flags]
#SBATCH --job-name=tetra_pred
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

LIST="${1:?READY_LIST}"
PRED_ROOT="${2:?PRED_ROOT}"
shift 2

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
PACKED_PRED_SCRIPT="${PACKED_PRED_SCRIPT:-${PROJECT_ROOT}/scripts/submit_packed_pred.slurm}"
NTASKS="${PACKED_TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

exec bash "${PACKED_PRED_SCRIPT}" \
  "${LIST}" "${PRED_ROOT}" "${NTASKS}" "$@"
