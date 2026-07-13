#!/bin/bash
# Batch ESMFold2 tetramer+ligand+MSA prediction as a SLURM array (one variant per task).
# Each task looks up its id from IDLIST, reads the local a3m (MSADIR/<id>.a3m), takes the query
# sequence from that a3m (guarantees the ProteinInput sequence matches the MSA), and runs the
# standalone predictor. Extra predictor flags (e.g. --ligand-smiles, --n-copies) pass through.
#
# Usage:
#   sbatch --array=0-N%8 --output=<logs>/esmf2a_%A_%a.out --error=<logs>/esmf2a_%A_%a.err \
#     scripts/submit_tetramer_esmfold2_array.slurm <IDLIST> <MSADIR> <OUTDIR> \
#       --ligand-smiles 'O=C1NC(=O)C2=C(N1)NC(=O)N2' --n-copies 4 --num-loops 8 --num-sampling-steps 100
#
#SBATCH --job-name=esmf2_arr
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -eo pipefail
export HF_HOME="${HF_HOME:-/scratch/gpfs/KAIYIJIANG/zijie/model_cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MKL_THREADING_LAYER=GNU
export PS1="${PS1:-}"
ESMF2_PY="${ESMF2_PY:-/home/zc1519/.conda/envs/esmfold2/bin/python}"

IDLIST="${1:?IDLIST}"; MSADIR="${2:?MSADIR}"; OUTDIR="${3:?OUTDIR}"; shift 3
ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDLIST")"
[ -n "$ID" ] || { echo "no id for array task ${SLURM_ARRAY_TASK_ID} in $IDLIST" >&2; exit 2; }
A3M="$MSADIR/$ID.a3m"
[ -f "$A3M" ] || { echo "missing a3m for $ID: $A3M" >&2; exit 3; }
SEQ="$(awk '!/^>/ && NF {print; exit}' "$A3M")"
[ -n "$SEQ" ] || { echo "empty query seq in $A3M" >&2; exit 4; }

echo "[esmf2_arr] task=${SLURM_ARRAY_TASK_ID} id=$ID len=${#SEQ} a3m=$A3M"
exec "$ESMF2_PY" /home/zc1519/src/Immune-Design/scripts/predict_tetramer_esmfold2.py \
  --sequence "$SEQ" --name "tetra_$ID" --out-dir "$OUTDIR" --msa-a3m "$A3M" "$@"
