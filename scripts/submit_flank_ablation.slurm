#!/bin/bash
#SBATCH --job-name=flank-ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/network/zc1519/run/flank-ablation-%j.out
#SBATCH --error=/scratch/network/zc1519/run/flank-ablation-%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
RUN_ROOT="/scratch/network/zc1519/run"
CHECKPOINT="/scratch/network/zc1519/run/best.pt"
OUTPUT_JSON="${RUN_ROOT}/ablation/flank_ablation_${SLURM_JOB_ID}.json"

mkdir -p "${RUN_ROOT}/ablation"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/network/zc1519/model_cache/torch"
export WANDB_MODE=offline

echo "============================================================"
echo "Flank Ablation Test (SLURM)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Checkpoint   : ${CHECKPOINT}"
echo "  Output json  : ${OUTPUT_JSON}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

python "${PROJECT_ROOT}/scripts/flank_ablation_test.py" \
  --checkpoint "${CHECKPOINT}" \
  --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
  --profile strict \
  --split val \
  --device cuda \
  --output-json "${OUTPUT_JSON}"

echo "Done: ${OUTPUT_JSON}"

