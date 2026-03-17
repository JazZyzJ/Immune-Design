#!/bin/bash
#SBATCH --job-name=cnn-strict
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/run/epitope_head/cnn-train-%j.out
#SBATCH --error=/scratch/network/zc1519/run/epitope_head/cnn-train-%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
OUTPUT_ROOT="/scratch/network/zc1519/run/epitope_head/cnn_strict_full"

mkdir -p "${OUTPUT_ROOT}"

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

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Epitope Head v2 CNN Training (SLURM)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Override config (optional, highest priority) ─────────────────────────
# Set OVERRIDE_CONFIG to a YAML file path to override any train config key.
# Example: OVERRIDE_CONFIG=/path/to/my_overrides.yaml sbatch scripts/submit_train_v2_cnn.slurm
OVERRIDE_FLAG=()
if [[ -n "${OVERRIDE_CONFIG:-}" ]]; then
  OVERRIDE_FLAG=(--override-config "${OVERRIDE_CONFIG}")
  echo "  Override cfg : ${OVERRIDE_CONFIG}"
fi

# ── Launch CNN training (LC1 variant, seed 42) ────────────────────────────
python "${PROJECT_ROOT}/scripts/train_v2_ablation.py" \
  --variant-id "LC1" \
  --seed 42 \
  --device cuda \
  --profile strict \
  --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
  "${OVERRIDE_FLAG[@]+"${OVERRIDE_FLAG[@]}"}" \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --wandb \
  --wandb-name "LC1_strict_full_seed42" \
  --wandb-entity "chenzijie040528-zhejiang-university" \
  --wandb-project "Immune-Design"

echo "CNN training job finished."

