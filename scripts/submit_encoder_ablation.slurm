#!/bin/bash
#SBATCH --job-name=enc-ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/run/ablation-%j.out
#SBATCH --error=/scratch/network/zc1519/run/ablation-%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
ABLATION_ROOT="/scratch/network/zc1519/run/ablation/encoder_v2"

mkdir -p "${ABLATION_ROOT}"

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
echo "Encoder Ablation v2 (SLURM)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Ablation root: ${ABLATION_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── 1. Encoder ablation grid (E0/E1/E2 × seeds) ───────────────────────────
ENCODERS=(E0 E1 E2)
SEEDS=(42 43 44)

FAIL_COUNT=0
for enc in "${ENCODERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> Starting ablation run encoder=${enc}, seed=${seed}"
    if python "${PROJECT_ROOT}/scripts/train_v2_ablation.py" \
      --encoder-id "${enc}" \
      --seed "${seed}" \
      --device cuda \
      --profile strict \
      --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
      --data-dir "${DATA_DIR}" \
      --output-root "${ABLATION_ROOT}" \
      --wandb \
      --wandb-entity "chenzijie040528-zhejiang-university" \
      --wandb-project "Immune-Design"; then
      echo "<<< Finished ablation run encoder=${enc}, seed=${seed}"
    else
      echo "!!! FAILED ablation run encoder=${enc}, seed=${seed} (exit=$?)"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "------------------------------------------------------------"
  done
done
echo "Training grid complete. Failures: ${FAIL_COUNT}/9"

# ── 2. Evaluation: ranking + mutation scan + latency ──────────────────────
echo ">>> Starting encoder ablation evaluation"
python "${PROJECT_ROOT}/scripts/eval_encoder_ablation.py" \
  --output-root "${ABLATION_ROOT}" \
  --encoder-ids E0 E1 E2 \
  --profile strict \
  --data-dir "${DATA_DIR}" \
  --device cuda \
  --mutation-scan \
  --latency-bench
echo "<<< Evaluation complete"

echo "All encoder ablation runs and evaluation finished."

