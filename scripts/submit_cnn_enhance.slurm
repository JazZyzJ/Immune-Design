#!/bin/bash
#SBATCH --job-name=cnn-enhance
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/run/cnn-enhance-%j.out
#SBATCH --error=/scratch/network/zc1519/run/cnn-enhance-%j.err

set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
ABLATION_ROOT="/scratch/network/zc1519/run/ablation/cnn_enhance_v2"

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
echo "Module I: CNN Enhancement Ablation (SLURM)"
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

# ── Variant matrix (B0/L1/C1/LC1 × 3 seeds) ─────────────────────────────
#
# | ID  | Encoder       | Loss          | Description                |
# |-----|---------------|---------------|----------------------------|
# | B0  | dilated_cnn   | infonce       | Control (E1 baseline)      |
# | L1  | dilated_cnn   | mixed_margin  | + margin hard mining       |
# | C1  | multiscale_cnn| infonce       | + multi-branch CNN         |
# | LC1 | multiscale_cnn| mixed_margin  | Both combined              |
#
# Monitor metric: pp_ap (Stage-I, AP-centric ranking)
# Output root:    cnn_enhance_v2/

VARIANTS=(LC1)
SEEDS=(42)

FAIL_COUNT=0
TOTAL=$((${#VARIANTS[@]} * ${#SEEDS[@]}))

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> Starting variant=${variant}, seed=${seed}"
    if python "${PROJECT_ROOT}/scripts/train_v2_ablation.py" \
      --variant-id "${variant}" \
      --seed "${seed}" \
      --device cuda \
      --profile strict \
      --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
      --data-dir "${DATA_DIR}" \
      --output-root "${ABLATION_ROOT}" \
      --wandb \
      --wandb-name "LC1_lite_seed${seed}" \
      --wandb-entity "chenzijie040528-zhejiang-university" \
      --wandb-project "Immune-Design"; then
      echo "<<< Finished variant=${variant}, seed=${seed}"
    else
      echo "!!! FAILED variant=${variant}, seed=${seed} (exit=$?)"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "------------------------------------------------------------"
  done
done
echo "Training grid complete. Failures: ${FAIL_COUNT}/${TOTAL}"

echo "All Module I CNN enhancement runs finished."
