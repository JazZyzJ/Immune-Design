#!/usr/bin/env bash
# Source this file to set up the Immune-Design environment.
# Usage:  source env.sh
#         python scripts/analysis/diagnose_e1.py --checkpoint ... --device cpu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/network/zc1519/model_cache/torch"
export WANDB_MODE=offline

export IMD_DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
export IMD_RUN_DIR="/scratch/network/zc1519/run"
export IMD_ABLATION_ROOT="/scratch/network/zc1519/run/ablation/encoder_v2"

cd "${PROJECT_ROOT}"
echo "Immune-Design env ready.  PROJECT_ROOT=${PROJECT_ROOT}"
