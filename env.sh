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
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export WANDB_MODE=offline

export IMD_DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/manifests"
export IMD_RUN_DIR="/scratch/gpfs/KAIYIJIANG/zijie/run"
export IMD_ABLATION_ROOT="/scratch/gpfs/KAIYIJIANG/zijie/run/ablation/encoder_v2"

# DSSP lives in a dedicated conda env (`dssp-tool`) because it cannot coexist
# with `immune-design`. Scripts that use DSSP (e.g. scripts/analysis/tier1_structural_analysis.py)
# accept --dssp-bin; export IMD_DSSP_BIN here and pass --dssp-bin "$IMD_DSSP_BIN".
# Uncomment + set the absolute path on first use:
# export IMD_DSSP_BIN="$(conda run -n dssp-tool which mkdssp 2>/dev/null)"

cd "${PROJECT_ROOT}"
echo "Immune-Design env ready.  PROJECT_ROOT=${PROJECT_ROOT}"
