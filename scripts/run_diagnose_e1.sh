#!/usr/bin/env bash
# Convenience wrapper for diagnose_e1.py with correct environment and data paths.
#
# Usage (example):
#   bash scripts/run_diagnose_e1.sh \
#     --checkpoint /scratch/network/zc1519/run/ablation/encoder_v2/runs/E1/seed_42/best.pt \
#     --encoder-id E1 \
#     --device cpu
#
# Any extra args are passed through to diagnose_e1.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load shared environment (modules, conda, PYTHONPATH, LD_PRELOAD, TORCH_HOME, etc.)
source "${PROJECT_ROOT}/env.sh"

# Ensure data dir is passed (defaults to IMD_DATA_DIR if set)
DATA_DIR="${IMD_DATA_DIR:-/scratch/network/zc1519/work/immune-design/manifests}"

cd "${PROJECT_ROOT}"
python scripts/analysis/diagnose_e1.py \
  --data-dir "${DATA_DIR}" \
  "$@"

