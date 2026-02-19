#!/usr/bin/env bash
# Epitope Head v1 — inference launcher
#
# Usage:
#   bash scripts/infer_v1.sh \
#     --checkpoint outputs/runs/<run>/best.pt \
#     --input-fasta data/infer_targets.fasta \
#     --device cuda

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "Epitope Head v1 Inference"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA         : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Args         : $*"
echo "============================================================"

python "${SCRIPT_DIR}/infer_v1.py" \
    --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
    "$@"

