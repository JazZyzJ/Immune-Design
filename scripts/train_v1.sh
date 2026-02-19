#!/usr/bin/env bash
# Epitope Head v1 — Reproducible training launcher
#
# Usage:
#   # Local smoke test (no ESM, CPU, quick):
#   bash scripts/train_v1.sh --smoke --mock-encoder
#
#   # Local full strict run on CPU:
#   bash scripts/train_v1.sh --profile strict
#
#   # Cluster GPU run:
#   bash scripts/train_v1.sh --device cuda --profile strict
#
#   # Pass any extra args to train_v1.py:
#   bash scripts/train_v1.sh --device cuda --profile balanced --run-dir /scratch/runs/exp01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Reproducibility environment ──────────────────────────────────────────
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Ensure project root is on PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Log banner ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Epitope Head v1 Training"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA         : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Args         : $*"
echo "============================================================"

# ── Launch training ──────────────────────────────────────────────────────
python "${SCRIPT_DIR}/train_v1.py" \
    --config-dir "${PROJECT_ROOT}/epitope_head/configs" \
    "$@"
