#!/bin/bash
#SBATCH --job-name=N1-filter
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding/N1_filter_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding/N1_filter_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
OUTPUT_ROOT="/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/inverse_folding"

# M3 sweep output (source for Level 1 re-selection)
M3_OUTPUT="${OUTPUT_ROOT}/guidance/sweep_seed42"
# Level 1 filter output
L1_OUTPUT="${OUTPUT_ROOT}/baselines/level1_filter/sweep_seed42"

mkdir -p "${L1_OUTPUT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export WANDB_MODE=offline

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "N1: Level 1 Post-Hoc Filter Baseline"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  M3 source    : ${M3_OUTPUT}"
echo "  Output       : ${L1_OUTPUT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/run_if_level1_filter.py" \
    --m3-output-dir "${M3_OUTPUT}" \
    --output-dir "${L1_OUTPUT}"

echo "N1 Level 1 filter finished."
