#!/bin/bash
#SBATCH --job-name=M3-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/logs/inverse_folding/M3_sweep_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/inverse_folding/M3_sweep_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design"
DPLM_ROOT="${PROJECT_ROOT}/inverse_folding/dplm"
OUTPUT_ROOT="/scratch/network/zc1519/run/inverse_folding"
LOG_DIR="/scratch/network/zc1519/logs/inverse_folding"

# ── FILL IN: paths ───────────────────────────────────────────────────────
DPLM_EXPERIMENT_PATH="${OUTPUT_ROOT}/dplm_v1_adapter/<FILL_IN_RUN_ID>"
DPLM_CKPT="${DPLM_EXPERIMENT_PATH}/checkpoints/<FILL_IN_BEST_CKPT>"
EPITOPE_CKPT="<FILL_IN: path to epitope head best.pt>"
TEST_PDB_DIR="<FILL_IN: path to test PDB directory>"

# ── Test set sampling ────────────────────────────────────────────────────
# Assembled parquet from L5 (tier-aware sampling for efficiency)
TEST_SET_PARQUET="${DATA_DIR}/if_test_set/test_proteins_HLA-DRB1_07_01.parquet"
TIER2_SAMPLE=40          # random sample from ~3000 tier 2 proteins
TIER3_SAMPLE=15          # random sample from ~126 tier 3 proteins
# Tier 1 (15 gold standard) always included in full

GUIDANCE_OUTPUT="${OUTPUT_ROOT}/guidance/sweep_seed42"

mkdir -p "${GUIDANCE_OUTPUT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/network/zc1519/model_cache/torch"
export HF_HOME="/scratch/network/zc1519/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "M3: Classifier Guidance Eta Sweep"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  DPLM ckpt    : ${DPLM_CKPT}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  Test PDBs    : ${TEST_PDB_DIR}"
echo "  Test parquet : ${TEST_SET_PARQUET}"
echo "  Tier2 sample : ${TIER2_SAMPLE}"
echo "  Tier3 sample : ${TIER3_SAMPLE}"
echo "  Output       : ${GUIDANCE_OUTPUT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

python "${PROJECT_ROOT}/scripts/run_if_guidance_sweep.py" \
    --dplm-root "${DPLM_ROOT}" \
    --dplm-experiment-path "${DPLM_EXPERIMENT_PATH}" \
    --dplm-ckpt "${DPLM_CKPT}" \
    --epitope-config-dir "${PROJECT_ROOT}/epitope_head/configs" \
    --epitope-ckpt "${EPITOPE_CKPT}" \
    --epitope-variant B0 \
    --test-pdb-dir "${TEST_PDB_DIR}" \
    --test-set-parquet "${TEST_SET_PARQUET}" \
    --tier2-sample "${TIER2_SAMPLE}" \
    --tier3-sample "${TIER3_SAMPLE}" \
    --output-dir "${GUIDANCE_OUTPUT}" \
    --num-candidates 8 \
    --seed 42 \
    --max-iter 10 \
    --temperature 1.0 \
    --device cuda

echo "M3 guidance sweep finished."
