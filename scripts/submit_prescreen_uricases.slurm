#!/bin/bash
#SBATCH --job-name=L-prescreen-uricase
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/logs/if_eval/prescreen_uricase_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/if_eval/prescreen_uricase_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/if_test_set"
CATH_DIR="/scratch/network/zc1519/work/immune-design/cath_4.3"
LOG_DIR="/scratch/network/zc1519/logs/if_eval"

INPUT_FASTA="${DATA_DIR}/uricases/filtered_uricases.fasta"
CATH_TRAIN_SOURCE="${CATH_DIR}/chain_set.jsonl"
EPITOPE_CKPT="/scratch/network/zc1519/run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt"
NETMHCIIPAN_BIN="/home/zc1519/src/Immune-Design/netMHCIIpan-4.3/netMHCIIpan"
OUTPUT_DIR="${DATA_DIR}/uricase_prescreen"

ALLELE="HLA-DRB1*04:01"
ALLELE_TAG="${ALLELE//[^[:alnum:].-]/_}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

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
echo "Uricase Pre-screening for Tier 3"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Input FASTA  : ${INPUT_FASTA}"
echo "  N sequences  : $(grep -c '^>' ${INPUT_FASTA})"
echo "  CATH train   : ${CATH_TRAIN_SOURCE}"
echo "  Allele       : ${ALLELE}"
echo "  Allele tag   : ${ALLELE_TAG}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  NMP binary   : ${NETMHCIIPAN_BIN}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python -u "${PROJECT_ROOT}/scripts/prescreen_uricases.py" \
    --input-fasta "${INPUT_FASTA}" \
    --cath-train-fasta "${CATH_TRAIN_SOURCE}" \
    --epitope-ckpt "${EPITOPE_CKPT}" \
    --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --nmp-batch-size 8 \
    --nmp-timeout 600 \
    --nmp-max-lengths-per-call 4 \
    --nmp-workers 8 \
    --head-prefilter-topk 3000 \
    --resume \
    --min-strong-windows 3 \
    --allele "${ALLELE}"

echo "Uricase pre-screening finished."
