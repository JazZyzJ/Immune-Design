#!/bin/bash
#SBATCH --job-name=L3-prescreen-tier2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/zc1519/logs/if_eval/prescreen_tier2_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/if_eval/prescreen_tier2_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/if_test_set"
OUTPUT_ROOT="/scratch/network/zc1519/work/immune-design/if_test_set"
LOG_DIR="/scratch/network/zc1519/logs/if_eval"

CANDIDATE_FASTA_DIR="${DATA_DIR}/tier2_candidates"
CATH_DIR="/scratch/network/zc1519/work/immune-design/cath_4.3"
CATH_TRAIN_SOURCE="${CATH_DIR}/chain_set.jsonl"
CATH_DOMAIN_LIST="${CATH_DIR}/chain_set.jsonl"
EPITOPE_CKPT="/scratch/network/zc1519/run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt"
NETMHCIIPAN_BIN="/home/zc1519/src/Immune-Design/netMHCIIpan-4.3/netMHCIIpan"
ALLELE="HLA-DRB1*04:01"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

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
echo "Tier 2 Pre-screening Pipeline"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Allele       : ${ALLELE}"
echo "  Candidates   : ${CANDIDATE_FASTA_DIR}"
echo "  CATH train   : ${CATH_TRAIN_SOURCE}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  NMP binary   : ${NETMHCIIPAN_BIN}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python -u "${PROJECT_ROOT}/scripts/prescreen_tier2.py" \
    --candidate-merged-fasta "${DATA_DIR}/tier2_candidates_merged.fasta" \
    --cath-train-fasta "${CATH_TRAIN_SOURCE}" \
    --epitope-ckpt "${EPITOPE_CKPT}" \
    --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
    --cath-domain-list "${CATH_DOMAIN_LIST}" \
    --output-dir "${OUTPUT_ROOT}" \
    --nmp-batch-size 8 \
    --nmp-timeout 600 \
    --nmp-max-lengths-per-call 4 \
    --nmp-workers 8 \
    --head-prefilter-topk 3000 \
    --resume \
    --skip-cath-diversity \
    --max-per-topology 2 \
    --target-total 50 \
    --allele "${ALLELE}"
echo "Tier 2 pre-screening finished."
