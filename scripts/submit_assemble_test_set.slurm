#!/bin/bash
#SBATCH --job-name=L5-assemble-test-set
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/network/zc1519/logs/if_eval/assemble_test_set_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/if_eval/assemble_test_set_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/if_test_set"
OUTPUT_ROOT="/scratch/network/zc1519/work/immune-design/if_test_set"
LOG_DIR="/scratch/network/zc1519/logs/if_eval"
ALLELE="HLA-DRB1*07:01"
ALLELE_TAG="${ALLELE//[^[:alnum:].-]/_}"

TIER1_JSON="${OUTPUT_ROOT}/tier1_candidates.json"
TIER2_CANDIDATE_FASTA="${DATA_DIR}/tier2_candidates_merged.fasta"
TIER2_HEAD_CKPT="${OUTPUT_ROOT}/_prescreen_head_results_${ALLELE_TAG}.parquet"
TIER2_NMP_CKPT="${OUTPUT_ROOT}/_prescreen_nmp_results_${ALLELE_TAG}.parquet"
TIER3_INPUT="${OUTPUT_ROOT}/uricase_prescreen/uricase_dual_pass_${ALLELE_TAG}.parquet"
PDB_DIR="${OUTPUT_ROOT}/pdbs"
EPITOPE_CKPT="/scratch/network/zc1519/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt"
NETMHCIIPAN_BIN="/home/zc1519/src/Immune-Design/netMHCIIpan-4.3/netMHCIIpan"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${PDB_DIR}"

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
echo "Test Set Assembly Pipeline"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Tier 1       : ${TIER1_JSON}"
echo "  Tier 2 FASTA : ${TIER2_CANDIDATE_FASTA}"
echo "  Tier 2 head  : ${TIER2_HEAD_CKPT}"
echo "  Tier 2 NMP   : ${TIER2_NMP_CKPT}"
echo "  Tier 3       : ${TIER3_INPUT}"
echo "  Allele       : ${ALLELE}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/assemble_if_test_set.py" \
    --tier1-json "${TIER1_JSON}" \
    --tier2-candidate-merged-fasta "${TIER2_CANDIDATE_FASTA}" \
    --tier2-head-ckpt-parquet "${TIER2_HEAD_CKPT}" \
    --tier2-nmp-ckpt-parquet "${TIER2_NMP_CKPT}" \
    --tier2-min-strong-windows 5 \
    --tier3-json "${TIER3_INPUT}" \
    --pdb-dir "${PDB_DIR}" \
    --epitope-ckpt "${EPITOPE_CKPT}" \
    --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
    --output-dir "${OUTPUT_ROOT}" \
    --allele "${ALLELE}"

echo "Test set assembly finished."
