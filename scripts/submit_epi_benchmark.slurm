#!/bin/bash
#SBATCH --job-name=bench-head-nmp
#SBATCH --constraint="a100&gpu40&pcie"
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi/epi_bench_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi/epi_bench_%j.err

set -euo pipefail

# ── Parameters (override via environment) ────────────────────────────────
# Example: ALLELE="HLA-DRB1*04:01" EPITOPE_CKPT="/.../drb0401/best.pt" sbatch scripts/submit_benchmark.slurm
INPUT_FASTA="${INPUT_FASTA:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/filtered_uricases-diverse.fasta}"
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
NMP_WORKERS="${NMP_WORKERS:-8}"
NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
VARIANT_ID="${VARIANT_ID:-LC1}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${EPITOPE_CKPT:-}" ]]; then
    if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
        EPITOPE_CKPT="/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt"
    elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
        EPITOPE_CKPT="/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_a1_res03_drb0401_seed42_cv5_fold0/runs/LC1/seed_42/best.pt"
    elif [[ "${ALLELE}" == "HLA-DRB1*15:01" ]]; then
        EPITOPE_CKPT="/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/cnn_himp_a1_res03_drb1501_seed42/runs/LC1/seed_42/best.pt"
    else
        echo "ERROR: EPITOPE_CKPT not set and no a1res03 default exists for ALLELE=${ALLELE}" >&2
        exit 2
    fi
fi

# ── Paths ────────────────────────────────────────────────────────────────���
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
OUTPUT_DIR="/scratch/gpfs/KAIYIJIANG/zijie/run/benchmark/epi/"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi/"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

# Derive allele tag for output filename (replace non-alnum with _)
ALLELE_TAG="$(echo "${ALLELE}" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"
# Derive a short input tag from fasta filename
INPUT_TAG="$(basename "${INPUT_FASTA}" .fasta)"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
export PS1="${PS1:-}"
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export WANDB_MODE=offline

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Benchmark: Epitope Head vs NetMHCIIpan"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  GPUs         : ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs         : ${SLURM_CPUS_PER_TASK:-16}"
echo "  Input FASTA  : ${INPUT_FASTA}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  NMP binary   : ${NETMHCIIPAN_BIN}"
echo "  Allele       : ${ALLELE}"
echo "  Variant ID   : ${VARIANT_ID}"
echo "  Device       : ${DEVICE}"
echo "  NMP workers  : ${NMP_WORKERS}"
echo "  NMP batch    : ${NMP_BATCH_SIZE}"
echo "  Output tag   : ${INPUT_TAG}_${ALLELE_TAG}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python -u "${PROJECT_ROOT}/scripts/benchmark_head_vs_nmp.py" \
    --input-fasta "${INPUT_FASTA}" \
    --epitope-ckpt "${EPITOPE_CKPT}" \
    --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
    --allele "${ALLELE}" \
    --output-json "${OUTPUT_DIR}/benchmark_${INPUT_TAG}_${ALLELE_TAG}.json" \
    --device "${DEVICE}" \
    --variant-id "${VARIANT_ID}" \
    --nmp-batch-size "${NMP_BATCH_SIZE}" \
    --nmp-workers "${NMP_WORKERS}" \
    --nmp-max-lengths-per-call 4 \
    --nmp-timeout 600

echo "Benchmark finished."
