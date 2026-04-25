#!/bin/bash
#SBATCH --job-name=bench-head-nmp
#SBATCH --partition=mig
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi/epi_bench_IEDB_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi/epi_bench_IEDB_%j.err

set -euo pipefail

# ── Mode switch ──────────────────────────────────────────────────────────
# MODE=fasta : head vs NMP on a FASTA (no ground truth). Uses accelerated NMP.
# MODE=iedb  : head + NMP vs IEDB ground truth from protein_samples parquet.
#              NMP_MODE controls original vs accelerated (default: original).
MODE="${MODE:-iedb}"

# ── Shared parameters (override via env) ─────────────────────────────────
EPITOPE_CKPT="${EPITOPE_CKPT:-/scratch/gpfs/KAIYIJIANG/zijie/run/ablation/cnn_enhance_v2/runs/LC1/seed_42/best.pt}"
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
VARIANT_ID="${VARIANT_ID:-LC1}"
DEVICE="${DEVICE:-cuda}"

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
OUTPUT_DIR="/scratch/gpfs/KAIYIJIANG/zijie/run/benchmark/epi"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

ALLELE_TAG="$(echo "${ALLELE}" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

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
echo "  MODE         : ${MODE}"
echo "  Epitope ckpt : ${EPITOPE_CKPT}"
echo "  NMP binary   : ${NETMHCIIPAN_BIN}"
echo "  Allele       : ${ALLELE}"
echo "  Variant ID   : ${VARIANT_ID}"
echo "  Device       : ${DEVICE}"
echo "  Python       : $(python --version 2>&1)"
echo "  PyTorch      : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail   : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
if [ "${MODE}" = "fasta" ]; then
    # FASTA mode: head vs NMP, no ground truth. Accelerated NMP by default.
    INPUT_FASTA="${INPUT_FASTA:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/uricases/filtered_uricases-diverse.fasta}"
    NMP_WORKERS="${NMP_WORKERS:-8}"
    NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
    INPUT_TAG="$(basename "${INPUT_FASTA}" .fasta)"

    echo "  Input FASTA  : ${INPUT_FASTA}"
    echo "  NMP workers  : ${NMP_WORKERS}"
    echo "  NMP batch    : ${NMP_BATCH_SIZE}"

    python -u "${PROJECT_ROOT}/scripts/benchmark_head_vs_nmp.py" \
        --input-fasta "${INPUT_FASTA}" \
        --epitope-ckpt "${EPITOPE_CKPT}" \
        --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
        --allele "${ALLELE}" \
        --output-json "${OUTPUT_DIR}/benchmark_${INPUT_TAG}_${ALLELE_TAG}_val_testFP.json" \
        --device "${DEVICE}" \
        --variant-id "${VARIANT_ID}" \
        --nmp-batch-size "${NMP_BATCH_SIZE}" \
        --nmp-workers "${NMP_WORKERS}" \
        --nmp-max-lengths-per-call 4 \
        --nmp-timeout 600

elif [ "${MODE}" = "iedb" ]; then
    # IEDB mode: head + NMP vs ground truth. Original NMP (no acceleration) by default.
    PROFILE="${PROFILE:-strict}"
    MANIFEST_ROOT="${MANIFEST_ROOT:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/manifests}"
    ALLELE_SUBDIR="${ALLELE_SUBDIR:-}"  # e.g. drb0401 (leave empty for default 0701 manifest)
    NMP_MODE_ARG="${NMP_MODE:-original}"
    NEAR_MISS_ANALYSIS="${NEAR_MISS_ANALYSIS:-0}"
    # Windows cache (write/read). Set WINDOWS_CACHE to write a cache after
    # inference; set RESUME_FROM_CACHE to skip steps 2/3 entirely. Mutually
    # exclusive — the Python script enforces this.
    WINDOWS_CACHE="${WINDOWS_CACHE:-}"
    RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-}"
    # Accelerated-mode knobs (ignored by the Python script when --nmp-mode=original)
    NMP_WORKERS="${NMP_WORKERS:-8}"
    NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
    NMP_MAX_LENGTHS_PER_CALL="${NMP_MAX_LENGTHS_PER_CALL:-4}"
    NMP_TIMEOUT="${NMP_TIMEOUT:-1800}"

    if [ -n "${ALLELE_SUBDIR}" ]; then
        BASE="${MANIFEST_ROOT}/${ALLELE_SUBDIR}"
    else
        BASE="${MANIFEST_ROOT}"
    fi
    PROTEIN_SAMPLES_PARQUET="${PROTEIN_SAMPLES_PARQUET:-${BASE}/protein_samples_${PROFILE}.parquet}"
    TEST_IDS="${TEST_IDS:-${BASE}/splits/${PROFILE}/val_ids.txt}"

    echo "  Profile            : ${PROFILE}"
    echo "  Parquet            : ${PROTEIN_SAMPLES_PARQUET}"
    echo "  Test IDs           : ${TEST_IDS}"
    echo "  NMP_MODE           : ${NMP_MODE_ARG}"
    echo "  Near-miss analysis : ${NEAR_MISS_ANALYSIS}"
    echo "  Windows cache      : ${WINDOWS_CACHE:-(disabled)}"
    echo "  Resume from cache  : ${RESUME_FROM_CACHE:-(no)}"
    if [ "${NMP_MODE_ARG}" = "accelerated" ]; then
        echo "  NMP workers        : ${NMP_WORKERS}"
        echo "  NMP batch size     : ${NMP_BATCH_SIZE}"
        echo "  NMP lengths/call   : ${NMP_MAX_LENGTHS_PER_CALL}"
    fi
    echo "  NMP timeout        : ${NMP_TIMEOUT}s"

    RESULT_JSON="${OUTPUT_DIR}/benchmark_iedb_${PROFILE}_${ALLELE_TAG}_${NMP_MODE_ARG}.json"
    EXTRA_ARGS=()
    if [ "${NEAR_MISS_ANALYSIS}" = "1" ]; then
        RESULT_JSON="${OUTPUT_DIR}/benchmark_iedb_${PROFILE}_${ALLELE_TAG}_${NMP_MODE_ARG}_near_miss.json"
        EXTRA_ARGS+=(--near-miss-analysis)
    fi
    if [ -n "${WINDOWS_CACHE}" ]; then
        EXTRA_ARGS+=(--windows-cache "${WINDOWS_CACHE}")
    fi
    if [ -n "${RESUME_FROM_CACHE}" ]; then
        EXTRA_ARGS+=(--resume-from-cache "${RESUME_FROM_CACHE}")
    fi

    python -u "${PROJECT_ROOT}/scripts/benchmark_iedb_test.py" \
        --protein-samples-parquet "${PROTEIN_SAMPLES_PARQUET}" \
        --test-ids "${TEST_IDS}" \
        --epitope-ckpt "${EPITOPE_CKPT}" \
        --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
        --allele "${ALLELE}" \
        --output-json "${RESULT_JSON}" \
        --nmp-mode "${NMP_MODE_ARG}" \
        --device "${DEVICE}" \
        --variant-id "${VARIANT_ID}" \
        --nmp-batch-size "${NMP_BATCH_SIZE}" \
        --nmp-max-lengths-per-call "${NMP_MAX_LENGTHS_PER_CALL}" \
        --nmp-workers "${NMP_WORKERS}" \
        --nmp-timeout "${NMP_TIMEOUT}" \
        "${EXTRA_ARGS[@]}"

    python -u "${PROJECT_ROOT}/scripts/plot_benchmark_score_histograms.py" \
        --input-json "${RESULT_JSON}"

else
    echo "ERROR: unknown MODE='${MODE}' (expected: fasta | iedb)"
    exit 1
fi

echo "Benchmark finished."
