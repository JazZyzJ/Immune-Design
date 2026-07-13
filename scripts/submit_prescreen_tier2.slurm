#!/bin/bash
#SBATCH --job-name=L3-prescreen-tier2
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/prescreen_tier2_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/prescreen_tier2_%j.err

set -euo pipefail
: "${PS1:=}"          # guard: conda activate references $PS1 under `set -u`
export PS1

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set"
OUTPUT_ROOT="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval"

CANDIDATE_FASTA_DIR="${DATA_DIR}/tier2_candidates"
CATH_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/cath_4.3"
CATH_TRAIN_SOURCE="${CATH_DIR}/chain_set.jsonl"
CATH_DOMAIN_LIST="${CATH_DIR}/chain_set.jsonl"
NETMHCIIPAN_BIN="/home/zc1519/src/Immune-Design/netMHCIIpan-4.3/netMHCIIpan"
ALLELE="${ALLELE:-HLA-DRB1*04:01}"

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

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
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

# ── Mode (dual_scorer = legacy; nmp_only = Tier 2 v2, PLAN_DATA_SEL §12) ─────
MODE="${MODE:-dual_scorer}"
NMP_WORKERS="${NMP_WORKERS:-8}"
NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"

# ── Launch ────────────────────────────────────────────────────────────────
if [ "${MODE}" = "nmp_only" ]; then
    # NMP-only screen. Pure CPU (no epitope head / GPU). For the S1 75k scan,
    # submit CPU-only with many cores, e.g.:
    #   MODE=nmp_only NMP_WORKERS=32 NMP_SCREEN_LENGTHS=15 \
    #   CANDIDATE_ID_LIST=.../_prescreen_head_results_<allele>.parquet \
    #   sbatch --partition=cpu --gres=none --cpus-per-task=32 --time=12:00:00 \
    #          scripts/submit_prescreen_tier2.slurm
    # For S2, point CANDIDATE_ID_LIST at the S1-selected parquet and drop
    # NMP_SCREEN_LENGTHS (defaults to all 12-25), add SAMPLE=gaussian.
    NMP_SCREEN_LENGTHS="${NMP_SCREEN_LENGTHS:-}"
    CANDIDATE_ID_LIST="${CANDIDATE_ID_LIST:-${DATA_DIR}/_prescreen_head_results_$(echo "${ALLELE}" | tr '*:' '__' | tr -d ' ').parquet}"
    SAMPLE="${SAMPLE:-none}"
    SAMPLE_N_TARGET="${SAMPLE_N_TARGET:-3000}"
    SAMPLE_N_BINS="${SAMPLE_N_BINS:-20}"
    SAMPLE_MIN_PER_BIN="${SAMPLE_MIN_PER_BIN:-40}"
    SAMPLE_PEAK_TO_TAIL="${SAMPLE_PEAK_TO_TAIL:-3.0}"
    SAMPLE_SEED="${SAMPLE_SEED:-42}"
    # Job-array sharding: SLURM_ARRAY_TASK_ID = shard index; N_SHARDS = array size.
    N_SHARDS="${N_SHARDS:-1}"
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
    python -u "${PROJECT_ROOT}/scripts/prescreen_tier2.py" \
        --selection-mode nmp_only \
        --candidate-merged-fasta "${DATA_DIR}/tier2_candidates_merged.fasta" \
        --candidate-id-list "${CANDIDATE_ID_LIST}" \
        --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
        --output-dir "${OUTPUT_ROOT}" \
        --nmp-screen-lengths "${NMP_SCREEN_LENGTHS}" \
        --nmp-batch-size "${NMP_BATCH_SIZE}" \
        --nmp-timeout 600 \
        --nmp-max-lengths-per-call 4 \
        --nmp-workers "${NMP_WORKERS}" \
        --n-shards "${N_SHARDS}" \
        --shard-index "${SHARD_INDEX}" \
        --resume \
        --sample "${SAMPLE}" \
        --sample-n-target "${SAMPLE_N_TARGET}" \
        --sample-n-bins "${SAMPLE_N_BINS}" \
        --sample-min-per-bin "${SAMPLE_MIN_PER_BIN}" \
        --sample-peak-to-tail "${SAMPLE_PEAK_TO_TAIL}" \
        --sample-seed "${SAMPLE_SEED}" \
        --allele "${ALLELE}"
else
    python -u "${PROJECT_ROOT}/scripts/prescreen_tier2.py" \
        --candidate-merged-fasta "${DATA_DIR}/tier2_candidates_merged.fasta" \
        --cath-train-fasta "${CATH_TRAIN_SOURCE}" \
        --epitope-ckpt "${EPITOPE_CKPT}" \
        --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
        --cath-domain-list "${CATH_DOMAIN_LIST}" \
        --output-dir "${OUTPUT_ROOT}" \
        --nmp-batch-size "${NMP_BATCH_SIZE}" \
        --nmp-timeout 600 \
        --nmp-max-lengths-per-call 4 \
        --nmp-workers "${NMP_WORKERS}" \
        --head-prefilter-topk 3000 \
        --resume \
        --skip-cath-diversity \
        --max-per-topology 2 \
        --target-total 50 \
        --allele "${ALLELE}"
fi
echo "Tier 2 pre-screening finished."
