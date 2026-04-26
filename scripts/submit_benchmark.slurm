#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/benchmark_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/benchmark_%j.err

set -euo pipefail

# ── Mode switch ──────────────────────────────────────────────────────────
# MODE=fasta : head vs NMP on a FASTA (no ground truth). Uses accelerated NMP.
# MODE=iedb  : head + NMP vs IEDB ground truth from protein_samples parquet.
#              NMP_MODE controls original vs accelerated (default: original).
# MODE=phase_c: evaluate Phase C generated.parquet into schema-aligned
#               immunogenicity / structural tables.
MODE="${MODE:-iedb}"

# ── Shared parameters (override via env) ─────────────────────────────────
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
VARIANT_ID="${VARIANT_ID:-LC1}"
DEVICE="${DEVICE:-cuda}"

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
OUTPUT_DIR="/scratch/gpfs/KAIYIJIANG/zijie/run/benchmark/epi_v2"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/benchmark/epi_v2"
NETMHCIIPAN_BIN="${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan"

ALLELE_TAG="$(echo "${ALLELE}" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

if [ -z "${EPITOPE_CKPT:-}" ]; then
    if [ "${ALLELE}" = "HLA-DRB1*07:01" ]; then
        EPITOPE_CKPT="/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_lite_aug/runs/LC1/seed_42/best.pt"
    elif [ "${ALLELE}" = "HLA-DRB1*04:01" ]; then
        EPITOPE_CKPT="/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head/LC1_drb0401_aug/runs/LC1/seed_42/best.pt"
    else
        echo "ERROR: EPITOPE_CKPT not set and no default is defined for ALLELE=${ALLELE}" >&2
        exit 2
    fi
fi

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
echo "Benchmark / Evaluation"
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

elif [ "${MODE}" = "phase_c" ]; then
    EVAL_MODE="${EVAL_MODE:-all}"
    REFOLD_MODEL="${REFOLD_MODEL:-esmfold}"
    FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.05}"
    PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
    OVERWRITE="${OVERWRITE:-0}"
    NMP_WORKERS="${NMP_WORKERS:-8}"
    NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
    NMP_MAX_LENGTHS_PER_CALL="${NMP_MAX_LENGTHS_PER_CALL:-4}"
    NMP_TIMEOUT="${NMP_TIMEOUT:-1800}"
    STRONG_BINDER_THRESHOLD="${STRONG_BINDER_THRESHOLD:-2.0}"
    TMALIGN_BIN="${TMALIGN_BIN:-TMalign}"

    if [ "${ALLELE}" = "HLA-DRB1*07:01" ]; then
        TEST_SET_PARQUET="${TEST_SET_PARQUET:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/test_proteins_HLA-DRB1_07_01.parquet}"
        PDB_ROOT="${PDB_ROOT:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs/0701}"
    elif [ "${ALLELE}" = "HLA-DRB1*04:01" ]; then
        TEST_SET_PARQUET="${TEST_SET_PARQUET:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/test_proteins_HLA-DRB1_04_01.parquet}"
        PDB_ROOT="${PDB_ROOT:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/pdbs/0401}"
    else
        echo "ERROR: TEST_SET_PARQUET/PDB_ROOT defaults only defined for 0701/0401; set them explicitly" >&2
        exit 2
    fi

    GENERATED_PARQUET="${GENERATED_PARQUET:-}"
    if [ -z "${GENERATED_PARQUET}" ]; then
        echo "ERROR: GENERATED_PARQUET is required for MODE=phase_c" >&2
        exit 2
    fi

    PHASE_C_OUTPUT_ROOT="${PHASE_C_OUTPUT_ROOT:-/scratch/gpfs/KAIYIJIANG/zijie/run/benchmark/if_phase_c}"
    ESMFOLD_CACHE_DIR="${ESMFOLD_CACHE_DIR:-${PHASE_C_OUTPUT_ROOT}/.cache/esmfold/${ALLELE_TAG}}"

    echo "  Eval mode         : ${EVAL_MODE}"
    echo "  Generated parquet : ${GENERATED_PARQUET}"
    echo "  Test-set parquet  : ${TEST_SET_PARQUET}"
    echo "  PDB root          : ${PDB_ROOT}"
    echo "  Refold model      : ${REFOLD_MODEL}"
    echo "  Phase-C output    : ${PHASE_C_OUTPUT_ROOT}"
    echo "  ESMFold cache     : ${ESMFOLD_CACHE_DIR}"
    echo "  NMP workers       : ${NMP_WORKERS}"
    echo "  NMP batch size    : ${NMP_BATCH_SIZE}"
    echo "  NMP lengths/call  : ${NMP_MAX_LENGTHS_PER_CALL}"
    echo "  NMP timeout       : ${NMP_TIMEOUT}s"
    echo "  Strong threshold  : ${STRONG_BINDER_THRESHOLD}%"

    EXTRA_ARGS=()
    if [ -n "${RUN_ID:-}" ]; then
        EXTRA_ARGS+=(--run-id "${RUN_ID}")
    fi
    if [ "${OVERWRITE}" = "1" ]; then
        EXTRA_ARGS+=(--overwrite)
    fi

    python -u "${PROJECT_ROOT}/scripts/evaluate_phase_c.py" \
        --generated-parquet "${GENERATED_PARQUET}" \
        --test-set-parquet "${TEST_SET_PARQUET}" \
        --allele "${ALLELE}" \
        --mode "${EVAL_MODE}" \
        --output-root "${PHASE_C_OUTPUT_ROOT}" \
        --epitope-ckpt "${EPITOPE_CKPT}" \
        --epitope-variant-id "${VARIANT_ID}" \
        --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
        --nmp-batch-size "${NMP_BATCH_SIZE}" \
        --nmp-workers "${NMP_WORKERS}" \
        --nmp-timeout "${NMP_TIMEOUT}" \
        --nmp-max-lengths-per-call "${NMP_MAX_LENGTHS_PER_CALL}" \
        --strong-binder-threshold "${STRONG_BINDER_THRESHOLD}" \
        --pdb-root "${PDB_ROOT}" \
        --refold-model "${REFOLD_MODEL}" \
        --tmalign-bin "${TMALIGN_BIN}" \
        --esmfold-cache-dir "${ESMFOLD_CACHE_DIR}" \
        --device "${DEVICE}" \
        --progress-every "${PROGRESS_EVERY}" \
        --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
        "${EXTRA_ARGS[@]}"

else
    echo "ERROR: unknown MODE='${MODE}' (expected: fasta | iedb | phase_c)"
    exit 1
fi

echo "Benchmark finished."
