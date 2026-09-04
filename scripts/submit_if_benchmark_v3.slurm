#!/bin/bash
#SBATCH --job-name=ifbench-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_benchmark_v3/%x_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_benchmark_v3/%x_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

MODE="${MODE:?set MODE=nmp_probe|mmseqs_probe|head_probe|head_verify|structure_smoke|mmseqs|head|production_task|production_seal}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
CONDA_ENV="${CONDA_ENV:-immune-design}"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/if_benchmark_v3"
if [[ "${MODE}" != "production_task" && "${MODE}" != "production_seal" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:?set an attempt-specific OUTPUT_DIR}"
  OUTPUT_DIR_RESOLVED="$(realpath -m "${OUTPUT_DIR}")"
  AUDIT_PATTERN='^/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set/builds/[^/]+/HLA-DRB1_15_01/audit(/|$)'
  if [[ "${OUTPUT_DIR_RESOLVED}" == *"/if_ready/main"* ]] || \
     [[ ! "${OUTPUT_DIR_RESOLVED}" =~ ${AUDIT_PATTERN} ]]; then
    echo "ERROR: OUTPUT_DIR must be inside one versioned DRB1501 build audit root" >&2
    exit 2
  fi
  OUTPUT_DIR="${OUTPUT_DIR_RESOLVED}"
  mkdir -p "$(dirname "${OUTPUT_DIR}")"
fi
mkdir -p "${LOG_DIR}"

module purge
module load anaconda3/2025.12
conda activate "${CONDA_ENV}"

export PYTHONHASHSEED=42
export LC_ALL=C
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export PYTHONPATH="${PROJECT_ROOT}/inverse_folding/dplm/src:${PROJECT_ROOT}/inverse_folding/dplm/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp/ifbench_v3_${SLURM_JOB_ID}}"
mkdir -p "${TMPDIR}"

RUNNER=(python -u "${PROJECT_ROOT}/scripts/build_if_benchmark_v3.py")

case "${MODE}" in
  nmp_probe)
    NMP_BINARY="${NMP_BINARY:?set NMP_BINARY}"
    NMP_INSTALL_ROOT="${NMP_INSTALL_ROOT:?set NMP_INSTALL_ROOT}"
    "${RUNNER[@]}" probe-nmp \
      --output-dir "${OUTPUT_DIR}" \
      --binary "${NMP_BINARY}" \
      --install-root "${NMP_INSTALL_ROOT}" \
      --allele "${ALLELE:-HLA-DRB1*15:01}"
    ;;
  mmseqs_probe)
    "${RUNNER[@]}" probe-mmseqs \
      --output-dir "${OUTPUT_DIR}" \
      --tool "${MMSEQS_TOOL:?set MMSEQS_TOOL}" \
      --threads "${SLURM_CPUS_PER_TASK:-2}"
    ;;
  head_probe)
    "${RUNNER[@]}" probe-head \
      --output-dir "${OUTPUT_DIR}" \
      --checkpoint "${HEAD_CHECKPOINT:?set HEAD_CHECKPOINT}" \
      --resolved-config "${HEAD_RESOLVED_CONFIG:?set HEAD_RESOLVED_CONFIG}" \
      --run-summary "${HEAD_RUN_SUMMARY:?set HEAD_RUN_SUMMARY}" \
      --model-config-dir "${HEAD_MODEL_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}" \
      --device "${HEAD_DEVICE:-cuda}" \
      --window-batch-size "${HEAD_WINDOW_BATCH_SIZE:-128}"
    ;;
  head_verify)
    "${RUNNER[@]}" verify-head-annotation \
      --manifest "${HEAD_MANIFEST:?set HEAD_MANIFEST}" \
      --checkpoint "${HEAD_CHECKPOINT:?set HEAD_CHECKPOINT}" \
      --resolved-config "${HEAD_RESOLVED_CONFIG:?set HEAD_RESOLVED_CONFIG}" \
      --run-summary "${HEAD_RUN_SUMMARY:?set HEAD_RUN_SUMMARY}" \
      --model-config-dir "${HEAD_MODEL_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}" \
      --cohort-parquet "${COHORT_PARQUET:?set COHORT_PARQUET}" \
      --replay
    ;;
  production_task)
    RESUME_ARGS=()
    if [ "${RESUME:-0}" = "1" ]; then
      RESUME_ARGS+=(--resume)
    fi
    "${RUNNER[@]}" run-production-task \
      --dag "${DAG_MANIFEST:?set DAG_MANIFEST}" \
      --task-table "${TASK_TABLE:?set TASK_TABLE}" \
      --stage "${STAGE:?set STAGE}" \
      --task-index "${SLURM_ARRAY_TASK_ID:?production_task requires a SLURM array}" \
      "${RESUME_ARGS[@]}"
    ;;
  production_seal)
    "${RUNNER[@]}" seal-production-stage \
      --dag "${DAG_MANIFEST:?set DAG_MANIFEST}" \
      --task-table "${TASK_TABLE:?set TASK_TABLE}" \
      --stage "${STAGE:?set STAGE}"
    ;;
  structure_smoke)
    STRUCTURE_SOURCE_MANIFEST="${STRUCTURE_SOURCE_MANIFEST:?set STRUCTURE_SOURCE_MANIFEST}"
    EXTRA=()
    if [ "${REQUIRE_REPEATED_AUTH_ID:-0}" = "1" ]; then
      EXTRA+=(--require-repeated-auth-id)
    fi
    "${RUNNER[@]}" smoke-structure \
      --output-dir "${OUTPUT_DIR}" \
      --source-manifest "${STRUCTURE_SOURCE_MANIFEST}" \
      --min-coverage 0.8 \
      "${EXTRA[@]}"
    ;;
  mmseqs)
    "${RUNNER[@]}" run-mmseqs \
      --tool "${MMSEQS_TOOL:?set MMSEQS_TOOL}" \
      --query-fasta "${QUERY_FASTA:?set QUERY_FASTA}" \
      --reference-fasta "${REFERENCE_FASTA:?set REFERENCE_FASTA}" \
      --output-dir "${OUTPUT_DIR}" \
      --cov-mode "${COV_MODE:?set COV_MODE=0|2}" \
      --coverage 0.8 --min-seq-id 0.3 \
      --threads "${SLURM_CPUS_PER_TASK:-8}"
    ;;
  head)
    HEAD_ARGS=()
    if [ -n "${DIAGNOSTIC_PARQUET:-}" ]; then
      HEAD_ARGS+=(--diagnostic-parquet "${DIAGNOSTIC_PARQUET}")
    fi
    mkdir -p "${OUTPUT_DIR}"
    "${RUNNER[@]}" annotate-head \
      --cohort-parquet "${COHORT_PARQUET:?set COHORT_PARQUET}" \
      "${HEAD_ARGS[@]}" \
      --checkpoint "${HEAD_CHECKPOINT:?set HEAD_CHECKPOINT}" \
      --resolved-config "${HEAD_RESOLVED_CONFIG:?set HEAD_RESOLVED_CONFIG}" \
      --run-summary "${HEAD_RUN_SUMMARY:?set HEAD_RUN_SUMMARY}" \
      --model-config-dir "${HEAD_MODEL_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}" \
      --output-parquet "${OUTPUT_DIR}/head.parquet" \
      --manifest "${OUTPUT_DIR}/head.manifest.json" \
      --device "${HEAD_DEVICE:-cuda}" \
      --window-batch-size "${HEAD_WINDOW_BATCH_SIZE:-1024}"
    ;;
  *)
    echo "ERROR: unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac
