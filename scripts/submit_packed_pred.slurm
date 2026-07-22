#!/bin/bash
# Persistent Protenix tetramer prediction for submit-capped partitions.
#
# Each array task stages its strided JSON subset and invokes protenix-run ONCE with that
# directory. Protenix v2 constructs and loads the 464M model once, then resets the requested
# seed for every JSON, preserving the previous one-process-per-JSON prediction semantics.
# Completed designs are skipped and every expected sample is checked after inference.
#
# H200 default:
#   sbatch --array=0-$((T-1)) scripts/submit_packed_pred.slurm \
#     <READY_LIST> <PRED_ROOT> <T> [protenix flags]
# PLI H100 override (the partition determines the GPU type; no H200 constraint is embedded):
#   sbatch --account=pli_x --qos=pli-low --partition=pli --gres=gpu:h100:1 \
#     --array=0-$((T-1)) \
#     scripts/submit_packed_pred.slurm <READY_LIST> <PRED_ROOT> <T> [protenix flags]
#
# Optional runtime controls: GPU_TELEMETRY=0 disables the 5-second nvidia-smi sampler;
# GPU_TELEMETRY_INTERVAL changes its cadence; MIN_GPU_MEMORY_MIB changes the 70000 MiB guard.
#SBATCH --job-name=packed_pred
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -uo pipefail

LIST="${1:?READY_LIST}"
PRED_ROOT="${2:?PRED_ROOT}"
NTASKS="${3:?NTASKS}"
shift 3
EXTRA_ARGS=("$@")

PROTENIX="${PROTENIX_RUN:-/scratch/gpfs/KAIYIJIANG/tools/protenix/bin/protenix-run}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
GPU_TELEMETRY="${GPU_TELEMETRY:-1}"
GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-5}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-70000}"

if [[ ! "${NTASKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: NTASKS must be a positive integer, got '${NTASKS}'" >&2
  exit 2
fi
if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID >= NTASKS )); then
  echo "FATAL: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside [0, ${NTASKS})" >&2
  exit 2
fi
if [ ! -f "${LIST}" ]; then
  echo "FATAL: ready list not found: ${LIST}" >&2
  exit 2
fi
if [ ! -x "${PROTENIX}" ]; then
  echo "FATAL: protenix-run is not executable: ${PROTENIX}" >&2
  exit 2
fi
if ! command -v jq > /dev/null 2>&1; then
  echo "FATAL: jq is required to validate local MSA paths before allocating Protenix" >&2
  exit 2
fi

mkdir -p "${PRED_ROOT}"
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
INPUT_DIR="${TMP_BASE}/protenix_packed_${JOB_ID}_${TASK_ID}_${BASHPID}"
mkdir -p "${INPUT_DIR}"

TELEMETRY_PID=""
TELEMETRY_FILE=""
TELEMETRY_IS_TEMP=0

summarize_telemetry() {
  [ -s "${TELEMETRY_FILE}" ] || return 0
  awk -F',' '
    {
      for (i = 1; i <= NF; i++) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
      }
      util = $3 + 0
      mem = $4 + 0
      total = $5 + 0
      util_sum += util
      below20 += (util < 20)
      if (util > util_max) util_max = util
      if (mem > mem_max) mem_max = mem
      if (total > mem_total) mem_total = total
      n++
    }
    END {
      if (n > 0) {
        printf "[gpu-telemetry] samples=%d mean_util=%.1f%% below20=%.1f%% max_util=%.0f%% peak_mem=%.0f/%.0f MiB\n", \
          n, util_sum / n, 100.0 * below20 / n, util_max, mem_max, mem_total
      }
    }
  ' "${TELEMETRY_FILE}"
}

cleanup() {
  local status=$?
  if [ -n "${TELEMETRY_PID}" ]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    summarize_telemetry || true
  fi
  if [ "${TELEMETRY_IS_TEMP}" -eq 1 ] && [ -n "${TELEMETRY_FILE}" ]; then
    rm -f "${TELEMETRY_FILE}"
  fi
  rm -rf "${INPUT_DIR}"
  return "${status}"
}
trap cleanup EXIT

EXPECTED_SAMPLES=5
EXPECTED_SEEDS="101"
NEED_ATOM_CONFIDENCE=false
SEEDS_ARG_PRESENT=0
for ((arg_idx = 0; arg_idx < ${#EXTRA_ARGS[@]}; arg_idx++)); do
  arg="${EXTRA_ARGS[$arg_idx]}"
  case "${arg}" in
    --sample|-e)
      EXPECTED_SAMPLES="${EXTRA_ARGS[$((arg_idx + 1))]:-}"
      arg_idx=$((arg_idx + 1))
      ;;
    --sample=*) EXPECTED_SAMPLES="${arg#*=}" ;;
    --seeds|-s)
      EXPECTED_SEEDS="${EXTRA_ARGS[$((arg_idx + 1))]:-}"
      SEEDS_ARG_PRESENT=1
      arg_idx=$((arg_idx + 1))
      ;;
    --seeds=*)
      EXPECTED_SEEDS="${arg#*=}"
      SEEDS_ARG_PRESENT=1
      ;;
    --need_atom_confidence)
      NEED_ATOM_CONFIDENCE="${EXTRA_ARGS[$((arg_idx + 1))]:-}"
      arg_idx=$((arg_idx + 1))
      ;;
    --need_atom_confidence=*) NEED_ATOM_CONFIDENCE="${arg#*=}" ;;
  esac
done

if [[ ! "${EXPECTED_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: --sample must be a positive integer, got '${EXPECTED_SAMPLES}'" >&2
  exit 2
fi
EXPECTED_SEEDS="${EXPECTED_SEEDS//[[:space:]]/}"
IFS=',' read -r -a SEED_VALUES <<< "${EXPECTED_SEEDS}"
if [ "${#SEED_VALUES[@]}" -eq 0 ] || [ -z "${SEED_VALUES[0]}" ]; then
  echo "FATAL: --seeds resolved to an empty list" >&2
  exit 2
fi
case "${NEED_ATOM_CONFIDENCE,,}" in
  true|1|yes) NEED_ATOM_CONFIDENCE=true ;;
  *) NEED_ATOM_CONFIDENCE=false ;;
esac

layout_is_complete() {
  local root="$1"
  local stem="$2"
  local seed sample_idx
  for seed in "${SEED_VALUES[@]}"; do
    for ((sample_idx = 0; sample_idx < EXPECTED_SAMPLES; sample_idx++)); do
      [ -f "${root}/seed_${seed}/predictions/${stem}_sample_${sample_idx}.cif" ] || return 1
      [ -f "${root}/seed_${seed}/predictions/${stem}_summary_confidence_sample_${sample_idx}.json" ] || return 1
      if [ "${NEED_ATOM_CONFIDENCE}" = true ]; then
        [ -f "${root}/seed_${seed}/predictions/${stem}_full_data_sample_${sample_idx}.json" ] || return 1
      fi
    done
  done
  return 0
}

is_complete() {
  local stem="$1"
  local root="${PRED_ROOT}/${stem}"
  local candidate
  layout_is_complete "${root}" "${stem}" && return 0
  for candidate in "${root}"/*; do
    [ -d "${candidate}" ] || continue
    layout_is_complete "${candidate}" "${stem}" && return 0
  done
  return 1
}

# Directory-mode Protenix writes <PRED_ROOT>/<name>/seed_*. The existing tetramer readers
# expect <PRED_ROOT>/<name>/<name>/seed_*, so move fresh seed directories into that layout.
# If a legacy directory is already partial, merge a completed direct seed into it atomically
# enough for resume: the direct copy is retained on copy failure and removed only after success.
normalize_layout() {
  local stem="$1"
  local root="${PRED_ROOT}/${stem}"
  local legacy="${root}/${stem}"
  local seed_dir seed_name
  [ -d "${root}" ] || return 0
  for seed_dir in "${root}"/seed_*; do
    [ -d "${seed_dir}" ] || continue
    seed_name="$(basename "${seed_dir}")"
    mkdir -p "${legacy}"
    if [ -e "${legacy}/${seed_name}" ]; then
      mkdir -p "${legacy}/${seed_name}"
      if ! cp -a "${seed_dir}/." "${legacy}/${seed_name}/"; then
        echo "[packed t${TASK_ID}] failed to merge output layout for ${stem}/${seed_name}" >&2
        return 1
      fi
      rm -rf -- "${seed_dir}"
    else
      if ! mv "${seed_dir}" "${legacy}/"; then
        echo "[packed t${TASK_ID}] failed to normalize output layout for ${stem}/${seed_name}" >&2
        return 1
      fi
    fi
  done
}

validate_local_msa() {
  local json_path="$1"
  local msa_paths msa_path
  if ! msa_paths="$(
    jq -r '.. | objects | (.pairedMsaPath?, .unpairedMsaPath?) |
      select(type == "string" and length > 0)' "${json_path}"
  )"; then
    echo "[packed t${TASK_ID}] invalid JSON: ${json_path}" >&2
    return 1
  fi
  if [ -z "${msa_paths}" ]; then
    echo "[packed t${TASK_ID}] no local paired/unpaired MSA paths in ${json_path}" >&2
    return 1
  fi
  while IFS= read -r msa_path; do
    if [ ! -f "${msa_path}" ]; then
      echo "[packed t${TASK_ID}] missing local MSA: ${msa_path}" >&2
      return 1
    fi
  done <<< "${msa_paths}"
}

mapfile -t JSONS < "${LIST}"
n="${#JSONS[@]}"
echo "[packed t${TASK_ID}/${NTASKS}] $(date +%H:%M:%S) striding over ${n} designs"

declare -a PENDING_STEMS=()
declare -A SEEN_STEMS=()
done_ct=0
skip_ct=0
fail_ct=0

for ((k = TASK_ID; k < n; k += NTASKS)); do
  JSON="${JSONS[$k]}"
  if [ -z "${JSON:-}" ] || [ ! -f "${JSON}" ]; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED missing JSON at ready-list index ${k}: ${JSON:-<empty>}"
    continue
  fi
  stem="$(basename "${JSON}" .json)"
  stem="${stem%-update-msa}"
  if [ -n "${SEEN_STEMS[$stem]+set}" ]; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED duplicate prediction stem: ${stem}"
    continue
  fi
  SEEN_STEMS["${stem}"]=1

  if ! normalize_layout "${stem}"; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED ${stem} (output layout normalization)"
    continue
  fi
  if is_complete "${stem}"; then
    skip_ct=$((skip_ct + 1))
    continue
  fi

  if ! validate_local_msa "${JSON}"; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED ${stem} (local MSA preflight)"
    continue
  fi

  stage_path="${INPUT_DIR}/$(printf '%08d_%s' "${k}" "$(basename "${JSON}")")"
  if ! ln -s "${JSON}" "${stage_path}"; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED staging ${JSON}"
    continue
  fi
  PENDING_STEMS+=("${stem}")
done

if [ "${#PENDING_STEMS[@]}" -eq 0 ]; then
  echo "[packed t${TASK_ID}] $(date +%H:%M:%S) DONE  done=0 skip=${skip_ct} fail=${fail_ct}"
  [ "${fail_ct}" -eq 0 ]
  exit $?
fi

if command -v nvidia-smi > /dev/null 2>&1; then
  GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [ -n "${GPU_INFO}" ]; then
    GPU_NAME="${GPU_INFO%,*}"
    GPU_MEMORY_MIB="${GPU_INFO##*,}"
    GPU_MEMORY_MIB="${GPU_MEMORY_MIB//[[:space:]]/}"
    echo "[gpu] name=${GPU_NAME} memory_total=${GPU_MEMORY_MIB} MiB min_required=${MIN_GPU_MEMORY_MIB} MiB"
    if [[ "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] && (( GPU_MEMORY_MIB < MIN_GPU_MEMORY_MIB )); then
      echo "FATAL: GPU memory ${GPU_MEMORY_MIB} MiB is below tetramer minimum ${MIN_GPU_MEMORY_MIB} MiB" >&2
      exit 3
    fi
  fi

  if [ "${GPU_TELEMETRY}" != "0" ]; then
    if [ -n "${GPU_TELEMETRY_FILE:-}" ]; then
      TELEMETRY_FILE="${GPU_TELEMETRY_FILE}"
      mkdir -p "$(dirname "${TELEMETRY_FILE}")"
    else
      TELEMETRY_FILE="${TMP_BASE}/protenix_gpu_${JOB_ID}_${TASK_ID}.csv"
      TELEMETRY_IS_TEMP=1
    fi
    nvidia-smi \
      --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits \
      -l "${GPU_TELEMETRY_INTERVAL}" > "${TELEMETRY_FILE}" 2>/dev/null &
    TELEMETRY_PID=$!
  fi
fi

echo "[packed t${TASK_ID}] persistent model input_count=${#PENDING_STEMS[@]} input_dir=${INPUT_DIR}"
runner_status=0
RUN_ARGS=(pred -i "${INPUT_DIR}" -o "${PRED_ROOT}" --use_msa true)
if [ "${SEEDS_ARG_PRESENT}" -eq 0 ]; then
  RUN_ARGS+=(--seeds 101)
fi
RUN_ARGS+=("${EXTRA_ARGS[@]}")
"${PROTENIX}" "${RUN_ARGS[@]}" || runner_status=$?

for stem in "${PENDING_STEMS[@]}"; do
  if ! normalize_layout "${stem}"; then
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED ${stem} (output layout normalization)"
    continue
  fi
  if is_complete "${stem}"; then
    done_ct=$((done_ct + 1))
  else
    fail_ct=$((fail_ct + 1))
    echo "[packed t${TASK_ID}] FAILED ${stem} (missing one or more expected outputs)"
  fi
done

if [ "${runner_status}" -ne 0 ]; then
  echo "[packed t${TASK_ID}] protenix-run exited ${runner_status}" >&2
  fail_ct=$((fail_ct + 1))
fi

echo "[packed t${TASK_ID}] $(date +%H:%M:%S) DONE  done=${done_ct} skip=${skip_ct} fail=${fail_ct}"
[ "${fail_ct}" -eq 0 ]
