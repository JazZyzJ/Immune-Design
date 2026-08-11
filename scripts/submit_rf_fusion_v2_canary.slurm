#!/bin/bash
#SBATCH --job-name=v2canary
#SBATCH --qos=gpu-short
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Sized on MEASURED usage, not on headroom-by-habit: the four executed cells peaked at
# MaxRSS 27.5-30.4 GB and a cell finishes in ~2 min, so 96G/8-core requests were ~3x over and only
# bought queue time on a busy rtx6000. The work is GPU-bound (DPLM + ESMC-6B + ESMFold2); the CPUs
# were idle. Raise `--mem` again if a longer protein pushes past ~35 GB.
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/fusion_v2/v2canary_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/fusion_v2/v2canary_%j.err

# One V2 state-transition / mechanism / V2F5A qualification CELL
# (doc/RF_Fusion_v2_Cluster_Runbook.md §§5, 7, 9).
#
# Every path this job needs comes from the cell's own ``<CELL>.args.sh``, which
# ``scripts/materialize_v2_canary_config.py`` writes NEXT TO the resolved config it signed.  That
# is deliberate: the driver's ``--input-file``/``--shard-input`` vectors must agree with the config's
# frozen/runtime content declaration, and a hand-maintained command in a second file is how the two
# drift.  This script therefore chooses nothing scientific -- it sources the vectors and adds the
# two things that belong to the JOB rather than to the cell: where to journal, and which device.
#
# Submit (one job per cell, all four before any feedback outcome is opened):
#   CELL=5zhv_r30   sbatch --job-name=v2canary_5zhv_r30   scripts/submit_rf_fusion_v2_canary.slurm
#   CELL=5zhv_r40   sbatch --job-name=v2canary_5zhv_r40   scripts/submit_rf_fusion_v2_canary.slurm
#   CELL=q00511_r30 sbatch --job-name=v2canary_q00511_r30 scripts/submit_rf_fusion_v2_canary.slurm
#   CELL=q00511_r40 sbatch --job-name=v2canary_q00511_r40 scripts/submit_rf_fusion_v2_canary.slurm
#
# rtx6000 is RTX PRO 6000 (sm_120), which needs the Blackwell env -- the same env the B(r) scan and
# the hotspot calibration ran under.  A different env would fold and score the Canary on a stack the
# threshold it is gated against was never measured on.

set -euo pipefail
: "${PS1:=}"
export PS1

# ── Cell + layout (override via env; no cluster path is hardcoded in a module) ──
# CELL runs one legacy cell. CELL_LIST runs one cell stem per line, serially, in this ONE GPU job.
# Exactly one is required; CELL_LIST never creates an array or background process.
CELL="${CELL:-}"
CELL_LIST="${CELL_LIST:-}"
if [[ -n "${CELL}" && -n "${CELL_LIST}" ]] || [[ -z "${CELL}" && -z "${CELL_LIST}" ]]; then
  echo "set exactly one of CELL or CELL_LIST" >&2
  exit 4
fi
# EXPORTED, not merely set: the DPLM checkpoint carries a stored hydra config whose paths are
# ``oc.env:PROJECT_ROOT`` interpolations, so omegaconf resolves them out of the ENVIRONMENT while
# loading the checkpoint.  As a plain shell variable it is invisible there and the load dies with
# "Environment variable 'PROJECT_ROOT' not found" -- after the GPU is allocated.
export PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
export SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/KAIYIJIANG/zijie}"
WORK="${WORK:-${SCRATCH_BASE}/work/immune-design/v2_canary}"
RUNDIR="${RUNDIR:-${SCRATCH_BASE}/run/inverse_folding/v2_canary}"
CONDA_ENV="${CONDA_ENV:-immune-design-blackwell}"
DEVICE="${DEVICE:-cuda}"
# 0 = the ordinary depth ladder (the Canary).  N > 0 = the runbook §7 MECHANISM cohort: N
# independent source prefixes, each carried through the matched arms.  A prefix runs five cycles
# where a Canary cell runs one, so it needs both more wall clock and more memory than the static
# header below, which stays sized for a Canary cell (MEASURED: a Canary cell peaks 27.5-30.4 GB, a
# one-prefix mechanism run peaked 39.2 GiB against a 40 GiB cap -- too close to run 16 of them):
#   MECHANISM_PREFIXES=16 CELL=q00511_mech \
#     sbatch --time=04:00:00 --mem=64G scripts/submit_rf_fusion_v2_canary.slurm
MECHANISM_PREFIXES="${MECHANISM_PREFIXES:-0}"
# First prefix index of THIS batch.  Batch 2 continues the sequence (e.g. 16) rather
# than repeating batch 1, whose seeds are derived from the index.
MECHANISM_PREFIX_START="${MECHANISM_PREFIX_START:-0}"
# 1 = V2F5A Head-directed support-policy qualification.  This is deliberately separate from the
# older mechanism view: it runs treatment/control support laws and must never be selected merely
# because a config happens to contain a Head-directed block.
QUALIFICATION="${QUALIFICATION:-0}"
QUALIFICATION_ARGS=()
case "${QUALIFICATION}" in
  0) ;;
  1)
    if [[ "${MECHANISM_PREFIXES}" -lt 1 ]]; then
      echo "QUALIFICATION=1 requires MECHANISM_PREFIXES>=1" >&2
      exit 4
    fi
    QUALIFICATION_ARGS+=(--qualification)
    ;;
  *)
    echo "QUALIFICATION must be exactly 0 or 1, got ${QUALIFICATION}" >&2
    exit 4
    ;;
esac

# Explicit, identity-bound exception for the non-confirmatory recursive capability sandbox in
# runbook §10.  It is never inferred from a D>1 config: both the resolved capability identity and
# this separate launch flag are required by the driver.
EXPLORATORY_DEPTH_OVERRIDE="${EXPLORATORY_DEPTH_OVERRIDE:-0}"
EXPLORATORY_DEPTH_ARGS=()
case "${EXPLORATORY_DEPTH_OVERRIDE}" in
  0) ;;
  1)
    if [[ "${MECHANISM_PREFIXES}" -ne 0 || "${QUALIFICATION}" -ne 0 ]]; then
      echo "EXPLORATORY_DEPTH_OVERRIDE=1 cannot be combined with mechanism/qualification mode" >&2
      exit 4
    fi
    EXPLORATORY_DEPTH_ARGS+=(--exploratory-depth-override)
    ;;
  *)
    echo "EXPLORATORY_DEPTH_OVERRIDE must be exactly 0 or 1, got ${EXPLORATORY_DEPTH_OVERRIDE}" >&2
    exit 4
    ;;
esac

# ── Env ──
module purge
module load anaconda3/2025.12
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Della compute nodes have no route to huggingface.co.  Without HF_HOME the DPLM task tries to
# fetch ``airkingbd/dplm_650m``'s config from the hub and spends its retry ladder on DNS failures;
# OFFLINE makes a genuinely missing model fail immediately and say so, instead of after five
# escalating timeouts.  The same cache holds the ESMFold2 snapshot bound as ``structure_backend``.
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/model_cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${PROJECT_ROOT}"

echo "[v2canary] git=$(git rev-parse HEAD) env=${CONDA_ENV} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[v2canary] mechanism_prefixes=${MECHANISM_PREFIXES} start=${MECHANISM_PREFIX_START} qualification=${QUALIFICATION} exploratory_depth_override=${EXPLORATORY_DEPTH_OVERRIDE}"

run_cell() (
  set -euo pipefail
  local cell=$1
  local out_dir="${RUNDIR}/${cell}"
  local args_file="${WORK}/resolved_configs/${cell}.args.sh"
  if [[ ! -r "${args_file}" ]]; then
    echo "[v2canary] unreadable args file: ${args_file}" >&2
    exit 4
  fi
  # shellcheck source=/dev/null
  source "${args_file}"   # -> V2_CONFIG, V2_COHORT, INPUT_FILES[], SHARD_INPUTS[]
  mkdir -p "${out_dir}"
  echo "[v2canary] cell=${cell} cohort=${V2_COHORT} config=${V2_CONFIG}"
  python scripts/run_rf_fusion_v2.py \
    --v2-config   "${V2_CONFIG}" \
    --out-dir     "${out_dir}" \
    --cohort      "${V2_COHORT}" \
    --mechanism-prefixes "${MECHANISM_PREFIXES}" \
    --mechanism-prefix-start "${MECHANISM_PREFIX_START}" \
    "${QUALIFICATION_ARGS[@]}" \
    "${EXPLORATORY_DEPTH_ARGS[@]}" \
    --input-file  "${INPUT_FILES[@]}" \
    --shard-input "${SHARD_INPUTS[@]}" \
                  "journal_dir=${out_dir}/journals" \
                  "device=${DEVICE}"
)

if [[ -n "${CELL}" ]]; then
  set +e
  run_cell "${CELL}"
  RC=$?
  set -e
else
  if [[ ! -r "${CELL_LIST}" ]]; then
    echo "CELL_LIST is not readable: ${CELL_LIST}" >&2
    exit 4
  fi
  TOTAL=0
  SUCCEEDED=0
  FAILED=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    read -r cell _ <<< "${line}"
    [[ -n "${cell:-}" ]] || continue
    TOTAL=$((TOTAL + 1))
    set +e
    run_cell "${cell}"
    cell_rc=$?
    set -e
    if [[ ${cell_rc} -eq 0 ]]; then
      SUCCEEDED=$((SUCCEEDED + 1))
    else
      FAILED=$((FAILED + 1))
      echo "[v2canary] cell=${cell} EXIT=${cell_rc}; continuing serial list" >&2
    fi
  done < "${CELL_LIST}"
  if [[ ${TOTAL} -eq 0 ]]; then
    echo "CELL_LIST contains no cell stems" >&2
    exit 4
  elif [[ ${FAILED} -eq 0 ]]; then
    RC=0
  elif [[ ${SUCCEEDED} -eq 0 ]]; then
    RC=2
  else
    RC=3
  fi
  echo "[v2canary] serial_summary total=${TOTAL} succeeded=${SUCCEEDED} failed=${FAILED}"
fi

# The driver's exit code is the whole interface: 0 complete with a success, 2 nothing usable or a
# realized cap breached, 3 partial, 4 not launchable.  Echoed so the log says which, unambiguously.
echo "EXIT=${RC}"
exit ${RC}
