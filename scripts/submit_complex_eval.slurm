#!/bin/bash
#SBATCH --job-name=cplx-iface
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/complex_iface_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/complex_iface_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
RUN_ROSETTA="${RUN_ROSETTA:-1}"

# Two modes:
#   passthrough : sbatch scripts/submit_complex_eval.slurm <eval_complex_gate.py args...>
#   sharded array (large sets): set SHARD_MANIFEST_DIR + SHARD_RUN_DIR + SHARD_LOG_DIR +
#     PRED_ROOT + USALIGN and
#     submit with --array=0-(N-1). Task i evaluates <SHARD_MANIFEST_DIR>/shard_<i>.parquet and
#     writes experiment outputs/runtime below <SHARD_RUN_DIR> and Rosetta stdout/stderr below
#     <SHARD_LOG_DIR>. Each shard
#     manifest must carry every parent's WT rows so --require-interface-wt / WT-normalization hold.
if [[ -z "${SHARD_MANIFEST_DIR:-}" && $# -eq 0 ]]; then
    echo "Usage: sbatch $0 <eval_complex_gate.py args>   OR   SHARD_MANIFEST_DIR=... SHARD_RUN_DIR=... SHARD_LOG_DIR=... PRED_ROOT=... USALIGN=... sbatch --array=0-N $0" >&2
    exit 2
fi

module purge
module load anaconda3/2025.12
if [[ "${RUN_ROSETTA}" == "1" ]]; then
    module load rosetta/3.15
elif [[ "${RUN_ROSETTA}" != "0" ]]; then
    echo "RUN_ROSETTA must be 0 or 1, got: ${RUN_ROSETTA}" >&2
    exit 2
fi
conda activate immune-design

export PYTHONHASHSEED=42
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export MPLCONFIGDIR="/tmp/${USER}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${RUN_ROSETTA}" == "1" ]]; then
    export ROSETTA_INTERFACE_ANALYZER
    ROSETTA_INTERFACE_ANALYZER="$(command -v InterfaceAnalyzer)"
else
    unset ROSETTA_INTERFACE_ANALYZER
fi

echo "[complex-eval] job=${SLURM_JOB_ID} host=${SLURMD_NODENAME} rosetta=${RUN_ROSETTA}"

if [[ -n "${SHARD_MANIFEST_DIR:-}" ]]; then
    i="${SLURM_ARRAY_TASK_ID:?SHARD_MANIFEST_DIR set but not an array job}"
    SH=$(printf "%s/shard_%02d.parquet" "${SHARD_MANIFEST_DIR}" "${i}")
    RUN_BASE="${SHARD_RUN_DIR:?SHARD_RUN_DIR}"
    OUT=$(printf "%s/gate_shard_%02d.parquet" "${RUN_BASE}" "${i}")
    ROSETTA_ARGS=()
    if [[ "${RUN_ROSETTA}" == "1" ]]; then
        RRD=$(printf "%s/rosetta_shard_%02d" "${RUN_BASE}" "${i}")
        RLD=$(printf "%s/rosetta_shard_%02d" "${SHARD_LOG_DIR:?SHARD_LOG_DIR}" "${i}")
        mkdir -p "${RRD}" "${RLD}"
        ROSETTA_ARGS=(--rosetta-run-dir "${RRD}" --rosetta-log-dir "${RLD}")
    fi
    mkdir -p "${RUN_BASE}"
    [[ -f "${SH}" ]] || { echo "no shard manifest: ${SH}" >&2; exit 3; }
    echo "[complex-eval] shard ${i}: ${SH} -> ${OUT}"
    # NOTE: run python directly (no srun). srun opens a PMIx step; the Rosetta InterfaceAnalyzer
    # subprocess is the MPI build (InterfaceAnalyzer.mpi.linuxgccrelease), and when several array
    # tasks are co-located on a node their per-step PMIx servers collide with the MPI singletons
    # -> PMIx errhandler kills the step (exit 137). No srun => no PMIx server => clean singletons.
    python -u "${PROJECT_ROOT}/scripts/eval_complex_gate.py" \
        --backend "${BACKEND:-protenix}" --pred-root "${PRED_ROOT:?PRED_ROOT}" \
        --manifest "${SH}" --usalign "${USALIGN:?USALIGN}" \
        --require-interface-wt "${ROSETTA_ARGS[@]}" --out "${OUT}" ${EVAL_EXTRA:-}
else
    echo "[complex-eval] args: $*"
    python -u "${PROJECT_ROOT}/scripts/eval_complex_gate.py" "$@"
fi
