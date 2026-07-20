#!/bin/bash
#SBATCH --job-name=tetra-iface
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/tetramer_iface_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/tetramer_iface_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
RUN_ROSETTA="${RUN_ROSETTA:-1}"

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch $0 <eval_tetramer_gate.py arguments>" >&2
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

echo "[tetramer-eval] job=${SLURM_JOB_ID} host=${SLURMD_NODENAME} rosetta=${RUN_ROSETTA}"
echo "[tetramer-eval] args: $*"

srun python -u "${PROJECT_ROOT}/scripts/eval_tetramer_gate.py" "$@"
