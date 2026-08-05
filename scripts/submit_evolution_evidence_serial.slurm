#!/bin/bash
# One-allocation, serial evolutionary-evidence analysis for a frozen parent panel.
#
# Stdout/stderr are intentionally supplied by the sbatch invocation so the
# caller chooses one explicit logs/ pair.  This worker never submits an array.
# The repository panel runner is supplied explicitly because sbatch executes a
# spool copy of this file.
#SBATCH --job-name=uric_evolution_all
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

PLMC_MANIFEST="${1:?PLMC_MANIFEST.tsv}"
WT_ROOT="${2:?WT_FASTA_ROOT}"
PLMC_ROOT="${3:?PLMC_OUTPUT_ROOT}"
OUTPUT_ROOT="${4:?EVOLUTION_OUTPUT_ROOT}"
EVCOUPLINGS_PYTHON="${5:?EVCOUPLINGS_PYTHON}"
PANEL_RUNNER="${6:?RUN_EVOLUTION_EVIDENCE_PANEL.py}"
EXPECTED_PARENT_COUNT="${7:?EXPECTED_PARENT_COUNT}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! "${EXPECTED_PARENT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: EXPECTED_PARENT_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ ! "${CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 2
fi
if [[ ! -f "${PLMC_MANIFEST}" ]]; then
    echo "FATAL: PLMC manifest not found: ${PLMC_MANIFEST}" >&2
    exit 2
fi
if [[ ! -d "${WT_ROOT}" ]]; then
    echo "FATAL: WT FASTA root not found: ${WT_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${PLMC_ROOT}" ]]; then
    echo "FATAL: PLMC output root not found: ${PLMC_ROOT}" >&2
    exit 2
fi
if [[ ! -x "${EVCOUPLINGS_PYTHON}" ]]; then
    echo "FATAL: EVcouplings Python is not executable: ${EVCOUPLINGS_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${PANEL_RUNNER}" || ! -r "${PANEL_RUNNER}" ]]; then
    echo "FATAL: panel runner is not readable: ${PANEL_RUNNER}" >&2
    exit 2
fi
case "/${OUTPUT_ROOT}/" in
    */work/*) ;;
    *) echo "FATAL: OUTPUT_ROOT must be inside a work/ layer" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export PYTHONHASHSEED=0

"${EVCOUPLINGS_PYTHON}" -u "${PANEL_RUNNER}" \
    --plmc-manifest "${PLMC_MANIFEST}" \
    --wt-root "${WT_ROOT}" \
    --plmc-root "${PLMC_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --expected-parent-count "${EXPECTED_PARENT_COUNT}" \
    --theta 0.8 \
    --natural-minimum-coverage 0.95 \
    --min-sequence-separation 6 \
    --score-chunk-size 256
