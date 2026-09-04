#!/bin/bash
# CPU-only shard-array worker for all-parent evolutionary evidence.
#
# Task i consumes <NORMALIZED_SHARD_ROOT>/<i:03d>/plmc_manifest.tsv and writes
# one isolated, atomic panel below <OUTPUT_SHARD_ROOT>/<i:03d>.  The strict
# panel runner requires WT_ROOT to equal the manifest cohort exactly, so this
# worker stages only the current shard's WT FASTAs in a node-local temporary
# directory.  The frozen global WT root remains read-only.
#SBATCH --job-name=uric_evolution
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

NORMALIZED_SHARD_ROOT="${1:?NORMALIZED_SHARD_ROOT}"
GLOBAL_WT_ROOT="${2:?GLOBAL_WT_FASTA_ROOT}"
PLMC_ROOT="${3:?PLMC_OUTPUT_ROOT}"
OUTPUT_SHARD_ROOT="${4:?EVOLUTION_OUTPUT_SHARD_ROOT}"
EVCOUPLINGS_PYTHON="${5:?EVCOUPLINGS_PYTHON}"
PANEL_RUNNER="${6:?RUN_EVOLUTION_EVIDENCE_PANEL.py}"
SHARD_COUNT="${7:?SHARD_COUNT}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: SHARD_COUNT must be a positive integer" >&2
    exit 2
fi
if ((TASK_ID >= SHARD_COUNT)); then
    echo "FATAL: array task ${TASK_ID} is outside SHARD_COUNT=${SHARD_COUNT}" >&2
    exit 2
fi
if [[ ! "${CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 2
fi
if [[ ! -d "${NORMALIZED_SHARD_ROOT}" ]]; then
    echo "FATAL: normalized shard root not found: ${NORMALIZED_SHARD_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${GLOBAL_WT_ROOT}" ]]; then
    echo "FATAL: global WT FASTA root not found: ${GLOBAL_WT_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${PLMC_ROOT}" ]]; then
    echo "FATAL: PLMC root not found: ${PLMC_ROOT}" >&2
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
case "/${OUTPUT_SHARD_ROOT}/" in
    */work/*) ;;
    *) echo "FATAL: OUTPUT_SHARD_ROOT must be inside a work/ layer" >&2; exit 2 ;;
esac

printf -v SHARD_TAG '%03d' "${TASK_ID}"
PLMC_MANIFEST="${NORMALIZED_SHARD_ROOT}/${SHARD_TAG}/plmc_manifest.tsv"
OUTPUT_ROOT="${OUTPUT_SHARD_ROOT}/${SHARD_TAG}"
if [[ ! -f "${PLMC_MANIFEST}" ]]; then
    echo "FATAL: shard manifest not found: ${PLMC_MANIFEST}" >&2
    exit 2
fi
expected_header=$'protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length'
actual_header="$(head -n 1 "${PLMC_MANIFEST}" | tr -d '\r')"
if [[ "${actual_header}" != "${expected_header}" ]]; then
    echo "FATAL: unexpected PLMC manifest header: ${actual_header}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "FATAL: shard output must be absent before execution: ${OUTPUT_ROOT}" >&2
    exit 2
fi

TMP_BASE="${TMPDIR:-/tmp}"
WT_STAGE="$(mktemp -d "${TMP_BASE}/uricase_evolution_${SLURM_ARRAY_JOB_ID:-manual}_${SHARD_TAG}.XXXXXX")"
cleanup() {
    if [[ -n "${WT_STAGE:-}" && -d "${WT_STAGE}" ]]; then
        rm -rf -- "${WT_STAGE}"
    fi
}
trap cleanup EXIT

declare -A SEEN_IDS=()
PARENT_COUNT=0
while IFS=$'\t' read -r PROTEIN_ID QUERY_ID _REST; do
    if [[ ! "${PROTEIN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "FATAL: unsafe protein_id in shard manifest: ${PROTEIN_ID}" >&2
        exit 2
    fi
    if [[ "${QUERY_ID}" != "${PROTEIN_ID}" ]]; then
        echo "FATAL: query_id differs from protein_id: ${QUERY_ID} != ${PROTEIN_ID}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_IDS[${PROTEIN_ID}]+x}" ]]; then
        echo "FATAL: duplicate protein_id in shard manifest: ${PROTEIN_ID}" >&2
        exit 2
    fi
    SEEN_IDS["${PROTEIN_ID}"]=1
    WT_SOURCE="${GLOBAL_WT_ROOT}/${PROTEIN_ID}.fasta"
    if [[ ! -s "${WT_SOURCE}" ]]; then
        echo "FATAL: WT FASTA missing or empty: ${WT_SOURCE}" >&2
        exit 2
    fi
    cp -- "${WT_SOURCE}" "${WT_STAGE}/${PROTEIN_ID}.fasta"
    PARENT_COUNT="$((PARENT_COUNT + 1))"
done < <(tail -n +2 "${PLMC_MANIFEST}")
if ((PARENT_COUNT < 1)); then
    echo "FATAL: shard manifest contains no parents: ${PLMC_MANIFEST}" >&2
    exit 2
fi

export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export PYTHONHASHSEED=0

echo "shard=${SHARD_TAG} parents=${PARENT_COUNT} manifest=${PLMC_MANIFEST} output=${OUTPUT_ROOT}"
"${EVCOUPLINGS_PYTHON}" -u "${PANEL_RUNNER}" \
    --plmc-manifest "${PLMC_MANIFEST}" \
    --wt-root "${WT_STAGE}" \
    --plmc-root "${PLMC_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --expected-parent-count "${PARENT_COUNT}" \
    --theta 0.8 \
    --natural-minimum-coverage 0.95 \
    --min-sequence-separation 6 \
    --score-chunk-size 256
