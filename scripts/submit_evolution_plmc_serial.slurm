#!/bin/bash
# One-allocation, sequential PLMC worker for a complete parent manifest.
#
# All manifest rows are preflighted before any model starts. Every row is then
# rerun in manifest order with atomic overwrite; no depth gate suppresses a model.
# The launcher supplies the repository runner explicitly because sbatch copies
# this worker to /var/spool/slurmd before execution.
#SBATCH --job-name=uric_plmc_serial
#SBATCH --account=kaiyijiang
#SBATCH --partition=cpu
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00

set -euo pipefail
: "${PS1:=}"
export PS1

MANIFEST="${1:?PLMC_MANIFEST.tsv}"
OUTPUT_ROOT="${2:?RUN_OUTPUT_ROOT}"
PLMC_BINARY="${3:?PLMC_BINARY}"
EVCOUPLINGS_PYTHON="${4:?EVCOUPLINGS_PYTHON}"
RUNNER="${5:?RUN_EVOLUTION_PLMC.py}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! "${CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "FATAL: PLMC manifest not found: ${MANIFEST}" >&2
    exit 2
fi
if [[ ! -x "${PLMC_BINARY}" ]]; then
    echo "FATAL: PLMC binary is not executable: ${PLMC_BINARY}" >&2
    exit 2
fi
if [[ ! -x "${EVCOUPLINGS_PYTHON}" ]]; then
    echo "FATAL: EVcouplings Python is not executable: ${EVCOUPLINGS_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${RUNNER}" || ! -r "${RUNNER}" ]]; then
    echo "FATAL: PLMC runner is not readable: ${RUNNER}" >&2
    exit 2
fi

NORMALIZED_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/plmc_manifest.XXXXXX.tsv")"
cleanup_manifest() {
    rm -f -- "${NORMALIZED_MANIFEST}"
}
trap cleanup_manifest EXIT
# Python csv writes CRLF by default. Remove only a terminal CR; embedded CRs
# remain and are rejected by the existing field/path contracts.
sed 's/\r$//' "${MANIFEST}" > "${NORMALIZED_MANIFEST}"

expected_header=$'protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length'
actual_header="$(head -n 1 "${NORMALIZED_MANIFEST}")"
if [[ "${actual_header}" != "${expected_header}" ]]; then
    echo "FATAL: unexpected PLMC manifest header: ${actual_header}" >&2
    exit 2
fi
awk -F '\t' '
    NR == 1 { next }
    NF == 0 { printf "FATAL: blank manifest row %d\n", NR > "/dev/stderr"; bad=1; next }
    NF != 6 { printf "FATAL: manifest row %d has %d fields, expected 6\n", NR, NF > "/dev/stderr"; bad=1; next }
    $1 !~ /^[A-Za-z0-9][A-Za-z0-9_.-]*$/ { printf "FATAL: unsafe protein_id at row %d\n", NR > "/dev/stderr"; bad=1 }
    $2 != $1 { printf "FATAL: query_id mismatch at row %d\n", NR > "/dev/stderr"; bad=1 }
    $3 == "" { printf "FATAL: empty alignment path at row %d\n", NR > "/dev/stderr"; bad=1 }
    $4 !~ /^(qualified|exploratory|unqualified|not_measured)$/ { printf "FATAL: invalid covariance gate at row %d\n", NR > "/dev/stderr"; bad=1 }
    seen[$1]++ { printf "FATAL: duplicate protein_id at row %d\n", NR > "/dev/stderr"; bad=1 }
    { n++ }
    END { if (n < 1) { print "FATAL: manifest has no model rows" > "/dev/stderr"; bad=1 }; exit bad }
' "${NORMALIZED_MANIFEST}"

while IFS=$'\t' read -r PROTEIN_ID QUERY_ID ALIGNMENT_PATH COVARIANCE_GATE NEFF NEFF_PER_L; do
    if [[ ! -f "${ALIGNMENT_PATH}" ]]; then
        echo "FATAL: focus alignment not found for ${PROTEIN_ID}: ${ALIGNMENT_PATH}" >&2
        exit 2
    fi
done < <(tail -n +2 "${NORMALIZED_MANIFEST}")

mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"

n_rows="$(awk 'NR > 1 { n++ } END { print n + 0 }' "${NORMALIZED_MANIFEST}")"
row_index=0
while IFS=$'\t' read -r PROTEIN_ID QUERY_ID ALIGNMENT_PATH COVARIANCE_GATE NEFF NEFF_PER_L; do
    row_index=$((row_index + 1))
    OUTPUT_PREFIX="${OUTPUT_ROOT}/${PROTEIN_ID}/plmc"
    echo "[$(date -Is)] model=${row_index}/${n_rows} protein_id=${PROTEIN_ID} gate=${COVARIANCE_GATE} neff=${NEFF} neff_per_L=${NEFF_PER_L}"
    "${EVCOUPLINGS_PYTHON}" -u "${RUNNER}" \
        --alignment "${ALIGNMENT_PATH}" \
        --query-id "${QUERY_ID}" \
        --output-prefix "${OUTPUT_PREFIX}" \
        --plmc-binary "${PLMC_BINARY}" \
        --theta 0.8 --lambda-h 0.01 --lambda-j-base 0.01 \
        --iterations 500 --cpu "${CPUS}" --overwrite
done < <(tail -n +2 "${NORMALIZED_MANIFEST}")
echo "[$(date -Is)] completed_models=${row_index}"
