#!/usr/bin/env bash
#
# Refiner inference-time sweep launcher.
#
# Submits 12 sbatch jobs over the matrix
#
#   apply_steps      ∈ {final_only, last_2}            ( 2 )
#   mask_ratio_center ∈ {0.05, 0.10, 0.20}              ( 3 )
#   mask_ratio_dev    = 0.0
#   fusion_mode       = residual
#   fusion_alpha      ∈ {0.10, 0.25}                    ( 2 )
#
# Total: 2 × 3 × 1 × 1 × 2 = 12 generate_refiner runs.
#
# All 12 use the same DPLM checkpoint and the same refiner checkpoint;
# only inference-time knobs change. ARM is fixed to ``refiner`` (no
# sidecar). Each submission gets a unique TAG so run dirs do not
# collide.
#
# Required env:
#   CHECKPOINT          : DPLM Module K checkpoint
#   REFINER_CHECKPOINT  : refiner .pt
# Optional env (forwarded to submit_if_imp.slurm; defaults there apply):
#   ALLELE / INPUT_SOURCE / CATH_SPLIT / CATH_MIN_RESOLVED_RATIO
#   LIMIT_PROTEINS  : subsample the test set for a fast pilot sweep
#   SEED / MAX_ITER / GEN_BATCH_SIZE / TEMPERATURE
#   DRY_RUN=1       : print the 12 sbatch commands without submitting
#
# Usage:
#   CHECKPOINT=.../best.ckpt \
#   REFINER_CHECKPOINT=.../refiner_last.pt \
#   LIMIT_PROTEINS=200 \
#       bash scripts/submit_if_imp_refiner_sweep.sh

set -euo pipefail
: "${PS1:=}"
export PS1

: "${CHECKPOINT:?set CHECKPOINT to the DPLM Module K .ckpt}"
: "${REFINER_CHECKPOINT:?set REFINER_CHECKPOINT to the trained refiner .pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/submit_if_imp.slurm"
if [[ ! -f "${SLURM_FILE}" ]]; then
    echo "ERROR: cannot find ${SLURM_FILE}" >&2
    exit 2
fi

APPLY_STEPS_MATRIX=(final_only last_2)
MASK_RATIO_CENTER_MATRIX=(0.05 0.10 0.20)
FUSION_ALPHA_MATRIX=(0.10 0.25)
MASK_RATIO_DEVIATION=0.0
FUSION_MODE=residual
ARM=refiner

DRY_RUN="${DRY_RUN:-0}"
SUBMITTED=0

for apply_steps in "${APPLY_STEPS_MATRIX[@]}"; do
    for mrc in "${MASK_RATIO_CENTER_MATRIX[@]}"; do
        for alpha in "${FUSION_ALPHA_MATRIX[@]}"; do
            # Encode the knob values into a TAG that survives into the
            # run_id, so output dirs are self-describing.
            mrc_tag="$(printf '%s' "${mrc}" | tr -d '.')"
            alpha_tag="$(printf '%s' "${alpha}" | tr -d '.')"
            TAG="sweep_${apply_steps}_mrc${mrc_tag}_a${alpha_tag}"

            env_args=(
                "MODE=generate_refiner"
                "ARM=${ARM}"
                "CHECKPOINT=${CHECKPOINT}"
                "REFINER_CHECKPOINT=${REFINER_CHECKPOINT}"
                "APPLY_STEPS=${apply_steps}"
                "MASK_RATIO_CENTER=${mrc}"
                "MASK_RATIO_DEVIATION=${MASK_RATIO_DEVIATION}"
                "FUSION_MODE=${FUSION_MODE}"
                "FUSION_ALPHA=${alpha}"
                "TAG=${TAG}"
            )
            # Forward any optional env the caller set, only when present.
            for var in \
                ALLELE INPUT_SOURCE CATH_ROOT CATH_SPLIT CATH_MAX_LENGTH \
                CATH_MIN_RESOLVED_RATIO LIMIT_PROTEINS SEED MAX_ITER \
                GEN_BATCH_SIZE TEMPERATURE MC_DROPOUT_PASSES \
                TEST_SET_PARQUET PDB_ROOT OUTPUT_ROOT \
                WANDB WANDB_PROJECT WANDB_GROUP WANDB_TAGS WANDB_MODE
            do
                if [[ -n "${!var:-}" ]]; then
                    env_args+=("${var}=${!var}")
                fi
            done

            cmd=(env "${env_args[@]}" sbatch "${SLURM_FILE}")
            echo "[sweep] ${TAG}"
            echo "  ${cmd[*]}"
            if [[ "${DRY_RUN}" != "1" ]]; then
                "${cmd[@]}"
            fi
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo "============================================================"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[sweep] DRY_RUN=1: would have submitted ${SUBMITTED} jobs"
else
    echo "[sweep] submitted ${SUBMITTED} jobs"
fi
echo "============================================================"
