#!/bin/bash
#SBATCH --job-name=if-phase-c
#SBATCH --account=kaiyijiang
#SBATCH --qos=gpu-short
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c/phase_c_%j.err

set -euo pipefail
: "${PS1:=}"
export PS1

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
WORK_DIR="${WORK_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding}"
LOG_DIR="${LOG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/logs/if_phase_c}"
export PROJECT_ROOT WORK_DIR RUN_DIR LOG_DIR

MODE="${MODE:-native}"   # native | reference_flow
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-immune-design}"

if [[ "${ALLELE}" == "HLA-DRB1*07:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_07_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_07_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_07_01.parquet"
elif [[ "${ALLELE}" == "HLA-DRB1*04:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_04_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_04_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_04_01.parquet"
elif [[ "${ALLELE}" == "HLA-DRB1*15:01" ]]; then
    ALLELE_FILE_TAG="HLA-DRB1_15_01"
    PDB_ROOT_DEFAULT="${WORK_DIR}/if_test_set/pdbs_if_ready/HLA-DRB1_15_01"
    H_MAPS_DEFAULT="${WORK_DIR}/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_15_01.parquet"
else
    echo "Unsupported ALLELE=${ALLELE}; expected HLA-DRB1*07:01, HLA-DRB1*04:01, or HLA-DRB1*15:01" >&2
    exit 2
fi

CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${WORK_DIR}/if_test_set/if_ready/fast/fast_v2_${ALLELE_FILE_TAG}.parquet}"
PDB_ROOT="${PDB_ROOT:-${PDB_ROOT_DEFAULT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/if_phase_c/${MODE}}"

N_DESIGNS_PER_PROTEIN="${N_DESIGNS_PER_PROTEIN:-1}"
SEED="${SEED:-42}"
# C0 (DPLM native) iterations. Aligned with the DPLM paper's inverse-folding
# evaluation default (README task.generator.max_iter=100). C1 iter count is
# read from the YAML config's sampler.n_steps (also 100) for fair comparison.
MAX_ITER="${MAX_ITER:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NATIVE_SAMPLING_STRATEGY="${NATIVE_SAMPLING_STRATEGY:-argmax}"
NATIVE_BATCH_SIZE="${NATIVE_BATCH_SIZE:-1}"

# h-map parquet is optional (run_if_phase_c1.py --h-maps-parquet is optional):
# leave H_MAPS_PARQUET unset to use the allele default; set it explicitly EMPTY
# (H_MAPS_PARQUET= sbatch ...) to run without an h-map (constant_one /
# controller-only arms, where g(h)=1 and h is inert). The `-` (not `:-`) keeps an
# explicit empty value instead of falling back to the default.
#
# Because an UNSET variable silently becomes the allele default, "non-empty" alone
# cannot distinguish caller intent from the fallback. MODE=v1_entry must REFUSE a
# caller-supplied h-map (V1-A runs g(h)=1 with no h-map at all), so record whether
# the caller set the variable at all BEFORE the default is applied.
H_MAPS_PARQUET_REQUESTED="${H_MAPS_PARQUET+requested}"
H_MAPS_PARQUET="${H_MAPS_PARQUET-${H_MAPS_DEFAULT}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/inverse_folding/reference_flow/configs/c1_null.yaml}"
FAIL_PCT_THRESHOLD="${FAIL_PCT_THRESHOLD:-0.05}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-0}"
H_CORPUS_STATS="${H_CORPUS_STATS:-}"
FLAG="${FLAG:-}"
RUN_ID="${RUN_ID:-}"
RF_BATCH_SIZE="${RF_BATCH_SIZE:-1}"

# Phase D1 controller wiring (optional; only consumed by MODE=reference_flow).
# CONTROLLER_CONFIG empty → controller is disabled and the head flags are
# ignored. MODE=v1_entry refuses a non-empty value outright.
CONTROLLER_CONFIG_REQUESTED="${CONTROLLER_CONFIG:+requested}"
CONTROLLER_CONFIG="${CONTROLLER_CONFIG:-}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-}"
HEAD_CONFIG_DIR="${HEAD_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"
HEAD_VARIANT_ID="${HEAD_VARIANT_ID:-}"
HEAD_DEVICE="${HEAD_DEVICE:-cuda}"
HEAD_WINDOW_BATCH_SIZE="${HEAD_WINDOW_BATCH_SIZE:-}"
HEAD_ALLELE_IDX="${HEAD_ALLELE_IDX:-0}"
# Stage C.1 global-pressure calibration JSON (PLAN_RF_UNI_CTRL.md C1.3). Required
# only when the controller config has global_pressure.enabled=true; empty → not
# forwarded (the controller fails fast at load if the band is needed).
GLOBAL_PRESSURE_CALIBRATION_JSON="${GLOBAL_PRESSURE_CALIBRATION_JSON:-}"

# Uricase enzyme mode v0 hard-anchor constraint manifest (optional; consumed only
# by MODE=reference_flow and MODE=v1_entry). Empty → unconstrained global RF.
CONSTRAINT_MANIFEST="${CONSTRAINT_MANIFEST:-}"

# ---------------------------------------------------------------------------
# MODE=v1_entry (RF-Refine Fusion v1 entry) inputs. Declared here with defaults
# so `set -u` cannot abort mid-job on an unbound variable AFTER the allocation is
# granted; the branch itself fails fast with a named reason on the required ones.
# ---------------------------------------------------------------------------
ENTRY_CONFIG="${ENTRY_CONFIG:-}"
TERMINAL_REPAIR_CONFIG="${TERMINAL_REPAIR_CONFIG:-}"
RF_SAMPLER_CONFIG="${RF_SAMPLER_CONFIG:-}"
FUSION_CONFIG="${FUSION_CONFIG:-}"
# TERMINAL arm only: manifest.json of the COMPLETED pre-terminal run. M_T is derived
# from its persisted per-protein C_reserved, which is what makes the two arms
# compute-matched; the terminal arm may not invent its own budget.
PRETERMINAL_RESERVATION="${PRETERMINAL_RESERVATION:-}"
V1_PROTEINS="${V1_PROTEINS:-}"
V1_SUBMODE="${V1_SUBMODE:-run}"
S_STEPS="${S_STEPS:-}"
R_PARENT="${R_PARENT:-}"
N_ROUNDS="${N_ROUNDS:-}"
SECONDS_PER_REFOLD="${SECONDS_PER_REFOLD:-}"
SECONDS_PER_DFE="${SECONDS_PER_DFE:-}"
# T0-only definitive structure handoff (runbook §11.5): T0 folds its own 3*Q_T0 subset with the
# SAME esmfold2_live evaluator + cache identity v0 uses (P1 arms defer structure to v0, so these
# are unused there). Defaults mirror submit_refine.slurm.
REFOLD_CACHE_DIR="${REFOLD_CACHE_DIR:-}"
ESMFOLD2_SITE_PACKAGES="${ESMFOLD2_SITE_PACKAGES:-}"
ESMFOLD2_MODEL="${ESMFOLD2_MODEL:-biohub/ESMFold2}"
ESMFOLD2_NUM_LOOPS="${ESMFOLD2_NUM_LOOPS:-3}"
ESMFOLD2_NUM_SAMPLING_STEPS="${ESMFOLD2_NUM_SAMPLING_STEPS:-50}"
ESMFOLD2_NUM_DIFFUSION_SAMPLES="${ESMFOLD2_NUM_DIFFUSION_SAMPLES:-1}"
ESMFOLD2_SEED="${ESMFOLD2_SEED:-0}"
# Head epitope-window bounds: they select WHICH sub-sequences are scored, so they
# are method knobs bound into the run provenance, not driver literals.
WINDOW_K_MIN="${WINDOW_K_MIN:-12}"
WINDOW_K_MAX="${WINDOW_K_MAX:-25}"
# A partial cohort exits 3 by default. Set to 1 only when a deliberately incomplete cohort is the
# intent -- accepting one silently would let a downstream stage treat outcome-correlated survivors
# as the frozen split.
ALLOW_PARTIAL_COHORT="${ALLOW_PARTIAL_COHORT:-0}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

module purge
module load anaconda3/2025.12
conda activate "${CONDA_ENV}"
module load proxy/default

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}/inverse_folding/dplm/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# wandb storage roots — keep all run state on /scratch, never in $HOME or cwd.
export WANDB_DIR="${WANDB_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/data}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/scratch/gpfs/KAIYIJIANG/zijie/wandb/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CONFIG_DIR}"
WANDB_FLAG="${WANDB:-1}"
WANDB_MODE_FOR_RUN="${WANDB_MODE:-online}"
WANDB_PROJECT_FOR_RUN="${WANDB_PROJECT:-mhc-if-phase-c-sampling}"
WANDB_GROUP_FOR_RUN="${WANDB_GROUP:-}"
WANDB_TAGS_FOR_RUN="${WANDB_TAGS:-}"
export WANDB_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb"
export WANDB_CACHE_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/cache"
export WANDB_DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/data"
export WANDB_CONFIG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/wandb/config"


echo "============================================================"
echo "Phase C generation"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Mode            : ${MODE}"
echo "  Conda env       : ${CONDA_ENV}"
echo "  Node            : ${SLURM_NODELIST}"
echo "  GPUs            : ${SLURM_GPUS_ON_NODE:-1}"
echo "  Project root    : ${PROJECT_ROOT}"
echo "  Allele          : ${ALLELE}"
echo "  Checkpoint      : ${CHECKPOINT}"
echo "  Test parquet    : ${TEST_SET_PARQUET}"
echo "  PDB root        : ${PDB_ROOT}"
echo "  Output root     : ${OUTPUT_ROOT}"
echo "  Device          : ${DEVICE}"
echo "  Python          : $(python --version 2>&1)"
echo "  PyTorch         : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA avail      : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  GPU name        : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')"
echo "  Timestamp       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

WANDB_ARGS=()
if [[ "${WANDB_FLAG}" == "1" ]]; then
    WANDB_ARGS+=(--wandb)
    WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT_FOR_RUN}")
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE_FOR_RUN}")
    if [[ -n "${WANDB_GROUP_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-group "${WANDB_GROUP_FOR_RUN}")
    fi
    if [[ -n "${WANDB_TAGS_FOR_RUN}" ]]; then
        WANDB_ARGS+=(--wandb-tags "${WANDB_TAGS_FOR_RUN}")
    fi
fi
echo "  wandb           : ${WANDB_FLAG} (project=${WANDB_PROJECT_FOR_RUN} mode=${WANDB_MODE_FOR_RUN})"

case "${MODE}" in
    native)
        echo "  Designs/protein : ${N_DESIGNS_PER_PROTEIN}"
        echo "  Seed            : ${SEED}"
        echo "  Max iter        : ${MAX_ITER}"
        echo "  Temperature     : ${TEMPERATURE}"
        echo "  Sampling strat  : ${NATIVE_SAMPLING_STRATEGY}"
        echo "  Batch size      : ${NATIVE_BATCH_SIZE}"
        python -u "${PROJECT_ROOT}/scripts/run_if_phase_c0.py" \
            --checkpoint "${CHECKPOINT}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            --allele "${ALLELE}" \
            --output-root "${OUTPUT_ROOT}" \
            --n-designs-per-protein "${N_DESIGNS_PER_PROTEIN}" \
            --seed "${SEED}" \
            --max-iter "${MAX_ITER}" \
            --temperature "${TEMPERATURE}" \
            --sampling-strategy "${NATIVE_SAMPLING_STRATEGY}" \
            --device "${DEVICE}" \
            --batch-size "${NATIVE_BATCH_SIZE}" \
            --flag-name "${FLAG}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    reference_flow)
        echo "  h_maps parquet  : ${H_MAPS_PARQUET:-<none: h absent, g(h)=1>}"
        echo "  Config          : ${CONFIG_PATH}"
        echo "  Config stem     : $(basename "${CONFIG_PATH}" .yaml)"
        echo "  Fail threshold  : ${FAIL_PCT_THRESHOLD}"
        echo "  Save traj       : ${SAVE_TRAJECTORIES}"
        echo "  Designs/protein : ${N_DESIGNS_PER_PROTEIN:-<from config>}"
        echo "  Seed            : ${SEED:-<from config>}"
        echo "  N steps         : ${N_STEPS:-<from config>}"
        echo "  h corpus stats  : ${H_CORPUS_STATS:-<none>}"
        echo "  Resume from     : ${RESUME_FROM:-<fresh run>}"
        echo "  Run ID          : ${RUN_ID:-<auto>}"
        echo "  RF batch size   : ${RF_BATCH_SIZE}"
        echo "  Controller YAML : ${CONTROLLER_CONFIG:-<disabled>}"
        if [[ -n "${CONTROLLER_CONFIG}" ]]; then
            echo "  Head checkpoint : ${HEAD_CHECKPOINT:-<missing>}"
            echo "  Head config dir : ${HEAD_CONFIG_DIR}"
            echo "  Head variant    : ${HEAD_VARIANT_ID:-<missing>}"
            echo "  Head device     : ${HEAD_DEVICE}"
            echo "  Head WBS        : ${HEAD_WINDOW_BATCH_SIZE:-<predictor default>}"
            echo "  Head allele idx : ${HEAD_ALLELE_IDX}"
            echo "  GR calibration  : ${GLOBAL_PRESSURE_CALIBRATION_JSON:-<none / pressure off>}"
        fi
        EXTRA_FLAGS=()
        if [[ -n "${H_CORPUS_STATS}" ]]; then
            EXTRA_FLAGS+=(--h-corpus-stats "${H_CORPUS_STATS}")
        fi
        if [[ -n "${RESUME_FROM:-}" ]]; then
            EXTRA_FLAGS+=(--resume-from "${RESUME_FROM}")
        fi
        if [[ "${SAVE_TRAJECTORIES}" == "1" ]]; then
            EXTRA_FLAGS+=(--save-trajectories)
        fi
        if [[ -n "${N_DESIGNS_PER_PROTEIN:-}" ]]; then
            EXTRA_FLAGS+=(--n-designs-per-protein "${N_DESIGNS_PER_PROTEIN}")
        fi
        if [[ -n "${SEED:-}" ]]; then
            EXTRA_FLAGS+=(--seed "${SEED}")
        fi
        if [[ -n "${N_STEPS:-}" ]]; then
            EXTRA_FLAGS+=(--n-steps "${N_STEPS}")
        fi
        EXTRA_FLAGS+=(--batch-size "${RF_BATCH_SIZE}")
        if [[ -n "${RUN_ID}" ]]; then
            EXTRA_FLAGS+=(--run-id "${RUN_ID}")
        fi
        if [[ -n "${CONTROLLER_CONFIG}" ]]; then
            EXTRA_FLAGS+=(--controller-config "${CONTROLLER_CONFIG}")
            if [[ -n "${HEAD_CHECKPOINT}" ]]; then
                EXTRA_FLAGS+=(--head-checkpoint "${HEAD_CHECKPOINT}")
            fi
            EXTRA_FLAGS+=(--head-config-dir "${HEAD_CONFIG_DIR}")
            if [[ -n "${HEAD_VARIANT_ID}" ]]; then
                EXTRA_FLAGS+=(--head-variant-id "${HEAD_VARIANT_ID}")
            fi
            EXTRA_FLAGS+=(--head-device "${HEAD_DEVICE}")
            EXTRA_FLAGS+=(--head-allele-idx "${HEAD_ALLELE_IDX}")
            if [[ -n "${HEAD_WINDOW_BATCH_SIZE}" ]]; then
                EXTRA_FLAGS+=(--head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}")
            fi
            # Stage C.1 global-pressure calibration (PLAN_RF_UNI_CTRL.md C1.3).
            if [[ -n "${GLOBAL_PRESSURE_CALIBRATION_JSON}" ]]; then
                EXTRA_FLAGS+=(--global-pressure-calibration-json "${GLOBAL_PRESSURE_CALIBRATION_JSON}")
            fi
        fi
        # Uricase enzyme mode v0 hard-anchor constraint manifest (optional;
        # PLAN_URICASE_ENZYME_MODE). Forwarded only in MODE=reference_flow, with or
        # without a controller.
        if [[ -n "${CONSTRAINT_MANIFEST}" ]]; then
            EXTRA_FLAGS+=(--constraint-manifest "${CONSTRAINT_MANIFEST}")
        fi

        # Pass --h-maps-parquet only when H_MAPS_PARQUET is non-empty; an empty
        # value means "no h-map" and the flag is omitted entirely.
        H_MAPS_FLAG=()
        if [[ -n "${H_MAPS_PARQUET}" ]]; then
            H_MAPS_FLAG=(--h-maps-parquet "${H_MAPS_PARQUET}")
        fi

        python -u "${PROJECT_ROOT}/scripts/run_if_phase_c1.py" \
            --checkpoint "${CHECKPOINT}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            "${H_MAPS_FLAG[@]+"${H_MAPS_FLAG[@]}"}" \
            --allele "${ALLELE}" \
            --config "${CONFIG_PATH}" \
            --output-root "${OUTPUT_ROOT}" \
            --device "${DEVICE}" \
            --fail-pct-threshold "${FAIL_PCT_THRESHOLD}" \
            --flag-name "${FLAG}" \
            "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
            "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"
        ;;
    v1_entry)
        # T0 (phase=t0) folds its own 3*Q_T0 structure subset (runbook §11.5) with ESMFold2 via the
        # SAME offline model cache the v0 refold stage uses. The default HF_HOME above holds only the
        # phase-C models, NOT biohub/ESMFold2, so point HF_HOME/TORCH_HOME at the shared cache (which
        # also holds airkingbd/dplm_650m, so the entry-stage DPLM load is unaffected). The P1 arms
        # defer structure and never fold, so this is inert for them.
        export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/hf"
        export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
        # RF-Refine Fusion v1 entry (Arm-C P1; Head-only, NO NMP). Reuses this launcher's
        # DPLM + Head environment. The entry stage emits OUTPUT_ROOT/generated.parquet (the
        # ordered facade); the TERMINAL complete-state stage is a SEPARATE v0 run launched via
        # submit_refine.slurm MODE=fusion reading that parquet (PLAN §6.2-6.3). V1_SUBMODE
        # selects run | print_config | dry_run (the latter two load no model).
        #
        # NULL-RUNTIME FIREWALL (PLAN_RF_REFINE_FUSION_V1 §2.12). V1-A measures pre-terminal
        # reward allocation under a frozen null entry kernel: g(h)=1, controller=None, no
        # h-map. Both position-dependency pathways are refused here, at the launcher, because
        # H_MAPS_PARQUET resolves to the ALLELE DEFAULT above whenever the caller leaves it
        # unset -- forwarding it would silently turn this into a different method.
        if [[ -n "${H_MAPS_PARQUET_REQUESTED}" && -n "${H_MAPS_PARQUET}" ]]; then
            echo "MODE=v1_entry refuses H_MAPS_PARQUET=${H_MAPS_PARQUET}: V1-A runs g(h)=1 with no h-map" >&2
            exit 2
        fi
        if [[ -n "${CONTROLLER_CONFIG_REQUESTED}" ]]; then
            echo "MODE=v1_entry refuses CONTROLLER_CONFIG=${CONTROLLER_CONFIG}: V1-A runs controller=None" >&2
            exit 2
        fi
        H_MAPS_PARQUET=""
        CONTROLLER_CONFIG=""
        # The driver's launch gate (§3.4) refuses a null content digest for a missing model input,
        # so every one of these must be present BEFORE the job spends its allocation finding out.
        for _required in ENTRY_CONFIG TERMINAL_REPAIR_CONFIG RF_SAMPLER_CONFIG FUSION_CONFIG \
                         V1_PROTEINS S_STEPS R_PARENT N_ROUNDS HEAD_CHECKPOINT HEAD_CONFIG_DIR; do
            if [[ -z "${!_required}" ]]; then
                echo "MODE=v1_entry requires ${_required}" >&2
                exit 2
            fi
        done
        echo "  Entry config    : ${ENTRY_CONFIG}"
        echo "  Terminal config : ${TERMINAL_REPAIR_CONFIG}"
        echo "  Proteins        : ${V1_PROTEINS}"
        echo "  Entry kernel    : ${RF_SAMPLER_CONFIG}"
        echo "  Fusion config   : ${FUSION_CONFIG}"
        echo "  Preterm reserv. : ${PRETERMINAL_RESERVATION:-<n/a: pre-terminal arm>}"
        echo "  Entry runtime   : g(h)=1 (null kernel), controller=None, h-map=<refused>"
        echo "  Constraint mfst : ${CONSTRAINT_MANIFEST:-<unconstrained>}"
        echo "  Head window k   : ${WINDOW_K_MIN}..${WINDOW_K_MAX} batch=${HEAD_WINDOW_BATCH_SIZE:-<default>}"
        echo "  S/R/rounds      : ${S_STEPS}/${R_PARENT}/${N_ROUNDS}"
        echo "  Unit costs s    : refold=${SECONDS_PER_REFOLD:-<unset>} dfe=${SECONDS_PER_DFE:-<unset>}"
        echo "  V1 sub-mode     : ${V1_SUBMODE}"
        V1_FLAGS=(--rf-sampler-config "${RF_SAMPLER_CONFIG}" --fusion-config "${FUSION_CONFIG}")
        if [[ -n "${PRETERMINAL_RESERVATION}" ]]; then V1_FLAGS+=(--preterminal-reservation "${PRETERMINAL_RESERVATION}"); fi
        if [[ -n "${CONSTRAINT_MANIFEST}" ]]; then V1_FLAGS+=(--constraint-manifest "${CONSTRAINT_MANIFEST}"); fi
        if [[ -n "${HEAD_CHECKPOINT}" ]]; then V1_FLAGS+=(--head-checkpoint "${HEAD_CHECKPOINT}"); fi
        if [[ -n "${HEAD_CONFIG_DIR}" ]]; then V1_FLAGS+=(--head-config-dir "${HEAD_CONFIG_DIR}"); fi
        if [[ -n "${HEAD_VARIANT_ID}" ]]; then V1_FLAGS+=(--head-variant-id "${HEAD_VARIANT_ID}"); fi
        if [[ -n "${HEAD_WINDOW_BATCH_SIZE}" ]]; then V1_FLAGS+=(--head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}"); fi
        if [[ -n "${SECONDS_PER_REFOLD}" ]]; then V1_FLAGS+=(--seconds-per-refold "${SECONDS_PER_REFOLD}"); fi
        if [[ -n "${SECONDS_PER_DFE}" ]]; then V1_FLAGS+=(--seconds-per-dfe "${SECONDS_PER_DFE}"); fi
        # T0 definitive structure handoff (§11.5): forward the refold/ESMFold2 evaluator inputs.
        if [[ -n "${REFOLD_CACHE_DIR}" ]]; then V1_FLAGS+=(--refold-cache-dir "${REFOLD_CACHE_DIR}"); fi
        if [[ -n "${ESMFOLD2_SITE_PACKAGES}" ]]; then V1_FLAGS+=(--esmfold2-site-packages "${ESMFOLD2_SITE_PACKAGES}"); fi
        V1_FLAGS+=(--esmfold2-model "${ESMFOLD2_MODEL}" \
                   --esmfold2-num-loops "${ESMFOLD2_NUM_LOOPS}" \
                   --esmfold2-num-sampling-steps "${ESMFOLD2_NUM_SAMPLING_STEPS}" \
                   --esmfold2-num-diffusion-samples "${ESMFOLD2_NUM_DIFFUSION_SAMPLES}" \
                   --esmfold2-seed "${ESMFOLD2_SEED}")
        if [[ "${ALLOW_PARTIAL_COHORT}" == "1" ]]; then V1_FLAGS+=(--allow-partial-cohort); fi
        case "${V1_SUBMODE}" in
            print_config) V1_FLAGS+=(--print-config) ;;
            dry_run)      V1_FLAGS+=(--dry-run) ;;
            run)          ;;
            *) echo "Unsupported V1_SUBMODE=${V1_SUBMODE}; expected run|print_config|dry_run" >&2; exit 2 ;;
        esac
        # Belt and braces: the driver also refuses these flags, but a launcher that can emit
        # them at all is one edit away from re-enabling a position-dependent entry runtime.
        for _flag in "${V1_FLAGS[@]}"; do
            case "${_flag}" in
                --h-map|--h-maps-parquet|--controller-config|--h-corpus-stats)
                    echo "internal error: banned runtime flag ${_flag} in V1_FLAGS" >&2
                    exit 2
                    ;;
            esac
        done
        # --proteins is unquoted on purpose (nargs; space-separated protein ids).
        #
        # The driver's exit code carries the cohort outcome (0 = every requested protein produced
        # a terminal parent, 2 = nothing usable, 3 = partial cohort). `set -e` would abort here and
        # skip the summary, so the status is captured explicitly, reported, and re-raised as the
        # job's own exit code -- a partial cohort must never be reported to sbatch as success.
        set +e
        python -u "${PROJECT_ROOT}/scripts/run_rf_fusion_v1_entry.py" \
            --entry-config "${ENTRY_CONFIG}" \
            --terminal-repair-config "${TERMINAL_REPAIR_CONFIG}" \
            --proteins ${V1_PROTEINS} \
            --out-dir "${OUTPUT_ROOT}" \
            --logs-dir "${LOG_DIR}" \
            --allele "${ALLELE}" \
            --test-set-parquet "${TEST_SET_PARQUET}" \
            --pdb-root "${PDB_ROOT}" \
            --base-if-checkpoint "${CHECKPOINT}" \
            --head-allele-idx "${HEAD_ALLELE_IDX}" \
            --head-device "${HEAD_DEVICE}" \
            --window-k-min "${WINDOW_K_MIN}" \
            --window-k-max "${WINDOW_K_MAX}" \
            --s-steps "${S_STEPS}" \
            --r-parent "${R_PARENT}" \
            --n-rounds "${N_ROUNDS}" \
            "${V1_FLAGS[@]}"
        V1_RC=$?
        set -e
        echo "Phase C v1_entry exit=${V1_RC} (0=cohort complete, 2=nothing produced, 3=partial)"
        exit "${V1_RC}"
        ;;
    *)
        echo "Unsupported MODE=${MODE}; expected native, reference_flow, or v1_entry" >&2
        exit 2
        ;;
esac

echo "Phase C ${MODE} finished."
