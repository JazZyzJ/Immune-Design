#!/bin/bash
#SBATCH --job-name=sigdiag
#SBATCH --account=pli_x
#SBATCH --qos=pli-low
#SBATCH --partition=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/signal_diag/%x_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/signal_diag/%x_%j.err

# SC-GR signal-direction diagnostic (PLAN_RF_SC_GR_signal_diag.md), P1/P2.
#   MODE=snapshot : replay B1 design-0 for each P0 protein, dump pre-D2 snapshots
#                   + decision_index (cluster GPU).
#   MODE=score    : reconstruct candidate pool + L/P, run K_Y oracle Y completions,
#                   write signal_diag_candidates.parquet (cluster GPU).
#   MODE=c0b      : Path-C headroom existence test (PLANNER_README §C0b) — one job:
#                   (1) select high-r_i registers from finished designs + run the
#                   single-position arm vs multi-step reconditioned joint resample,
#                   (2) ESMFold scTM on the shortlist facade (pli/h100, NOT mig),
#                   (3) GO gate. Writes c0b_candidates/registers/verdicts + c0b_gate.json.
# PLI/H100 by default to match the B1 source run's compute profile (B1 replay runs
# the full D2 4096-candidate head scoring per refresh; heavier than plain C1).

set -euo pipefail
: "${PS1:=}"
export PS1

PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
SCRATCH="/scratch/gpfs/KAIYIJIANG/zijie"
WORK_DIR="${WORK_DIR:-${SCRATCH}/work/immune-design}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/run/inverse_folding}"
export PROJECT_ROOT WORK_DIR RUN_DIR

module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}/inverse_folding/dplm/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="${SCRATCH}/model_cache/torch"
# Compute nodes are offline: point HF at the pre-cached store (dplm_650m base)
# and force offline so model load never tries to reach huggingface.co.
export HF_HOME="${SCRATCH}/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# ---- knobs (env-overridable) ----
MODE="${MODE:-snapshot}"
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
HEAD_DEVICE="${HEAD_DEVICE:-cuda}"

SRC_RUN="${SRC_RUN:-${SCRATCH}/run/inverse_folding/phase_d/stageB1/highrisk/HLA-DRB1_07_01/stageB1_highrisk_dplm_iter_aopen_beta5p0_n4_b4_seed42_pli_20260620T064542Z}"
P0_PARQUET="${P0_PARQUET:-${SCRATCH}/work/immune-design/rar_analysis_archive/current/Results/Analysis/signal_diag/p0_decision_points.parquet}"
CHECKPOINT="${CHECKPOINT:-${SCRATCH}/run/inverse_folding/dplm_v1_adapter/seed42_20260319_094244/checkpoints/best.ckpt}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${SCRATCH}/work/immune-design/if_test_set/if_ready/highrisk/highrisk_nod_v1_HLA-DRB1_07_01.parquet}"
PDB_ROOT="${PDB_ROOT:-${SCRATCH}/work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_07_01}"
H_MAPS_PARQUET="${H_MAPS_PARQUET:-${SCRATCH}/work/immune-design/if_test_set/if_ready/h_maps_v2/h_maps_HLA-DRB1_07_01.parquet}"
CONTROLLER_CONFIG="${CONTROLLER_CONFIG:-${SCRATCH}/work/RF/HLA-DRB1_07_01/experiment_configs/20260616T014400Z/d2_d3_full_stageB_aopen_beta5p0.yaml}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${SCRATCH}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt}"
HEAD_CONFIG_DIR="${HEAD_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"
HEAD_VARIANT_ID="${HEAD_VARIANT_ID:-LC1}"
HEAD_ALLELE_IDX="${HEAD_ALLELE_IDX:-0}"
HEAD_WINDOW_BATCH_SIZE="${HEAD_WINDOW_BATCH_SIZE:-4096}"

OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
SNAPSHOT_RUN_DIR="${SNAPSHOT_RUN_DIR:-}"
K_PROBE="${K_PROBE:-3}"
K_Y="${K_Y:-3}"
MAX_CANDIDATES_Y="${MAX_CANDIDATES_Y:-16}"
Y_BASE_SEED="${Y_BASE_SEED:-1000}"
PROBE_AGGREGATION="${PROBE_AGGREGATION:-topm_lse}"   # topm_lse | omega_lme (Y-aligned)
EMIT_Y_FORKS="${EMIT_Y_FORKS:-0}"                    # 1 -> write y_forks.parquet (split-half)
DECISION_SUBSET="${DECISION_SUBSET:-}"               # path to decision_id allowlist (optional)
PCOND_COLUMN="${PCOND_COLUMN:-P_cond}"               # cond_probe output column name

# ---- c0b knobs (env-overridable) ----
DESIGNS_PARQUET="${DESIGNS_PARQUET:-${SRC_RUN}/generated.parquet}"
N_REGISTERS="${N_REGISTERS:-40}"
REGISTER_WIDTH="${REGISTER_WIDTH:-9}"
REGISTERS_PER_DESIGN="${REGISTERS_PER_DESIGN:-2}"
REGISTER_QUANTILE="${REGISTER_QUANTILE:-0.6667}"
K_RESAMPLE="${K_RESAMPLE:-32}"
JOINT_MAX_ITER="${JOINT_MAX_ITER:-18}"
JOINT_MAX_ITER_DEEP="${JOINT_MAX_ITER_DEEP:-36}"     # true-joint depth sweep arm
SHORTLIST_K="${SHORTLIST_K:-3}"
PNLL_MARGIN="${PNLL_MARGIN:-2.0}"
RESAMPLE_TEMPERATURE="${RESAMPLE_TEMPERATURE:-1.0}"
C0B_SEED="${C0B_SEED:-0}"
DESIGN_IDX="${DESIGN_IDX:-}"                          # empty = lowest design_idx per protein
COORDINATION_MARGIN_THRESH="${COORDINATION_MARGIN_THRESH:-0.2}"  # true must beat fake by >= this (nats)
SCTM_TOL="${SCTM_TOL:-0.05}"                          # structure non-inferiority slack vs design scTM
SCTM_FLOOR="${SCTM_FLOOR:-0.5}"
WIN_FRACTION="${WIN_FRACTION:-0.3334}"               # Path-C GO iff >= this fraction of registers win

SOURCE_RUN_CONFIG="${SOURCE_RUN_CONFIG:-${SRC_RUN}/run_config.yaml}"

EXTRA_ARGS=( --probe-aggregation "${PROBE_AGGREGATION}" )
[ -n "${DECISION_SUBSET}" ] && EXTRA_ARGS+=( --decision-subset "${DECISION_SUBSET}" )
[ "${EMIT_Y_FORKS}" = "1" ] && EXTRA_ARGS+=( --emit-y-forks )

COMMON_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --checkpoint "${CHECKPOINT}"
  --source-run-config "${SOURCE_RUN_CONFIG}"
  --test-set-parquet "${TEST_SET_PARQUET}"
  --pdb-root "${PDB_ROOT}"
  --h-maps-parquet "${H_MAPS_PARQUET}"
  --allele "${ALLELE}"
  --device "${DEVICE}"
  --controller-config "${CONTROLLER_CONFIG}"
  --head-checkpoint "${HEAD_CHECKPOINT}"
  --head-config-dir "${HEAD_CONFIG_DIR}"
  --head-variant-id "${HEAD_VARIANT_ID}"
  --head-allele-idx "${HEAD_ALLELE_IDX}"
  --head-device "${HEAD_DEVICE}"
  --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}"
)

if [ "${MODE}" = "snapshot" ]; then
  srun python -u "${PROJECT_ROOT}/scripts/run_if_signal_diag.py" \
    --mode snapshot \
    --p0-parquet "${P0_PARQUET}" \
    "${COMMON_ARGS[@]}"
elif [ "${MODE}" = "score" ]; then
  srun python -u "${PROJECT_ROOT}/scripts/run_if_signal_diag.py" \
    --mode score \
    --snapshot-run-dir "${SNAPSHOT_RUN_DIR:?set SNAPSHOT_RUN_DIR for score mode}" \
    --k-probe "${K_PROBE}" \
    --k-y "${K_Y}" \
    --max-candidates-y "${MAX_CANDIDATES_Y}" \
    --y-base-seed "${Y_BASE_SEED}" \
    "${EXTRA_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
elif [ "${MODE}" = "cond_probe" ]; then
  # Augment an existing signal_diag_candidates.parquet with the conditioned-probe
  # P_cond column (re-run denoiser per committed candidate, then probe). Reuses P1
  # snapshots + the deterministic pool; does NOT re-run the oracle Y.
  srun python -u "${PROJECT_ROOT}/scripts/run_if_signal_diag.py" \
    --mode cond_probe \
    --snapshot-run-dir "${SNAPSHOT_RUN_DIR:?set SNAPSHOT_RUN_DIR for cond_probe mode}" \
    --candidates-parquet "${CANDIDATES_PARQUET:?set CANDIDATES_PARQUET for cond_probe mode}" \
    --k-probe "${K_PROBE}" \
    --y-base-seed "${Y_BASE_SEED}" \
    --pcond-column "${PCOND_COLUMN}" \
    "${EXTRA_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
elif [ "${MODE}" = "c0b" ]; then
  # Path-C headroom existence test, one self-contained job (PLANNER_README §C0b).
  C0B_ARGS=(
    --designs-parquet "${DESIGNS_PARQUET}"
    --n-registers "${N_REGISTERS}"
    --register-width "${REGISTER_WIDTH}"
    --registers-per-design "${REGISTERS_PER_DESIGN}"
    --register-quantile "${REGISTER_QUANTILE}"
    --k-resample "${K_RESAMPLE}"
    --joint-max-iter "${JOINT_MAX_ITER}"
    --joint-max-iter-deep "${JOINT_MAX_ITER_DEEP}"
    --shortlist-k "${SHORTLIST_K}"
    --pnll-margin "${PNLL_MARGIN}"
    --resample-temperature "${RESAMPLE_TEMPERATURE}"
    --seed "${C0B_SEED}"
  )
  [ -n "${DESIGN_IDX}" ] && C0B_ARGS+=( --design-idx "${DESIGN_IDX}" )
  # (1) select registers + single-position arm vs joint resample + cheap pNLL
  srun python -u "${PROJECT_ROOT}/scripts/run_if_signal_diag.py" \
    --mode c0b \
    "${C0B_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
  # (2) ESMFold scTM on the shortlist facade (refold on pli/h100, NEVER mig)
  srun python -u "${PROJECT_ROOT}/scripts/evaluate_phase_c.py" \
    --mode struct \
    --generated-parquet "${OUTPUT_DIR}/c0b_shortlist_facade.parquet" \
    --test-set-parquet "${TEST_SET_PARQUET}" \
    --allele "${ALLELE}" \
    --output-root "${OUTPUT_DIR}/struct" \
    --pdb-root "${PDB_ROOT}" \
    --refold-model esmfold \
    --device "${DEVICE}" \
    --run-id c0b_struct \
    --tag c0b
  # (3) GO gate (join scTM -> per-register verdict -> GO fraction)
  STRUCT_PARQUET="$(find "${OUTPUT_DIR}/struct" -path '*/c0b_struct/structural.parquet' | head -1)"
  [ -n "${STRUCT_PARQUET}" ] || { echo "ERROR: structural.parquet not found under ${OUTPUT_DIR}/struct" >&2; exit 3; }
  srun python -u "${PROJECT_ROOT}/scripts/analysis/c0b_gate.py" \
    --candidates-parquet "${OUTPUT_DIR}/c0b_candidates.parquet" \
    --structural-parquet "${STRUCT_PARQUET}" \
    --registers-parquet "${OUTPUT_DIR}/c0b_registers.parquet" \
    --output "${OUTPUT_DIR}/c0b_gate.json" \
    --verdicts-output "${OUTPUT_DIR}/c0b_verdicts.parquet" \
    --coordination-margin-thresh "${COORDINATION_MARGIN_THRESH}" \
    --sctm-tol "${SCTM_TOL}" \
    --sctm-floor "${SCTM_FLOOR}" \
    --win-fraction "${WIN_FRACTION}"
else
  echo "ERROR: MODE must be snapshot|score|cond_probe|c0b (got ${MODE})" >&2
  exit 2
fi
