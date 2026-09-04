#!/bin/bash
#SBATCH --job-name=rf-refine
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint="h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/refine/refine_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/refine/refine_%j.err

# RF refinement driver (PLAN_RF_REFINE.md) — one GPU job runs all three oracles:
#   immune head (GPU) + NetMHCIIpan (CPU, --nmp-workers==cpus-per-task) + ESMFold2 refold (GPU).
# Targets ailab H200 (8 cores/GPU); NMP_WORKERS defaults to 8 to match --cpus-per-task=8.
# PLI H100 fallback (busy): resubmit with
#   sbatch --account=pli_x --qos=pli-low --partition=pli --constraint= scripts/submit_refine.slurm
#
# Input is a self-contained --seed-table (protein_id/design_id/sequence parquet); the
# whole listed set is refined (PROTEINS filters it). Override any UPPER_CASE var at submit.
#   SEED_TABLE=<1-row.parquet> PROTEINS=Q0CXR4 OUT_DIR=<dir> sbatch scripts/submit_refine.slurm  # 1-seed smoke
#   sbatch --array=0-23%12 scripts/submit_refine.slurm          # full 123-seed run, 24 balanced shards
# Plain `sbatch` = one task = all seeds (no array). With `--array`, N_SHARDS auto-derives from
# SLURM_ARRAY_TASK_COUNT and each task refines its LPT-balanced shard (by nmp_n_strong_binders).
# Acceleration: incremental NMP splice (exact, ~6.5x fewer windows) is ON by default in the driver;
# NMP_MAX_LENGTHS_PER_CALL=2 keeps all 8 cores busy.
# Outputs (refined/): refined_designs.parquet + evaluator_ready.parquet + refine_trace.parquet, PLUS
# (EMIT_EVAL_METRICS=1, default) the evaluate_phase_c-schema tables imm_head/imm_nmp/structural/
# structural_residues for every final design (reuses evaluate row builders; refold cache hits => cheap).

set -euo pipefail
: "${PS1:=}"
export PS1

# ── Paths / model artifacts (override via env) ───────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
BASE="${BASE:-/scratch/gpfs/KAIYIJIANG/zijie}"
DPLM_ROOT="${DPLM_ROOT:-${PROJECT_ROOT}/inverse_folding/dplm}"
# MODE=fusion repair/reopen loads the DPLM base-IF task, whose .hydra/config.yaml resolves
# ${oc.env:PROJECT_ROOT}; export it (mirrors submit_if_phase_c.slurm) or that load KeyErrors.
export PROJECT_ROOT
C24_INPUTS="${C24_INPUTS:-${BASE}/work/RF/Uricases_RF/characterized24_inputs/20260703T201539Z}"
LOG_DIR="${LOG_DIR:-${BASE}/logs/refine}"

MODE="${MODE:-refine}"                       # refine | ceiling | fusion
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
PROTEINS="${PROTEINS:-all}"
TARGET_SOURCE="${TARGET_SOURCE:-nmp}"
OFFICIAL="${OFFICIAL:-0}"
FINAL_CANDIDATES_PER_PROTEIN="${FINAL_CANDIDATES_PER_PROTEIN:-}"

# MODE=fusion (Head-only RF-Refine Fusion) reads complete RF designs and runs the edit-and-repair
# loop; it does NOT use NetMHCIIpan. The static #SBATCH --cpus-per-task=8 is NMP-sized — a fusion
# run needs fewer cores, so submit with e.g. `MODE=fusion sbatch --cpus-per-task=2 ...`.
FUSION_CONFIG="${FUSION_CONFIG:-${PROJECT_ROOT}/inverse_folding/reference_flow/configs/rf_refine_fusion_smoke.yaml}"
RUN_DIR="${RUN_DIR:-}"                        # directory holding the generated designs parquet
# H3 (rf_reopen / edit-repair) needs the DPLM sampler: pass BOTH to enable the repair arm.
# Leave empty for the explicit-only S0 arm (the driver only requires them when repair/reopen on).
BASE_IF_CHECKPOINT="${BASE_IF_CHECKPOINT:-}"  # base inverse-folding checkpoint for the repair sampler
RF_SAMPLER_CONFIG="${RF_SAMPLER_CONFIG:-}"    # reference-flow sampler YAML (temperature/remask base)
ESMFOLD2_ENV="${ESMFOLD2_ENV:-${HOME}/.conda/envs/esmfold2}"
ESMFOLD2_SITE_PACKAGES="${ESMFOLD2_SITE_PACKAGES:-}"
ESMFOLD2_MODEL="${ESMFOLD2_MODEL:-biohub/ESMFold2}"
ESMFOLD2_NUM_LOOPS="${ESMFOLD2_NUM_LOOPS:-3}"
ESMFOLD2_NUM_SAMPLING_STEPS="${ESMFOLD2_NUM_SAMPLING_STEPS:-50}"
ESMFOLD2_NUM_DIFFUSION_SAMPLES="${ESMFOLD2_NUM_DIFFUSION_SAMPLES:-1}"
ESMFOLD2_SEED="${ESMFOLD2_SEED:-0}"
REFOLD_MODEL="${REFOLD_MODEL:-esmfold2_live}"

# Seeds: a --seed-table parquet (protein_id/design_id/sequence). Default = the full
# difficulty-stratified 0701 validation table; override SEED_TABLE for the smoke.
SEED_TABLE="${SEED_TABLE:-${BASE}/work/RF/Uricases_RF/refine_seed_selection_0701/refine_seeds_0701.parquet}"

# Immune head (a1res03 cv5 fold0 — the head the char24 generation used).
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${BASE}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt}"
HEAD_CONFIG_DIR="${HEAD_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"
HEAD_VARIANT_ID="${HEAD_VARIANT_ID:-LC1}"
HEAD_ALLELE_IDX="${HEAD_ALLELE_IDX:-0}"
HEAD_WINDOW_BATCH_SIZE="${HEAD_WINDOW_BATCH_SIZE:-4096}"
HEAD_RESIDUE_THRESHOLD="${HEAD_RESIDUE_THRESHOLD:-0.15}"
HEAD_MAX_TARGET_POSITIONS="${HEAD_MAX_TARGET_POSITIONS:-12}"
HEAD_AA_PER_POSITION="${HEAD_AA_PER_POSITION:-2}"

# NetMHCIIpan (CPU). NMP_WORKERS matches --cpus-per-task so the 8 ailab cores are saturated.
NETMHCIIPAN_BIN="${NETMHCIIPAN_BIN:-${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan}"
NMP_WORKERS="${NMP_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
NMP_MAX_LENGTHS_PER_CALL="${NMP_MAX_LENGTHS_PER_CALL:-2}"   # 14 lengths -> 7 groups ~= 8 cores
NMP_TIMEOUT="${NMP_TIMEOUT:-1800}"

# Seed job-array sharding: N_SHARDS auto-derives from --array (SLURM_ARRAY_TASK_COUNT); a plain
# sbatch (no array) => 1 shard = all seeds. Balanced (LPT) by the seed-table difficulty column.
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_IDX="${SHARD_IDX:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_BY="${SHARD_BY:-balanced}"
SHARD_COST_COL="${SHARD_COST_COL:-nmp_n_strong_binders}"

# Structure: WT/target backbones + the test-set rows resolve_structure_path needs.
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${C24_INPUTS}/uricase_characterized24_if_ready.parquet}"
PDB_ROOT="${PDB_ROOT:-${BASE}/work/immune-design/if_test_set/uricases/pdbs_if_ready}"
CONSTRAINT_MANIFEST="${CONSTRAINT_MANIFEST:-${C24_INPUTS}/uricase_characterized24_active_site_manifest.yaml}"

# Search knobs (PLAN §7).
STRONG_RANK="${STRONG_RANK:-0.02}"
MARGIN_BAND="${MARGIN_BAND:-0.10}"
STRUCTURE_GATE_PROFILE="${STRUCTURE_GATE_PROFILE:-protocol}"
# Protocol gate thresholds are intentionally unset: calibrate them per run and pass all three.
GATE_SCTM_MIN="${GATE_SCTM_MIN:-}"
GATE_CAT_MAX_SCRMSD_MAX="${GATE_CAT_MAX_SCRMSD_MAX:-}"
GATE_ACTIVE_SITE_MIN_PLDDT_MIN="${GATE_ACTIVE_SITE_MIN_PLDDT_MIN:-}"
# Explicit compatibility path only (`STRUCTURE_GATE_PROFILE=legacy`).
SCTM_EPS="${SCTM_EPS:-}"
TOPB="${TOPB:-64}"                           # head-prune to top-B before NMP (v0 refine default).
                                             # NMP-all (empty) proved infeasible on multi-core seeds
                                             # (pool 2208 -> 60min NMP -> walltime kill). Set TOPB= (empty)
                                             # only for a small few-core ceiling probe.
REFOLD_CAP="${REFOLD_CAP:-16}"               # max count-droppers refolded per round (bounds GPU)
BEAM_WIDTH="${BEAM_WIDTH:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
PATIENCE="${PATIENCE:-3}"
MAX_PAIRS="${MAX_PAIRS:-200}"
MAX_PATH_MUTATIONS="${MAX_PATH_MUTATIONS:-8}"

# Full-metrics emission (§ evaluate parity): every mode emits Head + structure tables;
# only TARGET_SOURCE=nmp emits imm_nmp. Refolds are cache hits. On by default.
EMIT_EVAL_METRICS="${EMIT_EVAL_METRICS:-1}"           # 0 => --no-eval-metrics
STRONG_BINDER_THRESHOLD="${STRONG_BINDER_THRESHOLD:-2.0}"  # imm_nmp rank_EL% strong cutoff (evaluate default)
HOTSPOT_THRESHOLD="${HOTSPOT_THRESHOLD:-0.5}"         # imm_head hotspot cutoff (evaluate default)
IMM_FULL="${IMM_FULL:-0}"                             # 1 => also imm_head_residues + imm_nmp_peptides
REFINEMENT_STRUCTURE_METRICS="${REFINEMENT_STRUCTURE_METRICS:-global_ca_rmsd,plddt,sidechain}"
MAX_ANCHOR_SIDECHAIN_RMSD_MAX="${MAX_ANCHOR_SIDECHAIN_RMSD_MAX:-}"

# Output run dir + per-run normalized refold cache (writes <key>.pdb/.plddt -> gives pdb_path).
# Base tag is stable across an array (SLURM_ARRAY_JOB_ID) so all tasks share one run dir; each
# task then writes to its own shardXXofNN/ subdir (no clobbering the shared refined/*.parquet).
TS="$(date +%Y%m%dT%H%M%SZ)"
JOB_TAG="${JOB_TAG:-refine_${MODE}_0701}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_TAG="${JOB_TAG}_A${SLURM_ARRAY_JOB_ID}"
else
    RUN_TAG="${JOB_TAG}_${TS}"
fi
OUT_DIR="${OUT_DIR:-${BASE}/run/inverse_folding/refine/${RUN_TAG}}"
if [ "${N_SHARDS}" -gt 1 ]; then
    OUT_DIR="${OUT_DIR}/shard$(printf '%02d' "${SHARD_IDX}")of${N_SHARDS}"
fi
REFOLD_CACHE_DIR="${REFOLD_CACHE_DIR:-${ESMFOLD_CACHE_DIR:-${OUT_DIR}/${REFOLD_MODEL}_cache}}"
ESMFOLD_CACHE_DIR="${ESMFOLD_CACHE_DIR:-${REFOLD_CACHE_DIR}}"  # deprecated env alias

mkdir -p "${OUT_DIR}" "${REFOLD_CACHE_DIR}" "${LOG_DIR}"

# ── Environment (mirrors submit_if_phase_c.slurm) ────────────────────────
# CONDA_ENV is overridable for the same reason submit_if_phase_c.slurm exposes it: the RTX PRO 6000
# (Blackwell, sm_120) partition needs `immune-design-blackwell` (torch 2.7.1+cu128), while
# `immune-design` (torch 2.5.1+cu121, sm<=90) is correct for ailab/H200, PLI/H100 and A100. A
# hardcoded env silently aborts on Blackwell AFTER the allocation is granted.
CONDA_ENV="${CONDA_ENV:-immune-design}"
module purge
module load anaconda3/2025.12
conda activate "${CONDA_ENV}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce head/ESMFold GPU fragmentation
export TORCH_HOME="${BASE}/model_cache/torch"           # ESMFold weights (esmfold_v1)
export HF_HOME="${HF_HOME:-${BASE}/model_cache/hf}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# TOPB flag is only emitted when non-empty (empty = NMP-all ceiling behavior).
TOPB_FLAG=()
[ -n "${TOPB}" ] && TOPB_FLAG=(--topB "${TOPB}")
OFFICIAL_FLAG=()
[ "${OFFICIAL}" = "1" ] && OFFICIAL_FLAG=(--official)
FINAL_CANDIDATES_FLAG=()
[ -n "${FINAL_CANDIDATES_PER_PROTEIN}" ] && FINAL_CANDIDATES_FLAG=(
    --final-candidates-per-protein "${FINAL_CANDIDATES_PER_PROTEIN}"
)

# Post-refine full-metrics emission flags (on by default; EMIT_EVAL_METRICS=0 to skip).
EVAL_METRICS_FLAG=(--emit-eval-metrics)
[ "${EMIT_EVAL_METRICS}" = "0" ] && EVAL_METRICS_FLAG=(--no-eval-metrics)
IMM_FULL_FLAG=()
[ "${IMM_FULL}" = "1" ] && IMM_FULL_FLAG=(--imm-full)
SIDECHAIN_GATE_FLAG=()
[ -n "${MAX_ANCHOR_SIDECHAIN_RMSD_MAX}" ] && SIDECHAIN_GATE_FLAG=(
    --max-anchor-sidechain-RMSD-max "${MAX_ANCHOR_SIDECHAIN_RMSD_MAX}"
)
STRUCTURE_GATE_FLAGS=()
if [ "${MODE}" != "fusion" ]; then
    if [ "${STRUCTURE_GATE_PROFILE}" = "protocol" ]; then
        if [ -z "${GATE_SCTM_MIN}" ] || [ -z "${GATE_CAT_MAX_SCRMSD_MAX}" ] || [ -z "${GATE_ACTIVE_SITE_MIN_PLDDT_MIN}" ]; then
            echo "FATAL: protocol gate requires GATE_SCTM_MIN, GATE_CAT_MAX_SCRMSD_MAX, and GATE_ACTIVE_SITE_MIN_PLDDT_MIN" >&2
            exit 2
        fi
        STRUCTURE_GATE_FLAGS=(
            --structure-gate-profile protocol
            --gate-scTM-min "${GATE_SCTM_MIN}"
            --gate-cat-max-scRMSD-max "${GATE_CAT_MAX_SCRMSD_MAX}"
            --gate-predicted-active-site-min-pLDDT-min "${GATE_ACTIVE_SITE_MIN_PLDDT_MIN}"
        )
    elif [ "${STRUCTURE_GATE_PROFILE}" = "legacy" ]; then
        STRUCTURE_GATE_FLAGS=(--structure-gate-profile legacy)
        [ -n "${SCTM_EPS}" ] && STRUCTURE_GATE_FLAGS+=(--scTM-eps "${SCTM_EPS}")
    else
        echo "FATAL: STRUCTURE_GATE_PROFILE must be protocol or legacy" >&2
        exit 2
    fi
fi

# Active-site manifest is optional: pass CONSTRAINT_MANIFEST=none to disable (no active site,
# e.g. non-enzyme highrisk inpaint runs) — the flag is then omitted and all positions are editable.
CONSTRAINT_FLAG=(--constraint-manifest "${CONSTRAINT_MANIFEST}")
[ "${CONSTRAINT_MANIFEST}" = "none" ] && CONSTRAINT_FLAG=()
NMP_RUNTIME="binary=${NETMHCIIPAN_BIN} workers=${NMP_WORKERS}"
[ "${TARGET_SOURCE}" = "head" ] && NMP_RUNTIME="disabled"

echo "============================================================"
echo "RF Refinement"
echo "  Job ID        : ${SLURM_JOB_ID:-<none>}"
echo "  Node          : ${SLURM_NODELIST:-<none>}"
echo "  GPU name      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo N/A)"
echo "  CPUs          : ${SLURM_CPUS_PER_TASK:-8}"
echo "  Mode          : ${MODE}   target-source: ${TARGET_SOURCE}"
echo "  Official      : ${OFFICIAL}   final candidates/protein: ${FINAL_CANDIDATES_PER_PROTEIN:-default}"
echo "  Head targeting: residue local-max >= ${HEAD_RESIDUE_THRESHOLD}; cap=${HEAD_MAX_TARGET_POSITIONS}; AA/pos=${HEAD_AA_PER_POSITION}"
echo "  Shard         : ${SHARD_IDX} / ${N_SHARDS}  (by ${SHARD_BY}:${SHARD_COST_COL})"
echo "  NMP runtime   : ${NMP_RUNTIME}"
echo "  Allele        : ${ALLELE}"
echo "  Seed table    : ${SEED_TABLE}"
echo "  Proteins      : ${PROTEINS}"
echo "  Head ckpt     : ${HEAD_CHECKPOINT}"
echo "  Test set      : ${TEST_SET_PARQUET}"
echo "  PDB root      : ${PDB_ROOT}"
echo "  Constraints   : ${CONSTRAINT_MANIFEST}"
echo "  Out dir       : ${OUT_DIR}"
echo "  Refold cache  : ${REFOLD_CACHE_DIR}"
echo "  Refold model  : ${REFOLD_MODEL}"
echo "  Eval metrics  : emit=${EMIT_EVAL_METRICS} strong=${STRONG_BINDER_THRESHOLD} hotspot=${HOTSPOT_THRESHOLD} imm_full=${IMM_FULL}"
echo "  Structure v2  : canonical; loop=${REFINEMENT_STRUCTURE_METRICS} gate=${STRUCTURE_GATE_PROFILE}"
echo "  Protocol gate : scTM>=${GATE_SCTM_MIN:-unset} cat_max<=${GATE_CAT_MAX_SCRMSD_MAX:-unset} active_site_min_pLDDT>=${GATE_ACTIVE_SITE_MIN_PLDDT_MIN:-unset}"
echo "============================================================"

if [ "${MODE}" = "fusion" ]; then
    # Head-only Fusion driver — NO --netmhciipan-bin / --nmp-* / --seed-table.
    # repair/reopen (H3) sampler flags: only forwarded when both are set (explicit-only S0 omits them).
    REPAIR_FLAGS=()
    if [ -n "${BASE_IF_CHECKPOINT}" ]; then REPAIR_FLAGS+=(--base-if-checkpoint "${BASE_IF_CHECKPOINT}"); fi
    if [ -n "${RF_SAMPLER_CONFIG}" ]; then REPAIR_FLAGS+=(--rf-sampler-config "${RF_SAMPLER_CONFIG}"); fi
    # The worker uses immune-design's current Python + torch. Only Biohub's
    # esm/transformers source path is overlaid in the isolated worker process.
    if [ -z "${ESMFOLD2_SITE_PACKAGES}" ]; then
        if [ ! -x "${ESMFOLD2_ENV}/bin/python" ]; then
            echo "FATAL: ESMFold2 env python not found: ${ESMFOLD2_ENV}/bin/python" >&2
            exit 2
        fi
        ESMFOLD2_SITE_PACKAGES="$("${ESMFOLD2_ENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    fi
    echo "  ESMFold2 code : ${ESMFOLD2_SITE_PACKAGES}"
    echo "  ESMFold2 model: ${ESMFOLD2_MODEL} loops/steps/samples=${ESMFOLD2_NUM_LOOPS}/${ESMFOLD2_NUM_SAMPLING_STEPS}/${ESMFOLD2_NUM_DIFFUSION_SAMPLES}"
    srun python "${PROJECT_ROOT}/scripts/run_rf_refine_fusion.py" \
        --fusion-config "${FUSION_CONFIG}" \
        --run-dir "${RUN_DIR}" \
        --proteins ${PROTEINS} \
        --allele "${ALLELE}" \
        "${CONSTRAINT_FLAG[@]}" \
        --test-set-parquet "${TEST_SET_PARQUET}" \
        --pdb-root "${PDB_ROOT}" \
        --head-checkpoint "${HEAD_CHECKPOINT}" \
        --head-config-dir "${HEAD_CONFIG_DIR}" \
        --head-variant-id "${HEAD_VARIANT_ID}" \
        --head-device cuda \
        --head-allele-idx "${HEAD_ALLELE_IDX}" \
        --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
        --refold-cache-dir "${REFOLD_CACHE_DIR}" \
        --esmfold2-site-packages "${ESMFOLD2_SITE_PACKAGES}" \
        --esmfold2-model "${ESMFOLD2_MODEL}" \
        --esmfold2-num-loops "${ESMFOLD2_NUM_LOOPS}" \
        --esmfold2-num-sampling-steps "${ESMFOLD2_NUM_SAMPLING_STEPS}" \
        --esmfold2-num-diffusion-samples "${ESMFOLD2_NUM_DIFFUSION_SAMPLES}" \
        --esmfold2-seed "${ESMFOLD2_SEED}" \
        "${REPAIR_FLAGS[@]}" \
        --out-dir "${OUT_DIR}" \
        --print-config
    echo "[fusion] done → ${OUT_DIR}"
    exit 0
fi

ESMFOLD2_FLAGS=()
if [ "${REFOLD_MODEL}" = "esmfold2_live" ]; then
    if [ -z "${ESMFOLD2_SITE_PACKAGES}" ]; then
        if [ ! -x "${ESMFOLD2_ENV}/bin/python" ]; then
            echo "FATAL: ESMFold2 env python not found: ${ESMFOLD2_ENV}/bin/python" >&2
            exit 2
        fi
        ESMFOLD2_SITE_PACKAGES="$("${ESMFOLD2_ENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    fi
    ESMFOLD2_FLAGS=(
        --esmfold2-site-packages "${ESMFOLD2_SITE_PACKAGES}"
        --esmfold2-model "${ESMFOLD2_MODEL}"
        --esmfold2-num-loops "${ESMFOLD2_NUM_LOOPS}"
        --esmfold2-num-sampling-steps "${ESMFOLD2_NUM_SAMPLING_STEPS}"
        --esmfold2-num-diffusion-samples "${ESMFOLD2_NUM_DIFFUSION_SAMPLES}"
        --esmfold2-seed "${ESMFOLD2_SEED}"
    )
fi

NMP_FLAGS=()
if [ "${TARGET_SOURCE}" = "nmp" ]; then
    NMP_FLAGS=(
        --netmhciipan-bin "${NETMHCIIPAN_BIN}"
        --nmp-workers "${NMP_WORKERS}"
        --nmp-batch-size "${NMP_BATCH_SIZE}"
        --nmp-max-lengths-per-call "${NMP_MAX_LENGTHS_PER_CALL}"
        --nmp-timeout "${NMP_TIMEOUT}"
    )
elif [ "${TARGET_SOURCE}" != "head" ]; then
    echo "FATAL: TARGET_SOURCE must be nmp or head" >&2
    exit 2
fi

srun python "${PROJECT_ROOT}/scripts/refine_rf_designs.py" \
    --mode "${MODE}" \
    --seed-table "${SEED_TABLE}" \
    --proteins "${PROTEINS}" \
    --allele "${ALLELE}" \
    --target-source "${TARGET_SOURCE}" \
    "${OFFICIAL_FLAG[@]}" \
    "${FINAL_CANDIDATES_FLAG[@]}" \
    "${CONSTRAINT_FLAG[@]}" \
    --test-set-parquet "${TEST_SET_PARQUET}" \
    --pdb-root "${PDB_ROOT}" \
    --head-checkpoint "${HEAD_CHECKPOINT}" \
    --head-config-dir "${HEAD_CONFIG_DIR}" \
    --head-variant-id "${HEAD_VARIANT_ID}" \
    --head-device cuda \
    --head-allele-idx "${HEAD_ALLELE_IDX}" \
    --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
    --head-residue-threshold "${HEAD_RESIDUE_THRESHOLD}" \
    --head-max-target-positions "${HEAD_MAX_TARGET_POSITIONS}" \
    --head-aa-per-position "${HEAD_AA_PER_POSITION}" \
    "${NMP_FLAGS[@]}" \
    --refold-cache-dir "${REFOLD_CACHE_DIR}" \
    --refold-model "${REFOLD_MODEL}" \
    "${ESMFOLD2_FLAGS[@]}" \
    --strong-rank "${STRONG_RANK}" \
    --margin-band "${MARGIN_BAND}" \
    "${STRUCTURE_GATE_FLAGS[@]}" \
    --refinement-structure-metrics "${REFINEMENT_STRUCTURE_METRICS}" \
    "${SIDECHAIN_GATE_FLAG[@]}" \
    "${TOPB_FLAG[@]}" \
    --beam-width "${BEAM_WIDTH}" \
    --max-rounds "${MAX_ROUNDS}" \
    --patience "${PATIENCE}" \
    --max-pairs "${MAX_PAIRS}" \
    --max-path-mutations "${MAX_PATH_MUTATIONS}" \
    --refold-cap "${REFOLD_CAP}" \
    --n-shards "${N_SHARDS}" \
    --shard-idx "${SHARD_IDX}" \
    --shard-by "${SHARD_BY}" \
    --shard-cost-col "${SHARD_COST_COL}" \
    "${EVAL_METRICS_FLAG[@]}" \
    --strong-binder-threshold "${STRONG_BINDER_THRESHOLD}" \
    --hotspot-threshold "${HOTSPOT_THRESHOLD}" \
    "${IMM_FULL_FLAG[@]}" \
    --out-dir "${OUT_DIR}" \
    --print-config

echo "[refine] done → ${OUT_DIR}"
