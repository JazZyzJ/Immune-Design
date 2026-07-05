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
#   immune head (GPU) + NetMHCIIpan (CPU, --nmp-workers==cpus-per-task) + ESMFold refold (GPU).
# Targets ailab H200 (8 cores/GPU); NMP_WORKERS defaults to 8 to match --cpus-per-task=8.
# PLI H100 fallback (busy): resubmit with
#   sbatch --account=pli_x --qos=pli-low --partition=pli --constraint= scripts/submit_refine.slurm
#
# Input is a self-contained --seed-table (protein_id/design_id/sequence parquet); the
# whole listed set is refined (PROTEINS filters it). Override any UPPER_CASE var at submit.
#   sbatch scripts/submit_refine.slurm                          # full refine_seeds_0701 table
#   SEED_TABLE=<1-row.parquet> PROTEINS=Q0CXR4 OUT_DIR=<dir> sbatch scripts/submit_refine.slurm  # smoke

set -euo pipefail
: "${PS1:=}"
export PS1

# ── Paths / model artifacts (override via env) ───────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
BASE="${BASE:-/scratch/gpfs/KAIYIJIANG/zijie}"
DPLM_ROOT="${DPLM_ROOT:-${PROJECT_ROOT}/inverse_folding/dplm}"
C24_INPUTS="${C24_INPUTS:-${BASE}/work/RF/Uricases_RF/characterized24_inputs/20260703T201539Z}"
LOG_DIR="${LOG_DIR:-${BASE}/logs/refine}"

MODE="${MODE:-refine}"                       # refine | ceiling
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
PROTEINS="${PROTEINS:-all}"
TARGET_SOURCE="${TARGET_SOURCE:-nmp}"

# Seeds: a --seed-table parquet (protein_id/design_id/sequence). Default = the full
# difficulty-stratified 0701 validation table; override SEED_TABLE for the smoke.
SEED_TABLE="${SEED_TABLE:-${BASE}/work/RF/Uricases_RF/refine_seed_selection_0701/refine_seeds_0701.parquet}"

# Immune head (a1res03 cv5 fold0 — the head the char24 generation used).
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-${BASE}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt}"
HEAD_CONFIG_DIR="${HEAD_CONFIG_DIR:-${PROJECT_ROOT}/epitope_head/configs}"
HEAD_VARIANT_ID="${HEAD_VARIANT_ID:-LC1}"
HEAD_ALLELE_IDX="${HEAD_ALLELE_IDX:-0}"
HEAD_WINDOW_BATCH_SIZE="${HEAD_WINDOW_BATCH_SIZE:-4096}"

# NetMHCIIpan (CPU). NMP_WORKERS matches --cpus-per-task so the 8 ailab cores are saturated.
NETMHCIIPAN_BIN="${NETMHCIIPAN_BIN:-${PROJECT_ROOT}/netMHCIIpan-4.3/netMHCIIpan}"
NMP_WORKERS="${NMP_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
NMP_BATCH_SIZE="${NMP_BATCH_SIZE:-8}"
NMP_MAX_LENGTHS_PER_CALL="${NMP_MAX_LENGTHS_PER_CALL:-4}"
NMP_TIMEOUT="${NMP_TIMEOUT:-1800}"

# Structure: WT/target backbones + the test-set rows resolve_structure_path needs.
TEST_SET_PARQUET="${TEST_SET_PARQUET:-${C24_INPUTS}/uricase_characterized24_if_ready.parquet}"
PDB_ROOT="${PDB_ROOT:-${BASE}/work/immune-design/if_test_set/uricases/pdbs_if_ready}"
CONSTRAINT_MANIFEST="${CONSTRAINT_MANIFEST:-${C24_INPUTS}/uricase_characterized24_active_site_manifest.yaml}"

# Search knobs (PLAN §7).
STRONG_RANK="${STRONG_RANK:-0.02}"
MARGIN_BAND="${MARGIN_BAND:-0.10}"
SCTM_EPS="${SCTM_EPS:-0.05}"
TOPB="${TOPB:-}"                             # empty → NMP-all (ceiling); set an int for cost mode
BEAM_WIDTH="${BEAM_WIDTH:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
PATIENCE="${PATIENCE:-3}"
MAX_PAIRS="${MAX_PAIRS:-200}"
MAX_PATH_MUTATIONS="${MAX_PATH_MUTATIONS:-8}"

# Output run dir + per-run ESMFold refold cache (writes <key>.pdb/.plddt → gives pdb_path).
TS="$(date +%Y%m%dT%H%M%SZ)"
JOB_TAG="${JOB_TAG:-refine_${MODE}_0701}"
OUT_DIR="${OUT_DIR:-${BASE}/run/inverse_folding/refine/${JOB_TAG}_${TS}}"
ESMFOLD_CACHE_DIR="${ESMFOLD_CACHE_DIR:-${OUT_DIR}/esmfold_cache}"

mkdir -p "${OUT_DIR}" "${ESMFOLD_CACHE_DIR}" "${LOG_DIR}"

# ── Environment (mirrors submit_if_phase_c.slurm) ────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${DPLM_ROOT}/vendor/openfold:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="${BASE}/model_cache/torch"           # ESMFold weights (esmfold_v1)
export HF_HOME="${BASE}/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# TOPB flag is only emitted when non-empty (empty = NMP-all ceiling behavior).
TOPB_FLAG=()
[ -n "${TOPB}" ] && TOPB_FLAG=(--topB "${TOPB}")

echo "============================================================"
echo "RF Refinement"
echo "  Job ID        : ${SLURM_JOB_ID:-<none>}"
echo "  Node          : ${SLURM_NODELIST:-<none>}"
echo "  GPU name      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo N/A)"
echo "  CPUs          : ${SLURM_CPUS_PER_TASK:-8}  (NMP workers: ${NMP_WORKERS})"
echo "  Mode          : ${MODE}   target-source: ${TARGET_SOURCE}"
echo "  Allele        : ${ALLELE}"
echo "  Seed table    : ${SEED_TABLE}"
echo "  Proteins      : ${PROTEINS}"
echo "  Head ckpt     : ${HEAD_CHECKPOINT}"
echo "  NMP binary    : ${NETMHCIIPAN_BIN}"
echo "  Test set      : ${TEST_SET_PARQUET}"
echo "  PDB root      : ${PDB_ROOT}"
echo "  Constraints   : ${CONSTRAINT_MANIFEST}"
echo "  Out dir       : ${OUT_DIR}"
echo "  ESMFold cache : ${ESMFOLD_CACHE_DIR}"
echo "============================================================"

srun python "${PROJECT_ROOT}/scripts/refine_rf_designs.py" \
    --mode "${MODE}" \
    --seed-table "${SEED_TABLE}" \
    --proteins "${PROTEINS}" \
    --allele "${ALLELE}" \
    --target-source "${TARGET_SOURCE}" \
    --constraint-manifest "${CONSTRAINT_MANIFEST}" \
    --test-set-parquet "${TEST_SET_PARQUET}" \
    --pdb-root "${PDB_ROOT}" \
    --head-checkpoint "${HEAD_CHECKPOINT}" \
    --head-config-dir "${HEAD_CONFIG_DIR}" \
    --head-variant-id "${HEAD_VARIANT_ID}" \
    --head-device cuda \
    --head-allele-idx "${HEAD_ALLELE_IDX}" \
    --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
    --netmhciipan-bin "${NETMHCIIPAN_BIN}" \
    --nmp-workers "${NMP_WORKERS}" \
    --nmp-batch-size "${NMP_BATCH_SIZE}" \
    --nmp-max-lengths-per-call "${NMP_MAX_LENGTHS_PER_CALL}" \
    --nmp-timeout "${NMP_TIMEOUT}" \
    --esmfold-cache-dir "${ESMFOLD_CACHE_DIR}" \
    --strong-rank "${STRONG_RANK}" \
    --margin-band "${MARGIN_BAND}" \
    --scTM-eps "${SCTM_EPS}" \
    "${TOPB_FLAG[@]}" \
    --beam-width "${BEAM_WIDTH}" \
    --max-rounds "${MAX_ROUNDS}" \
    --patience "${PATIENCE}" \
    --max-pairs "${MAX_PAIRS}" \
    --max-path-mutations "${MAX_PATH_MUTATIONS}" \
    --out-dir "${OUT_DIR}" \
    --print-config

echo "[refine] done → ${OUT_DIR}"
