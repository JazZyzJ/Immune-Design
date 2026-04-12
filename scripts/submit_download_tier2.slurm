#!/bin/bash
#SBATCH --job-name=L3-download-tier2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/download_tier2_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval/download_tier2_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/if_test_set"
LOG_DIR="/scratch/gpfs/KAIYIJIANG/zijie/logs/if_eval"

ENTITY_IDS="${PROJECT_ROOT}/outputs/if/test_set/tier2_all_entity_ids.txt"
OUTPUT_DIR="${DATA_DIR}/tier2_candidates"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Tier 2 Candidate Download & Filter"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  Entity IDs   : ${ENTITY_IDS}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Python       : $(python --version 2>&1)"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Launch ────────────────────────────────────────────────────────────────
python "${PROJECT_ROOT}/scripts/download_tier2_candidates.py" \
    --entity-ids "${ENTITY_IDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --merged-fasta "${DATA_DIR}/tier2_candidates_merged.fasta"

echo "Tier 2 download & filter complete."
echo "Next: sbatch scripts/submit_prescreen_tier2.slurm"
