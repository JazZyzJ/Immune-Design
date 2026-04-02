#!/bin/bash
#SBATCH --job-name=merge-alleles
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/network/zc1519/logs/augmentation/merge_multi_allele_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/augmentation/merge_multi_allele_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
AUG_BASE="/scratch/network/zc1519/work/immune-design/augmentation"
LOG_DIR="/scratch/network/zc1519/logs/augmentation"

ALLELES=("drb0401" "drb1501")

mkdir -p "${LOG_DIR}"

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage J: Merge registry shards (multi-allele)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  Alleles      : ${ALLELES[*]}"
echo "  Aug base     : ${AUG_BASE}"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Merge each allele ─────────────────────────────────────────────────────
for ALLELE in "${ALLELES[@]}"; do
    echo ""
    echo "── Merging ${ALLELE} ──────────────────────────────────────"
    AUG_DIR="${AUG_BASE}/${ALLELE}"

    python3 -c "
import pandas as pd
from pathlib import Path

aug_dir = Path('${AUG_DIR}')
shards = sorted(aug_dir.glob('mutation_registry_strict_shard*.parquet'))
print(f'  Found {len(shards)} shard files in ${ALLELE}')

if not shards:
    raise RuntimeError('No shard files found for ${ALLELE}! Check if array jobs completed.')

dfs = [pd.read_parquet(s) for s in shards]
merged = pd.concat(dfs, ignore_index=True)
print(f'  Merged: {len(merged)} total rows')

# Safety dedup
before = len(merged)
merged = merged.drop_duplicates(subset=['source_protein_id', 'mut_pos_0b', 'mut_aa'])
after = len(merged)
if before != after:
    print(f'  WARNING: removed {before - after} duplicate rows')

out_path = aug_dir / 'mutation_registry_strict.parquet'
merged.to_parquet(out_path, index=False)
print(f'  Written: {out_path}')

for s in shards:
    s.unlink()
    print(f'    removed {s.name}')
"
    echo "  ${ALLELE} merge done."
done

echo ""
echo "============================================================"
echo "All allele registries merged."
echo "  DRB0401: ${AUG_BASE}/drb0401/mutation_registry_strict.parquet"
echo "  DRB1501: ${AUG_BASE}/drb1501/mutation_registry_strict.parquet"
echo "============================================================"
