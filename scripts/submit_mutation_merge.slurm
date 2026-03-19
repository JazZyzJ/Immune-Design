#!/bin/bash
#SBATCH --job-name=mut-merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/network/zc1519/logs/augmentation/merge_%j.out
#SBATCH --error=/scratch/network/zc1519/logs/augmentation/merge_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/home/zc1519/src/Immune-Design"
DATA_DIR="/scratch/network/zc1519/work/immune-design/manifests"
AUG_OUTPUT="/scratch/network/zc1519/work/immune-design/augmentation"

mkdir -p /scratch/network/zc1519/logs/augmentation

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Diagnostics ───────────────────────────────────────────────────────────
echo "============================================================"
echo "Stage J: Merge shards + Materialize"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_NODELIST}"
echo "  Timestamp    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── Step 1: Merge shard parquets ─────────────────────────────────────────
python3 -c "
import pandas as pd
from pathlib import Path

aug_dir = Path('${AUG_OUTPUT}')
shards = sorted(aug_dir.glob('mutation_registry_strict_shard*.parquet'))
print(f'Found {len(shards)} shard files')

if not shards:
    raise RuntimeError('No shard files found! Check if array jobs completed.')

dfs = [pd.read_parquet(s) for s in shards]
merged = pd.concat(dfs, ignore_index=True)
print(f'Merged: {len(merged)} total rows')

# Safety dedup
before = len(merged)
merged = merged.drop_duplicates(subset=['source_protein_id', 'mut_pos_0b', 'mut_aa'])
after = len(merged)
if before != after:
    print(f'WARNING: removed {before - after} duplicate rows')

out_path = aug_dir / 'mutation_registry_strict.parquet'
merged.to_parquet(out_path, index=False)
print(f'Written: {out_path}')

for s in shards:
    s.unlink()
    print(f'  removed {s.name}')
"

# ── Step 2: Materialize augmented train samples ──────────────────────────
python -m scripts.materialize_augmented_train_set \
    --data-root "${DATA_DIR}" \
    --output-dir "${AUG_OUTPUT}"

echo "============================================================"
echo "Done. Augmented train artifact:"
echo "  ${AUG_OUTPUT}/protein_samples_strict_aug_train.parquet"
echo "============================================================"
