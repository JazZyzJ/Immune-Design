#!/bin/bash
# Phase 1 — deep MSA for Q00511 via MMseqs2 against the group's local ColabFold DBs.
#
# Why not the existing 03_alignment MSA: it gives N_eff/L = 3.59 at Q00511's columns,
# which is marginal for Potts inference (want >= 5, ideally >= 10). The metagenomic
# envdb is where the extra effective depth for this family should come from.
#
# Submit with:  sbatch coupling/scripts/11_msa.slurm
#
# Must run on GPU: the local uniref30/envdb .idx files were built with
# `mmseqs createindex --gpu 1` (see $DB/build_index.slurm line 30, GPU=1), and a
# GPU-format index omits the k-mer tables the CPU prefilter needs. Job 11464004
# tried --gpu 0 and died in 19 s with "getData: local id (4294967295) >= db size
# (19)" -> "Prefilter died". That is an index-format mismatch, not a memory or
# threading problem; it fails the same way at any --mem.
#
# 8 cores / 180G is host-side headroom for a single-query search; the GPU does the
# prefilter. 180G also stays under the 190000 MB commodity-node ceiling, which 192G
# crossed.
#SBATCH --job-name=uric_msa
# No --partition: DELLA's submit filter rejects an explicit "gpu" partition and
# routes the job by --gres instead.
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling/logs/%x_%j.err

set -euo pipefail

PROJ=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling
DB=/scratch/gpfs/KAIYIJIANG/databases/colabfold
OUT=$PROJ/01_msa/colabfold
QUERY=$PROJ/00_target/Q00511.fasta

# Environment Modules' init script dereferences PS1, which is unset in a batch
# shell — under `set -u` that aborts the job before the first echo. Job 11439346
# died this way in 1 s. Relax -u across the load only.
set +u
module load colabfold/1.5.5
set -u

# Keep mmseqs scratch off /tmp (small, node-local) and off /home (39/50 GB used).
export TMPDIR=$PROJ/01_msa/tmp
mkdir -p "$TMPDIR" "$OUT"

echo "[msa] host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "[msa] mmseqs=$(command -v mmseqs) ($(mmseqs version))"
[ -f "$DB/UNIREF30_READY" ] && [ -f "$DB/COLABDB_READY" ] || { echo "FATAL: DB ready markers missing" >&2; exit 2; }

# --filter 0 is the important one. ColabFold's default filter (--diff 3000, --qsc -20)
# is tuned to hand a structure predictor a diverse-but-capped MSA; for coupling
# inference it discards exactly the redundant-looking depth that N_eff is built from.
# We take everything here and do our own coverage/identity filtering in 12_filter_msa.py,
# where we can see what each threshold costs.
srun colabfold_search \
    --db1 uniref30_2302_db \
    --db3 colabfold_envdb_202108_db \
    --use-env 1 \
    --use-templates 0 \
    --filter 0 \
    --db-load-mode 2 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --gpu 1 \
    --unpack 1 \
    "$QUERY" "$DB" "$OUT"

echo "[msa] done. outputs:"
ls -la "$OUT"
for f in "$OUT"/*.a3m; do
    echo "[msa] $f : $(grep -c '^>' "$f") sequences"
done

# tmp can be tens of GB; drop it once the a3m is written
rm -rf "$TMPDIR"
echo "[msa] COMPLETE"
