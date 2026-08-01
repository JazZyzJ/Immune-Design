#!/bin/bash
# Phase 2 - fit the Potts model with plmc. One script serves both runs.
#
#   # baseline, on the existing nr90 alignment (N_eff/L = 3.6) - fast, early signal
#   sbatch coupling/scripts/13_couplings.slurm \
#          coupling/01_msa/nr90_baseline_focus.fasta coupling/02_couplings/nr90_baseline
#
#   # the real one, once 11_msa.slurm has finished and 12_prepare_alignment.py has run
#   sbatch coupling/scripts/13_couplings.slurm \
#          coupling/01_msa/deep_focus.fasta coupling/02_couplings/deep
#
#SBATCH --job-name=uric_plmc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 8 cores, not 32. The cpu partition is ~2500 jobs deep and only ONE node in it
# typically has >=32 idle cores, so a 32-core ask waits for a node to drain rather
# than backfilling: recent 32-core cpu jobs sat 7-11 DAYS (10939629_19, 11173484).
# 8 idle cores are common -- the 8-core MSA job backfilled in ~5 h. Trading ~4x
# runtime for days of queue is the right way round. Jobs 11475870/11475871 were
# cancelled at 32 cores for this reason.
#
# Override --time per run: baseline (N=3034) fits the 8 h default; pass
# --time=24:00:00 for the deep fit (N=12261), which is ~4x the work at 1/4 the cores.
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling/logs/%x_%j.err

set -euo pipefail

ALN=${1:?usage: sbatch 13_couplings.slurm <focus_alignment.fasta> <output_prefix>}
PREFIX=${2:?usage: sbatch 13_couplings.slurm <focus_alignment.fasta> <output_prefix>}

PROJ=/scratch/gpfs/KAIYIJIANG/kaiyi/Uricase

source /home/kj6118/anaconda3/etc/profile.d/conda.sh
conda activate "$PROJ/coupling/env/uricase-ec"

# plmc is OpenMP; keep it to the cores Slurm actually gave us
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "[plmc] host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK}"
echo "[plmc] alignment=$ALN"
echo "[plmc] prefix=$PREFIX"

srun python "$PROJ/coupling/scripts/13_run_plmc.py" \
    --alignment "$ALN" \
    --prefix "$PREFIX" \
    --cpu "${SLURM_CPUS_PER_TASK}"

echo "[plmc] COMPLETE"
