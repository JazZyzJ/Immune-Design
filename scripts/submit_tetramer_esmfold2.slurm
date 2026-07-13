#!/bin/bash
# ESMFold2 tetramer + ligand + MSA prediction on ailab H200 (PLAN_TETRAMER_GATE.md).
# Runs the standalone predictor in the `esmfold2` conda env (Biohub esm + transformers 4.57);
# NOT immune-design. All args after the script path pass through to predict_tetramer_esmfold2.py.
#
# Usage:
#   sbatch --output=<logs>/esmf2_%j.out --error=<logs>/esmf2_%j.err \
#     scripts/submit_tetramer_esmfold2.slurm --sequence <SEQ> --name <NAME> --out-dir <DIR> \
#       --msa-a3m <a3m> --ligand-smiles 'O=C1NC(=O)C2=C(N1)NC(=O)N2' --n-copies 4
#
#SBATCH --job-name=esmf2_tetra
#SBATCH --account=kaiyijiang
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/gpfs/KAIYIJIANG/zijie/model_cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # compute nodes have no internet; weights are cached
export MKL_THREADING_LAYER=GNU
ESMF2_PY="${ESMF2_PY:-/home/zc1519/.conda/envs/esmfold2/bin/python}"

exec "$ESMF2_PY" /home/zc1519/src/Immune-Design/scripts/predict_tetramer_esmfold2.py "$@"
