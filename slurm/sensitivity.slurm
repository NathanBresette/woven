#!/bin/bash
#SBATCH --job-name=grama_sens
#SBATCH --array=1-30
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nbhtd/grama/logs/sens_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/sens_%A_%a.err

# WOVEN hyperparameter sensitivity analysis
# ARM A only (V=2, fast): sweeps lambda, gamma_y, k_nn one at a time.
# 30 reps × ~19 fits each = 570 total grama_v2 calls (~15 min total on requeue).
#
# After completion:
#   Rscript aggregate_sensitivity.R ~/grama/results/sensitivity ~/grama/results/sensitivity

set -euo pipefail

DATA_DIR=/home/nbhtd/grama/data
OUT_DIR=/home/nbhtd/grama/results/sensitivity
WOVEN_SRC=/home/nbhtd/grama/R

mkdir -p "$OUT_DIR" /home/nbhtd/grama/logs

export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Sensitivity rep ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC /home/nbhtd/grama/scripts/sensitivity_one_rep.R \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA_DIR" \
    "$OUT_DIR" \
    "$WOVEN_SRC"

echo "Sensitivity rep ${SLURM_ARRAY_TASK_ID} done at $(date)"
