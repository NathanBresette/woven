#!/bin/bash
#SBATCH --job-name=grama_dense_d
#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=requeue
#SBATCH --output=/home/nbhtd/grama/logs/dense_d_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/dense_d_%A_%a.err

# Dense DIABLO (block.plsda, no sparsity) on ARM D
# Purpose: disentangle dense vs sparse W from missing-data handling
# 100 reps x 2 conditions (complete, mcar50) per job

set -euo pipefail

DATA=/home/nbhtd/grama/data
OUT=/home/nbhtd/grama/results/dense_diablo_armd
SRC=/home/nbhtd/grama/R
SCRIPT=/home/nbhtd/grama/scripts/dense_diablo_armd.R

mkdir -p "$OUT" /home/nbhtd/grama/logs

export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC "$SCRIPT" \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA" \
    "$OUT" \
    "$SRC"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
