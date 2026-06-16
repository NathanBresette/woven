#!/bin/bash
#SBATCH --job-name=grama_anchor
#SBATCH --array=1-50
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=requeue
#SBATCH --output=/home/nbhtd/grama/logs/anchor_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/anchor_%A_%a.err

# Anchor fraction sensitivity sweep
# Each job: one rep, all anchor_fracs (0.05-0.75) x 2 arms (A and C)
# 50 reps x 9 fracs x 2 arms = 900 conditions total

set -euo pipefail

DATA=/home/nbhtd/grama/data
OUT=/home/nbhtd/grama/results/sensitivity_anchor
SRC=/home/nbhtd/grama/R
SCRIPT=/home/nbhtd/grama/scripts/sensitivity_anchor_rep.R

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
