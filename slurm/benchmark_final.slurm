#!/bin/bash
#SBATCH --job-name=grama_final
#SBATCH --array=1-400
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nbhtd/grama/logs/final_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/final_%A_%a.err

# WOVEN Final Benchmark — all 4 arms, 100 reps each
# Optimized code: Omega/L/XtX caching, Cholesky precompute, vectorized Nystrom,
# B_chol/IKY_Xa outside als_run(), equivalent CV for all tunable methods.
# Timing recorded per method for paper comparisons.

set -euo pipefail

DATA_DIR=/home/nbhtd/grama/data
OUT_DIR=/home/nbhtd/grama/results/benchmark_final
WOVEN_SRC=/home/nbhtd/grama/R

mkdir -p "$OUT_DIR" /home/nbhtd/grama/logs

export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export RETICULATE_PYTHON=/home/nbhtd/miniconda3/bin/python
export WOVEN_MAX_ITER=200
export WOVEN_N_RESTARTS=3

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC /home/nbhtd/grama/scripts/benchmark_one_rep.R \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA_DIR" \
    "$OUT_DIR" \
    "$WOVEN_SRC"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
