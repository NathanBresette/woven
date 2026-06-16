#!/bin/bash
#SBATCH --job-name=grama_bench_v3
#SBATCH --array=1-400
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/nbhtd/grama/logs/bench_v3_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/bench_v3_%A_%a.err

# WOVEN Benchmarking v3 — mcca_dual unified solver (no iterations, global optimum)
# Each task: one rep × all conditions (complete, mcar30, mcar50, mar) × 4 methods
# Wall time reduced from 4h to 1.5h: mcca_dual is ~50x faster than primal

set -euo pipefail

DATA_DIR=/home/nbhtd/grama/data
OUT_DIR=/home/nbhtd/grama/results/benchmark_v3
WOVEN_SRC=/home/nbhtd/grama/R
LOG_DIR=/home/nbhtd/grama/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Conda R environment
export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

# Limit OpenBLAS threads to avoid oversubscription on shared nodes
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Python for MOFA2 via reticulate (mofapy2 installed in conda env)
export RETICULATE_PYTHON=/home/nbhtd/miniconda3/bin/python

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC /home/nbhtd/grama/scripts/benchmark_one_rep.R \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA_DIR" \
    "$OUT_DIR" \
    "$WOVEN_SRC"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
