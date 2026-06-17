#!/bin/bash
#SBATCH --job-name=woven_bench_v4
#SBATCH --array=1-400
#SBATCH --partition=general
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/nbhtd/grama/logs/bench_v4_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/bench_v4_%A_%a.err

# WOVEN Benchmarking v4 — speed improvements over v3:
#   - Vectorised project_all (BLAS matmul, not per-subject R loop)
#   - Fold Laplacians precomputed once before CV grid (27->3 extractions)
#   - mc.cores=3 (one per CV fold, matches cpus-per-task=3)
#   - Vectorised BER Z_val NA fill

set -euo pipefail

DATA_DIR=/home/nbhtd/grama/data
OUT_DIR=/home/nbhtd/grama/results/benchmark_v4
WOVEN_SRC=/home/nbhtd/grama/R
LOG_DIR=/home/nbhtd/grama/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Conda R environment
export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

# Limit OpenBLAS threads to avoid oversubscription on shared nodes
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Python for MOFA2 via reticulate (mofapy2 installed in conda env)
export RETICULATE_PYTHON=/home/nbhtd/miniconda3/bin/python

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC /home/nbhtd/grama/scripts/benchmark_one_rep.R \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA_DIR" \
    "$OUT_DIR" \
    "$WOVEN_SRC"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
