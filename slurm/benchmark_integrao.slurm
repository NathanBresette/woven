#!/bin/bash
#SBATCH --job-name=grama_integrao
#SBATCH --array=1-400
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=requeue
#SBATCH --output=/home/nbhtd/grama/logs/integrao_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/integrao_%A_%a.err

# IntegrAO benchmark: 400 reps x 4 arms x 4 conditions
# Uses integrAO conda env for Python/PyTorch dependencies
# Note: IntegrAO trains on complete-case anchors only (same ESS as DIABLO)
# 3h wall time because GNN training on CPU is slower than WOVEN eigendecomp

set -euo pipefail

DATA=/home/nbhtd/grama/data
OUT=/home/nbhtd/grama/results/benchmark_integrao
SRC=/home/nbhtd/grama/R
SCRIPT=/home/nbhtd/grama/scripts/benchmark_integrao_rep.R

mkdir -p "$OUT" /home/nbhtd/grama/logs

# Use base conda Rscript (has reticulate), not integrAO env
export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# reticulate will switch to integrAO Python inside the R script
export RETICULATE_PYTHON=/home/nbhtd/miniconda3/envs/integrAO/bin/python

echo "Task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"

$R_EXEC "$SCRIPT" \
    "${SLURM_ARRAY_TASK_ID}" \
    "$DATA" \
    "$OUT" \
    "$SRC"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
