#!/bin/bash
#SBATCH --job-name=grama_split
#SBATCH --array=1-64
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/nbhtd/grama/logs/split_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/split_%A_%a.err

# Split benchmark: one condition per task
# Tasks 1-16:  complete  for ARM B reps 76-91 (task_ids 176-191)
# Tasks 17-32: mcar30    for ARM B reps 76-91
# Tasks 33-48: mcar50    for ARM B reps 76-91
# Tasks 49-64: mar       for ARM B reps 76-91

set -euo pipefail

export PATH=/home/nbhtd/miniconda3/bin:$PATH
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export R_LIBS_USER=/home/nbhtd/grama/Rlib
export RETICULATE_PYTHON=/home/nbhtd/miniconda3/bin/python

TASK=${SLURM_ARRAY_TASK_ID}

COND_IDX=$(( (TASK - 1) / 16 ))
REP_IDX=$(( (TASK - 1) % 16 ))

CONDS=("complete" "mcar30" "mcar50" "mar")
WOVEN_COND="${CONDS[$COND_IDX]}"
TASK_ID=$(( 176 + REP_IDX ))   # maps to benchmark_one_rep.R task_ids 176-191

export WOVEN_COND

DATA_DIR=/home/nbhtd/grama/data
OUT_DIR=/home/nbhtd/grama/results/benchmark
WOVEN_SRC=/home/nbhtd/grama/R

echo "Split task ${TASK}: ARM B task_id=${TASK_ID} condition=${WOVEN_COND} on $(hostname) at $(date)"

/home/nbhtd/miniconda3/bin/Rscript /home/nbhtd/grama/scripts/benchmark_one_rep.R \
    "${TASK_ID}" "$DATA_DIR" "$OUT_DIR" "$WOVEN_SRC"

echo "Split task ${TASK} done at $(date)"
