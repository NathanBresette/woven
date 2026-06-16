#!/bin/bash
#SBATCH --job-name=grama_ber_v3
#SBATCH --array=1-400
#SBATCH --partition=requeue
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/nbhtd/grama/logs/patch_ber_v3_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/patch_ber_v3_%A_%a.err

set -euo pipefail

CONDA_BASE="/home/nbhtd/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate grama_env 2>/dev/null || true

export RETICULATE_PYTHON="${CONDA_BASE}/bin/python"
export R_LIBS_USER="/home/nbhtd/grama/Rlib"

DATA_DIR="/home/nbhtd/grama/data"
BENCH_DIR="/home/nbhtd/grama/results/benchmark_v3"
WOVEN_SRC="/home/nbhtd/grama/R"
SCRIPTS="/home/nbhtd/grama/scripts"

mkdir -p /home/nbhtd/grama/logs

Rscript "${SCRIPTS}/patch_ber_v3.R" \
  "${SLURM_ARRAY_TASK_ID}" \
  "${DATA_DIR}" \
  "${BENCH_DIR}" \
  "${WOVEN_SRC}"
