#!/bin/bash
#SBATCH --job-name=grama_sim
#SBATCH --array=1-200
#SBATCH --partition=requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nbhtd/grama/logs/sim_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/sim_%A_%a.err

# WOVEN Semi-Synthetic Simulation
# Tasks 1-100:   ARM C (InterSIM: RNA-seq + methylation + protein)
# Tasks 101-200: ARM D (NorTA: microbiome + metabolomics)
# Each task: simulate one rep + apply all three missingness conditions

set -euo pipefail

DATA_DIR=/home/nbhtd/grama/data
SCRIPTS=/home/nbhtd/grama/scripts
LOG_DIR=/home/nbhtd/grama/logs

mkdir -p "$DATA_DIR/complete" "$DATA_DIR/missing" "$LOG_DIR"

export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export R_LIBS_USER=/home/nbhtd/grama/Rlib

TASK=${SLURM_ARRAY_TASK_ID}

if [ "$TASK" -le 100 ]; then
  ARM="C"
  REP=$TASK
  SCRIPT="$SCRIPTS/simulate_arm_c.R"
else
  ARM="D"
  REP=$(( TASK - 100 ))
  SCRIPT="$SCRIPTS/simulate_arm_d.R"
fi

echo "Task ${TASK}: ARM ${ARM} rep ${REP} starting on $(hostname) at $(date)"

# Step 1: simulate complete data
$R_EXEC "$SCRIPT" "$REP" "$DATA_DIR"

# Step 2: apply missingness
COMPLETE_FILE="$DATA_DIR/complete/arm_${ARM}_rep_$(printf '%03d' $REP).rds"
$R_EXEC "$SCRIPTS/apply_missingness.R" "$COMPLETE_FILE" "$DATA_DIR"

echo "Task ${TASK}: ARM ${ARM} rep ${REP} done at $(date)"
