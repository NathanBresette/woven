#!/bin/bash
#SBATCH --job-name=grama_reinduce
#SBATCH --array=1-100
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=requeue
#SBATCH --output=/home/nbhtd/grama/logs/reinduce_%A_%a.out
#SBATCH --error=/home/nbhtd/grama/logs/reinduce_%A_%a.err

# Re-induce ARM C and D missingness with >=1 view per subject constraint.
# Fast: pure R, no model fitting, ~seconds per rep.

set -euo pipefail

DATA=/home/nbhtd/grama/data
SCRIPT=/home/nbhtd/grama/scripts/reinduce_missingness_cd.R

export PATH=/home/nbhtd/miniconda3/bin:$PATH
R_EXEC=/home/nbhtd/miniconda3/bin/Rscript

echo "Rep ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
$R_EXEC "$SCRIPT" "$DATA" "${SLURM_ARRAY_TASK_ID}"
echo "Rep ${SLURM_ARRAY_TASK_ID} done at $(date)"
