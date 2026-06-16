#!/bin/bash
# setup_and_submit.sh — install packages then submit benchmark array
# Run on Hellbender: bash ~/grama/slurm/setup_and_submit.sh

set -euo pipefail
RSCRIPT=/home/nbhtd/miniconda3/bin/Rscript

echo "=== Step 1: Installing R packages ==="
$RSCRIPT -e '
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4L)

cran <- c("cluster", "RANN", "RSpectra", "missForest")
for (p in cran) {
  if (!requireNamespace(p, quietly=TRUE)) {
    cat("Installing", p, "...\n")
    install.packages(p)
  } else cat(p, "already installed\n")
}

if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
bio <- c("mixOmics", "MOFA2")
for (p in bio) {
  if (!requireNamespace(p, quietly=TRUE)) {
    cat("Installing", p, "...\n")
    BiocManager::install(p, ask=FALSE, update=FALSE)
  } else cat(p, "already installed\n")
}
cat("Packages OK\n")
'

echo ""
echo "=== Step 2: Verify packages ==="
$RSCRIPT -e '
pkgs <- c("cluster", "RANN", "RSpectra", "Matrix", "mixOmics", "MOFA2", "missForest")
missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]
if (length(missing) > 0) {
  cat("MISSING:", paste(missing, collapse=", "), "\n")
  quit(status=1)
} else {
  cat("All packages verified OK\n")
}
'

echo ""
echo "=== Step 3: Create output dirs ==="
mkdir -p /home/nbhtd/grama/results/benchmark
mkdir -p /home/nbhtd/grama/logs

echo ""
echo "=== Step 4: Check simulation data ==="
N_FILES=$(ls /home/nbhtd/grama/data/arm_A_rep_*.rds 2>/dev/null | grep -v mcar | grep -v mar | wc -l)
echo "ARM A complete reps found: $N_FILES"
if [ "$N_FILES" -eq 0 ]; then
  echo "ERROR: No simulation data found in /home/nbhtd/grama/data/"
  exit 1
fi

# Only submit tasks for arms/reps that have data
N_TASKS=$(ls /home/nbhtd/grama/data/arm_*_rep_*.rds 2>/dev/null | grep -v mcar | grep -v mar | wc -l)
echo "Total complete rep files: $N_TASKS"

echo ""
echo "=== Step 5: Submit benchmark array ==="
# Submit only ARM A (tasks 1-100) first as a pilot; expand after spot check
sbatch --array=1-100 /home/nbhtd/grama/slurm/benchmark.slurm
echo "Pilot array (ARM A) submitted. Check with: squeue -u nbhtd"
echo ""
echo "After pilot completes and looks good, run:"
echo "  sbatch --array=101-400 /home/nbhtd/grama/slurm/benchmark.slurm"
echo ""
echo "To aggregate results when done:"
echo "  Rscript ~/grama/scripts/aggregate_results.R ~/grama/results/benchmark ~/grama/results/summary.csv"
