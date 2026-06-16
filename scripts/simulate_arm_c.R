#!/usr/bin/env Rscript
# ARM C: Semi-synthetic simulation using InterSIM (TCGA-OV reference)
# Modalities: RNA-seq (log2-CPM) + methylation (M-values) + protein (log2)
# 4 groups × 75 subjects = 300 total
# truth_Z: top K PCs of complete concatenated scaled data
# Usage: Rscript simulate_arm_c.R <rep_id> <out_dir>

.libPaths(c("/home/nbhtd/grama/Rlib", .libPaths()))
suppressPackageStartupMessages({
  library(InterSIM)
  library(Matrix)
})

args    <- commandArgs(trailingOnly = TRUE)
rep_id  <- as.integer(args[1])
out_dir <- args[2]

stopifnot(!is.na(rep_id), dir.exists(out_dir))

set.seed(rep_id * 7919L)

n_per_group <- 75L
K           <- 5L

# ── Generate 4 groups in one InterSIM call ────────────────────────────────────
# cluster.sample.prop of length 4 → 4 equal groups of 75 each
sim <- InterSIM(
  n.sample             = n_per_group * 4L,
  cluster.sample.prop  = c(0.25, 0.25, 0.25, 0.25),
  delta.methyl         = 2.0,
  delta.expr           = 2.0,
  delta.protein        = 2.0,
  p.DMP                = 0.20,
  p.DEG                = 0.20,
  p.DEP                = 0.20,
  do.plot              = FALSE,
  sample.cluster       = TRUE,
  feature.cluster      = TRUE
)

rnaseq      <- sim$dat.expr
methylation <- sim$dat.methyl
protein     <- sim$dat.protein
labels      <- as.integer(sim$clustering.assignment$cluster.id)
n           <- nrow(rnaseq)

# ── Align feature dimensions across reps ──────────────────────────────────────
# InterSIM returns all reference features; subsample to fixed dimensions
set.seed(rep_id * 3571L + 1L)
p_rna  <- ncol(rnaseq)    # InterSIM returns fixed dims from TCGA-OV reference
p_meth <- ncol(methylation)
p_prot <- ncol(protein)

# ── truth_Z: top K PCs of complete concatenated scaled data ───────────────────
X_concat <- cbind(
  scale(rnaseq, center = TRUE, scale = TRUE),
  scale(methylation, center = TRUE, scale = TRUE),
  scale(protein, center = TRUE, scale = TRUE)
)
# Replace any NA/Inf from zero-variance cols
X_concat[!is.finite(X_concat)] <- 0
pca     <- prcomp(X_concat, center = FALSE, scale. = FALSE, rank. = K)
truth_Z <- pca$x   # n × K

# ── Assemble and save ─────────────────────────────────────────────────────────
out <- list(
  data        = list(rnaseq = rnaseq, methylation = methylation, protein = protein),
  labels      = labels,
  truth_Z     = truth_Z,
  arm         = "C",
  rep         = rep_id,
  seed        = rep_id * 7919L,
  n_samples   = n,
  n_groups    = 4L,
  sim_params  = list(
    simulator   = "InterSIM",
    reference   = "TCGA-OV (built-in)",
    delta.methyl = 2.0, delta.expr = 2.0, delta.protein = 2.0,
    p_rna = p_rna, p_meth = p_meth, p_prot = p_prot
  )
)

fname <- file.path(out_dir, "complete", sprintf("arm_C_rep_%03d.rds", rep_id))
dir.create(file.path(out_dir, "complete"), showWarnings = FALSE, recursive = TRUE)
saveRDS(out, fname)
cat(sprintf("[ARM C rep %03d] saved: rnaseq %dx%d, meth %dx%d, prot %dx%d\n",
            rep_id, n, p_rna, n, p_meth, n, p_prot))
