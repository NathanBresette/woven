#!/usr/bin/env Rscript
# ARM D: Semi-synthetic simulation — microbiome (zero-inflated NB) + metabolomics (log-normal)
# Cross-modal correlation via NorTA (shared latent Gaussian copula)
# 4 groups × 75 subjects = 300 total, K=5 shared latent factors
# Spike-in cross-modal associations at 5% of feature pairs (known ground truth)
# Usage: Rscript simulate_arm_d.R <rep_id> <out_dir>

.libPaths(c("/home/nbhtd/grama/Rlib", .libPaths()))
suppressPackageStartupMessages({
  library(MASS)
  library(Matrix)
})

args    <- commandArgs(trailingOnly = TRUE)
rep_id  <- as.integer(args[1])
out_dir <- args[2]

stopifnot(!is.na(rep_id), dir.exists(out_dir))

set.seed(rep_id * 6271L)

n_per_group <- 75L
n           <- n_per_group * 4L
K           <- 5L
p_micro     <- 300L   # microbiome taxa
p_metab     <- 500L   # metabolite features
labels      <- rep(1L:4L, each = n_per_group)

# ── Shared latent factors (NorTA backbone) ────────────────────────────────────
# K shared factors drive cross-modal correlation and group structure
# Each group has distinct factor means
group_means <- matrix(c(
   2,  1,  0, -1, -2,   # group 1 factor means
   1,  2, -1,  0,  1,   # group 2
  -1, -1,  2,  1,  0,   # group 3
  -2,  0,  1,  2, -1    # group 4
), nrow = 4, ncol = K, byrow = TRUE)

Z_shared <- matrix(0, nrow = n, ncol = K)
for (g in 1L:4L) {
  idx <- which(labels == g)
  Z_shared[idx, ] <- mvrnorm(
    n    = length(idx),
    mu   = group_means[g, ],
    Sigma = diag(K) * 0.5 + 0.3   # within-group covariance
  )
}

# ── Microbiome simulation (zero-inflated log-normal → count → CLR) ───────────
# Realistic parameters: ~30% zero-inflation, overdispersed counts, ~200 depth
# Step 1: sparse loading matrix (taxa × K factors)
set.seed(rep_id * 6271L + 1L)
W_micro <- matrix(0, p_micro, K)
for (k in 1L:K) {
  active <- sample(p_micro, round(p_micro * 0.15))
  W_micro[active, k] <- rnorm(length(active), 0, 0.8)
}

# Step 2: latent log-abundances
log_abund <- Z_shared %*% t(W_micro) + matrix(rnorm(n * p_micro, 0, 0.5), n, p_micro)

# Step 3: zero-inflation (structural zeros, taxon-specific probability)
zero_prob <- rbeta(p_micro, 1, 4)   # most taxa ~20% structural zeros
zero_mask <- matrix(rbinom(n * p_micro, 1L, rep(zero_prob, each = n)), n, p_micro)

# Step 4: counts from NB (mean = exp(log_abund), dispersion = 0.5)
counts <- matrix(0L, n, p_micro)
for (j in 1L:p_micro) {
  mu_j <- exp(log_abund[, j] - mean(log_abund[, j]) + 3)   # mean count ~20
  counts[, j] <- ifelse(
    zero_mask[, j] == 1L, 0L,
    rnbinom(n, mu = mu_j, size = 0.5)
  )
}

# Step 5: CLR transform (compositional analysis standard)
counts_pos <- counts + 1L   # pseudocount
log_counts <- log(counts_pos)
clr_micro  <- log_counts - rowMeans(log_counts)

# Filter ultra-low prevalence taxa (< 10% prevalence)
prevalence <- colMeans(counts > 0)
keep_taxa  <- prevalence >= 0.10
clr_micro  <- clr_micro[, keep_taxa, drop = FALSE]
p_micro_final <- ncol(clr_micro)

# ── Metabolomics simulation (log-normal, MNAR zeros at LOD) ──────────────────
set.seed(rep_id * 6271L + 2L)
W_metab <- matrix(0, p_metab, K)
for (k in 1L:K) {
  active <- sample(p_metab, round(p_metab * 0.20))
  W_metab[active, k] <- rnorm(length(active), 0, 0.6)
}

log_intensity <- Z_shared %*% t(W_metab) + matrix(rnorm(n * p_metab, 0, 0.4), n, p_metab)

# Add realistic batch variation (2 batches, 150 samples each)
batch <- rep(c(0, 0.3), each = n / 2L)
log_intensity <- log_intensity + outer(batch, rnorm(p_metab, 0, 0.15))

# MNAR: metabolites below LOD appear missing (left-censoring at 10th percentile)
lod <- apply(log_intensity, 2, quantile, probs = 0.10)
metab_obs <- log_intensity
for (j in 1L:p_metab) {
  metab_obs[log_intensity[, j] < lod[j], j] <- NA_real_
}
# Impute MNAR with half-LOD (standard metabolomics practice before WOVEN analysis)
for (j in 1L:p_metab) metab_obs[is.na(metab_obs[, j]), j] <- lod[j] / 2

# ── Cross-modal spike-in associations (5% of feature pairs, known ground truth) ─
# Select 5% of micro × metab pairs as truly associated
n_spikein <- round(p_micro_final * p_metab * 0.05 / p_micro_final)
spike_micro <- sample(p_micro_final, n_spikein)
spike_metab <- sample(p_metab, n_spikein)
spike_effect <- rnorm(n_spikein, 0, 0.3)
for (i in seq_len(n_spikein)) {
  metab_obs[, spike_metab[i]] <- metab_obs[, spike_metab[i]] +
    spike_effect[i] * clr_micro[, spike_micro[i]]
}

# ── truth_Z: top K PCs of complete scaled concatenated data ──────────────────
X_concat <- cbind(
  scale(clr_micro,  center = TRUE, scale = TRUE),
  scale(metab_obs, center = TRUE, scale = TRUE)
)
X_concat[!is.finite(X_concat)] <- 0
pca     <- prcomp(X_concat, center = FALSE, scale. = FALSE, rank. = K)
truth_Z <- pca$x

# ── Assemble and save ─────────────────────────────────────────────────────────
out <- list(
  data        = list(microbiome = clr_micro, metabolomics = metab_obs),
  labels      = labels,
  truth_Z     = truth_Z,
  arm         = "D",
  rep         = rep_id,
  seed        = rep_id * 6271L,
  n_samples   = n,
  n_groups    = 4L,
  sim_params  = list(
    simulator       = "NorTA (custom)",
    micro_model     = "zero-inflated NB, CLR-transformed",
    metab_model     = "log-normal, MNAR LOD imputed",
    p_micro_final   = p_micro_final,
    p_metab         = p_metab,
    n_spikein_pairs = n_spikein,
    prevalence_filter = 0.10
  )
)

fname <- file.path(out_dir, "complete", sprintf("arm_D_rep_%03d.rds", rep_id))
dir.create(file.path(out_dir, "complete"), showWarnings = FALSE, recursive = TRUE)
saveRDS(out, fname)
cat(sprintf("[ARM D rep %03d] saved: microbiome %dx%d (CLR), metabolomics %dx%d\n",
            rep_id, n, p_micro_final, n, p_metab))
