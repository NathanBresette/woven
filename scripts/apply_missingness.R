#!/usr/bin/env Rscript
# Apply block-level missingness to a complete simulation rep
# Produces mcar30, mcar50, and mar versions alongside the complete file
# Usage: Rscript apply_missingness.R <complete_rds_path> <out_dir>
# Output: out_dir/missing/arm_X_rep_YYY_{mcar30,mcar50,mar}.rds

args         <- commandArgs(trailingOnly = TRUE)
complete_path <- args[1]
out_dir       <- args[2]

stopifnot(file.exists(complete_path), dir.exists(out_dir))

rep_dat <- readRDS(complete_path)
V       <- length(rep_dat$data)
n       <- nrow(rep_dat$data[[1]])
seed    <- rep_dat$seed %||% 42L
`%||%`  <- function(a, b) if (!is.null(a)) a else b

miss_dir <- file.path(out_dir, "missing")
dir.create(miss_dir, showWarnings = FALSE, recursive = TRUE)

# Extract basename for output naming
base <- tools::file_path_sans_ext(basename(complete_path))

apply_block_mask <- function(dat, mask_matrix) {
  # mask_matrix: n × V logical (TRUE = missing entire modality for that subject)
  lapply(seq_len(V), function(v) {
    X <- dat[[v]]
    X[mask_matrix[, v], ] <- NA_real_
    X
  })
}

# ── MCAR (each subject independently missing each modality) ──────────────────
for (rate in c(0.30, 0.50)) {
  set.seed(seed + round(rate * 1000))
  mask <- matrix(runif(n * V) < rate, nrow = n, ncol = V)
  # Ensure at least one complete subject per group (needed for anchor set)
  for (g in unique(rep_dat$labels)) {
    idx <- which(rep_dat$labels == g)
    mask[idx[1], ] <- FALSE   # force first subject in each group to be anchor
  }
  masked_data <- apply_block_mask(rep_dat$data, mask)
  names(masked_data) <- names(rep_dat$data)
  out <- rep_dat
  out$data <- masked_data
  out$missingness_mask <- mask
  out$missingness_type <- paste0("mcar", round(rate * 100))
  suffix <- paste0("_mcar", round(rate * 100))
  saveRDS(out, file.path(miss_dir, paste0(base, suffix, ".rds")))
}

# ── MAR (missingness correlated with group label — minority groups more missing) ──
set.seed(seed + 999L)
# Groups 3 and 4 have 2× the missingness rate of groups 1 and 2
base_rate <- 0.20
high_rate <- 0.45
miss_prob <- ifelse(rep_dat$labels %in% c(3L, 4L), high_rate, base_rate)
mask_mar  <- matrix(FALSE, nrow = n, ncol = V)
for (v in seq_len(V)) {
  mask_mar[, v] <- runif(n) < miss_prob
}
# Ensure at least one anchor per group
for (g in unique(rep_dat$labels)) {
  idx <- which(rep_dat$labels == g)
  mask_mar[idx[1], ] <- FALSE
}
masked_data <- apply_block_mask(rep_dat$data, mask_mar)
names(masked_data) <- names(rep_dat$data)
out_mar <- rep_dat
out_mar$data <- masked_data
out_mar$missingness_mask <- mask_mar
out_mar$missingness_type <- "mar"
saveRDS(out_mar, file.path(miss_dir, paste0(base, "_mar.rds")))

cat(sprintf("[%s] missingness applied: mcar30, mcar50, mar\n", base))
