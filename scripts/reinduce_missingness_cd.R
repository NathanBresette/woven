#!/usr/bin/env Rscript
# reinduce_missingness_cd.R
#
# Re-induces block-level missingness for ARM C and D with the correct design:
#   Every patient retains >= 1 modality. Patients with zero data across all
#   modalities would never be enrolled -- they carry no information and no
#   method (WOVEN or otherwise) can score them. Enforcing >=1 view per subject
#   makes all four simulation arms consistent with this enrollment assumption.
#
# Overwrites existing arm_C/D missing data files in data/missing/.
# Safe to re-run: complete data files are never modified.
#
# Usage: Rscript reinduce_missingness_cd.R <data_dir> <rep_num>
# SLURM array: task_id = rep_num (1-100)

args     <- commandArgs(trailingOnly=TRUE)
data_dir <- args[1]
rep_num  <- as.integer(args[2])
rep_str  <- sprintf("%03d", rep_num)

`%||%` <- function(a, b) if (!is.null(a)) a else b

induce_block_miss <- function(data_list, labels, rate, mechanism="mcar", seed=42L) {
  V <- length(data_list)
  n <- nrow(data_list[[1]])
  set.seed(seed)

  if (mechanism == "mcar") {
    mask <- matrix(runif(n * V) < rate, nrow=n, ncol=V)
  } else if (mechanism == "mar") {
    # Groups 3 and 4 have 2x missingness rate -- minority group structure
    base_rate <- rate * 0.55
    high_rate <- rate * 1.45
    miss_prob <- ifelse(labels %in% c(3L, 4L), high_rate, base_rate)
    miss_prob <- pmin(miss_prob, 0.90)
    mask <- matrix(FALSE, nrow=n, ncol=V)
    for (v in seq_len(V)) mask[, v] <- runif(n) < miss_prob
  }

  # Enforce: every patient has >= 1 modality observed.
  # Patients with zero data would not be enrolled in any study.
  all_miss <- which(rowSums(mask) == V)
  if (length(all_miss) > 0L) {
    for (i in all_miss) {
      mask[i, sample(V, 1L)] <- FALSE
    }
  }

  # Ensure at least 1 complete-case anchor per class (needed for WOVEN fitting)
  for (g in unique(labels)) {
    idx <- which(labels == g)
    if (all(rowSums(mask[idx, , drop=FALSE]) > 0L)) {
      mask[idx[1L], ] <- FALSE
    }
  }

  anchor_idx <- which(rowSums(mask) == 0L)

  masked <- lapply(seq_len(V), function(v) {
    X <- data_list[[v]]
    X[mask[, v], ] <- NA_real_
    X
  })

  list(data=masked, missingness_mask=mask, anchor_idx=anchor_idx,
       n_anchors=length(anchor_idx), rate=rate, mechanism=mechanism,
       effective_rate=mean(mask))
}

miss_dir <- file.path(data_dir, "missing")
dir.create(miss_dir, recursive=TRUE, showWarnings=FALSE)

for (arm in c("C", "D")) {
  comp_file <- file.path(data_dir, "complete",
    sprintf("arm_%s_rep_%s.rds", arm, rep_str))
  if (!file.exists(comp_file)) {
    cat(sprintf("[ARM %s rep %s] complete file not found, skipping\n", arm, rep_str))
    next
  }

  rep_dat <- readRDS(comp_file)
  base_seed <- rep_dat$seed %||% (rep_num * 1000L)

  for (cond in list(
    list(name="mcar30", rate=0.30, mechanism="mcar",  seed_off=1L),
    list(name="mcar50", rate=0.50, mechanism="mcar",  seed_off=2L),
    list(name="mar",    rate=0.35, mechanism="mar",   seed_off=3L)
  )) {
    out_file <- file.path(miss_dir,
      sprintf("arm_%s_rep_%s_%s.rds", arm, rep_str, cond$name))

    result <- induce_block_miss(
      rep_dat$data, rep_dat$labels,
      rate=cond$rate, mechanism=cond$mechanism,
      seed=base_seed + cond$seed_off
    )

    out <- rep_dat
    out$data             <- result$data
    out$missingness_mask <- result$missingness_mask
    out$anchor_idx       <- result$anchor_idx
    out$n_anchors        <- result$n_anchors
    out$condition        <- cond$name
    out$rate             <- cond$rate
    out$mechanism        <- cond$mechanism
    out$effective_rate   <- result$effective_rate

    saveRDS(out, out_file)
    cat(sprintf("[ARM %s rep %s %s] anchors=%d effective_rate=%.2f zero_view_subjects=0\n",
      arm, rep_str, cond$name, result$n_anchors, result$effective_rate))
  }
}

cat(sprintf("[rep %s] Done.\n", rep_str))
