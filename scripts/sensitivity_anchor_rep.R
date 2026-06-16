#!/usr/bin/env Rscript
# sensitivity_anchor_rep.R -- Anchor fraction sensitivity for WOVEN
#
# Sweeps anchor_frac (fraction of subjects designated as fully-observed anchors)
# from 0.05 to 0.75, stratified by class label. Non-anchor subjects receive
# MCAR 50% block-missingness (same as the hardest benchmark condition), so this
# directly tests: given realistic missing data, how few anchors are required?
#
# Two arms per rep:
#   ARM A  (V=2, RNA-seq + methylation, diffuse Gaussian signal -- hardest)
#   ARM C  (V=3, InterSIM TCGA-OV, concentrated signal -- easiest)
# Reporting both provides conservative and liberal bounds for reviewers.
#
# Hyperparameters held at benchmark defaults (lambda=0.01, gamma_y=1.0, k_nn=10)
# to isolate the effect of anchor fraction from tuning.
#
# Metrics per condition:
#   sil_all       silhouette over all scored subjects (anchors + Nystrom-projected)
#   sil_anchor    silhouette over anchor subjects only (fit quality)
#   sil_nonanc    silhouette over non-anchor subjects only (Nystrom projection quality)
#   nmi           NMI over all scored subjects
#   n_scored      number of subjects with a latent score
#   elapsed       wall time (seconds)
#
# Usage (SLURM array):
#   Rscript sensitivity_anchor_rep.R <rep_num> <data_dir> <out_dir> <grama_src>
# rep_num 1-50 -> ARM A and ARM C reps 001-050

suppressPackageStartupMessages({
  library(Matrix)
  library(RANN)
  library(RSpectra)
  library(cluster)
})

args      <- commandArgs(trailingOnly = TRUE)
rep_num   <- as.integer(args[1])
data_dir  <- args[2]
out_dir   <- args[3]
grama_src <- args[4]

for (f in c("utils.R", "laplacian.R", "solver_mcca_dual.R", "project.R", "metrics.R"))
  source(file.path(grama_src, f))

set.seed(rep_num * 777L)

# ── Anchor fraction grid ─────────────────────────────────────────────────────
anchor_fracs <- c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75)

# ── Helpers ──────────────────────────────────────────────────────────────────

# Stratified sample of anchor indices: equal fraction per class
stratified_anchors <- function(labels, frac, seed) {
  set.seed(seed)
  classes <- sort(unique(labels))
  unlist(lapply(classes, function(g) {
    idx <- which(labels == g)
    n_draw <- max(2L, round(length(idx) * frac))  # at least 2 per class
    n_draw <- min(n_draw, length(idx))
    sample(idx, n_draw)
  }))
}

# Apply MCAR block-missingness: for each non-anchor subject, each modality
# independently missing with prob 0.5 (at least one modality kept so subject
# has at least one score). Entire row of that modality set to NA.
apply_mcar50 <- function(X_list, non_anc_idx, seed) {
  set.seed(seed)
  V <- length(X_list)
  X_out <- lapply(X_list, function(X) X)  # deep copy
  for (i in non_anc_idx) {
    # draw which modalities are missing (independent Bernoulli 0.5 per view)
    # ensure at least one view is kept so subject contributes something
    repeat {
      miss <- rbinom(V, 1L, 0.5) == 1L
      if (!all(miss)) break
    }
    for (v in seq_len(V)) {
      if (miss[v]) X_out[[v]][i, ] <- NA_real_
    }
  }
  X_out
}

sil_mean <- function(Z, lbl) {
  if (is.null(Z) || nrow(Z) < 4L || length(unique(lbl)) < 2L) return(NA_real_)
  tryCatch(
    mean(silhouette(as.integer(factor(lbl)), dist(Z))[, 3], na.rm = TRUE),
    error = function(e) NA_real_
  )
}

nmi_val <- function(Z, lbl) {
  tryCatch(woven_nmi(Z, lbl), error = function(e) NA_real_)
}

# ── Per-arm runner ────────────────────────────────────────────────────────────
run_arm <- function(arm_ltr, rep_num, data_dir) {
  rep_str  <- sprintf("%03d", rep_num)
  rep_file <- file.path(data_dir, "complete",
                        sprintf("arm_%s_rep_%s.rds", arm_ltr, rep_str))
  if (!file.exists(rep_file)) {
    cat(sprintf("  [SKIP] %s not found\n", rep_file))
    return(NULL)
  }

  rep_dat <- readRDS(rep_file)
  X_full  <- rep_dat$data
  labels  <- rep_dat$labels
  V       <- length(X_full)
  n       <- nrow(X_full[[1]])
  K       <- 5L

  cat(sprintf("  ARM %s | n=%d V=%d p=%s\n",
              arm_ltr, n, V, paste(sapply(X_full, ncol), collapse="+")))

  rows <- list()

  for (frac in anchor_fracs) {
    cat(sprintf("    anchor_frac=%.2f...", frac))

    anchor_seed <- rep_num * 1000L + round(frac * 100)
    anchor_idx  <- stratified_anchors(labels, frac, seed = anchor_seed)
    non_anc_idx <- setdiff(seq_len(n), anchor_idx)
    n_a         <- length(anchor_idx)

    X_miss <- apply_mcar50(X_full, non_anc_idx, seed = anchor_seed + 1L)

    t0 <- proc.time()
    fit <- tryCatch({
      woven_mcca_dual(
        X_list  = X_miss,
        anchor_idx = anchor_idx,
        Y       = labels,
        K       = K,
        lambdas = 0.01,
        gamma_y = 1.0,
        k_nn    = 10L,
        verbose = FALSE
      )
    }, error = function(e) {
      cat(sprintf(" ERROR: %s\n", conditionMessage(e)))
      NULL
    })
    elapsed <- (proc.time() - t0)[["elapsed"]]

    if (is.null(fit)) {
      rows[[length(rows)+1]] <- data.frame(
        rep=rep_num, arm=arm_ltr, anchor_frac=frac,
        n_anchor=n_a, n_nonanc=length(non_anc_idx),
        sil_all=NA, sil_anchor=NA, sil_nonanc=NA, nmi=NA,
        n_scored=NA, elapsed=elapsed, error="fit failed",
        stringsAsFactors=FALSE
      )
      next
    }

    # Anchor latent scores (mean across views)
    Za_mean <- Reduce("+", fit$Za_list) / V

    # Project non-anchor subjects: direct W for available views
    Z_full <- matrix(NA_real_, nrow=n, ncol=K)
    Z_full[anchor_idx, ] <- Za_mean

    for (i in non_anc_idx) {
      scores <- lapply(seq_len(V), function(v) {
        xi <- X_miss[[v]][i, , drop=FALSE]
        if (all(is.na(xi))) return(NULL)
        cols <- fit$col_ok_list[[v]]
        xi_c <- xi[, cols, drop=FALSE]
        med  <- apply(fit$Xa_list[[v]], 2, median, na.rm=TRUE)
        med[!is.finite(med)] <- 0
        for (j in seq_len(ncol(xi_c))) {
          if (is.na(xi_c[1, j])) xi_c[1, j] <- med[j]
        }
        xi_c %*% fit$W_list[[v]]
      })
      valid <- Filter(Negate(is.null), scores)
      if (length(valid) > 0L) Z_full[i, ] <- colMeans(do.call(rbind, valid))
    }

    has_score   <- which(!is.na(Z_full[, 1]))
    n_scored    <- length(has_score)

    # Silhouette: all scored, anchors only, non-anchors only
    Z_all  <- Z_full[has_score, , drop=FALSE]
    lbl_all <- labels[has_score]
    sil_all <- sil_mean(Z_all, lbl_all)

    Za_scored <- Z_full[anchor_idx, , drop=FALSE]
    sil_anc   <- sil_mean(Za_scored, labels[anchor_idx])

    non_scored <- intersect(non_anc_idx, has_score)
    sil_non <- if (length(non_scored) >= 4L && length(unique(labels[non_scored])) >= 2L)
      sil_mean(Z_full[non_scored, , drop=FALSE], labels[non_scored])
    else NA_real_

    nmi <- nmi_val(Z_all, lbl_all)

    cat(sprintf(" sil_all=%.3f sil_nonanc=%.3f [%.0fs]\n", sil_all, sil_non, elapsed))

    rows[[length(rows)+1]] <- data.frame(
      rep=rep_num, arm=arm_ltr, anchor_frac=frac,
      n_anchor=n_a, n_nonanc=length(non_anc_idx),
      sil_all=sil_all, sil_anchor=sil_anc, sil_nonanc=sil_non,
      nmi=nmi, n_scored=n_scored, elapsed=elapsed, error="",
      stringsAsFactors=FALSE
    )
  }

  do.call(rbind, rows)
}

# ── Run both arms ─────────────────────────────────────────────────────────────
rep_str <- sprintf("%03d", rep_num)
cat(sprintf("[rep %s] Anchor fraction sweep\n", rep_str))

results_a <- run_arm("A", rep_num, data_dir)
results_c <- run_arm("C", rep_num, data_dir)

out_df <- rbind(results_a, results_c)

dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
out_file <- file.path(out_dir, sprintf("anchor_rep_%s.rds", rep_str))
saveRDS(out_df, out_file)
cat(sprintf("[rep %s] Done. Saved: %s\n", rep_str, out_file))
