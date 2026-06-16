#!/usr/bin/env Rscript
# sensitivity_one_rep.R — hyperparameter sensitivity analysis for one ARM A rep
#
# Sweeps lambda, gamma_y, and k_nn one at a time (others held at default).
# Runs only on ARM A (V=2, fast) complete-data condition so the effect of each
# parameter is cleanest. Anchor silhouette is the primary metric.
#
# Defaults: lambda=0.1, gamma_y=1.0, k_nn=15
#
# Usage (called by SLURM array):
#   Rscript sensitivity_one_rep.R <rep_num> <data_dir> <out_dir> <grama_src>
# rep_num 1-30 → ARM A reps 001-030

suppressPackageStartupMessages({
  library(Matrix); library(RANN); library(RSpectra); library(cluster)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

args    <- commandArgs(trailingOnly = TRUE)
rep_num <- as.integer(args[1])
data_dir <- args[2]
out_dir  <- args[3]
grama_src <- args[4]

for (f in c("utils.R","laplacian.R","solver_v2.R","solver_als.R","project.R","metrics.R"))
  source(file.path(grama_src, f))

rep_str  <- sprintf("%03d", rep_num)
rep_file <- file.path(data_dir, "complete", sprintf("arm_A_rep_%s.rds", rep_str))
if (!file.exists(rep_file)) stop("File not found: ", rep_file)

rep_dat  <- readRDS(rep_file)
X1       <- rep_dat$data[[1]]
X2       <- rep_dat$data[[2]]
labels   <- rep_dat$labels
n        <- nrow(X1)
anchor_idx <- seq_len(n)   # complete condition: all subjects are anchors
K        <- 5L

cat(sprintf("[rep %s] n=%d, p1=%d, p2=%d\n", rep_str, n, ncol(X1), ncol(X2)))

# ── Parameter grids ───────────────────────────────────────────────────────────
grids <- list(
  lambda  = c(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5),
  gamma_y = c(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
  k_nn    = c(3L, 5L, 10L, 15L, 25L, 50L)
)
defaults <- list(lambda = 0.01, gamma_y = 1.0, k_nn = 10L)

anchor_sil <- function(fit) {
  Za <- fit$Za1
  if (is.null(Za)) Za <- fit$Z_list[[1]]
  if (is.null(Za) || nrow(Za) < 4 || length(unique(labels[anchor_idx])) < 2)
    return(NA_real_)
  tryCatch(
    mean(silhouette(as.integer(factor(labels[anchor_idx])), dist(Za))[, 3]),
    error = function(e) NA_real_
  )
}

# Precompute L and Omega at default k_nn once — reused across lambda/gamma_y sweeps
cat("  Precomputing Laplacian (default k_nn=15)...\n")
L1_def    <- build_laplacian(X1, k = 15L)
Xc1       <- na_impute_median(X1)
Omega1_def <- as.matrix(crossprod(Xc1, as.matrix(L1_def %*% Xc1))) / nrow(Xc1)
XtX1_def  <- crossprod(Xc1)

L2_def    <- build_laplacian(X2, k = 15L)
Xc2       <- na_impute_median(X2)
Omega2_def <- as.matrix(crossprod(Xc2, as.matrix(L2_def %*% Xc2))) / nrow(Xc2)
XtX2_def  <- crossprod(Xc2)

rows <- list()

run_fit <- function(lambda, gamma_y, k_nn,
                    L1 = L1_def, L2 = L2_def,
                    Omega1 = Omega1_def, Omega2 = Omega2_def,
                    XtX1 = XtX1_def, XtX2 = XtX2_def) {
  tryCatch({
    t0  <- proc.time()
    fit <- woven_v2(X1, X2, anchor_idx = anchor_idx, Y = labels, K = K,
                    lambda1 = lambda, lambda2 = lambda,
                    gamma_y = gamma_y, k_nn = k_nn,
                    L1_precomp     = L1,     L2_precomp     = L2,
                    Omega1_precomp = Omega1, Omega2_precomp = Omega2,
                    XtX1_precomp   = XtX1,  XtX2_precomp   = XtX2)
    elapsed <- (proc.time() - t0)[["elapsed"]]
    list(sil = anchor_sil(fit), elapsed = elapsed, error = NULL)
  }, error = function(e) list(sil = NA_real_, elapsed = NA_real_, error = conditionMessage(e)))
}

# ── Lambda sweep (gamma_y=1.0, k_nn=15 fixed) ─────────────────────────────────
cat("  Lambda sweep...\n")
for (lam in grids$lambda) {
  r <- run_fit(lambda = lam, gamma_y = defaults$gamma_y, k_nn = defaults$k_nn)
  rows[[length(rows)+1]] <- data.frame(
    rep = rep_num, parameter = "lambda", value = lam,
    sil_anchor = r$sil, elapsed = r$elapsed, error = r$error %||% "",
    stringsAsFactors = FALSE
  )
  cat(sprintf("    lambda=%.3f  sil=%.4f\n", lam, r$sil))
}

# ── gamma_y sweep (lambda=0.1, k_nn=15 fixed) ─────────────────────────────────
cat("  gamma_y sweep...\n")
for (gam in grids$gamma_y) {
  r <- run_fit(lambda = defaults$lambda, gamma_y = gam, k_nn = defaults$k_nn)
  rows[[length(rows)+1]] <- data.frame(
    rep = rep_num, parameter = "gamma_y", value = gam,
    sil_anchor = r$sil, elapsed = r$elapsed, error = r$error %||% "",
    stringsAsFactors = FALSE
  )
  cat(sprintf("    gamma_y=%.2f  sil=%.4f\n", gam, r$sil))
}

# ── k_nn sweep (lambda=0.1, gamma_y=1.0; rebuild L/Omega per k) ───────────────
cat("  k_nn sweep...\n")
`%||%` <- function(a, b) if (!is.null(a)) a else b
for (knn in grids$k_nn) {
  # Need fresh L/Omega for each k_nn
  L1k    <- build_laplacian(X1, k = knn)
  Omega1k <- as.matrix(crossprod(Xc1, as.matrix(L1k %*% Xc1))) / nrow(Xc1)
  L2k    <- build_laplacian(X2, k = knn)
  Omega2k <- as.matrix(crossprod(Xc2, as.matrix(L2k %*% Xc2))) / nrow(Xc2)

  r <- run_fit(lambda = defaults$lambda, gamma_y = defaults$gamma_y, k_nn = knn,
               L1 = L1k, L2 = L2k, Omega1 = Omega1k, Omega2 = Omega2k)
  rows[[length(rows)+1]] <- data.frame(
    rep = rep_num, parameter = "k_nn", value = knn,
    sil_anchor = r$sil, elapsed = r$elapsed, error = r$error %||% "",
    stringsAsFactors = FALSE
  )
  cat(sprintf("    k_nn=%d  sil=%.4f\n", knn, r$sil))
}

# ── Save ──────────────────────────────────────────────────────────────────────
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, sprintf("sensitivity_rep_%s.rds", rep_str))
saveRDS(do.call(rbind, rows), out_file)
cat(sprintf("[rep %s] Done. Saved: %s\n", rep_str, out_file))
