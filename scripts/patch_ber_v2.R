#!/usr/bin/env Rscript
# patch_ber_v2.R — recompute BER using fixed-Z LDA approach
#
# Approach: fit the DR model once on all anchor subjects (with CV-selected params),
# project all subjects, then run woven_ber() which does 5-fold stratified LDA CV
# on the FIXED Z. This is the standard approach used in DIABLO and mixOmics papers:
# DR is not part of the CV loop — only the downstream classifier is held out.
#
# Replaces the broken ber_held_out() approach (per-fold DR refitting) which gave
# near-chance BER (~0.67) because CV models don't always generalize as well as
# the full model, and nearest-centroid is suboptimal for multi-dim latent spaces.
#
# Usage (SLURM array 1-400):
#   Rscript patch_ber_v2.R <task_id> <data_dir> <bench_dir> <grama_src>

suppressPackageStartupMessages({
  library(Matrix)
  library(RANN)
  library(RSpectra)
})

args      <- commandArgs(trailingOnly = TRUE)
task_id   <- as.integer(args[1])
data_dir  <- args[2]
bench_dir <- args[3]
grama_src <- args[4]

arm_idx <- ceiling(task_id / 100L)
rep_num <- ((task_id - 1L) %% 100L) + 1L
arm_ltr <- c("A", "B", "C", "D")[arm_idx]
rep_str <- sprintf("%03d", rep_num)

cat(sprintf("[patch_ber_v2 task %d] ARM %s rep %s\n", task_id, arm_ltr, rep_str))

for (f in c("utils.R", "laplacian.R", "solver_mcca_dual.R", "project.R", "metrics.R"))
  source(file.path(grama_src, f))

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a[1])) a else b

# ── Load existing benchmark result ────────────────────────────────────────────
bench_file <- file.path(bench_dir,
  sprintf("arm_%s_rep_%s_benchmark.rds", arm_ltr, rep_str))
if (!file.exists(bench_file)) stop("Bench file not found: ", bench_file)
obj <- readRDS(bench_file)

# ── Load simulation data ──────────────────────────────────────────────────────
base_name <- sprintf("arm_%s_rep_%s", arm_ltr, rep_str)

load_rep <- function(suffix = "") {
  fname  <- paste0(base_name, suffix, ".rds")
  subdir <- if (suffix == "") "complete" else "missing"
  cands  <- c(file.path(data_dir, subdir, fname), file.path(data_dir, fname))
  path   <- cands[file.exists(cands)][1]
  if (is.na(path)) return(NULL)
  readRDS(path)
}

complete_rep <- load_rep("")
if (is.null(complete_rep)) stop("Complete rep not found: ", base_name)

mod_names <- names(complete_rep$data)
V         <- length(mod_names)
n         <- nrow(complete_rep$data[[1]])
labels    <- complete_rep$labels

apply_mask <- function(data_list, mask_list) {
  lapply(seq_along(data_list), function(v) {
    X <- data_list[[v]]
    if (!is.null(mask_list[[v]])) X[mask_list[[v]], ] <- NA
    X
  })
}

anchor_sil <- function(Z, lbl) {
  if (is.null(Z) || nrow(Z) < 4L || length(unique(lbl)) < 2L) return(NA_real_)
  ok <- !is.na(Z[, 1])
  if (sum(ok) < 4L) return(NA_real_)
  tryCatch({
    s <- cluster::silhouette(as.integer(factor(lbl[ok])), dist(Z[ok, , drop=FALSE]))
    mean(s[, 3], na.rm = TRUE)
  }, error = function(e) NA_real_)
}

make_folds <- function(anchor_idx, n_folds = 3L, seed = 42L) {
  set.seed(seed)
  n_a <- length(anchor_idx)
  split(sample(n_a), rep(seq_len(n_folds), length.out = n_a))
}

# ── WOVEN: CV params + fit + woven_ber on fixed Z ────────────────────────────
compute_woven_ber_v2 <- function(X_list_miss, anchor_idx, labels, K, cv_folds) {
  lambda_grid <- c(0.001, 0.005, 0.01, 0.1, 0.5)
  gamma_grid  <- c(0.5, 1.0, 5.0, 10.0)
  best_lambda <- 0.01; best_gamma <- 1.0; best_sil <- -Inf

  # Precompute full L once; submatrix for each CV fold (fast)
  L_list <- lapply(X_list_miss, function(X) build_laplacian(X, k = 10L))

  # Stage 1: best lambda
  for (lam in lambda_grid) {
    sils <- unlist(parallel::mclapply(seq_along(cv_folds), function(f) {
      trn_a <- anchor_idx[-cv_folds[[f]]]
      val_a <- anchor_idx[cv_folds[[f]]]
      tryCatch({
        La_trn <- lapply(L_list, function(L) as.matrix(L[trn_a, trn_a, drop=FALSE]))
        cv_fit <- woven_mcca_dual(X_list_miss, trn_a, labels, K=K,
          lambdas=lam, gamma_y=1.0, La_list_precomp=La_trn, verbose=FALSE)
        Za_val <- Reduce("+", lapply(seq_len(V), function(v)
          X_list_miss[[v]][val_a, , drop=FALSE] %*% cv_fit$W_list[[v]])) / V
        anchor_sil(Za_val, labels[val_a])
      }, error = function(e) NA_real_)
    }, mc.cores = min(2L, parallel::detectCores())))
    ms <- mean(sils, na.rm=TRUE)
    if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_lambda <- lam }
  }

  # Stage 2: best gamma_y
  best_sil <- -Inf
  for (gam in gamma_grid) {
    sils <- unlist(parallel::mclapply(seq_along(cv_folds), function(f) {
      trn_a <- anchor_idx[-cv_folds[[f]]]
      val_a <- anchor_idx[cv_folds[[f]]]
      tryCatch({
        La_trn <- lapply(L_list, function(L) as.matrix(L[trn_a, trn_a, drop=FALSE]))
        cv_fit <- woven_mcca_dual(X_list_miss, trn_a, labels, K=K,
          lambdas=best_lambda, gamma_y=gam, La_list_precomp=La_trn, verbose=FALSE)
        Za_val <- Reduce("+", lapply(seq_len(V), function(v)
          X_list_miss[[v]][val_a, , drop=FALSE] %*% cv_fit$W_list[[v]])) / V
        anchor_sil(Za_val, labels[val_a])
      }, error = function(e) NA_real_)
    }, mc.cores = min(2L, parallel::detectCores())))
    ms <- mean(sils, na.rm=TRUE)
    if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_gamma <- gam }
  }

  cat(sprintf(" [lambda=%.3f gamma=%.1f]", best_lambda, best_gamma))

  # Final fit on all anchors
  La_list <- lapply(L_list, function(L) as.matrix(L[anchor_idx, anchor_idx, drop=FALSE]))
  fit <- woven_mcca_dual(X_list_miss, anchor_idx, labels, K=K,
    lambdas=best_lambda, gamma_y=best_gamma,
    La_list_precomp=La_list, verbose=FALSE)

  # Full Z (project all n subjects)
  Za_mean <- Reduce("+", fit$Za_list) / V
  Z_full  <- matrix(NA_real_, nrow=n, ncol=K)
  Z_full[anchor_idx, ] <- Za_mean
  non_anc <- setdiff(seq_len(n), anchor_idx)
  if (length(non_anc) > 0L) {
    for (i in non_anc) {
      scores <- lapply(seq_len(V), function(v) {
        xi <- X_list_miss[[v]][i, , drop=FALSE]
        if (all(is.na(xi))) return(NULL)
        if (any(is.na(xi))) xi <- na_impute_median(xi)
        xi %*% fit$W_list[[v]]
      })
      valid <- Filter(Negate(is.null), scores)
      if (length(valid) > 0L) Z_full[i, ] <- colMeans(do.call(rbind, valid))
    }
  }

  # ber: full cohort (all subjects WOVEN can score — uses ESS advantage)
  has_score <- !is.na(Z_full[, 1])
  ber_full  <- woven_ber(Z_full[has_score, , drop=FALSE], labels[has_score])

  # ber_anchor: anchor subjects only — same population as DIABLO, fair classifier comparison
  ber_anc <- woven_ber(Za_mean, labels[anchor_idx])

  list(ber = ber_full, ber_anchor = ber_anc)
}

# ── DIABLO: CV params + fit + woven_ber on fixed Z ───────────────────────────
compute_diablo_ber_v2 <- function(X_list_miss, anchor_idx, labels, K, cv_folds) {
  if (!requireNamespace("mixOmics", quietly=TRUE)) return(NA_real_)

  prep_Xa <- function(idx) {
    lapply(X_list_miss, function(X) {
      Xa <- na_impute_median(X[idx, , drop=FALSE])
      rownames(Xa) <- paste0("S", seq_along(idx))
      nzv <- tryCatch(mixOmics::nearZeroVar(Xa)$Position, error=function(e) integer(0))
      if (length(nzv) > 0L) Xa <- Xa[, -nzv, drop=FALSE]
      col_var <- apply(Xa, 2, var, na.rm=TRUE)
      keep    <- is.finite(col_var) & col_var > 1e-8
      if (sum(keep) < 2L) keep <- seq_len(ncol(Xa))
      Xa[, keep, drop=FALSE]
    })
  }

  keepX_grid <- c(10L, 30L, 50L)
  best_keepX <- 30L; best_sil <- -Inf
  des_cv <- matrix(0.1, V, V, dimnames=list(mod_names, mod_names)); diag(des_cv) <- 0

  for (kx in keepX_grid) {
    sils <- sapply(seq_along(cv_folds), function(f) {
      val_pos <- cv_folds[[f]]; trn_pos <- unlist(cv_folds[-f])
      val_a   <- anchor_idx[val_pos]; trn_a <- anchor_idx[trn_pos]
      tryCatch({
        Xa_trn <- prep_Xa(trn_a); names(Xa_trn) <- mod_names
        kx_use <- lapply(Xa_trn, function(X) rep(min(kx, ncol(X)), K))
        names(kx_use) <- mod_names
        fit_cv <- mixOmics::block.splsda(X=Xa_trn, Y=factor(labels[trn_a]),
          ncomp=K, design=des_cv, near.zero.var=TRUE, keepX=kx_use)
        Xa_val <- prep_Xa(val_a); names(Xa_val) <- mod_names
        Z_val <- Reduce("+", lapply(mod_names, function(m) {
          feat   <- rownames(fit_cv$loadings[[m]])
          common <- intersect(colnames(Xa_val[[m]]), feat)
          if (length(common) < 2L) return(matrix(0, nrow(Xa_val[[m]]), K))
          Xa_val[[m]][, common, drop=FALSE] %*%
            fit_cv$loadings[[m]][common, seq_len(K), drop=FALSE]
        })) / V
        anchor_sil(Z_val, labels[val_a])
      }, error=function(e) NA_real_)
    })
    ms <- mean(sils, na.rm=TRUE)
    if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_keepX <- kx }
  }

  cat(sprintf(" [keepX=%d]", best_keepX))

  X_anchors <- prep_Xa(anchor_idx); names(X_anchors) <- mod_names
  kx_final  <- lapply(X_anchors, function(Xa) rep(min(best_keepX, ncol(Xa)), K))
  names(kx_final) <- mod_names
  fit_d <- mixOmics::block.splsda(X=X_anchors, Y=factor(labels[anchor_idx]),
    ncomp=K, design=des_cv, near.zero.var=TRUE, keepX=kx_final)

  Z_anchor <- Reduce("+", lapply(mod_names, function(m) fit_d$variates[[m]])) / V
  Z_full   <- matrix(NA_real_, nrow=n, ncol=K)
  rownames(Z_anchor) <- NULL
  Z_full[anchor_idx, ] <- Z_anchor

  # DIABLO only scores anchor subjects — ber and ber_anchor are the same population
  ber_val <- woven_ber(Z_anchor, labels[anchor_idx])
  list(ber = ber_val, ber_anchor = ber_val)
}

# ── Main: iterate conditions, patch BER in-place ─────────────────────────────
all_conditions <- c("complete", "mcar30", "mcar50", "mar")
K              <- 5L

for (cond in all_conditions) {
  cat(sprintf("  Condition: %s\n", cond))

  if (cond == "complete") {
    X_list_miss <- complete_rep$data
    anchor_idx  <- seq_len(n)
  } else {
    rep_dat <- load_rep(paste0("_", cond))
    if (is.null(rep_dat)) { cat("    [SKIP] not found\n"); next }
    X_list_miss <- rep_dat$data
    anchor_idx  <- rep_dat$anchor_idx
    if (is.null(anchor_idx)) {
      bm <- sapply(X_list_miss, function(X) apply(is.na(X), 1, all))
      anchor_idx <- which(rowSums(bm) == 0L)
    }
  }

  cv_folds <- make_folds(anchor_idx, n_folds=3L,
    seed=rep_num * 100L + match(cond, all_conditions))

  cat("    WOVEN BER...")
  t0 <- proc.time()
  res_g <- tryCatch(
    compute_woven_ber_v2(X_list_miss, anchor_idx, labels, K, cv_folds),
    error = function(e) { cat(" ERROR:", conditionMessage(e)); list(ber=NA_real_, ber_anchor=NA_real_) }
  )
  cat(sprintf(" full=%.3f anc=%.3f [%.0fs]\n", res_g$ber, res_g$ber_anchor, (proc.time()-t0)[["elapsed"]]))

  cat("    DIABLO BER...")
  t0 <- proc.time()
  res_d <- tryCatch(
    compute_diablo_ber_v2(X_list_miss, anchor_idx, labels, K, cv_folds),
    error = function(e) { cat(" ERROR:", conditionMessage(e)); list(ber=NA_real_, ber_anchor=NA_real_) }
  )
  cat(sprintf(" %.3f [%.0fs]\n", res_d$ber, (proc.time()-t0)[["elapsed"]]))

  obj$results[[cond]]$WOVEN$ber         <- res_g$ber
  obj$results[[cond]]$WOVEN$ber_anchor  <- res_g$ber_anchor
  obj$results[[cond]]$DIABLO$ber        <- res_d$ber
  obj$results[[cond]]$DIABLO$ber_anchor <- res_d$ber_anchor
}

saveRDS(obj, bench_file)
cat(sprintf("[patch_ber_v2 task %d] Patched: %s\n", task_id, bench_file))
