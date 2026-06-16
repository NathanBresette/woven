#!/usr/bin/env Rscript
# patch_ber.R — recompute held-out BER for completed v3 benchmark results
#
# Fixes two bugs in benchmark_one_rep.R ber_held_out:
#   (1) WOVEN: Z_trn used Za_list (eigenvector space) but Z_val used X %*% W
#       (projection space). Mismatched coordinate systems => near-chance BER.
#       Fix: compute Z_trn = X_trn %*% W (same space as Z_val).
#   (2) DIABLO: fallback projection used colnames(loadings) as feature names.
#       colnames are component names ("comp1",...), not feature names.
#       rownames(loadings) are the feature names. Fix: use rownames.
#
# Patches existing benchmark RDS files in-place. All non-BER metrics unchanged.
#
# Usage (SLURM array 1-400):
#   Rscript patch_ber.R <task_id> <data_dir> <bench_dir> <grama_src>

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

cat(sprintf("[patch_ber task %d] ARM %s rep %s\n", task_id, arm_ltr, rep_str))

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

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# Fixed ber_held_out: both Z_trn and Z_val use same coordinate system
ber_held_out_fixed <- function(anchor_idx, labels, cv_folds, z_fold_fn) {
  n_a    <- length(anchor_idx)
  pred   <- integer(n_a)
  failed <- FALSE

  for (f in seq_along(cv_folds)) {
    val_pos <- cv_folds[[f]]
    trn_pos <- unlist(cv_folds[-f])
    trn_a   <- anchor_idx[trn_pos]
    val_a   <- anchor_idx[val_pos]

    res <- tryCatch(z_fold_fn(trn_a, val_a), error = function(e) NULL)
    if (is.null(res)) { failed <- TRUE; break }

    lbl_trn <- labels[trn_a]
    if (length(unique(lbl_trn)) < 2L) { failed <- TRUE; break }

    .nc_pred <- function(Z_trn, lbl, Z_val) {
      classes <- sort(unique(lbl))
      cents   <- do.call(rbind, lapply(classes, function(g)
        colMeans(Z_trn[lbl == g, , drop = FALSE])))
      d  <- as.matrix(dist(rbind(Z_val, cents)))
      nv <- nrow(Z_val)
      classes[apply(d[seq_len(nv), (nv+1L):(nv+length(classes)), drop=FALSE],
                    1L, which.min)]
    }

    pred_val <- tryCatch(
      .nc_pred(res$Z_trn, lbl_trn, res$Z_val),
      error = function(e) NULL
    )
    if (is.null(pred_val) || length(pred_val) != length(val_pos)) {
      failed <- TRUE; break
    }
    pred[val_pos] <- pred_val
  }

  if (failed) return(NA_real_)

  lbl_int <- as.integer(factor(labels[anchor_idx]))
  mean(vapply(sort(unique(lbl_int)), function(g) {
    idx <- which(lbl_int == g)
    if (!length(idx)) return(NA_real_)
    1 - mean(pred[idx] == g)
  }, numeric(1L)), na.rm = TRUE)
}

# ── WOVEN BER (fixed: Z_trn = X_trn %*% W, not Za_list) ─────────────────────
compute_woven_ber <- function(X_list_miss, anchor_idx, labels, K, cv_folds) {
  lambda_grid <- c(0.001, 0.005, 0.01, 0.1, 0.5)
  gamma_grid  <- c(0.5, 1.0, 5.0, 10.0)
  best_lambda <- 0.01; best_gamma <- 1.0; best_sil <- -Inf

  make_precomp_La <- function(idx) {
    lapply(X_list_miss, function(X) {
      L <- build_laplacian(X, k = 10L)
      as.matrix(L[idx, idx, drop=FALSE])
    })
  }

  # Stage 1: best lambda
  for (lam in lambda_grid) {
    sils <- unlist(parallel::mclapply(seq_along(cv_folds), function(f) {
      trn_a <- anchor_idx[-cv_folds[[f]]]
      val_a <- anchor_idx[cv_folds[[f]]]
      tryCatch({
        La_trn <- make_precomp_La(trn_a)
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
        La_trn <- make_precomp_La(trn_a)
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

  # Held-out BER with FIXED z_fold_fn (Z_trn = X_trn %*% W)
  ber_held_out_fixed(anchor_idx, labels, cv_folds, function(trn_a, val_a) {
    tryCatch({
      La_trn <- make_precomp_La(trn_a)
      f_trn  <- woven_mcca_dual(X_list_miss, trn_a, labels, K=K,
        lambdas=best_lambda, gamma_y=best_gamma,
        La_list_precomp=La_trn, verbose=FALSE)
      # FIX: both Z_trn and Z_val use X %*% W (same coordinate system)
      Z_trn <- Reduce("+", lapply(seq_len(V), function(v) {
        xv <- X_list_miss[[v]][trn_a, , drop=FALSE]
        if (any(is.na(xv))) xv <- na_impute_median(xv)
        xv %*% f_trn$W_list[[v]]
      })) / V
      Z_val <- Reduce("+", lapply(seq_len(V), function(v) {
        xv <- X_list_miss[[v]][val_a, , drop=FALSE]
        if (any(is.na(xv))) xv <- na_impute_median(xv)
        xv %*% f_trn$W_list[[v]]
      })) / V
      list(Z_trn = Z_trn, Z_val = Z_val)
    }, error = function(e) NULL)
  })
}

# ── DIABLO BER (fixed: rownames for feature selection) ────────────────────────
compute_diablo_ber <- function(X_list_miss, anchor_idx, labels, K, cv_folds) {
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

  # CV over keepX
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
          Xv <- Xa_val[[m]]
          feat   <- rownames(fit_cv$loadings[[m]])  # FIX: rownames
          common <- intersect(colnames(Xv), feat)
          if (length(common) < 2L) return(matrix(0, nrow(Xv), K))
          Xv[, common, drop=FALSE] %*% fit_cv$loadings[[m]][common, seq_len(K), drop=FALSE]
        })) / V
        anchor_sil(Z_val, labels[val_a])
      }, error=function(e) NA_real_)
    })
    ms <- mean(sils, na.rm=TRUE)
    if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_keepX <- kx }
  }

  cat(sprintf(" [keepX=%d]", best_keepX))

  ber_held_out_fixed(anchor_idx, labels, cv_folds, function(trn_a, val_a) {
    tryCatch({
      X_trn <- prep_Xa(trn_a); names(X_trn) <- mod_names
      kx    <- lapply(X_trn, function(Xa) rep(min(best_keepX, ncol(Xa)), K))
      names(kx) <- mod_names
      des   <- matrix(0.1, V, V, dimnames=list(mod_names, mod_names)); diag(des) <- 0
      f_trn <- mixOmics::block.splsda(X=X_trn, Y=factor(labels[trn_a]),
        ncomp=K, design=des, near.zero.var=TRUE, keepX=kx)
      Z_trn <- Reduce("+", lapply(mod_names, function(m) f_trn$variates[[m]]))/V
      rownames(Z_trn) <- NULL
      X_val <- prep_Xa(val_a); names(X_val) <- mod_names
      Z_val <- Reduce("+", lapply(mod_names, function(m) {
        Xv     <- X_val[[m]]
        feat   <- rownames(f_trn$loadings[[m]])  # FIX: rownames not colnames
        common <- intersect(colnames(Xv), feat)
        if (length(common) < 2L) return(matrix(0, nrow(Xv), K))
        Xv[, common, drop=FALSE] %*% f_trn$loadings[[m]][common, seq_len(K), drop=FALSE]
      })) / V
      list(Z_trn = Z_trn, Z_val = matrix(Z_val, nrow=length(val_a)))
    }, error = function(e) NULL)
  })
}

# ── Main: iterate conditions, patch BER in-place ─────────────────────────────
all_conditions <- c("complete", "mcar30", "mcar50", "mar")
K              <- 5L
patched        <- FALSE

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
  ber_g <- tryCatch(
    compute_woven_ber(X_list_miss, anchor_idx, labels, K, cv_folds),
    error = function(e) { cat(" ERROR:", conditionMessage(e)); NA_real_ }
  )
  cat(sprintf(" %.3f [%.0fs]\n", ber_g, (proc.time()-t0)[["elapsed"]]))

  cat("    DIABLO BER...")
  t0 <- proc.time()
  ber_d <- tryCatch(
    compute_diablo_ber(X_list_miss, anchor_idx, labels, K, cv_folds),
    error = function(e) { cat(" ERROR:", conditionMessage(e)); NA_real_ }
  )
  cat(sprintf(" %.3f [%.0fs]\n", ber_d, (proc.time()-t0)[["elapsed"]]))

  # Patch in-place
  obj$results[[cond]]$WOVEN$ber  <- ber_g
  obj$results[[cond]]$DIABLO$ber <- ber_d
  patched <- TRUE
}

if (patched) {
  saveRDS(obj, bench_file)
  cat(sprintf("[patch_ber task %d] Patched: %s\n", task_id, bench_file))
} else {
  cat(sprintf("[patch_ber task %d] No patches applied.\n", task_id))
}
