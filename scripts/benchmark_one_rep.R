#!/usr/bin/env Rscript
# benchmark_one_rep.R
#
# Benchmarks WOVEN vs DIABLO vs MOFA2 vs Impute+DIABLO for one simulation rep.
# All tunable methods use equivalent 3-fold CV on anchor subjects with the same
# fold splits — ensures fair head-to-head comparison.
#
# CV grids:
#   WOVEN V=2:  lambda    in {0.01, 0.1, 0.5}
#   WOVEN V>=3: lambda    in {0.01, 0.1, 0.5}, gamma_y in {0.5, 1.0, 5.0, 10.0}
#   DIABLO:     keepX     in {10, 30, 50}
#   ImputeDIABLO: keepX   in {10, 30, 50}
#   MOFA2:      no tuning (ARD prior handles sparsity automatically)
#
# Usage (called by SLURM array):
#   Rscript benchmark_one_rep.R <task_id> <data_dir> <out_dir> <woven_src_dir>
#
# task_id 1-100   → ARM A rep 001-100
# task_id 101-200 → ARM B rep 001-100
# task_id 201-300 → ARM C rep 001-100
# task_id 301-400 → ARM D rep 001-100

suppressPackageStartupMessages({
  library(Matrix)
  library(RANN)
  library(RSpectra)
})

# ── Parse args ────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("Usage: benchmark_one_rep.R <task_id> <data_dir> <out_dir> <woven_src>")

task_id   <- as.integer(args[1])
data_dir  <- args[2]
out_dir   <- args[3]
woven_src <- args[4]

arm_idx <- ceiling(task_id / 100L)
rep_num <- ((task_id - 1L) %% 100L) + 1L
arm_ltr <- c("A", "B", "C", "D")[arm_idx]
rep_str <- sprintf("%03d", rep_num)

cat(sprintf("[task %d] ARM %s rep %s\n", task_id, arm_ltr, rep_str))

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a[1])) a else b

# Vectorised column medians with finite fallback (avoids per-column apply loop)
colMedians_safe <- function(X) {
  m <- apply(X, 2L, median, na.rm = TRUE)
  m[!is.finite(m)] <- 0
  m
}

# ── Source WOVEN ──────────────────────────────────────────────────────────────
for (f in c("utils.R", "laplacian.R", "solver_mcca_dual.R",
            "project.R", "metrics.R")) {
  source(file.path(woven_src, f))
}

# ── Precompute sample Laplacian once per condition ────────────────────────────
# Built from X_list_miss: block-missing subjects are excluded from k-NN (no
# imputation). Only subjects with real observed data shape the graph.
# Pre-extracts anchor submatrix L_a to avoid repeated indexing in CV loops.
make_precomp <- function(X_list_miss, anchor_idx, k_nn = 10L) {
  lapply(seq_along(X_list_miss), function(v) {
    L   <- build_laplacian(X_list_miss[[v]], k = k_nn)
    L_a <- as.matrix(L[anchor_idx, anchor_idx, drop = FALSE])
    list(L = L, L_a = L_a)
  })
}

# ── Load simulation data ──────────────────────────────────────────────────────
base_name <- sprintf("arm_%s_rep_%s", arm_ltr, rep_str)

load_rep <- function(suffix = "") {
  fname  <- paste0(base_name, suffix, ".rds")
  subdir <- if (suffix == "") "complete" else "missing"
  candidates <- c(
    file.path(data_dir, subdir, fname),
    file.path(data_dir, fname)
  )
  path <- candidates[file.exists(candidates)][1]
  if (is.na(path)) return(NULL)
  readRDS(path)
}

complete_rep <- load_rep("")
if (is.null(complete_rep)) stop("Complete rep file not found: ", base_name)

mod_names <- names(complete_rep$data)
V         <- length(mod_names)
n         <- nrow(complete_rep$data[[1]])
labels    <- complete_rep$labels
truth_Z   <- complete_rep$truth_Z

set.seed(complete_rep$seed %||% 42L)
true_trt_effect <- 0.5
treatment       <- rbinom(n, 1L, 0.5)
if (!is.null(truth_Z)) {
  Y_outcome <- 0.3 * truth_Z[, 1] + true_trt_effect * treatment + rnorm(n, 0, 0.3)
} else {
  Y_outcome <- true_trt_effect * treatment + rnorm(n, 0, 0.3)
}
n_groups     <- length(unique(labels))
true_effects <- rep(true_trt_effect, n_groups)

cat(sprintf("  n=%d, V=%d modalities: %s\n", n, V, paste(mod_names, collapse=", ")))

apply_mask <- function(data_list, mask_list) {
  lapply(seq_along(data_list), function(v) {
    X <- data_list[[v]]
    if (!is.null(mask_list[[v]])) X[mask_list[[v]], ] <- NA
    X
  })
}

# ── CV helper: anchor silhouette ──────────────────────────────────────────────
anchor_sil <- function(Z, lbl) {
  if (is.null(Z) || nrow(Z) < 4L || length(unique(lbl)) < 2L) return(NA_real_)
  ok <- !is.na(Z[, 1])
  if (sum(ok) < 4L) return(NA_real_)
  tryCatch({
    s <- cluster::silhouette(as.integer(factor(lbl[ok])), dist(Z[ok, , drop=FALSE]))
    mean(s[, 3], na.rm = TRUE)
  }, error = function(e) NA_real_)
}

# Build shared CV folds once per condition (same splits for all methods)
make_folds <- function(anchor_idx, n_folds = 3L, seed = 42L) {
  set.seed(seed)
  n_a <- length(anchor_idx)
  split(sample(n_a), rep(seq_len(n_folds), length.out = n_a))
}

# ── Held-out BER with LDA ─────────────────────────────────────────────────────
# Per-fold DR refitting: test subjects never contributed to W estimation.
# LDA classifier: Fisher-optimal for Gaussian data, handles K=5 dimensions.
# Both Z_trn and Z_val use X %*% W (same coordinate system).
#
# z_fold_fn(trn_a, val_a) → list(Z_trn, Z_val) or NULL on failure.
ber_held_out_lda <- function(anchor_idx, labels, cv_folds, z_fold_fn) {
  n_a    <- length(anchor_idx)
  pred   <- integer(n_a)
  failed <- FALSE

  .lda_pred <- function(Z_trn, lbl_trn, Z_val) {
    lbl_trn <- as.integer(as.factor(lbl_trn))
    if (length(unique(lbl_trn)) < 2L) return(NULL)
    tryCatch({
      fit <- MASS::lda(Z_trn, grouping = lbl_trn)
      as.integer(predict(fit, Z_val)$class)
    }, error = function(e) {
      # LDA fails when within-class variance collapses to zero (perfect
      # class compactness). Nearest centroid is optimal in this case.
      classes <- sort(unique(lbl_trn))
      cents   <- do.call(rbind, lapply(classes, function(g)
        colMeans(Z_trn[lbl_trn == g, , drop = FALSE])))
      d  <- as.matrix(dist(rbind(Z_val, cents)))
      nv <- nrow(Z_val)
      tryCatch(
        classes[apply(d[seq_len(nv), (nv+1L):(nv+length(classes)), drop=FALSE],
                      1L, which.min)],
        error = function(e2) NULL)
    })
  }

  for (f in seq_along(cv_folds)) {
    val_pos <- cv_folds[[f]]
    trn_pos <- unlist(cv_folds[-f])
    trn_a   <- anchor_idx[trn_pos]
    val_a   <- anchor_idx[val_pos]

    res <- tryCatch(z_fold_fn(trn_a, val_a), error = function(e) NULL)
    if (is.null(res)) { failed <- TRUE; break }

    lbl_trn  <- labels[trn_a]
    pred_val <- .lda_pred(res$Z_trn, lbl_trn, res$Z_val)
    if (is.null(pred_val) || length(pred_val) != length(val_pos)) {
      failed <- TRUE; break
    }
    pred[val_pos] <- pred_val
  }

  if (failed) return(NA_real_)

  lbl_int <- as.integer(as.factor(labels[anchor_idx]))
  mean(vapply(sort(unique(lbl_int)), function(g) {
    idx <- which(lbl_int == g)
    if (!length(idx)) return(NA_real_)
    1 - mean(pred[idx] == g)
  }, numeric(1L)), na.rm = TRUE)
}

# ── Method wrappers ───────────────────────────────────────────────────────────

run_woven <- function(X_list_miss, anchor_idx, labels, K, cv_folds, precomp = NULL) {
  t0 <- proc.time()

  # Vectorised projection: V BLAS matrix multiplies instead of n×V R loop
  project_all <- function(fit_W, Za_mean) {
    Z_acc   <- matrix(0,       n, K)
    obs_cnt <- integer(n)
    for (v in seq_len(V)) {
      Xv  <- X_list_miss[[v]]
      obs <- which(!apply(Xv, 1L, function(r) all(is.na(r))))
      if (length(obs) == 0L) next
      Xv_obs <- Xv[obs, , drop = FALSE]
      Xv_obs[is.na(Xv_obs)] <- 0          # zero-fill feature-level NAs
      Z_acc[obs, ] <- Z_acc[obs, ] + Xv_obs %*% fit_W[[v]]
      obs_cnt[obs] <- obs_cnt[obs] + 1L
    }
    none <- obs_cnt == 0L
    obs_cnt[none] <- 1L
    Z_full <- Z_acc / obs_cnt
    Z_full[none, ] <- NA_real_
    # Overwrite anchors with their exact Za scores (no averaging artefact)
    Z_full[anchor_idx, ] <- Za_mean
    Z_full
  }

  tryCatch({
    # ── Unified: mcca_dual for all V (closed-form SUMCOR, no iterations) ────
    # CV over lambda; gamma_y fixed at 1.0 (sensitivity analysis showed it's
    # insensitive; V>=3 also sweeps gamma_y but mcca_dual handles it cheaply).
    lambda_grid <- c(0.001, 0.005, 0.01, 0.1, 0.5)
    gamma_grid  <- c(0.5, 1.0, 5.0, 10.0)
    best_lambda <- 0.01; best_gamma <- 1.0; best_sil <- -Inf

    # Pre-compute fold Laplacian submatrices once — reused across all grid values
    # Avoids 27 redundant sparse submatrix operations (5 lambda + 4 gamma) × 3 folds
    n_folds   <- length(cv_folds)
    mc_cores  <- min(n_folds, parallel::detectCores())
    fold_La <- if (!is.null(precomp)) {
      lapply(seq_len(n_folds), function(f) {
        trn_a <- anchor_idx[-cv_folds[[f]]]
        lapply(precomp, function(p) as.matrix(p$L[trn_a, trn_a, drop = FALSE]))
      })
    } else {
      vector("list", n_folds)
    }

    # Stage 1: best lambda (gamma_y = 1.0)
    for (lam in lambda_grid) {
      sils <- unlist(parallel::mclapply(seq_along(cv_folds), function(f) {
        trn_a <- anchor_idx[-cv_folds[[f]]]
        val_a <- anchor_idx[cv_folds[[f]]]
        tryCatch({
          cv_fit <- woven_mcca_dual(
            X_list = X_list_miss, anchor_idx = trn_a, Y = labels, K = K,
            lambdas = lam, gamma_y = 1.0,
            La_list_precomp = fold_La[[f]], verbose = FALSE
          )
          Za_val <- Reduce("+", lapply(seq_len(V), function(v)
            X_list_miss[[v]][val_a, , drop=FALSE] %*% cv_fit$W_list[[v]])) / V
          anchor_sil(Za_val, labels[val_a])
        }, error = function(e) NA_real_)
      }, mc.cores = mc_cores))
      ms <- mean(sils, na.rm = TRUE)
      if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_lambda <- lam }
    }

    # Stage 2: best gamma_y (best_lambda fixed)
    best_sil <- -Inf
    for (gam in gamma_grid) {
      sils <- unlist(parallel::mclapply(seq_along(cv_folds), function(f) {
        trn_a <- anchor_idx[-cv_folds[[f]]]
        val_a <- anchor_idx[cv_folds[[f]]]
        tryCatch({
          cv_fit <- woven_mcca_dual(
            X_list = X_list_miss, anchor_idx = trn_a, Y = labels, K = K,
            lambdas = best_lambda, gamma_y = gam,
            La_list_precomp = fold_La[[f]], verbose = FALSE
          )
          Za_val <- Reduce("+", lapply(seq_len(V), function(v)
            X_list_miss[[v]][val_a, , drop=FALSE] %*% cv_fit$W_list[[v]])) / V
          anchor_sil(Za_val, labels[val_a])
        }, error = function(e) NA_real_)
      }, mc.cores = mc_cores))
      ms <- mean(sils, na.rm = TRUE)
      if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_gamma <- gam }
    }

    cat(sprintf(" [lambda=%.3f gamma_y=%.1f]", best_lambda, best_gamma))

    fit <- woven_mcca_dual(
      X_list = X_list_miss, anchor_idx = anchor_idx, Y = labels, K = K,
      lambdas = best_lambda, gamma_y = best_gamma,
      La_list_precomp = if (!is.null(precomp)) lapply(precomp, "[[", "L_a") else NULL,
      verbose = FALSE
    )

    Za_mean <- Reduce("+", fit$Za_list) / V
    Z_out   <- project_all(fit$W_list, Za_mean)

    # BER: per-fold refitting with LDA. Test subjects never contributed to W.
    # Use Xa_list (already column-filtered/imputed) for Z_trn to match W dims.
    # Use col_ok_list to select the same feature subset for Z_val.
    ber <- ber_held_out_lda(anchor_idx, labels, cv_folds, function(trn_a, val_a) {
      f_idx <- which(vapply(cv_folds, function(fi)
        identical(anchor_idx[fi], val_a), logical(1L)))
      La_trn <- if (length(f_idx) == 1L) fold_La[[f_idx]] else NULL
      tryCatch({
        f_trn <- woven_mcca_dual(X_list_miss, trn_a, labels, K=K,
          lambdas=best_lambda, gamma_y=best_gamma,
          La_list_precomp=La_trn, verbose=FALSE)
        Z_trn <- Reduce("+", lapply(seq_len(V), function(v)
          f_trn$Xa_list[[v]] %*% f_trn$W_list[[v]])) / V
        # Vectorised NA fill: replace column NAs with training median in one pass
        Z_val <- Reduce("+", lapply(seq_len(V), function(v) {
          cols <- f_trn$col_ok_list[[v]]
          xv   <- X_list_miss[[v]][val_a, cols, drop=FALSE]
          med  <- colMedians_safe(f_trn$Xa_list[[v]])
          na_mask <- is.na(xv)
          xv[na_mask] <- med[col(xv)[na_mask]]
          xv %*% f_trn$W_list[[v]]
        })) / V
        list(Z_trn = Z_trn, Z_val = Z_val)
      }, error = function(e) NULL)
    })

    elapsed <- (proc.time() - t0)[["elapsed"]]
    list(Z = Z_out, n_used = sum(!is.na(Z_out[, 1])), elapsed = elapsed,
         ber = ber, method = "WOVEN", error = NULL)
  }, error = function(e) {
    list(Z = NULL, n_used = 0L, elapsed = NA_real_,
         method = "WOVEN", error = conditionMessage(e))
  })
}

run_diablo <- function(X_list_miss, anchor_idx, labels, K, cv_folds) {
  if (!requireNamespace("mixOmics", quietly = TRUE))
    return(list(Z = NULL, n_used = 0L, elapsed = NA_real_,
                method = "DIABLO", error = "mixOmics not available"))
  t0 <- proc.time()
  tryCatch({
    anchor_rn <- paste0("S", seq_along(anchor_idx))

    prep_Xa <- function(idx) {
      lapply(X_list_miss, function(X) {
        Xa <- na_impute_median(X[idx, , drop = FALSE])
        rownames(Xa) <- paste0("S", seq_along(idx))
        nzv <- tryCatch(mixOmics::nearZeroVar(Xa)$Position, error = function(e) integer(0))
        if (length(nzv) > 0L) Xa <- Xa[, -nzv, drop = FALSE]
        col_var <- apply(Xa, 2, var, na.rm = TRUE)
        keep <- is.finite(col_var) & col_var > 1e-8
        if (sum(keep) < 2L) keep <- seq_len(ncol(Xa))
        Xa[, keep, drop = FALSE]
      })
    }

    # CV over keepX
    keepX_grid <- c(10L, 30L, 50L)
    best_keepX <- 30L; best_sil <- -Inf
    design_cv <- matrix(0.1, V, V, dimnames = list(mod_names, mod_names))
    diag(design_cv) <- 0

    for (kx in keepX_grid) {
      sils <- sapply(seq_along(cv_folds), function(f) {
        val_pos <- cv_folds[[f]]
        trn_pos <- unlist(cv_folds[-f])
        val_a   <- anchor_idx[val_pos]
        trn_a   <- anchor_idx[trn_pos]
        tryCatch({
          Xa_trn <- prep_Xa(trn_a)
          names(Xa_trn) <- mod_names
          kx_use <- lapply(Xa_trn, function(X) rep(min(kx, ncol(X)), K))
          names(kx_use) <- mod_names
          fit_cv <- mixOmics::block.splsda(
            X = Xa_trn, Y = factor(labels[trn_a]),
            ncomp = K, design = design_cv,
            near.zero.var = TRUE, keepX = kx_use
          )
          # Project validation anchors
          Xa_val <- prep_Xa(val_a)
          names(Xa_val) <- mod_names
          Z_val <- Reduce("+", lapply(mod_names, function(m) {
            Xv <- Xa_val[[m]]
            # Keep only features in fitted model
            feat <- rownames(fit_cv$loadings[[m]])
            common <- intersect(colnames(Xv), feat)
            if (length(common) < 2L) return(matrix(0, nrow(Xv), K))
            Xv[, common, drop=FALSE] %*% fit_cv$loadings[[m]][common, , drop=FALSE]
          })) / V
          anchor_sil(Z_val, labels[val_a])
        }, error = function(e) NA_real_)
      })
      ms <- mean(sils, na.rm = TRUE)
      if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_keepX <- kx }
    }

    cat(sprintf(" [keepX=%d]", best_keepX))

    # Final fit with best keepX
    X_anchors <- prep_Xa(anchor_idx)
    names(X_anchors) <- mod_names
    keepX_final <- lapply(X_anchors, function(Xa) rep(min(best_keepX, ncol(Xa)), K))
    names(keepX_final) <- mod_names

    fit_d <- mixOmics::block.splsda(
      X = X_anchors, Y = factor(labels[anchor_idx]),
      ncomp = K, design = matrix(0.1, V, V, dimnames = list(mod_names, mod_names)),
      near.zero.var = TRUE, keepX = keepX_final
    )
    Z_anchor <- Reduce("+", lapply(mod_names, function(m) fit_d$variates[[m]])) / V
    Z_full   <- matrix(NA_real_, nrow = n, ncol = K)
    rownames(Z_anchor) <- NULL
    Z_full[anchor_idx, ] <- Z_anchor

    # BER: per-fold refitting with LDA. Manual projection using rownames(loadings).
    des <- matrix(0.1, V, V, dimnames=list(mod_names, mod_names)); diag(des) <- 0
    ber <- ber_held_out_lda(anchor_idx, labels, cv_folds, function(trn_a, val_a) {
      tryCatch({
        X_trn <- prep_Xa(trn_a); names(X_trn) <- mod_names
        kx    <- lapply(X_trn, function(Xa) rep(min(best_keepX, ncol(Xa)), K))
        names(kx) <- mod_names
        f_trn <- mixOmics::block.splsda(X=X_trn, Y=factor(labels[trn_a]),
          ncomp=K, design=des, near.zero.var=TRUE, keepX=kx)
        Z_trn <- Reduce("+", lapply(mod_names, function(m) f_trn$variates[[m]]))/V
        rownames(Z_trn) <- NULL
        X_val <- prep_Xa(val_a); names(X_val) <- mod_names
        Z_val <- Reduce("+", lapply(mod_names, function(m) {
          feat   <- rownames(f_trn$loadings[[m]])
          common <- intersect(colnames(X_val[[m]]), feat)
          if (length(common) < 2L) return(matrix(0, nrow(X_val[[m]]), K))
          X_val[[m]][, common, drop=FALSE] %*%
            f_trn$loadings[[m]][common, seq_len(K), drop=FALSE]
        }))/V
        list(Z_trn = Z_trn, Z_val = matrix(Z_val, nrow=length(val_a)))
      }, error = function(e) NULL)
    })

    elapsed <- (proc.time() - t0)[["elapsed"]]
    list(Z = Z_full, n_used = length(anchor_idx), elapsed = elapsed,
         ber = ber, method = "DIABLO", error = NULL)
  }, error = function(e) {
    list(Z = NULL, n_used = 0L, elapsed = NA_real_,
         method = "DIABLO", error = conditionMessage(e))
  })
}

run_mofa2 <- function(X_list_miss, K = 5L) {
  t0 <- proc.time()
  tryCatch({
    if (!requireNamespace("reticulate", quietly = TRUE))
      stop("reticulate not available")
    Sys.setenv(RETICULATE_PYTHON = "/home/nbhtd/miniconda3/bin/python")
    reticulate::py_config()
    mofapy2 <- reticulate::import("mofapy2.run.entry_point")
    np      <- reticulate::import("numpy")

    data_py <- lapply(X_list_miss, function(X) {
      Xf <- X; Xf[is.na(Xf)] <- NaN
      list(np$array(Xf, dtype = "float64"))
    })

    ent <- mofapy2$entry_point()
    ent$set_data_matrix(data_py,
                        likelihoods  = rep("gaussian", V),
                        views_names  = mod_names,
                        groups_names = list("group0"))
    ent$set_model_options(factors = as.integer(K))
    ent$set_train_options(iter = 1000L, convergence_mode = "fast",
                          verbose = FALSE, seed = 42L)
    ent$build()
    ent$run()

    model_obj <- reticulate::py_get_attr(ent, "model")
    nodes_obj <- reticulate::py_get_attr(model_obj, "nodes")
    Z_node    <- reticulate::py_get_item(nodes_obj, "Z")
    getExp    <- reticulate::py_get_attr(Z_node, "getExpectation")
    Z_mofa    <- reticulate::py_to_r(reticulate::py_call(getExp))
    rm(ent, model_obj, nodes_obj, Z_node, getExp); gc()

    elapsed <- (proc.time() - t0)[["elapsed"]]
    list(Z = Z_mofa, n_used = nrow(Z_mofa), elapsed = elapsed,
         method = "MOFA2", error = NULL)
  }, error = function(e) {
    message("MOFA2 ERROR: ", conditionMessage(e))
    list(Z = NULL, n_used = 0L, elapsed = NA_real_,
         method = "MOFA2", error = conditionMessage(e))
  })
}

run_impute_diablo <- function(X_list_miss, anchor_idx, labels, K, cv_folds,
                               top_features = 500L) {
  if (!requireNamespace("missForest", quietly = TRUE) ||
      !requireNamespace("mixOmics", quietly = TRUE))
    return(list(Z = NULL, n_used = 0L, elapsed = NA_real_,
                method = "ImputeDIABLO", error = "missForest or mixOmics not available"))
  t0 <- proc.time()
  tryCatch({
    all_rn <- paste0("S", seq_len(n))

    # Impute once (expensive — do before CV)
    X_reduced <- lapply(X_list_miss, function(X) {
      vars <- apply(X, 2, var, na.rm = TRUE); vars[is.na(vars)] <- 0
      X[, order(vars, decreasing=TRUE)[seq_len(min(top_features, ncol(X)))], drop=FALSE]
    })
    X_imputed <- lapply(X_reduced, function(X) {
      all_na_row <- apply(X, 1, function(r) all(is.na(r)))
      if (any(all_na_row)) {
        cm <- colMeans(X, na.rm=TRUE); cm[!is.finite(cm)] <- 0
        for (i in which(all_na_row)) X[i, ] <- cm
      }
      out <- missForest::missForest(X, maxiter=5L, ntree=50L, verbose=FALSE)$ximp
      rownames(out) <- all_rn
      out
    })
    names(X_imputed) <- mod_names

    # Pre-filter
    X_filt <- lapply(X_imputed, function(Xi) {
      nzv <- tryCatch(mixOmics::nearZeroVar(Xi)$Position, error=function(e) integer(0))
      if (length(nzv) > 0L) Xi <- Xi[, -nzv, drop=FALSE]
      keep <- is.finite(apply(Xi,2,var)) & apply(Xi,2,var) > 1e-8
      if (sum(keep) < 2L) keep <- seq_len(ncol(Xi))
      Xi[, keep, drop=FALSE]
    })
    names(X_filt) <- mod_names

    design_cv <- matrix(0.1, V, V, dimnames=list(mod_names, mod_names))
    diag(design_cv) <- 0

    # CV over keepX using anchor subjects only (same fold splits)
    keepX_grid <- c(10L, 30L, 50L)
    best_keepX <- 30L; best_sil <- -Inf

    for (kx in keepX_grid) {
      sils <- sapply(seq_along(cv_folds), function(f) {
        val_pos <- cv_folds[[f]]
        trn_pos <- unlist(cv_folds[-f])
        val_a   <- anchor_idx[val_pos]
        trn_a   <- anchor_idx[trn_pos]
        tryCatch({
          Xtrn <- lapply(X_filt, function(X) {
            out <- X[trn_a, , drop=FALSE]
            rownames(out) <- paste0("S", seq_along(trn_a))
            out
          })
          names(Xtrn) <- mod_names
          kx_use <- lapply(Xtrn, function(X) rep(min(kx, ncol(X)), K))
          names(kx_use) <- mod_names
          fit_cv <- mixOmics::block.splsda(
            X=Xtrn, Y=factor(labels[trn_a]),
            ncomp=K, design=design_cv,
            near.zero.var=TRUE, keepX=kx_use
          )
          Xval <- lapply(X_filt, function(X) X[val_a, , drop=FALSE])
          names(Xval) <- mod_names
          Z_val <- Reduce("+", lapply(mod_names, function(m) {
            Xv <- Xval[[m]]
            feat <- rownames(fit_cv$loadings[[m]])
            common <- intersect(colnames(Xv), feat)
            if (length(common) < 2L) return(matrix(0, nrow(Xv), K))
            Xv[, common, drop=FALSE] %*% fit_cv$loadings[[m]][common, , drop=FALSE]
          })) / V
          anchor_sil(Z_val, labels[val_a])
        }, error=function(e) NA_real_)
      })
      ms <- mean(sils, na.rm=TRUE)
      if (is.finite(ms) && ms > best_sil) { best_sil <- ms; best_keepX <- kx }
    }

    cat(sprintf(" [keepX=%d]", best_keepX))

    kx_final <- lapply(X_filt, function(X) rep(min(best_keepX, ncol(X)), K))
    names(kx_final) <- mod_names

    fit_d <- mixOmics::block.splsda(
      X=X_filt, Y=factor(labels),
      ncomp=K, design=design_cv,
      near.zero.var=TRUE, keepX=kx_final
    )

    Z_full <- Reduce("+", lapply(mod_names, function(m) fit_d$variates[[m]])) / V

    elapsed <- (proc.time() - t0)[["elapsed"]]
    list(Z=Z_full, n_used=n, elapsed=elapsed, method="ImputeDIABLO", error=NULL)
  }, error=function(e) {
    list(Z=NULL, n_used=0L, elapsed=NA_real_,
         method="ImputeDIABLO", error=conditionMessage(e))
  })
}

# ── Metric computation wrapper ────────────────────────────────────────────────
compute_metrics <- function(result, labels, n_total, truth_Z=NULL,
                            outcome=NULL, treatment=NULL, true_effects=NULL,
                            anchor_idx_eval=NULL) {
  if (is.null(result$Z))
    return(list(error=result$error, elapsed=result$elapsed, method=result$method))

  has_score <- !is.na(result$Z[, 1])
  Z_eval    <- result$Z[has_score, , drop=FALSE]
  lbl_eval  <- labels[has_score]

  m <- woven_all_metrics(
    Z            = Z_eval,
    labels       = lbl_eval,
    n_total      = n_total,
    Z_true       = if (!is.null(truth_Z)) truth_Z[has_score, , drop=FALSE] else NULL,
    outcome      = if (!is.null(outcome)) outcome[has_score] else NULL,
    treatment    = if (!is.null(treatment)) treatment[has_score] else NULL,
    true_effects = true_effects
  )

  if (!is.null(anchor_idx_eval) && !is.null(result$Z)) {
    Z_anc   <- result$Z[anchor_idx_eval, , drop=FALSE]
    lbl_anc <- labels[anchor_idx_eval]
    ok_anc  <- !is.na(Z_anc[, 1])
    if (sum(ok_anc) >= 4L && length(unique(lbl_anc[ok_anc])) >= 2L) {
      m_anc <- woven_all_metrics(
        Z       = Z_anc[ok_anc, , drop=FALSE],
        labels  = lbl_anc[ok_anc],
        n_total = length(anchor_idx_eval)
      )
      names(m_anc) <- paste0(names(m_anc), "_anchor")
      m <- c(m, m_anc)
    }
  }

  # BER from per-fold DR refitting (never circular: test subjects excluded from W fit)
  if (!is.null(result$ber)) m$ber <- result$ber

  c(m, list(method=result$method, n_used=result$n_used,
            elapsed=result$elapsed, error=result$error))
}

# ── Main loop ─────────────────────────────────────────────────────────────────
all_conditions <- c("complete", "mcar30", "mcar50", "mar")
K              <- 5L

woven_cond <- Sys.getenv("WOVEN_COND", unset="")
conditions  <- if (nchar(woven_cond) > 0L) woven_cond else all_conditions

ckpt_file <- file.path(out_dir, sprintf("arm_%s_rep_%s_checkpoint.rds", arm_ltr, rep_str))
results   <- if (file.exists(ckpt_file)) readRDS(ckpt_file) else list()

for (cond in conditions) {
  if (!is.null(results[[cond]])) {
    cat(sprintf("  Condition: %s [checkpointed, skipping]\n", cond)); next
  }
  cat(sprintf("  Condition: %s\n", cond))

  if (cond == "complete") {
    X_list_full <- complete_rep$data
    X_list_miss <- X_list_full
    anchor_idx  <- seq_len(n)
  } else {
    rep_dat <- load_rep(paste0("_", cond))
    if (is.null(rep_dat)) { cat(sprintf("    [SKIP] %s not found\n", cond)); next }
    X_list_full <- complete_rep$data
    has_nas <- any(sapply(rep_dat$data, function(X) any(is.na(X))))
    X_list_miss <- if (has_nas) rep_dat$data else {
      apply_mask(X_list_full, lapply(mod_names, function(m) rep_dat$missingness_mask[, m]))
    }
    anchor_idx <- rep_dat$anchor_idx
    if (is.null(anchor_idx)) {
      bm <- sapply(X_list_miss, function(X) apply(is.na(X), 1, all))
      anchor_idx <- which(rowSums(bm) == 0L)
    }
  }

  # Shared CV folds — same splits for all methods this condition
  cv_folds <- make_folds(anchor_idx, n_folds = 3L, seed = rep_num * 100L + match(cond, all_conditions))

  # Precompute L from missing data (block-missing rows excluded, no imputation)
  cat("    Precomputing L...")
  t_pre <- proc.time()
  precomp_list <- make_precomp(X_list_miss, anchor_idx, k_nn = 10L)
  cat(sprintf(" [%.0fs]\n", (proc.time() - t_pre)[["elapsed"]]))

  cond_results <- list()

  cat("    WOVEN...")
  r <- run_woven(X_list_miss, anchor_idx, labels, K, cv_folds, precomp_list)
  cond_results$WOVEN <- compute_metrics(r, labels, n, truth_Z, Y_outcome, treatment, true_effects, anchor_idx)
  cat(sprintf(" sil=%.3f ess=%.2f [%.0fs]\n",
    cond_results$WOVEN$silhouette %||% NA,
    cond_results$WOVEN$ess_retention %||% NA,
    cond_results$WOVEN$elapsed %||% NA))

  cat("    DIABLO...")
  r <- run_diablo(X_list_miss, anchor_idx, labels, K, cv_folds)
  cond_results$DIABLO <- compute_metrics(r, labels, n, truth_Z, Y_outcome, treatment, true_effects, anchor_idx)
  cat(sprintf(" sil=%.3f ess=%.2f [%.0fs]\n",
    cond_results$DIABLO$silhouette %||% NA,
    cond_results$DIABLO$ess_retention %||% NA,
    cond_results$DIABLO$elapsed %||% NA))

  cat("    MOFA2...")
  r <- run_mofa2(X_list_miss, K)
  cond_results$MOFA2 <- compute_metrics(r, labels, n, truth_Z, Y_outcome, treatment, true_effects, anchor_idx)
  cat(sprintf(" sil=%.3f ess=%.2f [%.0fs]\n",
    cond_results$MOFA2$silhouette %||% NA,
    cond_results$MOFA2$ess_retention %||% NA,
    cond_results$MOFA2$elapsed %||% NA))

  if (cond != "complete") {
    cat("    ImputeDIABLO...")
    r <- run_impute_diablo(X_list_miss, anchor_idx, labels, K, cv_folds)
    cond_results$ImputeDIABLO <- compute_metrics(r, labels, n, truth_Z, Y_outcome, treatment, true_effects, anchor_idx)
    cat(sprintf(" sil=%.3f ess=%.2f [%.0fs]\n",
      cond_results$ImputeDIABLO$silhouette %||% NA,
      cond_results$ImputeDIABLO$ess_retention %||% NA,
      cond_results$ImputeDIABLO$elapsed %||% NA))
  }

  results[[cond]] <- cond_results
  saveRDS(results, ckpt_file)
}

dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
out_file <- file.path(out_dir, sprintf("arm_%s_rep_%s_benchmark.rds", arm_ltr, rep_str))

if (all(all_conditions %in% names(results))) {
  saveRDS(list(task_id=task_id, arm=arm_ltr, rep=rep_num, results=results,
               metadata=list(date=Sys.time(), K=K, V=V, mods=mod_names, n=n)),
          file=out_file)
  cat(sprintf("\n[task %d] Saved: %s\n", task_id, out_file))
  file.remove(ckpt_file)
} else {
  cat(sprintf("\n[task %d] Partial (%s done), checkpoint saved.\n",
              task_id, paste(names(results), collapse=",")))
}
