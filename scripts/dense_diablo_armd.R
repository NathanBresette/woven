#!/usr/bin/env Rscript
# dense_diablo_armd.R -- Dense DIABLO (block.plsda, no sparsity) on ARM D
#
# Purpose: Disentangle whether WOVEN's BER advantage on ARM D complete data
# (WOVEN 0.131 vs sparse DIABLO 0.763) reflects the dense-vs-sparse W choice
# or WOVEN's missing-data handling. Runs block.plsda (all features, no keepX)
# so the only difference from WOVEN is the alignment objective, not sparsity.
#
# Conditions: complete, mcar50 (the two conditions where sparse DIABLO fails on ARM D)
# Reps: 100 (same as main ARM D benchmark)
# Output: arm_D_rep_XXX_dense_diablo.rds per rep
#
# Usage (SLURM array, task_id 1-100):
#   Rscript dense_diablo_armd.R <task_id> <data_dir> <out_dir> <grama_src>

suppressPackageStartupMessages({
  library(Matrix)
  library(RANN)
  library(RSpectra)
  library(cluster)
})

args      <- commandArgs(trailingOnly = TRUE)
task_id   <- as.integer(args[1])
data_dir  <- args[2]
out_dir   <- args[3]
grama_src <- args[4]

for (f in c("utils.R", "laplacian.R", "solver_mcca_dual.R", "project.R", "metrics.R"))
  source(file.path(grama_src, f))

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a[1])) a else b

rep_num <- task_id
rep_str <- sprintf("%03d", rep_num)
arm_ltr <- "D"

cat(sprintf("[task %d] Dense DIABLO ARM D rep %s\n", task_id, rep_str))

if (!requireNamespace("mixOmics", quietly = TRUE))
  stop("mixOmics not available")

# ── Load data ─────────────────────────────────────────────────────────────────
load_rep <- function(suffix = "") {
  fname  <- sprintf("arm_%s_rep_%s%s.rds", arm_ltr, rep_str, suffix)
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
if (is.null(complete_rep)) stop("Complete ARM D rep not found: ", rep_str)

mod_names <- names(complete_rep$data)
V         <- length(mod_names)
n         <- nrow(complete_rep$data[[1]])
labels    <- complete_rep$labels

set.seed(42L + rep_num)

# ── BER helpers ───────────────────────────────────────────────────────────────
make_folds <- function(anchor_idx, n_folds = 3L, seed = 42L) {
  set.seed(seed)
  n_a <- length(anchor_idx)
  split(sample(n_a), rep(seq_len(n_folds), length.out = n_a))
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

ber_lda <- function(anchor_idx, labels, cv_folds, z_fold_fn) {
  n_a  <- length(anchor_idx)
  pred <- integer(n_a)

  .lda_pred <- function(Z_trn, lbl_trn, Z_val) {
    lbl_trn <- as.integer(as.factor(lbl_trn))
    if (length(unique(lbl_trn)) < 2L) return(NULL)
    tryCatch({
      fit <- MASS::lda(Z_trn, grouping = lbl_trn)
      as.integer(predict(fit, Z_val)$class)
    }, error = function(e) {
      classes <- sort(unique(lbl_trn))
      cents   <- do.call(rbind, lapply(classes, function(g)
        colMeans(Z_trn[lbl_trn == g, , drop=FALSE])))
      d  <- as.matrix(dist(rbind(Z_val, cents)))
      nv <- nrow(Z_val)
      tryCatch(
        classes[apply(d[seq_len(nv), (nv+1L):(nv+length(classes)), drop=FALSE], 1L, which.min)],
        error = function(e2) NULL)
    })
  }

  for (f in seq_along(cv_folds)) {
    val_pos <- cv_folds[[f]]
    trn_pos <- unlist(cv_folds[-f])
    res <- tryCatch(z_fold_fn(anchor_idx[trn_pos], anchor_idx[val_pos]),
                   error = function(e) NULL)
    if (is.null(res)) return(NA_real_)
    pv <- .lda_pred(res$Z_trn, labels[anchor_idx[trn_pos]], res$Z_val)
    if (is.null(pv) || length(pv) != length(val_pos)) return(NA_real_)
    pred[val_pos] <- pv
  }

  lbl_int <- as.integer(as.factor(labels[anchor_idx]))
  mean(vapply(sort(unique(lbl_int)), function(g) {
    idx <- which(lbl_int == g)
    1 - mean(pred[idx] == g)
  }, numeric(1L)), na.rm = TRUE)
}

# ── Dense DIABLO runner ───────────────────────────────────────────────────────
run_dense_diablo <- function(X_list, anchor_idx, labels, K = 5L) {
  t0 <- proc.time()

  prep_Xa <- function(idx) {
    Xa <- lapply(X_list, function(X) {
      out <- na_impute_median(X[idx, , drop=FALSE])
      rownames(out) <- paste0("S", seq_along(idx))
      nzv <- tryCatch(mixOmics::nearZeroVar(out)$Position, error=function(e) integer(0))
      if (length(nzv) > 0L) out <- out[, -nzv, drop=FALSE]
      keep <- apply(out, 2, var, na.rm=TRUE)
      keep <- is.finite(keep) & keep > 1e-8
      if (sum(keep) < 2L) keep <- seq_len(ncol(out))
      out[, keep, drop=FALSE]
    })
    names(Xa) <- mod_names
    Xa
  }

  design <- matrix(0.1, V, V, dimnames=list(mod_names, mod_names))
  diag(design) <- 0

  cv_folds <- make_folds(anchor_idx, n_folds = 3L, seed = rep_num * 100L)

  tryCatch({
    # Final fit: block.plsda (dense — no keepX, uses all features)
    Xa  <- prep_Xa(anchor_idx)
    fit <- mixOmics::block.plsda(
      X = Xa, Y = factor(labels[anchor_idx]),
      ncomp = K, design = design, near.zero.var = TRUE
    )

    Z_anchor <- Reduce("+", lapply(mod_names, function(m) fit$variates[[m]])) / V
    Z_full   <- matrix(NA_real_, nrow=n, ncol=K)
    rownames(Z_anchor) <- NULL
    Z_full[anchor_idx, ] <- Z_anchor

    # Silhouette
    sil_all <- anchor_sil(Z_full[anchor_idx, , drop=FALSE], labels[anchor_idx])

    # BER: per-fold refitting
    ber <- ber_lda(anchor_idx, labels, cv_folds, function(trn_a, val_a) {
      tryCatch({
        Xtrn <- prep_Xa(trn_a)
        f_cv <- mixOmics::block.plsda(X=Xtrn, Y=factor(labels[trn_a]),
          ncomp=K, design=design, near.zero.var=TRUE)
        Z_trn <- Reduce("+", lapply(mod_names, function(m) f_cv$variates[[m]]))/V
        rownames(Z_trn) <- NULL
        Xval <- prep_Xa(val_a)
        Z_val <- Reduce("+", lapply(mod_names, function(m) {
          feat   <- rownames(f_cv$loadings[[m]])
          common <- intersect(colnames(Xval[[m]]), feat)
          if (length(common) < 2L) return(matrix(0, length(val_a), K))
          Xval[[m]][, common, drop=FALSE] %*%
            f_cv$loadings[[m]][common, seq_len(K), drop=FALSE]
        }))/V
        list(Z_trn=Z_trn, Z_val=matrix(Z_val, nrow=length(val_a)))
      }, error=function(e) NULL)
    })

    # NMI
    nmi <- tryCatch(woven_nmi(Z_full[anchor_idx,,drop=FALSE], labels[anchor_idx]),
                   error=function(e) NA_real_)

    elapsed <- (proc.time() - t0)[["elapsed"]]
    cat(sprintf("   sil=%.3f ber=%.3f nmi=%.3f [%.0fs]\n", sil_all, ber %||% NA, nmi, elapsed))

    list(sil=sil_all, ber=ber, nmi=nmi, n_used=length(anchor_idx),
         ess=length(anchor_idx)/n, elapsed=elapsed, error=NA_character_)
  }, error = function(e) {
    cat(sprintf("   ERROR: %s\n", conditionMessage(e)))
    list(sil=NA, ber=NA, nmi=NA, n_used=0L, ess=0,
         elapsed=(proc.time()-t0)[["elapsed"]], error=conditionMessage(e))
  })
}

# ── Run per condition ─────────────────────────────────────────────────────────
conditions <- c("complete", "mcar50")
results    <- list()

for (cond in conditions) {
  cat(sprintf("  Condition: %s\n", cond))

  if (cond == "complete") {
    X_list     <- complete_rep$data
    anchor_idx <- seq_len(n)
  } else {
    miss_rep <- load_rep(paste0("_", cond))
    if (is.null(miss_rep)) { cat("    [SKIP] not found\n"); next }
    has_nas <- any(sapply(miss_rep$data, function(X) any(is.na(X))))
    X_list <- if (has_nas) miss_rep$data else {
      lapply(seq_len(V), function(v) {
        X <- complete_rep$data[[v]]
        mask <- miss_rep$missingness_mask[, mod_names[v]]
        if (!is.null(mask)) X[mask, ] <- NA
        X
      })
    }
    anchor_idx <- miss_rep$anchor_idx
    if (is.null(anchor_idx)) {
      bm <- sapply(X_list, function(X) apply(is.na(X), 1, all))
      anchor_idx <- which(rowSums(bm) == 0L)
    }
  }

  results[[cond]] <- run_dense_diablo(X_list, anchor_idx, labels)
}

dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
out_file <- file.path(out_dir, sprintf("arm_D_rep_%s_dense_diablo.rds", rep_str))
saveRDS(list(rep=rep_num, arm="D", results=results), out_file)
cat(sprintf("[task %d] Saved: %s\n", task_id, out_file))
