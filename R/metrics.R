# metrics.R  -- Evaluation metric battery for WOVEN benchmarking
#
# All functions take a latent matrix Z (n x K) and return a scalar or named vector.
# Designed to work on: WOVEN output, DIABLO variate scores, MOFA+ factor scores.
#
# Metrics:
#   woven_silhouette       -- average silhouette width (cluster geometry)
#   woven_davies_bouldin   -- Davies-Bouldin index (cluster compactness/separation)
#   woven_nmi              -- normalized mutual information (label recovery)
#   woven_rv               -- RV coefficient vs ground-truth factor matrix
#   woven_ess_retention    -- effective sample size retention (N_used / N_total)
#   woven_effect_bias      -- subgroup effect estimate bias (CER-specific metric)
#   woven_nystrom_error    -- leave-anchor-out Nystrm projection error
#   woven_all_metrics      -- compute full battery, returns named list

#  Silhouette 

#' Average silhouette width
#'
#' @param Z numeric matrix n x K (latent scores)
#' @param labels integer or factor of length n (subgroup labels)
#' @return scalar in [-1, 1], higher is better
#' @export
woven_silhouette <- function(Z, labels) {
  labels <- as.integer(as.factor(labels))
  if (length(unique(labels)) < 2L) return(NA_real_)
  d  <- dist(Z)
  sw <- cluster::silhouette(labels, d)
  mean(sw[, "sil_width"])
}

#  Davies-Bouldin 

#' Davies-Bouldin index
#'
#' DB = (1/K) sum_i max_\{j != i\} (s_i + s_j) / d(c_i, c_j)
#' where s_i = mean intra-cluster distance, d(c_i, c_j) = centroid distance.
#'
#' @param Z numeric matrix n x K
#' @param labels integer or factor of length n
#' @return scalar >= 0, lower is better
#' @export
woven_davies_bouldin <- function(Z, labels) {
  labels <- as.integer(as.factor(labels))
  K_cl   <- length(unique(labels))
  if (K_cl < 2L) return(NA_real_)

  centroids <- do.call(rbind, lapply(sort(unique(labels)), function(g) {
    colMeans(Z[labels == g, , drop = FALSE])
  }))

  s <- sapply(sort(unique(labels)), function(g) {
    Zg <- Z[labels == g, , drop = FALSE]
    if (nrow(Zg) < 2L) return(0)
    mean(sqrt(rowSums(sweep(Zg, 2, centroids[g, ])^2)))
  })

  db_vals <- sapply(seq_len(K_cl), function(i) {
    others <- setdiff(seq_len(K_cl), i)
    max(sapply(others, function(j) {
      d_ij <- sqrt(sum((centroids[i, ] - centroids[j, ])^2))
      if (d_ij < 1e-12) return(0)
      (s[i] + s[j]) / d_ij
    }))
  })

  mean(db_vals)
}

#  NMI 

#' Normalized mutual information between cluster assignments and true labels
#'
#' Uses k-means on Z to get cluster assignments, then computes NMI.
#' k-means run 10 times to reduce initialization variance.
#'
#' @param Z numeric matrix n x K
#' @param labels integer or factor of length n (true labels)
#' @param n_cl integer, number of clusters (default = number of unique labels)
#' @param n_start integer, k-means random starts
#' @return scalar in [0, 1], higher is better
#' @export
woven_nmi <- function(Z, labels, n_cl = NULL, n_start = 10L) {
  labels <- as.integer(as.factor(labels))
  if (is.null(n_cl)) n_cl <- length(unique(labels))
  if (n_cl < 2L) return(NA_real_)

  km  <- kmeans(Z, centers = n_cl, nstart = n_start, iter.max = 100L)
  pred <- km$cluster

  # NMI via entropy decomposition (no external package needed)
  .entropy <- function(x) {
    px <- tabulate(x) / length(x)
    px <- px[px > 0]
    -sum(px * log(px))
  }
  .joint_entropy <- function(x, y) {
    n <- length(x)
    tbl <- table(x, y)
    pxy <- tbl / n
    pxy <- pxy[pxy > 0]
    -sum(pxy * log(pxy))
  }

  H_true <- .entropy(labels)
  H_pred <- .entropy(pred)
  H_joint <- .joint_entropy(labels, pred)
  MI <- H_true + H_pred - H_joint

  denom <- (H_true + H_pred) / 2
  if (denom < 1e-12) return(NA_real_)
  MI / denom
}

#  RV coefficient 

#' RV coefficient between latent scores and ground-truth factor matrix
#'
#' RV(X, Y) = trace(X X' Y Y') / sqrt(trace(X X' X X') * trace(Y Y' Y Y'))
#' Measures similarity of two cross-product matrices; 1 = identical subspace.
#'
#' @param Z numeric matrix n x K (inferred latent scores)
#' @param Z_true numeric matrix n x K_true (ground-truth factor scores from SUMO)
#' @return scalar in [0, 1], higher is better
#' @export
woven_rv <- function(Z, Z_true) {
  stopifnot(nrow(Z) == nrow(Z_true))
  # Center columns
  Z      <- scale(Z,      center = TRUE, scale = FALSE)
  Z_true <- scale(Z_true, center = TRUE, scale = FALSE)

  S  <- tcrossprod(Z)        # n x n
  T_ <- tcrossprod(Z_true)   # n x n

  num   <- sum(S * T_)
  denom <- sqrt(sum(S * S) * sum(T_ * T_))
  if (denom < 1e-12) return(NA_real_)
  num / denom
}

#  Effective sample size retention 

#' Effective sample size retention
#'
#' @param n_used integer, number of subjects with a latent score
#' @param n_total integer, total subjects in dataset
#' @return scalar in [0, 1], higher is better (DIABLO structurally caps at overlap fraction)
#' @export
woven_ess_retention <- function(n_used, n_total) {
  stopifnot(n_used >= 0, n_total > 0, n_used <= n_total)
  n_used / n_total
}

#  Subgroup effect estimate bias 

#' CER-specific: subgroup effect estimate bias
#'
#' Fits a linear model of a continuous outcome on a binary treatment indicator,
#' separately within each subgroup defined by `labels`. Compares estimated
#' treatment effect to the known true effect (from simulation ground truth).
#'
#' bias_g = |estimated_g - true_g| / |true_g|   (relative)
#' Returns mean bias across subgroups.
#'
#' @param Z numeric matrix n x K (latent scores; used as covariates)
#' @param outcome numeric vector of length n (simulated continuous outcome)
#' @param treatment integer/logical vector of length n (0/1 treatment indicator)
#' @param labels integer or factor of length n (subgroup labels)
#' @param true_effects named numeric vector, true treatment effect per subgroup level
#' @return scalar >= 0, lower is better
#' @export
woven_effect_bias <- function(Z, outcome, treatment, labels, true_effects) {
  labels    <- as.integer(as.factor(labels))
  treatment <- as.numeric(treatment)
  groups    <- sort(unique(labels))

  biases <- vapply(seq_along(groups), function(gi) {
    g   <- groups[gi]
    idx <- which(labels == g)
    if (length(idx) < 5L) return(NA_real_)

    df  <- data.frame(y = outcome[idx], trt = treatment[idx], Z[idx, , drop = FALSE])
    fit <- tryCatch(lm(y ~ ., data = df), error = function(e) NULL)
    if (is.null(fit)) return(NA_real_)

    est  <- coef(fit)["trt"]
    true <- true_effects[gi]
    if (is.na(true) || abs(true) < 1e-12) return(NA_real_)
    abs(est - true) / abs(true)
  }, numeric(1L))

  mean(biases, na.rm = TRUE)
}

#  Nystrm leave-anchor-out error 

#' Leave-anchor-out Nystrm projection error
#'
#' For each anchor subject, temporarily removes it from the anchor set,
#' projects it via Nystrm using remaining anchors, and computes ||Z_proj - Z_direct||.
#' Quantifies how well the Nystrm extension generalizes.
#'
#' @param fit output of woven_v2() or woven_als()
#' @param X1 n x p1 matrix for modality 1 (unmasked)
#' @param X2 n x p2 matrix for modality 2 (unmasked)
#' @param n_loo integer, number of anchors to hold out (default = all, slow)
#' @param sigma_proj optional bandwidth for Nystrm kernel
#' @return scalar >= 0, lower is better (mean Frobenius error per subject)
#' @export
woven_nystrom_error <- function(fit, X1, X2, n_loo = NULL, sigma_proj = NULL) {
  anchor_idx <- fit$anchor_idx
  n_a        <- length(anchor_idx)
  if (is.null(n_loo)) n_loo <- n_a

  n_loo  <- min(n_loo, n_a)
  loo_set <- sample(seq_len(n_a), n_loo)

  errors <- vapply(loo_set, function(j) {
    held_out   <- anchor_idx[j]
    remain_idx <- anchor_idx[-j]

    # Build a mini-fit with held-out anchor removed
    mini_fit           <- fit
    mini_fit$anchor_idx <- remain_idx
    if (!is.null(fit$Za1)) {
      mini_fit$Za1 <- fit$Za1[-j, , drop = FALSE]
      mini_fit$Za2 <- fit$Za2[-j, , drop = FALSE]
    } else if (!is.null(fit$Za_list)) {
      mini_fit$Za_list <- lapply(fit$Za_list, function(Z) Z[-j, , drop = FALSE])
    }

    # True direct score
    Z_true <- if (!is.null(fit$Za1)) {
      (fit$Za1[j, , drop = FALSE] + fit$Za2[j, , drop = FALSE]) / 2
    } else {
      Reduce("+", lapply(fit$Za_list, function(Z) Z[j, , drop = FALSE])) / fit$V
    }

    # Projected score via Nystrm
    X1_i <- X1[held_out, , drop = FALSE]
    X2_i <- X2[held_out, , drop = FALSE]
    proj  <- tryCatch(
      woven_project(mini_fit, X1_i, X2_i, sigma_proj = sigma_proj),
      error = function(e) NULL
    )
    if (is.null(proj) || any(is.na(proj$Z))) return(NA_real_)

    sqrt(sum((proj$Z - Z_true)^2))
  }, numeric(1L))

  mean(errors, na.rm = TRUE)
}

#  Balanced Error Rate (BER)

#' Balanced Error Rate via k-fold LDA cross-validation
#'
#' BER = (1/C) * sum_c (1 - sensitivity_c)  where C = number of classes.
#' Uses the same LDA classifier for all methods (WOVEN, DIABLO, MOFA2) so the
#' comparison is fair: DIABLO's built-in sparse classifier is NOT used.
#' Falls back to nearest-centroid if MASS::lda is unavailable.
#'
#' Lower is better. 0 = perfect, 1 - 1/C = chance level.
#'
#' @param Z numeric matrix n x K (consensus latent scores)
#' @param labels integer or factor of length n (class labels)
#' @param n_folds integer, CV folds (default 5; use n for LOO)
#' @return scalar in [0, 1], lower is better
#' @export
woven_ber <- function(Z, labels, n_folds = 5L) {
  labels <- as.integer(as.factor(labels))
  n      <- nrow(Z)
  C      <- length(unique(labels))
  if (C < 2L || n < C * 2L) return(NA_real_)
  n_folds <- min(n_folds, min(tabulate(labels)))  # can't have more folds than smallest class

  # Stratified k-fold: each class is represented in every training fold
  set.seed(42L)
  fold_id <- integer(n)
  for (g in sort(unique(labels))) {
    idx_g <- which(labels == g)
    fold_id[idx_g] <- sample(rep(seq_len(n_folds), length.out = length(idx_g)))
  }

  # Nearest-centroid classifier: no package dependency, works for any n/K
  .nc_predict <- function(Z_trn, lbl_trn, Z_val) {
    classes   <- sort(unique(lbl_trn))
    centroids <- do.call(rbind, lapply(classes, function(g)
      colMeans(Z_trn[lbl_trn == g, , drop = FALSE])))
    d <- as.matrix(dist(rbind(Z_val, centroids)))
    n_val <- nrow(Z_val)
    d_to_c <- d[seq_len(n_val), (n_val + 1L):(n_val + length(classes)), drop = FALSE]
    classes[apply(d_to_c, 1L, which.min)]
  }

  pred_all <- integer(n)
  ok <- tryCatch({
    for (f in seq_len(n_folds)) {
      trn <- which(fold_id != f)
      val <- which(fold_id == f)
      # Try LDA first; fall back to nearest centroid if it fails
      pred_val <- tryCatch({
        if (requireNamespace("MASS", quietly = TRUE)) {
          fit_lda <- MASS::lda(Z[trn, , drop = FALSE], grouping = labels[trn])
          as.integer(stats::predict(fit_lda, Z[val, , drop = FALSE])$class)
        } else {
          .nc_predict(Z[trn, , drop = FALSE], labels[trn], Z[val, , drop = FALSE])
        }
      }, error = function(e)
        .nc_predict(Z[trn, , drop = FALSE], labels[trn], Z[val, , drop = FALSE])
      )
      pred_all[val] <- pred_val
    }
    TRUE
  }, error = function(e) FALSE)

  if (!ok) return(NA_real_)

  # BER: mean per-class error rate
  per_class <- vapply(sort(unique(labels)), function(g) {
    idx <- which(labels == g)
    if (length(idx) == 0L) return(NA_real_)
    1 - mean(pred_all[idx] == g)
  }, numeric(1L))

  mean(per_class, na.rm = TRUE)
}

#  Full metric battery

#' Compute full WOVEN evaluation metric battery
#'
#' @param Z numeric matrix n x K (consensus latent scores)
#' @param labels integer or factor of length n
#' @param n_total integer, total subjects before any filtering (for ESS)
#' @param Z_true optional numeric matrix n x K_true (ground-truth factors for RV)
#' @param outcome optional numeric vector (for effect bias)
#' @param treatment optional 0/1 vector (for effect bias)
#' @param true_effects optional named numeric (for effect bias)
#' @param fit optional WOVEN fit object (for Nystrm LOO error)
#' @param X1 optional matrix (for Nystrm LOO error)
#' @param X2 optional matrix (for Nystrm LOO error)
#' @param n_loo integer, anchors to hold out for Nystrm LOO (default 20)
#' @return named list of metric values
#' @export
woven_all_metrics <- function(Z, labels, n_total,
                               Z_true       = NULL,
                               outcome      = NULL,
                               treatment    = NULL,
                               true_effects = NULL,
                               fit          = NULL,
                               X1           = NULL,
                               X2           = NULL,
                               n_loo        = 20L) {
  stopifnot(nrow(Z) == length(labels))

  out <- list(
    silhouette      = woven_silhouette(Z, labels),
    davies_bouldin  = woven_davies_bouldin(Z, labels),
    nmi             = woven_nmi(Z, labels),
    ess_retention   = woven_ess_retention(nrow(Z), n_total)
  )
  # BER is NOT computed here. It requires per-fold DR refitting to avoid
  # circularity: supervised methods encode labels into Z, so fixed-Z BER
  # recovers labels by construction. BER is computed in benchmark_one_rep.R
  # via ber_held_out_lda(): DR refit per fold, test subjects never seen during W fit.

  if (!is.null(Z_true))
    out$rv_coefficient <- woven_rv(Z, Z_true)

  if (!is.null(outcome) && !is.null(treatment) && !is.null(true_effects))
    out$effect_bias <- woven_effect_bias(Z, outcome, treatment, labels, true_effects)

  if (!is.null(fit) && !is.null(X1) && !is.null(X2))
    out$nystrom_error <- woven_nystrom_error(fit, X1, X2, n_loo = n_loo)

  out
}
