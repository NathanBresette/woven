# project.R  -- Nystrom out-of-sample projection for block-missing subjects
#
# Projects subjects missing one or more modalities into the shared latent space
# using kernel-weighted interpolation from anchor positions.
#
# Method: Nystrom extension (Bengio et al. 2004, NeurIPS)
# For a subject i missing modality v, with observed modality u:
#   z_i = sum_a k(x_i^(u), x_a^(u)) * z_a  /  sum_a k(x_i^(u), x_a^(u))
#
# Optimization: subjects are batched by type so kernel computation is a single
# matrix multiply rather than a per-subject loop.

# Vectorized Nystrom kernel: n_q x n_a normalized kernel matrix
# Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> for efficiency.
.nystrom_kernel_batch <- function(X_q, X_a, sigma) {
  sq_q <- rowSums(X_q^2)
  sq_a <- rowSums(X_a^2)
  D2   <- pmax(outer(sq_q, sq_a, "+") - 2 * tcrossprod(X_q, X_a), 0)
  K    <- exp(-D2 / sigma^2)
  K / (rowSums(K) + .Machine$double.eps)
}

# Batch impute feature-level NAs using anchor column medians
.batch_impute <- function(X, Xa_ref) {
  if (!any(is.na(X))) { X[!is.finite(X)] <- 0; return(X) }
  col_med <- apply(Xa_ref, 2, median, na.rm = TRUE)
  na_pos  <- which(is.na(X), arr.ind = TRUE)
  X[na_pos] <- col_med[na_pos[, 2L]]
  X[!is.finite(X)] <- 0
  X
}

#' Project all samples into the shared latent space
#'
#' @param fit output of woven_v2
#' @param X1 full modality 1 matrix (n x p1); NA rows = block-missing subject
#' @param X2 full modality 2 matrix (n x p2); NA rows = block-missing subject
#' @param sigma_proj RBF bandwidth for projection kernel (NULL = median heuristic from anchors)
#'
#' @return list:
#'   $Z1     n x K latent matrix (modality 1 perspective)
#'   $Z2     n x K latent matrix (modality 2 perspective)
#'   $Z      n x K consensus embedding (feature-count-weighted average)
#'   $method per-subject character: "direct", "partial", or "missing"
#' @export
woven_project <- function(fit, X1, X2, sigma_proj = NULL) {
  n   <- nrow(X1)
  K   <- fit$K
  W1  <- fit$W1;  W2  <- fit$W2
  anchor_idx <- fit$anchor_idx
  col_ok1 <- fit$col_ok1;  col_ok2 <- fit$col_ok2

  Za1 <- fit$Za1;  Za2 <- fit$Za2
  if (is.null(Za1)) { Za1 <- fit$Xa1 %*% W1;  Za2 <- fit$Xa2 %*% W2 }
  Xa1 <- fit$Xa1;  Xa2 <- fit$Xa2

  # Bandwidth from anchor pairwise distances
  if (is.null(sigma_proj)) {
    d1 <- as.vector(dist(Xa1));  sigma1 <- median(d1[d1 > 0])
    d2 <- as.vector(dist(Xa2));  sigma2 <- median(d2[d2 > 0])
    if (!is.finite(sigma1) || sigma1 == 0) sigma1 <- 1.0
    if (!is.finite(sigma2) || sigma2 == 0) sigma2 <- 1.0
  } else {
    sigma1 <- sigma_proj;  sigma2 <- sigma_proj
  }

  miss1 <- apply(X1, 1, function(r) all(is.na(r)))
  miss2 <- apply(X2, 1, function(r) all(is.na(r)))

  Z1     <- matrix(NA_real_, nrow = n, ncol = K)
  Z2     <- matrix(NA_real_, nrow = n, ncol = K)
  method <- character(n)

  # ── 1. Anchors (direct, always correct) ────────────────────────────────────
  Z1[anchor_idx, ] <- Za1
  Z2[anchor_idx, ] <- Za2
  method[anchor_idx] <- "direct"

  non_anc <- setdiff(seq_len(n), anchor_idx)
  if (length(non_anc) == 0L) {
    w1 <- fit$p1 / (fit$p1 + fit$p2);  w2 <- fit$p2 / (fit$p1 + fit$p2)
    Z  <- w1 * Z1 + w2 * Z2
    return(list(Z1 = Z1, Z2 = Z2, Z = Z, method = method))
  }

  obs1_na <- !miss1[non_anc]
  obs2_na <- !miss2[non_anc]

  # ── 2. Both views present: direct projection (no approximation) ────────────
  both_idx <- non_anc[obs1_na & obs2_na]
  if (length(both_idx) > 0L) {
    X1b <- .batch_impute(X1[both_idx, col_ok1, drop = FALSE], Xa1)
    X2b <- .batch_impute(X2[both_idx, col_ok2, drop = FALSE], Xa2)
    Z1[both_idx, ] <- X1b %*% W1
    Z2[both_idx, ] <- X2b %*% W2
    method[both_idx] <- "direct"
  }

  # ── 3. Only view 1 present: direct Z1, Nystrom Z2 ─────────────────────────
  only1_idx <- non_anc[obs1_na & !obs2_na]
  if (length(only1_idx) > 0L) {
    X1p <- .batch_impute(X1[only1_idx, col_ok1, drop = FALSE], Xa1)
    Z1[only1_idx, ] <- X1p %*% W1
    Km <- .nystrom_kernel_batch(X1p, Xa1, sigma1)
    Z2[only1_idx, ] <- Km %*% Za2
    method[only1_idx] <- "partial"
  }

  # ── 4. Only view 2 present: direct Z2, Nystrom Z1 ─────────────────────────
  only2_idx <- non_anc[!obs1_na & obs2_na]
  if (length(only2_idx) > 0L) {
    X2p <- .batch_impute(X2[only2_idx, col_ok2, drop = FALSE], Xa2)
    Z2[only2_idx, ] <- X2p %*% W2
    Km <- .nystrom_kernel_batch(X2p, Xa2, sigma2)
    Z1[only2_idx, ] <- Km %*% Za1
    method[only2_idx] <- "partial"
  }

  # ── 5. Consensus embedding ─────────────────────────────────────────────────
  w1 <- fit$p1 / (fit$p1 + fit$p2);  w2 <- fit$p2 / (fit$p1 + fit$p2)
  Z  <- matrix(NA_real_, nrow = n, ncol = K)
  valid  <- !is.na(Z1[, 1]) & !is.na(Z2[, 1])
  only_1 <- !is.na(Z1[, 1]) & is.na(Z2[, 1])
  only_2 <- is.na(Z1[, 1]) & !is.na(Z2[, 1])
  Z[valid,  ] <- w1 * Z1[valid,  ] + w2 * Z2[valid,  ]
  Z[only_1, ] <- Z1[only_1, ]
  Z[only_2, ] <- Z2[only_2, ]

  list(Z1 = Z1, Z2 = Z2, Z = Z, method = method)
}
