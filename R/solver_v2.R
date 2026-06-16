# solver_v2.R  -- Closed-form supervised WOVEN solver (V=2)
#
# Objective: supervised anchor alignment with Laplacian regularization
#
#   min  ||X_a^(1) W^(1) - X_a^(2) W^(2)||_F^2
#   s.t. W^(v)^T B_v W^(v) = I_K,   ||W^(v)_k||_1 <= s_v (optional sparsity)
#
# Supervision via label-augmented cross-covariance (no separate label weight matrix):
#
#   C_{12}^{sup} = X_a^(1)^T X_a^(2) + (gamma_y/n_a) * X_a^(1)^T (Y~Y~^T) X_a^(2)
#
#   Y~ = centered one-hot label matrix on anchor subjects (n_a x C)
#   Y~Y~^T is a label similarity kernel: entry (i,j) is positive when subjects
#   i and j share a class. Adding this to C_12 pulls canonical directions toward
#   those that are both cross-modally correlated AND class-discriminative.
#   Equivalent to DIABLO's design-matrix cross-view supervision (Singh et al. 2019).
#
# Solution: SVD of P = B_1^{-1/2} C_{12}^{sup} B_2^{-1/2}
#   W^(1) = B_1^{-1/2} U_K,   W^(2) = B_2^{-1/2} V_K
#
# With sparsity (alpha1/alpha2 > 0): PMD-style sparse SVD (Witten et al. 2009,
# J. Comput. Graph. Stat.)  -- power iteration with per-step soft-thresholding on
# U and V. Deflation after each component. Sparse W matches DIABLO's keepX.
#
# Nystrm projection for block-missing subjects: uses X features only.
# Y is not required for out-of-sample extension.

#  Sparse W via NIPALS with B-orthogonal soft-thresholding 
#
# Gives sparse projection matrices W^(1), W^(2) directly (not sparse U in
# transformed space). Equivalent to sparse CCA in the B-norm space.
#
# For each component k, iterate:
#   1. w2 <- B2^{-1} soft(C12^T w1, alpha2); normalize: w2 / sqrt(w2^T B2 w2)
#   2. w1 <- B1^{-1} soft(C12 w2, alpha1);   normalize: w1 / sqrt(w1^T B1 w1)
# Deflate C12 after each component.
#
# alpha=0 -> dense (recovers standard generalized CCA solution)
# alpha>0 -> sparse W (features with small covariance contribution zeroed)
# Heuristic: alpha_v ~ 0.05 * max(|C12|) produces ~50% sparsity
#
.sparse_nipals <- function(C12, B1_inv, B2_inv, B1, B2, K,
                           alpha1 = 0, alpha2 = 0,
                           max_iter = 200L, tol = 1e-8) {
  p1 <- nrow(C12); p2 <- ncol(C12)
  W1 <- matrix(0, p1, K)
  W2 <- matrix(0, p2, K)
  D  <- numeric(K)
  C_r <- C12

  soft  <- function(x, a) sign(x) * pmax(abs(x) - a, 0)
  bnorm <- function(w, B) { v <- as.numeric(crossprod(w, B %*% w)); sqrt(max(v, 1e-20)) }

  for (k in seq_len(K)) {
    # Warm start from leading singular vector
    sv0 <- if (min(nrow(C_r), ncol(C_r)) >= 2L) {
      tryCatch(RSpectra::svds(C_r, k = 1L),
               error = function(e) svd(C_r, nu = 1L, nv = 1L))
    } else svd(C_r, nu = 1L, nv = 1L)
    w1 <- sv0$u[, 1L]; w2 <- sv0$v[, 1L]

    for (iter in seq_len(max_iter)) {
      w1_prev <- w1

      # Update w2
      v_raw <- crossprod(C_r, w1)
      v_soft <- if (alpha2 > 0) soft(v_raw, alpha2) else v_raw
      w2_raw <- B2_inv %*% v_soft
      bn2 <- bnorm(w2_raw, B2); w2 <- w2_raw / bn2

      # Update w1
      u_raw <- C_r %*% w2
      u_soft <- if (alpha1 > 0) soft(u_raw, alpha1) else u_raw
      w1_raw <- B1_inv %*% u_soft
      bn1 <- bnorm(w1_raw, B1); w1 <- w1_raw / bn1

      if (sqrt(sum((w1 - w1_prev)^2)) < tol) break
    }

    d       <- as.numeric(crossprod(w1, C_r %*% w2))
    W1[, k] <- w1; W2[, k] <- w2; D[k] <- abs(d)
    # Deflate: remove rank-1 component
    C_r <- C_r - d * tcrossprod(B1 %*% w1, B2 %*% w2)
  }

  list(W1 = W1, W2 = W2, d = D)
}

#' Fit supervised WOVEN with two modalities (closed-form)
#'
#' @param X1 n x p1 matrix  -- modality 1 (NA rows = block-missing)
#' @param X2 n x p2 matrix  -- modality 2 (NA rows = block-missing)
#' @param anchor_idx integer vector  -- fully-observed subject indices
#' @param Y vector of length n  -- class labels (required; only anchors used in fit)
#' @param K integer  -- number of latent dimensions
#' @param lambda1,lambda2 numeric >= 0  -- Laplacian regularization per modality
#' @param gamma_y numeric >= 0  -- label supervision strength (0 = unsupervised)
#' @param alpha1,alpha2 numeric >= 0  -- L1 soft-threshold for sparse W1/W2
#'   (0 = dense; try 0.01-0.1 * max(abs(P)) for meaningful sparsity)
#' @param k_nn integer  -- k-NN for Laplacian
#' @param sigma1,sigma2 numeric or NULL  -- RBF bandwidth (NULL = median heuristic)
#' @param X1_full,X2_full optional unmasked matrices for Laplacian construction
#'
#' @return list: W1, W2, Z1, Z2, Za1, Za2, Xa1, Xa2, col_ok1, col_ok2,
#'   singular_values, B1, B2, L1, L2, anchor_idx, K, and fit metadata
#' @export
woven_v2 <- function(X1, X2, anchor_idx, Y,
                     K       = 5L,
                     lambda1 = 0.1,  lambda2 = 0.1,
                     gamma_y = 1.0,
                     alpha1  = 0,    alpha2  = 0,
                     k_nn    = 15L,  sigma1  = NULL, sigma2 = NULL,
                     X1_full = NULL, X2_full = NULL,
                     Omega1_precomp = NULL, Omega2_precomp = NULL,
                     L1_precomp = NULL,     L2_precomp = NULL,
                     XtX1_precomp = NULL,   XtX2_precomp = NULL) {

  check_woven_inputs(list(X1, X2), anchor_idx, K)
  stopifnot(length(Y) == nrow(X1))

  n   <- nrow(X1); p1 <- ncol(X1); p2 <- ncol(X2)
  n_a <- length(anchor_idx)

  cat(sprintf("WOVEN V=2 | n=%d, n_a=%d, p1=%d, p2=%d, K=%d, gamma_y=%.2f, alpha=[%.3f,%.3f]\n",
              n, n_a, p1, p2, K, gamma_y, alpha1, alpha2))

  #  Label kernel 
  Y_a      <- Y[anchor_idx]
  Y_onehot <- model.matrix(~ 0 + factor(Y_a))
  Y_tilde  <- scale(Y_onehot, center = TRUE, scale = FALSE)  # n_a x C, centered
  KY       <- tcrossprod(Y_tilde) / n_a                       # n_a x n_a

  #  Laplacians + B matrices 
  use_kernel1 <- p1 > 5L * n_a && p1 > 5000L
  use_kernel2 <- p2 > 5L * n_a && p2 > 5000L

  Xa1_raw <- X1[anchor_idx,, drop=FALSE]
  Xa2_raw <- X2[anchor_idx,, drop=FALSE]
  col_ok1 <- which(!apply(Xa1_raw, 2, function(v) all(is.na(v))))
  col_ok2 <- which(!apply(Xa2_raw, 2, function(v) all(is.na(v))))
  Xa1 <- na_impute_median(Xa1_raw)
  Xa2 <- na_impute_median(Xa2_raw)

  # Build Laplacian + Omega only if not precomputed
  if (is.null(L1_precomp) || (is.null(Omega1_precomp) && !use_kernel1)) {
    cat("  Building Laplacians...\n")
    L1 <- build_laplacian(if (!is.null(X1_full)) X1_full else X1[anchor_idx,, drop=FALSE],
                          k = k_nn, sigma = sigma1)
    if (!use_kernel1) {
      Xref1  <- na_impute_median(if (!is.null(X1_full)) X1_full else X1[anchor_idx,, drop=FALSE])
      Omega1 <- as.matrix(crossprod(Xref1, as.matrix(L1 %*% Xref1))) / nrow(Xref1)
    }
  } else {
    L1 <- L1_precomp
    Omega1 <- Omega1_precomp
  }

  if (is.null(L2_precomp) || (is.null(Omega2_precomp) && !use_kernel2)) {
    L2 <- build_laplacian(if (!is.null(X2_full)) X2_full else X2[anchor_idx,, drop=FALSE],
                          k = k_nn, sigma = sigma2)
    if (!use_kernel2) {
      Xref2  <- na_impute_median(if (!is.null(X2_full)) X2_full else X2[anchor_idx,, drop=FALSE])
      Omega2 <- as.matrix(crossprod(Xref2, as.matrix(L2 %*% Xref2))) / nrow(Xref2)
    }
  } else {
    L2 <- L2_precomp
    Omega2 <- Omega2_precomp
  }

  if (use_kernel1) {
    B1 <- compute_B_kernel(Xa1, lambda1, L1, anchor_idx)
  } else {
    B1 <- compute_B(X1, lambda1, Omega1, XtX_precomp = XtX1_precomp)
  }

  if (use_kernel2) {
    B2 <- compute_B_kernel(Xa2, lambda2, L2, anchor_idx)
  } else {
    B2 <- compute_B(X2, lambda2, Omega2, XtX_precomp = XtX2_precomp)
  }

  #  Matrix square roots 
  cat("  Computing matrix square roots...\n")
  sq1 <- mat_sqrt(B1, n_rank = n)
  sq2 <- mat_sqrt(B2, n_rank = n)

  #  Supervised cross-covariance 
  # C_{12}^{sup} = Xa1^T Xa2 + (gamma_y/n_a) * Xa1^T (Y~Y~^T) Xa2
  # The label kernel Y~Y~^T amplifies the cross-covariance between subjects
  # of the same class, pulling W toward class-discriminative directions.
  C12 <- crossprod(Xa1, Xa2)
  if (gamma_y > 0) C12 <- C12 + (gamma_y / n_a) * crossprod(Xa1, KY %*% Xa2)

  #  P matrix and sparse/dense SVD 
  cat("  Computing SVD of P...\n")
  P <- sq1$inv_sqrt %*% C12 %*% sq2$inv_sqrt

  K_use <- min(K, nrow(P), ncol(P))
  if (K_use < K) { warning(sprintf("K reduced %d -> %d.", K, K_use)); K <- K_use }

  use_sparse <- alpha1 > 0 || alpha2 > 0
  if (use_sparse) {
    # NIPALS with B-orthogonal soft-thresholding: sparse W directly
    # (PMD in U-space gives sparse U but dense W = B^{-1/2} U  -- not useful)
    B1_inv <- sq1$inv_sqrt %*% sq1$inv_sqrt
    B2_inv <- sq2$inv_sqrt %*% sq2$inv_sqrt
    nipals <- .sparse_nipals(C12, B1_inv, B2_inv, sq1$sqrt %*% sq1$sqrt,
                             sq2$sqrt %*% sq2$sqrt, K,
                             alpha1 = alpha1, alpha2 = alpha2)
    W1    <- nipals$W1
    W2    <- nipals$W2
    svals <- nipals$d
  } else if (min(nrow(P), ncol(P)) <= K + 1L || min(nrow(P), ncol(P)) < 3L) {
    sv    <- svd(P, nu = K, nv = K)
    U     <- sv$u[, seq_len(K), drop=FALSE]
    V     <- sv$v[, seq_len(K), drop=FALSE]
    svals <- sv$d[seq_len(K)]
    W1 <- sq1$inv_sqrt %*% U
    W2 <- sq2$inv_sqrt %*% V
  } else {
    sv    <- RSpectra::svds(P, k = K)
    U     <- sv$u; V <- sv$v; svals <- sv$d
    W1 <- sq1$inv_sqrt %*% U
    W2 <- sq2$inv_sqrt %*% V
  }

  #  Anchor scores 
  Z1 <- Xa1 %*% W1            # n_a x K
  Z2 <- Xa2 %*% W2            # n_a x K

  cat(sprintf("  Done. Top singular values: %s\n",
              paste(round(svals, 4), collapse = ", ")))

  list(
    W1 = W1, W2 = W2,
    Z1 = Z1, Z2 = Z2,
    Za1 = Z1, Za2 = Z2,
    Xa1 = Xa1, Xa2 = Xa2,
    col_ok1 = col_ok1, col_ok2 = col_ok2,
    Y_anchor = Y_tilde, KY = KY,
    singular_values = svals,
    B1 = B1, B2 = B2, L1 = L1, L2 = L2,
    anchor_idx = anchor_idx,
    K = K, lambda1 = lambda1, lambda2 = lambda2,
    gamma_y = gamma_y, alpha1 = alpha1, alpha2 = alpha2,
    k_nn = k_nn, n = n, p1 = p1, p2 = p2
  )
}

#' VIP scores for a fitted WOVEN V=2 model
#'
#' @param fit output of woven_v2
#' @return list with $vip1 (p1-vector) and $vip2 (p2-vector)
#' @export
woven_vip <- function(fit) {
  svals   <- fit$singular_values
  weights <- svals^2 / sum(svals^2)
  vip <- function(W, p) sqrt(p * rowSums(sweep(W^2, 2, weights, "*")))
  list(vip1 = vip(fit$W1, fit$p1), vip2 = vip(fit$W2, fit$p2))
}
