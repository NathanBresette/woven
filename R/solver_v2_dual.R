# solver_v2_dual.R  -- Dual-space closed-form supervised WOVEN solver (V=2)
#
# Exact dual reformulation of woven_v2. Replaces p x p matrix operations
# (mat_sqrt, SVD of p1 x p2 P matrix) with n_a x n_a operations.
#
# Derivation:
#   Primal: B_v = X_a^T M_v X_a,  M_v = I + lambda_v L_a / n_a
#           P = B1^{-1/2} C12^{sup} B2^{-1/2}  (p1 x p2)
#   Key identity: sing. values of P = sing. values of M1^{-1/2} IKY M2^{-1/2}
#   Proof: P = V_F1 [U_F1^T M1^{-1/2} IKY M2^{-1/2} U_F2] V_F2^T
#          where F_v = M_v^{1/2} X_av (n_a x p_v, full row rank when p_v >= n_a)
#          U_F1, U_F2 are n_a x n_a unitary -> same singular values.
#   Dual P: P_dual = M1^{-1/2} IKY M2^{-1/2}   [n_a x n_a]
#   Anchor latent scores: Za_v = M_v^{-1/2} (left/right sing. vectors of P_dual)
#   Projection matrices: W_v = Xa_v^T Ka_v^{-1} Za_v   [p x K, min-norm soln]
#
# Speedup at ARM A (p1=10000, n_a=150):
#   mat_sqrt: O(p^3) -> O(n_a^3): ~300000x cheaper
#   SVD:      O(p1 p2 K) -> O(n_a^2 K): ~45000x cheaper
#   End-to-end: ~109s -> <0.1s expected.

#' Fit supervised WOVEN with two modalities (dual closed-form)
#'
#' Exact dual of woven_v2: all matrix operations are n_a x n_a instead of p x p.
#' Requires L_list_precomp (sample Laplacians); anchor-only B computed internally.
#'
#' @param X1 n x p1 matrix -- modality 1 (NA rows = block-missing)
#' @param X2 n x p2 matrix -- modality 2 (NA rows = block-missing)
#' @param anchor_idx integer vector -- fully-observed subject indices
#' @param Y vector of length n -- class labels (required)
#' @param K integer -- number of latent dimensions
#' @param lambda1,lambda2 numeric >= 0 -- Laplacian regularization
#' @param gamma_y numeric >= 0 -- label supervision strength
#' @param k_nn integer -- k-NN for Laplacian (ignored if L1/L2 precomp provided)
#' @param sigma1,sigma2 numeric or NULL -- RBF bandwidth (NULL = median heuristic)
#' @param L1_precomp,L2_precomp precomputed sample Laplacians (strongly recommended)
#'
#' @return list with same fields as woven_v2: W1, W2, Z1, Z2, Za1, Za2,
#'   Xa1, Xa2, col_ok1, col_ok2, singular_values, anchor_idx, K, p1, p2, etc.
#' @export
woven_v2_dual <- function(X1, X2, anchor_idx, Y,
                           K       = 5L,
                           lambda1 = 0.1,  lambda2 = 0.1,
                           gamma_y = 1.0,
                           k_nn    = 10L,  sigma1 = NULL, sigma2 = NULL,
                           L1_precomp  = NULL, L2_precomp  = NULL,
                           La1_precomp = NULL, La2_precomp = NULL) {

  check_woven_inputs(list(X1, X2), anchor_idx, K)
  stopifnot(length(Y) == nrow(X1))

  n   <- nrow(X1); p1 <- ncol(X1); p2 <- ncol(X2)
  n_a <- length(anchor_idx)

  cat(sprintf("WOVEN V=2 dual | n=%d, n_a=%d, p1=%d, p2=%d, K=%d, gamma_y=%.2f\n",
              n, n_a, p1, p2, K, gamma_y))

  # ── Anchor data (anchors are always fully observed -- no block-missing) ──
  Xa1_raw <- X1[anchor_idx, , drop = FALSE]
  Xa2_raw <- X2[anchor_idx, , drop = FALSE]
  col_ok1 <- which(!apply(Xa1_raw, 2, function(v) all(is.na(v))))
  col_ok2 <- which(!apply(Xa2_raw, 2, function(v) all(is.na(v))))
  Xa1 <- na_impute_median(Xa1_raw)
  Xa2 <- na_impute_median(Xa2_raw)

  # ── Anchor Laplacian submatrices (n_a x n_a) ────────────────────────────
  # La_precomp: pre-extracted anchor submatrix (fastest path, no re-indexing).
  # L_precomp:  full-data Laplacian; anchor submatrix extracted here.
  # Neither:    build from scratch on X (block-missing rows excluded by build_laplacian).
  if (!is.null(La1_precomp)) {
    L_a1 <- as.matrix(La1_precomp)
    L_a2 <- as.matrix(La2_precomp)
  } else if (!is.null(L1_precomp)) {
    L_a1 <- as.matrix(L1_precomp[anchor_idx, anchor_idx, drop = FALSE])
    L_a2 <- as.matrix(L2_precomp[anchor_idx, anchor_idx, drop = FALSE])
  } else {
    cat("  Building Laplacians on observed subjects (no imputation)...\n")
    L_a1 <- as.matrix(build_laplacian(X1, k=k_nn, sigma=sigma1)[anchor_idx, anchor_idx])
    L_a2 <- as.matrix(build_laplacian(X2, k=k_nn, sigma=sigma2)[anchor_idx, anchor_idx])
  }

  # ── M matrices: n_a x n_a ────────────────────────────────────────────────
  M1 <- diag(n_a) + lambda1 * L_a1 / n_a
  M2 <- diag(n_a) + lambda2 * L_a2 / n_a
  # Tiny ridge for numerical stability
  M1 <- M1 + diag(1e-8 * max(diag(M1)), n_a)
  M2 <- M2 + diag(1e-8 * max(diag(M2)), n_a)

  # ── Label kernel (n_a x n_a) ─────────────────────────────────────────────
  Y_a      <- Y[anchor_idx]
  Y_onehot <- model.matrix(~ 0 + factor(Y_a))
  Y_tilde  <- scale(Y_onehot, center = TRUE, scale = FALSE)
  KY       <- tcrossprod(Y_tilde) / n_a
  IKY      <- diag(n_a) + gamma_y * KY

  # ── M square roots (n_a x n_a — cheap) ──────────────────────────────────
  cat("  Computing M matrix square roots (n_a x n_a)...\n")
  sq1 <- mat_sqrt(M1)
  sq2 <- mat_sqrt(M2)

  # ── Dual P = M1^{-1/2} IKY M2^{-1/2}  (no kernel matrices!) ────────────
  # Sing. values of P_dual = sing. values of primal P (proof: P = V_F1 P_small V_F2^T,
  # U_F1, U_F2 unitary -> same sing. values as P_small = U_F1^T P_dual U_F2).
  cat("  Computing SVD of dual P (n_a x n_a)...\n")
  P_dual <- sq1$inv_sqrt %*% IKY %*% sq2$inv_sqrt

  K_use <- min(K, n_a - 1L, nrow(P_dual), ncol(P_dual))
  if (K_use < K) { warning(sprintf("K reduced %d -> %d.", K, K_use)); K <- K_use }

  sv <- if (K_use < min(nrow(P_dual), ncol(P_dual)) - 1L && K_use >= 1L) {
    tryCatch(RSpectra::svds(P_dual, k = K_use),
             error = function(e) svd(P_dual, nu = K_use, nv = K_use))
  } else {
    svd(P_dual, nu = K_use, nv = K_use)
  }
  U_K   <- sv$u[, seq_len(K_use), drop = FALSE]
  V_K   <- sv$v[, seq_len(K_use), drop = FALSE]
  svals <- sv$d[seq_len(K_use)]

  # ── Anchor latent scores: Za_v = M_v^{-1/2} (sing. vectors of P_dual) ───
  # Derivation: Za_v = X_av W_v = X_av B_v^{-1/2} u_k = M_v^{-1/2} u_dual
  Za1 <- sq1$inv_sqrt %*% U_K   # n_a x K
  Za2 <- sq2$inv_sqrt %*% V_K   # n_a x K

  # ── Recover W: minimum-norm solution to Xa_v W_v = Za_v ─────────────────
  # W_v = Xa_v^T Ka_v^{-1} Za_v  (right pseudoinverse when p_v >= n_a)
  cat("  Recovering W from dual solution...\n")
  Ka1 <- tcrossprod(Xa1)
  Ka2 <- tcrossprod(Xa2)
  .recover_W <- function(Xa, Ka, Za) {
    Ka_reg <- Ka + diag(1e-8 * max(diag(Ka)), nrow(Ka))
    KaInvZ <- tryCatch(solve(Ka_reg, Za),
                       error = function(e) {
                         sv <- svd(Ka_reg)
                         sv$v %*% diag(1/pmax(sv$d, 1e-10*max(sv$d)), length(sv$d)) %*%
                           t(sv$u) %*% Za
                       })
    t(Xa) %*% KaInvZ   # p x K
  }
  W1 <- .recover_W(Xa1, Ka1, Za1)
  W2 <- .recover_W(Xa2, Ka2, Za2)

  cat(sprintf("  Done. Top singular values: %s\n",
              paste(round(svals, 4), collapse = ", ")))

  list(
    W1 = W1, W2 = W2,
    Z1 = Za1, Z2 = Za2,
    Za1 = Za1, Za2 = Za2,
    Xa1 = Xa1, Xa2 = Xa2,
    col_ok1 = col_ok1, col_ok2 = col_ok2,
    Y_anchor = Y_tilde, KY = KY,
    singular_values = svals,
    # Expose M as B for compatibility with code that checks fit$B1/B2
    B1 = M1, B2 = M2,
    L1 = L1_precomp, L2 = L2_precomp,
    anchor_idx = anchor_idx,
    K = K, lambda1 = lambda1, lambda2 = lambda2,
    gamma_y = gamma_y, k_nn = k_nn, n = n, p1 = p1, p2 = p2,
    dual = TRUE
  )
}
