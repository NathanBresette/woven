# solver_mcca_dual.R  -- Exact dual SUMCOR MCCA for WOVEN (V >= 2)
#
# Unified closed-form dual for all V. Replaces woven_v2_dual (V=2 SVD) and
# woven_als_dual (V>=3 ALS) with a single (V*n_a) x (V*n_a) eigendecomposition.
#
# Derivation:
#   Primal (anchor-only B): B_v = X_a^T M_v X_a,  M_v = I + lambda_v L_a / n_a
#   SUMCOR objective: max sum_{v<u} tr(Z_v^T IKY Z_u)  s.t. Z_v^T M_v Z_v = I_K
#
#   Block dual P (V*n_a x V*n_a, symmetric):
#     P_{vu} = M_v^{-1/2} IKY M_u^{-1/2}  (v != u)
#     P_{vv} = 0
#
#   Equivalence proof (all V):
#     Singular values of primal block = eigenvalues of P_dual.
#     Follows from B_v^{-1/2} X_a_v^T = V_Fv U_Fv^T M_v^{-1/2}, with
#     U_Fv unitary (full-row-rank X_a_v, p_v >= n_a) -- same argument as V=2.
#
#   V=2 reduces exactly: top singular values of M1^{-1/2} IKY M2^{-1/2} equal
#     eigenvalues of the 2-block P_dual (off-diagonal blocks +-M1^{-1/2} IKY M2^{-1/2}).
#
#   Za_v = M_v^{-1/2} phi_v  (v-th block of top-K eigenvectors)
#   W_v  = X_a_v^T K_v^{-1} Za_v  (min-norm solution, p_v x K)
#
# Cost:
#   Build P:     O(V^2 n_a^2)   -- V^2 mat_sqrts + block multiplications
#   Eigen:       O((V n_a)^3)   -- ARM B V=3 n_a=150: 450^3 ~ 91M ops (<1s)
#   W recovery:  O(V n_a^2 p_v) -- dominated by X_a^T multiply

#' Fit supervised WOVEN for V >= 2 views via dual SUMCOR MCCA (closed-form)
#'
#' Single eigendecomposition of a (V*n_a) x (V*n_a) block matrix. No iterations,
#' no random restarts, no local optima. Unified solver for all V.
#'
#' @param X_list  list of V matrices, each n x p_v (NA rows = block-missing)
#' @param anchor_idx integer vector of fully-observed subject indices
#' @param Y vector of length n, class labels (required)
#' @param K integer, number of latent dimensions
#' @param lambdas numeric scalar or length-V vector, Laplacian regularization
#' @param gamma_y numeric >= 0, label supervision strength
#' @param k_nn integer, k-NN for Laplacian (ignored if La_list_precomp supplied)
#' @param La_list_precomp optional list of pre-extracted n_a x n_a anchor Laplacians
#' @param verbose logical
#'
#' @return list with W_list, Za_list, Xa_list, singular_values, and metadata.
#'   Compatible with project_all() in benchmark_one_rep.R.
#' @examples
#' set.seed(1)
#' n <- 20
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' X3 <- matrix(rnorm(n * 3), n, 3)
#' Y <- rep(1:2, each = n / 2)
#' anchor_idx <- seq_len(14L)
#' fit <- woven_mcca_dual(list(X1, X2, X3), anchor_idx = anchor_idx, Y = Y, K = K)
#' length(fit$W_list)
#' @export
woven_mcca_dual <- function(X_list, anchor_idx, Y,
                            K = 5L,
                            lambdas = 0.1,
                            gamma_y = 1.0,
                            k_nn = 10L,
                            La_list_precomp = NULL,
                            verbose = TRUE) {
    V <- length(X_list)
    n <- nrow(X_list[[1]])
    n_a <- length(anchor_idx)
    check_woven_inputs(X_list, anchor_idx, K)
    stopifnot(length(Y) == n)

    if (length(lambdas) == 1L) lambdas <- rep(lambdas, V)
    stopifnot(length(lambdas) == V)

    if (verbose) {
        message(sprintf(
            "WOVEN MCCA-dual | V=%d, n=%d, n_a=%d, K=%d, gamma_y=%.2f\n",
            V, n, n_a, K, gamma_y
        ))
    }

    # ── Anchor data (fully observed by definition) ───────────────────────────
    Xa_raw <- lapply(X_list, function(X) X[anchor_idx, , drop = FALSE])
    col_ok <- lapply(Xa_raw, function(X) which(!apply(X, 2, function(v) all(is.na(v)))))
    Xa_list <- lapply(Xa_raw, na_impute_median)

    # ── Anchor Laplacian submatrices (n_a x n_a) ────────────────────────────
    if (verbose) message("  Extracting anchor Laplacians...\n")
    L_a_list <- if (!is.null(La_list_precomp)) {
        lapply(La_list_precomp, as.matrix)
    } else {
        lapply(X_list, function(X) {
            as.matrix(build_laplacian(X, k = k_nn)[anchor_idx, anchor_idx])
        })
    }

    # ── M_v matrices and their square roots (n_a x n_a) ─────────────────────
    if (verbose) message("  Computing M square roots (n_a x n_a)...\n")
    M_list <- lapply(seq_len(V), function(v) {
        M <- diag(n_a) + lambdas[v] * L_a_list[[v]] / n_a
        M + diag(1e-8 * max(diag(M)), n_a)
    })
    sq_list <- lapply(M_list, mat_sqrt) # list of list(sqrt, inv_sqrt, ...)
    M_chol <- lapply(M_list, chol) # for m_orth below

    # ── Label kernel (n_a x n_a) ────────────────────────────────────────────
    Y_a <- Y[anchor_idx]
    Y_onehot <- model.matrix(~ 0 + factor(Y_a))
    Y_tilde <- scale(Y_onehot, center = TRUE, scale = FALSE)
    KY <- tcrossprod(Y_tilde) / n_a
    IKY <- diag(n_a) + gamma_y * KY

    # ── Build (V*n_a) x (V*n_a) block dual matrix ───────────────────────────
    # P_{vu} = M_v^{-1/2} IKY M_u^{-1/2}  (v != u),  0  (v == u)
    if (verbose) {
        message(sprintf(
            "  Building %dx%d block matrix (V*n_a)...\n", V * n_a, V * n_a
        ))
    }

    # Pre-compute M_v^{-1/2} IKY once per row-view, reuse across columns
    MiIKY <- lapply(seq_len(V), function(v) {
        sq_list[[v]]$inv_sqrt %*% IKY
    }) # M_v^{-1/2} IKY, n_a x n_a

    P_full <- matrix(0.0, nrow = V * n_a, ncol = V * n_a)
    for (v in seq_len(V)) {
        row_idx <- ((v - 1L) * n_a + 1L):(v * n_a)
        for (u in seq_len(V)) {
            if (u == v) next
            col_idx <- ((u - 1L) * n_a + 1L):(u * n_a)
            P_full[row_idx, col_idx] <- MiIKY[[v]] %*% sq_list[[u]]$inv_sqrt
        }
    }

    # ── Top-K eigendecomposition of symmetric P_full ────────────────────────
    # Always use full eigen() -- P_full is (V*n_a)x(V*n_a) <= ~1200x1200 in
    # benchmarks, so full decomp is fast and avoids ARPACK convergence issues
    # with the anti-diagonal block structure. Keep only POSITIVE eigenvalues:
    # negative eigenvectors correspond to anti-correlated view directions and
    # would corrupt the SUMCOR solution.
    if (verbose) message("  Eigendecomposition...\n")
    ev <- eigen(P_full, symmetric = TRUE) # full, sorted descending
    pos_idx <- which(ev$values > 1e-10) # strictly positive only
    if (length(pos_idx) == 0L) stop("No positive eigenvalues in P_full.")
    K_use <- min(K, length(pos_idx))
    if (K_use < K) {
        warning(sprintf(
            "Only %d positive eigenvalues; K reduced %d -> %d.", length(pos_idx), K, K_use
        ))
    }
    K <- K_use
    keep <- pos_idx[seq_len(K_use)] # indices of top-K positive evals
    evals <- ev$values[keep]
    Phi <- ev$vectors[, keep, drop = FALSE] # (V*n_a) x K

    # ── Extract per-view blocks, compute Za and W ────────────────────────────
    if (verbose) message("  Extracting Za and recovering W...\n")
    Za_list <- vector("list", V)
    W_list <- vector("list", V)

    for (v in seq_len(V)) {
        row_idx <- ((v - 1L) * n_a + 1L):(v * n_a)
        phi_v <- Phi[row_idx, , drop = FALSE] # n_a x K

        # Za_v = M_v^{-1/2} phi_v, then M-orthonormalize (Za^T M Za = I)
        Za_v_raw <- sq_list[[v]]$inv_sqrt %*% phi_v
        ZMZ <- crossprod(M_chol[[v]] %*% Za_v_raw)
        ZMZ <- (ZMZ + t(ZMZ)) / 2
        R_za <- tryCatch(chol(ZMZ), error = function(e) NULL)
        Za_list[[v]] <- if (!is.null(R_za)) {
            t(backsolve(R_za, t(Za_v_raw)))
        } else {
            # fallback: eigendecomposition
            ev2 <- eigen(ZMZ, symmetric = TRUE)
            d_inv <- 1 / sqrt(pmax(ev2$values[seq_len(K_use)], 1e-14))
            Za_v_raw %*% ev2$vectors[, seq_len(K_use), drop = FALSE] %*% diag(d_inv, K_use)
        }

        # W_v = X_a_v^T K_v^{-1} Za_v  (min-norm solution, p_v x K)
        Ka_v <- tcrossprod(Xa_list[[v]])
        Ka_reg <- Ka_v + diag(1e-8 * max(diag(Ka_v)), n_a)
        W_list[[v]] <- t(Xa_list[[v]]) %*% tryCatch(
            solve(Ka_reg, Za_list[[v]]),
            error = function(e) {
                sv <- svd(Ka_reg)
                sv$v %*% diag(1 / pmax(sv$d, 1e-10 * max(sv$d)), length(sv$d)) %*%
                    t(sv$u) %*% Za_list[[v]]
            }
        )
    }

    if (verbose) {
        message(sprintf(
            "  Done. Top eigenvalues: %s\n",
            paste(round(evals[seq_len(min(5L, K_use))], 4), collapse = ", ")
        ))
    }

    list(
        W_list = W_list,
        Z_list = Za_list,
        Za_list = Za_list,
        Xa_list = Xa_list,
        col_ok_list = col_ok,
        singular_values = evals,
        M_list = M_list,
        L_a_list = L_a_list,
        anchor_idx = anchor_idx,
        K = K, V = V,
        lambdas = lambdas, gamma_y = gamma_y, k_nn = k_nn,
        n = n, p_v = vapply(X_list, ncol, integer(1L)),
        dual = TRUE
    )
}
