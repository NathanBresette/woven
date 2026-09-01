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
#' @param ridge_w numeric >= 0, relative ridge strength for the W recovery step
#'   (Ka_reg = Ka_v + ridge_w * max(diag(Ka_v)) * I). Default 1e-8 is a pure
#'   numerical-stability nudge (original behavior). Larger values shrink
#'   W_v toward zero, trading anchor fit for reduced variance when p_v >> n_a
#'   (v7 regularization; CV-selected like lambdas and gamma_y).
#' @param screen_top integer or Inf, per-modality feature count for
#'   univariate (ANOVA F-score vs Y) supervised pre-screening BEFORE the W
#'   recovery step. Default Inf disables this (all features used, original
#'   behavior). The eigendecomposition (Za_list) never depends on features --
#'   only the Laplacians and label kernel -- so screening only changes the
#'   W_v = X_a_v^T Ka_v^{-1} Za_v recovery: Ka_v = X_a_v X_a_v^T is built
#'   from the screened columns only, which is a genuine re-fit of the
#'   min-norm interpolation on a reduced feature set (not a post-hoc
#'   truncation of an already-complete solution). Screened-out features get
#'   an exact zero row in the returned W_v (same shape as unscreened, so
#'   all downstream code -- project_all(), woven_scores() -- is unaffected).
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
                            ridge_w = 1e-8,
                            screen_top = Inf,
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

        # ── Supervised univariate pre-screening (v7) ────────────────────────
        # Selects which columns of Xa_list[[v]] enter the Gram matrix BELOW,
        # before W is recovered -- a genuine re-fit of the min-norm
        # interpolation on a reduced feature set (Ka_v = X_screened X_screened^T
        # is a different matrix, not a truncation of the full-feature answer).
        # Za_list above is unaffected (never depended on features). Scored by
        # ANOVA F-statistic vs Y on anchor data only.
        p_v <- ncol(Xa_list[[v]])
        screen_idx <- seq_len(p_v)
        if (is.finite(screen_top) && screen_top < p_v) {
            Xv <- Xa_list[[v]]
            y_f <- as.factor(Y_a)
            grand_mean <- colMeans(Xv)
            ss_between <- numeric(p_v)
            ss_within  <- numeric(p_v)
            for (lev in levels(y_f)) {
                idx <- which(y_f == lev)
                n_c <- length(idx)
                if (n_c < 2L) next
                Xc <- Xv[idx, , drop = FALSE]
                mean_c <- colMeans(Xc)
                ss_between <- ss_between + n_c * (mean_c - grand_mean)^2
                ss_within  <- ss_within + colSums(sweep(Xc, 2, mean_c, "-")^2)
            }
            df_b <- nlevels(y_f) - 1L
            df_w <- n_a - nlevels(y_f)
            f_score <- (ss_between / max(df_b, 1L)) / (ss_within / max(df_w, 1L) + 1e-12)
            f_score[!is.finite(f_score)] <- 0
            screen_idx <- order(f_score, decreasing = TRUE)[seq_len(screen_top)]
        }

        # W_v = X_a_v^T K_v^{-1} Za_v  (min-norm solution, p_v x K)
        # Gram matrix built from screened columns only; unscreened rows of
        # W_v stay exactly zero (same p_v x K shape either way).
        Xa_screened <- Xa_list[[v]][, screen_idx, drop = FALSE]
        Ka_v <- tcrossprod(Xa_screened)
        Ka_reg <- Ka_v + diag(ridge_w * max(diag(Ka_v)), n_a)
        W_screened <- t(Xa_screened) %*% tryCatch(
            solve(Ka_reg, Za_list[[v]]),
            error = function(e) {
                sv <- svd(Ka_reg)
                sv$v %*% diag(1 / pmax(sv$d, 1e-10 * max(sv$d)), length(sv$d)) %*%
                    t(sv$u) %*% Za_list[[v]]
            }
        )
        Wv_full <- matrix(0, p_v, K_use)
        Wv_full[screen_idx, ] <- W_screened
        W_list[[v]] <- Wv_full
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
