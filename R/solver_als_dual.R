# solver_als_dual.R  -- Exact dual-space supervised ALS for WOVEN (V >= 2)
#
# Exact dual of woven_als. Replaces O(p^2 K) primal solve with O(n_a^2 K)
# dual solve. Mathematically identical to primal with anchor-only B_v.
#
# Key identity:
#   Primal (anchor-only B): B_v W_v = X_a^T T_v
#     where B_v = X_a^T M_v X_a,  M_v = I + lambda_v L_a_v / n_a
#   Dual: Z_v = M_v^{-1} T_v  (EXACT: same solution, n_a x n_a solve)
#     where Z_v = X_a W_v,  T_v = sum_{u!=v} (I + gamma_y KY) Z_u
#
# Anchor-only B is statistically preferable under block-missingness:
#   non-anchor subjects have median-imputed features that bias the full-data
#   covariance estimate; anchor subjects have real complete observations.
#   In the complete condition, anchor_idx = 1:n so no difference.
#
# Speedup: O((p/n_a)^2) per solve.
# ARM B (p=5000, n_a=150): ~1111x per solve, ~504s -> <1s.

#' Fit supervised WOVEN V>=2 via exact dual-space ALS
#'
#' @param X_list list of V matrices, each n x p_v (NA rows = block-missing)
#' @param anchor_idx integer vector of fully-observed subject indices
#' @param Y vector of length n -- class labels (required)
#' @param K integer, number of latent dimensions
#' @param lambdas numeric vector length V, Laplacian regularization
#' @param gamma_y numeric >= 0, label supervision strength
#' @param k_nn integer, k-NN for Laplacian (ignored if L_list_precomp provided)
#' @param max_iter integer, max ALS iterations
#' @param tol numeric, relative convergence threshold
#' @param n_restarts integer, random restarts; best objective wins
#' @param L_list_precomp optional precomputed sample Laplacians (strongly recommended)
#' @param La_list_precomp optional precomputed anchor submatrices of Laplacians
#' @param verbose logical
#'
#' @return list compatible with woven_als: W_list, Z_list, Za_list, Xa_list,
#'   col_ok_list, B_list, L_list, anchor_idx, K, V, objective, obj_trace, etc.
#' @examples
#' set.seed(1)
#' n <- 20
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' X3 <- matrix(rnorm(n * 3), n, 3)
#' Y <- rep(1:2, each = n / 2)
#' anchor_idx <- seq_len(14L)
#' fit <- woven_als_dual(list(X1, X2, X3), anchor_idx = anchor_idx, Y = Y, K = K)
#' length(fit$W_list)
#' @export
woven_als_dual <- function(X_list, anchor_idx, Y,
                           K = 5L,
                           lambdas = NULL,
                           gamma_y = 1.0,
                           k_nn = 10L,
                           max_iter = 200L,
                           tol = 1e-6,
                           n_restarts = 5L,
                           L_list_precomp = NULL,
                           La_list_precomp = NULL,
                           verbose = TRUE) {
    V <- length(X_list)
    n <- nrow(X_list[[1]])
    n_a <- length(anchor_idx)
    check_woven_inputs(X_list, anchor_idx, K)
    stopifnot(length(Y) == n)

    if (is.null(lambdas)) lambdas <- rep(0.1, V)
    if (length(lambdas) == 1L) lambdas <- rep(lambdas, V)
    stopifnot(length(lambdas) == V)

    if (verbose) {
        message(sprintf(
            "WOVEN ALS-dual | V=%d, n=%d, n_a=%d, K=%d, gamma_y=%.2f, restarts=%d\n",
            V, n, n_a, K, gamma_y, n_restarts
        ))
    }

    # ── Anchor data ────────────────────────────────────────────────────────────
    Xa_raw <- lapply(X_list, function(X) X[anchor_idx, , drop = FALSE])
    col_ok <- lapply(Xa_raw, function(X) which(!apply(X, 2, function(v) all(is.na(v)))))
    Xa_list <- lapply(Xa_raw, na_impute_median)

    # ── Anchor Laplacian submatrices (n_a x n_a, no imputation) ───────────────
    # La_list_precomp: fastest -- pre-extracted anchor submatrices from make_precomp.
    # L_list_precomp:  full-data Laplacians; anchor submatrix extracted here.
    # Neither:         build from scratch (block-missing rows excluded automatically).
    if (verbose) message("  Extracting anchor Laplacians...\n")
    L_a_list <- if (!is.null(La_list_precomp)) {
        lapply(La_list_precomp, as.matrix)
    } else if (!is.null(L_list_precomp)) {
        lapply(L_list_precomp, function(L) {
            as.matrix(L[anchor_idx, anchor_idx, drop = FALSE])
        })
    } else {
        lapply(X_list, function(X) {
            as.matrix(build_laplacian(X, k = k_nn)[anchor_idx, anchor_idx])
        })
    }
    L_list <- L_list_precomp # retained for output slot only

    # ── M_v = I + lambda_v * L_a_v / n_a  (n_a x n_a, fixed) ─────────────────
    M_list <- lapply(seq_len(V), function(v) {
        M <- diag(n_a) + lambdas[v] * L_a_list[[v]] / n_a
        M + diag(1e-8 * max(diag(M)), n_a)
    })

    # Cholesky of each M once — shared across ALL restarts and ALL iterations
    if (verbose) message("  Factorizing M matrices (n_a x n_a)...\n")
    M_chol <- lapply(M_list, function(M) {
        tryCatch(chol(M), error = function(e) NULL)
    })

    # ── Label kernel ───────────────────────────────────────────────────────────
    Y_a <- Y[anchor_idx]
    Y_onehot <- model.matrix(~ 0 + factor(Y_a))
    Y_tilde <- scale(Y_onehot, center = TRUE, scale = FALSE)
    KY <- tcrossprod(Y_tilde) / n_a
    IKY <- diag(n_a) + gamma_y * KY # fixed throughout

    # ── M-orthogonalization (dual analog of primal b_orth) ─────────────────────
    # Goal: Z_new s.t. Z_new^T M Z_new = I_K
    m_orth <- function(Z, R_M) {
        # Z^T M Z = Z^T R_M^T R_M Z = (R_M Z)^T (R_M Z)  [chol: t(R) R = M]
        ZMZ <- crossprod(R_M %*% Z)
        ZMZ <- (ZMZ + t(ZMZ)) / 2
        R2 <- tryCatch(chol(ZMZ), error = function(e) NULL)
        if (!is.null(R2)) {
            return(t(backsolve(R2, t(Z))))
        }
        ev <- eigen(ZMZ, symmetric = TRUE)
        K_use <- ncol(Z)
        d_inv <- 1 / sqrt(pmax(ev$values[seq_len(K_use)], 1e-14))
        Z %*% ev$vectors[, seq_len(K_use), drop = FALSE] %*% diag(d_inv, K_use)
    }

    # ── Precompute top-K left singular vectors of each Xa (for PCA warm start) ─
    Xa_svd_U <- lapply(Xa_list, function(Xa) {
        tryCatch(.svds_safe(Xa, k = K)$u,
            error = function(e) matrix(rnorm(n_a * K), n_a, K)
        )
    })

    # ── Warm start: restart 1 = PCA init; subsequent = random ─────────────────
    # PCA (top-K left singular vectors of Xa_v, M-orthonormalized) gives a
    # structured starting point that often lands in the right basin.
    cnt_env <- new.env(parent = emptyenv())
    cnt_env$restart_count <- 0L
    warm_start <- function() {
        cnt_env$restart_count <- cnt_env$restart_count + 1L
        lapply(seq_len(V), function(v) {
            Z0 <- if (cnt_env$restart_count == 1L) Xa_svd_U[[v]] else matrix(rnorm(n_a * K), n_a, K)
            if (!is.null(M_chol[[v]])) m_orth(Z0, M_chol[[v]]) else Z0
        })
    }

    # ── Dual ALS iteration ─────────────────────────────────────────────────────
    als_run <- function(Z_init) {
        Z_list <- Z_init
        obj_prev <- Inf
        obj_trace <- numeric(max_iter)

        for (iter in seq_len(max_iter)) {
            for (v in seq_len(V)) {
                others <- setdiff(seq_len(V), v)
                # T_v = sum_{u!=v} (I + gamma_y KY) Z_u  [n_a x K]
                T_v <- Reduce("+", lapply(others, function(u) IKY %*% Z_list[[u]]))

                # Dual update: Z_v = M_v^{-1} T_v  [exact, O(n_a^2 K)]
                Z_new <- if (!is.null(M_chol[[v]])) {
                    backsolve(
                        M_chol[[v]],
                        backsolve(M_chol[[v]], T_v, transpose = TRUE)
                    )
                } else {
                    tryCatch(solve(M_list[[v]], T_v), error = function(e) T_v)
                }
                Z_new[!is.finite(Z_new)] <- 0
                Z_list[[v]] <- m_orth(Z_new, M_chol[[v]])
            }

            # Objective: pairwise Frobenius distances under supervised metric
            obj <- 0
            for (v in seq_len(V - 1L)) {
                for (u in seq(v + 1L, V)) {
                    d <- Z_list[[v]] - Z_list[[u]]
                    obj <- obj + sum(d^2)
                    if (gamma_y > 0) obj <- obj + gamma_y * sum(d * (KY %*% d))
                }
            }
            if (!is.finite(obj)) obj <- obj_prev
            obj_trace[iter] <- obj

            if (verbose && iter %% 20L == 0L) {
                message(sprintf("    iter %d: obj=%.6f\n", iter, obj))
            }

            if (is.finite(obj_prev) &&
                abs(obj_prev - obj) / (max(abs(obj_prev), 1.0) + 1e-12) < tol) {
                obj_trace <- obj_trace[seq_len(iter)]
                break
            }
            obj_prev <- obj
        }
        list(Z_list = Z_list, objective = obj, obj_trace = obj_trace)
    }

    # ── Multiple restarts ──────────────────────────────────────────────────────
    best <- NULL
    for (r in seq_len(n_restarts)) {
        if (verbose) message(sprintf("  Restart %d/%d...\n", r, n_restarts))
        result <- als_run(warm_start())
        if (is.null(best) || result$objective < best$objective) best <- result
    }

    Za_list <- best$Z_list

    # ── Recover W_list: W_v = X_a^T K_v^{-1} Z_v  [once, p x K per view] ─────
    if (verbose) message("  Recovering W_list from dual solution...\n")
    W_list <- lapply(seq_len(V), function(v) {
        Ka <- tcrossprod(Xa_list[[v]])
        Ka_reg <- Ka + diag(1e-8 * max(diag(Ka)), n_a)
        KaInvZ <- tryCatch(
            solve(Ka_reg, Za_list[[v]]),
            error = function(e) {
                sv <- svd(Ka_reg)
                sv$v %*% diag(1 / pmax(sv$d, 1e-10 * max(sv$d)), length(sv$d)) %*%
                    t(sv$u) %*% Za_list[[v]]
            }
        )
        t(Xa_list[[v]]) %*% KaInvZ # p_v x K
    })

    # ── Singular values ────────────────────────────────────────────────────────
    svals <- if (V == 2L) {
        svd(crossprod(Za_list[[1]], Za_list[[2]]))$d[seq_len(K)]
    } else {
        tryCatch(.svds_safe(do.call(cbind, Za_list), k = K)$d,
            error = function(e) rep(NA_real_, K)
        )
    }

    if (verbose) {
        message(sprintf(
            "  Done. Objective=%.6f | Top svals: %s\n",
            best$objective,
            paste(round(svals[seq_len(min(5L, K))], 4), collapse = ", ")
        ))
    }

    list(
        W_list = W_list,
        Z_list = Za_list,
        Za_list = Za_list,
        Xa_list = Xa_list,
        col_ok_list = col_ok,
        Y_anchor = Y_tilde,
        KY = KY,
        singular_values = svals,
        B_list = M_list, # n_a x n_a; primal was p x p
        L_list = L_list,
        L_a_list = L_a_list,
        M_list = M_list,
        anchor_idx = anchor_idx,
        K = K, V = V,
        lambdas = lambdas,
        gamma_y = gamma_y,
        k_nn = k_nn,
        n = n,
        p_v = vapply(X_list, ncol, integer(1L)),
        objective = best$objective,
        obj_trace = best$obj_trace,
        dual = TRUE
    )
}
