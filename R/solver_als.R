# solver_als.R  -- Supervised ALS solver for WOVEN with V >= 2 modalities
#
# Supervised objective: label-augmented multi-view alignment
#
#   J = sum_{v<u} ||X_a^(v) W^(v) - X_a^(u) W^(u)||_F^2
#       + gamma_y * sum_{v<u} ||X_a^(v) W^(v) - X_a^(u) W^(u)||_{KY}^2
#   s.t. W^(v)^T B_v W^(v) = I_K,   sparse W via soft-thresholding (optional)
#
# Supervision enters through an augmented cross-covariance, consistent with
# the V=2 closed-form solver. For each X view v, the ALS update uses:
#
#   RHS_v = Xa_v^T * sum_{u!=v} (I + gamma_y * KY) * Xa_u * W_u
#
# where KY = Y~Y~^T / n_a is the n_a x n_a label similarity kernel.
# No separate label weight matrix W^(Y)  -- avoids rank-deficiency when K > C-1.
#
# Sparsity: optional L1 soft-thresholding on W after each solve, followed by
# re-B-orthogonalization. Mimics DIABLO's keepX sparse loadings.

#' Fit supervised WOVEN with V >= 2 modalities via ALS
#'
#' @param X_list list of V matrices, each n x p_v (NA rows = block-missing)
#' @param anchor_idx integer vector of fully-observed subject indices
#' @param Y vector of length n  -- class labels (required)
#' @param K integer, number of latent dimensions
#' @param lambdas numeric vector of length V, Laplacian regularization
#' @param gamma_y numeric >= 0, label supervision strength
#' @param alpha numeric >= 0, L1 soft-threshold on W (0 = dense, like glmnet alpha)
#' @param k_nn integer, k-NN for Laplacian
#' @param max_iter integer, max ALS iterations
#' @param tol numeric, relative convergence threshold
#' @param n_restarts integer, random restarts; best objective wins
#' @param X_list_full optional unmasked matrices for Laplacian construction
#' @param Omega_list_precomp optional precomputed Omega (graph smoothing) matrices
#' @param L_list_precomp optional precomputed sample Laplacians
#' @param XtX_list_precomp optional precomputed X^T X matrices
#' @param verbose logical
#'
#' @return list: W_list, Z_list, Za_list, Xa_list, col_ok_list,
#'   singular_values, objective_trace, B_list, L_list, metadata
#' @examples
#' set.seed(1)
#' n <- 20
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' X3 <- matrix(rnorm(n * 3), n, 3)
#' Y <- rep(1:2, each = n / 2)
#' anchor_idx <- seq_len(14L)
#' fit <- woven_als(list(X1, X2, X3), anchor_idx = anchor_idx, Y = Y, K = K)
#' length(fit$W_list)
#' @export
woven_als <- function(X_list, anchor_idx, Y,
                      K = 5L,
                      lambdas = NULL,
                      gamma_y = 1.0,
                      alpha = 0,
                      k_nn = 15L,
                      max_iter = 200L,
                      tol = 1e-6,
                      n_restarts = 5L,
                      X_list_full = NULL,
                      Omega_list_precomp = NULL,
                      L_list_precomp = NULL,
                      XtX_list_precomp = NULL,
                      verbose = TRUE) {
    V <- length(X_list)
    check_woven_inputs(X_list, anchor_idx, K)
    stopifnot(length(Y) == nrow(X_list[[1]]))

    n <- nrow(X_list[[1]])
    n_a <- length(anchor_idx)

    if (is.null(lambdas)) lambdas <- rep(0.1, V)
    if (length(lambdas) == 1L) lambdas <- rep(lambdas, V)
    stopifnot(length(lambdas) == V)

    if (verbose) {
        message(sprintf(
            "WOVEN ALS | V=%d, n=%d, n_a=%d, K=%d, gamma_y=%.2f, alpha=%.3f, restarts=%d\n",
            V, n, n_a, K, gamma_y, alpha, n_restarts
        ))
    }

    #  Label kernel (fixed throughout)
    Y_a <- Y[anchor_idx]
    Y_onehot <- model.matrix(~ 0 + factor(Y_a))
    Y_tilde <- scale(Y_onehot, center = TRUE, scale = FALSE) # n_a x C
    KY <- tcrossprod(Y_tilde) / n_a # n_a x n_a

    # Precompute (I + gamma_y * KY) once  -- applied to Xa_u in each RHS update
    IKY <- diag(n_a) + gamma_y * KY # n_a x n_a

    #  Laplacians + B matrices
    Xa_raw <- lapply(X_list, function(X) X[anchor_idx, , drop = FALSE])
    col_ok <- lapply(Xa_raw, function(X) which(!apply(X, 2, function(v) all(is.na(v)))))
    Xa_list <- lapply(Xa_raw, na_impute_median)

    # Build Laplacians only if not precomputed
    if (is.null(L_list_precomp) || is.null(Omega_list_precomp)) {
        if (verbose) message("  Building Laplacians...\n")
        L_list <- if (!is.null(L_list_precomp)) {
            L_list_precomp
        } else {
            lapply(seq_len(V), function(v) {
                Xref <- if (!is.null(X_list_full)) X_list_full[[v]] else X_list[[v]]
                build_laplacian(Xref, k = k_nn)
            })
        }
        Omega_list_local <- if (!is.null(Omega_list_precomp)) {
            Omega_list_precomp
        } else {
            lapply(seq_len(V), function(v) {
                Xref <- na_impute_median(
                    if (!is.null(X_list_full)) X_list_full[[v]] else X_list[[v]][anchor_idx, , drop = FALSE]
                )
                n_ref <- nrow(Xref)
                L <- L_list[[v]]
                if (nrow(L) == n_ref) {
                    as.matrix(crossprod(Xref, as.matrix(L %*% Xref))) / n_ref
                } else {
                    La <- as.matrix(L[seq_len(n_ref), seq_len(n_ref)])
                    as.matrix(crossprod(Xref, La %*% Xref)) / n_ref
                }
            })
        }
    } else {
        L_list <- L_list_precomp
        Omega_list_local <- Omega_list_precomp
    }

    if (verbose) message("  Computing B matrices...\n")
    B_list <- lapply(seq_len(V), function(v) {
        XtX_v <- if (!is.null(XtX_list_precomp)) XtX_list_precomp[[v]] else NULL
        B <- compute_B(X_list[[v]], lambdas[v], Omega_list_local[[v]], XtX_precomp = XtX_v)
        B + diag(1e-8 * max(diag(B)), nrow(B))
    })

    p_eff <- vapply(B_list, nrow, integer(1L))

    # Constant across all ALS restarts — precompute once here, not inside als_run()
    # IKY %*% Xa_list[[u]] does not depend on W; B does not change across restarts.
    IKY_Xa <- lapply(seq_len(V), function(u) IKY %*% Xa_list[[u]])
    B_chol <- lapply(seq_len(V), function(v) {
        tryCatch(chol(B_list[[v]]), error = function(e) NULL)
    })

    #  B-orthogonalization
    b_orth <- function(W, B) {
        WBW <- crossprod(W, B %*% W)
        WBW <- (WBW + t(WBW)) / 2 # symmetrize
        R <- tryCatch(chol(WBW), error = function(e) NULL)
        if (!is.null(R)) {
            return(t(backsolve(R, t(W))))
        }
        # Cholesky failed: use eigendecomposition of W^T B W for correct B-normalization
        ev <- eigen(WBW, symmetric = TRUE)
        # Always use top K eigenvectors to keep W dimensionality = K
        K_use <- ncol(W)
        D_inv_sqrt <- 1 / sqrt(pmax(ev$values[seq_len(K_use)], 1e-14))
        W %*% ev$vectors[, seq_len(K_use), drop = FALSE] %*% diag(D_inv_sqrt, K_use)
    }

    #  Soft-threshold (L1)
    soft_thresh <- function(W, a) {
        if (a <= 0) {
            return(W)
        }
        sign(W) * pmax(abs(W) - a, 0)
    }

    #  Random B-orthogonalized init
    warm_start <- function() {
        lapply(seq_len(V), function(v) {
            W0 <- matrix(rnorm(p_eff[v] * K), p_eff[v], K)
            b_orth(W0, B_list[[v]])
        })
    }

    #  ALS iteration
    als_run <- function(W_init) {
        W_list <- W_init
        obj_prev <- Inf
        obj_trace <- numeric(max_iter)

        for (iter in seq_len(max_iter)) {
            for (v in seq_len(V)) {
                others <- setdiff(seq_len(V), v)
                # RHS_v = Xa_v^T * sum_{u!=v} (I + gamma_y*KY) * Xa_u * W_u
                RHS <- Reduce("+", lapply(others, function(u) {
                    crossprod(Xa_list[[v]], IKY_Xa[[u]] %*% W_list[[u]])
                }))
                W_new <- if (!is.null(B_chol[[v]])) {
                    # Fast path: O(p^2 K) backsolve using precomputed Cholesky
                    backsolve(B_chol[[v]], backsolve(B_chol[[v]], RHS, transpose = TRUE))
                } else {
                    tryCatch(
                        solve(B_list[[v]], RHS),
                        error = function(e) {
                            sv <- svd(B_list[[v]])
                            sv$v %*% diag(1 / pmax(sv$d, 1e-10), length(sv$d)) %*% t(sv$u) %*% RHS
                        }
                    )
                }
                W_new[!is.finite(W_new)] <- 0
                W_new <- soft_thresh(W_new, alpha) # L1 sparsity
                W_list[[v]] <- b_orth(W_new, B_list[[v]])
            }

            # Objective: sum of pairwise Frobenius distances in supervised metric
            # ||Za_v - Za_u||^2 under label-augmented geometry
            obj <- 0
            for (v in seq_len(V - 1L)) {
                for (u in seq(v + 1L, V)) {
                    Zv <- Xa_list[[v]] %*% W_list[[v]]
                    Zu <- Xa_list[[u]] %*% W_list[[u]]
                    diff_vu <- Zv - Zu
                    # Standard alignment
                    obj <- obj + sum(diff_vu^2)
                    # Label-weighted penalty: same-class subjects penalized more for disagreement
                    if (gamma_y > 0) {
                        obj <- obj + gamma_y * sum(diff_vu * (KY %*% diff_vu))
                    }
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

        list(W_list = W_list, objective = obj, obj_trace = obj_trace)
    }

    #  Multiple restarts
    best <- NULL
    for (r in seq_len(n_restarts)) {
        if (verbose) message(sprintf("  Restart %d/%d...\n", r, n_restarts))
        result <- als_run(warm_start())
        if (is.null(best) || result$objective < best$objective) best <- result
    }

    W_list <- best$W_list
    Z_list <- lapply(seq_len(V), function(v) Xa_list[[v]] %*% W_list[[v]])

    svals <- if (V == 2L) {
        svd(crossprod(Z_list[[1]], Z_list[[2]]))$d[seq_len(K)]
    } else {
        .svds_safe(do.call(cbind, Z_list), k = K)$d
    }

    if (verbose) {
        message(sprintf(
            "  Converged. Objective=%.6f | Top svals: %s\n",
            best$objective,
            paste(round(svals[seq_len(min(5L, K))], 4), collapse = ", ")
        ))
    }

    list(
        W_list = W_list,
        Z_list = Z_list,
        Za_list = Z_list,
        Xa_list = Xa_list,
        col_ok_list = col_ok,
        Y_anchor = Y_tilde,
        KY = KY,
        singular_values = svals,
        B_list = B_list,
        L_list = L_list,
        Omega_list = Omega_list_local,
        anchor_idx = anchor_idx,
        K = K, V = V,
        lambdas = lambdas,
        gamma_y = gamma_y,
        alpha = alpha,
        k_nn = k_nn,
        n = n,
        p_v = p_eff,
        objective = best$objective,
        obj_trace = best$obj_trace
    )
}

#' VIP scores for a fitted ALS model
#'
#' @param fit output of woven_als
#' @return list of V VIP vectors, one per modality
#' @examples
#' set.seed(1)
#' n <- 20
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' X3 <- matrix(rnorm(n * 3), n, 3)
#' Y <- rep(1:2, each = n / 2)
#' fit <- woven_als(list(X1, X2, X3), anchor_idx = seq_len(14L), Y = Y, K = K)
#' woven_als_vip(fit)
#' @export
woven_als_vip <- function(fit) {
    svals <- fit$singular_values
    weights <- svals^2 / sum(svals^2)
    lapply(seq_len(fit$V), function(v) {
        W <- fit$W_list[[v]]
        sqrt(fit$p_v[v] * rowSums(sweep(W^2, 2, weights, "*")))
    })
}
