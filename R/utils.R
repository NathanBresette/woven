# utils.R  -- Internal matrix utilities for WOVEN

# RSpectra is Suggests: use it when available, fall back to base R silently.
# Both helpers return the same structure as the RSpectra originals.
.svds_safe <- function(X, k) {
    if (requireNamespace("RSpectra", quietly = TRUE)) {
        return(RSpectra::svds(X, k = k))
    }
    k <- min(k, nrow(X), ncol(X))
    sv <- svd(X, nu = k, nv = k)
    list(u = sv$u[, seq_len(k), drop = FALSE],
         d = sv$d[seq_len(k)],
         v = sv$v[, seq_len(k), drop = FALSE])
}

.eigs_sym_safe <- function(B, k) {
    if (requireNamespace("RSpectra", quietly = TRUE)) {
        return(RSpectra::eigs_sym(B, k = k, which = "LM"))
    }
    eig <- eigen(B, symmetric = TRUE)
    list(values  = eig$values[seq_len(k)],
         vectors = eig$vectors[, seq_len(k), drop = FALSE])
}

#' Compute regularized B matrix: B_v = X^T X + lambda * Omega
#'
#' @param X numeric matrix, n x p (all samples, full modality)
#' @param lambda non-negative regularization weight
#' @param Omega sparse p x p Laplacian (from build_laplacian, feature space)
#'   OR NULL to use sample-space Laplacian (when p >> n, use kernel trick)
#' @return dense p x p symmetric positive definite matrix
#' @keywords internal
compute_B <- function(X, lambda, Omega, XtX_precomp = NULL) {
    XtX <- if (!is.null(XtX_precomp)) {
        XtX_precomp
    } else {
        Xc <- na_impute_median(X)
        crossprod(Xc)
    }
    if (lambda == 0 || is.null(Omega)) {
        return(as.matrix(XtX))
    }
    as.matrix(XtX + lambda * Omega)
}

# Internal: drop all-NA rows, impute feature-level NAs with column median
na_impute_median <- function(X) {
    # Drop block-missing rows (all NA)
    all_na_row <- apply(X, 1, function(r) all(is.na(r)))
    Xc <- X[!all_na_row, , drop = FALSE]
    # Drop all-NA columns (can't impute)
    all_na_col <- apply(Xc, 2, function(v) all(is.na(v)))
    Xc <- Xc[, !all_na_col, drop = FALSE]
    # Impute remaining feature-level NAs with column median
    col_med <- apply(Xc, 2, function(v) median(v, na.rm = TRUE))
    for (j in seq_len(ncol(Xc))) {
        na_j <- is.na(Xc[, j])
        if (any(na_j)) Xc[na_j, j] <- col_med[j]
    }
    # Final safety: replace any remaining non-finite values with 0
    Xc[!is.finite(Xc)] <- 0
    Xc
}

#' Symmetric matrix square root via eigendecomposition
#'
#' Returns B^\{1/2\} and B^\{-1/2\} for a symmetric PD matrix.
#' When p > rank_thresh, uses truncated eigendecomposition via RSpectra  --
#' B = X^T M X has rank at most n, so only n eigenvectors are non-trivial.
#'
#' @param B symmetric PD matrix (p x p)
#' @param tol relative eigenvalue floor
#' @param n_rank known upper bound on rank of B (e.g. number of samples).
#'   If provided and p > n_rank, uses truncated eig with k = min(n_rank, p-1).
#' @return list with $sqrt and $inv_sqrt (both p x p)
#' @keywords internal
mat_sqrt <- function(B, tol = 1e-10, n_rank = NULL) {
    p <- nrow(B)

    # Truncated path: when p is large but rank is bounded by n_rank
    use_trunc <- !is.null(n_rank) && p > 500L && n_rank < p
    if (use_trunc) {
        k <- min(n_rank, p - 1L)
        eig <- .eigs_sym_safe(B, k = k)
        vals <- eig$values
        vecs <- eig$vectors
    } else {
        eig <- eigen(B, symmetric = TRUE)
        vals <- eig$values
        vecs <- eig$vectors
    }

    floor_val <- tol * max(abs(vals))
    vals_safe <- pmax(vals, floor_val)
    sqrt_vals <- sqrt(vals_safe)
    inv_sqrt_vals <- 1 / sqrt_vals

    # Reconstruct p x p matrices from potentially truncated eigenvectors
    list(
        sqrt     = vecs %*% diag(sqrt_vals, length(sqrt_vals)) %*% t(vecs),
        inv_sqrt = vecs %*% diag(inv_sqrt_vals, length(inv_sqrt_vals)) %*% t(vecs)
    )
}

#' Kernel trick: compute B in sample space when p >> n_a
#'
#' When p_v >> n_a, working in feature space costs O(p^2).
#' Instead compute the n_a x n_a kernel K = X X^T and solve in that space.
#' Used internally by solver_v2 when p > 5 * n_a.
#'
#' @param Xa anchor samples, n_a x p
#' @param lambda regularization
#' @param L_sample n x n sample-space Laplacian (for full data n, not just anchors)
#' @param anchor_idx integer vector of anchor row indices into full data
#' @return n_a x n_a kernel matrix K_reg = Xa Xa^T + lambda * L_a
#'   where L_a is the anchor submatrix of the sample Laplacian
#' @keywords internal
compute_B_kernel <- function(Xa, lambda, L_sample, anchor_idx) {
    K <- tcrossprod(Xa) # n_a x n_a
    if (lambda == 0) {
        return(K)
    }
    L_a <- as.matrix(L_sample[anchor_idx, anchor_idx])
    K + lambda * L_a
}

#' Validate inputs before solve
#'
#' @param X_list list of modality matrices (n_v x p_v each)
#' @param anchor_idx integer vector of anchor indices
#' @param K number of latent dimensions
#' @return Invisibly returns TRUE if all checks pass; stops with an error otherwise.
#' @keywords internal
check_woven_inputs <- function(X_list, anchor_idx, K) {
    V <- length(X_list)
    if (V < 2) stop("WOVEN requires at least 2 modalities.")

    n_a <- length(anchor_idx)
    if (n_a < K) {
        stop(sprintf(
            "Anchor set (n_a=%d) must be >= K=%d. Increase anchor set or reduce K.", n_a, K
        ))
    }
    if (n_a < 5 * K) {
        warning(sprintf(
            "Anchor set (n_a=%d) is small relative to K=%d (recommend n_a >= 5K). Results may be unstable.", n_a, K
        ))
    }

    ns <- vapply(X_list, nrow, integer(1L))
    if (length(unique(ns)) > 1) stop("All modality matrices must have the same number of rows.")

    invisible(TRUE)
}

# Suppress R CMD check notes for variables used inside closures/lapply
utils::globalVariables(c(
    "centroids", "Z_ref_knn", "uniform_p", # woven_predict closures
    "W_list", "B_list", "Xa_list", "KY", "IKY", # ALS closures
    "obj_prev", "obj_trace", # ALS iteration
    "b_orth", "soft_thresh" # ALS local fns seen by nested closures
))
