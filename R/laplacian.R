# laplacian.R  -- Sparse k-NN graph Laplacian for WOVEN
#
# Builds a combinatorial Laplacian from RBF-weighted k-NN graph.
# Uses RANN::nn2() for O(n log n) k-NN instead of O(n^2) distance matrix.

#' Build sparse k-NN graph Laplacian
#'
#' Block-missing (all-NA) rows are excluded from k-NN entirely -- no imputation
#' is performed for them. Only within-row feature NAs (partial missingness) are
#' median-imputed before distance computation. The returned matrix is always
#' n_full x n_full; rows/cols for block-missing subjects are structural zeros.
#'
#' @param X numeric matrix, n x p.
#' @param k integer, number of nearest neighbors (default 10)
#' @param sigma numeric, RBF bandwidth. NULL = median heuristic over observed edges.
#' @return sparse n_full x n_full symmetric combinatorial Laplacian (dgCMatrix).
#' @keywords internal
build_laplacian <- function(X, k = 10L, sigma = NULL) {
    n_full <- nrow(X)

    # Identify block-missing rows (entire row NA) -- excluded, never imputed
    all_na_row <- rowSums(!is.na(X)) == 0L
    obs_idx <- which(!all_na_row)
    n_obs <- length(obs_idx)

    if (n_obs == 0L) stop("No observed rows in X -- cannot build Laplacian.")
    k <- min(k, n_obs - 1L)

    # Impute only within observed rows (feature-level NAs only)
    X_obs <- na_impute_median(X[obs_idx, , drop = FALSE])

    # k-NN via kd-tree on observed subjects only
    nn   <- RANN::nn2(X_obs, k = k + 1L)
    idx  <- nn$nn.idx[,  -1L, drop = FALSE]   # n_obs x k
    dsts <- nn$nn.dists[, -1L, drop = FALSE]  # n_obs x k

    # RBF bandwidth: median heuristic over observed edges
    if (is.null(sigma)) {
        sigma <- stats::median(dsts[dsts > 0])
        if (!is.finite(sigma) || sigma == 0) sigma <- 1.0
    }

    # Build affinity in obs-space, embed into full n_full x n_full
    from_f <- obs_idx[rep(seq_len(n_obs), times = k)]
    to_f   <- obs_idx[as.vector(idx)]
    w      <- exp(-(as.vector(dsts)^2) / sigma^2)

    W <- Matrix::sparseMatrix(
        i    = c(from_f, to_f),
        j    = c(to_f, from_f),
        x    = rep(w / 2, 2),
        dims = c(n_full, n_full),
        repr = "C"
    )
    W@x <- pmin(W@x, 1.0)

    d <- Matrix::rowSums(W)
    Matrix::Diagonal(x = d) - W
}
