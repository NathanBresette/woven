# woven.R  -- Top-level user API for WOVEN
#
# Typical workflow:
#   fit  <- woven(X_list, Y, anchor_idx, K = 5)  # fit; fit$Z has scores for all n subjects
#   plot(fit, labels = Y)                          # plot latent space
#   scores  <- woven_scores(fit, X_list_new)       # latent scores for new subjects
#   pred    <- woven_predict(fit, X_list_new)      # class predictions for new subjects

#' Fit a supervised WOVEN model
#'
#' Learns a shared supervised latent space across V omics modalities, handling
#' block-missing data via anchor-restricted alignment and Nystrm projection.
#' Labels Y are required  -- WOVEN is a supervised method (cf. DIABLO).
#'
#' For V=2, uses the closed-form supervised CCA solver (fast, exact).
#' For V>=3, uses the ALS solver with label-kernel supervision.
#'
#' @param X_list list of V numeric matrices, each n x p_v.
#'   Subjects missing an entire modality should have that matrix row set to NA.
#' @param Y integer or factor vector of length n  -- class labels for all subjects.
#'   Only anchor subjects' labels enter the supervised objective.
#' @param anchor_idx integer vector  -- indices of fully-observed subjects
#'   (observed in all V modalities). Must have length >= K. If NULL (default),
#'   anchors are detected automatically as subjects with no block-missing modalities.
#' @param K integer  -- number of latent dimensions (default 5)
#' @param lambdas numeric scalar or length-V vector  -- Laplacian regularization
#'   strength per modality (default 0.1 for all)
#' @param gamma_y numeric >= 0  -- supervision strength. 0 = unsupervised CCA.
#'   Default 1.0 (equal weight to cross-modal alignment and label alignment).
#'   Tune via cross-validation on anchor set if labels are noisy.
#' @param alpha numeric >= 0 or length-2 vector [alpha1, alpha2]  -- L1 sparsity
#'   on projection matrices W (default 0 = dense). Analogous to DIABLO's keepX.
#'   Set to ~0.01-0.1 for interpretable sparse feature loadings.
#' @param k_nn integer  -- k-nearest-neighbors for Laplacian graph (default 15)
#' @param X_list_full optional list of complete unmasked matrices for Laplacian
#'   construction (use when X_list has many NA rows from block-missingness)
#' @param n_restarts integer  -- ALS random restarts, V>=3 only (default 5)
#' @param max_iter integer  -- ALS max iterations, V>=3 only (default 200)
#' @param verbose logical  -- print progress (default TRUE)
#'
#' @return object of class "woven" with:
#'   \describe{
#'     \item{$Z}{n x K matrix of consensus latent scores for ALL n subjects
#'       (anchors and block-missing). The primary output for downstream analysis.}
#'     \item{$W_list}{list of V projection matrices, each p_v x K}
#'     \item{$Z_anchors}{list of V anchor latent score matrices, each n_a x K}
#'     \item{$singular_values}{K supervised canonical correlations}
#'     \item{$anchor_idx}{indices of anchor (fully-observed) subjects}
#'     \item{$Y_levels}{class label levels used during fitting}
#'     \item{$K, $V, $n}{dimensions}
#'     \item{$lambdas, $gamma_y, $alpha}{hyperparameters}
#'   }
#'
#' @examples
#' set.seed(1)
#' n <- 60; p1 <- 20; p2 <- 15; K <- 2
#' Y <- rep(1:2, each = n / 2)
#' X1 <- matrix(rnorm(n * p1), n, p1)
#' X2 <- matrix(rnorm(n * p2), n, p2)
#' # 30% block missingness; enforce >= 1 view per subject
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' # anchor_idx auto-detected from NA pattern -- no need to specify
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' dim(fit$Z) # 60 x 2 -- all subjects scored
#'
#' @seealso [woven_scores()], [woven_predict()], [woven_all_metrics()]
#' @export
woven <- function(X_list, Y, anchor_idx = NULL,
                  K = 5L,
                  lambdas = 0.1,
                  gamma_y = 1.0,
                  alpha = 0,
                  k_nn = 15L,
                  X_list_full = NULL,
                  n_restarts = 5L,
                  max_iter = 200L,
                  verbose = TRUE) {
    #  Input validation
    if (!is.list(X_list)) stop("X_list must be a list of matrices.")
    V <- length(X_list)
    if (V < 2L) stop("X_list must contain at least 2 modalities.")
    n <- nrow(X_list[[1]])
    if (length(Y) != n) stop("length(Y) must equal nrow(X_list[[1]]).")

    #  Auto-detect anchors (subjects with all modalities observed)
    if (is.null(anchor_idx)) {
        block_missing <- vapply(X_list, function(X) {
            apply(X, 1L, function(row) all(is.na(row)))
        }, logical(n))
        anchor_idx <- which(rowSums(block_missing) == 0L)
        if (verbose) message(sprintf(
            "Auto-detected %d anchor subjects (%.0f%% of n=%d with all modalities observed).",
            length(anchor_idx), 100 * length(anchor_idx) / n, n
        ))
        if (length(anchor_idx) == 0L)
            stop("No fully-observed subjects found. Every subject is block-missing in at least one modality.")
    }

    if (length(anchor_idx) < K) {
        stop(sprintf("Need at least K=%d anchor subjects; got %d.", K, length(anchor_idx)))
    }

    Y <- as.integer(as.factor(Y)) # canonical integer labels

    if (length(lambdas) == 1L) lambdas <- rep(lambdas, V)
    stopifnot(length(lambdas) == V)

    alpha1 <- if (length(alpha) == 2L) alpha[1] else alpha
    alpha2 <- if (length(alpha) == 2L) alpha[2] else alpha

    #  Dispatch
    if (V == 2L) {
        raw <- woven_v2(
            X1 = X_list[[1]], X2 = X_list[[2]],
            anchor_idx = anchor_idx,
            Y = Y,
            K = K,
            lambda1 = lambdas[1], lambda2 = lambdas[2],
            gamma_y = gamma_y,
            alpha1 = alpha1, alpha2 = alpha2,
            k_nn = k_nn,
            X1_full = X_list_full[[1]],
            X2_full = X_list_full[[2]]
        )
        W_list <- list(raw$W1, raw$W2)
        Z_anchors <- list(raw$Z1, raw$Z2)
        svals <- raw$singular_values
        fit_v2 <- raw
        fit_als <- NULL
    } else {
        raw <- woven_als(
            X_list      = X_list,
            anchor_idx  = anchor_idx,
            Y           = Y,
            K           = K,
            lambdas     = lambdas,
            gamma_y     = gamma_y,
            alpha       = alpha,
            k_nn        = k_nn,
            max_iter    = max_iter,
            n_restarts  = n_restarts,
            X_list_full = X_list_full,
            verbose     = verbose
        )
        W_list <- raw$W_list
        Z_anchors <- raw$Z_list
        svals <- raw$singular_values
        fit_v2 <- NULL
        fit_als <- raw
    }

    #  Compute consensus Z for all n subjects (direct projection via W)
    Z_all <- matrix(NA_real_, n, K)
    for (i in seq_len(n)) {
        zv <- vector("list", V)
        n_obs <- 0L
        for (v in seq_len(V)) {
            xi <- X_list[[v]][i, , drop = TRUE]
            if (!all(is.na(xi))) {
                xi[is.na(xi)] <- 0
                zv[[v]] <- as.vector(xi %*% W_list[[v]])
                n_obs <- n_obs + 1L
            }
        }
        if (n_obs > 0L) {
            Z_all[i, ] <- Reduce("+", Filter(Negate(is.null), zv)) / n_obs
        }
    }

    #  Return unified object
    structure(
        list(
            Z               = Z_all,
            W_list          = W_list,
            Z_anchors       = Z_anchors,
            singular_values = svals,
            anchor_idx      = anchor_idx,
            Y_anchor        = Y[anchor_idx],
            Y_levels        = sort(unique(Y)),
            K               = K,
            V               = V,
            n               = n,
            lambdas         = lambdas,
            gamma_y         = gamma_y,
            alpha           = alpha,
            k_nn            = k_nn,
            fit_v2          = fit_v2,
            fit_als         = fit_als
        ),
        class = "woven"
    )
}

#' Print method for WOVEN fit
#' @param x a woven object from [woven()]
#' @param ... further arguments (unused)
#' @return Invisibly returns the woven object \code{x}.
#' @examples
#' set.seed(1)
#' n <- 20
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' Y <- rep(1:2, each = n / 2)
#' miss <- matrix(FALSE, n, 2)
#' miss[c(15, 17, 19), 1] <- TRUE
#' miss[c(16, 18, 20), 2] <- TRUE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' anchor_idx <- which(rowSums(miss) == 0)
#' fit <- woven(list(X1, X2), Y = Y, anchor_idx = anchor_idx, K = K)
#' print(fit)
#' @export
print.woven <- function(x, ...) {
    n_anchor <- length(x$anchor_idx)
    n_scored <- sum(!is.na(x$Z[, 1L]))
    cat("WOVEN fit\n")
    cat(sprintf(
        "  Modalities : %d    Subjects: %d    Dimensions: %d\n",
        x$V, x$n, x$K
    ))
    cat(sprintf(
        "  Anchors    : %d (%.0f%%)    Scored: %d (%.0f%%)\n",
        n_anchor, 100 * n_anchor / x$n,
        n_scored, 100 * n_scored / x$n
    ))
    cat(sprintf(
        "  gamma_y    : %.2f    lambda: %s    k_nn: %d\n",
        x$gamma_y,
        paste(round(x$lambdas, 3), collapse = "/"),
        x$k_nn
    ))
    cat(sprintf(
        "  Singular values: %s\n",
        paste(round(x$singular_values[seq_len(min(5L, x$K))], 3),
            collapse = ", "
        )
    ))
    cat("  Access: fit$Z (all scores), fit$W_list (loadings)\n")
    cat("  Functions: woven_scores(), woven_predict(), plot(fit, labels=Y)\n")
    invisible(x)
}

#' Plot the WOVEN latent space
#'
#' Plots the first two latent dimensions from `fit$Z`, colored by group label.
#'
#' @param x a woven object from [woven()]
#' @param labels integer or factor of length n for coloring points.
#'   If NULL, all points are plotted in the same color.
#' @param dims integer vector of length 2: which latent dimensions to plot
#'   (default c(1, 2))
#' @param highlight_anchors logical: whether to mark anchor subjects with
#'   filled points and block-missing subjects with open circles (default TRUE)
#' @param ... further arguments passed to [plot()]
#' @return invisibly returns x
#' @importFrom graphics legend plot
#' @export
plot.woven <- function(x, labels = NULL, dims = c(1L, 2L),
                       highlight_anchors = TRUE, ...) {
    Z <- x$Z
    if (is.null(Z)) stop("fit$Z is NULL. Refit with woven().")
    d1 <- dims[1L]
    d2 <- dims[2L]
    if (d1 > x$K || d2 > x$K) {
        stop(sprintf("dims out of range: fit has K=%d dimensions.", x$K))
    }

    pal <- c(
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#999999"
    )

    if (is.null(labels)) {
        cols <- rep(pal[1L], x$n)
    } else {
        lbl <- as.integer(as.factor(labels))
        cols <- pal[(lbl - 1L) %% length(pal) + 1L]
    }

    pch_vec <- if (highlight_anchors) {
        ifelse(seq_len(x$n) %in% x$anchor_idx, 19L, 1L)
    } else {
        rep(19L, x$n)
    }

    plot(Z[, d1], Z[, d2],
        col  = cols,
        pch  = pch_vec,
        xlab = sprintf("WOVEN dim %d", d1),
        ylab = sprintf("WOVEN dim %d", d2),
        ...
    )

    if (highlight_anchors) {
        legend("topright",
            bty = "n",
            legend = c("anchor (complete)", "projected (partial)"),
            pch = c(19L, 1L),
            col = "black",
            cex = 0.8
        )
    }
    invisible(x)
}

#' Extract latent scores for new subjects
#'
#' Projects new subjects into the trained WOVEN latent space and returns an
#' n_new x K score matrix. Uses direct linear projection (x %*% W_v) for each
#' available modality, then averages across observed views. No Nystrm kernel
#' required -- suitable for large new cohorts.
#'
#' For class predictions on new subjects, use [woven_predict()] instead.
#'
#' @param fit woven object from [woven()]
#' @param X_list_new list of V matrices (n_new x p_v). Set entire rows to NA
#'   for subjects missing that modality block. Every subject must have at least
#'   one non-missing view.
#' @return numeric matrix n_new x K of consensus latent scores. Subjects with
#'   no observed data in any view receive a row of NA.
#' @examples
#' \dontrun{
#' # Score a held-out test set
#' Z_test <- woven_scores(fit, X_list_test)
#' }
#' @seealso [woven_predict()] for class predictions, [woven()] for model fitting
#' @export
woven_scores <- function(fit, X_list_new) {
    stopifnot(inherits(fit, "woven"))
    stopifnot(is.list(X_list_new), length(X_list_new) == fit$V)

    n_new <- nrow(X_list_new[[1]])
    K <- fit$K
    V <- fit$V

    Z <- matrix(NA_real_, n_new, K)
    for (i in seq_len(n_new)) {
        zv <- vector("list", V)
        n_obs <- 0L
        for (v in seq_len(V)) {
            xi <- X_list_new[[v]][i, , drop = TRUE]
            if (!all(is.na(xi))) {
                xi[is.na(xi)] <- 0
                zv[[v]] <- as.vector(xi %*% fit$W_list[[v]])
                n_obs <- n_obs + 1L
            }
        }
        if (n_obs > 0L) {
            Z[i, ] <- Reduce("+", Filter(Negate(is.null), zv)) / n_obs
        }
    }
    Z
}

#' Predict class probabilities for new subjects
#'
#' Projects new subjects into the WOVEN latent space and returns soft class
#' assignments using a nearest-centroid classifier in latent space. Works for
#' complete subjects (direct projection) and block-missing subjects (Nystrm).
#'
#' @param fit woven object from [woven()]
#' @param X_list_new list of V matrices for new subjects (n_new x p_v each).
#'   Block-missing subjects should have their modality rows set to NA.
#' @param method "centroid" (default)  -- nearest centroid in latent space.
#'   "knn"  -- k-NN vote using anchor subjects as the reference set.
#' @param k_pred integer  -- number of neighbors for knn method (default 5)
#'
#' @return data.frame with n_new rows:
#'   $predicted_class  integer predicted class label
#'   $confidence       probability of predicted class (0-1)
#'   One column per class level with soft probabilities
#' @examples
#' set.seed(1)
#' n <- 40
#' K <- 2L
#' X1 <- matrix(rnorm(n * 5), n, 5)
#' X2 <- matrix(rnorm(n * 4), n, 4)
#' Y <- rep(1:2, each = n / 2)
#' miss <- matrix(FALSE, n, 2)
#' miss[c(31, 33, 35), 1] <- TRUE
#' miss[c(32, 34, 36), 2] <- TRUE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' anchor_idx <- which(rowSums(miss) == 0)
#' fit <- woven(list(X1, X2), Y = Y, anchor_idx = anchor_idx, K = K)
#' pred <- woven_predict(fit, list(X1[1:5, ], X2[1:5, ]))
#' pred$predicted_class
#' @export
woven_predict <- function(fit, X_list_new, method = "centroid", k_pred = 5L) {
    stopifnot(inherits(fit, "woven"))
    stopifnot(is.list(X_list_new), length(X_list_new) == fit$V)

    #  Project new subjects via Nystrm (all are out-of-sample)
    # woven_project() expects the original np matrices and splits by anchor_idx.
    # For genuinely new subjects, we apply Nystrm directly to all rows.
    n_new <- nrow(X_list_new[[1]])

    if (fit$V == 2L) {
        raw <- fit$fit_v2
        col1 <- raw$col_ok1
        col2 <- raw$col_ok2
        Z_new <- matrix(NA_real_, n_new, fit$K)
        for (i in seq_len(n_new)) {
            x1i <- X_list_new[[1]][i, col1, drop = TRUE]
            x2i <- X_list_new[[2]][i, col2, drop = TRUE]
            obs1 <- !all(is.na(x1i))
            obs2 <- !all(is.na(x2i))
            x1i[is.na(x1i)] <- 0
            x2i[is.na(x2i)] <- 0
            z1 <- if (obs1) {
                d1 <- sqrt(rowSums(sweep(raw$Xa1, 2, x1i)^2))
                s1 <- median(d1[d1 > 0])
                if (!is.finite(s1) || s1 == 0) s1 <- 1
                k1 <- exp(-d1^2 / s1^2)
                k1 <- k1 / sum(k1)
                crossprod(k1, raw$Za1)
            } else {
                NULL
            }
            z2 <- if (obs2) {
                d2 <- sqrt(rowSums(sweep(raw$Xa2, 2, x2i)^2))
                s2 <- median(d2[d2 > 0])
                if (!is.finite(s2) || s2 == 0) s2 <- 1
                k2 <- exp(-d2^2 / s2^2)
                k2 <- k2 / sum(k2)
                crossprod(k2, raw$Za2)
            } else {
                NULL
            }
            if (!is.null(z1) && !is.null(z2)) {
                p1e <- length(col1)
                p2e <- length(col2)
                Z_new[i, ] <- (p1e * z1 + p2e * z2) / (p1e + p2e)
            } else if (!is.null(z1)) {
                Z_new[i, ] <- z1
            } else if (!is.null(z2)) Z_new[i, ] <- z2
        }
    } else {
        Z_views <- lapply(seq_len(fit$V), function(v) {
            .nystrom_single_view(
                X_list_new[[v]], fit$fit_als$Xa_list[[v]],
                fit$Z_anchors[[v]], fit$W_list[[v]],
                fit$fit_als$col_ok_list[[v]]
            )
        })
        obs <- vapply(X_list_new, function(X) {
            !apply(X, 1, function(r) all(is.na(r)))
        }, logical(n_new))
        Z_new <- matrix(0, n_new, fit$K)
        w_sum <- numeric(n_new)
        for (v in seq_len(fit$V)) {
            ok <- if (is.matrix(obs)) obs[, v] else obs[[v]]
            Z_new[ok, ] <- Z_new[ok, ] + Z_views[[v]][ok, , drop = FALSE]
            w_sum[ok] <- w_sum[ok] + 1
        }
        w_sum[w_sum == 0] <- 1
        Z_new <- Z_new / w_sum
    }

    #  Classify in latent space
    Y_a <- fit$Y_anchor
    levels_Y <- fit$Y_levels
    C <- length(levels_Y)
    n_new <- nrow(Z_new)

    uniform_p <- rep(1 / C, C) # fallback for unscored subjects

    classify_row <- function(z) {
        if (any(is.na(z))) {
            return(uniform_p)
        }
        if (method == "centroid") {
            dists <- sqrt(rowSums(sweep(centroids, 2, z)^2))
            dists <- pmax(dists, 1e-10)
            w <- 1 / dists^2
            w / sum(w)
        } else {
            dists <- sqrt(rowSums(sweep(Z_ref_knn, 2, z)^2))
            nn_idx <- order(dists)[seq_len(min(k_pred, length(dists)))]
            nn_y <- Y_a[nn_idx]
            tabulate(match(nn_y, levels_Y), nbins = C) / length(nn_idx)
        }
    }

    Z_ref <- Reduce("+", fit$Z_anchors) / fit$V

    if (method == "centroid") {
        centroids <- do.call(rbind, lapply(levels_Y, function(g) {
            idx <- which(Y_a == g)
            if (length(idx) == 0L) {
                return(rep(NA_real_, fit$K))
            }
            colMeans(Z_ref[idx, , drop = FALSE])
        }))
        Z_ref_knn <- NULL # not used
    } else if (method == "knn") {
        centroids <- NULL
        Z_ref_knn <- Z_ref
    } else {
        stop("method must be 'centroid' or 'knn'")
    }

    probs <- do.call(rbind, lapply(seq_len(n_new), function(i) classify_row(Z_new[i, ])))

    colnames(probs) <- paste0("p_class", levels_Y)
    pred_idx <- apply(probs, 1, which.max)

    data.frame(
        predicted_class = levels_Y[pred_idx],
        confidence = probs[cbind(seq_len(n_new), pred_idx)],
        as.data.frame(probs),
        stringsAsFactors = FALSE
    )
}

# Internal helper: single-view Nystrm projection for V>=3 predict path
.nystrom_single_view <- function(X_new, Xa, Za, W, col_ok) {
    n_new <- nrow(X_new)
    Z_out <- matrix(NA_real_, n_new, ncol(Za))

    for (i in seq_len(n_new)) {
        xi <- X_new[i, col_ok, drop = TRUE]
        if (all(is.na(xi))) next
        xi[is.na(xi)] <- 0
        dists <- sqrt(rowSums(sweep(Xa, 2, xi)^2))
        sigma <- median(dists[dists > 0])
        if (!is.finite(sigma) || sigma == 0) sigma <- 1
        kw <- exp(-dists^2 / sigma^2)
        kw <- kw / sum(kw)
        Z_out[i, ] <- crossprod(kw, Za)
    }
    Z_out
}
