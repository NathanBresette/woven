# woven.R  -- Top-level user API for WOVEN
#
# Typical workflow:
#   precomp <- woven_precompute(X_list)             # build Laplacians once (optional)
#   fit  <- woven(X_list, Y, K = 5, precomp = precomp)
#   plot(fit, labels = Y)                           # plot latent space
#   scores  <- woven_scores(fit, X_list_new)        # latent scores for new subjects
#   pred    <- woven_predict(fit, X_list_new)       # class predictions for new subjects

#' Pre-compute Laplacian graphs for reuse across multiple woven() calls
#'
#' Builds k-NN RBF Laplacians for each modality from the observed data.
#' Pass the result to \code{woven(..., precomp = precomp)} to avoid
#' rebuilding the graph on every call -- useful for hyperparameter search,
#' cross-validation, or sensitivity analysis.
#'
#' @param X_list list of V matrices (n x p_v). Block-missing rows (all NA)
#'   are automatically excluded from the k-NN graph.
#' @param k_nn integer, number of nearest neighbours (default 10)
#' @return list of V sparse Laplacian matrices, one per modality. Pass directly
#'   to the \code{precomp} argument of [woven()].
#' @examples
#' set.seed(1)
#' n <- 40; K <- 2L
#' X1 <- matrix(rnorm(n * 8), n, 8)
#' X2 <- matrix(rnorm(n * 6), n, 6)
#' Y <- rep(1:2, each = n / 2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA; X2[miss[, 2], ] <- NA
#' precomp <- woven_precompute(list(X1, X2), k_nn = 10L)
#' fit <- woven(list(X1, X2), Y = Y, K = K, precomp = precomp)
#' @seealso [woven()]
#' @export
woven_precompute <- function(X_list, k_nn = 10L) {
    if (!is.list(X_list))
        stop("X_list must be a list of matrices.")
    V     <- length(X_list)
    max_p <- max(vapply(X_list, ncol, integer(1L)))
    # Parallelize V Laplacians when p > 500: fork overhead (~200ms) is
    # negligible vs build time for high-dim omics data, and V tasks are independent.
    # For small p (toy data, vignette examples), stay sequential.
    cores <- if (V > 1L && max_p > 500L)
        min(V, parallel::detectCores(logical = FALSE))
    else
        1L
    parallel::mclapply(X_list, function(X) build_laplacian(X, k = k_nn),
                       mc.cores = cores)
}

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
#' @param k_nn integer  -- k-nearest-neighbors for Laplacian graph (default 10).
#'   Ignored when \code{precomp} is supplied.
#' @param precomp optional output of [woven_precompute()] -- pre-built Laplacians.
#'   Pass this when calling \code{woven()} multiple times on the same data (e.g.
#'   hyperparameter search, cross-validation) to avoid rebuilding the graph each time.
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
#'     \item{$lambdas, $gamma_y}{hyperparameters}
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
                  k_nn = 10L,
                  precomp = NULL,
                  verbose = TRUE) {
    #  Input validation
    if (!is.list(X_list))
        stop("X_list must be a list of matrices, e.g. list(RNA = X_rna, Methyl = X_meth).")
    V <- length(X_list)
    if (V < 2L)
        stop(sprintf("X_list must contain at least 2 modalities; got %d.", V))
    for (v in seq_len(V)) {
        if (!is.matrix(X_list[[v]]))
            stop(sprintf(
                "X_list[[%d]] is not a matrix (got %s). Convert with as.matrix().",
                v, class(X_list[[v]])[1L]
            ))
    }
    n_rows <- vapply(X_list, nrow, integer(1L))
    if (length(unique(n_rows)) > 1L) {
        nm <- if (!is.null(names(X_list))) names(X_list) else paste0("[[", seq_len(V), "]]")
        stop(sprintf(
            "All matrices must have the same number of rows (subjects).\n  Row counts: %s",
            paste(sprintf("%s=%d", nm, n_rows), collapse = ", ")
        ))
    }
    n <- n_rows[1L]
    if (length(Y) != n)
        stop(sprintf(
            "length(Y) = %d but nrow(X_list[[1]]) = %d. Y must have one label per subject.",
            length(Y), n
        ))
    if (anyNA(Y))
        stop("Y contains NA values. Every subject must have a class label.")
    if (K < 1L) stop("K must be >= 1.")
    if (K >= n) stop(sprintf("K=%d must be less than n=%d.", K, n))

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

    Y_fct    <- as.factor(Y)
    Y_labels <- levels(Y_fct)   # original label names e.g. "CN","MCI","Dementia"
    Y        <- as.integer(Y_fct)

    if (length(lambdas) == 1L) lambdas <- rep(lambdas, V)
    stopifnot(length(lambdas) == V)

    #  Extract anchor Laplacian submatrices from precomp if supplied
    La_list_precomp <- if (!is.null(precomp)) {
        if (!is.list(precomp) || length(precomp) != V)
            stop(sprintf(
                "precomp must be the output of woven_precompute() -- a list of %d Laplacians.", V
            ))
        lapply(precomp, function(L) as.matrix(L[anchor_idx, anchor_idx, drop = FALSE]))
    } else {
        NULL
    }

    #  Fit: closed-form dual SUMCOR MCCA for all V
    #  Identical solver to benchmark_one_rep.R -- paper results exactly reproduced.
    raw <- woven_mcca_dual(
        X_list          = X_list,
        anchor_idx      = anchor_idx,
        Y               = Y,
        K               = K,
        lambdas         = lambdas,
        gamma_y         = gamma_y,
        k_nn            = k_nn,
        La_list_precomp = La_list_precomp,
        verbose         = verbose
    )
    W_list    <- raw$W_list
    Z_anchors <- raw$Za_list
    svals     <- raw$singular_values
    fit_mcca  <- raw

    #  Propagate feature names and modality names onto W_list
    mod_names <- if (!is.null(names(X_list))) names(X_list) else
        paste0("Modality ", seq_len(V))
    for (v in seq_len(V)) {
        rownames(W_list[[v]]) <- colnames(X_list[[v]])
    }
    names(W_list) <- mod_names

    #  Compute consensus Z for all n subjects via vectorised BLAS projection
    if (verbose) message("  Projecting all subjects...")
    t_proj <- proc.time()
    Z_acc   <- matrix(0,       n, K)
    obs_cnt <- integer(n)
    for (v in seq_len(V)) {
        Xv  <- X_list[[v]]
        obs <- which(!apply(Xv, 1L, function(r) all(is.na(r))))
        if (length(obs) == 0L) next
        Xv_obs <- Xv[obs, , drop = FALSE]
        Xv_obs[is.na(Xv_obs)] <- 0
        Z_acc[obs, ] <- Z_acc[obs, ] + Xv_obs %*% W_list[[v]]
        obs_cnt[obs] <- obs_cnt[obs] + 1L
    }
    none <- obs_cnt == 0L
    obs_cnt[none] <- 1L
    Z_all <- Z_acc / obs_cnt
    Z_all[none, ] <- NA_real_
    dimnames(Z_all) <- list(rownames(X_list[[1]]), paste0("Dim", seq_len(K)))
    if (verbose) {
        elapsed  <- round((proc.time() - t_proj)["elapsed"], 1)
        n_scored <- sum(!none)
        message(sprintf(
            "  Done. %d / %d subjects scored (%.0f%% ESS) in %.1fs.",
            n_scored, n, 100 * n_scored / n, elapsed
        ))
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
            Y_labels        = Y_labels,
            K               = K,
            V               = V,
            n               = n,
            mod_names       = mod_names,
            lambdas         = lambdas,
            gamma_y         = gamma_y,
            k_nn            = k_nn,
            fit_mcca        = fit_mcca
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
        "  Solver     : mcca_dual (closed-form, globally optimal)\n"
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
    if (!is.null(x$Y_labels)) {
        cat(sprintf("  Classes    : %s\n", paste(x$Y_labels, collapse = ", ")))
    }
    if (!is.null(x$mod_names)) {
        cat(sprintf("  Modalities : %s\n", paste(x$mod_names, collapse = ", ")))
    }
    first_mod <- if (!is.null(x$mod_names))
        sprintf('"%s"', x$mod_names[1L]) else "1"
    cat("\n")
    cat("  -- Next steps --\n")
    cat("  plot(fit, labels = Y)                         # latent space scatter\n")
    cat(sprintf(
        "  woven_plot_vip(fit, modality = %-12s  # top features by VIP\n",
        paste0(first_mod, ")")
    ))
    cat("  woven_plot_loadings(fit, dim = 1)             # loadings per modality\n")
    cat("  woven_plot_variance(fit)                      # variance per dimension\n")
    cat("  woven_metrics(fit, Y)                         # silhouette, NMI, ESS\n")
    cat("  woven_predict(fit, X_list_new)                # predict on new data\n")
    invisible(x)
}

#' Plot the WOVEN latent space
#'
#' Plots the first two latent dimensions from \code{fit$Z}, colored by group
#' label. Anchor subjects (complete cases used to learn W) are shown as filled
#' circles; block-missing subjects projected via available views are shown as
#' open triangles. Returns a ggplot object that can be further customized with
#' \code{+} layers.
#'
#' @param x a woven object from [woven()]
#' @param labels integer or factor of length n for coloring points.
#'   If NULL, all points are plotted in a single color.
#' @param dims integer vector of length 2: which latent dimensions to plot
#'   (default c(1, 2))
#' @param highlight_anchors logical: distinguish anchors from projected subjects
#'   via point shape (default TRUE)
#' @param ... unused; present for S3 compatibility
#' @return a ggplot object, invisibly. The plot is printed as a side effect.
#' @examples
#' set.seed(1)
#' n <- 20; K <- 2L
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
#' plot(fit, labels = Y)
#' @export
plot.woven <- function(x, labels = NULL, dims = c(1L, 2L),
                       highlight_anchors = TRUE, ...) {
    .require_ggplot2()
    Z <- x$Z
    if (is.null(Z)) stop("fit$Z is NULL. Refit with woven().")
    d1 <- dims[1L]
    d2 <- dims[2L]
    if (d1 > x$K || d2 > x$K)
        stop(sprintf("dims out of range: fit has K=%d dimensions.", x$K))

    is_anchor <- seq_len(x$n) %in% x$anchor_idx
    df <- data.frame(
        z1         = Z[, d1],
        z2         = Z[, d2],
        point_type = ifelse(is_anchor, "Anchor", "Projected"),
        stringsAsFactors = FALSE
    )
    df$group <- if (!is.null(labels)) as.factor(labels) else factor("all")

    # Drop rows with no latent score (subjects with zero observed views)
    df <- df[!is.na(df$z1), , drop = FALSE]

    n_groups <- length(levels(df$group))
    pal_use  <- .pal_woven[seq_len(n_groups)]

    df_anchor    <- df[df$point_type == "Anchor", , drop = FALSE]
    df_projected <- df[df$point_type == "Projected", , drop = FALSE]

    p_out <- ggplot2::ggplot(df, ggplot2::aes(x = z1, y = z2, color = group))

    if (highlight_anchors && nrow(df_projected) > 0L) {
        p_out <- p_out +
            ggplot2::geom_point(data = df_projected,
                       size = 1.4, alpha = 0.35, shape = 16L)
    }

    p_out <- p_out +
        ggplot2::geom_point(data = df_anchor,
                   size = 2.8, alpha = 0.9, shape = 16L)

    if (!is.null(labels) && nrow(df_anchor) >= 4L) {
        p_out <- p_out +
            ggplot2::stat_ellipse(data = df_anchor,
                         type = "norm", level = 0.68,
                         linewidth = 0.8, linetype = "dashed", alpha = 0.7)
    }

    if (!is.null(labels)) {
        p_out <- p_out +
            ggplot2::scale_color_manual(values = pal_use, name = "Group")
    } else {
        p_out <- p_out +
            ggplot2::scale_color_manual(values = pal_use[1L]) +
            ggplot2::guides(color = "none")
    }

    n_scored   <- sum(!is.na(x$Z[, 1L]))
    n_anchor   <- length(x$anchor_idx)
    n_proj     <- n_scored - n_anchor
    ess_pct    <- round(100 * n_scored / x$n)

    anchor_note <- if (highlight_anchors && n_proj > 0L)
        sprintf("solid = %d anchors, faded = %d projected", n_anchor, n_proj)
    else NULL

    p_out <- p_out +
        ggplot2::labs(
            x        = sprintf("WOVEN Dimension %d", d1),
            y        = sprintf("WOVEN Dimension %d", d2),
            title    = "WOVEN Latent Space",
            subtitle = sprintf("%d / %d subjects scored (%d%% ESS)%s",
                               n_scored, x$n, ess_pct,
                               if (!is.null(anchor_note))
                                   paste0("  |  ", anchor_note) else "")
        ) +
        .theme_woven() +
        ggplot2::theme(legend.position = "right")

    print(p_out)
    invisible(p_out)
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
#'   one non-missing view. Column order must match the training data.
#' @return numeric matrix n_new x K of consensus latent scores, with rownames
#'   from \code{X_list_new[[1]]} and colnames \code{Dim1}, \code{Dim2}, etc.
#'   Subjects with no observed data in any view receive a row of NA.
#' @examples
#' set.seed(1)
#' n <- 30; K <- 2L
#' X1 <- matrix(rnorm(n * 8), n, 8)
#' X2 <- matrix(rnorm(n * 6), n, 6)
#' Y <- rep(1:2, each = n / 2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA; X2[miss[, 2], ] <- NA
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' Z_new <- woven_scores(fit, list(X1[1:5, ], X2[1:5, ]))
#' dim(Z_new)
#' @seealso [woven_predict()] for class predictions, [woven()] for model fitting
#' @export
woven_scores <- function(fit, X_list_new) {
    stopifnot(inherits(fit, "woven"))
    if (!is.list(X_list_new) || length(X_list_new) != fit$V)
        stop(sprintf(
            "X_list_new must be a list of %d matrices (one per modality).", fit$V
        ))
    for (v in seq_len(fit$V)) {
        p_new  <- ncol(X_list_new[[v]])
        p_fit  <- nrow(fit$W_list[[v]])
        nm     <- if (!is.null(fit$mod_names)) fit$mod_names[v] else
            paste0("modality ", v)
        if (p_new != p_fit)
            stop(sprintf(
                "X_list_new[[%d]] (%s) has %d features but model was trained on %d.",
                v, nm, p_new, p_fit
            ))
    }

    n_new   <- nrow(X_list_new[[1]])
    K       <- fit$K
    V       <- fit$V
    rn      <- rownames(X_list_new[[1]])

    Z_acc   <- matrix(0, n_new, K)
    obs_cnt <- integer(n_new)
    for (v in seq_len(V)) {
        Xv  <- X_list_new[[v]]
        obs <- which(!apply(Xv, 1L, function(r) all(is.na(r))))
        if (length(obs) == 0L) next
        Xv_obs <- Xv[obs, , drop = FALSE]
        Xv_obs[is.na(Xv_obs)] <- 0
        Z_acc[obs, ] <- Z_acc[obs, ] + Xv_obs %*% fit$W_list[[v]]
        obs_cnt[obs] <- obs_cnt[obs] + 1L
    }
    none <- obs_cnt == 0L
    obs_cnt[none] <- 1L
    Z <- Z_acc / obs_cnt
    Z[none, ] <- NA_real_
    dimnames(Z) <- list(rn, paste0("Dim", seq_len(K)))
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
    if (!is.list(X_list_new) || length(X_list_new) != fit$V)
        stop(sprintf(
            "X_list_new must be a list of %d matrices (one per modality).", fit$V
        ))
    for (v in seq_len(fit$V)) {
        p_new <- ncol(X_list_new[[v]])
        p_fit <- nrow(fit$W_list[[v]])
        nm    <- if (!is.null(fit$mod_names)) fit$mod_names[v] else paste0("modality ", v)
        if (p_new != p_fit)
            stop(sprintf(
                "X_list_new[[%d]] (%s) has %d features but model was trained on %d.",
                v, nm, p_new, p_fit
            ))
    }
    pred_rownames <- rownames(X_list_new[[1L]])

    #  Project via direct W multiplication — identical to paper benchmark
    Z_new <- woven_scores(fit, X_list_new)

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

    # Use original label names if available (e.g. "CN","MCI","Dementia")
    display_labels <- if (!is.null(fit$Y_labels)) fit$Y_labels else
        as.character(levels_Y)
    colnames(probs) <- paste0("p_", display_labels)
    pred_idx <- apply(probs, 1, which.max)

    data.frame(
        predicted_class = display_labels[pred_idx],
        confidence      = probs[cbind(seq_len(n_new), pred_idx)],
        as.data.frame(probs),
        row.names       = pred_rownames,
        stringsAsFactors = FALSE
    )
}

