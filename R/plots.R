# plots.R -- ggplot2-based publication-ready plots for WOVEN fits
# ggplot2 is Suggests: every exported function checks for it at runtime.
#
# woven_plot_vip()      -- VIP scores per modality (lollipop chart)
# woven_plot_loadings() -- Feature loadings for one latent dimension
# woven_plot_variance() -- Variance explained per dimension (bar + cumulative)

# Suppress R CMD check notes for ggplot2 aesthetic variable names
utils::globalVariables(c(
    "feature", "vip_score",
    "loading", "sign_str", "modality_label",
    "dim_int", "pct_var", "cumulative",
    "z1", "z2", "group", "point_type",
    "abs_loading"
))

.require_ggplot2 <- function() {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop(
            "Package 'ggplot2' is needed for WOVEN plots.\n",
            "Install it with: install.packages(\"ggplot2\")",
            call. = FALSE
        )
    }
    invisible(TRUE)
}

# Resolve modality argument: accepts integer index or name string
.resolve_modality <- function(modality, fit) {
    if (is.character(modality)) {
        idx <- match(modality, fit$mod_names)
        if (is.na(idx)) {
            stop(sprintf(
                "Modality '%s' not found. Available: %s",
                modality, paste(fit$mod_names, collapse = ", ")
            ))
        }
        return(idx)
    }
    as.integer(modality)
}

# Colorblind-safe palette (Wong 2011, Nature Methods)
.pal_woven <- c(
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#999999"
)

# Shared minimal theme for all WOVEN plots
.theme_woven <- function(base_size = 12) {
    ggplot2::theme_bw(base_size = base_size) +
        ggplot2::theme(
            plot.title       = ggplot2::element_text(face = "bold", hjust = 0.5, size = base_size + 1),
            plot.subtitle    = ggplot2::element_text(hjust = 0.5, color = "gray40", size = base_size - 1),
            panel.grid.minor = ggplot2::element_blank(),
            strip.background = ggplot2::element_rect(fill = "gray95", color = "gray80"),
            strip.text       = ggplot2::element_text(face = "bold")
        )
}

#' Plot VIP scores for a WOVEN modality
#'
#' Displays the top features by Variable Importance in Projection (VIP) score
#' for one modality. VIP scores weight each feature's loading across all K
#' latent dimensions by the variance explained per dimension, producing a
#' single importance ranking analogous to DIABLO's contribution plot.
#' A dashed reference line at VIP = 1 marks above-average importance.
#'
#' @param fit woven object from [woven()]
#' @param modality integer: which modality to plot (1..V, default 1)
#' @param n_top integer: number of top features to display (default 20)
#' @param feature_names optional character vector of length p_v with feature
#'   labels. If NULL, uses rownames of W_list[[modality]] or "Feature_j".
#' @param main character: plot title. If NULL, a default is generated.
#' @return a ggplot object (printed automatically; add layers with \code{+})
#' @examples
#' set.seed(1)
#' n <- 60
#' p1 <- 30
#' p2 <- 20
#' K <- 3
#' Y <- rep(1:3, each = n / 3)
#' X1 <- matrix(rnorm(n * p1), n, p1)
#' X2 <- matrix(rnorm(n * p2), n, p2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' woven_plot_vip(fit, modality = 1L)
#' @seealso [woven_plot_loadings()], [woven_plot_variance()], [woven_vip()]
#' @export
woven_plot_vip <- function(fit, modality = 1L, n_top = 20L,
                           feature_names = NULL, main = NULL) {
    stopifnot(inherits(fit, "woven"))
    modality <- .resolve_modality(modality, fit)
    stopifnot(length(modality) == 1L, modality >= 1L, modality <= fit$V)

    W <- fit$W_list[[modality]]
    p <- nrow(W)
    svals <- fit$singular_values
    wts <- svals^2 / sum(svals^2)
    vip <- sqrt(p * rowSums(sweep(W^2, 2L, wts, "*")))

    if (is.null(feature_names)) {
        feature_names <- if (!is.null(rownames(W))) {
            rownames(W)
        } else {
            paste0("Feature_", seq_len(p))
        }
    }
    if (length(feature_names) != p) {
        stop(sprintf(
            "feature_names length %d != %d features in modality %d.",
            length(feature_names), p, modality
        ))
    }

    n_top <- min(as.integer(n_top), p)
    top_idx <- order(vip, decreasing = TRUE)[seq_len(n_top)]

    df <- data.frame(
        feature = factor(feature_names[top_idx],
            levels = feature_names[top_idx[order(vip[top_idx])]]
        ),
        vip_score = vip[top_idx],
        stringsAsFactors = FALSE
    )

    col_use <- .pal_woven[((modality - 1L) %% length(.pal_woven)) + 1L]

    mod_label <- if (!is.null(fit$mod_names)) {
        fit$mod_names[[modality]]
    } else {
        paste0("Modality ", modality)
    }

    if (is.null(main)) {
        main <- sprintf("VIP Scores - %s", mod_label)
    }
    subtitle <- sprintf("Top %d of %d features", n_top, p)

    p_out <- ggplot2::ggplot(df, ggplot2::aes(x = vip_score, y = feature)) +
        ggplot2::geom_col(fill = col_use, width = 0.7) +
        ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", vip_score)),
            hjust = -0.1, size = 3, color = "gray30"
        ) +
        ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.18))) +
        ggplot2::labs(
            x        = "VIP score",
            y        = NULL,
            title    = main,
            subtitle = subtitle
        ) +
        .theme_woven() +
        ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

    # Only draw VIP = 1 line if it falls within the data range
    if (max(df$vip_score) > 0.9) {
        p_out <- p_out +
            ggplot2::geom_vline(
                xintercept = 1, linetype = "dashed",
                color = "gray50", linewidth = 0.5
            ) +
            ggplot2::annotate("text",
                x = 1, y = Inf,
                label = "VIP = 1",
                hjust = -0.1, vjust = 1.5,
                size = 3, color = "gray45"
            )
    }

    p_out
}

#' Plot feature loadings for one WOVEN latent dimension
#'
#' For a given latent dimension, shows the top features by absolute loading
#' for each modality (or a selected subset), colored by loading sign.
#' Positive loadings are blue; negative loadings are red-orange.
#' Equivalent to DIABLO's \code{plotLoadings()}.
#'
#' @param fit woven object from [woven()]
#' @param dim integer: which latent dimension to plot (1..K, default 1)
#' @param n_top integer: number of top features per modality (default 15)
#' @param feature_names optional list of V character vectors (one per modality).
#'   If a plain character vector is passed for a single-modality call, it is
#'   used for that modality. If NULL, uses rownames of W or "Feature_j".
#' @param modality integer or NULL: if specified, plot only that modality.
#'   If NULL (default), all V modalities are shown in faceted panels.
#' @param main character: plot title. If NULL, a default is used.
#' @return a ggplot object (printed automatically; add layers with \code{+})
#' @examples
#' set.seed(1)
#' n <- 60
#' p1 <- 30
#' p2 <- 20
#' K <- 3
#' Y <- rep(1:3, each = n / 3)
#' X1 <- matrix(rnorm(n * p1), n, p1)
#' X2 <- matrix(rnorm(n * p2), n, p2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' woven_plot_loadings(fit, dim = 1L)
#' @seealso [woven_plot_vip()], [woven_plot_variance()]
#' @export
woven_plot_loadings <- function(fit, dim = 1L, n_top = 15L,
                                feature_names = NULL, modality = NULL,
                                main = NULL) {
    stopifnot(inherits(fit, "woven"))
    stopifnot(length(dim) == 1L, dim >= 1L, dim <= fit$K)

    mod_seq <- if (is.null(modality)) {
        seq_len(fit$V)
    } else {
        vapply(modality, .resolve_modality, integer(1L), fit = fit)
    }
    stopifnot(all(mod_seq >= 1L), all(mod_seq <= fit$V))

    # Build feature name list (length V)
    if (is.null(feature_names)) {
        feature_names <- lapply(seq_len(fit$V), function(v) {
            W <- fit$W_list[[v]]
            if (!is.null(rownames(W))) {
                rownames(W)
            } else {
                paste0("Feature_", seq_len(nrow(W)))
            }
        })
    } else if (is.character(feature_names)) {
        fn <- feature_names
        feature_names <- lapply(seq_len(fit$V), function(v) fn)
    }

    col_pos <- "#0072B2"
    col_neg <- "#D55E00"

    mod_names <- if (!is.null(fit$mod_names)) {
        fit$mod_names
    } else {
        paste0("Modality ", seq_len(fit$V))
    }

    rows <- lapply(mod_seq, function(v) {
        W <- fit$W_list[[v]]
        w <- W[, dim]
        p <- length(w)
        nm <- feature_names[[v]]
        if (length(nm) != p) {
            stop(sprintf("feature_names[[%d]] length %d != %d.", v, length(nm), p))
        }

        n_show <- min(as.integer(n_top), p)
        top_idx <- order(abs(w), decreasing = TRUE)[seq_len(n_show)]

        data.frame(
            modality_label = mod_names[[v]],
            feature = nm[top_idx],
            loading = w[top_idx],
            sign_str = ifelse(w[top_idx] >= 0, "positive", "negative"),
            abs_loading = abs(w[top_idx]),
            stringsAsFactors = FALSE
        )
    })
    df <- do.call(rbind, rows)

    # Reorder features within each modality panel by loading value
    df$feature <- stats::reorder(
        factor(df$feature),
        df$loading,
        FUN = function(x) x[1]
    )

    if (is.null(main)) {
        main <- sprintf("Feature Loadings - Dimension %d", dim)
    }

    n_mod <- length(mod_seq)
    ncol_facets <- min(n_mod, 3L)

    ggplot2::ggplot(df, ggplot2::aes(x = loading, y = feature, fill = sign_str)) +
        ggplot2::geom_col(width = 0.7) +
        ggplot2::geom_vline(xintercept = 0, color = "gray30", linewidth = 0.4) +
        ggplot2::facet_wrap(~modality_label, scales = "free_y", ncol = ncol_facets) +
        ggplot2::scale_fill_manual(
            values = c("positive" = col_pos, "negative" = col_neg),
            labels = c("positive" = "Positive", "negative" = "Negative")
        ) +
        ggplot2::labs(
            x = sprintf("Loading (Dimension %d)", dim),
            y = NULL,
            fill = NULL,
            title = main
        ) +
        .theme_woven() +
        ggplot2::theme(
            panel.grid.major.y = ggplot2::element_blank(),
            legend.position    = "bottom"
        )
}

#' Plot variance explained per WOVEN latent dimension
#'
#' Bar chart of the proportion of variance explained per latent dimension
#' (proportional to squared singular values), overlaid with a cumulative
#' variance curve on a secondary axis. Use this to choose K and to show
#' how much shared multi-omics signal is captured in the leading dimensions.
#'
#' @param fit woven object from [woven()]
#' @param main character: plot title (default "Variance Explained")
#' @return a ggplot object (printed automatically; add layers with \code{+})
#' @examples
#' set.seed(1)
#' n <- 60
#' p1 <- 30
#' p2 <- 20
#' K <- 4
#' Y <- rep(1:2, each = n / 2)
#' X1 <- matrix(rnorm(n * p1), n, p1)
#' X2 <- matrix(rnorm(n * p2), n, p2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' woven_plot_variance(fit)
#' @seealso [woven_plot_vip()], [woven_plot_loadings()]
#' @export
woven_plot_variance <- function(fit, main = "Variance Explained") {
    stopifnot(inherits(fit, "woven"))
    svals <- fit$singular_values
    K <- fit$K
    prop_var <- svals^2 / sum(svals^2)
    cumvar <- cumsum(prop_var)

    df <- data.frame(
        dim_int = seq_len(K),
        pct_var = prop_var * 100,
        cumulative = cumvar * 100,
        stringsAsFactors = FALSE
    )

    col_bar <- "#0072B2"
    col_cum <- "#D55E00"

    ggplot2::ggplot(df, ggplot2::aes(x = dim_int)) +
        ggplot2::geom_col(ggplot2::aes(y = pct_var),
            fill = col_bar, width = 0.65, alpha = 0.9
        ) +
        ggplot2::geom_line(ggplot2::aes(y = cumulative),
            color = col_cum, linewidth = 1.2
        ) +
        ggplot2::geom_point(ggplot2::aes(y = cumulative),
            color = col_cum, size = 3.2
        ) +
        ggplot2::geom_text(ggplot2::aes(y = pct_var / 2, label = sprintf("%.1f%%", pct_var)),
            size = 3.2, color = "white", fontface = "bold"
        ) +
        ggplot2::scale_x_continuous(
            breaks = seq_len(K),
            labels = paste0("Dim ", seq_len(K))
        ) +
        ggplot2::scale_y_continuous(
            name     = "% Per Dimension",
            limits   = c(0, 110),
            breaks   = seq(0, 100, 20),
            sec.axis = ggplot2::dup_axis(name = "Cumulative %")
        ) +
        ggplot2::labs(
            x     = "Latent Dimension",
            title = main
        ) +
        ggplot2::annotate(
            "text",
            x = df$dim_int, y = df$cumulative + 4,
            label = sprintf("%.0f%%", df$cumulative),
            color = col_cum, size = 3.2, fontface = "bold"
        ) +
        .theme_woven() +
        ggplot2::theme(
            axis.title.y.left  = ggplot2::element_text(color = col_bar),
            axis.text.y.left   = ggplot2::element_text(color = col_bar),
            axis.title.y.right = ggplot2::element_text(color = col_cum),
            axis.text.y.right  = ggplot2::element_text(color = col_cum)
        )
}
