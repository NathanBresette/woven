# metrics.R  -- Evaluation metric battery for WOVEN benchmarking
#
# All functions take a latent matrix Z (n x K) and return a scalar or named vector.
# Designed to work on: WOVEN output, DIABLO variate scores, MOFA+ factor scores.
#
# Metrics:
#   woven_silhouette       -- average silhouette width (cluster geometry)
#   woven_davies_bouldin   -- Davies-Bouldin index (cluster compactness/separation)
#   woven_nmi              -- normalized mutual information (label recovery)
#   woven_rv               -- RV coefficient vs ground-truth factor matrix
#   woven_ess_retention    -- effective sample size retention (N_used / N_total)
#   woven_effect_bias      -- subgroup effect estimate bias (CER-specific metric)
#   woven_nystrom_error    -- leave-anchor-out Nystrm projection error
#   woven_all_metrics      -- compute full battery, returns named list

#  Silhouette

#' Average silhouette width
#'
#' @param Z numeric matrix n x K (latent scores)
#' @param labels integer or factor of length n (subgroup labels)
#' @return scalar in [-1, 1], higher is better
#' @examples
#' set.seed(1)
#' Z <- matrix(rnorm(20 * 2), 20, 2)
#' labels <- rep(1:2, each = 10)
#' woven_silhouette(Z, labels)
#' @export
woven_silhouette <- function(Z, labels) {
    labels <- as.integer(as.factor(labels))
    if (length(unique(labels)) < 2L) {
        return(NA_real_)
    }
    d <- dist(Z)
    sw <- cluster::silhouette(labels, d)
    mean(sw[, "sil_width"])
}

#  Davies-Bouldin

#' Davies-Bouldin index
#'
#' DB = (1/K) sum_i max_\{j != i\} (s_i + s_j) / d(c_i, c_j)
#' where s_i = mean intra-cluster distance, d(c_i, c_j) = centroid distance.
#'
#' @param Z numeric matrix n x K
#' @param labels integer or factor of length n
#' @return scalar >= 0, lower is better
#' @examples
#' set.seed(1)
#' Z <- matrix(rnorm(20 * 2), 20, 2)
#' labels <- rep(1:2, each = 10)
#' woven_davies_bouldin(Z, labels)
#' @export
woven_davies_bouldin <- function(Z, labels) {
    labels <- as.integer(as.factor(labels))
    K_cl <- length(unique(labels))
    if (K_cl < 2L) {
        return(NA_real_)
    }

    centroids <- do.call(rbind, lapply(sort(unique(labels)), function(g) {
        colMeans(Z[labels == g, , drop = FALSE])
    }))

    s <- vapply(sort(unique(labels)), function(g) {
        Zg <- Z[labels == g, , drop = FALSE]
        if (nrow(Zg) < 2L) {
            return(0)
        }
        mean(sqrt(rowSums(sweep(Zg, 2, centroids[g, ])^2)))
    }, numeric(1L))

    db_vals <- vapply(seq_len(K_cl), function(i) {
        others <- setdiff(seq_len(K_cl), i)
        max(vapply(others, function(j) {
            d_ij <- sqrt(sum((centroids[i, ] - centroids[j, ])^2))
            if (d_ij < 1e-12) {
                return(0)
            }
            (s[i] + s[j]) / d_ij
        }, numeric(1L)))
    }, numeric(1L))

    mean(db_vals)
}

#  NMI

#' Normalized mutual information between cluster assignments and true labels
#'
#' Uses k-means on Z to get cluster assignments, then computes NMI.
#' k-means run 10 times to reduce initialization variance.
#'
#' @param Z numeric matrix n x K
#' @param labels integer or factor of length n (true labels)
#' @param n_cl integer, number of clusters (default = number of unique labels)
#' @param n_start integer, k-means random starts
#' @return scalar in [0, 1], higher is better
#' @examples
#' set.seed(1)
#' Z <- matrix(rnorm(40 * 2), 40, 2)
#' labels <- rep(1:2, each = 20)
#' woven_nmi(Z, labels)
#' @export
woven_nmi <- function(Z, labels, n_cl = NULL, n_start = 10L) {
    labels <- as.integer(as.factor(labels))
    if (is.null(n_cl)) n_cl <- length(unique(labels))
    if (n_cl < 2L) {
        return(NA_real_)
    }

    km <- kmeans(Z, centers = n_cl, nstart = n_start, iter.max = 100L)
    pred <- km$cluster

    # NMI via entropy decomposition (no external package needed)
    .entropy <- function(x) {
        px <- tabulate(x) / length(x)
        px <- px[px > 0]
        -sum(px * log(px))
    }
    .joint_entropy <- function(x, y) {
        n <- length(x)
        tbl <- table(x, y)
        pxy <- tbl / n
        pxy <- pxy[pxy > 0]
        -sum(pxy * log(pxy))
    }

    H_true <- .entropy(labels)
    H_pred <- .entropy(pred)
    H_joint <- .joint_entropy(labels, pred)
    MI <- H_true + H_pred - H_joint

    denom <- (H_true + H_pred) / 2
    if (denom < 1e-12) {
        return(NA_real_)
    }
    MI / denom
}

#  RV coefficient

#' RV coefficient between latent scores and ground-truth factor matrix
#'
#' RV(X, Y) = trace(X X' Y Y') / sqrt(trace(X X' X X') * trace(Y Y' Y Y'))
#' Measures similarity of two cross-product matrices; 1 = identical subspace.
#'
#' @param Z numeric matrix n x K (inferred latent scores)
#' @param Z_true numeric matrix n x K_true (ground-truth factor scores from SUMO)
#' @return scalar in [0, 1], higher is better
#' @examples
#' set.seed(1)
#' Z <- matrix(rnorm(20 * 2), 20, 2)
#' Z_true <- matrix(rnorm(20 * 3), 20, 3)
#' woven_rv(Z, Z_true)
#' @export
woven_rv <- function(Z, Z_true) {
    stopifnot(nrow(Z) == nrow(Z_true))
    # Center columns
    Z <- scale(Z, center = TRUE, scale = FALSE)
    Z_true <- scale(Z_true, center = TRUE, scale = FALSE)

    S <- tcrossprod(Z) # n x n
    T_ <- tcrossprod(Z_true) # n x n

    num <- sum(S * T_)
    denom <- sqrt(sum(S * S) * sum(T_ * T_))
    if (denom < 1e-12) {
        return(NA_real_)
    }
    num / denom
}

#  Effective sample size retention

#' Effective sample size retention
#'
#' @param n_used integer, number of subjects with a latent score
#' @param n_total integer, total subjects in dataset
#' @return scalar in [0, 1], higher is better (DIABLO structurally caps at overlap fraction)
#' @examples
#' woven_ess_retention(n_used = 80, n_total = 100)
#' @export
woven_ess_retention <- function(n_used, n_total) {
    stopifnot(n_used >= 0, n_total > 0, n_used <= n_total)
    n_used / n_total
}

#  Subgroup effect estimate bias

#' CER-specific: subgroup effect estimate bias
#'
#' Fits a linear model of a continuous outcome on a binary treatment indicator,
#' separately within each subgroup defined by `labels`. Compares estimated
#' treatment effect to the known true effect (from simulation ground truth).
#'
#' bias_g = |estimated_g - true_g| / |true_g|   (relative)
#' Returns mean bias across subgroups.
#'
#' @param Z numeric matrix n x K (latent scores; used as covariates)
#' @param outcome numeric vector of length n (simulated continuous outcome)
#' @param treatment integer/logical vector of length n (0/1 treatment indicator)
#' @param labels integer or factor of length n (subgroup labels)
#' @param true_effects named numeric vector, true treatment effect per subgroup level
#' @return scalar >= 0, lower is better
#' @examples
#' set.seed(1)
#' n <- 60
#' Z <- matrix(rnorm(n * 3), n, 3)
#' outcome <- rnorm(n)
#' treatment <- rep(0:1, n / 2)
#' labels <- rep(1:2, each = n / 2)
#' true_eff <- c(0.5, 1.0)
#' woven_effect_bias(Z, outcome, treatment, labels, true_eff)
#' @export
woven_effect_bias <- function(Z, outcome, treatment, labels, true_effects) {
    labels <- as.integer(as.factor(labels))
    treatment <- as.numeric(treatment)
    groups <- sort(unique(labels))

    biases <- vapply(seq_along(groups), function(gi) {
        g <- groups[gi]
        idx <- which(labels == g)
        if (length(idx) < 5L) {
            return(NA_real_)
        }

        df <- data.frame(y = outcome[idx], trt = treatment[idx], Z[idx, , drop = FALSE])
        fit <- tryCatch(lm(y ~ ., data = df), error = function(e) NULL)
        if (is.null(fit)) {
            return(NA_real_)
        }

        est <- coef(fit)["trt"]
        true <- true_effects[gi]
        if (is.na(true) || abs(true) < 1e-12) {
            return(NA_real_)
        }
        abs(est - true) / abs(true)
    }, numeric(1L))

    mean(biases, na.rm = TRUE)
}

#  Nystrm leave-anchor-out error

#' Leave-anchor-out Nystrm projection error
#'
#' For each held-out anchor subject, refits WOVEN without it, projects via
#' direct W scoring, and computes ||Z_proj - Z_direct||.
#' Quantifies how well the projection generalizes across anchor subsets.
#'
#' @param fit a woven object from [woven()]
#' @param X_list list of complete (no block-missing) modality matrices, same
#'   structure as passed to [woven()]
#' @param n_loo integer, number of anchors to hold out (default min(20, n_a))
#' @param sigma_proj unused, kept for compatibility
#' @return scalar >= 0, lower is better (mean Frobenius error per anchor)
#' @examples
#' data(woven_example)
#' fit <- woven(woven_example$X_complete, Y = woven_example$Y, K = 3L)
#' woven_nystrom_error(fit, woven_example$X_complete, n_loo = 5L)
#' @export
woven_nystrom_error <- function(fit, X_list, n_loo = NULL, sigma_proj = NULL) {
    if (!inherits(fit, "woven")) stop("fit must be a woven object.")
    anchor_idx <- fit$anchor_idx
    n_a <- length(anchor_idx)
    if (is.null(n_loo)) n_loo <- min(20L, n_a)
    n_loo <- min(n_loo, n_a)
    loo_set <- sample(seq_len(n_a), n_loo)
    V <- length(fit$W_list)

    errors <- vapply(loo_set, function(j) {
        held_out  <- anchor_idx[j]
        remain_idx <- anchor_idx[-j]

        # True score from full fit
        Z_true <- Reduce("+", lapply(fit$Za_list, function(Z) Z[j, , drop = FALSE])) / V

        # LOO fit without this anchor
        X_loo <- lapply(X_list, function(X) {
            Xi <- X; Xi[held_out, ] <- NA_real_; Xi
        })
        mini <- tryCatch(
            woven_mcca_dual(X_loo, anchor_idx = remain_idx,
                            Y = as.integer(as.factor(fit$Y_anchor[-j])),
                            K = fit$K, lambdas = fit$lambdas,
                            gamma_y = fit$gamma_y, verbose = FALSE),
            error = function(e) NULL
        )
        if (is.null(mini)) return(NA_real_)

        # Direct W projection of held-out subject
        Xh <- lapply(seq_len(V), function(v) X_list[[v]][held_out, , drop = FALSE])
        Z_proj <- Reduce("+", lapply(seq_len(V), function(v)
            Xh[[v]] %*% mini$W_list[[v]])) / V

        sqrt(sum((Z_proj - Z_true)^2))
    }, numeric(1L))

    mean(errors, na.rm = TRUE)
}

#  Balanced Error Rate (BER)


#  Full metric battery

#' Compute full WOVEN evaluation metric battery
#'
#' @param Z numeric matrix n x K (consensus latent scores)
#' @param labels integer or factor of length n
#' @param n_total integer, total subjects before any filtering (for ESS)
#' @param Z_true optional numeric matrix n x K_true (ground-truth factors for RV)
#' @param outcome optional numeric vector (for effect bias)
#' @param treatment optional 0/1 vector (for effect bias)
#' @param true_effects optional named numeric (for effect bias)
#' @param fit optional WOVEN fit object (for Nystrm LOO error)
#' @param X1 optional matrix (for Nystrm LOO error)
#' @param X2 optional matrix (for Nystrm LOO error)
#' @param n_loo integer, anchors to hold out for Nystrm LOO (default 20)
#' @return named list of metric values
#' @keywords internal
woven_all_metrics <- function(Z, labels, n_total,
                              Z_true = NULL,
                              outcome = NULL,
                              treatment = NULL,
                              true_effects = NULL,
                              fit = NULL,
                              X1 = NULL,
                              X2 = NULL,
                              n_loo = 20L) {
    stopifnot(nrow(Z) == length(labels))

    out <- list(
        silhouette      = woven_silhouette(Z, labels),
        davies_bouldin  = woven_davies_bouldin(Z, labels),
        nmi             = woven_nmi(Z, labels),
        ess_retention   = woven_ess_retention(nrow(Z), n_total)
    )
    # BER is NOT computed here. It requires per-fold DR refitting to avoid
    # circularity: supervised methods encode labels into Z, so fixed-Z BER
    # recovers labels by construction. BER is computed in benchmark_one_rep.R
    # via ber_held_out_lda(): DR refit per fold, test subjects never seen during W fit.

    if (!is.null(Z_true)) {
        out$rv_coefficient <- woven_rv(Z, Z_true)
    }

    if (!is.null(outcome) && !is.null(treatment) && !is.null(true_effects)) {
        out$effect_bias <- woven_effect_bias(Z, outcome, treatment, labels, true_effects)
    }

    if (!is.null(fit) && !is.null(X1) && !is.null(X2)) {
        out$nystrom_error <- woven_nystrom_error(fit, X1, X2, n_loo = n_loo)
    }

    out
}

#' Convenience wrapper: compute core metrics directly from a woven fit
#'
#' Calls [woven_all_metrics()] using \code{fit$Z} and \code{fit$n} so you
#' do not need to extract them manually. Returns silhouette, Davies-Bouldin,
#' NMI, and ESS retention as a named numeric vector.
#'
#' @param fit woven object from [woven()]
#' @param labels integer, factor, or character vector of length n with subgroup
#'   labels for all subjects (same Y passed to [woven()])
#' @param ... additional arguments forwarded to [woven_all_metrics()]
#' @return named numeric vector of metric values, printed as a tidy table
#' @examples
#' set.seed(1)
#' n <- 40; K <- 2L
#' X1 <- matrix(rnorm(n * 8), n, 8)
#' X2 <- matrix(rnorm(n * 6), n, 6)
#' Y <- rep(1:2, each = n / 2)
#' miss <- matrix(runif(n * 2) < 0.3, n, 2)
#' for (i in which(rowSums(miss) == 2)) miss[i, sample(2, 1)] <- FALSE
#' X1[miss[, 1], ] <- NA
#' X2[miss[, 2], ] <- NA
#' fit <- woven(list(X1, X2), Y = Y, K = K)
#' woven_metrics(fit, Y)
#' @seealso [woven_all_metrics()], [woven_silhouette()], [woven_nmi()]
#' @export
woven_metrics <- function(fit, labels, ...) {
    stopifnot(inherits(fit, "woven"))
    stopifnot(length(labels) == fit$n)

    scored <- !is.na(fit$Z[, 1L])
    res <- woven_all_metrics(
        Z       = fit$Z[scored, , drop = FALSE],
        labels  = labels[scored],
        n_total = fit$n,
        ...
    )

    out <- unlist(res[c("silhouette", "davies_bouldin", "nmi", "ess_retention")])
    names(out) <- c("Silhouette", "Davies-Bouldin", "NMI", "ESS")
    out
}
