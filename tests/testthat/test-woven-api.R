make_fixture <- function(n = 80, p1 = 60, p2 = 40, K = 3L, seed = 99L) {
    set.seed(seed)
    groups <- rep(1:4, each = n / 4)
    X1 <- matrix(rnorm(n * p1), n, p1)
    X2 <- matrix(rnorm(n * p2), n, p2)
    # Interleave missing rows so each class has anchors
    miss1 <- seq(3, n, by = 8)[1:8]
    miss2 <- seq(5, n, by = 8)[1:8]
    X1_m <- X1
    X1_m[miss1, ] <- NA
    X2_m <- X2
    X2_m[miss2, ] <- NA
    anchor_idx <- setdiff(seq_len(n), union(miss1, miss2))
    list(
        X1 = X1_m, X2 = X2_m, X1_full = X1, X2_full = X2,
        groups = groups, anchor_idx = anchor_idx, K = K, n = n
    )
}

test_that("woven() returns object of class 'woven'", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    expect_s3_class(fit, "woven")
})

test_that("woven() print method runs without error", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    expect_output(print(fit), "WOVEN fit")
})

test_that("woven() uses mcca_dual for all V", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    expect_false(is.null(fit$fit_mcca))
    expect_null(fit$fit_v2)
    expect_null(fit$fit_als)
    expect_length(fit$W_list, 2L)
    expect_equal(nrow(fit$W_list[[1]]), ncol(d$X1))
    expect_equal(ncol(fit$W_list[[1]]), d$K)
})

test_that("woven() scalar lambda broadcasts to all V modalities", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, lambdas = 0.5, verbose = FALSE
    )
    expect_equal(fit$lambdas, c(0.5, 0.5))
})

test_that("woven() errors if anchor_idx shorter than K", {
    d <- make_fixture()
    expect_error(
        woven(list(d$X1, d$X2), Y = d$groups, anchor_idx = 1:2, K = 5L),
        regexp = "anchor"
    )
})

test_that("woven_predict() returns data.frame with correct nrow", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    n_new <- 10L
    set.seed(1)
    X1_new <- matrix(rnorm(n_new * ncol(d$X1)), n_new, ncol(d$X1))
    X2_new <- matrix(rnorm(n_new * ncol(d$X2)), n_new, ncol(d$X2))
    pred <- woven_predict(fit, list(X1_new, X2_new))
    expect_s3_class(pred, "data.frame")
    expect_equal(nrow(pred), n_new)
    expect_true("predicted_class" %in% names(pred))
    expect_true("confidence" %in% names(pred))
    # Confidence in [0,1]
    expect_true(all(pred$confidence >= 0 & pred$confidence <= 1))
})

test_that("woven_predict() knn method also works", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    set.seed(2)
    X1_new <- matrix(rnorm(5 * ncol(d$X1)), 5, ncol(d$X1))
    X2_new <- matrix(rnorm(5 * ncol(d$X2)), 5, ncol(d$X2))
    pred <- woven_predict(fit, list(X1_new, X2_new), method = "knn")
    expect_equal(nrow(pred), 5L)
})

test_that("woven_predict() handles block-missing new subjects", {
    d <- make_fixture()
    fit <- woven(list(d$X1, d$X2),
        Y = d$groups, anchor_idx = d$anchor_idx,
        K = d$K, verbose = FALSE
    )
    set.seed(3)
    X1_new <- matrix(rnorm(8 * ncol(d$X1)), 8, ncol(d$X1))
    X2_new <- matrix(rnorm(8 * ncol(d$X2)), 8, ncol(d$X2))
    X1_new[c(2, 5), ] <- NA # 2 subjects missing modality 1
    pred <- woven_predict(fit, list(X1_new, X2_new))
    expect_equal(nrow(pred), 8L)
    # Subjects 2 and 5 should still get a prediction (via modality 2 only)
    expect_true(all(!is.na(pred$predicted_class)))
})
