# Small reproducible fixture
make_X <- function(n = 30, p = 20, seed = 1L) {
    set.seed(seed)
    matrix(rnorm(n * p), n, p)
}

test_that("build_laplacian returns valid combinatorial Laplacian", {
    X <- make_X()
    L <- build_laplacian(X, k = 5L)
    expect_equal(dim(L), c(30L, 30L))
    # Diagonal >= 0 (use Matrix::diag to avoid sparse long-vector issue)
    d <- Matrix::diag(L)
    expect_true(all(d >= 0))
    # Row sums == 0 (combinatorial Laplacian property)
    expect_true(all(abs(Matrix::rowSums(L)) < 1e-10))
    # Symmetric
    diff_sym <- max(abs(L - Matrix::t(L)))
    expect_lt(diff_sym, 1e-12)
})

test_that("build_laplacian handles block-missing rows", {
    X <- make_X(n = 30, p = 20)
    X[c(1, 5, 10), ] <- NA # 3 fully-missing rows
    L <- build_laplacian(X, k = 5L)
    expect_equal(dim(L), c(30L, 30L))
    # Missing rows get zero degree (no edges)
    d <- Matrix::diag(L)
    expect_equal(as.numeric(d[c(1, 5, 10)]), c(0, 0, 0))
})

test_that("build_laplacian handles feature-level NAs via imputation", {
    X <- make_X(n = 20, p = 15)
    X[3, 5] <- NA
    X[7, 2] <- NA # scattered NAs, rows not all-NA
    expect_no_error(build_laplacian(X, k = 4L))
})
