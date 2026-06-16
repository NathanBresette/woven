
test_that("woven_silhouette returns scalar in [-1, 1]", {
  set.seed(1); Z <- matrix(rnorm(100 * 5), 100, 5)
  labels <- rep(1:4, 25)
  s <- woven_silhouette(Z, labels)
  expect_true(is.finite(s))
  expect_true(s >= -1 && s <= 1)
})

test_that("woven_silhouette returns NA with one class", {
  Z <- matrix(rnorm(40 * 3), 40, 3)
  expect_true(is.na(woven_silhouette(Z, rep(1, 40))))
})

test_that("woven_davies_bouldin is non-negative", {
  set.seed(2); Z <- matrix(rnorm(80 * 4), 80, 4)
  labels <- rep(1:4, 20)
  db <- woven_davies_bouldin(Z, labels)
  expect_true(is.finite(db) && db >= 0)
})

test_that("woven_nmi is in [0, 1]", {
  set.seed(3); Z <- matrix(rnorm(60 * 3), 60, 3)
  labels <- rep(1:3, 20)
  nmi <- woven_nmi(Z, labels)
  expect_true(is.finite(nmi) && nmi >= 0 && nmi <= 1)
})

test_that("woven_nmi is 1 for perfectly separated clusters", {
  # Clusters perfectly separated — k-means should recover them exactly
  set.seed(4)
  Z <- rbind(
    matrix(rnorm(30 * 2, mean = 0),  30, 2),
    matrix(rnorm(30 * 2, mean = 10), 30, 2),
    matrix(rnorm(30 * 2, mean = 20), 30, 2)
  )
  labels <- rep(1:3, each = 30)
  nmi <- woven_nmi(Z, labels, n_cl = 3L)
  expect_gt(nmi, 0.95)
})

test_that("woven_rv returns value in [0, 1]", {
  set.seed(5)
  Z      <- matrix(rnorm(50 * 4), 50, 4)
  Z_true <- matrix(rnorm(50 * 5), 50, 5)
  rv <- woven_rv(Z, Z_true)
  expect_true(is.finite(rv) && rv >= 0 && rv <= 1)
})

test_that("woven_rv is 1 when matrices are identical", {
  set.seed(6)
  Z  <- matrix(rnorm(40 * 3), 40, 3)
  rv <- woven_rv(Z, Z)
  expect_gt(rv, 0.999)
})

test_that("woven_rv is higher for correlated than uncorrelated matrices", {
  set.seed(61)
  Z      <- matrix(rnorm(50 * 4), 50, 4)
  Z_corr <- Z + matrix(rnorm(50 * 4, sd = 0.1), 50, 4)   # nearly identical
  Z_rand <- matrix(rnorm(50 * 5), 50, 5)                  # random, uncorrelated
  expect_gt(woven_rv(Z, Z_corr), woven_rv(Z, Z_rand))
})

test_that("woven_ess_retention is n_used/n_total", {
  expect_equal(woven_ess_retention(145L, 300L), 145 / 300)
  expect_equal(woven_ess_retention(300L, 300L), 1.0)
})

test_that("woven_all_metrics returns finite values with supervised fit", {
  set.seed(7)
  n <- 60; K <- 3L
  groups <- rep(1:3, 20)
  X1 <- matrix(rnorm(n * 50), n, 50)
  X2 <- matrix(rnorm(n * 30), n, 30)
  miss <- 1:15; anchor_idx <- (max(miss) + 1):n
  X1_m <- X1; X1_m[miss, ] <- NA

  fit  <- woven_v2(X1_m, X2, anchor_idx, Y = groups, K = K,
                   lambda1 = 0.1, lambda2 = 0.1, gamma_y = 1.0)
  proj <- woven_project(fit, X1_m, X2)
  Z_true <- matrix(rnorm(n * K), n, K)

  m <- woven_all_metrics(proj$Z, groups, n_total = n, Z_true = Z_true)

  expect_true(is.finite(m$silhouette))
  expect_true(is.finite(m$davies_bouldin))
  expect_true(is.finite(m$nmi))
  expect_equal(m$ess_retention, 1.0)
  expect_true(is.finite(m$rv_coefficient))
})
