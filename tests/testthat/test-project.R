
make_fixture <- function(n = 60, p1 = 80, p2 = 40, K = 3L, seed = 1L) {
  set.seed(seed)
  groups <- rep(1:2, each = n / 2)
  X1 <- matrix(rnorm(n * p1), n, p1)
  X2 <- matrix(rnorm(n * p2), n, p2)
  miss1 <- 1:10; miss2 <- 11:20
  X1_m <- X1; X1_m[miss1, ] <- NA
  X2_m <- X2; X2_m[miss2, ] <- NA
  anchor_idx <- setdiff(seq_len(n), union(miss1, miss2))
  list(X1 = X1_m, X2 = X2_m, X1_full = X1, X2_full = X2,
       groups = groups, anchor_idx = anchor_idx, K = K, n = n)
}

test_that("woven_project scores all subjects with no NAs", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  X1_full = d$X1_full, X2_full = d$X2_full)
  proj <- woven_project(fit, d$X1, d$X2)

  expect_equal(nrow(proj$Z), d$n)
  expect_equal(ncol(proj$Z), d$K)
  expect_equal(sum(is.na(proj$Z[, 1])), 0L)
})

test_that("woven_project direct scores are weighted combo of Z1 and Z2", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  X1_full = d$X1_full, X2_full = d$X2_full)
  proj <- woven_project(fit, d$X1, d$X2)

  # Anchors should be scored "direct" — their projection should be close to
  # the feature-weighted consensus of Z1 and Z2
  p1e <- length(fit$col_ok1); p2e <- length(fit$col_ok2)
  Z_expected <- (p1e * fit$Z1 + p2e * fit$Z2) / (p1e + p2e)
  Z_proj_anchors <- proj$Z[d$anchor_idx, ]
  expect_lt(max(abs(Z_expected - Z_proj_anchors)), 1e-6)
})

test_that("woven_project method labels are correct", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  X1_full = d$X1_full, X2_full = d$X2_full)
  proj <- woven_project(fit, d$X1, d$X2)

  expect_true(all(proj$method[d$anchor_idx] == "direct"))
  nystrom_idx <- setdiff(seq_len(d$n), d$anchor_idx)
  expect_true(all(proj$method[nystrom_idx] %in% c("partial", "missing")))
})
