
# Reproducible synthetic fixture: 2-class problem with clear signal
make_fixture <- function(n = 60, p1 = 80, p2 = 40, K = 3L,
                         miss_frac = 0.3, seed = 42L) {
  set.seed(seed)
  # 2 groups with separation in first latent factor
  groups <- rep(1:2, each = n / 2)
  Z_true <- matrix(c(groups - 1.5, rnorm(n * (K - 1))), n, K)
  W1 <- matrix(rnorm(p1 * K), p1, K)
  W2 <- matrix(rnorm(p2 * K), p2, K)
  X1 <- Z_true %*% t(W1) + matrix(rnorm(n * p1, sd = 0.5), n, p1)
  X2 <- Z_true %*% t(W2) + matrix(rnorm(n * p2, sd = 0.5), n, p2)
  # Induce block missingness
  miss_idx <- sample(n, floor(n * miss_frac))
  miss1 <- miss_idx[seq_len(length(miss_idx) %/% 2)]
  miss2 <- miss_idx[seq(length(miss_idx) %/% 2 + 1, length(miss_idx))]
  X1_m <- X1; X1_m[miss1, ] <- NA
  X2_m <- X2; X2_m[miss2, ] <- NA
  anchor_idx <- setdiff(seq_len(n), union(miss1, miss2))
  list(X1 = X1_m, X2 = X2_m, X1_full = X1, X2_full = X2,
       groups = groups, anchor_idx = anchor_idx, K = K)
}

test_that("woven_v2 produces B-orthogonal projection matrices", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  lambda1 = 0.1, lambda2 = 0.1, gamma_y = 1.0,
                  X1_full = d$X1_full, X2_full = d$X2_full)

  WBW1 <- t(fit$W1) %*% fit$B1 %*% fit$W1
  WBW2 <- t(fit$W2) %*% fit$B2 %*% fit$W2

  # Diagonals should be 1
  expect_true(all(abs(diag(WBW1) - 1) < 1e-8),
              label = "W1 B-orthonormal diagonals")
  expect_true(all(abs(diag(WBW2) - 1) < 1e-8),
              label = "W2 B-orthonormal diagonals")
  # Off-diagonals should be near zero
  off1 <- max(abs(WBW1 - diag(diag(WBW1))))
  off2 <- max(abs(WBW2 - diag(diag(WBW2))))
  expect_lt(off1, 1e-8)
  expect_lt(off2, 1e-8)
})

test_that("woven_v2 anchor alignment error is near zero", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  lambda1 = 0.1, lambda2 = 0.1, gamma_y = 1.0,
                  X1_full = d$X1_full, X2_full = d$X2_full)
  align_err <- mean(rowSums((fit$Z1 - fit$Z2)^2))
  expect_lt(align_err, 1e-6)
})

test_that("woven_v2 singular values are positive and decreasing", {
  d <- make_fixture()
  fit <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                  lambda1 = 0.1, lambda2 = 0.1, gamma_y = 1.0,
                  X1_full = d$X1_full, X2_full = d$X2_full)
  sv <- fit$singular_values
  expect_true(all(sv > 0))
  expect_true(all(diff(sv) <= 1e-10))  # non-increasing
})

test_that("woven_v2 supervision improves group separation vs gamma_y=0", {
  d <- make_fixture(seed = 7L)
  Y <- d$groups

  fit_sup <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = Y, K = d$K,
                      lambda1 = 0.1, lambda2 = 0.1, gamma_y = 5.0,
                      X1_full = d$X1_full, X2_full = d$X2_full)
  fit_uns <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = Y, K = d$K,
                      lambda1 = 0.1, lambda2 = 0.1, gamma_y = 0.0,
                      X1_full = d$X1_full, X2_full = d$X2_full)

  # Between-class variance ratio in Z1
  between_var <- function(Z, y) {
    var(tapply(Z[, 1], y, mean))
  }

  Y_a <- Y[d$anchor_idx]
  bv_sup <- between_var(fit_sup$Z1, Y_a)
  bv_uns <- between_var(fit_uns$Z1, Y_a)

  # Supervised should have higher between-class variance
  expect_gt(bv_sup, bv_uns)
})

test_that("woven_v2 sparse W has fewer nonzero entries than dense", {
  d <- make_fixture()
  fit_dense <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                        gamma_y = 1.0, alpha1 = 0, alpha2 = 0,
                        X1_full = d$X1_full, X2_full = d$X2_full)
  # NIPALS thresholds in W-space directly; use 0.3 * max(|W1|) for ~50% sparsity
  alpha_use <- 0.3 * max(abs(fit_dense$W1))
  fit_sparse <- woven_v2(d$X1, d$X2, d$anchor_idx, Y = d$groups, K = d$K,
                         gamma_y = 1.0, alpha1 = alpha_use, alpha2 = alpha_use,
                         X1_full = d$X1_full, X2_full = d$X2_full)
  nnz_dense  <- sum(abs(fit_dense$W1)  > 1e-10)
  nnz_sparse <- sum(abs(fit_sparse$W1) > 1e-10)
  expect_lt(nnz_sparse, nnz_dense)
})
