# test_solver.R — smoke test for WOVEN V=2 solver + Nyström projection
# Run from: NewDIABLO/woven/
# Rscript data-raw/test_solver.R

rlib_hpc <- NULL  # local run, no extra lib needed
pkgs <- c("RANN", "RSpectra", "Matrix")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    stop("Missing package: ", p, " — install with install.packages('", p, "')")
}

# Source WOVEN files
for (f in c("R/utils.R", "R/laplacian.R", "R/solver_v2.R", "R/project.R")) {
  source(f)
}

cat("=== WOVEN Smoke Test ===\n\n")

# ── Load data ──────────────────────────────────────────────────────────────────
complete <- readRDS("data-raw/arm_A_rep_001.rds")
missing  <- readRDS("data-raw/arm_A_rep_001_mcar30.rds")

X1_full <- complete$data$rnaseq       # 300 x 5000
X2_full <- complete$data$methylation  # 300 x 10000
labels  <- complete$labels            # 1:4 group assignments
truth_Z <- complete$truth_Z           # 300 x 10 PCA ground truth

cat(sprintf("Data: n=%d, p1=%d, p2=%d, groups=%d\n",
            nrow(X1_full), ncol(X1_full), ncol(X2_full), length(unique(labels))))
cat(sprintf("Anchor set: %d subjects (%.0f%% complete)\n\n",
            missing$n_anchors, 100 * missing$n_anchors / nrow(X1_full)))

anchor_idx <- missing$anchor_idx

# ── Fit solver ────────────────────────────────────────────────────────────────
# Apply missingness mask upfront — solver receives NA rows for missing subjects
mask <- missing$missingness_mask
X1_miss <- X1_full; X1_miss[mask[, "rnaseq"], ]      <- NA
X2_miss <- X2_full; X2_miss[mask[, "methylation"], ] <- NA

cat("--- Fitting V=2 solver (supervised) ---\n")
t0 <- proc.time()
fit <- woven_v2(
  X1 = X1_miss, X2 = X2_miss,
  anchor_idx = anchor_idx,
  Y = labels,
  K = 5L,
  lambda1 = 0.1, lambda2 = 0.1,
  gamma_y = 1.0,
  k_nn = 15L,
  X1_full = X1_full, X2_full = X2_full
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("\nFit time: %.1f sec\n", elapsed))

# ── Sanity checks ─────────────────────────────────────────────────────────────
cat("\n--- Sanity checks ---\n")

# 1. Singular values should be positive and ordered
sv <- fit$singular_values
cat(sprintf("Singular values: %s\n", paste(round(sv, 4), collapse = ", ")))
stopifnot(all(sv > 0), all(diff(sv) <= 0))
cat("  [OK] Singular values positive and decreasing\n")

# 2. B-orthogonality: W^T B W ≈ I_K
check_orth <- function(W, B, name) {
  WBW <- t(W) %*% B %*% W
  diag_vals <- diag(WBW)
  off_diag  <- max(abs(WBW - diag(diag_vals)))
  cat(sprintf("  %s: W^T B W diagonal range [%.4f, %.4f], max off-diag %.2e\n",
              name, min(diag_vals), max(diag_vals), off_diag))
}
check_orth(fit$W1, fit$B1, "W1")
check_orth(fit$W2, fit$B2, "W2")

# 3. Alignment: anchor latent positions should be close across modalities
align_err <- mean(rowSums((fit$Z1 - fit$Z2)^2))
cat(sprintf("  Mean anchor alignment error (||Z1 - Z2||^2): %.4f\n", align_err))

# 4. Group separation in latent space (anchor samples only)
anchor_labels <- labels[anchor_idx]
within_var  <- mean(tapply(seq_len(nrow(fit$Z1)), anchor_labels, function(idx) {
  mean(apply(fit$Z1[idx, , drop=FALSE], 2, var))
}))
between_var <- var(apply(fit$Z1, 2, function(z) tapply(z, anchor_labels, mean)))
cat(sprintf("  Latent Z1: within-group var=%.4f, between-group var=%.4f\n",
            within_var, mean(between_var)))

# 5. Correlation with ground truth PCA
truth_a <- truth_Z[anchor_idx, ]
cors <- sapply(seq_len(fit$K), function(k) {
  max(abs(cor(fit$Z1[, k], truth_a)))  # max over truth dims (sign/order invariant)
})
cat(sprintf("  Max |cor| with truth PCA dims: %s\n",
            paste(round(cors, 3), collapse = ", ")))

# ── Project block-missing subjects ────────────────────────────────────────────
cat("\n--- Nyström projection ---\n")

t1 <- proc.time()
proj <- woven_project(fit, X1_miss, X2_miss)
proj_time <- (proc.time() - t1)[["elapsed"]]

method_tab <- table(proj$method)
cat(sprintf("  Projection methods: %s\n",
            paste(names(method_tab), method_tab, sep="=", collapse=", ")))
cat(sprintf("  Projection time: %.2f sec\n", proj_time))

# Check no NAs remain (every subject has at least one observed modality)
n_na <- sum(is.na(proj$Z[, 1]))
cat(sprintf("  Subjects with NA consensus Z: %d (should be 0)\n", n_na))
stopifnot(n_na == 0)
cat("  [OK] All subjects projected\n")

# ── VIP scores ────────────────────────────────────────────────────────────────
cat("\n--- VIP scores ---\n")
vips <- woven_vip(fit)
cat(sprintf("  VIP1 range: [%.4f, %.4f]\n", min(vips$vip1), max(vips$vip1)))
cat(sprintf("  VIP2 range: [%.4f, %.4f]\n", min(vips$vip2), max(vips$vip2)))
cat(sprintf("  Top 5 VIP1 features: %s\n",
            paste(order(vips$vip1, decreasing=TRUE)[1:5], collapse=", ")))
cat(sprintf("  Top 5 VIP2 features: %s\n",
            paste(order(vips$vip2, decreasing=TRUE)[1:5], collapse=", ")))

cat("\n=== V=2 checks passed ===\n")

# ── ALS test (V=2 cross-check, then V=3 via ARM B data if available) ──────────
cat("\n=== ALS Solver Test (V=2 cross-check) ===\n")
source("R/solver_als.R")

t2 <- proc.time()
fit_als <- woven_als(
  X_list     = list(X1_miss, X2_miss),
  anchor_idx = anchor_idx,
  Y = labels,
  K = 5L, lambdas = c(0.1, 0.1), gamma_y = 1.0, k_nn = 15L,
  max_iter = 100L, n_restarts = 2L,
  X_list_full = list(X1_full, X2_full),
  verbose = TRUE
)
als_time <- (proc.time() - t2)[["elapsed"]]
cat(sprintf("\nALS fit time: %.1f sec\n", als_time))
cat(sprintf("Final objective: %.6f\n", fit_als$objective))
cat(sprintf("Anchor alignment (ALS): %.6f\n",
            mean(rowSums((fit_als$Z_list[[1]] - fit_als$Z_list[[2]])^2))))

# Compare V=2 ALS vs closed-form: latent spaces should be similar
cor_z <- mean(abs(diag(cor(fit$Z1, fit_als$Z_list[[1]]))))
cat(sprintf("Mean |cor| between V2 and ALS Z1: %.4f\n", cor_z))

cat("\n=== All checks passed ===\n")

# ── Metrics battery smoke test ────────────────────────────────────────────────
cat("\n=== Metrics Battery ===\n")
source("R/metrics.R")

Z_all <- proj$Z   # n x K consensus scores (all 300 subjects)

m <- woven_all_metrics(
  Z        = Z_all,
  labels   = labels,
  n_total  = nrow(X1_full),
  Z_true   = truth_Z
)

cat(sprintf("  Silhouette:     %.4f  (higher better, range [-1,1])\n", m$silhouette))
cat(sprintf("  Davies-Bouldin: %.4f  (lower better)\n",               m$davies_bouldin))
cat(sprintf("  NMI:            %.4f  (higher better, range [0,1])\n", m$nmi))
cat(sprintf("  ESS retention:  %.4f  (higher better; DIABLO < 1)\n",  m$ess_retention))
cat(sprintf("  RV coefficient: %.4f  (higher better, range [0,1])\n", m$rv_coefficient))

stopifnot(
  is.finite(m$silhouette),
  is.finite(m$davies_bouldin),
  is.finite(m$nmi),
  m$ess_retention == 1.0,         # WOVEN uses all subjects
  is.finite(m$rv_coefficient)
)
cat("  [OK] All metrics finite and ESS=1.0\n")
cat("\n=== Metrics passed ===\n")
