#!/usr/bin/env Rscript
# aggregate_results.R — combine per-rep benchmark RDS files into one summary table
#
# Usage: Rscript aggregate_results.R <results_dir> <out_csv>
# Example: Rscript aggregate_results.R ~/woven/results/benchmark_final ~/woven/results/summary_final.csv

`%||` <- function(a, b) if (!is.null(a)) a else b

args    <- commandArgs(trailingOnly = TRUE)
res_dir <- if (length(args) >= 1) args[1] else "~/woven/results/benchmark_final"
out_csv <- if (length(args) >= 2) args[2] else "~/woven/results/summary_final.csv"

res_dir <- path.expand(res_dir)
out_csv <- path.expand(out_csv)

files <- list.files(res_dir, pattern = "_benchmark\\.rds$", full.names = TRUE)
cat(sprintf("Found %d result files in %s\n", length(files), res_dir))

metric_names <- c("silhouette", "davies_bouldin", "nmi", "ess_retention",
                  "rv_coefficient", "effect_bias", "nystrom_error",
                  "silhouette_anchor", "davies_bouldin_anchor", "nmi_anchor", "ess_retention_anchor",
                  "ber", "ber_anchor")

rows <- list()
for (f in files) {
  obj <- tryCatch(readRDS(f), error = function(e) NULL)
  if (is.null(obj)) { cat("SKIP (unreadable):", f, "\n"); next }
  for (cond in names(obj$results)) {
    for (method in names(obj$results[[cond]])) {
      r <- obj$results[[cond]][[method]]
      row <- data.frame(
        arm       = obj$arm,
        rep       = obj$rep,
        condition = cond,
        method    = method,
        n         = obj$metadata$n,
        V         = obj$metadata$V,
        n_used    = r$n_used   %||% NA_integer_,
        elapsed   = r$elapsed  %||% NA_real_,
        error_msg = r$error    %||% "",
        stringsAsFactors = FALSE
      )
      for (m in metric_names) row[[m]] <- r[[m]] %||% NA_real_
      rows[[length(rows) + 1L]] <- row
    }
  }
}

df <- do.call(rbind, rows)
write.csv(df, out_csv, row.names = FALSE)
cat(sprintf("Written %d rows to %s\n", nrow(df), out_csv))

ok <- df[df$error_msg == "", ]

# ── Timing CSV ──────────────────────────────────────────────────────────────────
timing_csv <- sub("\\.csv$", "_timing.csv", out_csv)
timing_rows <- list()
for (arm in sort(unique(ok$arm))) {
  for (cond in sort(unique(ok$condition))) {
    for (method in sort(unique(ok$method))) {
      sub <- ok[ok$arm == arm & ok$condition == cond & ok$method == method, ]
      if (nrow(sub) == 0) next
      timing_rows[[length(timing_rows) + 1L]] <- data.frame(
        arm = arm, condition = cond, method = method,
        n_reps     = nrow(sub),
        mean_sec   = round(mean(sub$elapsed, na.rm = TRUE), 1),
        sd_sec     = round(sd(sub$elapsed,   na.rm = TRUE), 1),
        median_sec = round(median(sub$elapsed, na.rm = TRUE), 1),
        stringsAsFactors = FALSE
      )
    }
  }
}
timing_df <- do.call(rbind, timing_rows)
write.csv(timing_df, timing_csv, row.names = FALSE)
cat(sprintf("Timing table written to %s\n", timing_csv))

# ── Metric summary ──────────────────────────────────────────────────────────────
cat("\n=== Mean metrics by method x condition (all arms) ===\n")
for (cond in c("complete","mcar30","mcar50","mar")) {
  if (!cond %in% ok$condition) next
  cat(sprintf("\n--- %s ---\n", cond))
  sub <- ok[ok$condition == cond, ]
  for (method in c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")) {
    if (!method %in% sub$method) next
    m <- sub[sub$method == method, ]
    cat(sprintf("  %-14s sil=%.3f  sil_anc=%.3f  NMI=%.3f  ESS=%.2f  RV=%.3f  n=%d\n",
      method,
      mean(m$silhouette,        na.rm=TRUE),
      mean(m$silhouette_anchor, na.rm=TRUE),
      mean(m$nmi,               na.rm=TRUE),
      mean(m$ess_retention,     na.rm=TRUE),
      mean(m$rv_coefficient,    na.rm=TRUE),
      nrow(m)
    ))
  }
}

# ── Timing summary ──────────────────────────────────────────────────────────────
cat("\n=== Mean runtime (seconds) per method x condition ===\n")
cat(sprintf("  %-14s  %-10s  %-8s  %-8s  %-8s  %-8s\n",
            "Method", "Condition", "ARM A", "ARM B", "ARM C", "ARM D"))
for (method in c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")) {
  for (cond in c("complete","mcar30","mcar50","mar")) {
    arm_times <- sapply(c("A","B","C","D"), function(arm) {
      sub <- ok[ok$method==method & ok$condition==cond & ok$arm==arm, ]
      if (nrow(sub) == 0) return(NA_real_)
      mean(sub$elapsed, na.rm=TRUE)
    })
    cat(sprintf("  %-14s  %-10s  %7.0fs  %7.0fs  %7.0fs  %7.0fs\n",
      method, cond,
      arm_times["A"], arm_times["B"], arm_times["C"], arm_times["D"]))
  }
}

# ── Paper timing table: method x arm, complete condition only ──────────────────
cat("\n=== Paper timing table: complete-data condition, mean (SD) seconds ===\n")
cat(sprintf("  %-14s  %-16s  %-16s  %-16s  %-16s\n",
            "Method", "ARM A (V=2 G)", "ARM B (V=3 G)", "ARM C (V=3 IS)", "ARM D (V=2 N)"))
for (method in c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")) {
  vals <- sapply(c("A","B","C","D"), function(arm) {
    sub <- ok[ok$method==method & ok$condition=="complete" & ok$arm==arm, ]
    if (nrow(sub) == 0) return("     N/A")
    sprintf("%5.0f (%4.0f)", mean(sub$elapsed, na.rm=TRUE), sd(sub$elapsed, na.rm=TRUE))
  })
  cat(sprintf("  %-14s  %-16s  %-16s  %-16s  %-16s\n", method, vals[1], vals[2], vals[3], vals[4]))
}
cat("\nG=Gaussian, IS=InterSIM, N=NorTA\n")
