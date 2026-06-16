#!/usr/bin/env Rscript
# aggregate_dense_diablo.R -- Combine dense DIABLO ARM D per-rep RDS → summary CSV
#
# Usage:
#   Rscript aggregate_dense_diablo.R <results_dir> <out_csv>

args        <- commandArgs(trailingOnly = TRUE)
results_dir <- args[1]
out_csv     <- args[2]

files <- list.files(results_dir, pattern = "^arm_D_rep_.*_dense_diablo\\.rds$", full.names = TRUE)
cat(sprintf("Found %d rep files\n", length(files)))

all_rows <- lapply(files, function(f) {
  tryCatch({
    d <- readRDS(f)
    do.call(rbind, lapply(names(d$results), function(cond) {
      r <- d$results[[cond]]
      data.frame(rep=d$rep, arm=d$arm, condition=cond, method="DenseDIABLO",
        sil=r$sil, ber=r$ber, nmi=r$nmi, n_used=r$n_used,
        ess=r$ess, elapsed=r$elapsed, error=r$error %||% NA_character_,
        stringsAsFactors=FALSE)
    }))
  }, error = function(e) {
    cat(sprintf("  SKIP: %s\n", basename(f)))
    NULL
  })
})
all_rows <- do.call(rbind, Filter(Negate(is.null), all_rows))
cat(sprintf("Total rows: %d\n", nrow(all_rows)))

`%||%` <- function(a, b) if (!is.null(a) && !is.na(a)) a else b

summary_df <- do.call(rbind, lapply(
  split(all_rows, list(all_rows$condition)),
  function(g) {
    data.frame(
      condition   = g$condition[1],
      method      = "DenseDIABLO",
      n_reps      = nrow(g),
      sil_mean    = mean(g$sil, na.rm=TRUE),
      sil_sd      = sd(g$sil, na.rm=TRUE),
      ber_mean    = mean(g$ber, na.rm=TRUE),
      ber_sd      = sd(g$ber, na.rm=TRUE),
      nmi_mean    = mean(g$nmi, na.rm=TRUE),
      ess_mean    = mean(g$ess, na.rm=TRUE),
      stringsAsFactors = FALSE
    )
  }
))

write.csv(all_rows, sub("\\.csv$", "_raw.csv", out_csv), row.names=FALSE)
write.csv(summary_df, out_csv, row.names=FALSE)
cat(sprintf("Saved: %s\n", out_csv))
print(summary_df)
