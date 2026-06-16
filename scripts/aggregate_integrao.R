#!/usr/bin/env Rscript
# aggregate_integrao.R -- Combine IntegrAO per-rep RDS → summary CSV
#
# Usage:
#   Rscript aggregate_integrao.R <results_dir> <out_csv>

args        <- commandArgs(trailingOnly = TRUE)
results_dir <- args[1]
out_csv     <- args[2]

files <- list.files(results_dir, pattern = "^arm_._rep_.*_integrao\\.rds$", full.names = TRUE)
cat(sprintf("Found %d rep files\n", length(files)))

all_rows <- lapply(files, function(f) {
  tryCatch({
    d <- readRDS(f)
    do.call(rbind, lapply(names(d$results), function(cond) {
      r <- d$results[[cond]]
      data.frame(rep=d$rep, arm=d$arm, condition=cond, method="IntegrAO",
        sil=r$sil, nmi=r$nmi, ber=r$ber,
        n_used=r$n_used, ess=r$ess,
        elapsed=r$elapsed, error=r$error,
        stringsAsFactors=FALSE)
    }))
  }, error = function(e) {
    cat(sprintf("  SKIP: %s\n", basename(f)))
    NULL
  })
})
all_rows <- do.call(rbind, Filter(Negate(is.null), all_rows))
cat(sprintf("Total rows: %d\n", nrow(all_rows)))

summary_df <- do.call(rbind, lapply(
  split(all_rows, list(all_rows$arm, all_rows$condition)),
  function(g) {
    if (nrow(g) == 0L) return(NULL)
    data.frame(
      arm        = g$arm[1],
      condition  = g$condition[1],
      method     = "IntegrAO",
      n_reps     = sum(!is.na(g$sil)),
      sil_mean   = mean(g$sil, na.rm=TRUE),
      sil_sd     = sd(g$sil, na.rm=TRUE),
      nmi_mean   = mean(g$nmi, na.rm=TRUE),
      nmi_sd     = sd(g$nmi, na.rm=TRUE),
      ber_mean   = mean(g$ber, na.rm=TRUE),
      ber_sd     = sd(g$ber, na.rm=TRUE),
      ess_mean   = mean(g$ess, na.rm=TRUE),
      elapsed_mean = mean(g$elapsed, na.rm=TRUE),
      stringsAsFactors = FALSE
    )
  }
))
summary_df <- summary_df[order(summary_df$arm, summary_df$condition), ]

write.csv(all_rows, sub("\\.csv$", "_raw.csv", out_csv), row.names=FALSE)
write.csv(summary_df, out_csv, row.names=FALSE)
cat(sprintf("Saved: %s\n", out_csv))
print(summary_df[, c("arm","condition","n_reps","sil_mean","nmi_mean","ber_mean","ess_mean")])
