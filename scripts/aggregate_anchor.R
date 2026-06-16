#!/usr/bin/env Rscript
# aggregate_anchor.R -- Combine per-rep anchor fraction sensitivity RDS into summary CSV
#
# Usage:
#   Rscript aggregate_anchor.R <results_dir> <out_csv>
#
# Output columns: arm, anchor_frac, n_anchor (mean), sil_all, sil_anchor,
#   sil_nonanc, nmi, n_scored (mean), n_reps (count)

args       <- commandArgs(trailingOnly = TRUE)
results_dir <- args[1]
out_csv     <- args[2]

files <- list.files(results_dir, pattern = "^anchor_rep_.*\\.rds$", full.names = TRUE)
cat(sprintf("Found %d rep files in %s\n", length(files), results_dir))

all_rows <- lapply(files, function(f) {
  tryCatch(readRDS(f), error = function(e) {
    cat(sprintf("  SKIP (error): %s\n", basename(f)))
    NULL
  })
})
all_rows <- do.call(rbind, Filter(Negate(is.null), all_rows))
cat(sprintf("Total rows: %d\n", nrow(all_rows)))

# Summarize: mean and SD per arm x anchor_frac
library(stats)
summary_df <- do.call(rbind, lapply(split(all_rows, list(all_rows$arm, all_rows$anchor_frac)), function(g) {
  if (nrow(g) == 0L) return(NULL)
  data.frame(
    arm         = g$arm[1],
    anchor_frac = g$anchor_frac[1],
    n_anchor_mean = mean(g$n_anchor, na.rm=TRUE),
    sil_all     = mean(g$sil_all, na.rm=TRUE),
    sil_all_sd  = sd(g$sil_all, na.rm=TRUE),
    sil_anchor  = mean(g$sil_anchor, na.rm=TRUE),
    sil_anchor_sd = sd(g$sil_anchor, na.rm=TRUE),
    sil_nonanc  = mean(g$sil_nonanc, na.rm=TRUE),
    sil_nonanc_sd = sd(g$sil_nonanc, na.rm=TRUE),
    nmi         = mean(g$nmi, na.rm=TRUE),
    nmi_sd      = sd(g$nmi, na.rm=TRUE),
    n_scored_mean = mean(g$n_scored, na.rm=TRUE),
    n_reps      = sum(!is.na(g$sil_all)),
    stringsAsFactors = FALSE
  )
}))
summary_df <- summary_df[order(summary_df$arm, summary_df$anchor_frac), ]

write.csv(summary_df, out_csv, row.names=FALSE)
cat(sprintf("Saved: %s\n", out_csv))
print(summary_df[, c("arm", "anchor_frac", "sil_all", "sil_anchor", "sil_nonanc", "nmi", "n_reps")])
