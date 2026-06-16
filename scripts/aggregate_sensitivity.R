#!/usr/bin/env Rscript
# aggregate_sensitivity.R — combine per-rep sensitivity RDS into summary and figures
#
# Usage: Rscript aggregate_sensitivity.R <results_dir> <out_dir>

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

args    <- commandArgs(trailingOnly = TRUE)
res_dir <- path.expand(if (length(args) >= 1) args[1] else "~/woven/results/sensitivity")
out_dir <- path.expand(if (length(args) >= 2) args[2] else res_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(res_dir, pattern = "^sensitivity_rep_.*\\.rds$", full.names = TRUE)
cat(sprintf("Found %d rep files\n", length(files)))

df <- do.call(rbind, lapply(files, function(f) {
  tryCatch(readRDS(f), error = function(e) { cat("SKIP:", f, "\n"); NULL })
}))

write.csv(df, file.path(out_dir, "sensitivity_all.csv"), row.names = FALSE)

ok <- df[df$error == "" & !is.na(df$sil_anchor), ]

smry <- ok %>%
  group_by(parameter, value) %>%
  summarise(
    mean_sil = mean(sil_anchor, na.rm = TRUE),
    se_sil   = sd(sil_anchor,   na.rm = TRUE) / sqrt(n()),
    n_reps   = n(),
    .groups  = "drop"
  )

# Defaults used in benchmark
defaults <- c(lambda = 0.1, gamma_y = 1.0, k_nn = 15)

param_labels <- c(
  lambda  = expression(lambda ~ "(Laplacian regularization)"),
  gamma_y = expression(gamma[Y] ~ "(label supervision strength)"),
  k_nn    = "k-NN (graph neighbors)"
)

plots <- lapply(c("lambda", "gamma_y", "k_nn"), function(param) {
  sub   <- smry[smry$parameter == param, ]
  dflt  <- defaults[param]
  xscale <- if (param %in% c("lambda", "gamma_y")) "log10" else "identity"

  p <- ggplot(sub, aes(x = value, y = mean_sil)) +
    geom_ribbon(aes(ymin = mean_sil - se_sil, ymax = mean_sil + se_sil),
                fill = "#2166AC", alpha = 0.2) +
    geom_line(colour = "#2166AC", linewidth = 1) +
    geom_point(colour = "#2166AC", size = 3) +
    geom_vline(xintercept = dflt, linetype = "dashed",
               colour = "grey40", linewidth = 0.6) +
    annotate("text", x = dflt, y = Inf, label = "default",
             vjust = 1.4, hjust = -0.1, size = 3, colour = "grey40") +
    scale_y_continuous(limits = c(
      min(sub$mean_sil - sub$se_sil, na.rm = TRUE) * 0.95,
      max(sub$mean_sil + sub$se_sil, na.rm = TRUE) * 1.05
    )) +
    labs(x = param_labels[[param]],
         y = "Anchor silhouette (mean ± SE)",
         title = param_labels[[param]]) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(size = 10))

  if (xscale == "log10")
    p <- p + scale_x_log10(breaks = sub$value,
                            labels = as.character(sub$value))
  p
})

pdf(file.path(out_dir, "fig_sensitivity.pdf"), width = 11, height = 4)
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  print(plots[[1]] | plots[[2]] | plots[[3]])
} else {
  gridExtra::grid.arrange(grobs = plots, ncol = 3)
}
dev.off()
cat(sprintf("Written: %s\n", file.path(out_dir, "fig_sensitivity.pdf")))

# Print summary table
cat("\n=== Sensitivity summary ===\n")
for (param in c("lambda", "gamma_y", "k_nn")) {
  cat(sprintf("\n%s (default = %s):\n", param, defaults[param]))
  sub <- smry[smry$parameter == param, ]
  for (i in seq_len(nrow(sub))) {
    marker <- if (abs(sub$value[i] - defaults[param]) < 1e-9) " <-- default" else ""
    cat(sprintf("  %6.3f  sil=%.4f (%.4f SE) n=%d%s\n",
                sub$value[i], sub$mean_sil[i], sub$se_sil[i], sub$n_reps[i], marker))
  }
}
