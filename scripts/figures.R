#!/usr/bin/env Rscript
# figures.R — paper figures for WOVEN benchmark results
#
# Usage: Rscript figures.R <summary_csv> <out_dir>
# Default: uses summary_final.csv; falls back to summary_v2_400.csv if not found
#
# Outputs:
#   fig1_main.pdf       — 2-panel: ESS retention + anchor silhouette (main figure)
#   fig2_perarm.pdf     — per-arm anchor silhouette breakdown (supplementary)
#   fig3_timing.pdf     — runtime comparison across methods and arms (supplementary)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
})

args    <- commandArgs(trailingOnly = TRUE)
csv_in  <- if (length(args) >= 1) args[1] else {
  candidates <- c(
    "~/woven/results/summary_final.csv",
    "~/woven/results/summary_v2_400.csv"
  )
  candidates <- path.expand(candidates)
  candidates[file.exists(candidates)][1]
}
out_dir <- if (length(args) >= 2) args[2] else dirname(csv_in)

csv_in  <- path.expand(csv_in)
out_dir <- path.expand(out_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Reading: %s\nOutput:  %s\n", csv_in, out_dir))

df <- read.csv(csv_in, na.strings = c("NA", ""), stringsAsFactors = FALSE)
ok <- df[df$error_msg == "" | is.na(df$error_msg), ]

# ── Shared aesthetics ─────────────────────────────────────────────────────────
method_order <- c("WOVEN", "DIABLO", "MOFA2", "ImputeDIABLO")
cond_order   <- c("complete", "mcar30", "mcar50", "mar")
cond_labels  <- c(complete = "Complete", mcar30 = "MCAR 30%",
                  mcar50 = "MCAR 50%", mar = "MAR")
arm_labels   <- c(A = "ARM A\n(V=2, Gaussian)",
                  B = "ARM B\n(V=3, Gaussian)",
                  C = "ARM C\n(V=3, InterSIM)",
                  D = "ARM D\n(V=2, NorTA)")

method_colors <- c(
  WOVEN        = "#2166AC",
  DIABLO       = "#D6604D",
  MOFA2        = "#4DAC26",
  ImputeDIABLO = "#878787"
)
method_shapes <- c(WOVEN = 16, DIABLO = 17, MOFA2 = 15, ImputeDIABLO = 18)

ok$condition <- factor(ok$condition, levels = cond_order, labels = cond_labels)
ok$method    <- factor(ok$method,    levels = method_order)

# Drop ImputeDIABLO from complete condition (not run there)
ok <- ok[!(ok$method == "ImputeDIABLO" & ok$condition == "Complete"), ]

# ── Summarise ─────────────────────────────────────────────────────────────────
smry <- ok %>%
  group_by(method, condition) %>%
  summarise(
    sil_anc_mean = mean(silhouette_anchor, na.rm = TRUE),
    sil_anc_se   = sd(silhouette_anchor,   na.rm = TRUE) / sqrt(sum(!is.na(silhouette_anchor))),
    sil_mean     = mean(silhouette,        na.rm = TRUE),
    sil_se       = sd(silhouette,          na.rm = TRUE) / sqrt(sum(!is.na(silhouette))),
    ess_mean     = mean(ess_retention,     na.rm = TRUE),
    ess_se       = sd(ess_retention,       na.rm = TRUE) / sqrt(sum(!is.na(ess_retention))),
    nmi_mean     = mean(nmi,               na.rm = TRUE),
    nmi_se       = sd(nmi,                 na.rm = TRUE) / sqrt(sum(!is.na(nmi))),
    .groups = "drop"
  )

smry_arm <- ok %>%
  group_by(method, condition, arm) %>%
  summarise(
    sil_anc_mean = mean(silhouette_anchor, na.rm = TRUE),
    sil_anc_se   = sd(silhouette_anchor,   na.rm = TRUE) / sqrt(sum(!is.na(silhouette_anchor))),
    ess_mean     = mean(ess_retention,     na.rm = TRUE),
    .groups = "drop"
  )

# ── Fig 1: ESS + anchor silhouette (main figure) ──────────────────────────────
pd <- position_dodge(width = 0.5)

p_ess <- ggplot(smry, aes(x = condition, y = ess_mean, colour = method,
                           group = method, shape = method)) +
  geom_line(position = pd, linewidth = 0.7) +
  geom_point(position = pd, size = 3) +
  geom_errorbar(aes(ymin = ess_mean - ess_se, ymax = ess_mean + ess_se),
                position = pd, width = 0.2, linewidth = 0.5) +
  scale_colour_manual(values = method_colors, name = NULL) +
  scale_shape_manual(values = method_shapes, name = NULL) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25),
                     labels = percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Effective sample size retention",
       title = "A") +
  theme_bw(base_size = 11) +
  theme(
    legend.position    = "bottom",
    legend.key.width   = unit(1.5, "cm"),
    panel.grid.minor   = element_blank(),
    plot.title         = element_text(face = "bold", size = 13),
    axis.text.x        = element_text(angle = 20, hjust = 1)
  )

p_sil <- ggplot(smry, aes(x = condition, y = sil_anc_mean, colour = method,
                           group = method, shape = method)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
  geom_line(position = pd, linewidth = 0.7) +
  geom_point(position = pd, size = 3) +
  geom_errorbar(aes(ymin = sil_anc_mean - sil_anc_se,
                    ymax = sil_anc_mean + sil_anc_se),
                position = pd, width = 0.2, linewidth = 0.5) +
  scale_colour_manual(values = method_colors, name = NULL) +
  scale_shape_manual(values = method_shapes, name = NULL) +
  scale_y_continuous(limits = c(-0.05, 0.65)) +
  labs(x = NULL, y = "Silhouette score (anchor subjects only)",
       title = "B") +
  theme_bw(base_size = 11) +
  theme(
    legend.position    = "bottom",
    legend.key.width   = unit(1.5, "cm"),
    panel.grid.minor   = element_blank(),
    plot.title         = element_text(face = "bold", size = 13),
    axis.text.x        = element_text(angle = 20, hjust = 1)
  )

# Shared legend via cowplot-style patchwork — use gridExtra as fallback
fig1_path <- file.path(out_dir, "fig1_main.pdf")
pdf(fig1_path, width = 9, height = 4.5)
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  print((p_ess | p_sil) + plot_layout(guides = "collect") &
          theme(legend.position = "bottom"))
} else {
  gridExtra::grid.arrange(p_ess, p_sil, ncol = 2)
}
dev.off()
cat(sprintf("Written: %s\n", fig1_path))

# ── Fig 2: Per-arm anchor silhouette (supplementary) ─────────────────────────
smry_arm$arm_label <- factor(smry_arm$arm, levels = c("A","B","C","D"),
                              labels = arm_labels)

p_arm <- ggplot(smry_arm[smry_arm$condition %in% c("Complete","MCAR 30%","MCAR 50%","MAR"), ],
                aes(x = condition, y = sil_anc_mean, colour = method,
                    group = method, shape = method)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
  geom_line(position = pd, linewidth = 0.6) +
  geom_point(position = pd, size = 2.5) +
  geom_errorbar(aes(ymin = sil_anc_mean - sil_anc_se,
                    ymax = sil_anc_mean + sil_anc_se),
                position = pd, width = 0.25, linewidth = 0.4) +
  facet_wrap(~ arm_label, nrow = 1) +
  scale_colour_manual(values = method_colors, name = NULL) +
  scale_shape_manual(values = method_shapes, name = NULL) +
  labs(x = NULL, y = "Silhouette score (anchor subjects only)",
       title = "Anchor silhouette by simulation arm") +
  theme_bw(base_size = 10) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.x      = element_text(angle = 25, hjust = 1),
    strip.background = element_rect(fill = "grey92")
  )

fig2_path <- file.path(out_dir, "fig2_perarm.pdf")
pdf(fig2_path, width = 11, height = 4)
print(p_arm)
dev.off()
cat(sprintf("Written: %s\n", fig2_path))

# ── Fig 3: Runtime (supplementary) ────────────────────────────────────────────
if ("elapsed" %in% names(ok) && any(!is.na(ok$elapsed))) {
  timing <- ok %>%
    filter(!is.na(elapsed)) %>%
    group_by(method, arm) %>%
    summarise(
      mean_sec = mean(elapsed, na.rm = TRUE),
      se_sec   = sd(elapsed,   na.rm = TRUE) / sqrt(sum(!is.na(elapsed))),
      .groups = "drop"
    ) %>%
    mutate(arm_label = factor(arm, levels = c("A","B","C","D"), labels = arm_labels))

  p_time <- ggplot(timing, aes(x = arm_label, y = mean_sec / 60,
                                colour = method, group = method, shape = method)) +
    geom_line(position = pd, linewidth = 0.7) +
    geom_point(position = pd, size = 3) +
    geom_errorbar(aes(ymin = (mean_sec - se_sec) / 60,
                      ymax = (mean_sec + se_sec) / 60),
                  position = pd, width = 0.25, linewidth = 0.5) +
    scale_colour_manual(values = method_colors, name = NULL) +
    scale_shape_manual(values = method_shapes, name = NULL) +
    scale_y_continuous(limits = c(0, NA)) +
    labs(x = NULL, y = "Mean runtime per replicate (minutes)",
         title = "Runtime comparison across simulation arms") +
    theme_bw(base_size = 11) +
    theme(
      legend.position  = "bottom",
      panel.grid.minor = element_blank()
    )

  fig3_path <- file.path(out_dir, "fig3_timing.pdf")
  pdf(fig3_path, width = 7, height = 4.5)
  print(p_time)
  dev.off()
  cat(sprintf("Written: %s\n", fig3_path))
}

cat("Done.\n")
