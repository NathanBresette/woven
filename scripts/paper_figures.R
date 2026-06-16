#!/usr/bin/env Rscript
# paper_figures.R — All paper figures and tables for WOVEN manuscript
#
# Outputs (written to figures/):
#   fig1_ess.pdf            ESS retention across conditions
#   fig2_silhouette.pdf     Silhouette by arm x condition x method
#   fig3_ber_nmi.pdf        BER and NMI: WOVEN, IntegrAO, DIABLO
#   fig4_adni.pdf           ADNI: ESS + DX stranded + latent space
#   fig_anchor_sensitivity.pdf  Supp. Fig. 2: sil vs anchor fraction
#   table1_main.tex         Main benchmark results (WOVEN, IntegrAO, DIABLO, MOFA2, Impute+DIABLO)
#   table2_adni.tex         ADNI coverage and results
#   table_s1_perarm.tex     Per-arm supplementary (WOVEN, IntegrAO, DIABLO)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
})

script_path <- tryCatch(normalizePath(sys.frames()[[1]]$ofile), error=function(e) NULL)
root <- if (!is.null(script_path)) {
  normalizePath(file.path(dirname(script_path), ".."))
} else {
  normalizePath("woven")
}
fig_dir <- file.path(root, "figures")
dir.create(fig_dir, recursive=TRUE, showWarnings=FALSE)

# ── Shared aesthetics ─────────────────────────────────────────────────────────
METHOD_COLORS <- c(
  WOVEN        = "#2166AC",
  IntegrAO     = "#E69F00",
  DIABLO       = "#D6604D",
  MOFA2        = "#4DAC26",
  ImputeDIABLO = "#762A83"
)
METHOD_LABELS <- c(
  WOVEN        = "WOVEN",
  IntegrAO     = "IntegrAO",
  DIABLO       = "DIABLO",
  MOFA2        = "MOFA2",
  ImputeDIABLO = "Impute+DIABLO"
)
COND_LEVELS <- c("complete","mcar30","mcar50","mar")
COND_LABELS <- c(complete="Complete", mcar30="MCAR 30%", mcar50="MCAR 50%", mar="MAR 30%")
ARM_LABELS  <- c(
  A="ARM A\n(RNA+Meth, V=2)",
  B="ARM B\n(RNA+Meth+Prot, V=3)",
  C="ARM C\n(InterSIM, V=3)",
  D="ARM D\n(Microbiome+Metab, V=2)"
)

theme_woven <- function() {
  theme_bw(base_size=11) +
  theme(
    panel.grid.minor  = element_blank(),
    strip.background  = element_rect(fill="grey92", color=NA),
    legend.position   = "bottom",
    legend.key.size   = unit(0.4,"cm"),
    plot.title        = element_text(face="bold", size=12),
    axis.title        = element_text(size=10),
    strip.text        = element_text(size=9)
  )
}

# ── Load primary benchmark data ───────────────────────────────────────────────
d <- read.csv(file.path(root,"results","summary_v3_400.csv"), stringsAsFactors=FALSE)
d <- d[d$error_msg=="" | is.na(d$error_msg), ]
d$method[d$method == "GRAMA"] <- "WOVEN"

smry <- d %>%
  group_by(arm, condition, method) %>%
  summarise(
    n_reps       = n(),
    sil_mean     = mean(silhouette,        na.rm=TRUE),
    sil_sd       = sd(silhouette,          na.rm=TRUE),
    sil_anc_mean = mean(silhouette_anchor, na.rm=TRUE),
    sil_anc_sd   = sd(silhouette_anchor,   na.rm=TRUE),
    nmi_mean     = mean(nmi,               na.rm=TRUE),
    nmi_sd       = sd(nmi,                na.rm=TRUE),
    ess_mean     = mean(ess_retention,     na.rm=TRUE),
    ess_sd       = sd(ess_retention,       na.rm=TRUE),
    ber_mean     = mean(ber,               na.rm=TRUE),
    ber_sd       = sd(ber,                na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(
    condition = factor(condition, levels=COND_LEVELS, labels=COND_LABELS),
    arm_label = ARM_LABELS[arm]
  )

smry_all <- d %>%
  group_by(condition, method) %>%
  summarise(
    sil_anc_mean = mean(silhouette_anchor, na.rm=TRUE),
    nmi_mean     = mean(nmi,               na.rm=TRUE),
    ess_mean     = mean(ess_retention,     na.rm=TRUE),
    ber_mean     = mean(ber,               na.rm=TRUE),
    ber_sd       = sd(ber,                na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(condition = factor(condition, levels=COND_LEVELS, labels=COND_LABELS))

# ── Load IntegrAO data (per-fold DR refitting, same protocol as WOVEN/DIABLO) ─
ia <- read.csv(file.path(root,"results","integrao_summary_v2.csv"),
               stringsAsFactors=FALSE)
# ia has: arm, condition, n_reps, sil_mean, nmi_mean, ber_mean, ess_mean
# sil_mean for IntegrAO = full-cohort silhouette (no anchor-specific split)

ia_smry <- ia %>%
  mutate(
    method       = "IntegrAO",
    sil_sd       = sil_mean * 0,   # SD not in v2 CSV; use 0 for point-only plots
    sil_anc_mean = sil_mean,        # best proxy; scored set = subjects with >=1 view
    sil_anc_sd   = 0,
    ber_sd        = 0,
    arm_label    = ARM_LABELS[arm],
    condition    = factor(condition, levels=COND_LEVELS, labels=COND_LABELS)
  )

ia_smry_all <- ia %>%
  group_by(condition) %>%
  summarise(
    method       = "IntegrAO",
    sil_anc_mean = mean(sil_mean,  na.rm=TRUE),
    nmi_mean     = mean(nmi_mean,  na.rm=TRUE),
    ess_mean     = mean(ess_mean,  na.rm=TRUE),
    ber_mean     = mean(ber_mean,  na.rm=TRUE),
    ber_sd       = 0,
    .groups="drop"
  ) %>%
  mutate(condition = factor(condition, levels=COND_LEVELS, labels=COND_LABELS))

# ── Figure 1: ESS retention ───────────────────────────────────────────────────
cat("Building Figure 1: ESS...\n")

ess_df <- smry %>%
  filter(method %in% c("WOVEN","DIABLO","ImputeDIABLO")) %>%
  mutate(method = factor(method, levels=names(METHOD_LABELS), labels=METHOD_LABELS))

# IntegrAO ESS = WOVEN ESS on every arm -- include as annotation rather than line
# to avoid crowding; note in caption instead.
fig1 <- ggplot(ess_df, aes(x=condition, y=ess_mean, color=method, group=method)) +
  geom_line(linewidth=0.8) +
  geom_point(size=2.2) +
  geom_errorbar(aes(ymin=ess_mean-ess_sd, ymax=ess_mean+ess_sd),
                width=0.15, linewidth=0.4, alpha=0.6) +
  facet_wrap(~arm_label, nrow=1) +
  scale_color_manual(
    values = METHOD_COLORS[c("WOVEN","DIABLO","ImputeDIABLO")],
    labels = METHOD_LABELS[c("WOVEN","DIABLO","ImputeDIABLO")],
    name   = NULL) +
  scale_y_continuous(labels=percent_format(accuracy=1), limits=c(0,1.05),
                     breaks=seq(0,1,0.25)) +
  labs(x=NULL, y="Effective Sample Size Retention",
       title="ESS retention: WOVEN (and IntegrAO) retain >90% vs DIABLO <50% under missingness",
       caption="Note: IntegrAO ESS is identical to WOVEN on all arms (both score subjects with >=1 view); line omitted to avoid overlap.") +
  theme_woven() +
  theme(axis.text.x=element_text(angle=35, hjust=1, size=8),
        plot.caption=element_text(size=7, color="grey40"))

ggsave(file.path(fig_dir,"fig1_ess.pdf"), fig1, width=10, height=5)
cat("  Saved fig1_ess.pdf\n")

# ── Figure 2: Silhouette ──────────────────────────────────────────────────────
cat("Building Figure 2: Silhouette...\n")

sil_base <- smry %>%
  filter(method %in% c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")) %>%
  mutate(
    method    = factor(method, levels=names(METHOD_LABELS), labels=METHOD_LABELS),
    sil_plot  = ifelse(as.character(condition)=="Complete", sil_mean, sil_anc_mean),
    sil_se    = ifelse(as.character(condition)=="Complete", sil_sd, sil_anc_sd)
  )

fig2 <- ggplot(sil_base, aes(x=condition, y=sil_plot, color=method, group=method)) +
  geom_hline(yintercept=0, linetype="dashed", color="grey60", linewidth=0.4) +
  geom_line(linewidth=0.8) +
  geom_point(size=2.2) +
  geom_errorbar(aes(ymin=sil_plot-sil_se, ymax=sil_plot+sil_se),
                width=0.15, linewidth=0.4, alpha=0.6) +
  facet_wrap(~arm_label, nrow=1) +
  scale_color_manual(values=METHOD_COLORS[c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")],
                     labels=METHOD_LABELS[c("WOVEN","DIABLO","MOFA2","ImputeDIABLO")],
                     name=NULL) +
  scale_y_continuous(limits=c(-0.15,1.05), breaks=seq(-0.2,1,0.2)) +
  labs(x=NULL, y="Average Silhouette Width\n(anchor subjects for missing conditions)",
       title="Latent space quality: WOVEN achieves >3× higher silhouette vs DIABLO") +
  theme_woven() +
  theme(axis.text.x=element_text(angle=35, hjust=1, size=8))

ggsave(file.path(fig_dir,"fig2_silhouette.pdf"), fig2, width=10, height=4.5)
cat("  Saved fig2_silhouette.pdf\n")

# ── Figure 3: BER + NMI (WOVEN, IntegrAO, DIABLO) ────────────────────────────
cat("Building Figure 3: BER and NMI...\n")

# Combine WOVEN/DIABLO from smry with IntegrAO from ia_smry
ber_nmi_gd <- smry %>%
  filter(method %in% c("WOVEN","DIABLO")) %>%
  select(arm, arm_label, condition, method, ber_mean, nmi_mean)

ber_nmi_ia <- ia_smry %>%
  select(arm, arm_label, condition, method, ber_mean, nmi_mean)

ber_nmi <- bind_rows(ber_nmi_gd, ber_nmi_ia) %>%
  pivot_longer(c(ber_mean, nmi_mean), names_to="metric", values_to="value") %>%
  mutate(
    metric = recode(metric,
      ber_mean = "BER (lower = better)",
      nmi_mean = "NMI (higher = better)"),
    method = factor(method, levels=c("WOVEN","IntegrAO","DIABLO"))
  )

chance_lines <- data.frame(
  metric = c("BER (lower = better)", "NMI (higher = better)"),
  yint   = c(0.75, 0)
)

fig3 <- ggplot(ber_nmi, aes(x=condition, y=value, color=method, group=method)) +
  geom_hline(data=chance_lines, aes(yintercept=yint),
             linetype="dashed", color="grey50", linewidth=0.4) +
  geom_line(linewidth=0.8) +
  geom_point(size=2.2) +
  facet_grid(metric ~ arm_label, scales="free_y") +
  scale_color_manual(
    values = METHOD_COLORS[c("WOVEN","IntegrAO","DIABLO")],
    labels = METHOD_LABELS[c("WOVEN","IntegrAO","DIABLO")],
    name   = NULL) +
  labs(x=NULL, y=NULL,
       title="Classification and cluster recovery: WOVEN, IntegrAO, and DIABLO across arms",
       caption="BER: per-fold DR refitting + LDA for all methods. ARM D: IntegrAO BER lower than WOVEN on complete data;\nWOVEN lower than IntegrAO under MCAR 50% missingness (reversal). Dashed = 4-class chance level (BER) / uninformative (NMI).") +
  theme_woven() +
  theme(axis.text.x=element_text(angle=35, hjust=1, size=7),
        plot.caption=element_text(size=7, color="grey40"))

ggsave(file.path(fig_dir,"fig3_ber_nmi.pdf"), fig3, width=12, height=6.5)
cat("  Saved fig3_ber_nmi.pdf\n")

# ── Figure 4: ADNI ────────────────────────────────────────────────────────────
cat("Building Figure 4: ADNI...\n")

adni_metrics <- read.csv(file.path(root,"..","ADNI","results","adni_metrics.csv"),
                          stringsAsFactors=FALSE)
adni_dx      <- read.csv(file.path(root,"..","ADNI","results","adni_dx_stranded.csv"),
                          stringsAsFactors=FALSE)

ess_bar <- data.frame(
  Method  = c("WOVEN","DIABLO"),
  ESS_pct = c(adni_metrics$ess_pct[adni_metrics$method=="WOVEN"],
               adni_metrics$ess_pct[adni_metrics$method=="DIABLO"])
)
p4a <- ggplot(ess_bar, aes(x=Method, y=ESS_pct, fill=Method)) +
  geom_col(width=0.5) +
  geom_text(aes(label=sprintf("%.0f%%",ESS_pct)), vjust=-0.4, fontface="bold", size=4.5) +
  scale_fill_manual(values=METHOD_COLORS[c("WOVEN","DIABLO")], guide="none") +
  scale_y_continuous(limits=c(0,100), labels=function(x) paste0(x,"%")) +
  labs(x=NULL, y="Subjects Scored (%)", title="(A) ESS Retention") +
  theme_woven() + theme(legend.position="none")

dx_long <- adni_dx %>%
  select(Group, CN_pct, MCI_pct, Dem_pct) %>%
  pivot_longer(c(CN_pct,MCI_pct,Dem_pct), names_to="DX", values_to="pct") %>%
  mutate(
    DX    = recode(DX, CN_pct="CN", MCI_pct="MCI", Dem_pct="Dementia"),
    DX    = factor(DX, levels=c("CN","MCI","Dementia")),
    Group = factor(Group, levels=c("DIABLO retained","WOVEN-only (recovered)","Neither (no data)"))
  )
p4b <- ggplot(dx_long, aes(x=Group, y=pct, fill=DX)) +
  geom_col(width=0.6, position="stack") +
  geom_text(aes(label=ifelse(!is.na(pct)&pct>4, sprintf("%.0f%%",pct),"")),
            position=position_stack(vjust=0.5), size=3.2, color="white", fontface="bold") +
  scale_fill_manual(values=c(CN="#4DAF4A",MCI="#FF7F00",Dementia="#E41A1C"), name="Diagnosis") +
  scale_y_continuous(labels=function(x) paste0(x,"%"), limits=c(0,105)) +
  labs(x=NULL, y="% of subjects", title="(B) Diagnosis distribution by subject group") +
  theme_woven() +
  theme(axis.text.x=element_text(angle=25, hjust=1, size=9))

metric_bar <- adni_metrics %>%
  filter(method %in% c("WOVEN","DIABLO")) %>%
  select(method, silhouette, nmi, ber) %>%
  pivot_longer(c(silhouette,nmi,ber), names_to="metric", values_to="value") %>%
  mutate(
    metric = recode(metric,
      silhouette="Silhouette\n(higher better)",
      nmi="NMI\n(higher better)",
      ber="BER\n(lower better)"),
    method = factor(method, levels=c("WOVEN","DIABLO"))
  )
p4c <- ggplot(metric_bar, aes(x=method, y=value, fill=method)) +
  geom_col(width=0.55) +
  geom_text(aes(label=sprintf("%.3f",value)), vjust=-0.4, size=3.5, fontface="bold") +
  facet_wrap(~metric, scales="free_y", nrow=1) +
  scale_fill_manual(values=METHOD_COLORS[c("WOVEN","DIABLO")], guide="none") +
  labs(x=NULL, y="Value", title="(C) Latent space quality metrics") +
  theme_woven()

fig4 <- (p4a | p4b | p4c) +
  plot_annotation(
    title="ADNI validation: WOVEN retains 70% of subjects vs DIABLO 31%;\nstranded patients have 2× higher Dementia rate",
    theme=theme(plot.title=element_text(face="bold", size=11))
  )
ggsave(file.path(fig_dir,"fig4_adni.pdf"), fig4, width=13, height=5.5)
cat("  Saved fig4_adni.pdf\n")

# ── Supp. Figure 2: Anchor fraction sensitivity ───────────────────────────────
cat("Building Supp. Figure 2: Anchor sensitivity...\n")

anc <- read.csv(file.path(root,"results","anchor_sensitivity.csv"), stringsAsFactors=FALSE)
anc_long <- anc %>%
  select(arm, anchor_frac, sil_all, sil_anchor, sil_nonanc) %>%
  pivot_longer(c(sil_all, sil_anchor, sil_nonanc), names_to="metric", values_to="sil") %>%
  mutate(
    metric = recode(metric,
      sil_all    = "All scored subjects",
      sil_anchor = "Anchor subjects only",
      sil_nonanc = "Non-anchor (Nystrom-projected)"),
    arm_label = paste0("ARM ", arm, if_else(arm=="A", "\n(diffuse Gaussian)", "\n(concentrated InterSIM)"))
  )

# ADNI reference anchor fraction
adni_ref <- 0.31

fig_anc <- ggplot(anc_long, aes(x=anchor_frac, y=sil, color=metric, group=metric)) +
  geom_hline(yintercept=0, linetype="dashed", color="grey60", linewidth=0.4) +
  geom_vline(xintercept=adni_ref, linetype="dotted", color="grey40", linewidth=0.5) +
  annotate("text", x=adni_ref+0.01, y=-0.12, label="ADNI\n(31%)", size=2.8,
           hjust=0, color="grey30") +
  geom_line(linewidth=0.9) +
  geom_point(size=2.0) +
  facet_wrap(~arm_label, nrow=1) +
  scale_color_manual(
    values = c("All scored subjects"="#2166AC",
               "Anchor subjects only"="#D6604D",
               "Non-anchor (Nystrom-projected)"="#4DAC26"),
    name=NULL) +
  scale_x_continuous(labels=percent_format(accuracy=1), breaks=seq(0.05,0.75,0.1)) +
  labs(x="Anchor fraction", y="Average Silhouette Width",
       title="Anchor fraction sensitivity: WOVEN is robust above ~15% anchors on concentrated signal",
       caption="Vertical dotted line = ADNI anchor fraction (31%). ARM A: non-anchor silhouette flat regardless of anchor fraction\n(SNR bottleneck). ARM C: non-anchor silhouette degrades only below ~10% anchors (anchor-quantity bottleneck).") +
  theme_woven() +
  theme(plot.caption=element_text(size=7, color="grey40"))

ggsave(file.path(fig_dir,"fig_anchor_sensitivity.pdf"), fig_anc, width=10, height=4.5)
cat("  Saved fig_anchor_sensitivity.pdf\n")

# ── Table 1: Main benchmark ───────────────────────────────────────────────────
cat("Building Table 1...\n")

# Combine all-arms summaries: WOVEN/DIABLO/MOFA2/ImputeDIABLO + IntegrAO
tab1_gd <- smry_all %>%
  filter(method %in% c("WOVEN","DIABLO","MOFA2","ImputeDIABLO"))

tab1_ia <- ia_smry_all %>%
  select(condition, method, sil_anc_mean, nmi_mean, ess_mean, ber_mean, ber_sd)

tab1_all <- bind_rows(tab1_gd, tab1_ia) %>%
  mutate(
    Method = METHOD_LABELS[method],
    Sil    = sprintf("%.3f", sil_anc_mean),
    NMI    = sprintf("%.3f", nmi_mean),
    ESS    = sprintf("%.2f", ess_mean),
    BER    = case_when(
      method == "MOFA2"        ~ "---",  # unsupervised; BER excluded
      is.na(ber_mean)          ~ "---",
      TRUE                     ~ sprintf("%.3f", ber_mean)
    )
  ) %>%
  select(condition, method, Method, Sil, NMI, ESS, BER)

# Row order: WOVEN, IntegrAO, DIABLO, MOFA2, ImputeDIABLO
method_order <- c("WOVEN","IntegrAO","DIABLO","MOFA2","ImputeDIABLO")
tab1_all <- tab1_all %>%
  mutate(method=factor(method, levels=method_order)) %>%
  arrange(condition, method)

lat <- c(
  "\\begin{table}[ht]",
  "\\centering",
  "\\small",
  paste0(
    "\\caption{Benchmark results averaged across 100 replicates per arm (all four arms combined). ",
    "Silhouette computed on anchor subjects (complete-case) for WOVEN and DIABLO; ",
    "on all scored subjects ($\\geq$1 view) for IntegrAO. See Fig.~2 for anchor-only comparison across methods. ",
    "BER uses 3-fold per-fold DR refitting + LDA for WOVEN, IntegrAO, DIABLO, and Impute+DIABLO; ",
    "IntegrAO per-fold fitting uses k-NN projection for held-out subjects (equivalent protocol). ",
    "MOFA2 BER excluded (unsupervised; labels not used during fitting). ",
    "Chance BER = 0.75 (4 classes). IntegrAO ESS equals WOVEN on all arms (both score subjects with $\\geq$1 view).}"
  ),
  "\\label{tab:main_benchmark}",
  "\\begin{tabular}{llrrrr}",
  "\\toprule",
  "Condition & Method & Silhouette & NMI & ESS & BER \\\\",
  "\\midrule"
)
prev_cond <- ""
for (i in seq_len(nrow(tab1_all))) {
  r <- tab1_all[i, ]
  cond_char <- as.character(r$condition)
  cond_str <- if (cond_char != prev_cond) {
    prev_cond <- cond_char
    paste0("\\textit{", cond_char, "} & ")
  } else "& "
  lat <- c(lat, sprintf("%s%s & %s & %s & %s & %s \\\\",
    cond_str, r$Method, r$Sil, r$NMI, r$ESS, r$BER))
  if (i < nrow(tab1_all) && as.character(tab1_all$condition[i+1]) != cond_char)
    lat <- c(lat, "\\midrule")
}
lat <- c(lat, "\\bottomrule", "\\end{tabular}", "\\end{table}")
writeLines(lat, file.path(fig_dir,"table1_main.tex"))
cat("  Saved table1_main.tex\n")

# ── Table 2: ADNI ─────────────────────────────────────────────────────────────
cat("Building Table 2: ADNI...\n")

adni_tex <- c(
  "\\begin{table}[ht]",
  "\\centering",
  paste0(
    "\\caption{ADNI real-data validation (V=3: MRI FreeSurfer + plasma lipidomics + NMR metabolomics; ",
    "n=2{,}422 baseline subjects with known diagnosis). ESS = fraction of subjects with at least one ",
    "latent score. BER via 5-fold stratified LDA on fixed latent representation; chance = 0.667 (3-class). ",
    "The 944 WOVEN-only subjects have twice the Dementia rate of the DIABLO-retained group.}"
  ),
  "\\label{tab:adni}",
  "\\begin{tabular}{lrrrrrr}",
  "\\toprule",
  "Method & N scored & ESS & Silhouette & NMI & BER \\\\",
  "\\midrule"
)
for (i in seq_len(nrow(adni_metrics))) {
  r <- adni_metrics[i,]
  adni_tex <- c(adni_tex, sprintf(
    "%s & %d & %.0f\\%% & %.3f & %.3f & %.3f \\\\",
    r$method, r$n_scored, r$ess_pct, r$silhouette, r$nmi, r$ber))
}
adni_tex <- c(adni_tex,
  "\\midrule",
  "\\multicolumn{6}{l}{\\textit{Diagnosis composition by subject group}} \\\\",
  "\\midrule",
  "Group & N & & CN & MCI & Dementia \\\\",
  "\\midrule"
)
for (i in seq_len(nrow(adni_dx))) {
  r <- adni_dx[i,]
  adni_tex <- c(adni_tex, sprintf(
    "%s & %d & & %.0f\\%% & %.0f\\%% & %.0f\\%% \\\\",
    gsub("_","\\_",r$Group), r$N, r$CN_pct, r$MCI_pct, r$Dem_pct))
}
adni_tex <- c(adni_tex, "\\bottomrule", "\\end{tabular}", "\\end{table}")
writeLines(adni_tex, file.path(fig_dir,"table2_adni.tex"))
cat("  Saved table2_adni.tex\n")

# ── Table S1: Per-arm (WOVEN, IntegrAO, DIABLO) ───────────────────────────────
cat("Building Table S1: Per-arm...\n")

# IntegrAO per-arm from ia_smry
ia_s1 <- ia_smry %>%
  mutate(
    sil_str = sprintf("%.3f", sil_mean),
    nmi_str = sprintf("%.3f", nmi_mean),
    ess_str = sprintf("%.2f", ess_mean),
    ber_str = sprintf("%.3f", ber_mean)
  )

s1_gd <- smry %>%
  filter(method %in% c("WOVEN","DIABLO")) %>%
  mutate(
    sil_str = sprintf("%.3f", sil_anc_mean),
    nmi_str = sprintf("%.3f", nmi_mean),
    ess_str = sprintf("%.2f", ess_mean),
    ber_str = ifelse(is.na(ber_mean), "---", sprintf("%.3f", ber_mean))
  )

s1_tex <- c(
  "\\begin{table}[ht]",
  "\\centering",
  "\\small",
  paste0(
    "\\caption{Per-arm benchmark results (mean over 100 replicates). ",
    "Silhouette on anchor subjects (WOVEN and DIABLO) or all scored subjects (IntegrAO). ",
    "BER via per-fold DR refitting + LDA for all three methods. ",
    "ARM A: RNA-seq+methylation (V=2, diffuse Gaussian). ",
    "ARM B: RNA-seq+methylation+protein (V=3, diffuse Gaussian). ",
    "ARM C: RNA-seq+methylation+protein (V=3, InterSIM concentrated). ",
    "ARM D: microbiome+metabolomics (V=2, NorTA). ",
    "Dagger (\\dag): IntegrAO BER lower than WOVEN on ARM D complete; ",
    "WOVEN BER lower than IntegrAO at MCAR 50\\% (see text).}"
  ),
  "\\label{tab:perarm}",
  "\\begin{tabular}{llcccccccccccc}",
  "\\toprule",
  paste0(" & & \\multicolumn{4}{c}{WOVEN} & \\multicolumn{4}{c}{IntegrAO}",
         " & \\multicolumn{4}{c}{DIABLO} \\\\"),
  "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
  "ARM & Condition & Sil & NMI & ESS & BER & Sil & NMI & ESS & BER & Sil & NMI & ESS & BER \\\\",
  "\\midrule"
)

prev_arm <- ""
for (arm in c("A","B","C","D")) {
  for (cond in levels(s1_gd$condition)) {
    g  <- s1_gd[s1_gd$arm==arm & as.character(s1_gd$condition)==cond & s1_gd$method=="WOVEN",]
    ia <- ia_s1[ia_s1$arm==arm & as.character(ia_s1$condition)==cond,]
    dd <- s1_gd[s1_gd$arm==arm & as.character(s1_gd$condition)==cond & s1_gd$method=="DIABLO",]
    if (nrow(g)==0 || nrow(dd)==0) next
    ia_sil <- if (nrow(ia)>0) ia$sil_str else "---"
    ia_nmi <- if (nrow(ia)>0) ia$nmi_str else "---"
    ia_ess <- if (nrow(ia)>0) ia$ess_str else "---"
    ia_ber <- if (nrow(ia)>0) ia$ber_str else "---"
    arm_str <- if (arm != prev_arm) { prev_arm <- arm; arm } else ""
    # Add dagger footnote marker for ARM D complete and mcar50
    ber_g_str  <- g$ber_str
    ber_ia_str <- ia_ber
    if (arm=="D" && cond %in% c("Complete","MCAR 50%")) {
      ber_g_str  <- paste0(ber_g_str,  "\\dag")
      ber_ia_str <- paste0(ber_ia_str, "\\dag")
    }
    s1_tex <- c(s1_tex, sprintf(
      "%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\",
      arm_str, cond,
      g$sil_str,  g$nmi_str,  g$ess_str,  ber_g_str,
      ia_sil,     ia_nmi,     ia_ess,     ber_ia_str,
      dd$sil_str, dd$nmi_str, dd$ess_str, dd$ber_str))
  }
  if (arm != "D") s1_tex <- c(s1_tex, "\\midrule")
}
s1_tex <- c(s1_tex,
  "\\bottomrule",
  "\\multicolumn{14}{l}{\\dag ARM D complete: IntegrAO BER $<$ WOVEN; MCAR 50\\%: WOVEN BER $<$ IntegrAO (reversal -- see text).}",
  "\\end{tabular}",
  "\\end{table}"
)
writeLines(s1_tex, file.path(fig_dir,"table_s1_perarm.tex"))
cat("  Saved table_s1_perarm.tex\n")

cat(sprintf("\nAll outputs written to: %s\n", fig_dir))
cat("Files:\n")
for (f in list.files(fig_dir)) cat(sprintf("  %s\n", f))
