# woven

**Weighted Omics View Embedding via Nystrom** — supervised multi-omics integration for incomplete patient data.

## The problem

Standard multi-omics methods (DIABLO, MOFA2) require every patient to have every data type. Patients missing even one modality are deleted. In clinical cohorts this typically discards 50–80% of subjects, introducing systematic bias: the excluded patients are often the sickest.

## What WOVEN does

WOVEN learns a shared latent space using only the fully-observed **anchor** subjects, then projects block-missing patients via graph similarity — no rows dropped, no feature-level imputation.

- Supervised: label-augmented cross-covariance pulls dimensions toward class-discriminative structure
- Closed-form: single eigendecomposition of an (V×n_a) × (V×n_a) matrix — globally optimal, no iterations
- Interpretable: dense linear projection matrices W per modality

## Installation

```r
# Bioconductor (submission pending)
BiocManager::install("woven")

# Development version
remotes::install_github("NathanBresette/woven")
```

## Quick start

```r
library(woven)
data(woven_example)  # 90 subjects, 3 modalities, ~33% block-missing

# Fit on block-missing data — all 90 subjects scored
fit <- woven(woven_example$X_missing, Y = woven_example$Y, K = 3L)
summary(fit, labels = woven_example$Y)

# Latent space plot
plot(fit, labels = woven_example$Y)

# Top features driving each dimension
woven_plot_vip(fit, modality = "RNA")

# Metrics
woven_metrics(fit, woven_example$Y)
```

## Documentation

A full walkthrough — simulating block-missing multi-omics data, fitting
WOVEN, projecting incomplete subjects, and interpreting loadings — is in the
package vignette:

```r
vignette("woven-intro", package = "woven")
```

See `?woven` for the main entry point and `?woven_predict`,
`?woven_metrics`, `?woven_plot_vip` for the supporting functions.

## Benchmark results (400 simulation reps × 4 arms)

| Condition | Method | Silhouette | NMI | ESS | BER |
|---|---|---|---|---|---|
| Complete | WOVEN | **0.828** | **1.000** | 1.00 | **0.375** |
| Complete | DIABLO | 0.271 | 0.539 | 1.00 | 0.528 |
| MCAR 50% | WOVEN | 0.218 (anchor: **0.710**) | **0.476** | **1.00** | **0.398** |
| MCAR 50% | DIABLO | 0.350 | 0.739 | 0.19 | 0.571 |

ADNI validation (MRI + Lipidomics + NMR metabolomics, 2422 subjects): WOVEN scores **70%** of patients vs DIABLO's **31%**. The 944 subjects DIABLO discards have 2× the Dementia rate.

## Citation

Bresette N, Lin A-L, Cheng J. Weighted Omics View Embedding via Nystrom (WOVEN): supervised multi-omics integration for block-missing clinical cohort data. *In preparation*, 2026.

## License

MIT
