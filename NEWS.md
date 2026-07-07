# woven 0.99.1

* `woven()` now standardizes each modality to zero mean and unit variance by
  default (`scale = TRUE`), matching the internal scaling of comparator methods
  so that no modality dominates by measurement scale. Stored centers and scales
  are reapplied to new subjects in `woven_scores()` and `woven_predict()`. Set
  `scale = FALSE` to retain the previous raw-feature behavior.

# woven 0.99.0

* Initial Bioconductor submission.
* Unified dual SUMCOR MCCA solver (`woven_mcca_dual`) for all V >= 2 modalities.
* Nyström out-of-sample extension for block-missing subjects.
* Graph Laplacian regularization built on observed rows only (no imputation).
* Label-augmented cross-covariance for supervised integration.
* Full benchmark against DIABLO, MOFA2, IntegrAO, and Impute+DIABLO across four simulation arms and ADNI real-data validation.
