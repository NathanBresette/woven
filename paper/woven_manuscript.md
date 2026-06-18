# Weighted Omics View Embedding via Nystrom (WOVEN): Supervised Multi-Omics Integration for Block-Missing Clinical Cohort Data

**Nathan Bresette**^1^, **Ai-Ling Lin**^1,2^ and **Jianlin Cheng**^3^

^1^ Department of Radiology, University of Missouri, Columbia, MO, USA
^2^ NextGen Precision Health Institute, University of Missouri, Columbia, MO, USA
^3^ Department of Electrical Engineering and Computer Science, University of Missouri, Columbia, MO, USA

Correspondence: lin.ailingATumsystem.edu

---

## Abstract

**Motivation:** Supervised multi-omics integration methods require complete observations across all molecular modalities, silently discarding patients missing any single data block. In clinical cohort studies, this intersection constraint introduces systematic bias because missingness correlates with disease severity and subgroup membership.

**Results:** We present WOVEN (Weighted Omics View Embedding via Nystrom), a supervised multi-omics integration method that learns a shared latent space from fully observed anchor subjects and projects incomplete patients using only their available modalities, without feature-level imputation. Across 400 simulation replicates spanning four benchmark arms, WOVEN retains 100% of subjects at 50% block-missing rates while DIABLO retains 19%, achieves three-fold higher silhouette score on complete data (0.828 vs 0.271), and substantially lower balanced error rate on NorTA microbiome-metabolomics data (0.145 vs 0.763). Missingness-targeted imputation followed by DIABLO does not recover discriminative signal (BER 0.579 vs WOVEN 0.398 at 50% MCAR). On 2,422 ADNI subjects, WOVEN scores 1,687 patients (70%) versus 743 (31%) for DIABLO; recovered subjects carry twice the dementia prevalence of the retained complete-case sample.

**Availability:** R package at https://github.com/NathanBresette/woven (Bioconductor submission in progress). Benchmark code at https://github.com/NathanBresette/woven_paper.

---

## 1. Introduction

Multi-omics integration has become central to precision medicine and comparative effectiveness research (CER), enabling the joint analysis of transcriptomics, methylation, proteomics, and metabolomics to identify patient subgroups and molecular drivers of clinical outcomes (Hasin *et al.*, 2017). Methods span a wide design space: unsupervised decompositions including JIVE (Lock *et al.*, 2013), MOFA+ (Argelaguet *et al.*, 2020), and iClusterPlus (Mo *et al.*, 2013) extract shared and modality-specific latent factors from complete matched datasets. Supervised methods, led by DIABLO (Singh *et al.*, 2019) which extends Sparse GCCA (Tenenhaus *et al.*, 2014) to a multi-class discriminative objective, additionally exploit class labels to pull canonical directions toward clinically relevant structure. These methods share a critical assumption: every subject must contribute data to every modality.

In clinical cohort studies, this assumption routinely fails. Patients miss blood draws, decline imaging visits, or lack biospecimens for a particular assay. The result is block-missing data, where entire modality matrices are absent for subsets of subjects. The standard response is the intersection constraint, retaining only patients who are fully observed across all modalities. DIABLO enforces this constraint before fitting. In simulation at 50% block-missing rates, this reduces the effective sample size to 19% of the cohort. In the ADNI multi-omics cohort combining MRI, lipidomics, and NMR metabolomics, DIABLO retains 743 of 2,422 enrolled subjects (31%). The 944 subjects recovered only by WOVEN carry twice the dementia prevalence of the retained complete-case sample (25% vs 12%), a direct demonstration of the selection bias introduced by the intersection constraint.

A natural response to missing data is imputation. However, missForest-imputed data followed by DIABLO (ImputeDIABLO) performs no better than, and consistently worse than, DIABLO on complete cases alone (balanced error rate 0.579 vs 0.571 at 50% block-missing). Feature-level imputation predicts individual molecular measurements from within-modality correlations, but does not reconstruct the cross-modal alignment structure that structural missingness disrupts.

Several recent methods handle block-missing multi-omics without imputation. IntegrAO (Ma *et al.*, 2025) uses graph neural networks to embed subjects from partially overlapping modality sets, achieving high performance on complete data but unable to project new block-missing subjects at inference time without all modalities present. MIMIR (Nambiar *et al.*, 2026) addresses both missing modalities and missing features via masked autoencoders, but performs feature-level reconstruction in the process. JASMINE (Ballard *et al.*, 2025) uses self-supervised representation learning for incomplete multi-omics, but operates in the deep generative model regime without interpretable projection matrices. OLFG (Chen *et al.*, 2023) applies graph Laplacian regularization with projection matrices to neuroimaging modalities, the closest structural analog to WOVEN, but requires feature-level imputation before projecting and is restricted to the neuroimaging domain. Sui *et al.* (2025) apply block-wise missing multi-task learning to ADNI data but frame the problem as supervised prediction rather than latent space alignment.

We present WOVEN (Weighted Omics View Embedding via Nystrom), a supervised multi-omics integration method for block-missing clinical data. WOVEN identifies a subset of fully observed anchor subjects, learns per-modality graph Laplacian regularizers from all available observations, and finds a shared latent space by solving a single closed-form eigendecomposition of a label-augmented anchor-restricted cross-modal alignment objective. Block-missing subjects are projected using their available modalities directly, or via Nystrom extension (Bengio *et al.*, 2003) when only a single modality is observed. No feature-level imputation is performed at any stage.

Laplacian-regularized CCA is an established subfield (Blaschko *et al.*, 2011; Chen *et al.*, 2019), but no prior work applies it to block-missing bulk clinical multi-omics. LapKCCA (Blaschko *et al.*, 2011) applies semi-supervised Laplacian kernel CCA to paired fMRI modalities without block-missing support or Nystrom projection. Graph MCCA (Chen *et al.*, 2019) requires all subjects to be fully paired. WOVEN's specific combination of anchor-restricted cross-modal alignment, full-data per-modality Laplacians, label augmentation, and Nystrom out-of-sample projection fills the gap identified in recent field benchmarks (Duan *et al.*, 2021; Hornung and Boulesteix, 2024).

We benchmark WOVEN against DIABLO (Singh *et al.*, 2019; Rohart *et al.*, 2017), MOFA+ (Argelaguet *et al.*, 2020), ImputeDIABLO (missForest + DIABLO), and IntegrAO (Ma *et al.*, 2025) across 400 simulation replicates spanning four data-generation arms under complete, 30% MCAR, 50% MCAR, and MAR conditions. We validate on 2,422 ADNI subjects. WOVEN is distributed as an open-source R package with Bioconductor submission in progress.

---

## 2. Methods

### 2.1 Problem Formulation

Let $X^{(v)} \in \mathbb{R}^{n \times p_v}$ denote the data matrix for modality $v \in \{1, \ldots, V\}$ over $n$ subjects. Under block-missing data, subject $i$ may be entirely absent from modality $v$. Let $\mathcal{A} \subseteq \{1, \ldots, n\}$ denote the anchor set, the subjects with observations in all $V$ modalities, with $n_a = |\mathcal{A}|$. Non-anchor subjects have data in at least one but not all modalities. Subjects with no data in any modality are excluded from all analyses.

WOVEN seeks projection matrices $W^{(v)} \in \mathbb{R}^{p_v \times K}$ for each modality such that the anchor latent scores $Z_a^{(v)} = X_a^{(v)} W^{(v)}$ are maximally correlated across modalities, geometrically regularized within each modality, and discriminative with respect to class labels $Y$.

### 2.2 Objective Function

The WOVEN objective is a label-augmented SUMCOR multiset CCA:

$$\max_{W^{(v)}} \sum_{v < u} \text{tr}\left( W^{(v)T} \tilde{C}_{vu} W^{(u)} \right) \quad \text{s.t.} \quad W^{(v)T} B_v W^{(v)} = I_K$$

where $\tilde{C}_{vu} = X_a^{(v)T} \left( I + \gamma_Y K_Y \right) X_a^{(u)}$ is the label-augmented cross-covariance between modalities $v$ and $u$ restricted to anchor subjects, $K_Y = \tilde{Y}\tilde{Y}^T / n_a$ is the centered one-hot label kernel (amplifying cross-covariance between same-class subjects), and $\gamma_Y \geq 0$ is the supervision strength hyperparameter.

The constraint matrix $B_v = X^{(v)T} M_v X^{(v)}$ encodes graph Laplacian regularization via $M_v = I + \lambda_v L^{(v)} / n$, where $L^{(v)}$ is a $k$-nearest-neighbor RBF graph Laplacian built from all observed rows of $X^{(v)}$ (block-missing subjects excluded from the $k$-NN graph, no imputation). The Laplacian trace penalty $\text{tr}(W^{(v)T} X^{(v)T} L^{(v)} X^{(v)} W^{(v)})$ encourages nearby subjects in the feature-space graph to project to nearby latent positions, providing geometric regularization that improves out-of-sample generalization for projected block-missing subjects.

### 2.3 Closed-Form Solution via Dual MCCA

The WOVEN objective is reformulated in dual space, enabling a single eigendecomposition of a $(V n_a) \times (V n_a)$ block matrix. Define the anchor-restricted regularizer $M_v = I_{n_a} + \lambda_v L_a^{(v)} / n_a$ and the regularized anchor kernel $\mathcal{K}_v = X_a^{(v)} M_v X_a^{(v)T}$ for modality $v$. The dual block matrix $P$ has zero diagonal blocks and off-diagonal blocks:

$$P_{vu} = M_v^{-1/2} \left( I + \gamma_Y K_Y \right) M_u^{-1/2}$$

The top-$K$ eigenvectors $\{\phi_v\}$ of $P$ corresponding to positive eigenvalues give anchor latent scores $Z_v = M_v^{-1/2} \phi_v$ (M-orthonormalized), and projection matrices are recovered as $W^{(v)} = X_a^{(v)T} \mathcal{K}_v^{-1} Z_v^{(a)}$. The eigendecomposition is $O((V n_a)^3)$, yielding a 5 to 20-fold speed advantage over iterative ALS solvers at biologically realistic feature dimensions. For $V = 2$ this reduces to the SVD of $M_1^{-1/2} (I + \gamma_Y K_Y) M_2^{-1/2}$, equivalent to dual supervised CCA. Only positive eigenvalues are retained; negative eigenvalues correspond to anti-correlated directions. Setting $\gamma_Y = 0$ recovers unsupervised SUMCOR MCCA.

### 2.4 Projection of Block-Missing Subjects

For non-anchor subjects with at least two observed modalities, WOVEN computes latent scores by averaging available-modality projections:

$$\hat{z}_i = \frac{1}{|\mathcal{V}_i|} \sum_{v \in \mathcal{V}_i} x_i^{(v)} W^{(v)}$$

where $\mathcal{V}_i$ is the set of observed modalities for subject $i$. No feature-level imputation is performed; absent modalities do not contribute to the latent score.

For subjects with only a single observed modality, WOVEN applies Nystrom extension (Bengio *et al.*, 2003): the subject is projected via a kernel-weighted average of anchor latent positions, normalized to a simplex, using an RBF kernel computed from the observed features. This preserves the eigenvector structure of the anchor latent space in the out-of-sample extension.

### 2.5 Hyperparameter Selection

The supervision strength $\gamma_Y$ is selected from $\{0.5, 1.0, 5.0, 10.0\}$ and the regularization strength $\lambda_v$ from $\{0.001, 0.005, 0.01, 0.1, 0.5\}$ by three-fold stratified cross-validation on anchor subjects, maximizing silhouette score (Rousseeuw, 1987) of held-out anchor latent scores. The $k$-NN graph uses $k = 10$ by default; sensitivity analysis across $k \in \{3, 5, 10, 20, 50\}$ shows silhouette varies by less than 0.044 across this range. Sensitivity to $\lambda$ across four orders of magnitude ($10^{-4}$ to $1.0$) changes silhouette by less than 1%, and any $\gamma_Y > 0$ recovers full performance (setting $\gamma_Y = 0$ degrades silhouette from 0.799 to -0.094, confirming that the supervision term is essential).

### 2.6 Benchmark Design

We evaluated WOVEN against DIABLO (Singh *et al.*, 2019; Rohart *et al.*, 2017), MOFA+ (Argelaguet *et al.*, 2020), ImputeDIABLO (missForest imputation followed by DIABLO), and IntegrAO (Ma *et al.*, 2025) across four semi-synthetic simulation arms (parameters estimated from reference datasets following the semisynthetic paradigm of Sankaran *et al.*, 2025):

- **ARM A:** Two-modality diffuse signal (RNA-seq + methylation; SPsimSeq, Assefa *et al.* 2020, 4,047 genes + 367 CpG sites; $n = 300$, four balanced groups; anchor fraction 25% at 50% MCAR)
- **ARM B:** Three-modality diffuse signal (ARM A + proteomics, 62 proteins; InterSIM, Chalise *et al.* 2016; anchor fraction 13% at 50% MCAR)
- **ARM C:** Three-modality concentrated signal (RNA-seq + methylation + proteomics; InterSIM from TCGA-OV reference, strong between-group separation; anchor fraction 13% at 50% MCAR)
- **ARM D:** Two-modality compositional signal (microbiome CLR + metabolomics; NorTA framework with SpiecEasi-estimated correlation structure, Mangnier *et al.* 2025; microbiome marginals via MIDASim, He *et al.* 2024, with cross-modal spike-in structure from SparseDOSSA2, Ma *et al.* 2021; anchor fraction 25% at 50% MCAR)

Ground-truth latent factor structure was embedded using SUMO (Osang'ir *et al.*, 2025) to provide a recoverable $K = 5$ shared factor backbone across all arms. Block missingness is induced post-simulation as MCAR 30%, MCAR 50%, and structured MAR. One hundred bootstrap replicates per arm yield 400 total replicates per method per condition.

Balanced error rate (BER) is computed via per-fold dimensionality reduction (DR) refitting with linear discriminant analysis (LDA), ensuring test subjects never contribute to parameter estimation. Fixed-Z BER is not used (circular for supervised methods). All methods use identical three-fold splits per replicate. Chance BER for four balanced classes is 0.75. Latent space geometry is assessed using silhouette score (Rousseeuw, 1987) and normalized mutual information (NMI).

### 2.7 ADNI Validation

We applied WOVEN to 2,422 baseline subjects from the Alzheimer's Disease Neuroimaging Initiative (ADNI; Jack *et al.*, 2008) combining three modalities: MRI FreeSurfer morphometrics (341 features, $n_{\text{obs}} = 902$), plasma lipidomics (781 features, $n_{\text{obs}} = 1,418$), and NMR metabolomics (267 features, $n_{\text{obs}} = 1,642$). The anchor set (complete across all three modalities) contains 743 subjects (31%). ADNI is used here as a methodological testbed for block-missing multi-omics integration; we make no claims about the biological specificity of plasma lipids for central nervous system pathology. The three modalities were selected to maximize subject coverage and to ensure genuinely independent missingness mechanisms: MRI requires a separate imaging visit, plasma lipidomics requires a blood draw at a dedicated lab visit, and NMR metabolomics is measured from the same blood draw as lipidomics but processed independently. CSF-based biomarkers, while biologically proximal to neurodegeneration, are available for fewer than 300 ADNI subjects and would substantially reduce the recoverable cohort, undermining the ESS demonstration. Labels are baseline diagnosis: cognitively normal (CN), mild cognitive impairment (MCI), and dementia (three-class; chance BER = 0.667). Data were obtained from the ADNI database (adni.loni.usc.edu). ADNI is funded by the National Institute on Aging and the National Institute of Biomedical Imaging and Bioengineering. A complete list of ADNI investigators is at adni.loni.usc.edu/about/faqs.

---

## 3. Results

### 3.1 WOVEN Recovers Latent Structure More Precisely on Complete Data

On complete data, WOVEN substantially outperforms all comparators on latent space geometry across all four simulation arms (Table 1). Mean silhouette score is 0.828 for WOVEN versus 0.271 for DIABLO and 0.204 for MOFA+ (all 400 replicates). Normalized mutual information (NMI) is 1.000 for WOVEN versus 0.539 for DIABLO and 0.500 for MOFA+. WOVEN achieves NMI = 1.00 in every replicate of every arm, a result of the label-augmented objective explicitly aligning canonical directions with class-discriminative cross-modal correlations. Note that NMI measures whether the latent space organizes geometrically into groups matching true labels, which WOVEN's supervision guarantees; BER measures linear separability of those groups via LDA, which depends on within-group signal strength. On diffuse signal arms (A and B), group structure is present in the latent space (NMI = 1.00) but the signal is too weak for reliable LDA classification (BER near chance at 0.673/0.682), consistent with no method recovering discriminative signal in that regime.

BER on complete data is 0.375 for WOVEN and 0.528 for DIABLO overall. The dominant contribution is ARM D (NorTA microbiome-metabolomics), where WOVEN BER is 0.145 versus 0.763 for DIABLO. In dense DIABLO (no sparsity constraint, 100 replicates), ARM D BER is 0.762, confirming that this gap is algorithmic rather than a sparsity artifact. WOVEN's label-augmented cross-covariance with Laplacian regularization captures the compositional correlation structure that DIABLO's PLS objective misses. On ARM C (concentrated InterSIM signal), both methods achieve BER = 0.000.

WOVEN runs in 3.2 to 12.6 seconds per replicate under complete-data conditions (ARM D fastest, ARM B slowest), compared to 4.3 to 72.8 seconds for DIABLO and 10.2 to 16.8 seconds for MOFA+. WOVEN is 5.8-fold faster than DIABLO on the high-dimensional ARM A and B arms.

### 3.2 ARM D Reversal: Nystrom Projection Degrades More Gracefully Than Graph Diffusion

The comparison with IntegrAO reveals the mechanistic advantage of WOVEN under block-missingness. On ARM D complete data, IntegrAO achieves BER = 0.086 versus WOVEN 0.131, confirming that unsupervised graph diffusion better captures NorTA microbiome-metabolomics structure when all data are available. Under 50% MCAR, this reverses: WOVEN BER is 0.107 versus IntegrAO 0.130. IntegrAO's graph diffusion constructs a subject-by-subject affinity matrix; when 50% of subjects are missing a modality, the affinity matrix is partially unobserved and must be estimated, introducing noise that degrades embedding quality. WOVEN's Nystrom projection operates on the anchor-estimated eigenvector structure, requiring only available-modality features per new subject, with no dependence on the full affinity matrix. The ARM D reversal is the clearest empirical demonstration of the mechanism WOVEN is designed to exploit: anchor-restricted estimation followed by closed-form projection degrades more gracefully than graph-based methods when modality availability is heterogeneous.

IntegrAO and WOVEN achieve equivalent performance on ARM C (BER = 0.000 for both) and WOVEN outperforms IntegrAO on diffuse arms (WOVEN 0.673/0.680 vs IntegrAO 0.747/0.763 complete), where label supervision provides a consistent advantage. IntegrAO also achieves ESS = 1.00 in all conditions, matching WOVEN on subject retention. The ESS advantage in this benchmark is WOVEN and IntegrAO together versus DIABLO.

### 3.3 WOVEN Retains All Subjects Under Block-Missingness

At 50% MCAR block-missing rates, DIABLO retains 19% of subjects by enforcing the intersection constraint. WOVEN retains 100% of subjects in every arm and condition. MOFA+ also achieves ESS = 1.00 via variational EM that skips missing entries. The distinction between WOVEN and MOFA+ under missingness lies in supervised geometry: WOVEN maintains anchor-only silhouette 0.710 at 50% MCAR versus MOFA+ 0.192, and provides BER estimates via per-fold DR refitting with labeled data, whereas MOFA+'s unsupervised objective does not.

DIABLO's apparent full-cohort silhouette advantage under missingness (0.350 vs WOVEN 0.218) is an artifact of selection: DIABLO only scores the 19% of subjects with complete data, who cluster more cleanly by construction. The anchor-only comparison, evaluating both methods on the same population, reverses this decisively (WOVEN 0.710 vs DIABLO 0.350).

BER at 50% MCAR is 0.398 for WOVEN and 0.571 for DIABLO overall, driven by ARM D (0.113 vs 0.777). MAR block-missingness produces comparable results (WOVEN BER 0.389 vs DIABLO 0.572), confirming that WOVEN's advantage is not specific to the MCAR mechanism.

Notably, WOVEN runs faster under block-missingness than on complete data: 1.6 to 6.2 seconds per replicate at 50% MCAR, versus 3.2 to 12.6 seconds on complete data. This speedup arises because the anchor set is smaller under missingness, reducing the eigendecomposition from $O((V n)^3)$ on complete data to $O((V n_a)^3)$ where $n_a \approx 0.25n$ for $V = 2$ and $n_a \approx 0.13n$ for $V = 3$ at 50% MCAR. DIABLO also speeds up under missingness (31.8 to 38.1 seconds for ARM A/B at mcar50 vs 72 seconds complete), for the same reason: it fits only on anchor subjects. WOVEN remains 5 to 7-fold faster than DIABLO at 50% MCAR on the high-dimensional arms, while scoring the full cohort rather than the 19% DIABLO retains.

### 3.4 Imputation Does Not Recover Discriminative Signal

ImputeDIABLO, which applies missForest random-forest imputation to all features before running DIABLO, performs no better than DIABLO on complete cases alone and consistently worse than WOVEN (Table 1). At 50% MCAR, ImputeDIABLO BER is 0.579 versus DIABLO 0.571 and WOVEN 0.398. On ARM D at 50% MCAR, ImputeDIABLO BER is 0.777, identical to DIABLO and far above WOVEN's 0.113. Anchor-only silhouette of ImputeDIABLO (0.267 at 50% MCAR) does not reach DIABLO's complete-case silhouette (0.350), let alone WOVEN's (0.710).

This result reflects a structural distinction between feature-level imputation and latent-space projection. missForest predicts individual molecular feature values from within-modality cross-feature correlations, then passes the imputed matrix to DIABLO. Imputation does not reconstruct the cross-modal alignment structure that block-missingness disrupts; it replaces missing entries with within-modality predictions that carry no information about how that modality relates to others. WOVEN projects incomplete subjects through the estimated cross-modal projection matrices $W^{(v)}$, preserving the geometric relationships learned from anchor subjects.

### 3.5 ADNI: WOVEN Recovers Clinically Relevant Subjects That DIABLO Discards

Applied to 2,422 ADNI subjects, WOVEN scores 1,687 patients (70%) versus 743 (31%) for DIABLO, recovering 944 additional patients that DIABLO discards. WOVEN achieves BER 0.398 versus DIABLO 0.541 (chance = 0.667), silhouette 0.174 versus -0.018, and NMI 0.114 versus 0.015.

The 944 WOVEN-recovered subjects have substantially different clinical characteristics than the 743 DIABLO-retained subjects (Table 2). Among DIABLO-retained subjects, dementia prevalence is 12%. Among WOVEN-recovered subjects, dementia prevalence is 25%, twice as high. The largest recoverable group (660 subjects with lipidomics and NMR but no MRI) disproportionately comprises patients who could not or did not complete imaging visits, consistent with greater functional impairment at baseline. Any CER analysis using a DIABLO-derived latent space as a covariate or stratification variable would systematically underrepresent the most severely affected patients, biasing downstream subgroup effect estimates.

---

## 4. Discussion

WOVEN addresses the intersection constraint that limits supervised multi-omics integration in clinical cohort studies. By combining anchor-restricted label-augmented cross-covariance, full-data per-modality graph Laplacian regularization, and Nystrom out-of-sample projection, WOVEN learns a shared latent space without feature-level imputation and without discarding incomplete subjects. The closed-form dual SUMCOR MCCA solution finds the global optimum of the WOVEN objective in a single eigendecomposition, with no iterations, no random restarts, and 5 to 20-fold speed improvement over iterative solvers on high-dimensional data.

The benchmark results establish four concrete claims. First, WOVEN substantially improves latent space geometry on complete data (silhouette 0.828 vs 0.271, NMI 1.000 vs 0.539 against DIABLO), reflecting the label-augmented objective's alignment of class-discriminative cross-modal correlations. Second, WOVEN retains 100% of enrolled subjects at all missingness levels, compared to 19% for DIABLO at 50% MCAR. Third, feature-level imputation does not recover discriminative structure lost to block-missingness: ImputeDIABLO performs no better than DIABLO on complete cases in any condition. Fourth, the ARM D reversal demonstrates the mechanistic advantage of Nystrom projection over graph diffusion under block-missingness: IntegrAO outperforms WOVEN on complete compositional data but is outperformed by WOVEN once modalities become structurally missing.

WOVEN belongs to a broader lineage of Laplacian-regularized CCA but extends it in ways not previously combined. LapKCCA (Blaschko *et al.*, 2011) applies semi-supervised Laplacian kernel CCA to paired fMRI modalities without block-missing support or Nystrom projection. Graph MCCA (Chen *et al.*, 2019) requires all subjects to be fully paired. OLFG (Chen *et al.*, 2023) is the closest structural analog: it also uses graph Laplacian regularization with linear projection matrices for neuroimaging modalities, but requires feature-level imputation before projecting and has not been extended to general bulk clinical multi-omics. WOVEN deliberately avoids feature-level imputation at every stage: projection via $W^{(v)}$ matrices or Nystrom extension operates entirely in the latent space. Deep generative approaches including MIMIR (Nambiar *et al.*, 2026) and JASMINE (Ballard *et al.*, 2025) reconstruct raw molecular features via decoder networks, performing feature-level imputation in the process and producing embeddings without interpretable projection matrices. The connection to WOVEN's predecessor framework, DIABLO (built on SGCCA; Tenenhaus *et al.*, 2014), is direct: WOVEN generalizes DIABLO's supervised SUMCOR objective to the anchor-restricted block-missing setting while replacing the iterative NIPALS solver with a closed-form eigendecomposition.

The ADNI equity finding deserves particular emphasis. The 944 subjects DIABLO discards carry twice the dementia prevalence of retained subjects. In ADNI, this pattern arises because MRI acquisition requires a separate imaging visit, which more severely impaired patients disproportionately do not complete. Complete-case analysis therefore systematically excludes the sickest patients, introducing bias into downstream causal inference. This pattern is likely general in multi-modal clinical studies where missingness is driven by patient burden and functional capacity rather than random equipment failure. WOVEN's ability to include these patients while producing a well-separated latent space (BER 0.398 vs 0.541, silhouette 0.174 vs -0.018) directly mitigates this bias.

Several limitations deserve acknowledgment. First, anchor-only parameter estimation means $W^{(v)}$ is estimated from the fully observed subset; at very low anchor fractions (below approximately 15% in our sensitivity analysis) performance degrades. For prospective study design, we recommend targeting an anchor fraction of at least 20% for two-modality studies and 15% for three-modality studies. In practice, this can be achieved by staggering data collection so that a designated subset of subjects completes all modality acquisitions in the same enrollment period, or by collecting a high-coverage "base" modality (e.g., blood-based assays) for all subjects while allowing imaging to be missing. When the anchor fraction cannot be controlled, users should consider pooling low-coverage modalities or relaxing the anchor definition to subjects observed in at least $V - 1$ modalities. Second, the Nystrom extension approximation for subjects with a single observed modality degrades as the anchor fraction decreases and anchor latent structure becomes less representative of the full cohort. Third, WOVEN produces dense linear projection matrices $W^{(v)}$; variable importance rankings derived from $W^{(v)}$ entries are valid for anchor subjects and approximately valid for non-anchor subjects, but should be interpreted cautiously for subjects whose contribution to $W$ estimation is indirect.

Future work includes extension to feature-level missingness within otherwise observed modalities, longitudinal cohorts where the anchor set shifts across time points, sparse variants of the WOVEN objective for settings where interpretable loadings are required, and Bayesian uncertainty quantification for the Nystrom projection.

---

## References

Argelaguet, R., Arnol, D., Bredikhin, D., Deloro, Y., Velten, B., Marioni, J.C. and Stegle, O. (2020). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. *Genome Biology*, 21, 111. https://doi.org/10.1186/s13059-020-02015-1

Assefa, A.T., Vandesompele, J. and Thas, O. (2020). SPsimSeq: semi-parametric simulation of bulk and single-cell RNA-sequencing data. *Bioinformatics*, 36, 3276-3278. https://doi.org/10.1093/bioinformatics/btaa105

Ballard, J.L., Dai, Z., Shen, L. and Long, Q. (2025). JASMINE: a powerful representation learning method for enhanced analysis of incomplete multi-omics data. *bioRxiv*. https://doi.org/10.1101/2025.06.16.659949

Bengio, Y., Paiement, J.F., Vincent, P., Delalleau, O., Le Roux, N. and Ouimet, M. (2003). Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps, and spectral clustering. In *Advances in Neural Information Processing Systems*, 16. https://proceedings.neurips.cc/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html

Blaschko, M.B., Shelton, J.A., Bartels, A., Lampert, C.H. and Gretton, A. (2011). Semi-supervised kernel canonical correlation analysis with application to human fMRI. *Pattern Recognition Letters*, 32, 1572-1583. https://doi.org/10.1016/j.patrec.2011.02.011

Chalise, P., Raghavan, R. and Fridley, B.L. (2016). InterSIM: simulation tool for multiple integrative omic datasets. *Computer Methods and Programs in Biomedicine*, 128, 69-74. https://doi.org/10.1016/j.cmpb.2016.02.011

Chen, Z., Liu, Y., Zhang, Y. and Li, Q. (2023). Orthogonal latent space learning with feature weighting and graph learning for multimodal Alzheimer's disease diagnosis. *Medical Image Analysis*, 84, 102698. https://doi.org/10.1016/j.media.2022.102698

Chen, J., Wang, G. and Giannakis, G.B. (2019). Graph multiview canonical correlation analysis. *IEEE Transactions on Signal Processing*, 67, 2826-2838. https://doi.org/10.1109/TSP.2019.2918944

Duan, R., Gao, L., Gao, Y., Hu, Y., Xu, H., Huang, M. *et al.* (2021). Evaluation and comparison of multi-omics data integration methods for cancer subtyping. *PLOS Computational Biology*, 17, e1009224. https://doi.org/10.1371/journal.pcbi.1009224

Hasin, Y., Seldin, M. and Lusis, A. (2017). Multi-omics approaches to disease. *Genome Biology*, 18, 83. https://doi.org/10.1186/s13059-017-1215-1

He, M., Zhao, N. and Satten, G.A. (2024). MIDASim: a fast and simple simulator for realistic microbiome data. *Microbiome*, 12, 133. https://doi.org/10.1186/s40168-024-01822-z

Hornung, R. and Boulesteix, A.L. (2024). Prediction approaches for partly missing multi-omics covariate data: a literature review and an empirical comparison study. *WIREs Computational Statistics*, 16, e1626. https://doi.org/10.1002/wics.1626

Jack, C.R., Bernstein, M.A., Fox, N.C., Thompson, P., Alexander, G., Harvey, D. *et al.* (2008). The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *Journal of Magnetic Resonance Imaging*, 27, 685-691. https://doi.org/10.1002/jmri.21049

Lock, E.F., Hoadley, K.A., Marron, J.S. and Nobel, A.B. (2013). Joint and individual variation explained (JIVE) for integrated analysis of multiple data types. *Annals of Applied Statistics*, 7, 523-542. https://doi.org/10.1214/12-AOAS597

Ma, S., Ren, B., Mallick, H., Moon, Y.S., Schwager, E., Maharjan, S. *et al.* (2021). A statistical model for describing and simulating microbial community profiles. *PLOS Computational Biology*, 17, e1008913. https://doi.org/10.1371/journal.pcbi.1008913

Ma, S., Zeng, A.G.X., Haibe-Kains, B., Goldenberg, A., Dick, J.E. and Wang, B. (2025). Moving towards genome-wide data integration for patient stratification with Integrate Any Omics. *Nature Machine Intelligence*, 7, 29-42. https://doi.org/10.1038/s42256-024-00942-3

Mangnier, L., Bodein, A., Droit, A. and Leclercq, M. (2025). A systematic benchmark of integrative strategies for microbiome-metabolome data. *Communications Biology*, 8, 1057. https://doi.org/10.1038/s42003-025-08515-9

Mo, Q., Wang, S., Seshan, V.E., Olshen, A.B., Schultz, N., Sander, C. *et al.* (2013). Pattern discovery and cancer gene identification in integrated cancer genomic data. *Proceedings of the National Academy of Sciences*, 110, 4245-4250. https://doi.org/10.1073/pnas.1208949110

Nambiar, A., Melendez, C. and Noble, W.S. (2026). Unified imputation of missing data modalities and features in multi-omic data via shared representation learning. *bioRxiv*. https://doi.org/10.64898/2026.02.04.703630

Osang'ir, B.I., Gupta, S., Shkedy, Z. and Claesen, J. (2025). SUMO: an R package for simulating multi-omics data for methods development and testing. *Bioinformatics Advances*, 5, vbaf264. https://doi.org/10.1093/bioadv/vbaf264

Rohart, F., Gautier, B., Singh, A. and Le Cao, K.A. (2017). mixOmics: an R package for omics feature selection and multiple data integration. *PLOS Computational Biology*, 13, e1005752. https://doi.org/10.1371/journal.pcbi.1005752

Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7

Sankaran, K., Kodikara, S., Li, J.J. and Le Cao, K.A. (2025). Semisynthetic simulation for microbiome data analysis. *Briefings in Bioinformatics*, 26, bbaf051. https://doi.org/10.1093/bib/bbaf051

Singh, A., Shannon, C.P., Gautier, B., Rohart, F., Vacher, M., Tebbutt, S.J. and Le Cao, K.A. (2019). DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays. *Bioinformatics*, 35, 3055-3062. https://doi.org/10.1093/bioinformatics/bty1054

Sui, X., Xue, Z., Li, X., Zhao, X., Hu, Y. and Tian, Y. (2025). Multi-task learning for heterogeneous multi-source block-wise missing data. *arXiv*:2505.24413.

Tenenhaus, A., Philippe, C., Guillemot, V., Le Cao, K.A., Grill, J. and Frouin, V. (2014). Variable selection for generalized canonical correlation analysis. *Biostatistics*, 15, 569-583. https://doi.org/10.1093/biostatistics/kxu001

---

## Tables

**Table 1.** Benchmark results across 400 replicates (4 arms x 100 reps). Sil: mean silhouette score over all scored subjects. Sil (anchor): silhouette restricted to anchor subjects only, comparable across methods regardless of ESS. BER computed via per-fold DR refitting + LDA; chance level = 0.75 (four balanced classes). MOFA+ BER: NA (unsupervised, per-fold DR refitting not applicable). ImputeDIABLO complete: NA (imputation not applicable when all data observed).

| Condition | Method | Silhouette | Sil (anchor) | NMI | ESS | BER |
|---|---|---|---|---|---|---|
| Complete | WOVEN | **0.828** | 0.828 | **1.000** | 1.00 | **0.375** |
| Complete | DIABLO | 0.271 | 0.271 | 0.539 | 1.00 | 0.528 |
| Complete | MOFA+ | 0.204 | 0.204 | 0.500 | 1.00 | NA |
| MCAR 30% | WOVEN | 0.293 | **0.776** | **0.535** | **1.00** | **0.378** |
| MCAR 30% | DIABLO | 0.298 | 0.298 | 0.592 | 0.42 | 0.545 |
| MCAR 30% | ImputeDIABLO | 0.209 | 0.271 | 0.499 | 1.00 | 0.561 |
| MCAR 50% | WOVEN | 0.218 | **0.710** | 0.476 | **1.00** | **0.398** |
| MCAR 50% | DIABLO | 0.350 | 0.350 | **0.739** | 0.19 | 0.571 |
| MCAR 50% | MOFA+ | 0.179 | 0.192 | 0.462 | 1.00 | NA |
| MCAR 50% | ImputeDIABLO | 0.195 | 0.267 | 0.498 | 1.00 | 0.579 |
| MAR | WOVEN | 0.297 | **0.757** | 0.546 | **1.00** | **0.389** |
| MAR | DIABLO | 0.313 | 0.313 | **0.597** | 0.41 | 0.572 |
| MAR | ImputeDIABLO | 0.201 | 0.280 | 0.495 | 1.00 | 0.590 |

**Table 2.** ADNI subject retention and clinical characteristics by group. Dementia prevalence among WOVEN-recovered subjects is twice that of DIABLO-retained subjects, demonstrating that the intersection constraint systematically excludes the most severely affected patients. The 735 subjects with no data in any modality cannot be scored by any method.

| Group | N | CN | MCI | Dementia |
|---|---|---|---|---|
| DIABLO retained (complete, all 3 modalities) | 743 | 34% | 54% | 12% |
| WOVEN-only (missing at least 1 modality) | 944 | 28% | 47% | **25%** |
| Unscored (no data in any modality) | 735 | 53% | 36% | 11% |
| All enrolled | 2,422 | 38% | 47% | 15% |

---

*Supplementary materials: dual MCCA proof, Nystrom extension derivation, M-orthonormalization, simulation specifications, per-arm results tables, sensitivity figures.*
