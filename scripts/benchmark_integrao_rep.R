#!/usr/bin/env Rscript
# benchmark_integrao_rep.R -- IntegrAO benchmark for one simulation rep
#
# BER methodology: per-fold DR refitting + k-NN out-of-sample projection.
# Matches WOVEN's ber_held_out_lda:
#   - Full fit (200 ep) on all scored subjects -> silhouette, NMI, ESS
#   - 3-fold BER: refit IntegrAO on 2/3 scored subjects (100 ep), project held-out
#     1/3 via k-NN weighted average of training embeddings (per available view,
#     then averaged). LDA on Z_trn, predict Z_val.
# k-NN only uses training subjects with OBSERVED data per view; handles
# block-missing training and test subjects.

suppressPackageStartupMessages({ library(Matrix); library(cluster) })

args <- commandArgs(trailingOnly=TRUE)
task_id <- as.integer(args[1]); data_dir <- args[2]
out_dir <- args[3];             grama_src <- args[4]
for (f in c("utils.R","metrics.R")) source(file.path(grama_src, f))
`%||%` <- function(a,b) if (!is.null(a)&&length(a)>0&&!is.na(a[1])) a else b

arm_idx <- ceiling(task_id/100L)
rep_num <- ((task_id-1L)%%100L)+1L
arm_ltr <- c("A","B","C","D")[arm_idx]
rep_str <- sprintf("%03d", rep_num)
cat(sprintf("[task %d] IntegrAO ARM %s rep %s\n", task_id, arm_ltr, rep_str))

load_rep <- function(suffix="") {
  fn <- sprintf("arm_%s_rep_%s%s.rds", arm_ltr, rep_str, suffix)
  sd <- if (suffix=="") "complete" else "missing"
  p  <- c(file.path(data_dir,sd,fn), file.path(data_dir,fn))
  p  <- p[file.exists(p)][1]
  if (is.na(p)) return(NULL); readRDS(p)
}
complete_rep <- load_rep("")
if (is.null(complete_rep)) stop("complete rep not found")
mod_names <- names(complete_rep$data)
V <- length(mod_names); n <- nrow(complete_rep$data[[1]])
labels <- complete_rep$labels
set.seed(rep_num*42L)

PY <- "/home/nbhtd/miniconda3/envs/integrAO/bin/python"
init_py <- function() {
  if (!requireNamespace("reticulate",quietly=TRUE)) stop("reticulate missing")
  if (!file.exists(PY)) stop("integrAO Python not found")
  reticulate::use_python(PY, required=TRUE)
  list(pd=reticulate::import("pandas"), np=reticulate::import("numpy"),
       it=reticulate::import("integrao.integrater"),
       nn=reticulate::import("sklearn.neighbors"))
}

# Build DataFrames; index = original R row numbers as strings.
# Subjects with all-NA rows in a view are excluded from that view's DataFrame.
make_dfs <- function(X_list_miss, subject_idx, py) {
  ids <- as.character(subject_idx)
  lapply(seq_len(V), function(v) {
    X  <- X_list_miss[[v]][subject_idx,,drop=FALSE]
    ok <- apply(X, 1, function(r) !all(is.na(r)))
    if (!any(ok)) return(NULL)
    Xs <- na_impute_median(X[ok,,drop=FALSE])
    kp <- apply(Xs, 2, var, na.rm=TRUE); kp <- is.finite(kp)&kp>1e-8
    if (sum(kp)<2L) kp <- seq_len(ncol(Xs))
    Xs <- Xs[,kp,drop=FALSE]
    py$pd$DataFrame(data=py$np$array(Xs,dtype="float64"),
                    index=reticulate::r_to_py(ids[ok]))
  })
}

# Run IntegrAO; return list(Z, orig_idx) where orig_idx are the original R indices
# that IntegrAO scored (may be subset of subject_idx if some have all views missing).
fit_integrao <- function(dfs, K, epochs, seed, py) {
  dfs <- Filter(Negate(is.null), dfs)
  if (length(dfs)<2L) return(NULL)
  ig <- py$it$integrao_integrater(datasets=dfs, embedding_dims=as.integer(K),
    alighment_epochs=as.integer(epochs), random_state=as.integer(seed))
  ig$network_diffusion()
  res <- ig$unsupervised_alignment()
  er  <- res[[1]]
  Z   <- as.matrix(if(is.data.frame(er)||is.matrix(er)) er
                   else reticulate::py_to_r(er$values))
  # rownames = original R indices (set by make_dfs)
  list(Z=Z, orig_idx=as.integer(rownames(Z)))
}

# k-NN out-of-sample projection.
# trn_scored: original R indices of training subjects scored by IntegrAO.
# Z_trn: rows correspond to trn_scored in ORDER (row i -> trn_scored[i]).
# val_orig: original R indices of test subjects to project.
# Returns Z_val (n_val x K), averaging over available views per test subject.
knn_project <- function(X_list_miss, trn_scored, val_orig, Z_trn, K, k=10L, py) {
  n_val <- length(val_orig)
  Z_acc <- matrix(0, n_val, K)
  cnt   <- integer(n_val)

  for (v in seq_len(V)) {
    Xt_raw <- X_list_miss[[v]][trn_scored,,drop=FALSE]
    Xv_raw <- X_list_miss[[v]][val_orig,,drop=FALSE]

    ok_trn <- which(apply(Xt_raw, 1, function(r) !all(is.na(r))))
    ok_val <- which(apply(Xv_raw, 1, function(r) !all(is.na(r))))
    if (length(ok_trn)<2L || length(ok_val)==0L) next

    Xt <- na_impute_median(Xt_raw[ok_trn,,drop=FALSE])
    Xv <- na_impute_median(Xv_raw[ok_val,,drop=FALSE])
    kp <- apply(Xt,2,var,na.rm=TRUE); kp <- is.finite(kp)&kp>1e-8
    if (sum(kp)<2L) next
    Xt <- Xt[,kp,drop=FALSE]; Xv <- Xv[,kp,drop=FALSE]

    nbrs <- py$nn$NearestNeighbors(n_neighbors=as.integer(min(k, nrow(Xt))))
    nbrs$fit(py$np$array(Xt, dtype="float64"))
    res_knn <- nbrs$kneighbors(py$np$array(Xv, dtype="float64"))
    dmat <- reticulate::py_to_r(res_knn[[1]])         # n_ok_val x k
    imat <- reticulate::py_to_r(res_knn[[2]]) + 1L   # 1-indexed rows of Xt / ok_trn

    wts <- 1/(dmat+1e-10); wts <- wts/rowSums(wts)
    for (i in seq_len(length(ok_val))) {
      # imat[i,] are positions in ok_trn -> use ok_trn[imat[i,]] as Z_trn rows
      zrow <- colSums(wts[i,] * Z_trn[ok_trn[imat[i,]],,drop=FALSE])
      Z_acc[ok_val[i],] <- Z_acc[ok_val[i],] + zrow
      cnt[ok_val[i]]    <- cnt[ok_val[i]] + 1L
    }
  }
  if (all(cnt==0L)) return(NULL)
  Z_acc / pmax(cnt, 1L)
}

# Per-fold BER: refit IntegrAO per fold, project test subjects via k-NN.
ber_pf <- function(X_list_miss, scored_idx, labels, K, cv_folds, py) {
  n_s <- length(scored_idx); pred <- integer(n_s)

  .lda_pred <- function(Ztr, ltr, Zval) {
    li <- as.integer(as.factor(ltr))
    if (length(unique(li))<2L) return(NULL)
    tryCatch({
      fit <- MASS::lda(Ztr, grouping=li); as.integer(predict(fit,Zval)$class)
    }, error=function(e) {
      cl <- sort(unique(li))
      ct <- do.call(rbind, lapply(cl, function(g) colMeans(Ztr[li==g,,drop=FALSE])))
      d  <- as.matrix(dist(rbind(Zval,ct))); nv <- nrow(Zval)
      tryCatch(cl[apply(d[seq_len(nv),(nv+1L):(nv+length(cl)),drop=FALSE],1L,which.min)],
               error=function(e2) rep(1L,nv))
    })
  }

  for (f in seq_along(cv_folds)) {
    vp <- cv_folds[[f]]; tp <- unlist(cv_folds[-f])
    to <- scored_idx[tp]; vo <- scored_idx[vp]

    dfs_trn <- make_dfs(X_list_miss, to, py)
    fr <- tryCatch(fit_integrao(dfs_trn, K, 100L, rep_num*1000L+f, py),
                   error=function(e) NULL)
    if (is.null(fr)) return(NA_real_)

    # fr$orig_idx = subset of `to` that IntegrAO scored (those with >= 1 view)
    # fr$Z rows correspond to fr$orig_idx in order (no reordering needed)
    trn_scored <- fr$orig_idx
    Z_trn      <- fr$Z

    if (length(trn_scored)<4L || length(unique(labels[trn_scored]))<2L)
      return(NA_real_)

    Z_val <- tryCatch(knn_project(X_list_miss, trn_scored, vo, Z_trn, K, 10L, py),
                      error=function(e) NULL)
    if (is.null(Z_val)) return(NA_real_)

    pv <- .lda_pred(Z_trn, labels[trn_scored], Z_val)
    if (is.null(pv)||length(pv)!=length(vp)) return(NA_real_)
    pred[vp] <- pv
  }

  li <- as.integer(as.factor(labels[scored_idx]))
  mean(vapply(sort(unique(li)), function(g) {
    idx <- which(li==g); 1-mean(pred[idx]==g)
  }, numeric(1L)), na.rm=TRUE)
}

run_integrao <- function(X_list_miss, labels, K=5L) {
  t0 <- proc.time()
  tryCatch({
    py <- init_py()
    dfs_all <- make_dfs(X_list_miss, seq_len(n), py)
    full <- fit_integrao(dfs_all, K, 200L, rep_num, py)
    if (is.null(full)) stop("full fit failed")

    Z_mat <- full$Z; idx_int <- full$orig_idx; n_used <- length(idx_int)
    Z_full <- matrix(NA_real_, nrow=n, ncol=K); Z_full[idx_int,] <- Z_mat
    lbl_sc <- labels[idx_int]

    sil <- tryCatch(mean(cluster::silhouette(as.integer(factor(lbl_sc)),dist(Z_mat))[,3],na.rm=TRUE),
                    error=function(e) NA_real_)
    nmi <- tryCatch(woven_nmi(Z_mat,lbl_sc), error=function(e) NA_real_)

    set.seed(rep_num*100L)
    folds <- split(sample(n_used), rep(seq_len(3L), length.out=n_used))
    ber   <- ber_pf(X_list_miss, idx_int, labels, K, folds, py)

    elapsed <- (proc.time()-t0)[["elapsed"]]
    cat(sprintf("   n=%d ess=%.2f sil=%.3f nmi=%.3f ber=%.3f [%.0fs]\n",
                n_used, n_used/n, sil%||%NA, nmi%||%NA, ber%||%NA, elapsed))
    list(Z=Z_full, n_used=n_used, ess=n_used/n, sil=sil, nmi=nmi, ber=ber,
         elapsed=elapsed, error=NA_character_)
  }, error=function(e) {
    cat(sprintf("   ERROR: %s\n", conditionMessage(e)))
    list(Z=NULL, n_used=0L, ess=0, sil=NA, nmi=NA, ber=NA,
         elapsed=(proc.time()-t0)[["elapsed"]], error=conditionMessage(e))
  })
}

all_conditions <- c("complete","mcar30","mcar50","mar")
K <- 5L; results <- list()
ckpt <- file.path(out_dir, sprintf("arm_%s_rep_%s_integrao_ckpt.rds", arm_ltr, rep_str))
if (file.exists(ckpt)) results <- readRDS(ckpt)

for (cond in all_conditions) {
  if (!is.null(results[[cond]])) { cat(sprintf("  %s [ckpt]\n",cond)); next }
  cat(sprintf("  Condition: %s\n", cond))
  X_list_miss <- if (cond=="complete") complete_rep$data else {
    mr <- load_rep(paste0("_",cond))
    if (is.null(mr)) { cat("    [SKIP]\n"); next }
    if (any(sapply(mr$data, function(X) any(is.na(X))))) mr$data else {
      lapply(seq_along(mod_names), function(v) {
        X <- complete_rep$data[[v]]; m <- mr$missingness_mask[,mod_names[v]]
        if (!is.null(m)) X[m,] <- NA; X
      })
    }
  }
  results[[cond]] <- run_integrao(X_list_miss, labels, K)
  saveRDS(results, ckpt)
}

dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
out_file <- file.path(out_dir, sprintf("arm_%s_rep_%s_integrao.rds", arm_ltr, rep_str))
if (all(all_conditions %in% names(results))) {
  saveRDS(list(task_id=task_id, arm=arm_ltr, rep=rep_num, results=results,
               metadata=list(date=Sys.time(), K=K, V=V, n=n, mods=mod_names)), out_file)
  cat(sprintf("[task %d] Saved: %s\n", task_id, out_file))
  if (file.exists(ckpt)) file.remove(ckpt)
} else {
  cat(sprintf("[task %d] Partial (%s)\n", task_id, paste(names(results),collapse=",")))
}
