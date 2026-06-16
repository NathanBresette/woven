\name{woven_scores}
\alias{woven_scores}
\title{Extract latent scores for new subjects}
\usage{
woven_scores(fit, X_list_new)
}
\arguments{
\item{fit}{woven object from \code{\link{woven}()}}
\item{X_list_new}{list of V matrices (n_new x p_v). Set entire rows to NA
  for subjects missing that modality block. Every subject must have at least
  one non-missing view.}
}
\value{
Numeric matrix n_new x K of consensus latent scores. Subjects with
  no observed data in any view receive a row of NA.
}
\description{
Projects new subjects into the trained WOVEN latent space and returns an
n_new x K score matrix. Uses direct linear projection (x \%*\% W_v) for each
available modality, then averages across observed views.

For class predictions on new subjects, use \code{\link{woven_predict}()} instead.
}
\seealso{
\code{\link{woven_predict}()} for class predictions,
\code{\link{woven}()} for model fitting.
}
\examples{
# minimal example data (n=20, 14 anchors, 6 partial)
set.seed(1); n <- 20; K <- 2L
X1 <- matrix(rnorm(n*5), n, 5); X2 <- matrix(rnorm(n*4), n, 4)
Y  <- rep(1:2, each = n/2)
# Rows 15-20: alternate missing view 1 or view 2 (never both)
miss <- matrix(FALSE, n, 2)
miss[c(15,17,19), 1] <- TRUE   # miss view 1
miss[c(16,18,20), 2] <- TRUE   # miss view 2
X1[miss[,1],] <- NA; X2[miss[,2],] <- NA
anchor_idx <- which(rowSums(miss)==0)
fit <- woven(list(X1,X2),Y=Y,anchor_idx=anchor_idx,K=K)
dim(woven_scores(fit, list(X1,X2)))
}
