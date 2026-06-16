\name{plot.woven}
\alias{plot.woven}
\title{Plot the WOVEN latent space}
\usage{
\method{plot}{woven}(x, labels = NULL, dims = c(1L, 2L),
     highlight_anchors = TRUE, ...)
}
\arguments{
\item{x}{a woven object from \code{\link{woven}()}}
\item{labels}{integer or factor of length n for coloring points.
  If NULL, all points are plotted in the same color.}
\item{dims}{integer vector of length 2: which latent dimensions to plot
  (default c(1, 2)).}
\item{highlight_anchors}{logical: whether to mark anchor subjects with
  filled points and projected subjects with open circles (default TRUE).}
\item{...}{further arguments passed to \code{\link[graphics]{plot}()}.}
}
\value{
Invisibly returns \code{x}.
}
\description{
Plots two latent dimensions from \code{fit$Z}, colored by group label.
Anchor subjects (complete cases used to learn W) are shown as filled points;
block-missing subjects projected into the space are shown as open circles.
}
\seealso{
\code{\link{woven}()}, \code{\link{woven_scores}()}
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
plot(fit, labels=Y)
}
