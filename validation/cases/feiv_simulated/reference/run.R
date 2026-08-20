# Independent base-R implementation of the declared within-2SLS contract.

case_dir <- "validation/cases/feiv_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

demean <- function(values, groups) {
  values - ave(values, groups, FUN = mean)
}

y <- demean(df$y, df$id)
X <- matrix(demean(df$x, df$id), ncol = 1)
Z <- matrix(demean(df$z, df$id), ncol = 1)
colnames(X) <- "x"

X_hat <- Z %*% solve(crossprod(Z), crossprod(Z, X))
beta <- solve(crossprod(X_hat, X), crossprod(X_hat, y))
residuals <- y - X %*% beta

n <- nrow(X)
k <- ncol(X)
G <- length(unique(df$id))
df_resid <- n - k - (G - 1)
if (df_resid <= 0) {
  stop("FE-IV reference has no residual degrees of freedom")
}

sigma2 <- as.numeric(crossprod(residuals)) / df_resid
covariance <- sigma2 * solve(crossprod(X_hat, X))
std_error <- sqrt(diag(covariance))

cat(sprintf(
  '{"coefficients":{"x":%.17g},"standard_errors":{"x":%.17g}}',
  beta[1],
  std_error[1]
))
