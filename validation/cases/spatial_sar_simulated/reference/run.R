# Independent base-R implementation of the concentrated MLE SAR model.

case_dir <- "validation/cases/spatial_sar_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
W <- as.matrix(read.csv(file.path(case_dir, "data", "W.csv"), header = FALSE))

y <- df$y
x <- df$x
n <- length(y)
X <- cbind(1, x)

# Eigenvalues of W for the log-determinant
w_eigvals <- eigen(W, only.values = TRUE)$values
w_eigvals <- Re(w_eigvals)

sar_loglik <- function(rho) {
  A <- diag(n) - rho * W
  y_star <- y - rho * W %*% y
  beta <- solve(crossprod(X), crossprod(X, y_star))
  resid <- y_star - X %*% beta
  rss <- as.numeric(crossprod(resid))
  sigma2 <- rss / n
  log_det <- sum(log(abs(1 - rho * w_eigvals)))
  log_det - n / 2 * log(2 * pi * sigma2) - rss / (2 * sigma2)
}

# Grid search over rho
rhos <- seq(-0.99, 0.99, length.out = 199)
lls <- vapply(rhos, sar_loglik, numeric(1))
best_rho <- rhos[which.max(lls)]

# Golden-section refinement
lo <- best_rho - 0.05
hi <- best_rho + 0.05
phi <- 0.6180339887498949
a <- lo
b <- hi
c <- b - phi * (b - a)
d <- a + phi * (b - a)
fc <- sar_loglik(c)
fd <- sar_loglik(d)
for (i in 1:50) {
  if (fc > fd) {
    b <- d
    d <- c
    fd <- fc
    c <- b - phi * (b - a)
    fc <- sar_loglik(c)
  } else {
    a <- c
    c <- d
    fc <- fd
    d <- a + phi * (b - a)
    fd <- sar_loglik(d)
  }
}
best_rho <- if (fc > fd) c else d

# Final beta and SEs
wy <- W %*% y
y_star <- y - best_rho * wy
beta <- solve(crossprod(X), crossprod(X, y_star))
resid <- y_star - X %*% beta
sigma2 <- as.numeric(crossprod(resid)) / n
xtx_inv <- solve(crossprod(X))
beta_se <- sqrt(diag(xtx_inv * sigma2))

result <- list(
  coefficients = list(
    rho = as.numeric(best_rho),
    `_cons` = as.numeric(beta[1]),
    x = as.numeric(beta[2])
  ),
  standard_errors = list(
    rho = 0.0,
    `_cons` = as.numeric(beta_se[1]),
    x = as.numeric(beta_se[2])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE))
