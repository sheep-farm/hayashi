# Independent base-R implementation of the concentrated MLE SEM model.

case_dir <- "validation/cases/spatial_sem_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
W <- as.matrix(read.csv(file.path(case_dir, "data", "W.csv"), header = FALSE))

y <- df$y
x <- df$x
n <- length(y)
X <- cbind(1, x)

w_eigvals <- eigen(W, only.values = TRUE)$values
w_eigvals <- Re(w_eigvals)

sem_loglik <- function(lam) {
  B <- diag(n) - lam * W
  y_tilde <- B %*% y
  X_tilde <- B %*% X
  beta <- solve(crossprod(X_tilde), crossprod(X_tilde, y_tilde))
  resid <- y_tilde - X_tilde %*% beta
  rss <- as.numeric(crossprod(resid))
  sigma2 <- rss / n
  log_det <- sum(log(abs(1 - lam * w_eigvals)))
  log_det - n / 2 * log(2 * pi * sigma2) - rss / (2 * sigma2)
}

lams <- seq(-0.99, 0.99, length.out = 199)
lls <- vapply(lams, sem_loglik, numeric(1))
best_lam <- lams[which.max(lls)]

lo <- best_lam - 0.05
hi <- best_lam + 0.05
phi <- 0.6180339887498949
a <- lo
b <- hi
c <- b - phi * (b - a)
d <- a + phi * (b - a)
fc <- sem_loglik(c)
fd <- sem_loglik(d)
for (i in 1:50) {
  if (fc > fd) {
    b <- d
    d <- c
    fd <- fc
    c <- b - phi * (b - a)
    fc <- sem_loglik(c)
  } else {
    a <- c
    c <- d
    fc <- fd
    d <- a + phi * (b - a)
    fd <- sem_loglik(d)
  }
}
best_lam <- if (fc > fd) c else d

B <- diag(n) - best_lam * W
y_tilde <- B %*% y
X_tilde <- B %*% X
beta <- solve(crossprod(X_tilde), crossprod(X_tilde, y_tilde))
resid <- y_tilde - X_tilde %*% beta
sigma2 <- as.numeric(crossprod(resid)) / n
xtx_inv <- solve(crossprod(X_tilde))
beta_se <- sqrt(diag(xtx_inv * sigma2))

result <- list(
  coefficients = list(
    lambda = as.numeric(best_lam),
    `_cons` = as.numeric(beta[1]),
    x = as.numeric(beta[2])
  ),
  standard_errors = list(
    lambda = 0.0,
    `_cons` = as.numeric(beta_se[1]),
    x = as.numeric(beta_se[2])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE))
