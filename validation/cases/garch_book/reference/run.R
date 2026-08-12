library(jsonlite)

case_dir <- "validation/cases/garch_book"
df <- read.csv(file.path(case_dir, "data", "garch.csv"))
r <- as.numeric(df$e[!is.na(df$e)])
T <- length(r)

garch_nll <- function(theta) {
  mu <- theta[1]
  omega <- exp(theta[2])
  ex <- exp(theta[3:4])
  denom <- 1 + sum(ex)
  alpha <- ex[1] / denom
  beta <- ex[2] / denom
  if (alpha + beta >= 0.9999) return(1e12)
  h <- numeric(T)
  h[1] <- omega / max(1e-12, 1 - alpha - beta)
  eps2 <- (r - mu)^2
  for (t in 2:T) {
    h[t] <- omega + alpha * eps2[t - 1] + beta * h[t - 1]
  }
  h <- pmax(h, 1e-12)
  ll <- -0.5 * (log(2 * pi) + log(h) + eps2 / h)
  return(-sum(ll))
}

rmean <- mean(r)
rvar <- mean((r - rmean)^2)
init <- c(rmean, log(0.1 * rvar), -2, 2)
res <- optim(init, garch_nll, method = "Nelder-Mead",
             control = list(maxit = 5000, reltol = 1e-8))
if (res$convergence != 0) {
  res <- optim(res$par, garch_nll, method = "BFGS")
}

mu <- res$par[1]
omega <- exp(res$par[2])
ex <- exp(res$par[3:4])
denom <- 1 + sum(ex)
alpha <- ex[1] / denom
beta <- ex[2] / denom

result <- list(
  coefficients = list(
    mu = as.numeric(mu),
    omega = as.numeric(omega),
    "alpha[1]" = as.numeric(alpha),
    "beta[1]" = as.numeric(beta)
  ),
  standard_errors = list(
    mu = NaN,
    omega = NaN,
    "alpha[1]" = NaN,
    "beta[1]" = NaN
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
