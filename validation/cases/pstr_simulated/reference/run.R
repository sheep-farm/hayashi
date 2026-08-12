df <- read.csv("validation/cases/pstr_simulated/data/data.csv")
y <- df$y
x <- df$x
q <- df$q
ids <- df$id
n <- length(y)

# Within demeaning
for (uid in unique(ids)) {
  mask <- ids == uid
  y[mask] <- y[mask] - mean(y[mask])
  x[mask] <- x[mask] - mean(x[mask])
  q[mask] <- q[mask] - mean(q[mask])
}

# Grid over c and gamma
cs <- seq(min(q), max(q), length.out = 41)
gammas <- seq(0.1, 15.0, length.out = 61)

best <- list(gamma = 1.0, c = 0.0, sse = Inf)
for (gamma in gammas) {
  for (c in cs) {
    g <- 1 / (1 + exp(-gamma * (q - c)))
    Z <- cbind(x, x * g)
    beta <- qr.solve(Z, y)
    res <- y - Z %*% beta
    sse <- sum(res^2)
    if (sse < best$sse) {
      best <- list(gamma = gamma, c = c, sse = sse)
    }
  }
}

gamma <- best$gamma
c <- best$c
g <- 1 / (1 + exp(-gamma * (q - c)))
Z <- cbind(x, x * g)
beta <- qr.solve(Z, y)
res <- y - Z %*% beta
sigma2 <- sum(res^2) / (n - 2 - 2)
cov <- solve(t(Z) %*% Z + 1e-8 * diag(2)) * sigma2
se <- sqrt(diag(cov))

result <- list(
  coefficients = list(
    gamma = gamma,
    c = c,
    beta0_x = as.numeric(beta[1]),
    beta1_x = as.numeric(beta[2])
  ),
  standard_errors = list(
    gamma = 0.0,
    c = 0.0,
    beta0_x = as.numeric(se[1]),
    beta1_x = as.numeric(se[2])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"), "\n")
