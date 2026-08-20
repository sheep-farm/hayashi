df <- read.csv("validation/cases/nardl_simulated/data/data.csv")
y <- df$y
x <- df$x
t <- length(y)
lags <- 1

# Decompose x into positive/negative partial sums
dx <- diff(x)
x_pos <- numeric(t)
x_neg <- numeric(t)
cum_pos <- 0
cum_neg <- 0
x_pos[1] <- x[1]
x_neg[1] <- x[1]
for (i in 2:t) {
  if (dx[i-1] > 0) {
    cum_pos <- cum_pos + dx[i-1]
  } else {
    cum_neg <- cum_neg + dx[i-1]
  }
  x_pos[i] <- x[1] + cum_pos
  x_neg[i] <- x[1] + cum_neg
}

# Build ECM regression
n_eff <- t - lags - 1
n_reg <- 4 + lags * 3
Z <- matrix(0, n_eff, n_reg)
dy <- numeric(n_eff)

for (i in 1:n_eff) {
  t_i <- lags + i
  dy[i] <- y[t_i + 1] - y[t_i]
  Z[i, 1] <- 1.0
  Z[i, 2] <- y[t_i]
  Z[i, 3] <- x_pos[t_i]
  Z[i, 4] <- x_neg[t_i]
  for (j in 0:(lags-1)) {
    Z[i, 4 + j + 1] <- y[t_i - j] - y[t_i - j - 1]
    Z[i, 4 + lags + j + 1] <- x_pos[t_i - j] - x_pos[t_i - j - 1]
    Z[i, 4 + 2 * lags + j + 1] <- x_neg[t_i - j] - x_neg[t_i - j - 1]
  }
}

# OLS
fit <- lm(dy ~ 0 + Z)

# Use OLS manually for SE
eps <- fit$residuals
sigma2 <- sum(eps^2) / (n_eff - n_reg)
xtx <- t(Z) %*% Z + diag(1e-8, n_reg)
cov <- solve(xtx) * sigma2
se <- sqrt(diag(cov))

coef <- as.numeric(coef(fit))
names <- c("const", "y_{t-1}", "x^+_{t-1}", "x^-_{t-1}", "Dy_{t-1}", "Dx^+_{t-1}", "Dx^-_{t-1}")

result <- list(
  coefficients = setNames(as.list(coef), names),
  standard_errors = setNames(as.list(se), names)
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"), "\n")
