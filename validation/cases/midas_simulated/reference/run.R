ylow <- read.csv("validation/cases/midas_simulated/data/ylow.csv")
xhigh <- read.csv("validation/cases/midas_simulated/data/xhigh.csv")

y <- ylow$y
x <- xhigh$x

T <- length(y)
freq <- 3
n_lags <- 12
poly_degree <- 2

build_x_midas <- function(gamma) {
  k <- 0:(n_lags - 1)
  Z <- cbind(1, k, k^2)
  g <- c(0, gamma[1], gamma[2])
  raw <- exp(Z %*% g)
  weights <- as.numeric(raw / sum(raw))
  xm <- numeric(T)
  for (t in 1:T) {
    base <- (t - 1) * freq + (freq - 1) + 1  # 1-based
    val <- 0
    for (lag in 1:n_lags) {
      idx <- base - (lag - 1)
      if (idx >= 1 && idx <= length(x)) {
        val <- val + weights[lag] * x[idx]
      }
    }
    xm[t] <- val
  }
  list(xm = xm, weights = weights)
}

sse <- function(gamma) {
  obj <- build_x_midas(gamma)
  xm <- obj$xm
  xm_mean <- mean(xm)
  y_mean <- mean(y)
  sxx <- sum((xm - xm_mean)^2)
  sxy <- sum((xm - xm_mean) * (y - y_mean))
  if (abs(sxx) < 1e-15) return(1e18)
  beta <- sxy / sxx
  alpha <- y_mean - beta * xm_mean
  sum((y - alpha - beta * xm)^2)
}

res <- optim(c(0, 0), sse, method = "Nelder-Mead")
best_gamma <- res$par
obj <- build_x_midas(best_gamma)
xm <- obj$xm
xm_mean <- mean(xm)
y_mean <- mean(y)
sxx <- sum((xm - xm_mean)^2)
sxy <- sum((xm - xm_mean) * (y - y_mean))
beta <- sxy / sxx
alpha <- y_mean - beta * xm_mean
r2 <- 1.0 - sum((y - alpha - beta * xm)^2) / sum((y - y_mean)^2)

result <- list(
  coefficients = list(
    alpha = alpha,
    beta = beta,
    r_squared = r2
  ),
  standard_errors = list(
    alpha = 0.0,
    beta = 0.0,
    r_squared = 0.0
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"), "\n")
