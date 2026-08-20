df <- read.csv(file.path("validation", "cases", "setar_simulated", "data", "data.csv"))
y <- df$y
n <- length(y)
delay <- 1

y_t <- y[(delay + 1):n]
y_lag <- y[1:(n - delay)]

# Grid search over threshold (between 15th and 85th percentiles of the threshold variable)
thresh_var <- y_lag
sorted_th <- sort(unique(thresh_var))
lo <- sorted_th[ceiling(0.15 * length(sorted_th))]
hi <- sorted_th[floor(0.85 * length(sorted_th))]
candidates <- seq(lo, hi, length.out = 50)

best_rss <- Inf
best <- NULL
for (th in candidates) {
  low_idx <- thresh_var < th
  high_idx <- !low_idx

  x_low <- cbind(1, y_lag[low_idx])
  x_high <- cbind(1, y_lag[high_idx])
  y_low <- y_t[low_idx]
  y_high <- y_t[high_idx]

  beta_low <- as.numeric(qr.solve(x_low, y_low))
  beta_high <- as.numeric(qr.solve(x_high, y_high))

  pred_low <- x_low %*% beta_low
  pred_high <- x_high %*% beta_high
  rss <- sum((y_low - pred_low)^2) + sum((y_high - pred_high)^2)

  if (rss < best_rss) {
    best_rss <- rss
    best <- list(th = th, beta_low = beta_low, beta_high = beta_high,
                 x_low = x_low, x_high = x_high, y_low = y_low, y_high = y_high)
  }
}

# OLS standard errors
se <- function(x, y, beta) {
  nobs <- length(y)
  k <- ncol(x)
  pred <- x %*% beta
  resid <- y - pred
  sigma2 <- sum(resid^2) / (nobs - k)
  sqrt(diag(solve(t(x) %*% x)) * sigma2)
}

se_low <- se(best$x_low, best$y_low, best$beta_low)
se_high <- se(best$x_high, best$y_high, best$beta_high)

result <- list(
  coefficients = list(
    low_const = best$beta_low[1],
    low_AR.L1 = best$beta_low[2],
    high_const = best$beta_high[1],
    high_AR.L1 = best$beta_high[2]
  ),
  standard_errors = list(
    low_const = se_low[1],
    low_AR.L1 = se_low[2],
    high_const = se_high[1],
    high_AR.L1 = se_high[2]
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
