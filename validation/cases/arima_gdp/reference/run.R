# Reference implementation in R for the ARIMA GDP case.
#
# This script reports the global maximum of the exact Gaussian likelihood for
# an ARIMA(1,1,1) model on log US real GDP. This matches the MLE estimator
# exposed by Hayashi's `arima(..., method="mle")`.

library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/arima_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from local CSV or statsmodels mirror (Rdatasets).
local_csv <- "validation/cases/arima_gdp/data/macrodata.csv"
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- if (file.exists(local_csv)) read.csv(local_csv) else read.csv(url)
if (!"gdp" %in% names(macro)) {
  macro <- macro[, c("year", "quarter", "realgdp")]
  names(macro)[names(macro) == "realgdp"] <- "gdp"
}

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# ARIMA(1,1,1) on log GDP.
macro$lgdp <- log(macro$gdp)
z <- diff(macro$lgdp)
n <- length(z)
zc <- z - mean(z)

# Exact Gaussian log-likelihood for ARMA(1,1) on the centred differences.
exact_loglik <- function(phi, theta) {
  # MA(infinity) coefficients
  psi <- numeric(1000)
  psi[1] <- 1.0
  for (j in 2:1000) {
    val <- phi * psi[j - 1]
    if (j == 2) {
      val <- val + theta
    }
    psi[j] <- val
    if (j > n && abs(val) < 1e-12) {
      break
    }
  }

  # Autocovariances (sigma^2 = 1)
  max_lag <- min(n, 50)
  gamma <- numeric(max_lag + 1)
  for (k in 0:max_lag) {
    s <- 0.0
    for (j in 1:1000) {
      if (j + k > 1000) {
        break
      }
      s <- s + psi[j] * psi[j + k]
      if (j > n && abs(psi[j]) < 1e-12 && abs(psi[j + k]) < 1e-12) {
        break
      }
    }
    gamma[k + 1] <- s
  }

  # Durbin-Levinson innovations algorithm
  v <- numeric(n)
  v[1] <- gamma[1]
  phi_coefs <- list()
  phi_coefs[[1]] <- numeric(0)
  sum_log_v <- 0.0
  sum_eps2_v <- 0.0

  for (t in 1:n) {
    xhat <- 0.0
    if (t > 1) {
      prev <- phi_coefs[[t - 1]]
      for (j in seq_along(prev)) {
        xhat <- xhat + prev[j] * zc[t - 1 - j]
      }
    }
    eps <- zc[t] - xhat
    sum_log_v <- sum_log_v + log(v[t])
    sum_eps2_v <- sum_eps2_v + eps * eps / v[t]

    if (t < n) {
      k <- t
      num <- if (k <= max_lag) gamma[k + 1] else 0.0
      prev <- phi_coefs[[t]]
      for (j in seq_along(prev)) {
        lag <- k - 1 - (j - 1)
        num <- num - prev[j] * (if (lag <= max_lag) gamma[lag + 1] else 0.0)
      }
      phi_kk <- if (v[t] > 0) num / v[t] else 0.0
      new_phi <- numeric(min(k, max_lag))
      for (j in 1:min(k - 1, max_lag)) {
        prev_j <- prev[j]
        prev_kj <- if (k - 1 - j >= 1 && k - 1 - j <= length(prev)) prev[k - 1 - j] else 0.0
        new_phi[j] <- prev_j - phi_kk * prev_kj
      }
      new_phi[min(k, max_lag)] <- phi_kk
      v[k + 1] <- v[t] * (1.0 - phi_kk * phi_kk)
      phi_coefs[[k + 1]] <- new_phi
    }
  }

  sigma2 <- sum_eps2_v / n
  log_lik <- -0.5 * n * (1.0 + log(2.0 * pi * sigma2)) - 0.5 * sum_log_v
  list(log_lik = log_lik, sigma2 = sigma2)
}

# Coarse grid.
best_ll <- -Inf
best_phi <- 0.0
best_theta <- 0.0
best_sigma2 <- 0.0
phi_grid <- seq(-0.95, 0.95, by = 0.05)
theta_grid <- seq(-0.95, 0.95, by = 0.05)
for (phi in phi_grid) {
  for (theta in theta_grid) {
    if (phi <= -0.999 || phi >= 0.999 || theta <= -0.999 || theta >= 0.999) {
      next
    }
    res <- exact_loglik(phi, theta)
    if (!is.na(res$log_lik) && res$sigma2 > 0 && res$log_lik > best_ll) {
      best_ll <- res$log_lik
      best_phi <- phi
      best_theta <- theta
      best_sigma2 <- res$sigma2
    }
  }
}

# Fine refinement.
for (phi in seq(best_phi - 0.05, best_phi + 0.05, by = 0.01)) {
  for (theta in seq(best_theta - 0.05, best_theta + 0.05, by = 0.01)) {
    if (phi <= -0.999 || phi >= 0.999 || theta <= -0.999 || theta >= 0.999) {
      next
    }
    res <- exact_loglik(phi, theta)
    if (!is.na(res$log_lik) && res$sigma2 > 0 && res$log_lik > best_ll) {
      best_ll <- res$log_lik
      best_phi <- phi
      best_theta <- theta
      best_sigma2 <- res$sigma2
    }
  }
}

result <- list(
  coefficients = list(intercept = mean(z), "ar.L1" = best_phi, "ma.L1" = best_theta),
  standard_errors = list(intercept = 0.0, "ar.L1" = 0.0, "ma.L1" = 0.0)
)

out_dir <- "validation/cases/arima_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
