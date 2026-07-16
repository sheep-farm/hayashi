# Reference implementation in R for DCC-GARCH.

library(wooldridge)
library(jsonlite)

data(nyse)

# Rename return to ret so Hayashi can use it (return is a reserved word).
nyse$ret <- nyse$return
nyse$return_1 <- c(NA, nyse$ret[-nrow(nyse)])

# Ensure the data directory exists.
data_dir <- "validation/cases/dcc_garch_nyse/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(nyse[, c("ret", "return_1")], file.path(data_dir, "nyse.csv"), row.names = FALSE, na = "")

# Use only complete cases for the reference.
Z <- cbind(nyse$ret, nyse$return_1)
Z <- Z[complete.cases(Z), ]
t <- nrow(Z)

# GARCH(1,1) grid search
garch11 <- function(r) {
  var_init <- mean(r^2)
  best <- c(0.01, 0.05, 0.90)
  best_ll <- -Inf
  best_vols <- rep(var_init, length(r))
  n_grid <- 8
  for (oi in 0:(n_grid-1)) {
    omega <- 0.001 + 0.1 * oi / (n_grid - 1) * var_init
    for (ai in 0:(n_grid-1)) {
      alpha <- 0.01 + 0.3 * ai / (n_grid - 1)
      for (bi in 0:(n_grid-1)) {
        beta <- 0.5 + 0.48 * bi / (n_grid - 1)
        if (alpha + beta >= 0.99) next
        vols <- rep(var_init, length(r))
        ll <- 0
        for (i in 1:length(r)) {
          if (i > 1) {
            vols[i] <- omega + alpha * r[i-1]^2 + beta * vols[i-1]
          }
          vol <- max(vols[i], 1e-10)
          ll <- ll - 0.5 * log(2 * pi) - log(vol) - 0.5 * (r[i]^2) / vol
        }
        if (ll > best_ll) {
          best_ll <- ll
          best <- c(omega, alpha, beta)
          best_vols <- vols
        }
      }
    }
  }
  list(params = best, vols = best_vols, ll = best_ll)
}

k <- 2
std_resids <- matrix(0, t, k)
conditional_vols <- matrix(0, t, k)
garch_params <- matrix(0, k, 3)
garch_ll <- 0
for (j in 1:k) {
  fit <- garch11(Z[, j])
  garch_params[j, ] <- fit$params
  conditional_vols[, j] <- fit$vols
  std_resids[, j] <- Z[, j] / sqrt(pmax(fit$vols, 1e-10))
  garch_ll <- garch_ll + fit$ll
}

# Unconditional correlation of standardized residuals
q_bar <- matrix(0, k, k)
for (i in 1:t) {
  q_bar <- q_bar + outer(std_resids[i, ], std_resids[i, ])
}
q_bar <- q_bar / t

# DCC log-likelihood
dcc_ll <- function(alpha, beta) {
  q_prev <- q_bar
  ll <- 0
  for (i in 1:t) {
    s <- std_resids[i, ]
    q_t <- (1 - alpha - beta) * q_bar + alpha * outer(s, s) + beta * q_prev
    d_inv <- 1 / sqrt(pmax(diag(q_t), 1e-10))
    r_t <- diag(d_inv) %*% q_t %*% diag(d_inv)
    r_det <- max(det(r_t), 1e-300)
    r_inv <- solve(r_t + diag(1e-8, k))
    quad <- as.numeric(t(s) %*% r_inv %*% s)
    ll <- ll - 0.5 * (log(r_det) + quad)
    q_prev <- q_t
  }
  ll
}

# DCC grid search
best_alpha <- 0.01
best_beta <- 0.95
best_ll <- -Inf
n_grid <- 15
for (i in 0:(n_grid-1)) {
  for (j in 0:(n_grid-1)) {
    alpha <- 0.01 + 0.48 * i / (n_grid - 1)
    beta <- 0.01 + 0.48 * j / (n_grid - 1)
    if (alpha + beta >= 0.99) next
    ll <- dcc_ll(alpha, beta)
    if (ll > best_ll) {
      best_ll <- ll
      best_alpha <- alpha
      best_beta <- beta
    }
  }
}

total_ll <- garch_ll + best_ll
n_params <- k * 3 + 2
aic <- -2 * total_ll + 2 * n_params
bic <- -2 * total_ll + t * n_params

result <- list(
  dcc_alpha = best_alpha,
  dcc_beta = best_beta,
  ret_omega = garch_params[1, 1],
  ret_alpha = garch_params[1, 2],
  ret_beta = garch_params[1, 3],
  return_1_omega = garch_params[2, 1],
  return_1_alpha = garch_params[2, 2],
  return_1_beta = garch_params[2, 3],
  log_likelihood = total_ll,
  aic = aic,
  bic = bic
)

out_dir <- "validation/cases/dcc_garch_nyse/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
