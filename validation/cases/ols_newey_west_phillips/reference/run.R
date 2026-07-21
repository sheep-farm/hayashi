# Reference implementation in R for the Wooldridge phillips Newey-West OLS case.
#
# The HAC covariance is computed manually to make the kernel, lag length, and
# finite-sample correction match the Hayashi/Greeners convention.

library(wooldridge)
library(jsonlite)

data(phillips)

case_dir <- "validation/cases/ols_newey_west_phillips"
data_dir <- file.path(case_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

phillips_clean <- phillips[complete.cases(phillips[, c("cinf", "unem")]), ]
write.csv(phillips_clean, file.path(data_dir, "phillips.csv"), row.names = FALSE)

model <- lm(cinf ~ unem, data = phillips_clean)

newey_west_vcov <- function(model, lags) {
  x <- model.matrix(model)
  residuals <- resid(model)
  n <- nrow(x)
  k <- ncol(x)
  xtx_inv <- solve(crossprod(x))

  meat <- crossprod(x, x * as.numeric(residuals^2))

  for (lag in seq_len(lags)) {
    weight <- 1 - lag / (lags + 1)
    omega_l <- matrix(0, nrow = k, ncol = k)

    for (t in (lag + 1):n) {
      scale <- residuals[t] * residuals[t - lag]
      omega_l <- omega_l + scale * tcrossprod(x[t, ], x[t - lag, ])
    }

    meat <- meat + weight * (omega_l + t(omega_l))
  }

  correction <- n / (n - k)
  correction * xtx_inv %*% meat %*% xtx_inv
}

vcov_nw <- newey_west_vcov(model, lags = 4)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- sqrt(diag(vcov_nw))
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- file.path(case_dir, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
