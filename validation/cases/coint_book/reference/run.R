# Reference implementation in R for the book cointegration/VECM case.
# Implements a manual Johansen ML procedure for a bivariate VECM(1) with rank 1.
# Requires only base R packages.

johansen_vecm <- function(data, lags, rank) {
  t_total <- nrow(data)
  k <- ncol(data)
  n_eff <- t_total - lags
  p_vecm <- lags - 1
  n_z_cols <- k * p_vecm + 1

  z_mat <- matrix(0, nrow = n_eff, ncol = n_z_cols)
  dy_target <- matrix(0, nrow = n_eff, ncol = k)
  y_lag_level <- matrix(0, nrow = n_eff, ncol = k)

  for (i in seq_len(n_eff)) {
    t_original <- lags + i
    dy_target[i, ] <- data[t_original, ] - data[t_original - 1, ]
    y_lag_level[i, ] <- data[t_original - 1, ]
    z_mat[i, 1] <- 1.0
    if (p_vecm > 0) {
      for (l in seq_len(p_vecm)) {
        lag_time <- t_original - l
        dy_lag <- data[lag_time, ] - data[lag_time - 1, ]
        start_col <- 1 + (l - 1) * k
        z_mat[i, start_col:(start_col + k - 1)] <- dy_lag
      }
    }
  }

  ztz <- crossprod(z_mat)
  ztz_inv <- solve(ztz)
  beta_0 <- ztz_inv %*% crossprod(z_mat, dy_target)
  r0 <- dy_target - z_mat %*% beta_0
  beta_1 <- ztz_inv %*% crossprod(z_mat, y_lag_level)
  r1 <- y_lag_level - z_mat %*% beta_1

  t_float <- n_eff
  s00 <- crossprod(r0) / t_float
  s11 <- crossprod(r1) / t_float
  s01 <- crossprod(r0, r1) / t_float
  s10 <- t(s01)

  s11_chol <- chol(s11)
  s11_inv_chol <- solve(s11_chol)
  s00_inv <- solve(s00)

  temp <- t(s11_inv_chol) %*% s10 %*% s00_inv %*% s01 %*% s11_inv_chol
  eig <- eigen(temp, symmetric = FALSE)

  # Keep real eigenvalues and sort descending.
  pairs <- list()
  for (i in seq_along(eig$values)) {
    if (abs(Im(eig$values[i])) < 1e-12) {
      pairs[[length(pairs) + 1]] <- list(value = as.numeric(Re(eig$values[i])), vector = Re(eig$vectors[, i]))
    }
  }
  pairs <- pairs[order(sapply(pairs, function(p) p$value), decreasing = TRUE)]

  beta_est <- matrix(0, nrow = k, ncol = rank)
  for (r in seq_len(rank)) {
    beta_vec <- s11_inv_chol %*% pairs[[r]]$vector
    beta_est[, r] <- beta_vec
  }

  cointegration_term <- r1 %*% beta_est
  alpha_est <- crossprod(r0, cointegration_term) %*% solve(crossprod(cointegration_term))

  # Simple OLS conditional standard errors for alpha: regress r0_j on the
  # cointegration term (already orthogonal to the constant) without intercept.
  alpha_se <- matrix(0, nrow = k, ncol = rank)
  for (r in seq_len(rank)) {
    ec <- cointegration_term[, r]
    ss_ec <- sum(ec * ec)
    for (j in seq_len(k)) {
      a <- alpha_est[j, r]
      resid <- r0[, j] - a * ec
      sigma2 <- sum(resid * resid) / max(1, n_eff - 2)
      alpha_se[j, r] <- sqrt(sigma2 / ss_ec)
    }
  }

  # Approximate beta standard errors from the static long-run OLS regression
  # y ~ x (with intercept). The slope SE is used for beta_y2; the intercept
  # SE is used as a rough proxy for beta_y1 (the Johansen vector is not
  # normalized here, so this is intentionally approximate).
  y_level <- data[, 1]
  x_level <- data[, 2]
  X_ols <- cbind(1.0, x_level)
  beta_ols <- solve(crossprod(X_ols), crossprod(X_ols, y_level))
  resid_ols <- as.vector(y_level - X_ols %*% beta_ols)
  sigma2_ols <- sum(resid_ols * resid_ols) / max(1, t_total - 2)
  cov_ols <- sigma2_ols * solve(crossprod(X_ols))
  beta_se <- matrix(0, nrow = k, ncol = rank)
  beta_se[1, 1] <- sqrt(cov_ols[1, 1])  # intercept SE as proxy for beta_y1
  beta_se[2, 1] <- sqrt(cov_ols[2, 2])  # slope SE as proxy for beta_y2

  list(alpha = alpha_est, beta = beta_est, alpha_se = alpha_se, beta_se = beta_se)
}

data_path <- "validation/cases/coint_book/data/coint.csv"
data <- as.matrix(read.csv(data_path)[, c("y", "x")])

res <- johansen_vecm(data, lags = 1, rank = 1)

result <- list(
  coefficients = list(
    beta_1_y1 = as.numeric(res$beta[1, 1]),
    beta_1_y2 = as.numeric(res$beta[2, 1]),
    alpha_1_y1 = as.numeric(res$alpha[1, 1]),
    alpha_1_y2 = as.numeric(res$alpha[2, 1])
  ),
  standard_errors = list(
    beta_1_y1 = as.numeric(res$beta_se[1, 1]),
    beta_1_y2 = as.numeric(res$beta_se[2, 1]),
    alpha_1_y1 = as.numeric(res$alpha_se[1, 1]),
    alpha_1_y2 = as.numeric(res$alpha_se[2, 1])
  )
)

json_str <- paste0(
  '{"coefficients":{',
  '"beta_1_y1":', format(result$coefficients$beta_1_y1, digits = 17),
  ',"beta_1_y2":', format(result$coefficients$beta_1_y2, digits = 17),
  ',"alpha_1_y1":', format(result$coefficients$alpha_1_y1, digits = 17),
  ',"alpha_1_y2":', format(result$coefficients$alpha_1_y2, digits = 17),
  '},"standard_errors":{',
  '"beta_1_y1":', format(result$standard_errors$beta_1_y1, digits = 6),
  ',"beta_1_y2":', format(result$standard_errors$beta_1_y2, digits = 6),
  ',"alpha_1_y1":', format(result$standard_errors$alpha_1_y1, digits = 6),
  ',"alpha_1_y2":', format(result$standard_errors$alpha_1_y2, digits = 6),
  '}}'
)
out_dir <- "validation/cases/coint_book/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(json_str, file.path(out_dir, "expected.json"), sep = "")
cat(json_str, "\n")
