# Reference implementation in R for the GLSAR housing price case.

library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/glsar_hprice1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "hprice1.csv")
if (!file.exists(csv_path)) {
  # Fallback to the lasso/wls dataset if this case has not downloaded yet.
  fallback <- "validation/cases/wls_hprice1/data/hprice1.csv"
  if (file.exists(fallback)) {
    file.copy(fallback, csv_path)
  } else {
    stop("hprice1 CSV not found. Run wls_hprice1 reference first.")
  }
}

df <- read.csv(csv_path)

# Iterative GLSAR(1), matching statsmodels' adjusted Yule-Walker update.
y <- df$price
X <- model.matrix(~ lotsize + sqrft + bdrms, data = df)
rho <- 0.0
last_beta <- NULL

# statsmodels performs maxiter - 1 update steps, then one final fit.
for (iteration in seq_len(9)) {
  y_white <- y[-1] - rho * y[-length(y)]
  X_white <- X[-1, , drop = FALSE] - rho * X[-nrow(X), , drop = FALSE]
  fit <- lm.fit(X_white, y_white)
  beta <- fit$coefficients

  if (!is.null(last_beta)) {
    relative_change <- max(abs(last_beta - beta) / abs(last_beta))
    if (relative_change < 1e-4) {
      break
    }
  }
  last_beta <- beta

  # Estimate AR(1) rho from demeaned original residuals using the adjusted
  # Yule-Walker denominator n - 1, as statsmodels does by default.
  residuals <- y - drop(X %*% beta)
  centered <- residuals - mean(residuals)
  r0 <- sum(centered^2) / length(centered)
  r1 <- sum(centered[-length(centered)] * centered[-1]) / (length(centered) - 1)
  rho <- r1 / r0
}

y_white <- y[-1] - rho * y[-length(y)]
X_white <- X[-1, , drop = FALSE] - rho * X[-nrow(X), , drop = FALSE]
fit <- lm.fit(X_white, y_white)
sigma2 <- sum(fit$residuals^2) / fit$df.residual
coefs <- setNames(fit$coefficients, colnames(X))
std_errors <- setNames(
  sqrt(diag(sigma2 * solve(crossprod(X_white)))),
  colnames(X)
)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/glsar_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
