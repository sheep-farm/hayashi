# Reference implementation in R for Double Machine Learning.

library(jsonlite)

# Simulate data following the same DGP as Hayashi.
set.seed(42)
n <- 1000

# Generate controls
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
x4 <- rnorm(n)
x5 <- rnorm(n)

# Generate treatment (propensity score model)
ps <- 0.5 + 0.2 * x1 - 0.1 * x2 + 0.1 * x3
d <- rbinom(n, 1, plogis(ps))

# Generate outcome (treatment effect = 0.5)
theta <- 0.5
confounder <- 0.3 * x1 + 0.4 * x2 - 0.2 * x3 + 0.1 * x4 - 0.1 * x5
y <- theta * d + confounder + rnorm(n) * 0.5

# Create DataFrame
df <- data.frame(y = y, d = d, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5)

# Ensure the data directory exists.
data_dir <- "validation/cases/double_ml_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(df, file.path(data_dir, "double_ml.csv"), row.names = FALSE)

# Simple reference: partial out controls via OLS of y and d on X, then regress residuals.
X <- cbind(1, x1, x2, x3, x4, x5)

# Project y on X
y_coef <- lm(y ~ x1 + x2 + x3 + x4 + x5)$coefficients
y_hat <- cbind(1, x1, x2, x3, x4, x5) %*% y_coef
y_tilde <- y - y_hat

# Project d on X
d_coef <- lm(d ~ x1 + x2 + x3 + x4 + x5)$coefficients
d_hat <- cbind(1, x1, x2, x3, x4, x5) %*% d_coef
d_tilde <- d - d_hat

# Regress y_tilde on d_tilde
fit <- lm(y_tilde ~ d_tilde - 1)
ate_coef <- as.numeric(coef(fit))
ate_se <- as.numeric(summary(fit)$coefficients[, "Std. Error"])

result <- list(
  ate_coefficient = ate_coef,
  ate_standard_error = ate_se
)

out_dir <- "validation/cases/double_ml_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
