# Explicit R reference for the Wooldridge wagepan panel FE clustered-SE case.
#
# This intentionally avoids plm::vcovHC because package defaults can use a
# different finite-sample correction from Greeners. The calculation mirrors
# within-transformed OLS with one-way CR1 clustered covariance.

library(jsonlite)

case_dir <- "validation/cases/panel_fe_cluster_wagepan"
csv_path <- file.path(case_dir, "data", "wagepan.csv")

if (!file.exists(csv_path)) {
  stop("wagepan.csv is missing; run data/gen.py before the reference script")
}

df <- read.csv(csv_path)
variables <- c("lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year")
df <- df[complete.cases(df[, variables]), variables]

y_name <- "lwage"
x_names <- c("union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87")
cluster_name <- "nr"

within_vector <- function(values, groups) {
  values - ave(values, groups, FUN = mean)
}

y <- within_vector(df[[y_name]], df[[cluster_name]])
X <- as.matrix(df[, x_names])
for (j in seq_along(x_names)) {
  X[, j] <- within_vector(X[, j], df[[cluster_name]])
}

xtx_inv <- solve(crossprod(X))
beta <- as.numeric(xtx_inv %*% crossprod(X, y))
names(beta) <- x_names

residuals <- as.numeric(y - X %*% beta)
n <- nrow(X)
k <- ncol(X)
clusters <- df[[cluster_name]]
g <- length(unique(clusters))

meat <- matrix(0, nrow = k, ncol = k)
for (cluster in unique(clusters)) {
  idx <- clusters == cluster
  score <- crossprod(X[idx, , drop = FALSE], residuals[idx])
  meat <- meat + score %*% t(score)
}

finite_sample_correction <- (g / (g - 1)) * ((n - 1) / (n - k))
vcov_cluster <- finite_sample_correction * xtx_inv %*% meat %*% xtx_inv
se <- sqrt(diag(vcov_cluster))
names(se) <- x_names

result <- list(
  coefficients = as.list(beta),
  standard_errors = as.list(se)
)

out_dir <- file.path(case_dir, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(
  result,
  file.path(out_dir, "expected.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 16
)

cat(toJSON(result, auto_unbox = TRUE, digits = 16))
