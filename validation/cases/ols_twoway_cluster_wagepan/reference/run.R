# Reference implementation in R for the Wooldridge wagepan two-way clustered
# OLS case.
#
# Uses base R for the Cameron-Gelbach-Miller covariance so the validation
# workflow does not depend on package-specific finite-sample corrections.

library(wooldridge)
library(jsonlite)

data(wagepan)

case_dir <- "validation/cases/ols_twoway_cluster_wagepan"
data_dir <- file.path(case_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

formula <- lwage ~ educ + exper + expersq + union + married
model <- lm(formula, data = wagepan)
model_rows <- as.integer(rownames(model.frame(model)))

X <- model.matrix(model)
u <- residuals(model)
cluster_nr <- wagepan$nr[model_rows]
cluster_year <- wagepan$year[model_rows]
cluster_intersection <- paste(cluster_nr, cluster_year, sep = "::")

cluster_meat <- function(X, u, clusters) {
  k <- ncol(X)
  meat <- matrix(0, nrow = k, ncol = k)

  for (cluster in unique(clusters)) {
    idx <- clusters == cluster
    xg <- X[idx, , drop = FALSE]
    ug <- u[idx]
    score <- crossprod(xg, ug)
    meat <- meat + score %*% t(score)
  }

  meat
}

n <- nrow(X)
k <- ncol(X)
g_nr <- length(unique(cluster_nr))
g_year <- length(unique(cluster_year))
g_min <- min(g_nr, g_year)

xtx_inv <- solve(crossprod(X))
meat_nr <- cluster_meat(X, u, cluster_nr)
meat_year <- cluster_meat(X, u, cluster_year)
meat_intersection <- cluster_meat(X, u, cluster_intersection)
meat <- meat_nr + meat_year - meat_intersection

finite_sample_correction <- (g_min / (g_min - 1)) * ((n - 1) / (n - k))
vcov_twoway <- finite_sample_correction * xtx_inv %*% meat %*% xtx_inv
se <- sqrt(pmax(diag(vcov_twoway), 0))

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))
names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"

std_errors <- as.numeric(se)
names(std_errors) <- colnames(X)
names(std_errors)[names(std_errors) == "(Intercept)"] <- "Intercept"

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
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
