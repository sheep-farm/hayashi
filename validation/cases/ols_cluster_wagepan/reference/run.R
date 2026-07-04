# Reference implementation in R for the Wooldridge wagepan clustered-OLS case.
#
# Uses base R for the one-way cluster-robust sandwich covariance so the
# validation workflow does not need extra R dependencies beyond wooldridge and
# jsonlite.

library(wooldridge)
library(jsonlite)

data(wagepan)

data_dir <- "validation/cases/ols_cluster_wagepan/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

formula <- lwage ~ educ + exper + expersq + union + married
model <- lm(formula, data = wagepan)

X <- model.matrix(model)
u <- residuals(model)
clusters <- wagepan$nr[as.integer(rownames(model.frame(model)))]

n <- nrow(X)
k <- ncol(X)
g <- length(unique(clusters))

xtx_inv <- solve(crossprod(X))
meat <- matrix(0, nrow = k, ncol = k)

for (cluster in unique(clusters)) {
  idx <- clusters == cluster
  xg <- X[idx, , drop = FALSE]
  ug <- u[idx]
  score <- crossprod(xg, ug)
  meat <- meat + score %*% t(score)
}

finite_sample_correction <- (g / (g - 1)) * ((n - 1) / (n - k))
vcov_cluster <- finite_sample_correction * xtx_inv %*% meat %*% xtx_inv
se <- sqrt(diag(vcov_cluster))

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

out_dir <- "validation/cases/ols_cluster_wagepan/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(
  result,
  file.path(out_dir, "expected.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 16
)

cat(toJSON(result, auto_unbox = TRUE, digits = 16))
