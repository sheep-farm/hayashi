# Independent R reference for two-step heteroskedastic linear IV-GMM.

library(gmm)
library(jsonlite)

case_dir <- "validation/cases/gmm_overidentified_simulated"
data <- read.csv(file.path(case_dir, "data", "data.csv"))

# MDS with prewhitening disabled estimates a heteroskedastic, serially
# independent moment covariance. This is not a translation of Greeners code.
fit <- gmm(
  y ~ x + endog,
  x = ~ x + z1 + z2,
  data = data,
  type = "twoStep",
  wmatrix = "optimal",
  vcov = "MDS",
  prewhite = FALSE,
  centeredVcov = FALSE
)

j_test <- specTest(fit)
standard_errors <- sqrt(diag(vcov(fit)))
result <- list(
  coef_const = as.numeric(coef(fit)["(Intercept)"]),
  coef_x = as.numeric(coef(fit)["x"]),
  coef_endog = as.numeric(coef(fit)["endog"]),
  se_const = as.numeric(standard_errors["(Intercept)"]),
  se_x = as.numeric(standard_errors["x"]),
  se_endog = as.numeric(standard_errors["endog"]),
  j_stat = as.numeric(j_test$test[1, "J-test"]),
  df_overid = as.numeric(fit$df),
  n_obs = as.numeric(fit$n)
)

cat(toJSON(result, auto_unbox = TRUE, digits = NA), "\n")
