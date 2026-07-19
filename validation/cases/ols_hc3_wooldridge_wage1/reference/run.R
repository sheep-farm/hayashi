# Reference implementation in R for the Wooldridge wage1 HC3 OLS case.
#
# HC3 is computed manually to avoid introducing an additional R dependency and
# to make the finite-sample convention explicit.

library(wooldridge)
library(jsonlite)

data(wage1)

data_dir <- "validation/cases/ols_hc3_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

model <- lm(lwage ~ educ + exper + tenure, data = wage1)

x <- model.matrix(model)
residuals <- resid(model)
hat <- hatvalues(model)
xtx_inv <- solve(crossprod(x))
scaled_residuals <- residuals / (1 - hat)
meat <- crossprod(x, x * as.numeric(scaled_residuals^2))
vcov_hc3 <- xtx_inv %*% meat %*% xtx_inv

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- sqrt(diag(vcov_hc3))
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/ols_hc3_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
