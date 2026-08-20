# Reference implementation in R for the Wooldridge mroz robust-IV case.
#
# The IV HC1 covariance is computed manually to match the Hayashi/Greeners
# convention and avoid package-specific finite-sample defaults.

library(wooldridge)
library(jsonlite)

data(mroz)

case_dir <- "validation/cases/iv_robust_mroz"
data_dir <- file.path(case_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

vars <- c("lwage", "educ", "exper", "expersq", "fatheduc", "motheduc")
mroz_clean <- mroz[complete.cases(mroz[, vars]), vars]
write.csv(mroz_clean, file.path(data_dir, "mroz.csv"), row.names = FALSE)

y <- as.matrix(mroz_clean[, "lwage"])
x <- model.matrix(~ educ + exper + expersq, data = mroz_clean)
z <- model.matrix(~ fatheduc + motheduc + exper + expersq, data = mroz_clean)

ztz_inv <- solve(crossprod(z))
x_hat <- z %*% ztz_inv %*% crossprod(z, x)
xhx_inv <- solve(crossprod(x_hat))
beta <- xhx_inv %*% crossprod(x_hat, y)

resid <- y - x %*% beta
n <- nrow(x)
k <- ncol(x)

meat <- crossprod(x_hat, x_hat * as.numeric(resid^2))
vcov_hc1 <- (n / (n - k)) * xhx_inv %*% meat %*% xhx_inv
std_errors <- sqrt(diag(vcov_hc1))

coefs <- as.numeric(beta)
names(coefs) <- colnames(x)
names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"

names(std_errors) <- colnames(x)
names(std_errors)[names(std_errors) == "(Intercept)"] <- "Intercept"

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- file.path(case_dir, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
