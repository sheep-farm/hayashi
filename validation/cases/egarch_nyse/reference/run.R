# Reference implementation in R for the EGARCH NYSE returns case.

library(wooldridge)
library(rugarch)
library(jsonlite)

data(nyse)

# Ensure the data directory exists.
data_dir <- "validation/cases/egarch_nyse/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only non-missing returns.
nyse <- nyse[, c("price", "return")]
nyse <- nyse[complete.cases(nyse), ]

# Write CSV for Hayashi to read.
write.csv(nyse, file.path(data_dir, "nyse.csv"), row.names = FALSE)

# EGARCH(1,1) on NYSE returns.
spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)
model <- ugarchfit(spec, data = nyse$return)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))
# rugarch order for eGARCH is mu, omega, alpha1, beta1, gamma1.
# Hayashi order is mu, omega, alpha[1], gamma[1], beta[1].
names(coefs) <- c("mu", "omega", "gamma[1]", "beta[1]", "alpha[1]")

std_errors <- as.numeric(model@fit$matcoef[, " Std. Error"])
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/egarch_nyse/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, auto_unbox = TRUE))
