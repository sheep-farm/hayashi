# Reference implementation in R for the WLS hprice1 case.

library(wooldridge)
library(jsonlite)

data(hprice1)

# Ensure the data directory exists.
data_dir <- "validation/cases/wls_hprice1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Weight assumption: error variance proportional to lotsize.
hprice1$w <- 1 / hprice1$lotsize

# Write CSV for Hayashi to read.
write.csv(hprice1, file.path(data_dir, "hprice1.csv"), row.names = FALSE)

# Estimate weighted least squares.
model <- lm(price ~ lotsize + sqrft + bdrms, data = hprice1, weights = w)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/wls_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
