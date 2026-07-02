:
# Reference implementation in R for the Wooldridge wage1 OLS case.

library(wooldridge)
library(jsonlite)

data(wage1)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

# Estimate the canonical Mincer wage equation.
model <- lm(wage ~ educ + exper + tenure, data = wage1)

# Extract comparable quantities.
summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors),
  r_squared = summary_model$r.squared,
  nobs = nrow(wage1)
)

out_dir <- "validation/cases/ols_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also write a CSV aligned with the Hayashi output format for direct diffing.
out_csv <- data.frame(
  Variable = names(coefs),
  Coef = coefs,
  Std_Err = std_errors
)
write.csv(out_csv, file.path(out_dir, "expected.csv"), row.names = FALSE)
