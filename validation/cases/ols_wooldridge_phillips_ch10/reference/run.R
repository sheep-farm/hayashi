# Reference implementation in R for Wooldridge phillips, Chapter 10, Example 10.1.

library(wooldridge)
library(jsonlite)

data(phillips)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_phillips_ch10/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(phillips, file.path(data_dir, "phillips.csv"), row.names = FALSE)

# Estimate the model from Chapter 10, Example 10.1.
model <- lm(inf ~ unem, data = phillips)

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
  nobs = nrow(phillips)
)

out_dir <- "validation/cases/ols_wooldridge_phillips_ch10/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
