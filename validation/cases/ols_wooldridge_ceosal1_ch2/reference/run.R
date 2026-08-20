# Reference implementation in R for Wooldridge ceosal1, Chapter 2, Example 2.11.

library(wooldridge)
library(jsonlite)

data(ceosal1)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_ceosal1_ch2/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(ceosal1, file.path(data_dir, "ceosal1.csv"), row.names = FALSE)

# Estimate the model from Chapter 2, Example 2.11.
model <- lm(lsalary ~ lsales, data = ceosal1)

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
  nobs = nrow(ceosal1)
)

out_dir <- "validation/cases/ols_wooldridge_ceosal1_ch2/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
