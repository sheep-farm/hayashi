# Reference implementation in R for Wooldridge mroz IV, Chapter 15, Example 15.1.

library(wooldridge)
library(AER)
library(jsonlite)

data(mroz)

# Ensure the data directory exists.
data_dir <- "validation/cases/iv_wooldridge_mroz_ch15/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only observations used in the IV model.
mroz_clean <- mroz[complete.cases(mroz[, c("lwage", "educ", "exper", "expersq", "fatheduc", "motheduc")]), ]

# Write CSV for Hayashi to read.
write.csv(mroz_clean, file.path(data_dir, "mroz.csv"), row.names = FALSE)

# Estimate the IV model from Chapter 15, Example 15.1.
model <- ivreg(lwage ~ educ + exper + expersq | fatheduc + motheduc + exper + expersq, data = mroz_clean)

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
  nobs = nrow(mroz_clean)
)

out_dir <- "validation/cases/iv_wooldridge_mroz_ch15/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
