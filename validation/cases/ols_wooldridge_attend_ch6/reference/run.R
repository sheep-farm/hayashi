# Reference implementation in R for Wooldridge attend, Chapter 6, Example 6.3.

library(wooldridge)
library(jsonlite)

data(attend)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_attend_ch6/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(attend, file.path(data_dir, "attend.csv"), row.names = FALSE)

# Estimate the model from Chapter 6, Example 6.3.
model <- lm(stndfnl ~ atndrte + priGPA + ACT, data = attend)

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
  nobs = nrow(attend)
)

out_dir <- "validation/cases/ols_wooldridge_attend_ch6/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
