# Reference implementation in R for Wooldridge campus, Chapter 4, Example 4.4.

library(wooldridge)
library(jsonlite)

data(campus)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_campus_ch4/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(campus, file.path(data_dir, "campus.csv"), row.names = FALSE)

# Estimate the model from Chapter 4, Example 4.4.
model <- lm(lcrime ~ lenroll, data = campus)

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
  nobs = nrow(campus)
)

out_dir <- "validation/cases/ols_wooldridge_campus_ch4/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
