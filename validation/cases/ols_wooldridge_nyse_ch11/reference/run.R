# Reference implementation in R for Wooldridge nyse, Chapter 11, Example 11.4.

library(wooldridge)
library(jsonlite)

data(nyse)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_nyse_ch11/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Rename column to avoid reserved keyword in Hayashi/Python.
nyse_renamed <- nyse
names(nyse_renamed)[names(nyse_renamed) == "return"] <- "ret"

# Remove rows with missing lagged returns to match the textbook sample.
nyse_renamed <- nyse_renamed[complete.cases(nyse_renamed$ret, nyse_renamed$return_1), ]

# Write CSV for Hayashi to read.
write.csv(nyse_renamed, file.path(data_dir, "nyse.csv"), row.names = FALSE)

# Estimate the model from Chapter 11, Example 11.4.
model <- lm(ret ~ return_1, data = nyse_renamed)

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
  nobs = nrow(nyse)
)

out_dir <- "validation/cases/ols_wooldridge_nyse_ch11/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
