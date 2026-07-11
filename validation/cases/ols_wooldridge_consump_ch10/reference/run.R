# Reference implementation in R for Wooldridge consump, Chapter 10, Example 10.4.

library(wooldridge)
library(jsonlite)

data(consump)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_consump_ch10/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Drop rows with missing values for any model variable to match the textbook sample.
consump_clean <- consump[complete.cases(consump[, c('gc', 'gy')]), ]

# Write CSV for Hayashi to read.
write.csv(consump_clean, file.path(data_dir, "consump.csv"), row.names = FALSE)

# Estimate the model from Chapter 10, Example 10.4.
model <- lm(gc ~ gy, data = consump_clean)

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
  nobs = nrow(consump_clean)
)

out_dir <- "validation/cases/ols_wooldridge_consump_ch10/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
