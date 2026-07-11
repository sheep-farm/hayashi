# Reference implementation in R for Wooldridge kielmc, Chapter 13, Example 13.1.

library(wooldridge)
library(jsonlite)

data(kielmc)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_kielmc_ch13/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Drop rows with missing values for any model variable to match the textbook sample.
kielmc_clean <- kielmc[complete.cases(kielmc[, c('lprice', 'y81', 'nearinc', 'y81nrinc')]), ]

# Write CSV for Hayashi to read.
write.csv(kielmc_clean, file.path(data_dir, "kielmc.csv"), row.names = FALSE)

# Estimate the model from Chapter 13, Example 13.1.
model <- lm(lprice ~ y81 + nearinc + y81nrinc, data = kielmc_clean)

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
  nobs = nrow(kielmc_clean)
)

out_dir <- "validation/cases/ols_wooldridge_kielmc_ch13/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
