# Reference implementation in R for Wooldridge 401k, Chapter 3, Example 3.3.

library(wooldridge)
library(jsonlite)

data(k401k)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_401k_ch3/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(k401k, file.path(data_dir, "401k.csv"), row.names = FALSE)

# Estimate the 401(k) participation equation from Example 3.3.
model <- lm(prate ~ mrate + age, data = k401k)

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
  nobs = nrow(k401k)
)

out_dir <- "validation/cases/ols_wooldridge_401k_ch3/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
