# Reference implementation in R for Wooldridge barium, Chapter 10, Example 10.5.

library(wooldridge)
library(jsonlite)

data(barium)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_barium_ch10/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(barium, file.path(data_dir, "barium.csv"), row.names = FALSE)

# Estimate the model from Chapter 10, Example 10.5.
model <- lm(lchnimp ~ lchempi + lgas + lrtwex + befile6 + affile6 + afdec6, data = barium)

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
  nobs = nrow(barium)
)

out_dir <- "validation/cases/ols_wooldridge_barium_ch10/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
