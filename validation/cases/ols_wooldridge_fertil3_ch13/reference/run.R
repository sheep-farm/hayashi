# Reference implementation in R for Wooldridge fertil3, Chapter 13, Example 13.3.

library(wooldridge)
library(jsonlite)

data(fertil3)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_fertil3_ch13/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Drop rows with missing values for any model variable to match the textbook sample.
fertil3_clean <- fertil3[complete.cases(fertil3[, c('gfr', 'pe', 'pe_1', 'pe_2', 'ww2', 'pill')]), ]

# Write CSV for Hayashi to read.
write.csv(fertil3_clean, file.path(data_dir, "fertil3.csv"), row.names = FALSE)

# Estimate the model from Chapter 13, Example 13.3.
model <- lm(gfr ~ pe + pe_1 + pe_2 + ww2 + pill, data = fertil3_clean)

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
  nobs = nrow(fertil3_clean)
)

out_dir <- "validation/cases/ols_wooldridge_fertil3_ch13/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
