# Reference implementation in R for Wooldridge jtrain, Chapter 14, Example 14.3.

library(wooldridge)
library(jsonlite)

data(jtrain)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_jtrain_pooled_ch14/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Drop rows with missing values for any model variable to match the textbook sample.
jtrain_clean <- jtrain[complete.cases(jtrain[, c('lscrap', 'd88', 'd89', 'grant', 'grant_1', 'lsales', 'lemploy')]), ]

# Write CSV for Hayashi to read.
write.csv(jtrain_clean, file.path(data_dir, "jtrain.csv"), row.names = FALSE)

# Estimate the model from Chapter 14, Example 14.3.
model <- lm(lscrap ~ d88 + d89 + grant + grant_1 + lsales + lemploy, data = jtrain_clean)

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
  nobs = nrow(jtrain_clean)
)

out_dir <- "validation/cases/ols_wooldridge_jtrain_pooled_ch14/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
