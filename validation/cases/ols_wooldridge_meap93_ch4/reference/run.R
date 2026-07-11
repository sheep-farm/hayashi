# Reference implementation in R for Wooldridge meap93, Chapter 4, Examples 4.2 and 4.10.

library(wooldridge)
library(jsonlite)

data(meap93)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_meap93_ch4/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(meap93, file.path(data_dir, "meap93.csv"), row.names = FALSE)

# Estimate the model from Chapter 4, Examples 4.2 and 4.10.
model <- lm(math10 ~ ltotcomp + lstaff + lenroll, data = meap93)

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
  nobs = nrow(meap93)
)

out_dir <- "validation/cases/ols_wooldridge_meap93_ch4/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
