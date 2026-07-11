# Reference implementation in R for Wooldridge intdef, Chapter 10, Example 10.2.

library(wooldridge)
library(jsonlite)

data(intdef)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_intdef_ch10/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Rename columns to avoid Python/patsy reserved literals ('inf' and 'def').
intdef_renamed <- intdef
names(intdef_renamed)[names(intdef_renamed) == "inf"] <- "inflation"
names(intdef_renamed)[names(intdef_renamed) == "def"] <- "deficit"

# Write CSV for Hayashi to read.
write.csv(intdef_renamed, file.path(data_dir, "intdef.csv"), row.names = FALSE)

# Estimate the model from Chapter 10, Example 10.2.
model <- lm(i3 ~ inflation + deficit, data = intdef_renamed)

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
  nobs = nrow(intdef)
)

out_dir <- "validation/cases/ols_wooldridge_intdef_ch10/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
