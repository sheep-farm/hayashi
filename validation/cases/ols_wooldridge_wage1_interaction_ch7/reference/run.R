# Reference implementation in R for Wooldridge wage1, Chapter 7, Example 7.6.

library(wooldridge)
library(jsonlite)

data(wage1)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_wage1_interaction_ch7/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Preprocess variables for the model.
wage1_proc <- wage1
wage1_proc$marrmale <- as.integer((wage1_proc$female == 0) & (wage1_proc$married == 1))
wage1_proc$marrfem <- as.integer((wage1_proc$female == 1) & (wage1_proc$married == 1))
wage1_proc$singfem <- as.integer((wage1_proc$female == 1) & (wage1_proc$married == 0))
# Write CSV for Hayashi to read.
write.csv(wage1_proc, file.path(data_dir, "wage1.csv"), row.names = FALSE)

# Estimate the model from Chapter 7, Example 7.6.
model <- lm(lwage ~ marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq, data = wage1_proc)

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
  nobs = nrow(wage1_proc)
)

out_dir <- "validation/cases/ols_wooldridge_wage1_interaction_ch7/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
