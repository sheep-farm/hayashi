# Reference implementation in R for Wooldridge wagepan panel FE, Chapter 14, Example 14.4.

library(wooldridge)
library(plm)
library(jsonlite)

data(wagepan)

# Ensure the data directory exists.
data_dir <- "validation/cases/panel_fe_wooldridge_wagepan_ch14/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only observations with complete model variables.
wagepan_clean <- wagepan[complete.cases(wagepan[, c("lwage", "union", "married", "d81", "d82", "d83", "d84", "d85", "d86", "d87", "nr", "year")]), ]

# Write CSV for Hayashi to read.
write.csv(wagepan_clean, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

# Estimate the fixed-effects model from Chapter 14, Example 14.4.
model <- plm(lwage ~ union + married + d81 + d82 + d83 + d84 + d85 + d86 + d87,
             data = wagepan_clean,
             model = "within",
             index = c("nr", "year"))

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
  nobs = nrow(wagepan_clean)
)

out_dir <- "validation/cases/panel_fe_wooldridge_wagepan_ch14/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
