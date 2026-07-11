# Reference implementation in R for Wooldridge hprice2, Chapter 6, Example 6.2.

library(wooldridge)
library(jsonlite)

data(hprice2)

# Ensure the data directory exists.
data_dir <- "validation/cases/ols_wooldridge_hprice2_ch6/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Preprocess variables for the model.
hprice2_proc <- hprice2
hprice2_proc$ldist <- log(hprice2_proc$dist)
hprice2_proc$roomsq <- hprice2_proc$rooms^2
# Write CSV for Hayashi to read.
write.csv(hprice2_proc, file.path(data_dir, "hprice2.csv"), row.names = FALSE)

# Estimate the model from Chapter 6, Example 6.2.
model <- lm(lprice ~ lnox + ldist + rooms + roomsq + stratio, data = hprice2_proc)

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
  nobs = nrow(hprice2_proc)
)

out_dir <- "validation/cases/ols_wooldridge_hprice2_ch6/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
