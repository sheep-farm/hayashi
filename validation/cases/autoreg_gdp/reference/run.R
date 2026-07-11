# Reference implementation in R for the autoregressive GDP case.

library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/autoreg_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from local CSV or statsmodels mirror (Rdatasets).
local_csv <- "validation/cases/autoreg_gdp/data/macrodata.csv"
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- if (file.exists(local_csv)) read.csv(local_csv) else read.csv(url)
if (!"gdp" %in% names(macro)) {
  macro <- macro[, c("year", "quarter", "realgdp")]
  names(macro)[names(macro) == "realgdp"] <- "gdp"
}

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# Conditional AR(1) on GDP with a constant and linear trend.
y <- macro$gdp[-1]
y_lag <- macro$gdp[-nrow(macro)]
trend <- seq.int(2, nrow(macro))
model <- lm(y ~ y_lag + trend)

coefs <- c(
  const = unname(coef(model)["(Intercept)"]),
  trend = unname(coef(model)["trend"]),
  "y.L1" = unname(coef(model)["y_lag"])
)

std_errors <- c(
  const = unname(summary(model)$coefficients["(Intercept)", "Std. Error"]),
  trend = unname(summary(model)$coefficients["trend", "Std. Error"]),
  "y.L1" = unname(summary(model)$coefficients["y_lag", "Std. Error"])
)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/autoreg_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
