# Reference implementation in R for the ARIMA GDP case.

library(forecast)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/arima_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from statsmodels mirror (Rdatasets).
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- read.csv(url)
macro <- macro[, c("year", "quarter", "realgdp")]
names(macro)[names(macro) == "realgdp"] <- "gdp"

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# ARIMA(1,1,1) on log GDP.
macro$lgdp <- log(macro$gdp)
model <- Arima(macro$lgdp, order = c(1, 1, 1))

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(sqrt(diag(model$var.coef)))
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/arima_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
