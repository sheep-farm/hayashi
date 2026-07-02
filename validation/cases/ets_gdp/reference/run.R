# Reference implementation in R for the ETS GDP case.

library(forecast)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/ets_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from statsmodels mirror (Rdatasets).
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- read.csv(url)
macro <- macro[, c("year", "quarter", "realgdp")]
names(macro)[names(macro) == "realgdp"] <- "gdp"

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# ETS on GDP (auto-selected model).
model <- ets(macro$gdp)

coefs <- as.numeric(model$par)
names(coefs) <- names(model$par)

std_errors <- as.numeric(sqrt(diag(model$variance)))
# Ensure names match the parameter vector.
std_errors <- rep(NA, length(coefs))
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/ets_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
