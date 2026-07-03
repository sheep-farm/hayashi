# Reference implementation in R for the ARDL GDP case.

library(forecast)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/ardl_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from statsmodels mirror (Rdatasets).
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- read.csv(url)
macro <- macro[, c("year", "quarter", "realgdp", "realcons")]
names(macro)[names(macro) == "realgdp"] <- "gdp"
names(macro)[names(macro) == "realcons"] <- "cons"

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# ARDL(1,1): y_t on y_{t-1}, x_t and x_{t-1}.
macro <- macro[complete.cases(macro[, c("gdp", "cons")]), ]
n <- nrow(macro)
y <- macro$gdp[-1]
y_lag <- macro$gdp[-n]
x <- macro$cons[-1]
x_lag <- macro$cons[-n]
model <- lm(y ~ y_lag + x + x_lag)

summary_model <- summary(model)

name_map <- c(
  "(Intercept)" = "const",
  "y_lag" = "y.L1",
  "x" = "x1",
  "x_lag" = "x1.L1"
)

coefs <- as.numeric(coef(model))
names(coefs) <- name_map[names(coef(model))]

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- name_map[rownames(summary_model$coefficients)]

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/ardl_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
