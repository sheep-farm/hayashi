# Reference implementation in R for the ARIMA(1,1,0) book case.
# Uses arima(..., order=c(1,1,0), method="CSS-ML") and maps ar1 to ar.L1
# to match the naming used by Python/statsmodels.  The intercept reported by
# Hayashi is not compared.

library(jsonlite)

case_dir <- dirname(dirname(normalizePath("validation/cases/arima_book/reference/run.R", mustWork = FALSE)))
df <- read.csv(file.path(case_dir, "data", "arima.csv"))

m <- arima(df$rw, order = c(1, 1, 0), method = "CSS-ML")

result <- list(
  coefficients = list(
    "ar.L1" = as.numeric(coef(m)["ar1"])
  ),
  standard_errors = list(
    "ar.L1" = as.numeric(sqrt(diag(vcov(m)))["ar1"])
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
