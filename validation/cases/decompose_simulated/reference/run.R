# Reference: classical additive seasonal decomposition using R's base decompose.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)][1])
case_dir <- normalizePath(file.path(dirname(script_path), ".."))
df <- read.csv(file.path(case_dir, "data", "data.csv"))

y <- df$y
n <- length(y)
period <- 12
first <- 6
last <- n - period / 2

result <- decompose(ts(y, frequency = period), type = "additive")

out <- list(
  coefficients = list(
    trend_first = as.numeric(result$trend[first]),
    trend_last = as.numeric(result$trend[last]),
    seasonal_first = as.numeric(result$seasonal[first]),
    seasonal_last = as.numeric(result$seasonal[last]),
    resid_first = as.numeric(result$random[first]),
    resid_last = as.numeric(result$random[last])
  ),
  standard_errors = list(
    trend_first = NaN,
    trend_last = NaN,
    seasonal_first = NaN,
    seasonal_last = NaN,
    resid_first = NaN,
    resid_last = NaN
  )
)

json_str <- jsonlite::toJSON(out, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)
