library(jsonlite)
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)][1])
case_dir <- normalizePath(file.path(dirname(script_path), ".."))
df <- read.csv(file.path(case_dir, "data", "data.csv"))
fit <- isoreg(df$x, df$y)
steps <- c(fit$yf[1], fit$yf[41], fit$yf[81])
result <- list(
  coefficients = list(step_1 = as.numeric(steps[1]), step_2 = as.numeric(steps[2]), step_3 = as.numeric(steps[3])),
  standard_errors = list(step_1 = NA_real_, step_2 = NA_real_, step_3 = NA_real_)
)
cat(toJSON(result, pretty = FALSE, digits = 15, auto_unbox = TRUE))
