library(jsonlite)
setwd("/home/flavio/Documents/GitHub/hayashi")
df <- read.csv("validation/cases/isotonic_simulated/data/data.csv")
fit <- isoreg(df$x, df$y)
steps <- c(fit$yf[1], fit$yf[41], fit$yf[81])
result <- list(
  coefficients = list(step_1 = as.numeric(steps[1]), step_2 = as.numeric(steps[2]), step_3 = as.numeric(steps[3])),
  standard_errors = list(step_1 = NA_real_, step_2 = NA_real_, step_3 = NA_real_)
)
cat(toJSON(result, pretty = FALSE, digits = 10, auto_unbox = TRUE))
