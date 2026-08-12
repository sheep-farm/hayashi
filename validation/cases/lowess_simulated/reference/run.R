options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/lowess_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

fit <- lowess(df$x, df$y, f = 0.4, iter = 3)
# interpola os y suavizados nos x originais (já estão ordenados)
yhat <- approx(fit$x, fit$y, xout = df$x, rule = 2)$y

result <- list(
  coefficients = list(
    mean_yhat = mean(yhat),
    yhat_first = yhat[1],
    yhat_mid = yhat[100],
    yhat_last = yhat[200]
  ),
  standard_errors = list(
    mean_yhat = 0,
    yhat_first = 0,
    yhat_mid = 0,
    yhat_last = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
