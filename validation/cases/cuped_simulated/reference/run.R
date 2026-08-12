library(jsonlite)
setwd("/home/flavio/Documents/GitHub/hayashi")
df <- read.csv("validation/cases/cuped_simulated/data/data.csv")
theta <- cov(df$y, df$pre_y) / var(df$pre_y)
y_adj <- df$y - theta * (df$pre_y - mean(df$pre_y))
ate <- mean(y_adj[df$treated == 1]) - mean(y_adj[df$treated == 0])
unadj <- mean(df$y[df$treated == 1]) - mean(df$y[df$treated == 0])
vr <- 1 - var(y_adj) / var(df$y)
result <- list(
  coefficients = list(treatment_effect = ate, unadjusted_effect = unadj, theta = theta, variance_reduction = vr),
  standard_errors = list(treatment_effect = NA_real_, unadjusted_effect = NA_real_, theta = NA_real_, variance_reduction = NA_real_)
)
cat(toJSON(result, pretty = FALSE, digits = 10, auto_unbox = TRUE))
