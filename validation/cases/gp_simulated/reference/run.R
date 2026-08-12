library(kernlab)
library(jsonlite)

case_dir <- "validation/cases/gp_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

# Gaussian process regression with RBF kernel and automatic length-scale estimation.
model <- gausspr(as.matrix(df$x), df$y, scaled = FALSE, kernel = "rbfdot")

y_hat <- as.numeric(predict(model, as.matrix(df$x)))

mse <- mean((df$y - y_hat) ^ 2)
ss_res <- sum((df$y - y_hat) ^ 2)
ss_tot <- sum((df$y - mean(df$y)) ^ 2)
r2 <- 1 - ss_res / ss_tot

result <- list(
  coefficients = list(r2 = r2, mse = mse),
  standard_errors = list(r2 = 0, mse = 0)
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
