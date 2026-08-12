library(jsonlite)

if (!requireNamespace("grf", quietly = TRUE)) {
  install.packages("grf", repos = "https://cloud.r-project.org")
}
library(grf)

set.seed(42)

case_dir <- "validation/cases/qrf_inf_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

X <- df[, c("x1", "x2")]
y <- df$y

model <- quantile_forest(X, y, quantiles = c(0.75), num.trees = 50)
pred <- as.numeric(predict(model, X, quantiles = c(0.75))$predictions)

tss <- sum((y - mean(y))^2)
ss <- sum((y - pred)^2)
r2 <- 1 - ss / tss

result <- list(
  coefficients = list(r_squared = as.numeric(r2)),
  standard_errors = list(r_squared = NaN)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
