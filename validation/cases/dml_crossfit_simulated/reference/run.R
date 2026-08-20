options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/dml_crossfit_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))
df$d <- df$dvar - 0.5
n <- nrow(df)

set.seed(42)
K <- 5
folds <- cut(seq(1, n), breaks = K, labels = FALSE)[sample(n)]

df$ytilde <- rep(0, n)
df$dtilde <- rep(0, n)

for (k in 1:K) {
  train <- which(folds != k)
  test <- which(folds == k)
  gy <- lm(y ~ x1 + x2, data = df[train, ])
  gd <- lm(d ~ x1 + x2, data = df[train, ])
  df$ytilde[test] <- df$y[test] - predict(gy, newdata = df[test, ])
  df$dtilde[test] <- df$d[test] - predict(gd, newdata = df[test, ])
}

mod <- lm(ytilde ~ dtilde - 1, data = df)
theta <- as.numeric(coef(mod))
se <- as.numeric(summary(mod)$coefficients[, "Std. Error"])

result <- list(
  coefficients = list(
    theta = theta
  ),
  standard_errors = list(
    theta = se
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
