library(jsonlite)

if (!requireNamespace("gbm", quietly = TRUE)) {
  install.packages("gbm", repos = "https://cloud.r-project.org")
}
library(gbm)

set.seed(42)

case_dir <- "validation/cases/gbm_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

model <- gbm(y ~ x1 + x2, data = df, distribution = "gaussian",
             n.trees = 50, interaction.depth = 3, shrinkage = 0.1,
             bag.fraction = 1, n.minobsinnode = 10,
             keep.data = FALSE, cv.folds = 0, verbose = FALSE)

pred <- predict(model, newdata = df, n.trees = 50, type = "response")
mse <- mean((df$y - pred)^2)
tss <- sum((df$y - mean(df$y))^2)
r2 <- 1 - sum((df$y - pred)^2) / tss

result <- list(
  coefficients = list(mse = as.numeric(mse), r_squared = as.numeric(r2)),
  standard_errors = list(mse = NaN, r_squared = NaN)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
