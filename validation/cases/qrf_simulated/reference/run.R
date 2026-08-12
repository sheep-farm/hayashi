library(jsonlite)

if (!requireNamespace("quantregForest", quietly = TRUE)) {
  install.packages("quantregForest", repos = "https://cloud.r-project.org")
}
library(quantregForest)

set.seed(42)

case_dir <- "validation/cases/qrf_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
X <- df[, c("x1", "x2")]
y <- df$y

model <- quantregForest(x = X, y = y, ntree = 50, mtry = 2,
                        nodesize = 1, keep.inbag = TRUE)
pred <- predict(model, X, quantiles = 0.75)

tss <- sum((y - mean(y))^2)
ss <- sum((y - pred)^2)
r2 <- 1 - ss / tss

result <- list(
  coefficients = list(r_squared = as.numeric(r2)),
  standard_errors = list(r_squared = NaN)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
