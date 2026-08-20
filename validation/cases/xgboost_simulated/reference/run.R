library(jsonlite)

library(xgboost)

set.seed(42)

case_dir <- "validation/cases/xgboost_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
X <- as.matrix(df[, c("x1", "x2")])
y <- df$y

dtrain <- xgb.DMatrix(data = X, label = y)
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 3,
  subsample = 1,
  colsample_bytree = 1,
  lambda = 1,
  alpha = 0,
  gamma = 0,
  nthread = 1
)
model <- xgb.train(params = params, data = dtrain, nrounds = 50)

pred <- predict(model, dtrain)
mse <- mean((y - pred)^2)
tss <- sum((y - mean(y))^2)
r2 <- 1 - sum((y - pred)^2) / tss

result <- list(
  coefficients = list(mse = as.numeric(mse), r_squared = as.numeric(r2)),
  standard_errors = list(mse = NaN, r_squared = NaN)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
