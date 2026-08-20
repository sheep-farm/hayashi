# Independent random-forest in-sample R² reference.
# randomForest does not expose a direct `max_depth` equivalent, so `maxnodes`
# is set to 2^5 = 32 to approximate scikit-learn's `max_depth=5`.

case_dir <- "validation/cases/rf_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

X <- df[, c("x1", "x2")]
y <- df$y

library(randomForest)

set.seed(123)
model <- randomForest(
  x = X,
  y = y,
  ntree = 50,
  mtry = 2,
  maxnodes = 32,
  nodesize = 1,
  importance = FALSE
)

pred <- predict(model, X)
ss_res <- as.numeric(crossprod(y - pred))
ss_tot <- as.numeric(crossprod(y - mean(y)))
r2 <- 1 - ss_res / ss_tot

result <- list(
  coefficients = list(r_squared = r2),
  standard_errors = list(r_squared = NaN)
)

json_str <- jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)

