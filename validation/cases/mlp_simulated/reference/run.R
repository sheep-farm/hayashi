df <- read.csv("validation/cases/mlp_simulated/data/data.csv")
y <- df$y
X <- df[, c("x1", "x2")]

set.seed(42)
n <- nrow(X)
# nnet with logistic hidden and linear output
m <- nnet::nnet(X, y, size = 10, linout = TRUE, maxit = 200, decay = 0, trace = FALSE)
pred <- as.numeric(predict(m, X))
ss_res <- sum((y - pred)^2)
ss_tot <- sum((y - mean(y))^2)
r2 <- 1.0 - ss_res / ss_tot

result <- list(
  coefficients = list(r_squared = r2),
  standard_errors = list(r_squared = 0.0)
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"), "\n")
