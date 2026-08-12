library(jsonlite)

case_dir <- "validation/cases/be_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

means <- aggregate(cbind(y, x) ~ id, data = df, FUN = mean)
n <- nrow(means)
X <- cbind(1, means$x)
Y <- means$y

b <- as.vector(solve(crossprod(X), crossprod(X, Y)))
resid <- as.vector(Y - X %*% b)
s2 <- sum(resid * resid) / (n - 2)
XX_inv <- solve(crossprod(X))
se <- sqrt(s2 * diag(XX_inv))

result <- list(
  coefficients = list(x0 = b[1], x1 = b[2]),
  standard_errors = list(x0 = se[1], x1 = se[2])
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
