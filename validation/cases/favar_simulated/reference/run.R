library(jsonlite)

case_dir <- "validation/cases/favar_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
Y <- as.matrix(df[, c("y1", "y2", "y3")])
n <- nrow(Y)

# standardise with population sd (ddof = 0) to match Python
mu <- colMeans(Y)
sigma <- sqrt(colMeans((Y - matrix(mu, n, 3, byrow = TRUE))^2))
Xs <- (Y - matrix(mu, n, 3, byrow = TRUE)) / matrix(sigma, n, 3, byrow = TRUE)

cov <- crossprod(Xs) / n
ev <- eigen(cov, symmetric = TRUE)
ord <- order(ev$values, decreasing = TRUE)
pc <- ev$vectors[, ord[1]]
factor <- as.vector(Xs %*% pc)
if (cor(factor, Xs[, 2]) < 0) factor <- -factor

# VAR(1) on [factor, y1]
Yvar <- cbind(factor, df$y1)
Ylag <- rbind(c(0, 0), Yvar[-n, , drop = FALSE])
Xmat <- cbind(1, Ylag[-1, , drop = FALSE])
Y_t <- Yvar[-1, , drop = FALSE]
coef <- as.matrix(solve(crossprod(Xmat), crossprod(Xmat, Y_t)))

result <- list(
  coefficients = list(
    const_F1 = as.numeric(coef[1, 1]),
    L1.F1_F1 = as.numeric(coef[2, 1]),
    L1.y1_F1 = as.numeric(coef[3, 1]),
    const_y1 = as.numeric(coef[1, 2]),
    L1.F1_y1 = as.numeric(coef[2, 2]),
    L1.y1_y1 = as.numeric(coef[3, 2])
  ),
  standard_errors = list(
    const_F1 = NaN,
    L1.F1_F1 = NaN,
    L1.y1_F1 = NaN,
    const_y1 = NaN,
    L1.F1_y1 = NaN,
    L1.y1_y1 = NaN
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
