library(jsonlite)

case_dir <- "validation/cases/varma_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
Y <- as.matrix(df[, c("y1", "y2")])
n <- nrow(Y)

# Hannan-Rissanen two-stage estimator for a bivariate VARMA(1,1)
# Stage 1: VAR(1) to obtain initial residuals
Y_t1 <- Y[2:n, , drop = FALSE]
Y_l1 <- Y[1:(n - 1), , drop = FALSE]
Phi <- t(sapply(1:2, function(i) lm.fit(Y_l1, Y_t1[, i])$coefficients))
a <- Y_t1 - Y_l1 %*% t(Phi)

# Stage 2: regress Y_t on lagged Y and lagged residuals (from stage 1)
Y_t2 <- Y[3:n, , drop = FALSE]
Y_l2 <- Y[2:(n - 1), , drop = FALSE]
a_l <- a[1:(nrow(a) - 1), , drop = FALSE]
X <- cbind(Y_l2, a_l)
coefs <- t(sapply(1:2, function(i) lm.fit(X, Y_t2[, i])$coefficients))
Phi2 <- coefs[, 1:2, drop = FALSE]
Theta <- coefs[, 3:4, drop = FALSE]

result <- list(
  coefficients = list(
    ar0 = 0.0,
    ar1 = 0.0,
    ar2 = as.numeric(Phi2[1, 1]),
    ar3 = as.numeric(Phi2[2, 1]),
    ar4 = as.numeric(Phi2[1, 2]),
    ar5 = as.numeric(Phi2[2, 2]),
    ma0 = as.numeric(Theta[1, 1]),
    ma1 = as.numeric(Theta[2, 1]),
    ma2 = as.numeric(Theta[1, 2]),
    ma3 = as.numeric(Theta[2, 2])
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
