df <- read.csv(file.path("validation", "cases", "svec_simulated", "data", "data.csv"))

Y <- as.matrix(df[, c("y1", "y2")])
T <- nrow(Y)
k <- ncol(Y)
lags <- 1

# VAR(1) with intercept by OLS
Y1 <- Y[(lags + 1):T, , drop = FALSE]
X <- cbind(1, Y[1:(T - lags), , drop = FALSE])

B_ols <- solve(t(X) %*% X) %*% t(X) %*% Y1
resid <- Y1 - X %*% B_ols

# Remove intercept, keep slope matrix A1
A1 <- t(B_ols[2:(1 + k), , drop = FALSE])

n_obs <- T - lags
n_cols_x <- 1 + k * lags
Sigma_u <- (t(resid) %*% resid) / (n_obs - n_cols_x)

# C(1) = (I - A1)^{-1}
C1_inv <- diag(k) - A1
C1 <- solve(C1_inv)

# Long-run covariance
long_run_cov <- C1 %*% Sigma_u %*% t(C1)

# Lower Cholesky
lr_chol <- t(chol(long_run_cov))

# B = C(1)^{-1} %*% long_run_chol
B <- C1_inv %*% lr_chol
A <- diag(k)

a_matrix <- as.vector(t(A))
b_matrix <- as.vector(t(B))

result <- list(
  a_matrix = setNames(as.list(a_matrix), paste0("a", seq_along(a_matrix) - 1)),
  b_matrix = setNames(as.list(b_matrix), paste0("b", seq_along(b_matrix) - 1))
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), "\n")
