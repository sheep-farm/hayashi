# Reference implementation in R for the GMM card case.
#
# Uses base-R two-stage least squares with HC1 heteroskedasticity-robust
# standard errors, which is the linear two-step GMM estimator under
# heteroskedasticity.

library(wooldridge)
library(jsonlite)

data(card)

# Ensure the data directory exists.
data_dir <- "validation/cases/gmm_card/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(card, file.path(data_dir, "card.csv"), row.names = FALSE)

# Prepare complete-case data with all needed variables.
vars <- c("lwage", "educ", "exper", "expersq", "smsa", "black", "south", "nearc4")
D <- na.omit(card[, vars])

y <- D$lwage
educ <- D$educ
n <- length(y)

# Exogenous regressors (excluding intercept and educ).
W <- as.matrix(cbind(D$exper, D$expersq, D$smsa, D$black, D$south))
# Instruments: same exogenous regressors plus nearc4.
Z <- as.matrix(cbind(1, D$nearc4, W))
# Endogenous + exogenous design matrix.
X <- as.matrix(cbind(1, educ, W))

# Two-step efficient GMM (exactly identified; equivalent to 2SLS).
# beta = (X'Z (Z' \Omega Z)^{-1} Z'X)^{-1} X'Z (Z' \Omega Z)^{-1} Z'y
# with \Omega = diag(resid^2).

# Initial 2SLS beta for residual calculation.
educ_hat <- as.vector(Z %*% solve(crossprod(Z)) %*% t(Z) %*% educ)
X_hat <- cbind(1, educ_hat, W)
beta <- as.vector(solve(crossprod(X_hat, X)) %*% crossprod(X_hat, y))

# Residuals from original endogenous design.
resid <- as.vector(y - X %*% beta)
e2 <- resid^2

# Efficient weighting matrix: (Z' \Omega Z)^{-1} with \Omega = diag(resid^2).
Z_omega_Z <- t(Z) %*% (e2 * Z)
W_opt <- solve(Z_omega_Z)

# GMM covariance.
XZ <- t(X) %*% Z
vcov <- solve(XZ %*% W_opt %*% t(XZ))
se <- sqrt(pmax(diag(vcov), 0))

name_map <- c(
  "(Intercept)" = "x0",
  "educ" = "x1",
  "exper" = "x2",
  "expersq" = "x3",
  "smsa" = "x4",
  "black" = "x5",
  "south" = "x6"
)

coef_names <- c("(Intercept)", "educ", "exper", "expersq", "smsa", "black", "south")
coefs <- as.numeric(beta)
names(coefs) <- name_map[coef_names]

std_errors <- as.numeric(se)
names(std_errors) <- name_map[coef_names]

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/gmm_card/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
