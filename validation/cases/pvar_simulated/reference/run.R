df <- read.csv(file.path("validation", "cases", "pvar_simulated", "data", "data.csv"))

ids <- unique(df$id)
T <- max(table(df$id))
N <- length(ids)

Y1 <- matrix(0, N, T)
Y2 <- matrix(0, N, T)
for (i in 1:N) {
  idx <- df$id == ids[i]
  Y1[i, ] <- head(df$y1[idx], T)
  Y2[i, ] <- head(df$y2[idx], T)
}

# Within transformation
Y1w <- Y1 - rowMeans(Y1)
Y2w <- Y2 - rowMeans(Y2)

# Stacked data
y1_dep <- as.vector(Y1w[, 2:T])
y2_dep <- as.vector(Y2w[, 2:T])
X <- cbind(as.vector(Y1w[, 1:(T - 1)]), as.vector(Y2w[, 1:(T - 1)]))

ols <- function(y, X) {
  beta <- qr.solve(crossprod(X), crossprod(X, y))
  resid <- y - X %*% beta
  nobs <- length(y)
  k <- ncol(X)
  sigma2 <- sum(resid^2) / (nobs - k)
  se <- sqrt(diag(solve(crossprod(X))) * sigma2)
  list(beta = as.numeric(beta), se = as.numeric(se))
}

fit_y1 <- ols(y1_dep, X)
fit_y2 <- ols(y2_dep, X)

result <- list(
  coefficients = list(
    y1_L1.y1 = fit_y1$beta[1],
    y1_L1.y2 = fit_y1$beta[2],
    y2_L1.y1 = fit_y2$beta[1],
    y2_L1.y2 = fit_y2$beta[2]
  ),
  standard_errors = list(
    y1_L1.y1 = fit_y1$se[1],
    y1_L1.y2 = fit_y1$se[2],
    y2_L1.y1 = fit_y2$se[1],
    y2_L1.y2 = fit_y2$se[2]
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
