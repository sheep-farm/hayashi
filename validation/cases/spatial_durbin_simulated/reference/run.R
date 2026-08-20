# Independent base-R implementation of the concentrated MLE spatial Durbin model.

case_dir <- "validation/cases/spatial_durbin_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
W <- as.matrix(read.csv(file.path(case_dir, "data", "W.csv"), header = FALSE))

y <- df$y
x <- df$x
n <- length(y)
X <- cbind(1, x)
Wx <- W %*% x
Z <- cbind(X, Wx)
Wy <- W %*% y

w_eigvals <- eigen(W, only.values = TRUE)$values
w_eigvals <- Re(w_eigvals)

durbin_loglik <- function(rho) {
  y_star <- y - rho * Wy
  beta <- solve(crossprod(Z), crossprod(Z, y_star))
  resid <- y_star - Z %*% beta
  rss <- as.numeric(crossprod(resid))
  sigma2 <- rss / n
  log_det <- sum(log(abs(1 - rho * w_eigvals)))
  log_det - n / 2 * log(2 * pi * sigma2) - rss / (2 * sigma2)
}

rhos <- seq(-0.99, 0.99, length.out = 199)
lls <- vapply(rhos, durbin_loglik, numeric(1))
best_rho <- rhos[which.max(lls)]

lo <- best_rho - 0.05
hi <- best_rho + 0.05
phi <- 0.6180339887498949
a <- lo
b <- hi
c <- b - phi * (b - a)
d <- a + phi * (b - a)
fc <- durbin_loglik(c)
fd <- durbin_loglik(d)
for (i in 1:50) {
  if (fc > fd) {
    b <- d
    d <- c
    fd <- fc
    c <- b - phi * (b - a)
    fc <- durbin_loglik(c)
  } else {
    a <- c
    c <- d
    fc <- fd
    d <- a + phi * (b - a)
    fd <- durbin_loglik(d)
  }
}
best_rho <- if (fc > fd) c else d

result <- list(
  coefficients = list(rho = as.numeric(best_rho)),
  standard_errors = list(rho = NaN)
)

json_str <- jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)

