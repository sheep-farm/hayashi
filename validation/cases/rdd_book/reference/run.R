# Reference implementation in R for RDD using the Hayashi book DGP.
#
# Local linear regression with triangular kernel and an Imbens-Kalyanaraman
# bandwidth selector.

library(jsonlite)

CASE_DIR <- "validation/cases/rdd_book"
DATA_DIR <- file.path(CASE_DIR, "data")
CSV_PATH <- file.path(DATA_DIR, "rdd_book.csv")

if (!file.exists(CSV_PATH)) {
  stop("rdd_book.csv not found; run the Python reference first")
}

df <- read.csv(CSV_PATH)
x <- df$x
y <- df$y
cutoff <- 0.0

triangular_weight <- function(u) max(1 - abs(u), 0)

local_poly_wls <- function(y_s, x_c, w, order) {
  n <- length(y_s)
  p <- order + 1
  X <- outer(x_c, 0:order, function(x, j) x^j)
  XtWX <- crossprod(X, X * w)
  XtWy <- crossprod(X, y_s * w)
  XtWX_inv <- solve(XtWX)
  beta <- XtWX_inv %*% XtWy
  y_hat <- X %*% beta
  resid <- y_s - y_hat
  scale <- n / max(n - p, 1)
  meat <- matrix(0, p, p)
  for (i in seq_len(n)) {
    xi <- X[i, ]
    meat <- meat + scale * (w[i]^2) * (resid[i]^2) * (xi %o% xi)
  }
  vcov <- XtWX_inv %*% meat %*% XtWX_inv
  list(beta = beta, vcov = vcov)
}

side_fit <- function(y_s, x_s, cutoff, h, order, side) {
  if (side == "left") {
    mask <- x_s < cutoff
  } else {
    mask <- x_s >= cutoff
  }
  xs <- (x_s[mask] - cutoff)
  ys <- y_s[mask]
  ws <- sapply(xs / h, triangular_weight)
  keep <- ws > 0
  xs <- xs[keep]
  ys <- ys[keep]
  ws <- ws[keep]
  fit <- local_poly_wls(ys, xs, ws, order)
  list(beta = fit$beta, vcov = fit$vcov, n = length(ys))
}

ik_bandwidth <- function(y_s, x_s, cutoff, order) {
  n <- length(x_s)
  if (n < 10) return(1.0)
  x_sd <- sd(x_s)
  if (x_sd < 1e-15) return(1.0)
  h0 <- 1.84 * x_sd * n^(-0.2)
  q <- order + 1

  side_fit_pilot <- function(side) {
    if (side == "left") {
      mask <- x_s < cutoff
    } else {
      mask <- x_s >= cutoff
    }
    xs <- (x_s[mask] - cutoff)
    ys <- y_s[mask]
    keep <- abs(xs) <= h0
    xs <- xs[keep]
    ys <- ys[keep]
    if (length(ys) < q + 2) return(c(0.0, 1.0))
    ws <- rep(1.0, length(ys))
    fit <- local_poly_wls(ys, xs, ws, q)
    deriv_coeff <- fit$beta[q + 1]
    y_hat <- outer(xs, 0:q, function(x, j) x^j) %*% fit$beta
    resid <- ys - y_hat
    resid_var <- sum(resid^2) / max(length(ys) - (q + 1), 1)
    c(deriv_coeff, resid_var)
  }

  left <- side_fit_pilot("left")
  right <- side_fit_pilot("right")
  m_left <- left[1]
  sigma2_left <- left[2]
  m_right <- right[1]
  sigma2_right <- right[2]

  b_jump <- m_right - m_left
  if (abs(b_jump) < 1e-12) return(h0)

  n_window <- sum(abs(x_s - cutoff) <= h0)
  f_c <- max(n_window / (2.0 * h0 * n), 1e-10)

  c_k <- 3.4375
  exponent <- 1.0 / (2.0 * order + 3.0)
  h_star <- (c_k * (sigma2_left + sigma2_right) / (n * f_c * b_jump^2))^exponent
  max(min(h_star, 2.0 * x_sd), 0.05 * x_sd)
}

order <- 1
h <- ik_bandwidth(y, x, cutoff, order)

left_fit <- side_fit(y, x, cutoff, h, order, "left")
right_fit <- side_fit(y, x, cutoff, h, order, "right")

tau <- as.numeric(right_fit$beta[1] - left_fit$beta[1])
var_tau <- max(left_fit$vcov[1, 1] + right_fit$vcov[1, 1], 0.0)
se <- sqrt(var_tau)

result <- list(
  coefficients = list(tau = tau),
  standard_errors = list(tau = se)
)

out_dir <- file.path(CASE_DIR, "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
