library(jsonlite)

case_dir <- "validation/cases/cpoisson_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

Y <- split(df$y, df$group)
X <- split(df$x, df$group)

ll_and_grad_hess <- function(b) {
  ll <- 0
  grad <- 0
  hess <- 0
  for (i in seq_along(Y)) {
    yg <- Y[[i]]
    xg <- X[[i]]
    S <- sum(yg)
    if (S == 0) next
    xb <- xg * b
    xb_max <- max(xb)
    exb <- exp(xb - xb_max)
    sum_exb <- sum(exb)
    ll <- ll + sum(yg * xb) - S * (xb_max + log(sum_exb))
    w <- exb / sum_exb
    grad <- grad + sum((yg - S * w) * xg)
    mx <- sum(w * xg)
    hess <- hess - S * (sum(w * xg * xg) - mx * mx)
  }
  list(ll = ll, grad = grad, hess = hess)
}

b <- 0.5
for (iter in 1:100) {
  g <- ll_and_grad_hess(b)
  if (abs(g$hess) < 1e-12) break
  db <- -g$grad / g$hess
  if (!is.finite(db)) break
  b <- b + db
  if (abs(db) < 1e-12) break
}

g <- ll_and_grad_hess(b)
se <- 1 / sqrt(-g$hess)

result <- list(
  coefficients = list(x = as.numeric(b)),
  standard_errors = list(x = as.numeric(se))
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
