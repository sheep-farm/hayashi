# Reference implementation in R for the book ARMA(1,1) case.
# Replicates Hayashi's default Hannan-Rissanen two-step estimator.

df <- read.csv("validation/cases/arma_book/data/arma.csv")
z <- df$y
t <- length(z)
p <- 1
q <- 1

# Step 1: long AR regression to get proxy residuals.
p_long <- max(p + q, as.integer(t^0.25) + 2, 4)
n_long <- t - p_long

x_long <- matrix(1.0, nrow = n_long, ncol = p_long + 1)
y_long <- z[(p_long + 1):t]
for (i in 1:n_long) {
  ti <- p_long + i - 1
  for (l in 1:p_long) {
    x_long[i, l + 1] <- z[ti - l + 1]
  }
}
phi_long <- solve(t(x_long) %*% x_long) %*% t(x_long) %*% y_long
u_hat <- y_long - x_long %*% phi_long

# Step 2: regression y_t = c + ar1*z_{t-1} + ma1*u_hat_{t-1}.
start2 <- q
n_final <- n_long - start2

x_final <- matrix(1.0, nrow = n_final, ncol = 1 + p + q)
y_final <- z[(p_long + start2 + 1):t]
for (i in 1:n_final) {
  j <- start2 + i - 1
  zi <- p_long + j + 1
  x_final[i, 1] <- 1.0
  x_final[i, 2] <- z[zi - 1]
  x_final[i, 3] <- u_hat[j]
}

beta <- solve(t(x_final) %*% x_final) %*% t(x_final) %*% y_final
resid <- as.vector(y_final - x_final %*% beta)
sigma2 <- sum(resid^2) / n_final
vc <- sigma2 * solve(t(x_final) %*% x_final)
se <- sqrt(diag(vc))

result <- list(
  coefficients = list(
    intercept = as.numeric(beta[1]),
    "ar.L1" = as.numeric(beta[2]),
    "ma.L1" = as.numeric(beta[3])
  ),
  standard_errors = list(
    intercept = as.numeric(se[1]),
    "ar.L1" = as.numeric(se[2]),
    "ma.L1" = as.numeric(se[3])
  )
)

out_dir <- "validation/cases/arma_book/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fmt_json <- function(obj) {
  keys <- names(obj)
  parts <- vapply(seq_along(keys), function(i) {
    sprintf('"%s": %.10g', keys[i], obj[[i]])
  }, character(1))
  paste0("{", paste(parts, collapse = ", "), "}")
}

json <- paste0(
  '{"coefficients": ', fmt_json(result$coefficients),
  ', "standard_errors": ', fmt_json(result$standard_errors), "}"
)

writeLines(json, file.path(out_dir, "expected.json"), sep = "")
cat(json)
