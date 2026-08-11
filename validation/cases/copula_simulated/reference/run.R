df <- read.csv(file.path("validation", "cases", "copula_simulated", "data", "data.csv"))

y <- df$y
x <- df$x

tau <- cor(y, x, method = "kendall")
spearman <- cor(y, x, method = "spearman")
# Gaussian copula parameter derived from Kendall's tau: rho = sin(pi * tau / 2)
corr <- sin(pi * tau / 2)

result <- list(
  coefficients = list(
    corr_yx = as.numeric(corr),
    kendall_yx = as.numeric(tau),
    spearman_yx = as.numeric(spearman)
  ),
  standard_errors = list(
    corr_yx = 0.0,
    kendall_yx = 0.0,
    spearman_yx = 0.0
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
