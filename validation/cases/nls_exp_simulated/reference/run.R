df <- read.csv(file.path("validation", "cases", "nls_exp_simulated", "data", "data.csv"))

fit <- nls(y ~ a * exp(b * x), data = df, start = list(a = 2.0, b = -1.0), control = nls.control(maxiter = 200, tol = 1e-8))
sm <- summary(fit)

result <- list(
  coefficients = list(
    a = as.numeric(coef(fit)["a"]),
    b = as.numeric(coef(fit)["b"])
  ),
  standard_errors = list(
    a = as.numeric(sm$coefficients["a", "Std. Error"]),
    b = as.numeric(sm$coefficients["b", "Std. Error"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, pretty = TRUE), "\n")
