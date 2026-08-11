df <- read.csv(file.path("validation", "cases", "nls_power_simulated", "data", "data.csv"))

fit <- nls(y ~ a * I(x^b), data = df, start = list(a = 1.5, b = 0.5), control = nls.control(maxiter = 200, tol = 1e-8))
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

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
