df <- read.csv(file.path("validation", "cases", "nls_cobb_douglas_simulated", "data", "data.csv"))

fit <- nls(y ~ a * I(x1^b1) * I(x2^b2), data = df, start = list(a = 1.0, b1 = 0.3, b2 = 0.5), control = nls.control(maxiter = 200, tol = 1e-8))
sm <- summary(fit)

result <- list(
  coefficients = list(
    a = as.numeric(coef(fit)["a"]),
    b0 = as.numeric(coef(fit)["b1"]),
    b1 = as.numeric(coef(fit)["b2"])
  ),
  standard_errors = list(
    a = as.numeric(sm$coefficients["a", "Std. Error"]),
    b0 = as.numeric(sm$coefficients["b1", "Std. Error"]),
    b1 = as.numeric(sm$coefficients["b2", "Std. Error"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
