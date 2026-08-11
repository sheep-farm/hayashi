df <- read.csv(file.path("validation", "cases", "nls_logistic_simulated", "data", "data.csv"))

model <- function(x, a, b, c) { a / (1 + exp(-b * (x - c))) }
fit <- nls(y ~ model(x, a, b, c), data = df, start = list(a = 8.0, b = 1.5, c = 2.0), control = nls.control(maxiter = 200, tol = 1e-8))
sm <- summary(fit)

result <- list(
  coefficients = list(
    a = as.numeric(coef(fit)["a"]),
    b = as.numeric(coef(fit)["b"]),
    c = as.numeric(coef(fit)["c"])
  ),
  standard_errors = list(
    a = as.numeric(sm$coefficients["a", "Std. Error"]),
    b = as.numeric(sm$coefficients["b", "Std. Error"]),
    c = as.numeric(sm$coefficients["c", "Std. Error"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
