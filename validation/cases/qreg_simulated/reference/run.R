df <- read.csv(file.path("validation", "cases", "qreg_simulated", "data", "data.csv"))

fit <- quantreg::rq(y ~ x, tau = 0.75, data = df)
sm <- summary(fit, se = "boot", R = 100)
cf <- sm$coefficients

result <- list(
  coefficients = list(
    const = as.numeric(cf["(Intercept)", "Value"]),
    x = as.numeric(cf["x", "Value"])
  ),
  standard_errors = list(
    const = as.numeric(cf["(Intercept)", "Std. Error"]),
    x = as.numeric(cf["x", "Std. Error"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
