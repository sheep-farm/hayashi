df <- read.csv("validation/cases/panel_tobit_simulated/data/data.csv")

# Pooled Tobit as reference (coefficients should be close to panel RE Tobit)
fit <- censReg::censReg(y ~ x, data = df, left = 0)
sm <- summary(fit)

coef <- as.numeric(sm$estimate[, 1])
se <- as.numeric(sm$estimate[, 2])

result <- list(
  coefficients = list(
    "_cons" = coef[1],
    "x" = coef[2]
  ),
  standard_errors = list(
    "_cons" = se[1],
    "x" = se[2]
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"), "\n")
