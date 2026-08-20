library(geepack)

df <- read.csv(file.path("validation", "cases", "xtpoisson_simulated", "data", "data.csv"))
df$id <- as.factor(df$id)

fit <- geeglm(y ~ x, id = id, data = df, family = poisson, corstr = "exchangeable")
sm <- summary(fit)

result <- list(
  coefficients = list(
    const = as.numeric(coef(fit)["(Intercept)"]),
    x = as.numeric(coef(fit)["x"])
  ),
  standard_errors = list(
    const = as.numeric(sm$coefficients["(Intercept)", "Std.err"]),
    x = as.numeric(sm$coefficients["x", "Std.err"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
