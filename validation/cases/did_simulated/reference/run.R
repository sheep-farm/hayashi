df <- read.csv(file.path("validation", "cases", "did_simulated", "data", "data.csv"))

mod <- lm(y ~ treat + post + treat:post, data = df)
V <- sandwich::vcovHC(mod, type = "HC0")
cf <- coef(mod)
se <- sqrt(diag(V))

result <- list(
  coefficients = list(
    const = as.numeric(cf["(Intercept)"]),
    treated = as.numeric(cf["treat"]),
    post = as.numeric(cf["post"]),
    `treated:post` = as.numeric(cf["treat:post"])
  ),
  standard_errors = list(
    const = as.numeric(se["(Intercept)"]),
    treated = as.numeric(se["treat"]),
    post = as.numeric(se["post"]),
    `treated:post` = as.numeric(se["treat:post"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
