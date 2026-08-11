df <- read.csv(file.path("validation", "cases", "panel_qreg_simulated", "data", "data.csv"))

# Within transformation
df <- within(df, {
  y_dem <- ave(y, id, FUN = function(z) z - mean(z))
  x_dem <- ave(x, id, FUN = function(z) z - mean(z))
})

fit <- quantreg::rq(y_dem ~ x_dem - 1, tau = 0.75, data = df)
sm <- summary(fit, se = "ker")
cf <- sm$coefficients

result <- list(
  coefficients = list(
    x = as.numeric(cf["x_dem", "Value"])
  ),
  standard_errors = list(
    x = as.numeric(cf["x_dem", "Std. Error"])
  )
)

cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
