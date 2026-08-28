df <- read.csv(file.path("validation", "cases", "fmols_simulated", "data", "data.csv"))
suppressPackageStartupMessages(library(cointReg))
suppressPackageStartupMessages(library(jsonlite))

x <- as.matrix(df[, "x", drop = FALSE])
y <- as.numeric(df$y)

deter <- cbind(level = rep(1, length(y)))
fit <- cointRegFM(x = x, y = y, deter = deter, kernel = "ba", bandwidth = "and")

result <- list(
  coefficients = list(x = as.numeric(fit$theta[2])),
  standard_errors = list(x = as.numeric(fit$sd.theta[2]))
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
