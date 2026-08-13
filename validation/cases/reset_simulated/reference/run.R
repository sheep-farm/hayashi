# Reference implementation in R for reset test

library(lmtest)
library(car)
library(jsonlite)

set.seed(42)
n <- 200
x <- rnorm(n)
z <- rnorm(n)
e <- arima.sim(n = n, list(ar = 0.3))
y <- 1 + 2*x + 0.5*z + as.numeric(e)
df <- data.frame(y = y, x = x, z = z)

data_dir <- "validation/cases/reset_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(df, file.path(data_dir, "data.csv"), row.names = FALSE)

m <- lm(y ~ x + z, data = df)

r <- resettest(m, power = 2)
result <- list(
  f_stat = as.numeric(r$statistic),
  p_value = as.numeric(r$p.value)
)

out_dir <- "validation/cases/reset_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, digits = NA, auto_unbox = TRUE), "\n")
