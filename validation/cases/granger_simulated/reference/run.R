# Reference implementation in R for granger_simulated

library(lmtest)
library(jsonlite)

set.seed(42)
n <- 200
e <- rnorm(n)
y <- arima.sim(n = n, list(ar = 0.5), innov = e)
x <- arima.sim(n = n, list(ar = 0.5), innov = e + rnorm(n))

data_dir <- "validation/cases/granger_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(y = as.numeric(y), x = as.numeric(x)), file.path(data_dir, "data.csv"), row.names = FALSE)

r <- grangertest(as.numeric(y), as.numeric(x), order = 4)
F <- r$F[2]
p <- r$Pr[2]
result <- list(
  fit = list(
    statistic = as.numeric(F),
    p_value = as.numeric(p)
  )
)

out_dir <- "validation/cases/granger_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, digits = NA, auto_unbox = TRUE), "\n")
