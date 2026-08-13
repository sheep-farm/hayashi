# Reference implementation in R for ljungbox_simulated

library(jsonlite)

set.seed(42)
n <- 200
x <- arima.sim(n = n, list(ar = 0.5))

data_dir <- "validation/cases/ljungbox_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(x = as.numeric(x)), file.path(data_dir, "data.csv"), row.names = FALSE)

r <- Box.test(as.numeric(x), lag = 10, type = "Ljung-Box")
result <- list(
  fit = list(
    q_stat = as.numeric(r$statistic),
    p_value = as.numeric(r$p.value)
  )
)

out_dir <- "validation/cases/ljungbox_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
