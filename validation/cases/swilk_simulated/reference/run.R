# Reference implementation in R for swilk_simulated

library(jsonlite)

set.seed(42)
n <- 200
x <- rnorm(n, mean = 5, sd = 2)

data_dir <- "validation/cases/swilk_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(x = x), file.path(data_dir, "data.csv"), row.names = FALSE)

r <- shapiro.test(x)
result <- list(
  fit = list(
    statistic = as.numeric(r$statistic),
    p_value = as.numeric(r$p.value)
  )
)

out_dir <- "validation/cases/swilk_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, auto_unbox = TRUE, digits = 15))
