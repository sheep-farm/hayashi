# Reference implementation in R for the Jarque-Bera normality test.

library(moments)
library(jsonlite)

set.seed(42)
n <- 200
x <- rnorm(n, mean = 5, sd = 2)

data_dir <- "validation/cases/jb_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(x = x), file.path(data_dir, "data.csv"), row.names = FALSE)

stat <- jarque.test(x)$statistic
p <- jarque.test(x)$p.value

result <- list(
  jb_stat = as.numeric(stat),
  p_value = as.numeric(p)
)

out_dir <- "validation/cases/jb_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
