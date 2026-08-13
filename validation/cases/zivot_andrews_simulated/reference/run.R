# Reference implementation in R for zivot_andrews_simulated

library(urca)
library(jsonlite)

set.seed(42)
n <- 200
x <- cumsum(rnorm(n))

data_dir <- "validation/cases/zivot_andrews_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(x = x), file.path(data_dir, "data.csv"), row.names = FALSE)

r <- ur.za(x)
result <- list(
  fit = list(
    statistic = as.numeric(r@teststat[1])
  )
)

out_dir <- "validation/cases/zivot_andrews_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, digits = NA, auto_unbox = TRUE), "\n")
