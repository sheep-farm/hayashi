# Reference implementation in R for the Jarque-Bera normality test.

library(moments)
library(jsonlite)

x <- read.csv("validation/cases/jb_simulated/data/data.csv")$x

stat <- jarque.test(x)$statistic
p <- jarque.test(x)$p.value

result <- list(
  jb_stat = as.numeric(stat),
  p_value = as.numeric(p)
)

out_dir <- "validation/cases/jb_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, digits = NA, auto_unbox = TRUE), "\n")
