# Reference implementation in R for harveycollier_simulated

library(lmtest)
library(jsonlite)

set.seed(42)
n <- 200
x <- rnorm(n)
z <- rnorm(n)
y <- 1 + 2*x + 0.5*z + rnorm(n)
df <- data.frame(y = y, x = x, z = z)

data_dir <- "validation/cases/harveycollier_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(df, file.path(data_dir, "data.csv"), row.names = FALSE)

m <- lm(y ~ x + z, data = df)
r <- harvtest(m)
result <- list(
  fit = list(
    statistic = as.numeric(r$statistic),
    p_value = as.numeric(r$p.value)
  )
)

out_dir <- "validation/cases/harveycollier_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, digits = NA, auto_unbox = TRUE), "\n")
