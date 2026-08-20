library(boot)
library(jsonlite)

case_dir <- "validation/cases/bootstrap_ols_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

set.seed(42)

boot_stat <- function(d, i) {
  fit <- lm(y ~ x, data = d[i, , drop = FALSE])
  as.vector(coef(fit))
}

b <- boot(data = df, statistic = boot_stat, R = 2000, stype = "i")

means <- colMeans(b$t)
sds <- apply(b$t, 2, sd)

result <- list(
  coefficients = list(
    const = as.numeric(means[1]),
    x = as.numeric(means[2])
  ),
  standard_errors = list(
    const = as.numeric(sds[1]),
    x = as.numeric(sds[2])
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
