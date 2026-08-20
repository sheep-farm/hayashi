options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/kde_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))
x <- df$x

fit <- density(x, bw = 0.5, kernel = "gaussian", n = 512)
peak_idx <- which.max(fit$y)

result <- list(
  coefficients = list(
    bandwidth = 0.5,
    peak_density = as.numeric(fit$y[peak_idx]),
    peak_x = as.numeric(fit$x[peak_idx])
  ),
  standard_errors = list(
    bandwidth = 0,
    peak_density = 0,
    peak_x = 0
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
