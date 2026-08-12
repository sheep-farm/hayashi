library(jsonlite)
library(zoo)

case_dir <- "validation/cases/rolling_ols_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

window <- 20

betas <- rollapply(
  seq_len(nrow(df)),
  width = window,
  FUN = function(i) as.vector(coef(lm(y ~ x, data = df[i, , drop = FALSE]))),
  by.column = FALSE,
  align = "right"
)

last <- betas[nrow(betas), ]

result <- list(
  coefficients = list(
    const = as.numeric(last[1]),
    x = as.numeric(last[2])
  ),
  standard_errors = list(
    const = 0.0,
    x = 0.0
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
