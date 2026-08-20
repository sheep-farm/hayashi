options(warn=-1)
library(jsonlite)

case_dir <- "validation/cases/bayes_lm_simulated"
data_dir <- file.path(case_dir, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

mod <- lm(y ~ x1 + x2, data = df)
s <- summary(mod)$coefficients

result <- list(
  coefficients = list(
    b1 = as.numeric(s["x1", "Estimate"]),
    b2 = as.numeric(s["x2", "Estimate"])
  ),
  standard_errors = list(
    b1 = as.numeric(s["x1", "Std. Error"]),
    b2 = as.numeric(s["x2", "Std. Error"])
  )
)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
