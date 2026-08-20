library(jsonlite)

library(grf)

set.seed(42)

case_dir <- "validation/cases/orf_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

# Use x features and confounders together as the covariate set.
X <- df[, c("x1", "x2", "w1", "w2")]
Y <- df$y
W <- df$treated

model <- causal_forest(X, Y, W, num.trees = 200, seed = 42)
ate <- average_treatment_effect(model, target.sample = "all")

result <- list(
  coefficients = list(ate = as.numeric(ate["estimate"])),
  standard_errors = list(ate = as.numeric(ate["std.err"]))
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
