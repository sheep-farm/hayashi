# Reference implementation in R for the Elastic Net hprice1 case.

library(glmnet)
library(jsonlite)

case_dir <- "validation/cases/elasticnet_hprice1"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "hprice1.csv")
df <- read.csv(csv_path)

predictors <- c("llotsize", "lsqrft", "bdrms", "colonial")
X <- as.matrix(df[, predictors])
y <- df$lprice

# Elastic Net with alpha=0.5 and lambda=0.1 (matching the Hayashi parameterisation).
model <- glmnet(X, y, alpha = 0.5, lambda = 0.1, standardize = TRUE)

coefs <- as.numeric(coef(model))
names(coefs) <- rownames(coef(model))

std_errors <- as.numeric(rep(0.0, length(coefs)))
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/elasticnet_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
