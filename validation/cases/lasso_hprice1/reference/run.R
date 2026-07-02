# Reference implementation in R for the Lasso housing price case.

library(wooldridge)
library(glmnet)
library(jsonlite)

data(hprice1)

# Ensure the data directory exists.
data_dir <- "validation/cases/lasso_hprice1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(hprice1, file.path(data_dir, "hprice1.csv"), row.names = FALSE)

# Standardise predictors for fair comparison with glmnet/sklearn.
predictors <- c("lotsize", "sqrft", "bdrms")
X <- as.matrix(hprice1[, predictors])
X <- scale(X)
y <- hprice1$price

# Lasso with a small penalty that does not fully shrink coefficients.
model <- glmnet(X, y, alpha = 1, lambda = 100.0, standardize = FALSE)

coefs <- as.numeric(coef(model))
names(coefs) <- rownames(coef(model))

# glmnet does not provide analytical standard errors for Lasso.
std_errors <- as.numeric(rep(NA, length(coefs)))
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/lasso_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
