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

# Use raw predictors so coefficients are on the original scale.
predictors <- c("lotsize", "sqrft", "bdrms")
X <- as.matrix(hprice1[, predictors])
y <- hprice1$price

# Lasso with a small penalty that does not fully shrink coefficients.
model <- glmnet(X, y, alpha = 1, lambda = 1.0, standardize = TRUE)

coefs <- as.numeric(coef(model))
names(coefs) <- rownames(coef(model))
names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"

# glmnet does not provide analytical standard errors for Lasso.
std_errors <- as.numeric(rep(0.0, length(coefs)))
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/lasso_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
