library(glmnet)
library(jsonlite)

case_dir <- "validation/cases/reg_path_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))

X <- as.matrix(df[, c("x1", "x2")])
y <- df$y

# Match Hayashi reg_path: elastic net with alpha=0.5 and BIC-selected lambda.
fit <- glmnet(X, y, alpha = 0.5, lambda = 0.003642, standardize = TRUE)
coefs <- as.numeric(coef(fit))
names <- rownames(coef(fit))
names(coefs) <- names

result <- list(
  coefficients = list(
    const = as.numeric(coefs["(Intercept)"]),
    x1 = as.numeric(coefs["x1"]),
    x2 = as.numeric(coefs["x2"])
  ),
  standard_errors = list(
    const = 0.0,
    x1 = 0.0,
    x2 = 0.0
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
