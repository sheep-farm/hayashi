# Reference implementation in R for the Ridge hprice1 case.

library(glmnet)

case_dir <- "validation/cases/ridge_hprice1"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "hprice1.csv")
df <- read.csv(csv_path)

predictors <- c("llotsize", "lsqrft", "bdrms", "colonial")
X <- as.matrix(df[, predictors])
y <- df$lprice

# Ridge regression (alpha = 0) matching Hayashi's implementation.
# glmnet's lambda corresponds to alpha/n relative to sklearn's Ridge penalty.
model <- glmnet(X, y, alpha = 0, lambda = 0.1 / length(y), standardize = FALSE)

coefs <- as.numeric(coef(model))
names(coefs) <- rownames(coef(model))
names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"

std_errors <- as.numeric(rep(0.0, length(coefs)))
names(std_errors) <- names(coefs)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

# Build a JSON string without requiring jsonlite.
format_json <- function(x) {
  if (is.list(x)) {
    pairs <- vapply(names(x), function(n) {
      sprintf("\"%s\": %s", n, format_json(x[[n]]))
    }, character(1))
    sprintf("{%s}", paste(pairs, collapse = ", "))
  } else {
    format(x, digits = 17, scientific = FALSE)
  }
}

out <- format_json(result)
writeLines(out, file.path(ref_dir, "expected.json"))
cat(out)
