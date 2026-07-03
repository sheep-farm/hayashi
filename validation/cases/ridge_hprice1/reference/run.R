# Reference implementation in R for the Ridge hprice1 case.

library(MASS)

case_dir <- "validation/cases/ridge_hprice1"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "hprice1.csv")
df <- read.csv(csv_path)

# Ridge regression via MASS::lm.ridge (standardises X, intercept not penalised).
model <- lm.ridge(lprice ~ llotsize + lsqrft + bdrms + colonial, data = df, lambda = 0.1)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

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
