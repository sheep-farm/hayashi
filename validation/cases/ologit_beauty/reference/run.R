# Reference implementation in R for the ordered logit beauty case.

library(MASS)

case_dir <- "validation/cases/ologit_beauty"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")

csv_path <- file.path(data_dir, "beauty.csv")

# Read the CSV prepared by the Python reference.
df <- read.csv(csv_path)

# Ordered logit regression of looks on female, educ, exper, black.
model <- polr(
  factor(looks) ~ female + educ + exper + black,
  data = df,
  method = "logistic",
  Hess = TRUE
)

summary_model <- summary(model)

# Extract only the regression coefficients, not the thresholds/cuts.
reg_vars <- c("female", "educ", "exper", "black")
coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- rownames(summary_model$coefficients)

result <- list(
  coefficients = as.list(coefs[reg_vars]),
  standard_errors = as.list(std_errors[reg_vars])
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

dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(out, file.path(ref_dir, "expected.json"))

cat(out)
