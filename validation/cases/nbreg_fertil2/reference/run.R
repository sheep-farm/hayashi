# Reference implementation in R for the Negative Binomial fertil2 case.

library(MASS)

case_dir <- "validation/cases/nbreg_fertil2"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "fertil2.csv")
df <- read.csv(csv_path)

# Negative binomial regression for number of children.
model <- glm.nb(children ~ age + educ + electric + urban, data = df)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

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
