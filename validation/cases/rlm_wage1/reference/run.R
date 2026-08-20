# Reference implementation in R for robust linear model on Wooldridge wage1.

library(wooldridge)
library(MASS)
library(jsonlite)

data(wage1)

case_dir <- "validation/cases/rlm_wage1"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

model <- rlm(lwage ~ educ + exper + tenure, data = wage1)
sm <- summary(model)

coefs <- coef(model)
se <- sm$coefficients[, "Std. Error"]
names <- c("(Intercept)", "educ", "exper", "tenure")

result <- list(
  coefficients = as.list(setNames(as.numeric(coefs[names]), c("const", "educ", "exper", "tenure"))),
  standard_errors = as.list(setNames(as.numeric(se[names]), c("const", "educ", "exper", "tenure")))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
