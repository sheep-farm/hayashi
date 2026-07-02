# Reference implementation in R for the logit labour-force participation case.

library(wooldridge)
library(jsonlite)

data(mroz)

# Ensure the data directory exists.
data_dir <- "validation/cases/logit_mroz/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(mroz, file.path(data_dir, "mroz.csv"), row.names = FALSE)

# Estimate labour-force participation logit.
model <- glm(inlf ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6, data = mroz, family = binomial(link = "logit"))

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/logit_mroz/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
