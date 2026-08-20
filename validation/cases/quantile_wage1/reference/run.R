# Reference implementation in R for the quantile regression wage1 case.

library(wooldridge)
library(quantreg)
library(jsonlite)

data(wage1)

# Ensure the data directory exists.
data_dir <- "validation/cases/quantile_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

# Quantile regression at the median.
model <- rq(wage ~ educ + exper + tenure, data = wage1, tau = 0.5)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

# summary.rq reports coefficients, lower bd and upper bd. Approximate SE as
# half the width of the 95% confidence interval divided by the normal quantile.
lower <- summary_model$coefficients[, "lower bd"]
upper <- summary_model$coefficients[, "upper bd"]
std_errors <- as.numeric((upper - lower) / (2 * qnorm(0.975)))
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/quantile_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
