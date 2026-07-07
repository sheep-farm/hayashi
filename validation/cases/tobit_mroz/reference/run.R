# Reference implementation in R for the Tobit hours-worked case.

library(wooldridge)
library(AER)
library(jsonlite)

data(mroz)

# Ensure the data directory exists.
data_dir <- "validation/cases/tobit_mroz/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(mroz, file.path(data_dir, "mroz.csv"), row.names = FALSE)

# Tobit: hours censored at zero.
model <- tobit(hours ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6, data = mroz, left = 0)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))
names(coefs)[names(coefs) == "(Intercept)"] <- "const"

# Remove the log(Sigma) entry that AER appends to the coefficient vector.
se_all <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(se_all) <- rownames(summary_model$coefficients)
names(se_all)[names(se_all) == "(Intercept)"] <- "const"
std_errors <- se_all[names(coefs)]

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors),
  diagnostics = list(
    log_likelihood = as.numeric(logLik(model)),
    nobs = as.integer(nobs(model)),
    censored = as.integer(sum(mroz$hours <= 0)),
    uncensored = as.integer(sum(mroz$hours > 0))
  )
)

out_dir <- "validation/cases/tobit_mroz/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, auto_unbox = TRUE))
