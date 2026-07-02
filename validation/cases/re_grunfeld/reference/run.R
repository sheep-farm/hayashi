# Reference implementation in R for the random-effects Grunfeld case.

library(plm)
library(jsonlite)

# Load Grunfeld data from the plm package (same as statsmodels/wooldridge).
data(Grunfeld)

# Ensure the data directory exists.
data_dir <- "validation/cases/re_grunfeld/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# The plm Grunfeld data already has firm and year.
# Write CSV for Hayashi to read.
write.csv(Grunfeld, file.path(data_dir, "grunfeld.csv"), row.names = FALSE)

# Random-effects investment demand model.
model <- plm(inv ~ value + capital, data = Grunfeld, index = c("firm", "year"), model = "random")

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/re_grunfeld/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
