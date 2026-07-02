# Reference implementation in R for the Poisson fertility case.

library(wooldridge)
library(jsonlite)

data(fertil2)

# Ensure the data directory exists.
data_dir <- "validation/cases/poisson_fertil2/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(fertil2, file.path(data_dir, "fertil2.csv"), row.names = FALSE)

# Poisson regression for number of children.
model <- glm(children ~ educ + age + agesq + evermarr + urban + electric + tv, data = fertil2, family = poisson)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/poisson_fertil2/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
