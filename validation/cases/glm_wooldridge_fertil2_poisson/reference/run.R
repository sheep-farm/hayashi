# Reference implementation in R for Wooldridge fertil2 Poisson GLM.

library(wooldridge)
library(jsonlite)

data(fertil2)

# Ensure the data directory exists.
data_dir <- "validation/cases/glm_wooldridge_fertil2_poisson/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Keep only observations with complete model variables.
fertil2_clean <- fertil2[complete.cases(fertil2[, c("children", "age", "electric", "educ", "urban", "tv")]), ]

# Write CSV for Hayashi to read.
write.csv(fertil2_clean, file.path(data_dir, "fertil2.csv"), row.names = FALSE)

# Estimate Poisson GLM.
model <- glm(children ~ age + electric + educ + urban + tv,
             data = fertil2_clean,
             family = poisson(link = "log"))

# Extract comparable quantities.
summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors),
  nobs = nrow(fertil2_clean)
)

out_dir <- "validation/cases/glm_wooldridge_fertil2_poisson/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
