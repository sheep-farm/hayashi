# Reference implementation in R for the DiD Kiel-McClain case.

library(wooldridge)
library(jsonlite)

data(kielmc)

# Ensure the data directory exists.
data_dir <- "validation/cases/did_kielmc/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(kielmc, file.path(data_dir, "kielmc.csv"), row.names = FALSE)

# Difference-in-differences via interaction.
model <- lm(lprice ~ nearinc * y81, data = kielmc)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- rownames(summary_model$coefficients)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/did_kielmc/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
