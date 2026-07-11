# Reference implementation in R for the panel FE Grunfeld case.

library(plm)
library(jsonlite)

data(Grunfeld, package = "plm")
grunfeld <- Grunfeld

# Ensure the data directory exists.
data_dir <- "validation/cases/panel_fe_grunfeld/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(grunfeld, file.path(data_dir, "grunfeld.csv"), row.names = FALSE)

# Fixed-effects investment demand model.
model <- plm(inv ~ value + capital, data = grunfeld, index = c("firm", "year"), model = "within")

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- names(coef(model))

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/panel_fe_grunfeld/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
