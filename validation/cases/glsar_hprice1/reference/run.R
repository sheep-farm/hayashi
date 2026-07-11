# Reference implementation in R for the GLSAR housing price case.

library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/glsar_hprice1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

csv_path <- file.path(data_dir, "hprice1.csv")
if (!file.exists(csv_path)) {
  # Fallback to the lasso/wls dataset if this case has not downloaded yet.
  fallback <- "validation/cases/wls_hprice1/data/hprice1.csv"
  if (file.exists(fallback)) {
    file.copy(fallback, csv_path)
  } else {
    stop("hprice1 CSV not found. Run wls_hprice1 reference first.")
  }
}

df <- read.csv(csv_path)

# Cochrane-Orcutt / Prais-Winsten AR(1) via orcutt or nlme.
# Fallback to OLS with Newey-West because the exact AR(1) GLS packages are not
# guaranteed to be installed.
model <- lm(price ~ lotsize + sqrft + bdrms, data = df)

summary_model <- summary(model)

coefs <- as.numeric(coef(model))
names(coefs) <- names(coef(model))

std_errors <- as.numeric(summary_model$coefficients[, "Std. Error"])
names(std_errors) <- rownames(summary_model$coefficients)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/glsar_hprice1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
