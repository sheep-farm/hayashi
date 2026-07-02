# Reference implementation in R for the VAR macro case.

library(vars)
library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/var_macro/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from statsmodels mirror (Rdatasets).
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- read.csv(url)
macro <- macro[, c("year", "quarter", "realgdp", "realcons")]
names(macro)[names(macro) == "realgdp"] <- "gdp"
names(macro)[names(macro) == "realcons"] <- "cons"

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

# VAR(2) on GDP and consumption.
macro <- macro[complete.cases(macro[, c("gdp", "cons")]), ]
model <- VAR(macro[, c("gdp", "cons")], p = 2, type = "const")

# Extract coefficients and standard errors per equation.
all_coefs <- list()
all_ses <- list()
for (eq in names(coef(model))) {
  mat <- coef(model)[[eq]]
  for (i in seq_len(nrow(mat))) {
    name <- rownames(mat)[i]
    all_coefs[[paste0(eq, "_", name)]] <- as.numeric(mat[i, "Estimate"])
    all_ses[[paste0(eq, "_", name)]] <- as.numeric(mat[i, "Std. Error"])
  }
}

coefs <- as.numeric(unlist(all_coefs))
names(coefs) <- names(all_coefs)

std_errors <- as.numeric(unlist(all_ses))
names(std_errors) <- names(all_ses)

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/var_macro/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
