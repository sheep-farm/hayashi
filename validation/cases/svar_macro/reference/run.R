# Reference implementation in R for SVAR on US macro data.

library(jsonlite)

case_dir <- "validation/cases/svar_macro"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(file.path(data_dir, "macrodata.csv"))

if (!requireNamespace("vars", quietly = TRUE)) {
  install.packages("vars", repos = "http://cran.r-project.org")
}
library(vars)

# Use log levels (same as VAR macro case)
df <- df[, c("gdp", "cons")]
var_model <- VAR(df, p = 2, type = "const")
# Cholesky identification: A = identity, B is lower-triangular Cholesky of residual covariance
Amat <- diag(2)
resid <- residuals(var_model)
T <- nrow(resid)
k <- ncol(resid)
p <- 2
Sigma <- t(resid) %*% resid / (T - (1 + k * p))
B <- t(chol(Sigma))
A <- diag(2)

result <- list(
  a_matrix = as.list(setNames(as.numeric(A), paste0("a", 0:3))),
  b_matrix = as.list(setNames(as.numeric(B), paste0("b", 0:3)))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
