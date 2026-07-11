# Reference implementation in R for xtgls on Wooldridge wagepan.

library(wooldridge)
library(jsonlite)

data(wagepan)

case_dir <- "validation/cases/xtgls_wagepan"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

if (!requireNamespace("plm", quietly = TRUE)) {
  install.packages("plm", repos = "http://cran.r-project.org")
}
library(plm)

pdata <- pdata.frame(wagepan, index = c("nr", "year"))
model <- plm(lwage ~ educ + exper + expersq + married + union, data = pdata,
             model = "pooling")
# FGLS with heteroskedastic panels via pggls (Parks-Kmenta)
if (!requireNamespace("pggls", quietly = TRUE)) {
  install.packages("pggls", repos = "http://cran.r-project.org")
}

# Use plm GLS with heteroskedastic error
model_fgls <- pggls(lwage ~ educ + exper + expersq + married + union, data = pdata,
                    model = "pooling", effect = "individual")
coefs <- coef(model_fgls)
se <- summary(model_fgls)$coefficients[, "Std. Error"]

vars <- c("(Intercept)", "educ", "exper", "expersq", "married", "union")
names <- c("const", "educ", "exper", "expersq", "married", "union")

result <- list(
  coefficients = as.list(setNames(as.numeric(coefs[vars]), names)),
  standard_errors = as.list(setNames(as.numeric(se[vars]), names))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
