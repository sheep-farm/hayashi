# Reference implementation in R for PCSE on Wooldridge wagepan.

library(wooldridge)
library(jsonlite)
library(plm)

data(wagepan)

case_dir <- "validation/cases/pcse_wagepan"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

pdata <- pdata.frame(wagepan, index = c("nr", "year"))
model <- plm(lwage ~ educ + exper + expersq + married + union, data = pdata, model = "pooling")

# Panel-corrected standard errors (Beck-Katz), matching the
# Hayashi/Greeners balanced-panel convention validated by the Python reference.
vcv <- vcovBK(model, cluster = "time", type = "HC0")
coefs <- coef(model)
se <- sqrt(diag(vcv))

vars <- c("(Intercept)", "educ", "exper", "expersq", "married", "union")
names <- c("const", "educ", "exper", "expersq", "married", "union")

result <- list(
  coefficients = as.list(setNames(as.numeric(coefs[vars]), names)),
  standard_errors = as.list(setNames(as.numeric(se[vars]), names))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
