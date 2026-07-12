# Reference implementation in R for system GMM on Wooldridge wagepan.

library(wooldridge)
library(jsonlite)

data(wagepan)

case_dir <- "validation/cases/sysgmm_wagepan"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

# Keep balanced subset and create lagged lwage
wagepan <- wagepan[order(wagepan$nr, wagepan$year), ]
wagepan$lwage_lag <- ave(wagepan$lwage, wagepan$nr, FUN = function(x) c(NA, head(x, -1)))
wagepan <- wagepan[complete.cases(wagepan[, c("lwage", "lwage_lag", "exper", "expersq")]), ]
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

if (!requireNamespace("plm", quietly = TRUE)) {
  install.packages("plm", repos = "http://cran.r-project.org")
}
library(plm)

pdata <- pdata.frame(wagepan, index = c("nr", "year"))
model <- pgmm(lwage ~ lag(lwage, 1) + exper + expersq + married + union | lag(lwage, 2:3),
              data = pdata, effect = "individual", model = "twosteps", transformation = "ld")
sm <- summary(model)

coefs <- coef(model)
se <- sm$coefficients[, "Std. Error"]

vars <- c("lag(lwage, 1)", "exper", "expersq", "married", "union")
names <- c("lwage_lag", "exper", "expersq", "married", "union")

result <- list(
  coefficients = as.list(setNames(as.numeric(coefs[vars]), names)),
  standard_errors = as.list(setNames(as.numeric(se[vars]), names))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
