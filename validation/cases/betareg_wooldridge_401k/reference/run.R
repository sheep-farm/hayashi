# Reference implementation in R for beta regression on Wooldridge 401k.

library(wooldridge)
library(jsonlite)

data(k401k)
df <- k401k
df$prate <- df$prate / 100.0
df$prate <- pmin(pmax(df$prate, 1e-4), 1 - 1e-4)

case_dir <- "validation/cases/betareg_wooldridge_401k"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(df, file.path(data_dir, "401k.csv"), row.names = FALSE)

if (!requireNamespace("betareg", quietly = TRUE)) {
  install.packages("betareg", repos = "http://cran.r-project.org")
}
library(betareg)

model <- betareg(prate ~ mrate + age + sole, data = df)
sm <- summary(model)

coefs <- coef(model)
names <- c("(Intercept)", "mrate", "age", "sole")
result <- list(
  coefficients = as.list(setNames(as.numeric(coefs[names]), c("const", "mrate", "age", "sole"))),
  standard_errors = as.list(setNames(as.numeric(sm$coefficients$mean[names, "Std. Error"]), c("const", "mrate", "age", "sole")))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
