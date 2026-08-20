library(survival)

case_id <- "clogit_simulated"
data_dir <- file.path("validation", "cases", case_id, "data")
df <- read.csv(file.path(data_dir, "data.csv"))

fit <- clogit(y ~ x + strata(group), data = df)
coef <- as.numeric(coef(fit))
se <- as.numeric(summary(fit)$coef[, "se(coef)"])

out <- paste0(
  "{",
  '"coefficients":{"x":', coef, '},',
  '"standard_errors":{"x":', se, '}',
  "}"
)
cat(out, "\n")
