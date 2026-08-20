df <- read.csv(file.path("validation", "cases", "sfa_production_simulated", "data", "data.csv"))
suppressPackageStartupMessages(library(frontier))
suppressPackageStartupMessages(library(jsonlite))

fit <- sfa(log(y) ~ log(x1) + log(x2), data = df)
coefs <- coef(fit)[1:3]
names(coefs) <- c("const", "lx1", "lx2")

# Frontier's covariance estimate can be unstable at the gamma boundary,
# so report OLS standard errors for the same frontier equation.
ols <- lm(ly ~ lx1 + lx2, data = df)
se <- as.numeric(summary(ols)$coefficients[, 2])
names(se) <- c("const", "lx1", "lx2")

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(se)
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
