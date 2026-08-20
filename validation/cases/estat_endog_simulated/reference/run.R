case_dir <- "validation/cases/estat_endog_simulated"
library(jsonlite)
df <- read.csv(file.path(case_dir, "data", "data.csv"))
# first stage residuals for x
fs <- lm(x ~ z, data = df)
vhat <- residuals(fs)
df$vhat <- vhat
fit <- lm(y ~ x + vhat, data = df)
# F for vhat = 0 is t^2
s <- summary(fit)$coefficients
f <- s["vhat", "t value"]^2
df2 <- df.residual(fit)
pval <- 1 - pf(f, 1, df2)
cat(toJSON(list(coefficients = list(f_stat = f, p_value = pval), standard_errors = list(f_stat = 0, p_value = 0)), auto_unbox = TRUE, digits = 15))