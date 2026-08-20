case_dir <- "validation/cases/weak_iv_simulated"
library(jsonlite)
df <- read.csv(file.path(case_dir, "data", "data.csv"))
fit <- lm(x ~ z, data = df)
# anova against intercept-only
red <- lm(x ~ 1, data = df)
res <- anova(red, fit)
f <- res$F[2]
pval <- res$`Pr(>F)`[2]
if (is.na(f)) { f <- summary(fit)$fstatistic[1]; pval <- 1 - pf(f, 1, df.residual(fit)) }
cat(toJSON(list(coefficients = list(first_stage_f = f, first_stage_p = pval), standard_errors = list(first_stage_f = 0, first_stage_p = 0)), auto_unbox = TRUE, digits = 15))