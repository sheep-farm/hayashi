case_dir <- "validation/cases/testparm_simulated"
library(jsonlite)
df <- read.csv(file.path(case_dir, "data", "data.csv"))
full <- lm(y ~ x1 + x2, data = df)
red <- lm(y ~ 1, data = df)
res <- anova(red, full)
f <- res$F[2]
pval <- res$`Pr(>F)`[2]
cat(toJSON(list(coefficients = list(f_stat = f, p_value = pval), standard_errors = list(f_stat = 0, p_value = 0)), auto_unbox = TRUE, digits = 15))