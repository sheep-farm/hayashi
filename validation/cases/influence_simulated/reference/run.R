case_dir <- "validation/cases/influence_simulated"
library(jsonlite)
df <- read.csv(file.path(case_dir, "data", "data.csv"))
fit <- lm(y ~ x, data = df)
d <- unname(dffits(fit))
maxd <- max(abs(d))
cat(toJSON(list(coefficients = list(max_dffits = maxd), standard_errors = list(max_dffits = 0)), auto_unbox = TRUE, digits = 15))