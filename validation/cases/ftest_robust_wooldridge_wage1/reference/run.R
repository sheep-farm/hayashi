# Reference implementation in R for robust F-test.

library(wooldridge)
library(jsonlite)

data(wage1)

# Ensure the data directory exists.
data_dir <- "validation/cases/ftest_robust_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

# Estimate OLS model.
model <- lm(wage ~ educ + exper + tenure, data = wage1)

# Extract coefficients and standard errors for the tested variables.
coefs <- coef(model)
ses <- summary(model)$coefficients[, "Std. Error"]
names <- names(coefs)

# Test joint significance of exper and tenure (indices 3 and 4 with intercept as 1).
idx <- which(names %in% c("exper", "tenure"))
q <- length(idx)
p <- length(coefs)
n <- nrow(wage1)

beta_r <- coefs[idx]
vcov_r <- diag(ses[idx]^2)

wald <- as.numeric(t(beta_r) %*% solve(vcov_r) %*% beta_r)
f_stat <- wald / q
df_num <- q
df_denom <- n - p
p_value <- 1 - pf(f_stat, df_num, df_denom)

result <- list(
  test_statistic = f_stat,
  p_value = p_value,
  degrees_of_freedom_num = df_num,
  degrees_of_freedom_denom = df_denom
)

out_dir <- "validation/cases/ftest_robust_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
