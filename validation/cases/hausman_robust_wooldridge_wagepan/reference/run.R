# Reference implementation in R for robust Hausman test.

library(wooldridge)
library(plm)
library(jsonlite)

data(wagepan)

# Ensure the data directory exists.
data_dir <- "validation/cases/hausman_robust_wooldridge_wagepan/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Write CSV for Hayashi to read.
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

# Estimate FE and RE models on common time-varying variables.
fe_model <- plm(lwage ~ exper + expersq + married + union,
                data = wagepan, model = "within", index = c("nr", "year"))
re_model <- plm(lwage ~ exper + expersq + married + union,
                data = wagepan, model = "random", index = c("nr", "year"))

beta_fe <- coef(fe_model)
beta_re <- coef(re_model)

# Common coefficients (FE drops time-invariant variables).
common_names <- intersect(names(beta_fe), names(beta_re))
beta_fe_aligned <- beta_fe[common_names]
beta_re_aligned <- beta_re[common_names]

# Use diagonal variance approximation to match Hayashi's implementation.
se_fe <- summary(fe_model)$coefficients[, "Std. Error"][common_names]
se_re <- summary(re_model)$coefficients[, "Std. Error"][common_names]

diff <- beta_fe_aligned - beta_re_aligned
var_diff <- se_fe^2 - se_re^2
chi2 <- sum((diff^2) / var_diff)
df <- length(diff)
p_value <- 1 - pchisq(chi2, df)

result <- list(
  test_statistic = chi2,
  p_value = p_value,
  degrees_of_freedom = df
)

out_dir <- "validation/cases/hausman_robust_wooldridge_wagepan/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
