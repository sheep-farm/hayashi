# Reference implementation in R for the deterministic Fama-MacBeth validation
# case.

library(jsonlite)

data_dir <- "validation/cases/fmb_simulated/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

rows <- list()
idx <- 1
for (period in 1:8) {
  for (firm in 1:12) {
    beta <- 0.55 +
      0.08 * firm +
      0.012 * period +
      0.015 * ((firm + 2 * period) %% 4)
    size <- 7.0 +
      0.35 * ((firm * firm) %% 13) +
      0.05 * firm +
      0.10 * period +
      0.02 * ((firm + period) %% 5)
    eps <- 0.006 * (((firm * 3 + period * 2) %% 7) - 3)
    alpha <- 0.015 + 0.0015 * period
    beta_slope <- 0.040 + 0.0025 * ((period %% 4) - 1.5)
    size_slope <- 0.0045 + 0.0004 * ((period %% 3) - 1)
    ret <- alpha + beta_slope * beta + size_slope * size + eps
    rows[[idx]] <- data.frame(
      ret = ret,
      beta = beta,
      size = size,
      firm = firm,
      period = period
    )
    idx <- idx + 1
  }
}

df <- do.call(rbind, rows)
write.csv(df, file.path(data_dir, "fmb_panel.csv"), row.names = FALSE)

periods <- sort(unique(df$period))
period_coefs <- lapply(periods, function(p) {
  coef(lm(ret ~ beta + size, data = df[df$period == p, ]))
})
coef_mat <- do.call(rbind, period_coefs)

coefs <- colMeans(coef_mat)
std_errors <- apply(coef_mat, 2, sd) / sqrt(nrow(coef_mat))

names(coefs)[names(coefs) == "(Intercept)"] <- "Intercept"
names(std_errors)[names(std_errors) == "(Intercept)"] <- "Intercept"

result <- list(
  coefficients = as.list(coefs),
  standard_errors = as.list(std_errors)
)

out_dir <- "validation/cases/fmb_simulated/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(
  result,
  file.path(out_dir, "expected.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 16
)

cat(toJSON(result, auto_unbox = TRUE, digits = 16))
