# Reference implementation in R for local-level Kalman filter on NYSE returns.

library(wooldridge)
library(jsonlite)

data(nyse)

case_dir <- "validation/cases/kalman_nyse"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
nyse <- nyse[, c("price", "return")]
nyse <- nyse[complete.cases(nyse), ]
write.csv(nyse, file.path(data_dir, "nyse.csv"), row.names = FALSE)

if (!requireNamespace("dlm", quietly = TRUE)) {
  install.packages("dlm", repos = "http://cran.r-project.org")
}
library(dlm)

# Local level model: y_t = mu_t + v_t, mu_t = mu_{t-1} + w_t
model <- dlmModPoly(order = 1, dV = 1, dW = 0.1)
fit <- dlmMLE(nyse$return, parm = c(0, 0), build = function(p) {
  dlmModPoly(order = 1, dV = exp(p[1]), dW = exp(p[2]))
})
result <- list(
  sigma_obs = sqrt(exp(fit$par[1])),
  sigma_state = sqrt(exp(fit$par[2]))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
