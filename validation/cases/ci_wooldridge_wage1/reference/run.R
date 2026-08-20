# Reference implementation in R for the Wooldridge wage1 confidence interval case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/ci_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

x <- wage1$wage
n <- length(x)
mean_x <- mean(x)
sd_x <- sd(x)
std_err <- sd_x / sqrt(n)
t_crit <- qt(0.975, n - 1)

result <- list(
  variable = "wage",
  n = n,
  mean = mean_x,
  sd = sd_x,
  std_err = std_err,
  ci_lower = mean_x - t_crit * std_err,
  ci_upper = mean_x + t_crit * std_err,
  level = 0.95
)

out_dir <- "validation/cases/ci_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
