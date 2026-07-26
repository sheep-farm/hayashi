# Reference implementation in R for the Wooldridge wage1 one-sample t-test case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/ttest_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

x <- wage1$wage
n <- length(x)
mu0 <- 5.0
mean_x <- mean(x)
sd_x <- sd(x)
std_err <- sd_x / sqrt(n)
t_stat <- (mean_x - mu0) / std_err
df <- n - 1
p_value <- 2 * (1 - pt(abs(t_stat), df))
t_crit <- qt(0.975, df)

result <- list(
  test = "one-sample t-test",
  variable = "wage",
  n = n,
  mean = mean_x,
  mu = mu0,
  std_err = std_err,
  t_stat = t_stat,
  df = df,
  p_value = p_value,
  ci_lower = mean_x - t_crit * std_err,
  ci_upper = mean_x + t_crit * std_err
)

out_dir <- "validation/cases/ttest_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
