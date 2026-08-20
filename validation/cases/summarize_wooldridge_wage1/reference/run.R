# Reference implementation in R for the Wooldridge wage1 summary statistics case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/summarize_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

x <- wage1$wage
n <- length(x)
mean_x <- mean(x)
sd_x <- sd(x)
min_x <- min(x)
max_x <- max(x)

# Quantiles using type 2 to match Hayashi's nearest-rank rounding method.
q <- quantile(x, probs = c(0.25, 0.50, 0.75), type = 2)
p25 <- as.numeric(q[1])
p50 <- as.numeric(q[2])
p75 <- as.numeric(q[3])

# Skewness and kurtosis use the same moment formulas as Hayashi.
skew <- sum(((x - mean_x) / sd_x)^3) * n / ((n - 1) * (n - 2))
kurt <- mean(((x - mean_x) / sd_x)^4)

result <- list(
  N = n,
  mean = mean_x,
  sd = sd_x,
  min = min_x,
  max = max_x,
  p25 = p25,
  p50 = p50,
  p75 = p75,
  skewness = skew,
  kurtosis = kurt
)

out_dir <- "validation/cases/summarize_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)

# Also emit JSON on stdout so the orchestrator can avoid reading files.
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
