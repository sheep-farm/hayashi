# Reference implementation in R for the Wooldridge wage1 codebook case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/codebook_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write a one-column CSV for Hayashi to read.
wage_df <- data.frame(wage = wage1$wage)
write.csv(wage_df, file.path(data_dir, "wage.csv"), row.names = FALSE)

x <- wage1$wage
n <- length(x)

# Quantiles using type 2 to match Hayashi.
q <- quantile(x, probs = c(0.25, 0.50, 0.75), type = 2)

result <- list(
  variable = list("wage"),
  type = list("float"),
  obs = list(n),
  missing = list(sum(is.na(x))),
  unique = list(length(unique(x))),
  mean = list(mean(x)),
  sd = list(sd(x)),
  min = list(min(x)),
  p25 = list(as.numeric(q[1])),
  p50 = list(as.numeric(q[2])),
  p75 = list(as.numeric(q[3])),
  max = list(max(x))
)

out_dir <- "validation/cases/codebook_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
