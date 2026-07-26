# Reference implementation in R for the Wooldridge wage1 tabstat case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/tabstat_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

vars <- c("wage", "educ", "exper", "tenure")
stats <- c("mean", "sd", "min", "max", "p50")

variable <- character(0)
stat <- character(0)
value <- numeric(0)

for (v in vars) {
  x <- wage1[[v]]
  x <- x[!is.na(x)]
  n <- length(x)
  for (s in stats) {
    variable <- c(variable, v)
    stat <- c(stat, s)
    val <- switch(s,
      mean = mean(x),
      sd = sd(x),
      min = min(x),
      max = max(x),
      p50 = sort(x)[floor(0.50 * (length(x) - 1) + 0.5) + 1]
    )
    value <- c(value, val)
  }
}

result <- list(
  variable = variable,
  stat = stat,
  value = value,
  group = rep("all", length(variable))
)

out_dir <- "validation/cases/tabstat_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
