# Reference implementation in R for the Wooldridge wage1 centile case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/centile_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

x <- sort(wage1$wage)
n <- length(x)

pcts <- c(10, 25, 50, 75, 90)
# Hayashi uses round(p/100 * (n-1)) with half away from zero.
idx <- sapply(pcts, function(p) floor(p / 100 * (n - 1) + 0.5) + 1)
values <- x[idx]

result <- list(
  centile = pcts,
  value = as.numeric(values)
)

out_dir <- "validation/cases/centile_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
