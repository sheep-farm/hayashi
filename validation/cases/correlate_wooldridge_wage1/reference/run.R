# Reference implementation in R for the Wooldridge wage1 correlation case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/correlate_wooldridge_wage1/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(wage1)

# Write CSV for Hayashi to read.
write.csv(wage1, file.path(data_dir, "wage1.csv"), row.names = FALSE)

vars <- c("wage", "educ", "exper", "tenure")
sub <- wage1[, vars]
mat <- cor(sub, use = "complete.obs")

# Build lower-triangular pairwise list (including diagonal) in sorted name order.
sorted_vars <- sort(vars)
var1 <- character(0)
var2 <- character(0)
r_vals <- numeric(0)
p_vals <- numeric(0)
n <- nrow(sub)

for (i in sorted_vars) {
  for (j in sorted_vars) {
    if (which(sorted_vars == j) > which(sorted_vars == i)) next
    rij <- mat[i, j]
    # p-value from t-test on correlation: t = r * sqrt((n-2)/(1-r^2))
    if (i == j) {
      pij <- 0.0
    } else if (n <= 2 || (1 - rij^2) <= 0) {
      pij <- 1.0
    } else {
      t <- rij * sqrt((n - 2) / (1 - rij^2))
      pij <- 2 * (1 - pt(abs(t), n - 2))
    }
    var1 <- c(var1, i)
    var2 <- c(var2, j)
    r_vals <- c(r_vals, rij)
    p_vals <- c(p_vals, pij)
  }
}

result <- list(
  var1 = var1,
  var2 = var2,
  r = r_vals,
  p = p_vals
)

out_dir <- "validation/cases/correlate_wooldridge_wage1/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)

cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
