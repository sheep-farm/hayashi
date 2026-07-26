# Reference implementation in R for the Wooldridge mroz cross-tabulation case.

library(wooldridge)
library(jsonlite)

data_dir <- "validation/cases/tabulate_mroz_inlf_kidslt6/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

data(mroz)

# Write CSV for Hayashi to read.
write.csv(mroz, file.path(data_dir, "mroz.csv"), row.names = FALSE)

# Cross-tabulation.
tab <- table(mroz$inlf, mroz$kidslt6)
chi <- chisq.test(tab, correct = FALSE)

# Build long-format table matching Hayashi's output.
rows <- as.character(rownames(tab))
cols <- as.character(colnames(tab))
var1 <- character(0)
var2 <- character(0)
freq <- integer(0)
row_total <- integer(0)
col_total <- integer(0)

for (r in rows) {
  rt <- 0
  for (c in cols) {
    f <- tab[r, c]
    var1 <- c(var1, r)
    var2 <- c(var2, c)
    freq <- c(freq, f)
    rt <- rt + f
    col_total <- c(col_total, sum(tab[, c]))
  }
  row_total <- c(row_total, rep(rt, length(cols)))
}

# Row and column totals are duplicated per cell in Hayashi's output.
result <- list(
  chi2 = as.numeric(chi$statistic),
  df = as.numeric(chi$parameter),
  p_value = as.numeric(chi$p.value),
  table = list(
    inlf = var1,
    kidslt6 = var2,
    freq = freq,
    row_total = row_total,
    col_total = col_total
  )
)

out_dir <- "validation/cases/tabulate_mroz_inlf_kidslt6/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(toJSON(result, pretty = FALSE, digits = NA, auto_unbox = TRUE))
