#!/usr/bin/env Rscript
# Fixed Effects panel benchmark with R plm()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)

library(plm)

# warmup
for (i in seq_len(warmup)) {
  m <- plm(y ~ x, data = df, index = c("firm", "year"), model = "within")
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- plm(y ~ x, data = df, index = c("firm", "year"), model = "within")
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.4fs\n", t1 - t0))
}
