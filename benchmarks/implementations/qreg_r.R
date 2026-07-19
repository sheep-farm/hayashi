#!/usr/bin/env Rscript
# Quantile regression benchmark with quantreg

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)

library(quantreg)

# warmup
for (i in seq_len(warmup)) {
  m <- suppressWarnings(rq(y ~ x1 + x2, data = df, tau = 0.5))
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- suppressWarnings(rq(y ~ x1 + x2, data = df, tau = 0.5))
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.4fs\n", t1 - t0))
}
