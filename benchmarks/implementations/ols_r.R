#!/usr/bin/env Rscript
# OLS benchmark with R lm()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)

# warmup
for (i in seq_len(warmup)) {
  m <- lm(y ~ x1 + x2 + x3, data = df)
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- lm(y ~ x1 + x2 + x3, data = df)
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.6fs\n", t1 - t0))
}
