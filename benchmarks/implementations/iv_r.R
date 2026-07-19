#!/usr/bin/env Rscript
# IV/2SLS benchmark with ivreg

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)

library(ivreg)

# warmup
for (i in seq_len(warmup)) {
  m <- suppressWarnings(ivreg(y ~ x | z, data = df))
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- suppressWarnings(ivreg(y ~ x | z, data = df))
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.4fs\n", t1 - t0))
}
