#!/usr/bin/env Rscript
# VAR benchmark with vars

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)
y <- df[, c("y1", "y2")]

library(vars)

# warmup
for (i in seq_len(warmup)) {
  m <- suppressWarnings(VAR(y, p = 1, type = "const"))
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- suppressWarnings(VAR(y, p = 1, type = "const"))
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.4fs\n", t1 - t0))
}
