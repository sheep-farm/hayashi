#!/usr/bin/env Rscript
# ARIMA benchmark with R arima()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)
y <- df$y

# warmup
for (i in seq_len(warmup)) {
  m <- arima(y, order = c(1, 0, 0))
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- arima(y, order = c(1, 0, 0))
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.4fs\n", t1 - t0))
}
