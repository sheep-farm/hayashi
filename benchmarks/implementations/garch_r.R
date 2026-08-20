#!/usr/bin/env Rscript
# GARCH(1,1) benchmark with rugarch

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)
y <- df$y

library(rugarch)
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# warmup
for (i in seq_len(warmup)) {
  capture.output(invisible(ugarchfit(spec, y, solver = "hybrid")), file = nullfile())
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  capture.output(invisible(ugarchfit(spec, y, solver = "hybrid")), file = nullfile())
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.6fs\n", t1 - t0))
}
