#!/usr/bin/env Rscript
# Benchmark GARCH(1,1) com rugarch

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
reps <- as.integer(args[2])

df <- read.csv(path)
y <- df$y

library(rugarch)
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# warmup
invisible(ugarchfit(spec, y, solver = "hybrid"))

times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- Sys.time()
  m <- ugarchfit(spec, y, solver = "hybrid")
  t1 <- Sys.time()
  times[i] <- as.numeric(t1 - t0, units = "secs")
}

cat(jsonlite::toJSON(list(
  mean = mean(times),
  std = sd(times),
  min = min(times),
  max = max(times),
  reps = reps
), auto_unbox = TRUE), "\n")
