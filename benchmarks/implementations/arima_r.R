#!/usr/bin/env Rscript
# Benchmark ARIMA com arima() do R

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
reps <- as.integer(args[2])

df <- read.csv(path)
y <- df$y

# warmup
invisible(arima(y, order = c(1, 0, 0)))

times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- Sys.time()
  m <- arima(y, order = c(1, 0, 0))
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
