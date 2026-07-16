#!/usr/bin/env Rscript
# Benchmark Fixed Effects panel com plm()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
reps <- as.integer(args[2])

df <- read.csv(path)

library(plm)

# warmup
invisible(plm(y ~ x, data = df, index = c("firm", "year"), model = "within"))

times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- Sys.time()
  m <- plm(y ~ x, data = df, index = c("firm", "year"), model = "within")
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
