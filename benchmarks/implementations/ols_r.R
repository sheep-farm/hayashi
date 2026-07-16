#!/usr/bin/env Rscript
# OLS benchmark with R lm()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
reps <- as.integer(args[2])

df <- read.csv(path)

# warmup
invisible(lm(y ~ x1 + x2 + x3, data = df))

times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- Sys.time()
  m <- lm(y ~ x1 + x2 + x3, data = df)
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
