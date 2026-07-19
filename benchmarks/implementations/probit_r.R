#!/usr/bin/env Rscript
# Probit benchmark with R glm()

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
iters <- as.integer(args[2])
warmup <- as.integer(args[3])

df <- read.csv(path)

# warmup
for (i in seq_len(warmup)) {
  m <- suppressWarnings(glm(y ~ x1 + x2, data = df, family = binomial(link = "probit")))
}

for (i in seq_len(iters)) {
  t0 <- proc.time()[["elapsed"]]
  m <- suppressWarnings(glm(y ~ x1 + x2, data = df, family = binomial(link = "probit")))
  t1 <- proc.time()[["elapsed"]]
  cat(sprintf("  elapsed: %.6fs\n", t1 - t0))
}
