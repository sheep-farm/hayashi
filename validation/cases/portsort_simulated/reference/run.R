library(jsonlite)
setwd("/home/flavio/Documents/GitHub/hayashi")
df <- read.csv("validation/cases/portsort_simulated/data/data.csv")
df <- df[order(df$size), ]
n <- 5
N <- nrow(df)
per <- N %/% n
means <- c()
ses <- c()
labels <- c("Low", "P2", "P3", "P4", "High")
for (p in 1:n) {
  start <- (p - 1) * per + 1
  end <- if (p == n) N else p * per
  rets <- df$ret[start:end]
  means <- c(means, mean(rets))
  ses <- c(ses, sd(rets) / sqrt(length(rets)))
}
names(means) <- labels
names(ses) <- labels
result <- list(
  coefficients = as.list(means),
  standard_errors = as.list(ses)
)
result$coefficients[["hl_mean"]] <- means[5] - means[1]
result$standard_errors[["hl_mean"]] <- NA_real_
cat(toJSON(result, pretty = FALSE, digits = 10, auto_unbox = TRUE))
