# Independent base-R MODWT (stationary wavelet) reference using Haar filters.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)][1])
case_dir <- normalizePath(file.path(dirname(script_path), ".."))
df <- read.csv(file.path(case_dir, "data", "data.csv"))
y <- df$y
n <- length(y)

# Level-3 stationary wavelet transform with unnormalised Haar.
# Each level divides by sqrt(2) (the standard Haar normalisation) and uses
# the full signal without decimation, as pywt.swt(..., norm=FALSE) does.
V <- y
W <- numeric(3)
for (j in 1:3) {
  step <- 2^(j - 1)
  Vj <- numeric(n)
  Wj <- numeric(n)
  for (t in 1:n) {
    idx1 <- t
    idx2 <- ((t - 1 + step) %% n) + 1
    Vj[t] <- (V[idx1] + V[idx2]) / sqrt(2)
    Wj[t] <- (V[idx1] - V[idx2]) / sqrt(2)
  }
  W[j] <- as.numeric(crossprod(Wj))
  V <- Vj
}

result <- list(
  coefficients = list(
    W_1 = W[1],
    W_2 = W[2],
    W_3 = W[3]
  ),
  standard_errors = list(
    W_1 = NaN,
    W_2 = NaN,
    W_3 = NaN
  )
)

json_str <- jsonlite::toJSON(result, auto_unbox = TRUE, digits = 15, na = "null", pretty = FALSE)
json_str <- gsub(":null", ":NaN", json_str, fixed = TRUE)
cat(json_str)
