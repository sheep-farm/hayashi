df <- read.csv(file.path("validation", "cases", "johansen_break_simulated", "data", "data.csv"))
suppressPackageStartupMessages(library(urca))
suppressPackageStartupMessages(library(jsonlite))

Y <- df[, c("y1", "y2")]
shift <- df$shift

jo <- ca.jo(Y, type = "trace", ecdet = "trend", K = 2, spec = "longrun", dumvar = cbind(shift))
s <- summary(jo)

teststat <- s@teststat
cval5 <- s@cval[, 2]

rank <- 0
if (teststat[2] > cval5[2]) {
  rank <- 1
}

result <- list(
  coefficients = list(
    rank = rank,
    trace_0 = teststat[2],
    trace_1 = teststat[1]
  ),
  standard_errors = list(
    rank = 0,
    trace_0 = 0,
    trace_1 = 0
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15), "\n")
