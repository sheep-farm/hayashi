# Reference implementation in R for the ETS GDP case.
#
# Simple exponential smoothing (SES, ETS(A,N,N)) matching the Hayashi `ses(df, gdp)`
# call.  Only alpha is reported because the Hayashi text output exposes only the
# smoothing parameter.

library(jsonlite)

# Ensure the data directory exists.
data_dir <- "validation/cases/ets_gdp/data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Load macrodata from statsmodels mirror (Rdatasets).
url <- "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/statsmodels/macrodata.csv"
macro <- read.csv(url)
macro <- macro[, c("year", "quarter", "realgdp")]
names(macro)[names(macro) == "realgdp"] <- "gdp"

# Write CSV for Hayashi to read.
write.csv(macro, file.path(data_dir, "macrodata.csv"), row.names = FALSE)

y <- macro$gdp
n <- length(y)

# One-step SSE for a given alpha.
ses_sse <- function(alpha) {
  # Initial level is the first observation.
  l <- y[1]
  sse <- 0.0
  for (t in 2:n) {
    forecast <- l
    error <- y[t] - forecast
    sse <- sse + error * error
    l <- alpha * y[t] + (1 - alpha) * l
  }
  sse
}

# Grid search for alpha on [0, 1].
alpha_grid <- seq(0.001, 0.999, length.out = 1000)
best_alpha <- 0.0
best_sse <- Inf
for (a in alpha_grid) {
  s <- ses_sse(a)
  if (s < best_sse) {
    best_sse <- s
    best_alpha <- a
  }
}

# Fine refinement around the best candidate.
refine <- seq(max(0.0, best_alpha - 0.01), min(1.0, best_alpha + 0.01), length.out = 1000)
for (a in refine) {
  s <- ses_sse(a)
  if (s < best_sse) {
    best_sse <- s
    best_alpha <- a
  }
}

result <- list(
  coefficients = list(alpha = best_alpha),
  standard_errors = list(alpha = 0.0)
)

out_dir <- "validation/cases/ets_gdp/reference"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_json(result, file.path(out_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE)

cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE))
