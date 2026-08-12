library(jsonlite)

case_dir <- "validation/cases/hawkes_simulated"
df <- read.csv(file.path(case_dir, "data", "data.csv"))
events <- as.numeric(df$time)
T <- as.numeric(readLines(file.path(case_dir, "data", "T.txt")))

nll <- function(theta) {
  mu <- theta[1]
  alpha <- theta[2]
  beta <- theta[3]
  if (mu <= 0 || alpha < 0 || beta <= 0 || alpha >= beta) return(1e9)
  comp <- numeric(length(events))
  for (i in seq_along(events)) {
    if (i > 1) {
      comp[i] <- sum(exp(-beta * (events[i] - events[1:(i - 1)])))
    }
  }
  lambdas <- mu + alpha * comp
  if (any(lambdas <= 0)) return(1e9)
  Lambda <- mu * T + (alpha / beta) * sum(1 - exp(-beta * (T - events)))
  ll <- sum(log(lambdas)) - Lambda
  return(-ll)
}

res <- optim(c(0.5, 0.3, 2.0), nll, method = "L-BFGS-B",
             lower = c(0.001, 0, 0.001), upper = rep(Inf, 3))
theta <- res$par

result <- list(
  coefficients = list(
    mu = as.numeric(theta[1]),
    alpha = as.numeric(theta[2]),
    beta = as.numeric(theta[3]),
    branching_ratio = as.numeric(theta[2] / theta[3])
  ),
  standard_errors = list(
    mu = NaN,
    alpha = NaN,
    beta = NaN,
    branching_ratio = NaN
  )
)

cat(toJSON(result, auto_unbox = TRUE, digits = 15, na = "null"))
