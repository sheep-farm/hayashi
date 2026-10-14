# Native cointReg plus the explicitly labelled N-to-N-1 bias convention change.
d <- read.csv("validation/cases/fmols_simulated/data/data.csv")
stopifnot(identical(names(d), c("y", "x")), nrow(d) == 300L,
          all(is.finite(as.matrix(d))),
          as.character(packageVersion("cointReg")) == "0.2.0")
N <- nrow(d)
n <- N - 1L
B <- max(1L, floor(4 * (N / 100)^(2 / 9)))
fit <- cointReg::cointRegFM(as.matrix(d["x"]), d$y, deter = matrix(1, N, 1),
                           kernel = "ba", bandwidth = B + 1L, demeaning = FALSE)
eta <- cbind(as.numeric(fit$u.OLS)[-1], diff(d$x))
lr <- cointReg::getLongRunVar(eta, kernel = "ba", bandwidth = B + 1L,
                             demeaning = FALSE)
stopifnot(all(is.finite(lr$Omega)), all(is.finite(lr$Delta)),
          lr$Omega[2, 2] > 0, all(is.finite(fit$theta)),
          all(is.finite(fit$sd.theta)), all(fit$sd.theta > 0))
c <- lr$Omega[2, 1] / lr$Omega[2, 2]
b <- lr$Delta[2, 1] - lr$Delta[2, 2] * c
Z <- cbind(1, d$x[-1])
change <- as.numeric(solve(crossprod(Z), c(0, b)))
chosen <- as.numeric(fit$theta) + change
stopifnot(all(is.finite(chosen)), all(is.finite(change)))
result <- list(
  coefficients = list(Intercept = chosen[1], x = chosen[2], n_obs = n, bandwidth = B),
  standard_errors = list(Intercept = unname(fit$sd.theta[1]), x = unname(fit$sd.theta[2])),
  diagnostics = list(native_R_coefficients = unname(fit$theta),
                     coefficient_convention_change = change,
                     native_bias_multiplier = N, selected_bias_multiplier = n)
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA), "\n")
