# Native cointReg with the existing labelled N-to-N-1 coefficient adjustment.
reference_results <- function(d, fixtures=c("iid", "endogeneity", "serial_correlation", "holdout")) {
  all_fixtures <- c("iid", "endogeneity", "serial_correlation", "holdout")
  if (!identical(names(d), c("fixture", "row", "y", "x")) || nrow(d) != 1200L ||
      anyNA(d$fixture) || !setequal(unique(d$fixture), all_fixtures) ||
      !all(vapply(d[c("row", "y", "x")], is.numeric, logical(1))) ||
      !all(is.finite(as.matrix(d[c("row", "y", "x")]))) ||
      !all(vapply(all_fixtures, function(name) identical(d$row[d$fixture == name], 0:299), logical(1)))) {
    stop("Invalid FMOLS fixture data: require four ordered, finite 300-row designs")
  }
  stopifnot(length(fixtures) > 0L, !anyDuplicated(fixtures), all(fixtures %in% all_fixtures),
            as.character(packageVersion("cointReg")) == "0.2.0")
  result <- list(coefficients=list(), standard_errors=list(), diagnostics=list(
    versions=list(R=as.character(getRversion()), cointReg=as.character(packageVersion("cointReg")),
                  jsonlite=as.character(packageVersion("jsonlite")),
                  matrixStats=as.character(packageVersion("matrixStats"))), fixtures=list()))
  for (name in fixtures) {
    frame <- d[d$fixture == name, c("y", "x")]
    N <- nrow(frame)
    n <- N - 1L
    B <- max(1L, floor(4 * (N / 100)^(2 / 9)))
    fit <- cointReg::cointRegFM(as.matrix(frame["x"]), frame$y, deter=matrix(1, N, 1),
                               kernel="ba", bandwidth=B+1L, demeaning=FALSE)
    eta <- cbind(as.numeric(fit$u.OLS)[-1], diff(frame$x))
    lr <- cointReg::getLongRunVar(eta, kernel="ba", bandwidth=B+1L, demeaning=FALSE)
    # cointReg pads the omitted initial residual with NA; the FM sample is n.
    stopifnot(length(fit$residuals) == N, is.na(fit$residuals[1]),
              sum(is.finite(fit$residuals)) == n, fit$bandwidth$number == B+1L,
              all(is.finite(lr$Omega)), all(is.finite(lr$Delta)), lr$Omega[2,2] > 0,
              all(is.finite(fit$theta)), all(is.finite(fit$sd.theta)), all(fit$sd.theta > 0))
    c <- lr$Omega[2,1] / lr$Omega[2,2]
    b <- lr$Delta[2,1] - lr$Delta[2,2]*c
    Z <- cbind(1, frame$x[-1])
    change <- as.numeric(solve(crossprod(Z), c(0, b)))
    chosen <- as.numeric(fit$theta) + change
    stopifnot(all(is.finite(chosen)), all(is.finite(change)))
    prefix <- if (name == "iid") "" else paste0(name, "_")
    values <- c(chosen, n, B)
    fields <- c("Intercept", "x", "n_obs", "bandwidth")
    for (j in seq_along(fields)) {
      result$coefficients[[paste0(prefix, fields[j])]] <- values[j]
    }
    result$standard_errors[[paste0(prefix, "Intercept")]] <- unname(fit$sd.theta[1])
    result$standard_errors[[paste0(prefix, "x")]] <- unname(fit$sd.theta[2])
    result$diagnostics$fixtures[[name]] <- list(N=N, n=n, returned_residual_rows=length(fit$residuals), requested_bandwidth=B+1L,
      realised_bandwidth=fit$bandwidth$number, selected_B=B, kernel="ba", demeaning=FALSE,
      native_R_coefficients=unname(fit$theta), coefficient_convention_change=change,
      native_bias_multiplier=N, selected_bias_multiplier=n)
  }
  result
}

if (sys.nframe() == 0L) {
  result <- reference_results(read.csv("validation/cases/fmols_simulated/data/data.csv"))
  cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA), "\n")
}
