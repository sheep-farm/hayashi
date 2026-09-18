# Optional diagnostic only: not a declared validation reference.
d <- read.csv("validation/cases/stl_airpassengers/data/data.csv")
stopifnot(identical(names(d), c("index", "passengers")),
          identical(d$index, 1:144), all(is.finite(d$passengers)),
          identical(as.integer(d$passengers), as.integer(datasets::AirPassengers)))
y <- ts(log(d$passengers), start = c(1949, 1), frequency = 12)
message(R.version.string, "; stats=", packageVersion("stats"),
        "; datasets=", packageVersion("datasets"), "; R.home=", R.home())
cat("outer,index,observed,trend,seasonal,remainder,weight\n")
for (outer in 0:1) {
    fit <- stats::stl(y, s.window = 7, t.window = 23, l.window = 23,
                      s.degree = 1, t.degree = 1, l.degree = 1,
                      s.jump = 1, t.jump = 1, l.jump = 1,
                      robust = outer == 1, inner = 2, outer = outer)
    stopifnot(all(fit$win == c(7, 23, 23)), all(fit$deg == 1),
              all(fit$jump == 1), fit$inner == 2, fit$outer == outer,
              identical(dim(fit$time.series), c(144L, 3L)),
              identical(tsp(fit$time.series), tsp(y)),
              all(is.finite(fit$time.series)), length(fit$weights) == 144L,
              all(is.finite(fit$weights)),
              max(abs(y - rowSums(fit$time.series))) <= 1e-12)
    for (i in seq_along(y)) {
        cat(sprintf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", outer, i, y[i],
                    fit$time.series[i, "trend"], fit$time.series[i, "seasonal"],
                    fit$time.series[i, "remainder"], fit$weights[i]))
    }
}
