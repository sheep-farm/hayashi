series <- datasets::AirPassengers
stopifnot(is.ts(series), frequency(series) == 12,
          all(start(series) == c(1949, 1)), all(end(series) == c(1960, 12)),
          length(series) == 144L, all(is.finite(series)),
          all(series > 0), all(series == as.integer(series)))
write.csv(data.frame(index = seq_along(series), passengers = as.integer(series)),
          "validation/cases/stl_airpassengers/data/data.csv", row.names = FALSE)
