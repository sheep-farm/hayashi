# Ramsey RESET test on simulated OLS data

The R reference generates a seeded OLS design with AR(1) errors. Hayashi runs
the RESET diagnostic on the same generated CSV. The oracle is
`lmtest::resettest` with `power = 2`.

Compared quantities: F statistic and p-value with tolerance `1e-4`.
