# Shapiro-Wilk test on simulated normal data

The R reference generates a seeded normal sample and writes the CSV consumed by
Hayashi. The oracle is R `shapiro.test`.

Compared quantities: W statistic and p-value with tolerances `0.001` and `0.1`.
