# Granger causality test on simulated AR data

The R reference generates seeded paired AR(1) series and tests whether `x`
Granger-causes `y` at lag order four. Hayashi runs the same diagnostic on the
generated CSV. The oracle is `lmtest::grangertest`.

Compared quantities: F statistic and p-value with tolerances `0.1` and `0.05`.
