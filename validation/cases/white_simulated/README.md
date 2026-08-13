# White test on simulated OLS data

The R reference generates a seeded OLS design with AR(1) errors. Hayashi runs
the White diagnostic on the same generated CSV. The oracle is `lmtest::bptest`
with the declared squared and interaction auxiliary regressors.

Compared quantities: LM statistic and p-value with tolerance `1e-3`.
