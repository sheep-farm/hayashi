# Durbin-Watson test on simulated OLS data

The R reference generates a seeded OLS design with AR(1) errors. Hayashi runs
the Durbin-Watson diagnostic on the same generated CSV. The oracle is
`lmtest::dwtest`.

Compared quantity: `dw` with absolute tolerance `1e-4`.
