# Condition number on simulated OLS data

The R reference generates a seeded OLS design with two regressors and compares
the condition number of the non-intercept model matrix. Hayashi runs `condnum`
on the same generated CSV. The oracle is R `kappa(..., exact = TRUE)`.

Compared quantity: `condition_number` with absolute tolerance `0.05`.
