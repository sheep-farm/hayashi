# Variance inflation factors on simulated OLS data

The R reference generates a seeded OLS design and computes variance inflation
factors for `x` and `z`. Hayashi runs `vif` on the same generated CSV. This is
an R-only validation case; no Python reference is declared. The oracle is
`car::vif`.

Compared quantities: VIF for `x` and `z` with tolerance `1e-6`.
