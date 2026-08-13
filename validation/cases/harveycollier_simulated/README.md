# Harvey-Collier test on simulated OLS data

The R reference generates a seeded linear OLS design. Hayashi runs the
Harvey-Collier diagnostic on the same generated CSV. The oracle is
`lmtest::harvtest`.

Compared quantities: test statistic and p-value with tolerance `0.05`.
