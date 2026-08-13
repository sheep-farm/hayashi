# Ljung-Box test on simulated AR data

The R reference generates a seeded AR(1) series. Hayashi runs the Ljung-Box
diagnostic at lag ten on the generated CSV. The oracle is R `Box.test` with
`type = "Ljung-Box"`.

Compared quantities: Q statistic and p-value with tolerance `1e-10`.
