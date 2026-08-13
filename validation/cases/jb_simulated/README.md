# Jarque-Bera test on simulated normal data

`data/gen.R` generates one seeded normal sample. Hayashi, R, and Python all
read that CSV; references must not generate their own samples. The R oracle is
`moments::jarque.test`; the Python oracle is `scipy.stats.jarque_bera`.

Compared quantities: Jarque-Bera statistic and p-value with tolerance `1e-4`.
