# factor_simulated

Factor analysis validation on simulated two-factor data.

- DGP: four variables with two underlying factors and small noise.
- Hayashi: `factor(df, x1, x2, x3, x4, n=2)`.
- R reference: `eigen(cor(df))$values` sorted in decreasing order.
- Python reference: `np.linalg.eigvalsh` of the correlation matrix, sorted.
- Output: first and second eigenvalues of the correlation matrix.
