# PCA on Wooldridge `wage1`

This case validates Hayashi's standardised principal-component analysis on
four numeric variables from the public Wooldridge `wage1` teaching dataset:
`educ`, `exper`, `tenure`, and `wage`.

## Protocol

- Hayashi `pca(df, educ, exper, tenure, wage, n=2)`.
- R `prcomp(..., center = TRUE, scale. = TRUE)`.
- Python `numpy.linalg.eigh` of the corresponding sample correlation matrix.
- Compare the first two eigenvalues, explained-variance ratios, and absolute
  loadings.

PCA eigenvectors are defined only up to sign. A reference implementation may
therefore return the negative of a valid component. The case compares
absolute loadings rather than raw signed loadings, while eigenvalues and
variance ratios are sign-invariant.

The four-decimal Hayashi display is reflected in the declared tolerances.
