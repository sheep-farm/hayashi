# kde_simulated

Kernel density estimation validation on simulated Gaussian data.

- DGP: `x ~ N(2, 1.5)` with 500 observations.
- Hayashi: `kde(df, x, bw=0.5, kernel="gaussian")`.
- R reference: `density(x, bw=0.5, kernel="gaussian", n=512)`.
- Python reference: manual Gaussian KDE on a 512-point support grid.
- Output: fixed bandwidth, peak density and the x-coordinate of the density peak.
