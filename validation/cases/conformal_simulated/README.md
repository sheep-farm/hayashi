# conformal_simulated

Conformal prediction validation on simulated linear data.

- DGP: `y = 1 + 2x1 - 1.5x2 + N(0, 0.25)` with 300 observations.
- Hayashi: `conformal(y ~ x1 + x2, df, alpha=0.1, calib=0.3)`.
- R reference: split-conformal with OLS base predictor.
- Python reference: `statsmodels` OLS with split-conformal scores.
- Status: `blocked` because Hayashi's `conformal()` currently returns near-zero-width intervals (empirical coverage ~1% instead of ~90%).
