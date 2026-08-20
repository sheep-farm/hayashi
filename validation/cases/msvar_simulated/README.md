# msvar_simulated

Markov-switching VAR validation on simulated bivariate data.

- DGP: two-regime VAR with known intercepts and transition matrix.
- Hayashi: `msvar(y1 ~ y2, df, regimes=2, lags=1)`.
- References: `MSwM::msmFit` (R) and `statsmodels.tsa.MarkovRegression` (Python).
- Output: regime-specific `y1` intercepts and transition probabilities.
