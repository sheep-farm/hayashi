# johansen_break_simulated

Johansen cointegration with structural break validation on simulated data.

- DGP: cointegrated bivariate system with a known break date.
- Hayashi: `johansen_break(..., breaks=[...])`.
- R reference: `urca::ca.jo` with `dumvar=` break dummies.
- Python reference: `statsmodels.tsa.vector_ar.vecm.coint_johansen` (does not support exogenous break dummies, so the case is blocked).
- Output: blocked trace-statistic comparison.
