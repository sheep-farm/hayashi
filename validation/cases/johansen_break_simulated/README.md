# johansen_break_simulated

Johansen cointegration with structural break validation on simulated data.

- DGP: cointegrated bivariate system with a known break date.
- Hayashi: `johansen_break(..., breaks=[...])`.
- R reference: `urca::ca.jo` with `dumvar=` break dummies.
- Python reference: `statsmodels.tsa.vector_ar.vecm.coint_johansen` does not support exogenous break dummies.
- R's rank decision uses conventional critical values. Hayashi uses break-adjusted
  critical values, so neither available reference is an independent oracle for
  the declared rank-and-trace contract.

This is marked `not-supported` by the validation programme. It does not state
that Hayashi lacks `johansen_break`; it records that the declared independent
reference contract cannot currently be reproduced.
