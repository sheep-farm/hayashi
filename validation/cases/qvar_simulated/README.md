# qvar_simulated

Quantile VAR on simulated data.

Notes: Simulated bivariate VAR(1) process. Both R (quantreg::rq) and Python
(statsmodels QuantReg) are run separately for each equation at the median
(tau=0.5). Quantile-regression standard errors are algorithm- and
implementation-specific, so only coefficients are compared and std_err is
set to NaN.

