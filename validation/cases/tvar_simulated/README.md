# tvar_simulated

Threshold VAR validation on simulated bivariate data.

- DGP: bivariate TVAR with an exogenous threshold `q`.
- Hayashi: `tvar` on the generated data.
- References: `tsDyn::TVAR` (R) and a Python threshold-regime OLS (Python).
- Output: `variable,coef,std_err` CSV with regime coefficients.
