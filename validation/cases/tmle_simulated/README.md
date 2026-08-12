# tmle_simulated

Targeted maximum likelihood estimation (TMLE) validation on simulated observational data.

- DGP: `y ~ t + x1 + x2` with true ATE 0.7.
- Hayashi: `tmle(y ~ t, df, w="x1,x2")`.
- References: `tmle::tmle` (R) and a manual TMLE using `sklearn` (Python).
- Output: `variable,coef,std_err` CSV with the average treatment effect and its standard error.
