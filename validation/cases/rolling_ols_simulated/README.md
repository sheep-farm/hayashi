# rolling_ols_simulated

Rolling OLS on simulated time-series data.

Notes: Simulated linear trend (y = 1 + 2x + noise). Compares the last-window rolling OLS coefficients produced by Hayashi's rolling() against a zoo::rollapply reference and a statsmodels OLS fit on the final window.
