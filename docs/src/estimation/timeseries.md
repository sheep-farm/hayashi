# Time Series

## Setting the Time Index

```hay
use "macro_quarterly.csv"
tsset(macro, time=date)
```

Once `tsset` is declared, lag/lead/difference operators become available.

## Lag, Lead, and Difference Operators

```hay
// L. = lag, F. = lead, D. = first difference
reg(inflation ~ L.inflation + L2.inflation + unemployment, macro)
reg(gdp_growth ~ F.fed_rate, macro)
reg(D.gdp ~ D.investment + D.consumption, macro)
```

`L2.` means two-period lag. `D.` is equivalent to `x - L.x`.

## OLS with Newey-West SE

For time series regressions with autocorrelated and heteroskedastic errors:

```hay
// Phillips curve
reg(inflation ~ unemployment + L.inflation, macro, nw=4)
```

```
Newey-West SE (lag = 4)
──────────────────────────────────────
              coef      SE       t      p
──────────────────────────────────────
unemployment -0.523   0.187   -2.80   0.006
L.inflation   0.684   0.095    7.21   0.000
_cons         4.312   1.248    3.45   0.001
──────────────────────────────────────
```

Interpretation: a 1pp increase in unemployment is associated with a 0.52pp decrease in inflation, controlling for inflation persistence.

## ARIMA

```hay
// ARIMA(1,1,1)
arima(gdp, macro, order=(1,1,1))

// ARIMA with exogenous regressors (ARIMAX)
arima(inflation ~ oil_price, macro, order=(2,0,1))
```

Reports AR and MA coefficients, sigma-squared, AIC, and BIC.

## GARCH Family

Volatility models for financial returns:

```hay
use "sp500_daily.csv"

// GARCH(1,1)
garch(ret, sp500, order=(1,1))

// EGARCH (captures asymmetric volatility)
egarch(ret, sp500, order=(1,1))

// GJR-GARCH (threshold GARCH)
gjrgarch(ret, sp500, order=(1,1))
```

EGARCH and GJR-GARCH capture the leverage effect: negative shocks increase volatility more than positive shocks of the same magnitude.

## VAR

Vector autoregression for multivariate time series:

```hay
// VAR(2) with two endogenous variables
var(inflation + unemployment, macro, lags=2)
```

Reports coefficients for each equation, Granger causality tests, and information criteria for lag selection.

Impulse response functions:

```hay
let v = var(inflation + unemployment, macro, lags=2)
irf(v, impulse=unemployment, response=inflation, steps=12)
```

## VECM

Vector error correction model for cointegrated series:

```hay
vecm(consumption + income, macro, lags=2, rank=1)
```

`rank=` specifies the number of cointegrating relationships. Use the Johansen test to determine rank:

```hay
johansen(consumption + income, macro, lags=2)
```

## Phillips Curve Example

```hay
use "us_macro_quarterly.csv"
tsset(macro, time=quarter)

let m1 = reg(inflation ~ unemployment, macro, nw=4)
let m2 = reg(inflation ~ unemployment + L.inflation, macro, nw=4)
let m3 = reg(D.inflation ~ unemployment, macro, nw=4)
esttab(m1, m2, m3)
```
