# Instrumental Variables

## The Endogeneity Problem

OLS is inconsistent when a regressor is correlated with the error term. IV/2SLS uses instruments -- variables correlated with the endogenous regressor but uncorrelated with the error -- to obtain consistent estimates.

## Basic IV/2SLS Syntax

```hay
iv(Y ~ X_exog + X_endog, ~ Z1 + Z2 + X_exog, df)
```

- First formula: the structural equation (outcome ~ exogenous + endogenous regressors)
- Second formula: the instrument list (all exogenous variables + excluded instruments)
- Exogenous regressors must appear in both formulas

## Classic Example: Returns to Education

Using the MROZ dataset, `educ` is endogenous (correlated with unobserved ability). Parents' education serves as instruments:

```hay
use "mroz.csv"

// OLS (inconsistent if educ is endogenous)
let m_ols = reg(lwage ~ educ + exper + exper2, mroz, if=(inlf == 1))

// IV: educ instrumented by fatheduc and motheduc
let m_iv = iv(lwage ~ exper + exper2 + educ, ~ exper + exper2 + fatheduc + motheduc, mroz,
              if=(inlf == 1))

esttab(m_ols, m_iv)
```

```
──────────────────────────────────────
                (1)        (2)
             OLS         IV/2SLS
──────────────────────────────────────
educ         0.107***    0.061*
            (0.014)     (0.031)
exper        0.042***    0.044***
            (0.013)     (0.013)
exper2      -0.001*     -0.001*
            (0.000)     (0.000)
──────────────────────────────────────
N              428        428
──────────────────────────────────────
```

The IV estimate of returns to education is smaller, suggesting OLS overestimates due to ability bias.

## Robust Standard Errors

```hay
iv(lwage ~ exper + exper2 + educ, ~ exper + exper2 + fatheduc + motheduc, mroz,
   cov=robust, if=(inlf == 1))
```

All `cov=` options from OLS apply: `robust`, `hc0`--`hc3`, `cluster()`.

## First Stage Diagnostics

Hayashi automatically reports first-stage diagnostics when running `iv`:

```
First stage: educ ~ fatheduc + motheduc + exper + exper2
  F-statistic on excluded instruments:  17.56
  Partial R-sq:                          0.073
```

Rules of thumb:
- F < 10 indicates weak instruments (Staiger & Stock, 1997)
- With weak instruments, 2SLS is biased toward OLS

## Overidentification

When there are more instruments than endogenous regressors, the model is overidentified. Hayashi reports the Sargan-Hansen J-test:

```
Overidentification test (Sargan):
  J-statistic:  0.378
  p-value:      0.539
```

Failure to reject (high p-value) supports instrument validity.

## Comparing OLS and IV

```hay
let m1 = reg(lwage ~ educ + exper + exper2, mroz, if=(inlf == 1))
let m2 = iv(lwage ~ exper + exper2 + educ, ~ exper + exper2 + fatheduc, mroz,
            if=(inlf == 1))
let m3 = iv(lwage ~ exper + exper2 + educ, ~ exper + exper2 + fatheduc + motheduc, mroz,
            if=(inlf == 1))

esttab(m1, m2, m3)
```

Column (2) is exactly identified (one instrument for one endogenous variable). Column (3) is overidentified (two instruments), enabling the J-test.

## Multiple Endogenous Regressors

```hay
// educ and hours both endogenous
iv(lwage ~ exper + exper2 + educ + hours,
   ~ exper + exper2 + fatheduc + motheduc + kidsl6 + nwifeinc,
   mroz, if=(inlf == 1))
```

The number of excluded instruments must be at least equal to the number of endogenous regressors (order condition).
