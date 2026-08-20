# IV returns to schooling for married women on Wooldridge mroz (Chapter 15, Example 15.1)

This validation case estimates the returns to schooling IV equation for married women from Wooldridge's *Introductory Econometrics: A Modern Approach* (7e), Chapter 15, Example 15.1:

```
lwage ~ educ + exper + expersq
instruments: fatheduc, motheduc, exper, expersq
```

## Dataset

- **Name:** `wooldridge::mroz`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge-python`.
- **Licence:** Public teaching dataset.
- **Size:** 753 observations × 22 variables.

## Reference implementation

- **R:** `ivreg(lwage ~ educ + exper + expersq | fatheduc + motheduc + exper + expersq, data = mroz)`
- **Python linearmodels:** `IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ fatheduc + motheduc]", data=mroz).fit()`
- **Stata:** `ivreg lwage (educ=fatheduc motheduc) exper expersq` (optional)

## Compared quantities

- coefficients
- standard errors (homoskedastic, non-robust)

## Tolerances and rationale

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-4 | IV/2SLS should match closely across implementations |
| standard_errors | 1e-4 | Same precision target as coefficients |
