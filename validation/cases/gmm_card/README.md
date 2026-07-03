# GMM returns to schooling on Wooldridge `card`

This validation case estimates a GMM model of returns to schooling with `nearc4` as an instrument for `educ`.

## Model

```
lwage ~ educ + exper + expersq + smsa + black + south
```

where `educ` is endogenous and instrumented by `nearc4`.

## Dataset

- **Name:** `wooldridge::card`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 3,010 observations × 34 variables.

## Reference implementation

- **R:** `AER::ivreg(lwage ~ educ + exper + expersq + smsa + black + south | nearc4 + exper + expersq + smsa + black + south, data = card)`
- **Python:** `linearmodels.iv.IVGMM.from_formula("lwage ~ 1 + [educ ~ nearc4] + exper + expersq + smsa + black + south", data=card).fit()`
- **Hayashi:** `gmm(lwage ~ educ + fexper + fexpersq + fsmsa + fblack + fsouth, ~ fnearc4 + fexper + fexpersq + fsmsa + fblack + fsouth, df)`

## Compared quantities

- coefficients
- standard errors

## Tolerances

| Quantity | Tolerance | Rationale |
|---|---|---|
| coefficients | 1e-3 | GMM/2SLS should match closely |
| standard_errors | 1e-3 | Same tolerance as coefficients |
