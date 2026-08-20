# Panel feasible GLS on Wooldridge `wagepan`

This validation case would estimate a panel feasible GLS (Parks/Kmenta) model of log wages.

## Model

```
xtgls(lwage ~ educ + exper + expersq + married + union, df,
      id=nr, time=year, panels=hetero)
```

## Dataset

- **Name:** `wooldridge::wagepan`
- **Source:** R package `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **Python:** Two-step feasible GLS with panel-level heteroskedasticity, reproducing the Parks/Kmenta `xtgls panels(heteroskedastic)` estimator.
- **Hayashi:** `xtgls(...)`

## Status

Pass — Hayashi matches the Python reference for coefficients and standard errors.
