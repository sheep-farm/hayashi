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

- **R:** to be determined — no suitable CRAN package for Stata-style `xtgls` was available.
- **Hayashi:** `xtgls(...)`

## Status

Blocked — no R reference implementation for panel FGLS (Parks/Kmenta) could be installed.
See [sheep-farm/hayashi#66](https://github.com/sheep-farm/hayashi/issues/66).
