# Complementary log-log on Wooldridge `affairs`

This validation case estimates a complementary log-log model for having an affair.

## Model

```
glm(affair ~ age + yrsmarr + kids + educ + relig + ratemarr, df,
    family = binomial, link = cloglog)
```

## Dataset

- **Name:** `wooldridge::affairs`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.

## Reference implementation

- **R:** `glm(..., family = binomial(link = "cloglog"))`
- **Python:** `statsmodels GLM(..., family=Binomial(link=CLogLog()))`
- **Hayashi:** `glm(..., family=binomial, link=cloglog)`

## Status

Blocked — Hayashi's binomial/cloglog GLM overflows during IRLS.
See [sheep-farm/hayashi#64](https://github.com/sheep-farm/hayashi/issues/64).
