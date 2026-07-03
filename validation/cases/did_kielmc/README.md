# Difference-in-differences on Kiel-McClain housing prices

This validation case estimates a difference-in-differences model for the effect of incinerator proximity on log house prices.

## Model

```
lprice ~ nearinc + y81 + nearinc:y81
```

The treatment effect is the interaction coefficient `nearinc:y81`.

## Dataset

- **Name:** `wooldridge::kielmc`
- **Source:** R package `wooldridge`; also available in Python via `wooldridge`.
- **Licence:** Public teaching dataset.
- **Size:** 321 observations × 25 variables.

## Reference implementation

- **R:** `lm(lprice ~ nearinc * y81, data = kielmc)`
- **Python:** `statsmodels.formula.ols("lprice ~ nearinc * y81", data = kielmc).fit()`
- **Hayashi:** `did(lprice ~ nearinc + y81, df, treat=nearinc, post=y81)`

## Known differences

| Source | Output |
|---|---|
| Hayashi | ATT, group means, parallel trend diff |
| R/Python | Intercept, nearinc, y81, nearinc:y81 coefficients |

## Compared quantities

- coefficients
- standard errors
