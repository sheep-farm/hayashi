# Breusch-Pagan LM test for individual effects

Validates `bplm(lwage ~ union + married, df, id="nr")` on the Wooldridge
`wagepan` panel dataset.

## Data

`data/gen.py` downloads `wooldridge::wagepan` (or uses a cached copy) and saves
it as `wagepan.csv`.

## Comparison

- `lm_stat`: the Breusch-Pagan Lagrange Multiplier statistic for individual
  effects.
- `p_value`: right-tail p-value from χ²(1).

The Python reference computes the same closed-form expression used by
Hayashi/Greeners:

```
LM = (n * T) / (2 * (T - 1)) * ((A / B) - 1)^2
A = (1 / T) * sum_i (sum_t e_it)^2
B = sum_i,t e_it^2
```
