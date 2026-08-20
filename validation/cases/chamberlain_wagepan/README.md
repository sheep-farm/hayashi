# Chamberlain test for period-specific correlation

Validates `chamberlain(lwage ~ union + married, df, id="nr", time="year")`
on the Wooldridge `wagepan` panel dataset.

## Data

`data/gen.py` downloads `wooldridge::wagepan` and saves it as `wagepan.csv`.

## Comparison

- `f_stat`: the Chamberlain F-statistic.
- `p_value`: right-tail p-value from F(df_num, df_denom).

The Python reference builds the unrestricted pooled-OLS model
`y_it = const + X_it beta + sum_s X_i,s Pi_s`, where `X_i,s` is the value of
each regressor for entity `i` in period `s`, and computes the F-test for
H0: all `Pi_s = 0`. This matches the Hayashi/Greeners Chamberlain formulation.
