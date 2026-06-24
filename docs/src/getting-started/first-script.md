# First Script

Hayashi scripts are plain text files with the `.hay` extension. Run them with `hay <file>`.

## Inline data and OLS

Create a file `hello.hay`:

```
// hello.hay — wage equation with inline data

input df
wage educ exper tenure
12.5 12 10 5
15.2 14  8 3
10.1 10 15 7
18.7 16 12 6
11.3 12  5 2
20.5 18 20 10
13.8 14  6 4
 9.5 10  3 1
16.4 16 10 8
14.2 14 12 5
end

generate df lwage = log(wage)

let m = ols(lwage ~ educ + exper + tenure, df)
print(m)
```

Run it:

```bash
hay hello.hay
```

This prints OLS output: coefficients, standard errors, t-statistics, R-squared -- the same table format a Stata user would expect.

## Loading a CSV and comparing models

A more realistic workflow loads external data, estimates multiple specifications, and compares them side by side:

```
// compare.hay — returns to education

load "wages.csv" as df

generate df lwage = log(wage)
generate df exper2 = exper * exper

let m1 = reg(lwage ~ educ, df)
let m2 = reg(lwage ~ educ + exper + exper2, df)
let m3 = reg(lwage ~ educ + exper + exper2 + tenure, df, cov=robust)

esttab(m1, m2, m3)
```

`esttab` prints a publication-style comparison table with coefficients, standard errors, N, and R-squared for each model in columns.

Run:

```bash
hay compare.hay
```

## What to read next

- [The REPL](./repl.md) -- interactive exploration without writing a file
- [Loading Data](../data/loading.md) -- CSV, DTA, Parquet, Excel, and more
- [OLS](../estimation/ols.md) -- full reference for linear regression
