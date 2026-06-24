# Comparison Tables

## Basic usage

Pass models directly:

```
let m1 = ols(Y ~ X1, df)
let m2 = ols(Y ~ X1 + X2, df)
let m3 = ols(Y ~ X1 + X2 + X3, df)

esttab(m1, m2, m3)
```

Prints a formatted comparison table to stdout with coefficients, standard errors, significance stars, N, R-squared, and F-statistic.

## Store-and-display pattern

```
eststo m1
eststo m2
eststo m3
esttab()
```

`eststo` pushes a model to a global store. Calling `esttab()` with no arguments displays all stored models and clears the store.

## Building models in a loop

```
let models = []

for c in ["nonrobust", "HC1", "HC3"] {
    let m = ols(Y ~ X1 + X2, df, cov=c)
    push(models, m)
}

esttab(models)
```

`esttab` accepts a list in place of positional arguments.

## Export to file

```
esttab(m1, m2, m3, fmt=latex, path="table.tex")
esttab(m1, m2, m3, fmt=html, path="table.html")
```

Supported formats: `ascii` (default), `latex`, `html`. When `path` is given, the table is written to the file. Without `path`, the formatted output goes to stdout.

## Complete example

```
load "wages.csv" as df
generate df lwage = log(wage)

let specs = [
    "lwage ~ educ",
    "lwage ~ educ + exper",
    "lwage ~ educ + exper + tenure",
]

let results = []
for s in specs {
    push(results, ols(s, df, cov=robust))
}

esttab(results, fmt=latex, path="wage_models.tex")
```

## Notes

- Column headers default to `(1)`, `(2)`, ... Override with `labels=["Base", "Main", "Full"]`.
- Standard errors appear in parentheses below coefficients by default. Use `se=false` to hide them.
- `esttab` aligns coefficients across models automatically, showing blanks where a variable is absent.
