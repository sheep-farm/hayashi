# Hayashi

**An interpreted language for applied econometrics.**

Hayashi combines Stata-like syntax with modern language features — closures, pipe operators, pattern matching, namespaces — all running on a pure Rust engine with zero system dependencies.

```
load "wages.dta" as df

let m1 = ols(lwage ~ educ + exper + tenure, df)
let m2 = ols(lwage ~ educ + exper + tenure, df, cov=robust)
let m3 = fe(lwage ~ married + union, df)

esttab(m1, m2, m3)
```

## Why Hayashi?

- **Free and open source** (GPL-3.0) — no license fees, no restrictions
- **Single binary** (~20 MB) — no installers, no dependencies, runs anywhere
- **46 estimators** — OLS, IV, Panel, Logit/Probit, ARIMA, GARCH, VAR, DID, and more
- **Modern language** — closures, pipe (`|>`), namespaces (`mod::func()`), f-strings, try/catch
- **8 I/O formats** — CSV, DTA, Excel, Parquet, JSON, SQLite, TSV, ODBC
- **Built-in `esttab`** — publication-ready comparison tables in one line
- **100% Rust** — no C, no Fortran, no BLAS/LAPACK to install

## Quick taste

```
// Load data, transform, estimate, compare — 10 lines
load "data.csv" as df

generate df lwage = log(wage)
generate df exper2 = exper |> |x| x * x

let models = []
for c in ["nonrobust", "HC1", "HC3"] {
    push(models, ols("lwage ~ educ + exper + exper2", df, cov=c))
}
esttab(models)
```

## Named after

[Fumio Hayashi](https://en.wikipedia.org/wiki/Fumio_Hayashi), author of *Econometrics* (Princeton, 2000) — the definitive graduate textbook on econometric theory.

## Links

- **Source**: [github.com/sheep-farm/hayashi](https://github.com/sheep-farm/hayashi)
- **Crate**: [crates.io/crates/hayashi-lang](https://crates.io/crates/hayashi-lang)
- **License**: GPL-3.0
