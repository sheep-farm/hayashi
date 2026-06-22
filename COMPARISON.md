# Hayashi vs Stata

## Overview

| | Hayashi | Stata 18 |
|---|---|---|
| Price | Free (GPL-3.0) | US$ 595–2,985/year |
| Binary | ~20 MB, zero system deps | ~500 MB + license |
| Language | Rust | C/Java |
| Interface | Terminal (REPL + script) + VS Code | GUI + terminal |
| I/O | CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC | DTA, CSV, Excel, ODBC |
| Graphics | SVG + ASCII | PNG/SVG/PDF native |
| Tests | 382 automated + 59 examples | Internal proprietary suite |
| Scoping | Block-scoped, const, no GC | Global |
| DataFrames | Multiple simultaneous, Rc COW | Single active dataset (frames since v16) |
| Types | int, float, bool, str, list, dict, closures | Numeric + string |
| License | GPL-3.0 | Proprietary |

## Syntax side by side

```
// Stata                              // Hayashi
reg Y X1 X2                           reg(Y ~ X1 + X2, df)
reg Y X1 X2, vce(robust)              reg(Y ~ X1 + X2, df, cov=robust)
reg Y X1 X2, vce(cluster firm)        reg(Y ~ X1 + X2, df, cluster=firm)
reg Y X1 X2 if year==2020             reg(Y ~ X1 + X2, df, if=year==2020)
ivregress 2sls Y (X1=Z1 Z2) X2       iv(Y ~ X1 + X2, ~ Z1 + Z2, df)
xtset firm year                       xtset(df, firm, year)
xtreg Y X1 X2, fe                     fe(Y ~ X1 + X2, df)
xtreg Y X1 X2, re                     re(Y ~ X1 + X2, df)
hausman fe re                         hausman(m_fe, m_re)
test X1 X2                            test(m, "X1", "X2")
test X1 = X2                          test(m, "X1 = X2")
nlcom _b[X1]/_b[X2]                   nlcom(m, X1 / X2)
margins, dydx(X1)                     margins(m, dydx=[X1])
margins, at(X2=0)                     margins(m, at_X2=0)
estat ic                              estat(m1, m2)
predict yhat                          predict df yhat = m
predict e, resid                      predict df e = m, "residuals"
eststo: reg Y X1                      eststo(reg(Y ~ X1, df))
esttab, se                            esttab()
esttab using "t.tex", tex             esttab(fmt=latex, path="t.tex")
scatter Y X                           graph_scatter(df, X, Y, path="f.svg")
histogram Y                           graph_hist(df, Y, path="f.svg")
gen lnY = log(Y)                      generate df lnY = log(Y)
gen D = regexm(name, "^Dr")           generate df D = regexm(name, "^Dr")
replace Y = 0 if X > 10               replace df Y = 0 if X > 10
winsor2 Y, cuts(1 99)                 winsor(df, Y, p=0.01)
encode str_var, gen(num)               encode(df, str_var)
tab group, gen(d_)                     tabgen(df, group)
summarize, detail                      summarize(df, detail=true)
ci means Y                            ci(df, Y)
pwcorr X1 X2 X3, star(0.05)           pwcorr(df, X1, X2, X3)
preserve                              preserve(df)
restore                               restore(df)
quietly reg Y X                        quietly(ols(Y ~ X, df))
capture reg Y X                        capture(ols(Y ~ X, df))
assert price > 0                       assert(X > 0, "msg")
.                                      // no Stata equivalent:
.                                      let r = if x > 0 { "pos" } else { "neg" }
.                                      [1,2,3] |> map(|x| x*2) |> sort
.                                      let d = {"a": 1, "b": 2}
.                                      match x { 1 => "one", _ => "other" }
.                                      try { ... } catch e { display e }
.                                      f"mean = {mu:.2f}"

foreach v in X1 X2 X3 {               for v in ["X1", "X2", "X3"] {
    reg Y `v'                              eststo(ols("Y ~ " + v, df))
    est store m_`v'                    }
}                                      esttab()
esttab m_*
```

## Estimator coverage

| Category | Stata | Hayashi | Status |
|---|---|---|---|
| OLS + HC1-HC4 | `reg` | `ols`/`reg` | Parity |
| Cluster SEs | `vce(cluster)` | `cluster=` | Parity |
| Two-way cluster | `vce(cluster c1 c2)` | `cluster= cluster2=` | Parity |
| Newey-West | `newey` | `nw=` | Parity |
| IV/2SLS | `ivregress` | `iv` | Parity |
| Panel FE/RE | `xtreg` | `fe`/`re` + `xtset` | Parity |
| Arellano-Bond | `xtabond`/`xtdpdsys` | `ab`/`sysgmm` | Parity |
| Hausman | `hausman` | `hausman` | Parity |
| Logit/Probit | `logit`/`probit` | `logit`/`probit` | Parity |
| Margins AME + SEs | `margins` | `margins` | Parity |
| Poisson/NegBin | `poisson`/`nbreg` | `poisson`/`nbreg` | Parity |
| ZIP/ZINB | `zip` | `zip`/`zinb` | Parity |
| Ordered logit/probit | `ologit`/`oprobit` | `ologit`/`oprobit` | Parity |
| Multinomial logit | `mlogit` | `mlogit` | Parity |
| Tobit/Heckman | `tobit`/`heckman` | `tobit`/`heckman` | Parity |
| Quantile | `qreg` | `qreg` | Parity |
| ARIMA/GARCH | `arima`/`arch` | `arima`/`garch`/`egarch` | Parity |
| VAR/VECM/SVAR | `var`/`vec` | `var`/`vecm`/`svar` | Parity |
| Lasso/Ridge | `lasso` | `lasso`/`ridge`/`elasticnet` | Parity |
| Cox PH | `stcox` | `cox` | Parity |
| DID/RD/Synth/PSM | addons | builtins | Parity |
| GLM | `glm` | `glm` | Parity |
| Robust (M-est) | — | `rlm` | Hayashi only |
| GEE | — | `gee` | Hayashi only |
| Beta regression | — | `betareg` | Hayashi only |
| Fama-MacBeth | `xtfmb` (paid addon) | `fmb` (builtin + NW) | Hayashi superior |
| Portfolio sorts | manual coding | `portsort`/`doublesort` | Hayashi superior |
| Mixed/HLM | `mixed` | `mixed` | Partial |
| Survey | `svy:` | — | Missing |
| SEM | `sem`/`gsem` | — | Missing |
| Bayesian | `bayes:` | — | Missing |
| Spatial | `spregress` | — | Missing |

## Where Hayashi wins

**Cost and deployment:**
- Free and open source (GPL-3.0) vs US$ 595+/year
- Single 20 MB binary, no system dependencies
- `cargo install hayashi` — done

**I/O:**
- 8 input formats: CSV, TSV, JSON, DTA, Excel, Parquet, SQLite, ODBC
- 8 export formats: CSV, JSON, TSV, XLSX, Parquet, SQLite, LaTeX, HTML
- Stata has no native Parquet support

**Language features Stata lacks:**
- Multiple simultaneous DataFrames (Stata: one active dataset)
- F-strings: `f"mean = {mu:.2f}"`
- Closures: `|x| x * 2` with `map`, `filter`
- Pipe operator: `data |> sort |> filter(|x| x > 0)`
- Pattern matching: `match x { 1 => "one", _ => "other" }`
- Try/catch: structured error handling with error variable
- If-expression: `let r = if x > 0 { "yes" } else { "no" }`
- `in` operator: `if x in [1, 2, 3]`, `if "key" in dict`
- Dict type: `{"key": value}` with full operations
- List operations: push, pop, map, filter, sort, unique, flatten, etc.
- Const declarations: `const PI = 3.14` — immutable variables
- Block scoping with deterministic destruction (no GC)
- Function parameters are const by default (immutable input)
- No variable shadowing — prevents subtle bugs
- Type conversions: `int()`, `float()`, `str()`, `bool()`, `type()`

**Econometrics-specific:**
- Fama-MacBeth with Newey-West built-in (Stata requires paid addon)
- Portfolio sorts (`portsort`, `doublesort`) built-in
- Generic bootstrap with any estimator
- Dynamic formulas: `ols("Y ~ " + v, df)` native
- Row-wise regex in formulas: `ols(Y ~ X, df, if = regexm(name, "Dr"))`
- Copy-on-write DataFrames: zero-copy in functions

**Developer experience:**
- 382 automated tests, 59 examples, `cargo test` in <1s
- `help()` with ~110 topics, signature + example for every command
- VS Code extension (syntax highlighting, run/debug)
- Error messages with line numbers
- Multi-line expressions inside parentheses

## Where Stata wins

- **Maturity**: 40+ years, battle-tested in production
- **Documentation**: 15,000+ page manual
- **GUI**: full graphical interface
- **Ecosystem**: 10,000+ SSC packages
- **Survey**: `svy:` for complex samples
- **SEM/Bayesian/Spatial**: specialized niches
- **Graphics**: 50+ plot types vs 4 SVG + 8 ASCII
- **Academic acceptance**: de facto standard in journals
- **Support**: StataCorp + paid support

## Conclusion

Hayashi covers ~97% of the applied econometrics workflow at the graduate level with full functional parity in estimation, post-estimation, data manipulation, and publishable output. The remaining gaps are specialized niches (survey, SEM, Bayesian, spatial) that few researchers use simultaneously.

The language goes beyond Stata with modern features (closures, pattern matching, pipe, f-strings, dict, const, try/catch) that make scripts more expressive and robust. Multiple simultaneous DataFrames, block scoping, and immutable function parameters are architectural improvements over Stata's global-state model.
