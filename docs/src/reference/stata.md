# Comparison with Stata

## Syntax side by side

### Data I/O

| Stata | Hay |
|---|---|
| `use "data.dta"` | `load "data.dta" as df` |
| `import delimited "f.csv"` | `load "f.csv" as df` |
| `import excel "f.xlsx"` | `load "f.xlsx" as df` |
| `save "out.dta"` | `export(df, "csv", "out.csv")` |
| `export delimited "f.csv"` | `export(df, "csv", "f.csv")` |
| `outsheet` | `export(df, "tsv", "f.tsv")` |
| -- | `load "f.parquet" as df` |
| -- | `load "db.db" as df, table=t` |

### Variables & Columns

| Stata | Hay |
|---|---|
| `gen lnY = log(Y)` | `generate df lnY = log(Y)` |
| `gen D = (X > 0)` | `generate df D = (X > 0)` |
| `gen X2 = X * X` | `generate df X2 = X^2` |
| `gen lbl = regexm(s, "^Dr")` | `generate df lbl = regexm(s, "^Dr")` |
| `replace Y = 0 if X > 10` | `replace df Y = 0 if X > 10` |
| `drop var1 var2` | `drop(df, var1, var2)` |
| `keep var1 var2` | `keep(df, var1, var2)` |
| `rename old new` | `rename(df, old, new)` |
| `sort year firm` | `sort(df, year, firm)` |
| `encode str_var, gen(n)` | `encode(df, str_var)` |
| `tab group, gen(d_)` | `tabgen(df, group)` |
| `winsor2 Y, cuts(1 99)` | `winsor(df, Y, p=0.01)` |
| `label var Y "desc"` | `label(df, Y, "desc")` |

### Filtering & Subsets

| Stata | Hay |
|---|---|
| `keep if age >= 18` | `filter(df, age >= 18)` |
| `drop if missing(wage)` | `dropna(df, wage)` |
| `duplicates drop id, force` | `duplicates(df, id, action=drop)` |
| `preserve` / `restore` | `preserve(df)` / `restore(df)` |

### Merging & Reshaping

| Stata | Hay |
|---|---|
| `merge 1:1 id using "f.dta"` | `merge(df1, df2, key=id)` |
| `append using "f.dta"` | `append(df1, df2)` |
| `collapse (mean) Y, by(g)` | `collapse(df, mean, Y, by=g)` |
| `reshape long inc, i(id) j(yr)` | `reshape(df, id=id, stubs=[inc])` |

### Descriptive Statistics

| Stata | Hay |
|---|---|
| `summarize` | `summarize(df)` |
| `summarize, detail` | `summarize(df, detail=true)` |
| `tabulate var` | `tabulate(df, var)` |
| `tab v1 v2, chi2` | `tabulate(df, v1, v2, chi2=true)` |
| `correlate X1 X2 X3` | `correlate(df, X1, X2, X3)` |
| `pwcorr X1 X2, star(0.05)` | `pwcorr(df, X1, X2)` |
| `ci means Y` | `ci(df, Y)` |
| `ttest Y, by(group)` | `ttest(df, Y, by=group)` |
| `count if X > 0` | `count df if X > 0` |
| `describe` | `describe(df)` |
| `list in 1/10` | `list(df, n=10)` |

### Estimation

| Stata | Hay |
|---|---|
| `reg Y X1 X2` | `ols(Y ~ X1 + X2, df)` |
| `reg Y X, vce(robust)` | `ols(Y ~ X, df, cov=robust)` |
| `reg Y X, vce(cluster firm)` | `ols(Y ~ X, df, cluster=firm)` |
| `newey Y X, lag(4)` | `ols(Y ~ X, df, nw=4)` |
| `ivregress 2sls Y (X=Z1 Z2) C` | `iv(Y ~ X + C, ~ Z1 + Z2, df)` |
| `logit Y X` | `logit(Y ~ X, df)` |
| `probit Y X` | `probit(Y ~ X, df)` |
| `ologit Y X` | `ologit(Y ~ X, df)` |
| `mlogit Y X` | `mlogit(Y ~ X, df)` |
| `poisson Y X` | `poisson(Y ~ X, df)` |
| `nbreg Y X` | `nbreg(Y ~ X, df)` |
| `zip Y X` | `zip(Y ~ X, df)` |
| `tobit Y X, ll(0)` | `tobit(Y ~ X, df, ll=0)` |
| `heckman Y X, select(S=Z)` | `heckman(Y ~ X, df, select=S ~ Z)` |
| `qreg Y X, q(0.25)` | `qreg(Y ~ X, df, q=0.25)` |
| `xtset firm year` | `xtset(df, firm, year)` |
| `xtreg Y X, fe` | `fe(Y ~ X, df)` |
| `xtreg Y X, re` | `re(Y ~ X, df)` |
| `xtabond Y X` | `ab(Y ~ X, df, id=firm)` |
| `lasso linear Y X1-X50` | `lasso(Y ~ X1 + ... + X50, df)` |
| `arima Y, ar(1) ma(1)` | `arima(df, Y, p=1, d=0, q=1)` |
| `arch Y, arch(1) garch(1)` | `garch(df, Y, p=1, q=1)` |
| `var Y1 Y2, lags(2)` | `var(df, Y1, Y2, lags=2)` |
| `stcox X1 X2` | `cox(Y ~ X1 + X2, df, time=t)` |
| `glm Y X, family(binomial)` | `glm(Y ~ X, df, family=binomial)` |
| `diff Y, treat(T) post(P)` | `did(Y ~ X, df, treat=T, post=P)` |
| `rdrobust Y X, c(50)` | `rd(Y ~ X, df, running=X, cutoff=50)` |
| `xtfmb Y X` (paid addon) | `fmb(Y ~ X, df, time=t)` |

### Post-Estimation

| Stata | Hay |
|---|---|
| `test X1 X2` | `test(m, "X1", "X2")` |
| `test X1 = X2` | `test(m, "X1 = X2")` |
| `nlcom _b[X1]/_b[X2]` | `nlcom(m, X1 / X2)` |
| `margins, dydx(X1)` | `margins(m, dydx=[X1])` |
| `margins, at(X2=0)` | `margins(m, at_X2=0)` |
| `predict yhat` | `predict df yhat = m` |
| `predict e, resid` | `predict df e = m, "residuals"` |
| `estat ic` | `estat(m1, m2)` |
| `hausman fe re` | `hausman(m_fe, m_re)` |
| `vif` | `vif(m)` |
| `eststo` / `esttab` | `eststo(m)` / `esttab()` |
| `esttab using "t.tex"` | `esttab(fmt=latex, path="t.tex")` |
| `quietly reg Y X` | `quietly(ols(Y ~ X, df))` |
| `capture reg Y X` | `capture(ols(Y ~ X, df))` |
| `assert price > 0` | `assert(X > 0, "msg")` |

### Graphs

| Stata | Hay |
|---|---|
| `scatter Y X` | `scatter(df, X, Y)` |
| `histogram Y` | `histogram(df, Y)` |
| `graph export "f.png"` | `graph_scatter(df, X, Y, path="f.svg")` |

---

## What Hay has that Stata does not

| Feature | Syntax |
|---|---|
| Closures | `\|x\| x * 2` with `map`, `filter` |
| Pipe operator | `data \|> sort \|> filter(\|x\| x > 0)` |
| Pattern matching | `match x { 1 => "one", _ => "other" }` |
| If-expression | `let r = if x > 0 { "yes" } else { "no" }` |
| F-strings | `f"mean = {mu:.2f}"` |
| `in` operator | `if x in [1, 2, 3]` |
| Dict type | `{"key": value}` with full operations |
| List operations | push, pop, map, filter, sort, unique, flatten, ... |
| Try/catch | `try { } catch e { display e }` |
| Const declarations | `const PI = 3.14` |
| Block scoping | Variables scoped to `{ }` |
| Immutable params | Function parameters are const by default |
| No shadowing | Redeclaring a const is an error |
| Multiple DataFrames | All DataFrames are regular variables |
| Namespaces | `import("mod")` -> `mod::func()` |
| Type system | int, float, bool, str, list, dict, nil, closures |
| Parquet I/O | Native read/write |
| Fama-MacBeth | Built-in with Newey-West (Stata requires paid addon) |
| Portfolio sorts | `portsort`, `doublesort` built-in |
| Auto collinearity | Detected in all estimators, Stata-style `(omitted)` display |

---

## What Stata has that Hay does not

| Category | Stata |
|---|---|
| Survey data | `svy:` prefix for complex samples |
| SEM | `sem` / `gsem` structural equation models |
| Bayesian | `bayes:` prefix |
| Spatial | `spregress`, `spmatrix` |
| GUI | Full graphical interface |
| Graphics depth | 50+ plot types vs 4 SVG + 8 ASCII |
| Ecosystem | 10,000+ SSC community packages |
| Documentation | 15,000+ page manual |
| Academic acceptance | De facto standard in many journals |
| Maturity | 40+ years of production use |
