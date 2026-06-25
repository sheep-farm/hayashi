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
| Tests | 428 automated + 60 examples | Internal proprietary suite |
| Scoping | Block-scoped, const, no GC | Global |
| DataFrames | Multiple simultaneous, Rc COW | Single active dataset (frames since v16) |
| Types | int, float, bool, str, list, dict, closures | Numeric + string |
| Collinearity | Auto-detect, Stata-style (omitted) | Manual or addon |
| REPL | Tab completion, syntax highlighting, hints | Basic |
| Namespaces | Module-based (import as/only) | — |
| Date/time | date(), year(), month(), dow() | Built-in |
| License | GPL-3.0 | Proprietary |

## Syntax side by side

```
// Stata                              // Hay
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
.                                      import("finance")  // finance::func()
.                                      import("mod", as=m)  // m::func()
.                                      date("2024-01-15")
.                                      generate df Y = year(date_col)
gen X2 = exper * exper                 generate df X2 = exper |> |x| x * x
.                                      mutate(df, X2 = exper^2, X3 = ln(Y))
.                                      df |> mutate(z = x^2) |> filter(z > 5)
.                                      group_by(df, setor, mean, ret, vol)
.                                      pivot_longer(df, stubs=["gdp"], i=id, j=year)
.                                      codebook(df)
.                                      swilk(df, Y)  // Shapiro-Wilk
.                                      sktest(df, Y) // JB + D'Agostino
.                                      print("x =", x, sep=", ")

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
| Conditional MNLogit | `asclogit` | `cmnlogit` | Parity |
| GMM | `gmm` | `gmm` | Parity |
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
| Collinearity detection | manual `_rmcoll` | auto in all estimators | Hayashi superior |
| Date/time in generate | `year()` etc. | `year()` `month()` etc. | Parity |
| Mixed/HLM | `mixed` | `mixed` | Partial |
| Survey | `svy:` | — | Missing |
| SEM | `sem`/`gsem` | — | Missing |
| Bayesian | `bayes:` | — | Missing |
| Spatial | `spregress` | — | Missing |

## Language structures — Hayashi vs Stata

Stata is a command-oriented language with global state. Hayashi is a block-scoped, expression-oriented language with modern constructs.

### Type system

| Feature | Stata | Hayashi |
|---|---|---|
| Integer | numeric (no distinction) | `int` (64-bit) |
| Float | numeric | `float` (64-bit) |
| Boolean | 0/1 | `bool` (`true`/`false`) |
| String | `"text"` | `"text"` |
| List/array | — | `[1, 2, 3]` with 16 operations |
| Dictionary | — | `{"key": value}` with 7 operations |
| Nil/missing | `.` | `nil` |
| Type check | — | `type(x)` returns `"int"`, `"list"`, etc. |
| Conversions | `real()`, `string()` | `int()`, `float()`, `str()`, `bool()` |

### Variables and mutability

| Feature | Stata | Hayashi |
|---|---|---|
| Declaration | implicit global | `let x = 10` (mutable) |
| Constant | — | `const PI = 3.14` (immutable) |
| Scoping | global only | block-scoped, deterministic destruction |
| Shadowing | allowed (no warning) | **forbidden** — error on redeclare over const |
| Function params | mutable | `const` by default (immutable input) |
| DataFrame passing | single active dataset | `Rc` copy-on-write, zero-copy reads |

### Control flow

| Feature | Stata | Hayashi |
|---|---|---|
| If/else | `if` / `else` | `if cond { } else { }` |
| If expression | — | `let r = if x > 0 { "yes" } else { "no" }` |
| Match | — | `match x { 1 => "one", _ => "other" }` |
| For loop | `forvalues` / `foreach` | `for i in 1..10 { }` / `for v in list { }` |
| While loop | `while` | `while cond { }` |
| Try/catch | `capture` (no error access) | `try { } catch e { display e }` |
| Break/continue | — (only in Mata) | `break` / `continue` |

### Functions

| Feature | Stata | Hayashi |
|---|---|---|
| User functions | Mata only (separate language) | `fn name(x) { return x * x }` |
| Closures | — | `\|x\| x * 2` — anonymous, captures scope |
| First-class | no | yes — assign to variable, pass as argument |
| `map`/`filter` | — | `map(list, \|x\| x * 2)`, `filter(list, \|x\| x > 0)` |

### Operators Stata lacks

| Operator | Syntax | Example |
|---|---|---|
| Pipe | `\|>` | `data \|> sort \|> filter(\|x\| x > 0) \|> map(\|x\| x * 10)` |
| Membership | `in` | `if 3 in [1, 2, 3]`, `if "key" in dict` |
| F-string | `f"..."` | `f"mean = {mu:.2f}, n = {n}"` |
| Match | `match` | `match status { 1 => "active", _ => "unknown" }` |
| Ternary | if-expr | `let label = if x > 0 { "pos" } else { "neg" }` |

### Collections Stata lacks

```
// List — 16 operations, push/pop mutate in-place (like Python/JS/Rust)
let nums = [3, 1, 2] |> sort |> map(|x| x * 10)   // [10, 20, 30]
push  pop  insert  remove  clear  reverse  index  slice
join  map  filter  unique  flatten  sort  range  len

// Dict — 7 operations
let config = {"alpha": 0.05, "n_boot": 1000}
keys  values  has_key  dict_set  dict_remove  dict_merge  len

// esttab accepts lists: build models with push in loop
let models = []
for v in ["X1", "X2", "X3"] {
    push(models, ols("Y ~ " + v, df))
}
esttab(models)

// Pipe chaining (no Stata equivalent)
raw_data
    |> filter(|row| row > 0)
    |> sort
    |> unique
    |> map(|x| x * 100)
```

### Error handling

| Feature | Stata | Hayashi |
|---|---|---|
| Suppress output | `quietly` | `quietly(expr)` |
| Ignore errors | `capture` (no error info) | `capture(expr)` — returns Nil on error |
| Structured handling | — | `try { } catch e { display f"Error: {e}" }` |
| Assertions | `assert` | `assert(cond, "message")` |
| Error messages | generic | source preview, "did you mean?", stack traces, type mismatch |

### Multiple DataFrames

Stata supports only one active dataset (frames added in v16, verbose syntax). Hayashi treats DataFrames as regular variables:

```
// Hay — natural
load "sales.csv" as sales
load "clients.parquet" as clients
load "stock.db" as stock, table=products
let merged = merge(sales, clients, key=id)

// Stata — requires frame switching
frame create clients
frame change clients
use "clients.dta"
frame change default
frlink 1:1 id, frame(clients)
```

## Where Hayashi wins

**Cost and deployment:**
- Free and open source (GPL-3.0) vs US$ 595+/year
- Single 20 MB binary, no system dependencies
- `cargo install hay` — done

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
- Module namespaces: `import("mod")` -> `mod::func()`
- Pipe with inline closures: `exper |> |x| x * x`
- `push`/`pop` mutate in-place (standard behavior)
- Date/time extraction: `year()`, `month()`, `day()`, `dow()` in generate

**Econometrics-specific:**
- Fama-MacBeth with Newey-West built-in (Stata requires paid addon)
- Portfolio sorts (`portsort`, `doublesort`) built-in
- Generic bootstrap with any estimator
- Dynamic formulas: `ols("Y ~ " + v, df)` native
- Row-wise regex in formulas: `ols(Y ~ X, df, if = regexm(name, "Dr"))`
- Copy-on-write DataFrames: zero-copy in functions
- Auto collinearity detection across all estimators (Stata-style (omitted) display)

**Developer experience:**
- 428 automated tests, 60 examples, `cargo test` in <1s
- `help()` with ~115 topics, signature + example for every command
- VS Code extension (syntax highlighting, run/debug)
- Tab completion + syntax highlighting in REPL
- Error messages with source preview, "did you mean?", stack traces, type mismatch
- Multi-line expressions inside parentheses
- Domain: haylang.dev

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

Hayashi covers ~97% of the applied econometrics workflow at the graduate level with 53 estimators, 428 automated tests, and full functional parity in estimation, post-estimation, data manipulation, and publishable output. The remaining gaps are specialized niches (survey, SEM, Bayesian, spatial) that few researchers use simultaneously.

The language goes beyond Stata with modern features (closures, pattern matching, pipe, f-strings, dict, const, try/catch, namespaces, mutate, group_by, pivot) that make scripts more expressive and robust. Pipe assign-back semantics (`df |> f()` modifies source), rich error messages with "did you mean?", stack traces, and source preview make debugging intuitive. Multiple simultaneous DataFrames, block scoping, immutable function parameters, and auto collinearity detection are architectural improvements over Stata's global-state model. Scripts use the `.hay` extension and documentation is available at haylang.dev.
