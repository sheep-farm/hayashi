# Hayashi

An interpreted language for applied econometrics. Named after [Fumio Hayashi](https://en.wikipedia.org/wiki/Fumio_Hayashi).

Single binary, Stata-like syntax. Built in Rust on top of [Greeners](https://github.com/sheep-farm/Greeners).

## Install

```bash
git clone https://github.com/sheep-farm/hayashi.git
cd hayashi
cargo build --release
# Binary at target/release/hayashi

# Optional: ODBC support (requires unixodbc)
cargo build --release --features odbc
```

## Usage

```bash
hayashi                  # interactive REPL (multi-line)
hayashi script.hy        # run a script
hayashi -                # read from stdin
hayashi --help           # list commands
```

## Quick start

```stata
load "data.dta" as df

// OLS with clustered standard errors
let m1 = reg(Y ~ X1 + X2, df, cluster=firm)
let m2 = reg(Y ~ X1 + X2 + X3, df, cluster=firm)

// Comparison table (ASCII or LaTeX)
esttab(m1, m2)
esttab(m1, m2, fmt=latex, path="table1.tex")

// Export to multiple formats
export(df, "csv", "data.csv")
export(df, "xlsx", "data.xlsx")
export(df, "parquet", "data.parquet")
export(m1, "latex", "table.tex")

// Post-estimation
test(m2, "X2", "X3")          // joint F-test
test(m2, "X2 = X3")           // Wald: beta_X2 = beta_X3
nlcom(m2, X2 / X3)            // delta method
margins(m_logit)              // AME with standard errors
coefplot(m2)                  // ASCII coefficient plot

// SVG graphs for papers
graph_scatter(df, X, Y, path="fig1.svg")
graph_coef(m2, path="fig2.svg")

// Bootstrap (works with any estimator)
bootstrap(ols, Y ~ X1 + X2, df, n=1000)

// Automation
for spec in ["X1", "X1 + X2", "X1 + X2 + X3"] {
    eststo(ols("Y ~ " + spec, df, cluster=firm))
}
esttab(fmt=latex, path="table1.tex")
```

## Estimators

| Category | Commands |
|---|---|
| Linear | `ols` `reg` `iv` `wls` `glsar` |
| Panel | `fe` `re` `ab` `sysgmm` `pcse` `xtgls` |
| Binary | `logit` `probit` `cloglog` `clogit` |
| Count | `poisson` `nbreg` `zip` `zinb` |
| Ordinal | `ologit` `oprobit` |
| Censored | `tobit` `heckman` `truncreg` |
| Survival | `cox` `km` |
| Quantile | `qreg` |
| Regularization | `lasso` `ridge` `elasticnet` |
| Time series | `arima` `sarima` `garch` `egarch` `gjrgarch` |
| VAR | `var` `vecm` `svar` `irf` `fevd` |
| Causal | `did` `rd` `synth` `psmatch` |
| Finance | `fmb` `portsort` `doublesort` |
| Robust | `rlm` `gee` `glm` `betareg` |

All estimators support `if=` for subsamples and `cov=`/`cluster=`/`nw=` for robust SEs.

## Post-estimation

```stata
test(m, "X1", "X2")          // joint F-test
test(m, "X1 = X2")           // linear restriction
test(m, "white")             // White heteroskedasticity
test(m, "bp")                // Breusch-Pagan
test(m, "dw")                // Durbin-Watson
nlcom(m, X1 / X2)            // nonlinear combination (delta method)
margins(m)                   // AME with SEs, z-values, p-values
margins(m, dydx=[X1])        // specific variables
margins(m, at_X2=0)          // at fixed values
coefplot(m)                  // ASCII coefficient plot with 95% CI
estat(m1, m2, m3)            // AIC/BIC comparison
hausman(m_fe, m_re)          // Hausman test
predict df yhat = m              // fitted values
predict df e = m, "residuals"    // residuals
bootstrap(ols, Y ~ X, df, n=1000)
```

## Data

```stata
load "file.csv" as df
load "file.dta" as df                    // Stata .dta files
load "file.json" as df                   // JSON (array or column-oriented)
load "file.tsv" as df                    // tab-separated
load "file.xlsx" as df, sheet=Plan1      // Excel (xlsx/xls/ods)
load "file.parquet" as df                // Apache Parquet
load "file.db" as df, table=prices       // SQLite
load "file.db" as df, query="SELECT * FROM prices WHERE year > 2020"
load "odbc://DSN=mydb" as df, query="SELECT * FROM t"  // ODBC (feature flag)
load "https://...csv" as df              // URLs (auto-download)
load "data.csv" as df, sep=";"           // custom delimiter

generate df lnY = log(Y)
generate df D = (X == 1)       // dummy from condition
generate df D = (col == "val") // string comparison
generate df dr = regexm(name, "^Dr")  // regex over string column
generate df Z = std(Y)         // z-score
generate df row = _n           // row number
replace df Y = 0 if X > 10
drop(df, col)
keep(df, col1, col2)
winsor(df, Y, p=0.01)
encode(df, str_col)
tabgen(df, group)
recode(df, X, from=[1,2], to=[10,20])
duplicates(df, id, action=drop)
label(df, Y, "GDP per capita")
preserve(df)
restore(df)

summarize(df, detail=true)     // percentiles, skewness, kurtosis
ci(df, Y)                      // confidence interval for mean
centile(df, Y)                 // arbitrary percentiles
tabulate(df, group)
correlate(df, X1, X2, X3)
pwcorr(df, X1, X2, X3)         // with significance stars
list(df, vars=[X1, X2], n=10)
ttest(df, Y, by=group)
```

## Graphs

```stata
// SVG (publishable, LaTeX-ready)
graph_scatter(df, X, Y, path="fig.svg")
graph_line(df, X, Y, path="fig.svg")
graph_hist(df, Y, path="fig.svg", bins=30)
graph_coef(m, path="fig.svg")

// ASCII (terminal)
scatter(df, X, Y)
histogram(df, Y)
coefplot(m)
boxplot(df, Y)
kdensity(df, Y)
acfplot(df, Y)
qqplot(df, Y)
corrplot(df, X1, X2, X3)
```

## Panel

```stata
xtset(df, firm, year)

let m_fe = fe(Y ~ X1 + X2, df)
let m_re = re(Y ~ X1 + X2, df)
hausman(m_fe, m_re)
esttab(m_fe, m_re)
```

## Finance

```stata
fmb(ret ~ beta + size + bm, df, time=month, nw=4)
portsort(df, ret, beta, n=5)
doublesort(df, ret, size, bm, n1=5, n2=5)
```

## Regex

```stata
regexm(s, "[0-9]+")              // match → bool
regexr(s, "[0-9]+", "NUM")       // replace first
regexra(s, "[0-9]+", "NUM")      // replace all
regexs(s, "([0-9]+\\.[0-9]+)")   // extract capture group

// Row-wise over string columns
generate df is_dr = regexm(name, "^Dr")
ols(Y ~ X, df, if = regexm(name, "Dr"))
```

## Scoping

Variables are block-scoped. `{}` controls lifetime. Deterministic destruction via `Rc` — no GC.

```stata
let x = 10
{
    let temp = 42   // dies at }
    x = x + temp    // modifies outer (assign without let)
}
// temp is gone, x = 52
```

DataFrames use copy-on-write (`Rc<DataFrame>`). Function parameters are zero-copy reads.

## Language

```stata
let x = 5
scalar mu = mean(df, Y, if = X == 1)
display mu

if x > 3 { ... } else { ... }
for i in 1..10 { ... }
for v in ["a", "b"] { ... }
while cond { ... }
fn square(x) { return x * x }

quietly(ols(Y ~ X, df))       // suppress output
capture(ols(Y ~ X, df))       // ignore errors
assert(n > 0, "empty data")

set_seed(42)
timer(ols(Y ~ X, df))
source("other_script.hy")
help(ols)
```

## Build & Test

```bash
cargo build --release      # optimized binary
cargo test                 # 338 tests, <1s
```

59 example scripts in `exemplos/`, all passing.

## License

MIT
